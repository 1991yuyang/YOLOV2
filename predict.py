import numpy as np
from torchvision import transforms as T
import os
from utils import nms
from model_define import MyNet
from PIL import Image
import torch as t
from torch import nn
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with open("./class_name_to_index.json", "r", encoding="utf-8") as file:
    class_name_to_index = eval(file.read())
index_to_class_name = {v: k for k, v in class_name_to_index.items()}
with open("./conf.json", "r", encoding="utf-8") as file:
    config = eval(file.read())
predict_use_best_model = config["predict_use_best_model"]
predict_model_input_img_size = config["predict_model_input_img_size"]
assert predict_model_input_img_size % 32 == 0, "predict_input_img_size should be an integer multiple of 32"
best_model_save_path = config["best_model_save_path"]
epoch_model_save_path = config["epoch_model_save_path"]
have_obj_confidence_thresh = config["have_obj_confidence_thresh"]
nms_iou_thresh = config["nms_iou_thresh"]
predict_result_save_path = config["predict_result_save_path"]
test_img_path = config["test_img_path"]
anchor_boxes = np.load("./cluster_anchor_boxes.npy")
Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]
colors = {index_to_class_name[i]: Color[i] for i in range(len(class_name_to_index))}


def get_img_tensor(img_path, model_input_img_size):
    transformer = T.Compose([
        T.Resize((model_input_img_size, ) * 2),
        T.ToTensor()
    ])
    pil_img = Image.open(img_path)
    cv2_img = cv2.imread(img_path)
    original_img_size = pil_img.size[::-1]  # (h, w)
    img = transformer(pil_img).unsqueeze(0)
    return cv2_img, img, original_img_size


def load_model(num_classes):
    model = MyNet(num_classes, anchor_boxes.shape[0])
    model = nn.DataParallel(module=model, device_ids=[0])
    model = model.cuda(0)
    if predict_use_best_model:
        print("load best model......")
        model.load_state_dict(t.load(os.path.join(best_model_save_path, "best_model.pth")))
    else:
        print("load epoch model......")
        assert len(os.listdir(epoch_model_save_path)) != 0, "there is not a epoch_model"
        model.load_state_dict(t.load(os.path.join(epoch_model_save_path, os.listdir(epoch_model_save_path)[0])))
    return model


def predict_one_img(img_path, model_input_img_size, model):
    num_classes = len(class_name_to_index)
    label_size = model_input_img_size // 32
    cv2_img, img, original_img_size = get_img_tensor(img_path, model_input_img_size)
    y_grid_lenth = original_img_size[0] / label_size
    x_grid_lenth = original_img_size[1] / label_size
    img = img.cuda(0)
    model.eval()
    with t.no_grad():
        model_output = model(img)[0]  # shape: [label_size, label_size, anchor_boxe_count * (5 + num_classes)]，[10, 10, 125]
    all_anchor_box_confidence = model_output[:, :, 4::(5 + num_classes)]  # shape: [label_size, label_size, anchor_boxes_count]
    # 将每个anchor box后面的类别概率向量乘以confidence值
    for i in range(anchor_boxes.shape[0]):
        model_output[:, :, i * (num_classes + 5) + 5:(i + 1) * (num_classes + 5)] = all_anchor_box_confidence[:, :, i].unsqueeze(-1) * model_output[:, :, i * (num_classes + 5) + 5:(i + 1) * (num_classes + 5)]
    have_obj_index = all_anchor_box_confidence > have_obj_confidence_thresh
    # 获取有物体的grid索引和对应有物体的anchor box的索引
    have_obj_ygridindex_xgridindex_anchorindex = list(zip(*np.where(have_obj_index.cpu().detach().numpy() == 1)))
    all_box_info = {}
    for y_grid_index, x_grid_index, anchor_index in have_obj_ygridindex_xgridindex_anchorindex:
        box_info = model_output[y_grid_index, x_grid_index, anchor_index * (num_classes + 5):(anchor_index + 1) * (num_classes + 5)]
        class_index = t.argmax(box_info[5:])
        class_prob = box_info[5:][class_index].item()
        class_name = index_to_class_name[class_index.item()]
        if class_name not in all_box_info:
            all_box_info[class_name] = []
        anchor_box = anchor_boxes[anchor_index]
        pred_x_center = (box_info[0] + x_grid_index) * x_grid_lenth
        pred_y_center = (box_info[1] + y_grid_index) * y_grid_lenth
        pred_w = t.exp(box_info[2]) * anchor_box[1]
        pred_h = t.exp(box_info[3]) * anchor_box[0]
        pred_xmin = (pred_x_center - pred_w / 2).item()
        pred_ymin = (pred_y_center - pred_h / 2).item()
        pred_xmax = (pred_x_center + pred_w / 2).item()
        pred_ymax = (pred_y_center + pred_h / 2).item()
        pred_xmin = np.max([0, pred_xmin])
        pred_ymin = np.max([0, pred_ymin])
        pred_xmax = np.min([pred_xmax, original_img_size[1]])
        pred_ymax = np.min([pred_ymax, original_img_size[0]])
        all_box_info[class_name].append({"coord": [pred_xmin, pred_ymin, pred_xmax, pred_ymax], "class_prob": class_prob})
    all_box_info = nms(all_box_info, nms_iou_thresh)
    for class_name, boxes in all_box_info.items():
        # 遍历每一个类别
        bbx_color_of_current_class = colors[class_name]
        for box in boxes:
            # 遍历每一个类别的bounding box
            class_prob = box["class_prob"]
            coord = box["coord"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(cv2_img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), bbx_color_of_current_class, 2)
            cv2.putText(cv2_img, "%s:%.2f" % (class_name, np.round(class_prob, 2)), (int(coord[0]), int(coord[1]) - 5), font, 0.4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(predict_result_save_path, img_name), cv2_img)


if __name__ == "__main__":
    for i in os.listdir(predict_result_save_path):
        os.remove(os.path.join(predict_result_save_path, i))
    model = load_model(len(class_name_to_index))
    for img_name in os.listdir(test_img_path):
        print("predict %s" % (img_name,))
        img_path = os.path.join(test_img_path, img_name)
        predict_result = predict_one_img(img_path, predict_model_input_img_size, model)
        # predict_result.save(os.path.join(predict_result_save_path, img_name))