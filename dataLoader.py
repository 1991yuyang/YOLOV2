from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from utils import data_augmentation
import os
import torch as t
import numpy as np
from xml.etree import ElementTree as ET
from utils import calc_iou


class MySet(data.Dataset):

    def __init__(self, img_size, is_train, txt_file_path, img_file_path, xml_file_path, anchor_boxes, class_name_to_index):
        """

        :param anchor_boxes: 聚类举出的anchor box
        :param img_size: 模型输入的图片尺寸，因为yolov2是多尺度训练，而不同尺度模型输出的预测tensor的尺寸不一致，因此需要对应改变label的尺寸
        :param img_path: 图片存放路径
        :param xml_file_path: xml标记文档存放路径
        """
        self.class_name_to_index = class_name_to_index
        self.num_classes = len(class_name_to_index)
        self.anchor_boxes = anchor_boxes
        self.label_size = int(img_size / 32)
        self.transfomer = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])
        if is_train:
            with open(os.path.join(txt_file_path, "train.txt"), "r", encoding="utf-8") as file:
                img_names = file.read().strip("\n").split("\n")
        else:
            with open(os.path.join(txt_file_path, "val.txt"), "r", encoding="utf-8") as file:
                img_names = file.read().strip("\n").split("\n")
        self.img_paths = [os.path.join(img_file_path, "%s.jpg" % (name,)) for name in img_names]
        self.xml_file_paths = [os.path.join(xml_file_path, "%s.xml" % (name,)) for name in img_names]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        xml_file_path = self.xml_file_paths[index]
        label = np.zeros((self.label_size, self.label_size, self.anchor_boxes.shape[0] * (5 + self.num_classes)), np.float) # （y_grid, x_grid, boxes_info），注意每个grid cell都对应着一个(5 + num_classes) * anchor_boxes_count长度的向量，
                                                                                                                               # 第i * (5 + num_classes)到（i + 1） * (5 + num_classes)，(i=0,1,...anchor_boxes_count - 1)位置为索引为i的anchor box的信息
        img, h_shift, w_shift = data_augmentation(img_path)                                                                     # 形如(x_offset, y_offset, w_offset, h_offset, confidence, c1_prob, ..., cc_prob)，confidence需要实时在线计算
        original_img_size = img.size[::-1]  # (h, w)
        img = self.transfomer(img)
        xml_tree = ET.parse(xml_file_path)
        self.grid_y_lenth = original_img_size[0] / self.label_size
        self.grid_x_lenth = original_img_size[1] / self.label_size
        for obj in xml_tree.findall("object"):
            # 遍历当前图片上的每一个物体
            bbx = obj.find("bndbox")
            xmin = float(bbx.find("xmin").text) + w_shift
            ymin = float(bbx.find("ymin").text) + h_shift
            xmax = float(bbx.find("xmax").text) + w_shift
            ymax = float(bbx.find("ymax").text) + h_shift
            if xmin >= original_img_size[1] or ymin >= original_img_size[0] or xmax <= 0 or ymax <= 0:
                continue
            xmin = xmin if xmin > 0 else 0
            ymin = ymin if ymin > 0 else 0
            xmax = xmax if xmax < original_img_size[1] else original_img_size[1]
            ymax = ymax if ymax < original_img_size[0] else original_img_size[0]
            ground_truth_w = xmax - xmin  # 需要在anchor box确定之后计算w_offset和h_offset
            ground_truth_h = ymax - ymin
            class_name = obj.find("name").text
            class_index = self.class_name_to_index[class_name]  # 随后在对应anchor box的类别概率向量中对应类别处填入1
            # 判断当前物体的中心属于哪个grid cell
            x_grid_index, y_grid_index, x_offset, y_offset = self.get_grid_index([xmin, ymin, xmax, ymax])  # x_offset, y_offset在确定哪一个anchor box后填入对应位置
            # 判断当前grid cell是否还有未占用的anchor box
            not_ocupy_anchor_box = label[y_grid_index, x_grid_index, 4::(5 + self.num_classes)] == 0
            if np.sum(not_ocupy_anchor_box) == 0:
                # 当前grid cell的所有anchor boxes都已被占用
                continue
            # 获取当前grid cell所有未占用的anchor box的索引
            not_ocupy_anchor_box_index = np.where(not_ocupy_anchor_box)[0]
            # 遍历所有未占用的anchor box找到与当前物体iou最大的anchor box，并在其对应的label处填充
            current_biggest_iou = -float("inf")
            current_biggest_iou_anchor_box_index = None
            current_w_offset = None
            current_h_offset = None
            for i, anchor_box in enumerate(self.anchor_boxes[not_ocupy_anchor_box_index]):
                current_anchor_box_index = not_ocupy_anchor_box[i]
                anchor_box_w = anchor_box[1]
                anchor_box_h = anchor_box[0]
                anchor_box_x_center = self.grid_x_lenth / 2 + x_grid_index * self.grid_x_lenth
                anchor_box_y_center = self.grid_y_lenth / 2 + y_grid_index * self.grid_y_lenth
                anchor_box_xmin = anchor_box_x_center - anchor_box_w / 2
                anchor_box_xmax = anchor_box_x_center + anchor_box_w / 2
                anchor_box_ymin = anchor_box_y_center - anchor_box_h / 2
                anchor_box_ymax = anchor_box_y_center + anchor_box_h / 2
                ############################################################
                calc_iou_xmin = anchor_box_x_center - ground_truth_w / 2
                calc_iou_xmax = anchor_box_x_center + ground_truth_w / 2
                calc_iou_ymin = anchor_box_y_center - ground_truth_h / 2
                calc_iou_ymax = anchor_box_y_center + ground_truth_h / 2
                ############################################################
                current_iou = calc_iou([anchor_box_xmin, anchor_box_ymin, anchor_box_xmax, anchor_box_ymax], [calc_iou_xmin, calc_iou_ymin, calc_iou_xmax, calc_iou_ymax])
                if current_iou > current_biggest_iou:
                    current_biggest_iou = current_iou
                    current_biggest_iou_anchor_box_index = current_anchor_box_index
                    current_w_offset = np.log(ground_truth_w / anchor_box_w)
                    current_h_offset = np.log(ground_truth_h / anchor_box_h)
            label[y_grid_index, x_grid_index, current_biggest_iou_anchor_box_index * (5 + self.num_classes)] = x_offset
            label[y_grid_index, x_grid_index, current_biggest_iou_anchor_box_index * (5 + self.num_classes) + 1] = y_offset
            label[y_grid_index, x_grid_index, current_biggest_iou_anchor_box_index * (5 + self.num_classes) + 2] = current_w_offset
            label[y_grid_index, x_grid_index, current_biggest_iou_anchor_box_index * (5 + self.num_classes) + 3] = current_h_offset
            label[y_grid_index, x_grid_index, current_biggest_iou_anchor_box_index * (5 + self.num_classes) + 4] = 1
            label[y_grid_index, x_grid_index, current_biggest_iou_anchor_box_index * (5 + self.num_classes) + 5 + class_index] = 1
        return img, t.tensor(label).type(t.FloatTensor), t.tensor(original_img_size)

    def __len__(self):
        return len(self.xml_file_paths)

    def get_grid_index(self, box):
        """
        获取当前物体中心落入哪个grid cell，以及中心相对于当前grid cell左上角偏移量，并计算与其具有最大iou的anchor box的索引，以及当前
        :param box: 物体的bounding box，[xmin, ymin, xmax, ymax]
        :return: y_grid_index, x_grid_index，biggest_iou_anchor_box_index
        """
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        y_grid_index = int(y_center / self.grid_y_lenth)
        x_grid_index = int(x_center / self.grid_x_lenth)
        y_offset = y_center / self.grid_y_lenth - y_grid_index
        x_offset = x_center / self.grid_x_lenth - x_grid_index
        return x_grid_index, y_grid_index, x_offset, y_offset


if __name__ == "__main__":
    # anchor_boxes = np.load("./cluster_anchor_boxes.npy")
    # with open("./class_name_to_index.json", "r", encoding="utf-8") as file:
    #     class_name_to_index = eval(file.read())
    # s = MySet(320, r"H:\VOC2012\VOCdevkit\VOC2012\JPEGImages", "H:/VOC2012/VOCdevkit/VOC2012/Annotations", anchor_boxes, class_name_to_index)
    # img, label = s[1]
    # print(np.where(label > 0))
    with open(r"H:\VOC2012\VOCdevkit\VOC2012\ImageSets\Main\train.txt", "r", encoding="utf-8") as file:
        img_names = file.read().strip("\n")
    print(img_names.split("\n"))