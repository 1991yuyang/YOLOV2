import torch as t
from torch import nn, optim
import os
from model_define import MyNet
from yolov2_loss import Loss
from dataLoader import MySet
import numpy as np
from torch.utils import data


with open("./conf.json", "r", encoding="utf-8") as file:
    config = eval(file.read())
input_img_sizes = [int(i) for i in config["input_img_sizes"].split(",")]
for sz in input_img_sizes:
    if sz % 32 != 0:
        raise Exception("input_img_sizes should be an integer multiple of 32")
change_img_size_epoch = config["change_img_size_epoch"]
CUDA_VISIBLE_DEVICES = config["CUDA_VISIBLE_DEVICES"]
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
device_ids = list(range(len(CUDA_VISIBLE_DEVICES)))
epoch = config["epoch"]
batch_size = config["batch_size"]
lr = config["lr"]
lr_decrease_epoch = config["lr_decrease_epoch"]
lr_decrease_rate = config["lr_decrease_rate"]
img_file_path = config["img_file_path"]
txt_file_path = config["txt_file_path"]
xml_file_path = config["xml_file_path"]
epoch_model_save_path = config["epoch_model_save_path"]
best_model_save_path = config["best_model_save_path"]
lamda_coord = config["lamda_coord"]
lamda_noobj = config["lamda_noobj"]
lamda_obj = config["lamda_obj"]
lamda_class = config["lamda_class"]
weight_decay = config["weight_decay"]
save_best_model_good_performence_count = config["save_best_model_good_performence_count"]
anchor_boxes = np.load("./cluster_anchor_boxes.npy")
with open("./class_name_to_index.json", "r", encoding="utf-8") as file:
    class_name_to_index = eval(file.read())
num_classes = len(class_name_to_index)


def train():
    global lr
    start_epoch = 1
    model = MyNet(num_classes, anchor_boxes.shape[0])
    model = nn.DataParallel(module=model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
    input_img_size_index = 0
    if os.listdir(epoch_model_save_path):
        # 如果有已经训练至一半的模型需要加载模型，加载当前模型学习率，加载当前模型开始训练epoch
        print("load epoch model......")
        model_name = os.listdir(epoch_model_save_path)[0]
        start_epoch, lr, input_img_size_index = model_name.strip(".pth").split("_")
        start_epoch = int(start_epoch)
        lr = float(lr)
        input_img_size_index = int(input_img_size_index)
        model.load_state_dict(t.load(os.path.join(epoch_model_save_path, model_name)))
    valid_losses = [float("inf")] * save_best_model_good_performence_count
    current_minist_valid_loss = float("inf")
    if os.path.exists(os.path.join(best_model_save_path, "best_model.pth")):
        with open(os.path.join(best_model_save_path, "current_minist_valid_loss.txt"), "r", encoding="utf-8") as file:
            current_minist_valid_loss = float(file.read())
            valid_losses = [current_minist_valid_loss] * save_best_model_good_performence_count
    criterion = Loss(num_classes, anchor_boxes, lamda_coord, lamda_noobj, lamda_obj, lamda_class)
    optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    for e in range(start_epoch, 1 + epoch):
        if change_img_size_epoch != 1:
            if e % change_img_size_epoch == 1 and e != 1:
                if input_img_size_index != len(input_img_sizes) - 1:
                    input_img_size_index += 1
                else:
                    input_img_size_index = 0
        else:
            if e != 1:
                input_img_size_index += 1
                if input_img_size_index == len(input_img_sizes):
                    input_img_size_index = 0
        input_img_size = input_img_sizes[input_img_size_index]
        train_loader = iter(data.DataLoader(MySet(input_img_size, True, txt_file_path, img_file_path, xml_file_path, anchor_boxes, class_name_to_index), batch_size=batch_size, shuffle=True, drop_last=False))
        valid_loader = iter(data.DataLoader(MySet(input_img_size, False, txt_file_path, img_file_path, xml_file_path, anchor_boxes, class_name_to_index), batch_size=batch_size, shuffle=True, drop_last=False))
        step = 0
        for d_train, l_train, original_img_sizes in train_loader:
            model.train()
            original_img_sizes_cuda = original_img_sizes.cuda(device_ids[0])
            d_train_cuda = d_train.cuda(device_ids[0])
            l_train_cuda = l_train.cuda(device_ids[0])
            train_output = model(d_train_cuda)
            train_loss = criterion(train_output, l_train_cuda, original_img_sizes_cuda)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            step += 1
            try:
                d_valid, l_valid, original_img_sizes_valid = next(valid_loader)
            except:
                valid_loader = iter(data.DataLoader(MySet(input_img_size, False, txt_file_path, img_file_path, xml_file_path, anchor_boxes, class_name_to_index), batch_size=batch_size, shuffle=True, drop_last=False))
                d_valid, l_valid, original_img_sizes_valid = next(valid_loader)
            d_valid_cuda = d_valid.cuda(device_ids[0])
            l_valid_cuda = l_valid.cuda(device_ids[0])
            original_img_sizes_valid_cuda = original_img_sizes_valid.cuda(device_ids[0])
            model.eval()
            with t.no_grad():
                valid_output = model(d_valid_cuda)
                valid_loss = criterion(valid_output, l_valid_cuda, original_img_sizes_valid_cuda).item()
                valid_losses.append(valid_loss)
                valid_losses.pop(0)
            print("input_img_size: %d, epoch: %d, step: %d, valid_loss: %f, train_loss: %f" % (input_img_size, e, step, valid_loss, train_loss.item()))
            if np.mean(valid_losses) < current_minist_valid_loss:
                print("saving best model......")
                current_minist_valid_loss = np.mean(valid_losses)
                with open(os.path.join(best_model_save_path, "current_minist_valid_loss.txt"), "w", encoding="utf-8") as file:
                    file.write(str(current_minist_valid_loss))
                t.save(model.state_dict(), os.path.join(best_model_save_path, "best_model.pth"))
        if e % lr_decrease_epoch == 0:
            lr *= lr_decrease_rate
            optimizer = optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        print("saving epoch model......")
        if os.listdir(epoch_model_save_path):
            os.remove(os.path.join(epoch_model_save_path, os.listdir(epoch_model_save_path)[0]))
        t.save(model.state_dict(), os.path.join(epoch_model_save_path, "%d_%.10f_%d.pth" % (e + 1, lr, input_img_size_index)))


if __name__ == "__main__":
    train()