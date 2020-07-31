import torch as t
from torch import nn
from utils import calc_iou
import numpy as np


class Loss(nn.Module):

    def __init__(self, num_classes, anchor_boxes, lamda_coord, lamda_noobj, lamda_obj, lamda_class):
        """

        :param num_classes: 物体类别总数
        :param anchor_boxes: 聚类举出的anchor_boxes
        """
        super(Loss, self).__init__()
        self.lamda_coord = lamda_coord
        self.lamda_noobj = lamda_noobj
        self.lamda_obj = lamda_obj
        self.lamda_class = lamda_class
        self.anchor_boxes = anchor_boxes
        self.anchor_box_count = int(self.anchor_boxes.shape[0])
        self.num_classes = num_classes
        self.corss_entropy = nn.CrossEntropyLoss()

    def forward(self, model_output, label, original_img_sizes):
        obj_index_memory = []  # obj_index_memory[i]表示第i个anchor box有物体的索引，obj_index_memory[i]形状为[N, label_size, label_size]
        noobj_index_memory = []
        # TODO：将label中有物体的confidence乘以iou
        for i in range(self.anchor_box_count):
            anchor_box_size = self.anchor_boxes[i]
            ious = t.zeros(model_output.size()[0], label.size()[1], label.size()[2]).cuda()
            # 计算含有物体的第i个anchor box的grid cell索引
            obj_index = label[:, :, :, i * (5 + self.num_classes) + 4] >= 1  # shape: [N, label_size, label_size]
            noobj_index = label[:, :, :, i * (5 + self.num_classes) + 4] == 0  # shape: [N, label_size, label_size]
            obj_index_memory.append(obj_index)
            noobj_index_memory.append(noobj_index)
            img_indexs, y_grid_indexs, x_grid_indexs = np.where(obj_index.cpu().detach().numpy() == 1)
            # 将这些grid cell中的第i个anchor box的预测框和真实物体的bounding box计算iou并将iou乘在label对应位置的confidence处
            current_all_anchor_box_pred_info = model_output[obj_index][:, i * (5 + self.num_classes):(i + 1) * (5 + self.num_classes)][:, :5]  # shape: [k, 5]
            current_all_anchor_box_label_info = label[obj_index][:, i * (5 + self.num_classes):(i + 1) * (5 + self.num_classes)][:, :5]  # shape: [k, 5]
            for j in range(obj_index.sum().item()):
                anchor_box_pred_info = current_all_anchor_box_pred_info[j]  # shape: [5]
                anchor_box_label_info = current_all_anchor_box_label_info[j]  # shape: [5]
                img_index = img_indexs[j]
                y_grid_index = y_grid_indexs[j]
                x_grid_index = x_grid_indexs[j]
                original_img_size = original_img_sizes[img_index]

                y_grid_lenth = original_img_size[0] / label.size()[1]
                x_grid_lenth = original_img_size[1] / label.size()[2]
                pred_box_x_center = (x_grid_index + anchor_box_pred_info[0]) * x_grid_lenth
                pred_box_y_center = (y_grid_index + anchor_box_pred_info[1]) * y_grid_lenth
                pred_box_w = t.exp(anchor_box_pred_info[2]) * anchor_box_size[1]
                pred_box_h = t.exp(anchor_box_pred_info[3]) * anchor_box_size[0]
                pred_box_xmin = pred_box_x_center - pred_box_w / 2
                pred_box_xmax = pred_box_x_center + pred_box_w / 2
                pred_box_ymin = pred_box_y_center - pred_box_h / 2
                pred_box_ymax = pred_box_x_center + pred_box_h / 2
                true_box_x_center = (x_grid_index + anchor_box_label_info[0]) * x_grid_lenth
                true_box_y_center = (y_grid_index + anchor_box_label_info[1]) * y_grid_lenth
                true_box_w = t.exp(anchor_box_label_info[2]) * anchor_box_size[1]
                true_box_h = t.exp(anchor_box_label_info[3]) * anchor_box_size[0]
                true_box_xmin = true_box_x_center - true_box_w / 2
                true_box_xmax = true_box_x_center + true_box_w / 2
                true_box_ymin = true_box_y_center - true_box_h / 2
                true_box_ymax = true_box_x_center + true_box_h / 2
                iou = calc_iou([pred_box_xmin, pred_box_ymin, pred_box_xmax, pred_box_ymax], [true_box_xmin, true_box_ymin, true_box_xmax, true_box_ymax])
                ious[img_index, y_grid_index, x_grid_index] = iou
            label[:, :, :, i * (5 + self.num_classes) + 4] = label[:, :, :, i * (5 + self.num_classes) + 4].clone() * ious
        # TODO:根据乘iou之后的label和model_output计算损失
        part_one = []
        part_two = []
        part_three = []
        part_four = []
        part_five = []
        for i in range(self.anchor_box_count):
            current_anchor_box_obj = obj_index_memory[i]
            current_anchor_box_noobj = noobj_index_memory[i]
            current_anchor_obj_output_info = model_output[current_anchor_box_obj][:, i * (5 + self.num_classes):(i + 1) * (5 + self.num_classes)]
            current_anchor_noobj_output_info = model_output[current_anchor_box_noobj][:, i * (5 + self.num_classes):(i + 1) * (5 + self.num_classes)]
            current_anchor_obj_label_info = label[current_anchor_box_obj][:, i * (5 + self.num_classes):(i + 1) * (5 + self.num_classes)]
            current_anchor_noobj_label_info = label[current_anchor_box_noobj][:, i * (5 + self.num_classes):(i + 1) * (5 + self.num_classes)]
            part_one.append(((current_anchor_obj_output_info[:, :2] - current_anchor_obj_label_info[:, :2]) ** 2).sum().view(1))
            # part_two.append(((current_anchor_obj_output_info[:, 2:4] - current_anchor_obj_label_info[:, 2:4]) ** 2).sum().view(1))
            part_two.append(smoooth_l1(current_anchor_obj_output_info[:, 2:4] - current_anchor_obj_label_info[:, 2:4]).view(1))
            part_three.append(((current_anchor_obj_output_info[:, 4] - current_anchor_obj_label_info[:, 4]) ** 2).sum().view(1))
            part_four.append(((current_anchor_noobj_output_info[:, 4] - current_anchor_noobj_label_info[:, 4]) ** 2).sum().view(1))
            part_five.append(((current_anchor_obj_output_info[:, 5:] - current_anchor_obj_label_info[:, 5:]) ** 2).sum().view(1))
        total_loss = self.lamda_coord * t.cat(tuple(part_one)).sum() + self.lamda_coord * t.cat(tuple(part_two)).sum() + self.lamda_obj * t.cat(tuple(part_three)).sum() + self.lamda_noobj * t.cat(tuple(part_four)).sum() + self.lamda_class * t.cat(tuple(part_five)).sum()
        return total_loss


def smoooth_l1(x):
    part_one = (0.5 * x[t.abs(x) < 1] ** 2).sum()
    part_two = (t.abs(x[t.abs(x) >= 1]) - 0.5).sum()
    return part_one + part_two

