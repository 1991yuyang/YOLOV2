import numpy as np
import cv2
from numpy import random as rd
from PIL import Image
from torchvision import transforms as T


def nms(all_box_info, nms_iou_thresh):
    """

    :param all_box_info: 所有预测bounding box的信息： {"
                                                            bird": [
                                                                        {"coord": [xmin1, ymin1, xmax1, ymax1], "class_prob": 0.2},
                                                                        {"coord": [xmin2, ymin2, xmax2, ymax2], "class_prob": 0.2},
                                                                  ....],
                                                            "person": [
                                                                        {"coord": [xmin3, ymin3, xmax3, ymax3], "class_prob": 0.2},
                                                                        ["coord": [xmin4, ymin4, xmax4, ymax4], "class_prob": 0.1],
                                                                  ....],
                                                        ....}
    :param nms_iou_thresh: nms的iou阈值，同一物体类别的所有框中，iou大于阈值的框需要删除
    :return:
    """
    for class_name in all_box_info:
        current_class_boxes_after_nms = []
        current_class_boxes = all_box_info[class_name]
        # 对所有当前类别的box按照类别概率排序
        current_class_boxes = sorted(current_class_boxes, key=lambda x: x["class_prob"], reverse=True)

        while current_class_boxes:
            current_class_boxes_after_nms.append(current_class_boxes.pop(0))
            delete_indexs = []  # 记录要删除的box的index
            for i in range(len(current_class_boxes)):
                # 一次找出和current_class_boxes_after_nms[-1]的iou大于阈值的索引
                box1 = current_class_boxes_after_nms[-1]["coord"]
                box2 = current_class_boxes[i]["coord"]
                iou = calc_iou(box1, box2)
                if iou > nms_iou_thresh:
                    delete_indexs.append(i)
            delete_indexs = np.array(delete_indexs)
            for i in range(len(delete_indexs)):
                index_ = delete_indexs[i]
                current_class_boxes.pop(index_)
                delete_indexs -= 1
        all_box_info[class_name] = current_class_boxes_after_nms
    return all_box_info


def calc_iou(box1, box2):
    """
    计算两个bounidng box的iou
    :param box1: [xmin, ymin, xmax, ymax]
    :param box2: [xmin, ymin, xmax, ymax]
    :return: iou值
    """
    right_most_left = np.min([box1[2], box2[2]])
    left_most_right = np.max([box1[0], box2[0]])
    top_most_bottom = np.max([box1[1], box2[1]])
    bottom_most_top = np.min([box1[3], box2[3]])
    if right_most_left <= left_most_right or top_most_bottom >= bottom_most_top:
        return 0
    s_inter = (right_most_left - left_most_right) * (bottom_most_top - top_most_bottom)
    s_sum = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = s_inter / (s_sum - s_inter)
    return iou


def RandomShift(cv_img):
    is_shift = rd.random() < 0.5
    original_h = cv_img.shape[0]
    original_w = cv_img.shape[1]
    h_shift_distance = rd.randint(int(-cv_img.shape[0] / 4), int(cv_img.shape[0] / 4))
    w_shift_distance = rd.randint(int(-cv_img.shape[1] / 4), int(cv_img.shape[1] / 4))
    dst = np.zeros(cv_img.shape, dtype=np.uint8)
    if is_shift:
        dst_h_begin = 0 + h_shift_distance if 0 + h_shift_distance > 0 else 0
        dst_h_end = original_h + h_shift_distance if original_h + h_shift_distance < original_h else original_h
        dst_w_begin = 0 + w_shift_distance if 0 + w_shift_distance > 0 else 0
        dst_w_end = original_w + w_shift_distance if original_w + w_shift_distance < original_w else original_w
        ori_h_begin = 0 if h_shift_distance > 0 else -h_shift_distance
        ori_h_end = original_h - h_shift_distance if h_shift_distance > 0 else original_h
        ori_w_begin = 0 if w_shift_distance > 0 else -w_shift_distance
        ori_w_end = original_w - w_shift_distance if w_shift_distance > 0 else original_w
        dst[dst_h_begin:dst_h_end, dst_w_begin:dst_w_end, :] = cv_img[ori_h_begin:ori_h_end, ori_w_begin:ori_w_end, :]
        return Image.fromarray(dst), h_shift_distance, w_shift_distance
    return Image.fromarray(cv_img), 0, 0


def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def HSV2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def RandomBrightness(bgr):
    if rd.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = rd.choice(np.linspace(0.5, 2, 5, endpoint=True))
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomSaturation(bgr):
    if rd.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv2.split(hsv)
        adjust = rd.choice(np.linspace(0.5, 2, 5, endpoint=True))
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def data_augmentation(img_pth):
    cv_img = cv2.imread(img_pth)
    cv_img = RandomBrightness(cv_img)
    cv_img = RandomSaturation(cv_img)
    img, h_shift, w_shift = RandomShift(cv_img)
    return img, h_shift, w_shift