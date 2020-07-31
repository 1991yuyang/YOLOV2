import numpy as np
from xml.etree import ElementTree as ET
import os
from utils import calc_iou
from numpy import random as rd
import matplotlib.pyplot as plt


with open("./conf.json", "r", encoding="utf-8") as file:
    conf = eval(file.read())
txt_file_path = conf["txt_file_path"]
xml_file_path = conf["xml_file_path"]
cluster_count = conf["cluster_count"]
kmeans_iter_times = conf["kmeans_iter_times"]
with open(os.path.join(txt_file_path, "train.txt"), "r", encoding="utf-8") as file:
    train_xml_paths = [os.path.join(xml_file_path, "%s.xml" % (name,)) for name in file.read().strip("\n").split("\n")]


def draw(clusters, cluster_center):
    """
    将所有样本点聚类结果画出
    """
    ax = plt.subplot(1, 1, 1)
    for i in range(cluster_center.shape[0]):
        current_cluster = np.array(clusters[i])
        current_center = cluster_center[i]
        ax.scatter(current_cluster[:, 0], current_cluster[:, 1], s=3)
        ax.scatter(current_center[0], current_center[1], marker="*", c="black", s=30)
    ax.set_xlabel("height")
    ax.set_ylabel("width")
    plt.show()


def kmeans_double_plus(cluster_count, train_xml_paths, kmeans_iter_times):
    """
    对训练集中的所有bounidng box进行kmeans++聚类
    :param cluster_count: 簇的个数
    :param train_xml_path: 训练集标记xml文件目录
    :param kmeans_iter_times: kmeans迭代次数
    :return: None
    """
    all_box_w_h = []
    cluster_center_point = []
    for pth in train_xml_paths:
        xml_etree = ET.parse(pth)
        for obj in xml_etree.findall("object"):
            bbx = obj.find("bndbox")
            xmin = float(bbx.find("xmin").text)
            xmax = float(bbx.find("xmax").text)
            ymin = float(bbx.find("ymin").text)
            ymax = float(bbx.find("ymax").text)
            w = xmax - xmin
            h = ymax - ymin
            all_box_w_h.append([h, w])
    all_box_w_h = np.array(all_box_w_h)
    for i in range(cluster_count):
        if i == 0:
            cluster_center_point.append(all_box_w_h[rd.randint(all_box_w_h.shape[0])])
            continue
        closest_distace_of_every_sample = []  # 用于存储每个样本距其最近的聚类中心的距离
        for sample in all_box_w_h:
            box2_xmin = 0 - sample[1] / 2
            box2_xmax = 0 + sample[1] / 2
            box2_ymin = 0 - sample[0] / 2
            box2_ymax = 0 + sample[0] / 2
            box2 = [box2_xmin, box2_ymin, box2_xmax, box2_ymax]
            current_minist = float("inf")
            for center in cluster_center_point:
                box1_xmin = 0 - center[1] / 2
                box1_xmax = 0 + center[1] / 2
                box1_ymin = 0 - center[0] / 2
                box1_ymax = 0 + center[0] / 2
                box1 = [box1_xmin, box1_ymin, box1_xmax, box1_ymax]
                distance = 1 - calc_iou(box1, box2)
                if distance < current_minist:
                    current_minist = distance
            closest_distace_of_every_sample.append(current_minist)
        p_cumsum = np.cumsum(np.array(closest_distace_of_every_sample) / np.sum(closest_distace_of_every_sample))
        rand_num = rd.random()
        next_center_index = np.sum(rand_num >= p_cumsum)
        cluster_center_point.append(all_box_w_h[next_center_index])
    # 对所有cluster_center_point中的聚类中心使用kmeans进行调整
    for i in range(kmeans_iter_times):
        clusters = [[] for i in range(cluster_count)]
        for sample in all_box_w_h:
            box2_xmin = 0 - sample[1] / 2
            box2_xmax = 0 + sample[1] / 2
            box2_ymin = 0 - sample[0] / 2
            box2_ymax = 0 + sample[0] / 2
            box2 = [box2_xmin, box2_ymin, box2_xmax, box2_ymax]
            distances = []
            for center_index in range(cluster_count):
                center = cluster_center_point[center_index]
                box1_xmin = 0 - center[1] / 2
                box1_xmax = 0 + center[1] / 2
                box1_ymin = 0 - center[0] / 2
                box1_ymax = 0 + center[0] / 2
                box1 = [box1_xmin, box1_ymin, box1_xmax, box1_ymax]
                distance = 1 - calc_iou(box1, box2)
                distances.append(distance)
            cluster_index = int(np.argmin(distances))
            clusters[cluster_index].append(sample)
        for j in range(cluster_count):
            cluster_center_point[j] = np.mean(clusters[j], axis=0)
        print("kmeans iteration %d cluster center point:" % (i + 1), cluster_center_point)
    cluster_center_point = np.array(cluster_center_point)
    np.save("./cluster_anchor_boxes.npy", cluster_center_point)
    draw(clusters, cluster_center_point)


if __name__ == "__main__":
    kmeans_double_plus(cluster_count, train_xml_paths, kmeans_iter_times)