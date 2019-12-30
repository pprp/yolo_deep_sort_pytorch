# -*- encoding: utf-8 -*-
'''
@File    :   predict.py
@Time    :   2019/12/29 16:33:04
@Author  :   pprp
@Contact :   1115957667@qq.com
@License :   (C)Copyright 2018-2019
@Desc    :   None
'''

# here put the import lib
import torch
import time
import cv2
import numpy as np
import os
from PIL import Image

from models import *
from utils.datasets import *
from utils.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class InferYOLOv3(object):
    def __init__(self,
                 cfg,
                 img_size,
                 weight_path,
                 data_cfg,
                 device,
                 conf_thres=0.5,
                 nms_thres=0.5):
        self.cfg = cfg
        self.img_size = img_size
        self.weight_path = weight_path
        # self.img_file = img_file
        self.device = device
        self.model = Darknet(cfg).to(device)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=device)['model'])
        self.model.to(device).eval()
        self.classes = load_classes(parse_data_cfg(data_cfg)['names'])
        self.colors = [random.randint(0, 255) for _ in range(3)]
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def predict(self, im0):
        # singleDataloader = LoadSingleImages(img_file, img_size=img_size)
        # path, img, im0 = singleDataloader.__next__()

        img, _, _ = letterbox(im0, new_shape=self.img_size)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0

        # TODO: how to get img and im0

        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        det = non_max_suppression(pred, self.conf_thres, self.nms_thres)[0]

        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                      im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, self.classes[int(c)]), end=', ')

            img = np.array(img.cpu())
            # Draw bounding boxes and labels of detections

            bboxes, confs, cls_confs, cls_ids = [], [], [], []

            for *xyxy, conf, cls_conf, cls_id in det:
                # label = '%s %.2f' % (classes[int(cls_id)], conf)
                bboxes.append(xyxy)
                confs.append(conf)
                cls_confs.append(cls_conf)
                cls_ids.append(cls_id)
                # plot_one_box(xyxy, im0, label=label, color=colors)
            return np.array(bboxes), np.array(cls_confs), np.array(cls_ids)
        else:
            return None, None, None

    def plot_bbox(self, ori_img, boxes):
        img = ori_img
        height, width = img.shape[:2]
        for box in boxes:
            # get x1 x2 x3 x4
            x1 = int(round(((box[0] - box[2] / 2.0) * width).item()))
            y1 = int(round(((box[1] - box[3] / 2.0) * height).item()))
            x2 = int(round(((box[0] + box[2] / 2.0) * width).item()))
            y2 = int(round(((box[1] + box[3] / 2.0) * height).item()))
            cls_conf = box[5]
            cls_id = box[6]
            # import random
            # color = random.choices(range(256),k=3)
            color = [int(x) for x in np.random.randint(256, size=3)]
            # put texts and rectangles
            img = cv2.putText(img, self.class_names[cls_id], (x1, y1),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img

    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3,
                                     thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img,
                        label, (c1[0], c1[1] - 2),
                        0,
                        tl / 3, [225, 255, 255],
                        thickness=tf,
                        lineType=cv2.LINE_AA)


if __name__ == "__main__":
    #################################################
    cfg = './yolov3-cbam.cfg'
    img_size = 416
    weight_path = './miniversion/best.pt'
    img_file = "./miniversion/test.jpg"  #"./images/train2014/0137-2112.jpg"
    data_cfg = "./miniversion/dataset1.data"
    conf_thres = 0.5
    nms_thres = 0.5
    device = torch_utils.select_device()
    #################################################
    yolo = InferYOLOv3(cfg, img_size, weight_path, data_cfg, device)
    # bbox_xcycwh, cls_conf, cls_ids = yolo(img_file)
    # print(bbox_xcycwh.shape, cls_conf.shape, cls_ids.shape)

    img = cv2.imread(img_file)
    print(img.shape)
    # im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = img
    print(im.shape)
    bbox_xcycwh, cls_conf, cls_ids = yolo.predict(im)
    print(bbox_xcycwh.shape, cls_conf.shape, cls_ids.shape)

    bboxs = []
    for i in range(len(bbox_xcycwh)):
        bboxs.append(tuple(int(bbox_xcycwh[i][j].tolist()) for j in range(4)))
    
    print(bboxs)

