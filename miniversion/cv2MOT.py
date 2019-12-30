import glob
import os
import sys
import time
from random import randint

import cv2
import numpy as np
import torch
from PIL import Image

from models import *
from utils.datasets import *
from utils.utils import *
# from utils.utils import xyxy2xywh

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

trackerTypes = [
    'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'
]


def coordTrans(x):
    # from x1,y1,x2,y2 to x1,y1,w,h
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]  #(x[:, 0] + x[:, 2]) / 2
    y[:, 1] = x[:, 1]  #(x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


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
    weight_path = './best.pt'
    img_file = "./test.jpg"  #"./images/train2014/0137-2112.jpg"
    data_cfg = "./dataset1.data"
    conf_thres = 0.5
    nms_thres = 0.5
    device = torch_utils.select_device()
    trackerType = "BOOSTING"
    videoPath = "./demo.mp4"
    display_width = 800
    display_height = 600
    #################################################
    yolo = InferYOLOv3(cfg,
                       img_size,
                       weight_path,
                       data_cfg,
                       device,
                       conf_thres=conf_thres,
                       nms_thres=nms_thres)
    cap = cv2.VideoCapture(videoPath)
    _, frame = cap.read()

    bbox_xyxy, cls_conf, cls_ids = yolo.predict(frame)
    print("Shape of Frame:", frame.shape)
    print("Using %s algorithm." % trackerType)
    bboxes = []
    colors = []
    if bbox_xyxy is not None:
        for i in range(len(bbox_xyxy)):
            # we need left, top, w, h
            bbox_cxcywh = coordTrans(bbox_xyxy)
            bboxes.append(
                tuple(int(bbox_cxcywh[i][j].tolist()) for j in range(4)))
            colors.append((randint(64, 255), randint(64,
                                                     255), randint(64, 255)))

    print('Selected bounding boxes {}[x1,y1,w,h]'.format(bboxes))

    del yolo

    # '''
    # test for the first image
    # '''

    # for i, bbox in enumerate(bboxes):
    #     p1 = (int(bbox[0]), int(bbox[1]))
    #     p2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    #     cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
    # cv2.imwrite("./test_output.jpg", frame)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("out.avi", fourcc, 24,
                          (frame.shape[1], frame.shape[0]))
    multiTracker = cv2.MultiTracker_create()

    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("test", display_width, display_height)

    cnt = 0

    # Process video and track objects
    while cap.isOpened():
        success, frame = cap.read()
        cnt += 1
        print(cnt, end='\r')
        sys.stdout.flush()

        if cnt > 1000:
            break

        if not success:
            break

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            # x1,y1,w,h =
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[2]+newbox[0]), int(newbox[1]+newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
        out.write(frame)
        # show frame
        # cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        # if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        #     break
    os.system("mv out.avi %s.avi"%(trackerType))
    os.system("ffmpeg -y -i out.avi -r 10 -b:a 32k output.mp4")
