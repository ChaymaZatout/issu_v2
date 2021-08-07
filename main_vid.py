"""
Name : main_vid.py
Author : Chayma Zatout
Contact : github.com/ChaymaZatout
Time    : 05/08/21 08:56 م
Desc:
"""
import cv2
import os
import time
import argparse

import pycuda.autoinit  # This is needed for initializing CUDA driver

from Yolo_TensorRT.utils.yolo_classes import get_cls_dict
from Yolo_TensorRT.utils.camera import add_camera_args, Camera
from Yolo_TensorRT.utils.display import open_window, set_display, show_fps
from Yolo_TensorRT.utils.visualization import BBoxVisualization
from Yolo_TensorRT.utils.yolo_with_plugins import TrtYOLO

import numpy as np
from client import Client

objects_size = {'large': 416 * 416 * 75 / 100, 'medium': 416 * 416 * 50 / 100,
                'small': 416 * 416 * 25 / 100}


def compute_distance(bb, cls):
    x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
    s = abs(x_max - x_min) * abs(y_max - y_min)
    if s >= objects_size[cls] * 75 / 100:
        return 3
    elif s >= objects_size[cls] * 25 / 100:
        return 2
    else:
        return 1


if __name__ == '__main__':
    # video reader:
    path = '/home/chim/Documents/aim-x challenge/v2/demos/'
    video_name = 'chair_2.avi'
    cap = cv2.VideoCapture(path+video_name)
    h, w = 480, 640
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # video writer:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path+'detection_'+video_name, fourcc, 20.0, (416, 416))

    # classes :
    print('Client initialization ...')
    semantic_classes = {"bench": 2, "chair": 2, "sofa": 2, "bed": 2,
                        "diningtable": 3, "sink": 3, "toilet": 4}
    distance_classes = {"bench": 'small', "chair": 'small', "sofa": "medium",
                        "bed": "large",
                        "diningtable": "medium", "sink": "medium", "toilet": 'small'}
    previous_class = -1
    client = Client()

    # YOLO conf:
    print('Yolo initialization ...')
    model = "yolov4-416"  # in yolo dir
    category_num = 80
    letter_box = False
    start = time.time()
    trt_yolo = TrtYOLO(model, category_num, letter_box)
    print('__time__' + str(time.time() - start) + ' s.')
    # Inference variables :
    conf_th = 0.3
    cls_dict = get_cls_dict(category_num)
    vis = BBoxVisualization(cls_dict)

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # resize:
            y = (w - h) // 2
            img = frame[:, y:y + h]
            img = cv2.resize(img, (416, 416))

            # Inference:
            start = time.time()
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            print('__time__' + str(time.time() - start) + ' s.')

            # Visualize the detection:
            img = vis.draw_bboxes(img, boxes, confs, clss)
            cv2.imshow("Obstacle detection", img)
            out.write(img)

            # Get classes:
            semantic_class = 0
            distance_class = 0
            if len(confs) > 0:
                max_p = np.argmax(confs)
                cl = int(clss[max_p])
                cls_name = cls_dict.get(cl, 'CLS{}'.format(cl))
                print(cls_name + " : " + str(confs[max_p]))
                if cls_name in semantic_classes.keys():
                    semantic_class = semantic_classes[cls_name]
                    distance_class = compute_distance(boxes[max_p],
                                                      distance_classes[cls_name])
                else:
                    semantic_class = 1
                    distance_class = compute_distance(boxes[max_p], 'large')

            # Send to server:
            if semantic_class != previous_class:
                previous_class = semantic_class
                data = semantic_class, distance_class
                client.send_data(*data)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
