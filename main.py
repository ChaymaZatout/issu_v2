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


def gstreamer_pipeline(
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=60,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


if __name__ == "__main__":
    # YOLO conf:
    print('Yolo initialization ...')
    model = "yolov4-416"  # in yolo dir
    category_num = 80
    letter_box = False
    start = time.time()
    trt_yolo = TrtYOLO(model, category_num, letter_box)
    print('__time__' + str(time.time() - start) + ' s.')
    conf_th = 0.3
    cls_dict = get_cls_dict(category_num)

    # CAMERA:
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        h, w = 480, 640
        window_handle = cv2.namedWindow("Yolo", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("Yolo", 0) >= 0:
            # Read data:
            ret_val, img = cap.read()
            # resize:
            y = (w - h) // 2
            img = img[:, y:y + h]
            img = cv2.resize(img, (416, 416))

            # Inference:
            start = time.time()
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            print('__time__' + str(time.time() - start) + ' s.')

            # Get classes:

            # Visualize the detection:
            vis = BBoxVisualization(cls_dict)
            img = vis.draw_bboxes(img, boxes, confs, clss)
            cv2.imshow("Yolo", img)

            # Send to server:

            # Stop the program on the ESC key
            keyCode = cv2.waitKey(1) & 0xFF
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
