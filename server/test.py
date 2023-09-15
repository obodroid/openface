import cv2
import os
import glob
import time
import numpy as np
# import facenet
import dlib
import config

args = config.loadConfig()
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.facePredictor)

for image_path in glob.glob(os.path.join("../images/examples", "test_*.jpg")):
    print("Processing file: {}".format(image_path))
    time.sleep(1)
    image = cv2.imread(image_path)

    if image is not None:
        image_np = np.asarray(image)
        height, width, channels = image_np.shape
        print("height {}, width {}".format(height, width))
        try:
            # bbs = facenet.facenetWorkers[0].cnn_face_detector(image_np, 0)
            bbs = cnn_face_detector(image_np, 0)
            print("bbs: {}".format(bbs))
        except Exception as e:
            print("An unexpected error occurred: {}".format(e))
