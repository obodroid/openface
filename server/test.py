import cv2
import numpy as np
import facenet

image_path = 'test_small.jpg'
image = cv2.imread(image_path)

if image is not None:
    image_np = np.asarray(image)
    height, width, channels = image_np.shape
    print("height {}, width {}".format(height, width))
    try:
        bbs = facenet.facenetWorkers[0].cnn_face_detector(image_np, 0)
        print("bbs: {}".format(bbs))
    except Exception as e:
        print("An unexpected error occurred: {}".format(e))