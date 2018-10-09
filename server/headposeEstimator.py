import numpy as np
import cv2
from datetime import datetime

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords

def show_landmarks_and_headpose(image, landmarks, p1, p2):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    cv2.line(image, p1, p2, (0,0,255), 2)

def pose_estimate(image, shape):
    # """
    # Given an image and a set of facial landmarks generates the direction of pose
    # """
    landmarks = shape_to_np(shape)
    size = image.shape
    print("image size {}".format(size))
    image_points = np.array([
        (landmarks[33, 0], landmarks[33, 1]),     # Nose tip
        (landmarks[8, 0], landmarks[8, 1]),       # Chin
        (landmarks[36, 0], landmarks[36, 1]),     # Left eye left corner
        (landmarks[45, 0], landmarks[45, 1]),     # Right eye right corner
        (landmarks[48, 0], landmarks[48, 1]),     # Left Mouth corner
        (landmarks[54, 0], landmarks[54, 1])      # Right mouth corner
        ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
        ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    print("rotation_vector - {}".format(rotation_vector))
    nose_end_point2D, _ = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    print("p1 - {}, p2 - {}".format(p1,p2))

    show_landmarks_and_headpose(image, landmarks, p1, p2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, p1, p2