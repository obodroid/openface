import numpy as np
import cv2
from datetime import datetime

landmarks_size = 5  # not working
landmarks_size = 68


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((landmarks_size, 2), dtype=dtype)

    # loop over the all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, landmarks_size):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)


def draw_head_direction(image, p1, p2):
    cv2.line(image, p1, p2, (0, 0, 255), 2)


def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, dist_coeefs, color=(255, 255, 255), line_width=2):
    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


def pose_estimate(image, shape):
    # """
    # Given an image and a set of facial landmarks generates the direction of pose
    # """
    landmarks = shape_to_np(shape)
    size = image.shape
    print("image size {}".format(size))

    if landmarks_size == 68:
        image_points = np.array([
            (landmarks[30, 0], landmarks[30, 1]),     # Nose tip
            (landmarks[8, 0], landmarks[8, 1]),       # Chin
            (landmarks[36, 0], landmarks[36, 1]),     # Left eye left corner
            (landmarks[45, 0], landmarks[45, 1]),     # Right eye right corner
            (landmarks[48, 0], landmarks[48, 1]),     # Left Mouth corner
            (landmarks[54, 0], landmarks[54, 1])      # Right mouth corner
        ], dtype="double")
    elif landmarks_size == 5:
        image_points = np.array([
            (landmarks[4, 0], landmarks[4, 1]),     # Nose Bottom
            (landmarks[0, 0], landmarks[0, 1]),     # Left eye left corner
            (landmarks[1, 0], landmarks[1, 1]),     # Left eye right corner
            (landmarks[2, 0], landmarks[2, 1]),     # Right eye right corner
            (landmarks[3, 0], landmarks[3, 1])      # Right eye left corner
        ], dtype="double")

    if landmarks_size == 68:
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
    elif landmarks_size == 5:
        model_points = np.array([
            (0.0, 0.0, 0.0),            # Nose Bottom
            (-225.0, 170.0, -65.0),     # Left eye left corner
            (-150.0, 170.0, -65.0),     # Left eye right corner
            (225.0, 170.0, -65.0),      # Right eye right corner
            (150.0, 170.0, -65.0)       # Right eye left corner
        ])

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    nose_end_point2D, _ = cv2.projectPoints(np.array(
        [(0.0, 0.0, 100)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    print("p1 - {}, p2 - {}".format(p1, p2))

    draw_landmarks(image, landmarks)
    draw_head_direction(image, p1, p2)
    draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    return image, p1, p2
