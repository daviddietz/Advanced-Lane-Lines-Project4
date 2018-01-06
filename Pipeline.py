import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import glob
import logging

logger = logging.getLogger('Pipline_log')

calibrationImages = glob.glob("camera_cal/calibration*.jpg")

testPath = "camera_cal/calibration3.jpg"
testImage = cv2.imread(testPath)

object_points = []  # 3d point in real world space
image_points = []  # 2d points in image plane.


def calibrateCamera(images):

    object_point = np.zeros((9 * 6, 3), np.float32)
    object_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for file in images:
        image = mpimg.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_shape = gray.shape[::1]
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            image_points.append(corners)
            object_points.append(object_point)
    return cv2.calibrateCamera(object_points, image_points, gray_image_shape, None, None)


def getUndistoredImage(image, cam_matrix, dist_co):
    return cv2.undistort(image, cam_matrix, dist_co, None, cam_matrix)


ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector = calibrateCamera(calibrationImages)

# testUndistored = getUndistoredImage(testImage, camera_matrix, distortion_coefficients)
#
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(testImage)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(testUndistored)
# ax2.set_title('Undistorted Image', fontsize=30)
# #plt.show(block=True)
# #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#
# f.savefig('Undistort_test.png')
