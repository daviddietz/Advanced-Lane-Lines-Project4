import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from HelperFunctions import HelperFunctions

from moviepy.editor import VideoFileClip

from Line import leftLine, rightLine


class Pipline:

    calibrationImages = glob.glob("camera_cal/calibration*.jpg")
    testPath = "test_images/test16.jpg"
    testImage = mpimg.imread(testPath)

    testPath = "test_images/test17.jpg"
    testImage1 = mpimg.imread(testPath)

    testPath = "test_images/test18.jpg"
    testImage2 = mpimg.imread(testPath)

    testPath = "test_images/test19.jpg"
    testImage3 = mpimg.imread(testPath)

    testPath = "test_images/test20.jpg"
    testImage4 = mpimg.imread(testPath)

    testPath = "test_images/test21.jpg"
    testImage5 = mpimg.imread(testPath)

    testPath = "test_images/test22.jpg"
    testImage6 = mpimg.imread(testPath)

    testPath = "test_images/test23.jpg"
    testImage7 = mpimg.imread(testPath)

    global ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector
    ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector = HelperFunctions.calibrateCamera(
    calibrationImages)

    left_fit = np.array([])
    right_fit = np.array([])


    def process_image(image):
        imageUndistorted = HelperFunctions.getUndistoredImage(image, camera_matrix, distortion_coefficients)

        binary_image = HelperFunctions.createThresholdBinaryImage(imageUndistorted)

        img_size = (imageUndistorted.shape[1], imageUndistorted.shape[0])

        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

        warped_binary_test, perspective_M, inverse_perspective = HelperFunctions.performPerspectiveTransform(binary_image,
                                                                                                             src, dst)

        if not leftLine.detected or not rightLine.detected:
            ploty = HelperFunctions.findLaneLinesFromBinaryImage(
                warped_binary_test)
        else:
            ploty = HelperFunctions.findLaneLinesSkipSlidingWindow(
                warped_binary_test)

        return HelperFunctions.drawLaneLineOnOriginalImage(warped_binary_test, imageUndistorted, inverse_perspective, ploty)


    process_image(testImage)
    process_image(testImage1)
    process_image(testImage2)
    process_image(testImage3)
    process_image(testImage4)
    process_image(testImage5)
    process_image(testImage6)
    process_image(testImage7)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # f.tight_layout()
    # ax1.imshow(binary_image, cmap='gray')
    # ax1.set_title('Binary Image', fontsize=30)
    # ax2.imshow(warped_binary_test, cmap='gray')
    # ax2.set_title('Warped Image Lines Drawn', fontsize=30)
    # plt.show(block=True)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    # f.savefig('writeup_images/warped_image_dest_points_drawn.jpg')


    # cv2.line(copy, (int(src[0, 0]), int(src[0, 1])), (int(src[1, 0]), int(src[1, 1])), color=[255, 0, 0], thickness=5)
    # cv2.line(copy, (int(src[2, 0]), int(src[2, 1])), (int(src[3, 0]), int(src[3, 1])), color=[255, 0, 0], thickness=5)

    # project_video = 'testVideo.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # test_clip = clip1.fl_image(process_image)
    # test_clip.write_videofile(project_video, audio=False)
