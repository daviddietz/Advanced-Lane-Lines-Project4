import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ImageInfo import imageInfo
from CameraSettings import cameraSettings
from PIL import ImageDraw, Image, ImageFont
from Line import leftLine, rightLine


class HelperFunctions(object):

    @staticmethod
    def setCameraSettings(ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector):
        cameraSettings.ret = ret
        cameraSettings.camera_matrix = camera_matrix
        cameraSettings.distortion_coefficients = distortion_coefficients
        cameraSettings.rotation_vector = rotation_vector
        cameraSettings.translation_vector = translation_vector

    def calibrateCamera(self, images):
        object_points = []  # 3d point in real world space
        image_points = []  # 2d points in image plane.

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
        ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector = cv2.calibrateCamera(
            object_points, image_points, gray_image_shape, None, None)
        self.setCameraSettings(ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector)

    @staticmethod
    def getUndistoredImage(image, cam_matrix, dist_co):
        return cv2.undistort(image, cam_matrix, dist_co, None, cam_matrix)

    @staticmethod
    def createThresholdBinaryImage(img):
        image = np.copy(img)
        HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        HSV_H_channel = HSV[:, :, 0]
        HSV_S_channel = HSV[:, :, 1]
        HSV_V_channel = HSV[:, :, 2]

        HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        HLS_H_channel = HLS[:, :, 0]
        HLS_L_channel = HLS[:, :, 1]
        HLS_S_channel = HLS[:, :, 2]

        sensitivity_1 = 65
        sensitivity_2 = 60

        # For yellow
        yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

        # For white
        white = cv2.inRange(HSV, (0, 0, 255 - sensitivity_1), (255, 20, 255))
        white_2 = cv2.inRange(HLS, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
        white_3 = cv2.inRange(image, (200, 200, 200), (255, 255, 255))

        # Sobel x
        # sobelx = cv2.Sobel(HSV_V_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        # abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        # scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        # sx_thresh_min = 190
        # sx_thresh_max = 255
        # sxbinary = np.zeros_like(scaled_sobel)
        # sxbinary[(scaled_sobel >= sx_thresh_min) & (scaled_sobel <= sx_thresh_max)] = 1

        # Threshold color channel s
        # s_thresh_min = 170
        # s_thresh_max = 255
        # s_binary = np.zeros_like(HLS_S_channel)
        # s_binary[(HLS_S_channel >= s_thresh_min) & (HLS_S_channel <= s_thresh_max)] = 1

        # Combine the two binary thresholds
        # combined_binary = np.zeros_like(sxbinary)
        # combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        binary_output = yellow | white | white_2 | white_3

        return binary_output

    @staticmethod
    def region_of_interest(img):
        imageShape = img.shape
        imageInfo.imageShape = imageShape

        # currently vertices are hardcoded for project images. These should be calculated or part of camera calibration
        vertices = np.array([[(150, imageShape[0]), (600, 425), (725, 425), (1250, imageShape[0])]], dtype=np.int32)
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(imageShape) > 2:
            channel_count = imageShape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    @staticmethod
    def performPerspectiveTransform(image, source, destination):
        m = cv2.getPerspectiveTransform(source, destination)
        inverse_perspective_transform = cv2.getPerspectiveTransform(destination, source)
        image_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, m, image_size, flags=cv2.INTER_LINEAR)
        return warped, m, inverse_perspective_transform

    @staticmethod
    def sourceVertices(src):
        verticesLeft = np.array([[(int(src[0, 0]), int(src[0, 1]))], [(int(src[1, 0]), int(src[1, 1]))]])
        verticesRight = np.array([[(int(src[2, 0]), int(src[2, 1]))], [(int(src[3, 0]), int(src[3, 1]))]])
        points = np.concatenate((verticesLeft, verticesRight))
        return points

    def findLaneLinesFromBinaryImage(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 10

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 25

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        leftLine.current_fit = left_fit
        rightLine.current_fit = right_fit

        # Generate x and y values for plotting
        left_fitx = left_fit[0] * imageInfo.ploty ** 2 + left_fit[1] * imageInfo.ploty + left_fit[2]
        right_fitx = right_fit[0] * imageInfo.ploty ** 2 + right_fit[1] * imageInfo.ploty + right_fit[2]

        leftLine.bestx = left_fitx
        rightLine.bestx = right_fitx

        leftLine.best_fit = left_fit
        rightLine.best_fit = right_fit

        left_curve_radius, right_curve_radius = self.calculateRadiusCurvature(left_fitx, right_fitx)
        leftLine.radius_of_curvature = left_curve_radius
        rightLine.radius_of_curvature = right_curve_radius
        leftLine.detected = True
        rightLine.detected = True

    def findLaneLinesSkipSlidingWindow(self, binary_warped):
        leftLine.detected = False
        rightLine.detected = False
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (leftLine.current_fit[0] * (nonzeroy ** 2) + leftLine.current_fit[1] * nonzeroy +
                                       leftLine.current_fit[2] - margin)) & (
                              nonzerox < (leftLine.current_fit[0] * (nonzeroy ** 2) +
                                          leftLine.current_fit[1] * nonzeroy + leftLine.current_fit[
                                              2] + margin)))

        right_lane_inds = (
            (nonzerox > (rightLine.current_fit[0] * (nonzeroy ** 2) + rightLine.current_fit[1] * nonzeroy +
                         rightLine.current_fit[2] - margin)) & (nonzerox < (rightLine.current_fit[0] * (nonzeroy ** 2) +
                                                                            rightLine.current_fit[1] * nonzeroy +
                                                                            rightLine.current_fit[
                                                                                2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        leftLine.current_fit = left_fit
        rightLine.current_fit = right_fit

        # Generate x and y values for plotting
        imageInfo.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * imageInfo.ploty ** 2 + left_fit[1] * imageInfo.ploty + left_fit[2]
        right_fitx = right_fit[0] * imageInfo.ploty ** 2 + right_fit[1] * imageInfo.ploty + right_fit[2]

        left_curve_radius, right_curve_radius = self.calculateRadiusCurvature(left_fitx, right_fitx)
        good = self.sanityCheck(left_curve_radius, right_curve_radius)

        if good:
            leftLine.bestx = left_fitx
            rightLine.bestx = right_fitx
            leftLine.radius_of_curvature = left_curve_radius
            rightLine.radius_of_curvature = right_curve_radius
            leftLine.detected = True
            rightLine.detected = True
            leftLine.best_fit = left_fit
            rightLine.best_fit = right_fit

    @staticmethod
    def calculateRadiusCurvature(leftx, rightx):
        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(imageInfo.ploty, leftx, 2)
        left_fitx = left_fit[0] * imageInfo.ploty ** 2 + left_fit[1] * imageInfo.ploty + left_fit[2]
        right_fit = np.polyfit(imageInfo.ploty, rightx, 2)
        right_fitx = right_fit[0] * imageInfo.ploty ** 2 + right_fit[1] * imageInfo.ploty + right_fit[2]

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(imageInfo.ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(imageInfo.ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(imageInfo.ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        return round(left_curverad, 1), round(right_curverad, 1)

    def sanityCheck(self, left_curverad, right_curverad):
        # Check for similar curvature
        if abs(left_curverad - right_curverad) > 500:
            return False

        # Check for distance apart

        # Check for roughly parallelism

        return True

    @staticmethod
    def drawLaneLineOnOriginalImage(warped_binary, undistored_image, inverse_perspective):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftLine.bestx, imageInfo.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.bestx, imageInfo.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Calculate center line offset
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        image_center = imageInfo.imageShape[1] / 2
        LeftIntercept = leftLine.best_fit[0] * 720 ** 2 + leftLine.best_fit[1] * 720 + leftLine.best_fit[2]
        RightIntercept = rightLine.best_fit[0] * 720 ** 2 + rightLine.best_fit[1] * 720 + rightLine.best_fit[2]
        Center = (LeftIntercept + RightIntercept) / 2
        CenterOffset = abs((Center - image_center) * xm_per_pix)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newWarp = cv2.warpPerspective(color_warp, inverse_perspective,
                                      (undistored_image.shape[1], undistored_image.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undistored_image, 1, newWarp, 0.3, 0)
        # Add text from Line object
        font = cv2.QT_FONT_NORMAL
        cv2.putText(result, "Radius of Curvature Left = {} (meters)".format(leftLine.radius_of_curvature), (10, 50), font, 1,
                    (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(result, "Radius of Curvature Right = {} (meters)".format(rightLine.radius_of_curvature), (10, 125), font,
                    1, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(result, "Distance from left line to center: {} (meters)".format(round(CenterOffset, 2)), (10, 200), font, 1,
                    (255, 255, 255), 2, cv2.LINE_4)
        # plt.imshow(result)
        # plt.show(block=True)
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        return result

    def process_image(self, image):
        imageUndistorted = self.getUndistoredImage(image, cameraSettings.camera_matrix,
                                                   cameraSettings.distortion_coefficients)

        binary_image = self.createThresholdBinaryImage(imageUndistorted)

        img_size = (imageUndistorted.shape[1], imageUndistorted.shape[0])

        maskedImage = self.region_of_interest(binary_image)

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

        warped_binary, perspective_M, inverse_perspective = self.performPerspectiveTransform(
            maskedImage,
            src, dst)

        imageInfo.ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])

        if leftLine.detected and rightLine.detected:
            self.findLaneLinesSkipSlidingWindow(warped_binary)
        else:
            self.findLaneLinesFromBinaryImage(warped_binary)

        return self.drawLaneLineOnOriginalImage(warped_binary, imageUndistorted, inverse_perspective)

    def runTestImages(self):
        testPath = "test_images/test3.jpg"
        testImage = mpimg.imread(testPath)

        testPath = "test_images/test4.jpg"
        testImage1 = mpimg.imread(testPath)

        testPath = "test_images/test5.jpg"
        testImage2 = mpimg.imread(testPath)

        testPath = "test_images/test23.jpg"
        testImage3 = mpimg.imread(testPath)

        testPath = "test_images/test34.jpg"
        testImage4 = mpimg.imread(testPath)

        testPath = "test_images/test35.jpg"
        testImage5 = mpimg.imread(testPath)

        testPath = "test_images/test36.jpg"
        testImage6 = mpimg.imread(testPath)

        testPath = "test_images/test37.jpg"
        testImage7 = mpimg.imread(testPath)

        self.process_image(testImage)
        self.process_image(testImage1)
        self.process_image(testImage2)
        self.process_image(testImage3)
        self.process_image(testImage4)
        self.process_image(testImage5)
        self.process_image(testImage6)
        self.process_image(testImage7)
