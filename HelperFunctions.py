import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import ImageDraw, Image, ImageFont
from Line import leftLine, rightLine
from statistics import mean


class HelperFunctions:

    def calibrateCamera(images):
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
        return cv2.calibrateCamera(object_points, image_points, gray_image_shape, None, None)

    def getUndistoredImage(image, cam_matrix, dist_co):
        return cv2.undistort(image, cam_matrix, dist_co, None, cam_matrix)

    def createThresholdBinaryImage(img, s_thresh=(170, 255), h_thresh=(45, 70), sx_thresh=(30, 150)):
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        h_channel = hls[:, :, 0]
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channels
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        h_binary = np.zeros_like(h_channel)
        h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1) | (h_binary == 1)] = 1

        return combined_binary

    def performPerspectiveTransform(image, source, destination):
        m = cv2.getPerspectiveTransform(source, destination)
        inverse_perspective_transform = cv2.getPerspectiveTransform(destination, source)
        image_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, m, image_size, flags=cv2.INTER_LINEAR)
        return warped, m, inverse_perspective_transform

    def sourceVertices(src):
        verticesLeft = np.array([[(int(src[0, 0]), int(src[0, 1]))], [(int(src[1, 0]), int(src[1, 1]))]])
        verticesRight = np.array([[(int(src[2, 0]), int(src[2, 1]))], [(int(src[3, 0]), int(src[3, 1]))]])
        points = np.concatenate((verticesLeft, verticesRight))
        return points

    def findLaneLinesFromBinaryImage(binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
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
        minpix = 50
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
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
            # (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
            # (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
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

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_curve, right_curve = HelperFunctions.calculateRadiusCurvature(leftx, rightx, lefty, righty)
        good = HelperFunctions.sanityCheck(leftx, rightx, left_curve, right_curve)

        leftLine.allx = leftx
        leftLine.ally = lefty
        rightLine.allx = rightx
        rightLine.ally = righty
        leftLine.bestx = left_fitx
        rightLine.bestx = right_fitx

        leftLine.detected = True
        rightLine.detected = True
        # plt.imshow(out_img)
        # plt.show(block=True)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        leftLine.current_fit = left_fit
        rightLine.current_fit = right_fit
        leftLine.iterations = 1
        return ploty

    def findLaneLinesSkipSlidingWindow(binary_warped):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (leftLine.current_fit[0] * (nonzeroy ** 2) + leftLine.current_fit[1] * nonzeroy +
                                       leftLine.current_fit[2] - margin)) & (nonzerox < (leftLine.current_fit[0] * (nonzeroy ** 2) +
                                                                                         leftLine.current_fit[1] * nonzeroy + leftLine.current_fit[
                                                                                 2] + margin)))

        right_lane_inds = ((nonzerox > (rightLine.current_fit[0] * (nonzeroy ** 2) + rightLine.current_fit[1] * nonzeroy +
                                        rightLine.current_fit[2] - margin)) & (nonzerox < (rightLine.current_fit[0] * (nonzeroy ** 2) +
                                                                                           rightLine.current_fit[1] * nonzeroy + rightLine.current_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # plt.imshow(result)
        # plt.show(block=True)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        left_curve, right_curve = HelperFunctions.calculateRadiusCurvature(leftx, rightx, lefty, righty)
        good = HelperFunctions.sanityCheck(leftx, rightx, left_curve, right_curve)

        if good:
            leftLine.allx = leftx
            leftLine.ally = lefty
            rightLine.allx = rightx
            rightLine.ally = righty
            leftLine.current_fit = left_fit
            rightLine.current_fit = right_fit
            leftLine.bestx = left_fitx
            rightLine.bestx = right_fitx

        return ploty

    def calculateRadiusCurvature(leftx, rightx, lefty, righty):
        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = 720

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        return left_curverad, right_curverad

    def sanityCheck(leftx, rightx, left_curverad, right_curverad):
        midpointFromLeft = (((min(leftx) + max(rightx)) / 2) - min(leftx)) * (3.7 / 700)

        if not (midpointFromLeft < 2):
            leftLine.iterations = leftLine.iterations + 1
            if leftLine.iterations > 10:
                leftLine.detected = False
                rightLine.detected = False
            return False
        leftLine.line_base_pos = midpointFromLeft
        leftLine.radius_of_curvature = left_curverad
        rightLine.radius_of_curvature = right_curverad
        return True

    def drawLaneLineOnOriginalImage(warped_binary, undistored_image, inverse_perspective, ploty):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([leftLine.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rightLine.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, inverse_perspective,
                                      (undistored_image.shape[1], undistored_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistored_image, 1, newwarp, 0.3, 0)

        # Add text from Line object
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(result, "Radius of Curvature Left = {} (m)".format(leftLine.radius_of_curvature), (10, 50), font, 2, (255, 255, 255), 2, cv2.LINE_4)
        cv2.putText(result, "Distance from left line to center: {}".format(leftLine.line_base_pos), (10, 150), font, 2, (255, 255, 255), 2, cv2.LINE_4)
        plt.imshow(result)
        plt.show(block=True)
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        return result
