import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import copy

calibrationImages = glob.glob("camera_cal/calibration*.jpg")

testPath = "test_images/straight_lines1.jpg"
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


def createThresholdBinaryImage(img, s_thresh=(170, 255), h_thresh=(45, 70), sx_thresh=(30, 150)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
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
    image_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, m, image_size, flags=cv2.INTER_LINEAR)
    return warped, m


def region_of_interest(img, vertices):
    """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, color=[152, 251, 152])

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def sourceVertices(src):
    verticesLeft = np.array([[(int(src[0, 0]), int(src[0, 1]))], [(int(src[1, 0]), int(src[1, 1]))]])
    verticesRight = np.array([[(int(src[2, 0]), int(src[2, 1]))], [(int(src[3, 0]), int(src[3, 1]))]])
    points = np.concatenate((verticesLeft, verticesRight))
    return points

ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector = calibrateCamera(calibrationImages)

testUndistored = getUndistoredImage(testImage, camera_matrix, distortion_coefficients)

binary_image = createThresholdBinaryImage(testUndistored)

img_size = (testUndistored.shape[1], testUndistored.shape[0])
copy = copy.copy(testUndistored)

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

cv2.line(copy, (int(src[0, 0]), int(src[0, 1])), (int(src[1, 0]), int(src[1, 1])), color=[255, 0, 0], thickness=5)
cv2.line(copy, (int(src[2, 0]), int(src[2, 1])), (int(src[3, 0]), int(src[3, 1])), color=[255, 0, 0], thickness=5)

cv2.fillPoly(copy, [sourceVertices(src)], color=[152, 251, 152])

warped_binary_test, perspective_M = performPerspectiveTransform(binary_image, src, dst)

#cv2.line(warped_binary_test, (int(dst[0, 0]), int(dst[0, 1])), (int(dst[1, 0]), int(dst[1, 1])), color=[255, 66, 66], thickness=5)
#cv2.line(warped_binary_test, (int(dst[2, 0]), int(dst[2, 1])), (int(dst[3, 0]), int(dst[3, 1])), color=[255, 66, 66], thickness=5)

# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(binary_image, cmap='gray')
# ax1.set_title('Binary Image', fontsize=30)
# ax2.imshow(warped_binary_test, cmap='gray')
# ax2.set_title('Warped Image Lines Drawn', fontsize=30)
# plt.show(block=True)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#f.savefig('writeup_images/warped_image_dest_points_drawn.jpg')


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
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
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
    return left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds


left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = findLaneLinesFromBinaryImage(warped_binary_test)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_image.shape[0]-1, binary_image.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.show(block=True)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
