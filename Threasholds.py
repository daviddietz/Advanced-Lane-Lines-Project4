import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# Read in an image and grayscale it
image = mpimg.imread('test_images/test6.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Choose a Sobel kernel size
ksize = 7


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'y':
        sorbel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    else:
        # Default to x
        sorbel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)

    # Take the absolute value of the derivative or gradient
    absolute_sobel = np.absolute(sorbel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * absolute_sobel / np.max(absolute_sobel))

    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    masked_image = np.zeros_like(scaled_sobel)
    masked_image[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return this mask as your binary_output image
    grad_binary = masked_image
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Calculate the magnitude
    sobelxy = np.square(sobelx) + np.square(sobely)
    magnitude = np.sqrt(sobelxy)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))

    # Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # Return this mask as your binary_output image
    magnitude_binary = sxbinary
    return magnitude_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction_gradient = np.arctan2(abs_sobely, abs_sobelx)

    # Create a binary mask where direction thresholds are met
    sxbinary = np.zeros_like(direction_gradient)
    sxbinary[(direction_gradient >= thresh[0]) & (direction_gradient <= thresh[1])] = 1

    # Return this mask as your binary_output image
    direction_binary = sxbinary
    return direction_binary


def s_channel_color_threshold(img, thresh):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    # retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
    return binary


# Gradient Thresholds
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(30, 150))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 150))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 150))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Color Thresholds
s_channel_color = s_channel_color_threshold(image, thresh=(90, 255))

# Run the function
# mag_thresh = mag_thresh(image, sobel_kernel=3, thresh=(20, 100))
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(gray, cmap='gray')
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(s_channel_color, cmap='gray')
ax2.set_title('S Color', fontsize=30)
plt.show(block=True)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
