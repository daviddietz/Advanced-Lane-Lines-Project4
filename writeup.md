**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistort_test.png "Undistorted Checkerboard"
[image2]: ./output_images/Undistort_test2.png "Undistorted Road Image"
[image3]: ./output_images/binary_combo_test.jpg "Binary Example"
[image4]: ./output_images/warped_image_dest_points_drawn.jpg "Warp Example"
[image5]: ./output_images/lane_lines_calc.jpg "Fit Visual"
[image6]: ./output_images/lines_drawn.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---


### Camera Calibration

This step is executed after the calibration images are read into the program's memory. The code can be found in the HelperFuncdtions.py class within the method calibrateCamera (lines 20-37), which takes an array of images as an argument.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `object_point` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a calibration image.  `Image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `object_points` and `image_points` to compute the camera calibration, distortion coefficients, rotation_vector, and translation_vector using the `cv2.calibrateCamera()` function and set them on the CameraSettings class. I apply this distortion correction for each image using the `cv2.undistort()` function within the getUndistortedImage method. Here is this process applied to a calibration image:

![Undistorted Checkerboard][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. Here is an example:
![Undistorted Road Image][image2]

#### 2. Using color transforms to create a thresholded binary image.

After many trial and error attempts, I used color thresholding (with HSV and HLS) to generate a binary image. The method that handles this (createThresholdBinaryImage) is located in the Helperfunctions class (lines 44-90) and takes an image as an argument. I kept and commented out code that could be useful for future enhancement. Here's an example of my output for this step.

![Binary Example][image3]

#### 3. Perform a perspective transform.

Prior to performing perspective transform on the image, I masked the image in order to identify only the lane lines. This calculation assumes a center mounted camera and currently the vertices are hardcoded, which could be improved upon. The code for this process can be found in the HelperFunctions class in the regionOfInterest method (lines 93-120).
The code for my perspective transform can be found in HelperFunctions class in the performPerspectiveTransform method (lines 123-140) and takes an image as an argument. The source and destination points are hardcoded in the method using the image size. Here is what that looks like:

```python
source = np.float32(
            [[(image_size[0] / 2) - 55, image_size[1] / 2 + 100],
             [((image_size[0] / 6) - 10), image_size[1]],
             [(image_size[0] * 5 / 6) + 60, image_size[1]],
             [(image_size[0] / 2 + 55), image_size[1] / 2 + 100]])
        destination = np.float32(
            [[(image_size[0] / 4), 0],
             [(image_size[0] / 4), image_size[1]],
             [(image_size[0] * 3 / 4), image_size[1]],
             [(image_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203.333, 720      | 320, 720      |
| 1126.666, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `source` and `destination` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Warp Example][image4]

#### 4. Identifying lane-line pixels and fitting their positions with a polynomial.

Using the binary image, the findLaneLinesFromBinaryImage() method (lines 149-240 in HelperFunctions.py) identifies lane lines by created a histogram of the bottom half, and using a sliding window technique locates indices that meet specifications and recenters the window. I then fit a second order polynomial to each line and generate x,y values for plotting. Finally I set a boolean indicating each line has been detected. While processing each image if the lines have been detected the findLaneLinesSkipSlidingWindow() method (lines 242-292 in HelperFunctions.py) is used to identify the lane lines, which uses the indices from the previous image and skips the sliding window technique. Here is an example of detected lane lines on a warped binary image after perspective transform.

![Fit Visual][image5]

#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

The calcualteRadiusCurvature() method (lines 297-326 in HelperFunctions.py) takes generated x,y values that are generated after the second order polynomials are fit to each lane line. This is accomplished by fitting new polynomials in x,y space and calculating the radius of each curve.
The position of the vehicle with respect to center is calculated when drawing the lane lines on the image (lines 344-350 in HelperFunctions.py). This is done by finding the x intercepts of the polynomials and calculating half the distance between them.

#### 6. Example image of the result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the drawLaneLinesOnOriginalImage() method (lines 334-377 in HelperFunctions.py). Here is an example of my result on a test image:

![Output][image6]

---

### Pipeline (video)

#### 1. Link to final video output.  The pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Problems / issues faced in the implementation of this project.  Where will the pipeline likely fail?  What could be done to make it more robust?

One of the issues that arose during the implementation of this project is variation of hue and brightness of both the lane lines and the road. While there were some techniques used to
detect lane lines in various conditions I think this was the biggest hurdle. This program/pipline also makes many assumptions (position of camera, non drastic changes in elevation, no other road markings, etc.),
which could lead to failure. Some things that could be improved upon is: defining better sanity checks/calculations, using a neural network to help identify lines, better fault tolerance and the
ability to take in more information rather than from just one camera.

