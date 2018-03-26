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
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---


### Camera Calibration

This step is executed after the calibration images are read into the program's memory. The code can be found in the HelperFuncdtions.py class within the method calibrateCamera (lines 21-38), which takes an array of images as an argument.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `object_point` is just a replicated array of coordinates, and `object_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a calibration image.  `Image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `object_points` and `image_points` to compute the camera calibration, distortion coefficients, rotation_vector, and translation_vector using the `cv2.calibrateCamera()` function and set them on the CameraSettings class. I apply this distortion correction for each image using the `cv2.undistort()` function within the getUndistortedImage method. Here is this process applied to a calibration image:

![Undistorted Checkerboard][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images. Here is an example:
![Undistorted Road Image][image2]

#### 2. Using color transforms to create a thresholded binary image.

After many trial and error attempts, I used color thresholding (with HSV and HLS) to generate a binary image. The method that handles this (createThresholdBinaryImage) is located in the Helperfunctions class (lines 45-91) and takes an image as an argument. I kept and commented out code that could be useful for future enhancement. Here's an example of my output for this step.

![Binary Example][image3]

#### 3. Perform a perspective transform.

Prior to performing perspective transform on the image, I masked the image in order to identify only the lane lines. This calculation assumes a center mounted camera and currently the vertices are hardcoded, which could be improved upon. The code for this process can be found in the HelperFunctions class in the regionOfInterest method (lines 94-121).
The code for my perspective transform can be found in HelperFunctions class in the performPerspectiveTransform method (lines 124-141) and takes an image as an argument. The source and destination points are hardcoded in the method using the image size. Here is what that looks like:

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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
