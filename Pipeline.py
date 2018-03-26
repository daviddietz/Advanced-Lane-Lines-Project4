import glob
from HelperFunctions import HelperFunctions

from moviepy.editor import VideoFileClip


class Pipeline:
    calibrationImages = glob.glob("camera_cal/calibration*.jpg")

    HelperFunctions().calibrateCamera(calibrationImages)

    HelperFunctions().runTestImages()

    # project_video = 'output_video.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # test_clip = clip1.fl_image(HelperFunctions().process_image)
    # test_clip.write_videofile(project_video, audio=False)

