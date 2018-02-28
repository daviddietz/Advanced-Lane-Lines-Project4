import numpy as np


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self, detected=False, recent_xfitted=None, bestx=[], best_fit=None, current_fit=None, radius_of_curvature=None, line_base_pos=None, diffs=np.array([0, 0, 0], dtype='float'), allx=None, ally=None, iterations=1, ploty=None):
        # was the line detected in the last iteration?
        self.detected = detected

        # x values of the last n fits of the line
        if recent_xfitted is None:
            recent_xfitted = []
        self.recent_xfitted = recent_xfitted

        # average x values of the fitted line over the last n iterations
        self.bestx = bestx

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = best_fit

        # polynomial coefficients for the most recent fit
        if current_fit is None:
            current_fit = [np.array([False])]
        self.current_fit = current_fit

        # radius of curvature of the line in some units
        self.radius_of_curvature = radius_of_curvature

        # distance in meters of vehicle center from the line
        self.line_base_pos = line_base_pos

        # difference in fit coefficients between last and new fits
        self.diffs = diffs

        # x values for detected line pixels
        self.allx = allx

        # y values for detected line pixels
        self.ally = ally

        self.iterations = iterations


leftLine = Line()
rightLine = Line()
