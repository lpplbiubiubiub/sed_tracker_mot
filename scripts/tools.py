import math
import numpy as np
from shared_parameter import P

# standard gaussian function
def gaussian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2)))

# range [vs_lo, vs_hi)
vs_lo = - P['velocity_smooth_range'] // 2
vs_hi = 1 - vs_lo
# smoothing kernel
velocity_smoothing_kernel = np.array([gaussian(x, 0, P['velocity_smooth_sigma']) for x in range(vs_lo, vs_hi)])
velocity_smoothing_kernel = velocity_smoothing_kernel / velocity_smoothing_kernel.sum()

# Compute rectified distance between two pedestrians using precomputed calibration map and Bresenham algorithm
def bresenham_line_integral(c1, c2, frame_w, frame_h, nb_grid_w, nb_grid_h, w_mat, h_mat):
    """
        [Reference of Bresenham algorithm](http://www.geeksforgeeks.org/bresenhams-line-generation-algorithm)
        Since it is error-prone if we use the same distance threshold when computing distance between any two
        locations, we need to make the threshold adaptive to the distance between the objects and the camera.
        For example, if A and B are close to the camera, then we need to apply a rather large distance threshold.
        If A and B are far from camera, then we need to apply a rather small distance threshold.
        To accomplish this, we first need to figure out how close an object (or pedestrian) is to the camera.
        Unfortunately, it is impossible to get the exact value since we are ignorant of the camera specs. A
        workaround is to make use of the result of pedestrian detector, i.e. the bounding boxes of pedestrians.
        The closer a pedestrian, the larger bounding box. We rasterize the frame into gw * gh grids
        (self._calib_grid_w x self._calib_grid_h), and compute the mean width and height of bounding boxes that
        reside in each grid. This is done using ../tools/calibration/calibration.py
        To get the distance between A and B, we have to compute line integral on the computed closeness.
        Here, we use Bresenham algorithm to determine the pixels over which the line between A and B passes and
        the corresponding closeness of each pixel. Then we can compute a coarse approximation of rectified
        distance between A and B.
        [Param]
            c1:             One of the end of line (x, y)
            c2:             Another end of line (x, y)
            frame_w & h:    width and height of frame
            nb_grid_w:      Number of grids horizontally
            nb_grid_h:      Number of grids vertically
            w_mat & h_mat:  Calibration map of size nb_grid_h * nb_grid_w
    """
    x1, y1, x2, y2 = c1[0], c1[1], c2[0], c2[1]
    if x2 < x1:
        return bresenham_line_integral(c2, c1, frame_w, frame_h, nb_grid_w, nb_grid_h, w_mat, h_mat)
    if abs(x2 - x1) < abs(y2 - y1):
        # avoid vertical line
        return bresenham_line_integral(c1[::-1], c2[::-1], frame_h, frame_w, nb_grid_h, nb_grid_w, h_mat.T, w_mat.T)

    last_pixel = None
    dist, slope, delta = 0., 2*(y2-y1), x2-x1
    slope_error = slope - delta
    dx, dy, last_dy = x1, y1, None

    while dx <= x2:
        gx = max(0, min(int(math.floor(float(dx)/frame_w * nb_grid_w)), nb_grid_w - 1))
        gy = max(0, min(int(math.floor(float(dy)/frame_h * nb_grid_h)), nb_grid_h - 1))
        rx, ry = w_mat[gy, gx], h_mat[gy, gx]
        if last_pixel is not None:
            if last_dy == dy:
                dist += ry
            else:
                dist += math.sqrt(rx*rx + ry*ry)
        last_pixel = (dx, dy)
        last_dy = dy
        slope_error += slope
        if slope_error >= 0:
            dy += 1
            slope_error -= 2 * delta
        dx += 1
    return dist
