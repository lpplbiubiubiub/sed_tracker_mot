# Bounding box class
import cv2
import math

import cPickle as pickle
import numpy as np
from collections import Iterable
from shared_parameter import P
from tools import bresenham_line_integral
BACKGROUND_CLS_ID = -1

class Box(object):
    def __init__(self, pos=(0, 0, 0, 0), frame_id=1, track_id=0, confidence=0.0, occlusion=0.0):
        self._pos, self._size, self._frame_id, self._track_id, self._confidence, self._occlusion = \
            None, None, None, None, None, None
        self.pos = pos
        self.frame_id = frame_id
        self.track_id = track_id
        self.confidence = confidence
        self.occlusion = occlusion
        self._extend_occlusion = occlusion # because head shoulder always extend head, so self.occlusion is not proper
        self._extend_pos = (pos[0], pos[1], (pos[2] - pos[0]) * P['head_shoulder_to_body_scale_x'] + pos[0],
                            (pos[3] - pos[1]) * P['head_shoulder_to_body_scale_y'] + pos[1])

        # other members
        self._color_hist = None     # color histogram
        self._velocity = None       # velocity, tuple of 2
        self._gt_track_id_list = []


    def __repr__(self):
        return '[{}]  Position:[x1={}, y1={}, x2={}, y2={}] | Frame:{} | Track Id:{}'.format(
            self.__class__.__name__,
            self._pos[0], self._pos[1], self._pos[2], self._pos[3],
            self.frame_id,
            self.track_id,
        )

    def __str__(self):
        return self.__repr__()

    @property
    def width(self):
        return self._pos[2] - self._pos[0]

    @property
    def height(self):
        return self._pos[3] - self._pos[1]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, p):
        x1, x2, y1, y2 = int(p[0]), int(p[2]), int(p[1]), int(p[3])
        assert (0 <= x1 < P['frame_width']) and (0 <= x2 < P['frame_width']), \
            "x1 and x2 must be in range [0, {}], got {} and {}".format(P['frame_width'], x1, x2)
        assert (0 <= y1 < P['frame_height']) and (0 <= y2 < P['frame_height']), \
            "y1 and y2 must be in range [0, {}], got {} and {}".format(P['frame_height'], y1, y2)
        self._pos = (x1, y1, x2, y2)
        self._size = (x2 - x1 + 1) * (y2 - y1 + 1)
    @property
    def extend_pos(self):
        extend_pos = (self._pos[0], self._pos[1], (self._pos[2] - self._pos[0]) * P['head_shoulder_to_body_scale_x'] + self._pos[0],
                            (self._pos[3] - self._pos[1]) * P['head_shoulder_to_body_scale_y'] + self._pos[1])
        extend_pos = map(lambda x: int(x), extend_pos)
        return extend_pos
    @property
    def center(self):
        return (self._pos[0] + self._pos[2])//2, (self._pos[1] + self._pos[3])//2


    @property
    def size(self):
        return self._size

    @property
    def extend_size(self):
        return self._size * P['head_shoulder_to_body_scale_x'] * P['head_shoulder_to_body_scale_y']

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, c):
        assert 0.0 <= c <= 1.0, "confidence must be in range [0, 1]"
        self._confidence = c

    @property
    def frame_id(self):
        return self._frame_id

    @frame_id.setter
    def frame_id(self, frame_id):
        assert frame_id >= 0, "frame id must be greater than 0"
        self._frame_id = frame_id

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self, track_id):
        assert track_id >= 0, "track id must be greater than 0"
        self._track_id = track_id

    @property
    def occlusion(self):
        return self._occlusion

    @occlusion.setter
    def occlusion(self, occlusion):
        assert 0.0 <= occlusion <= 1.0, "occlusion ratio must be in range[0, 1], got {}".format(occlusion)
        self._occlusion = occlusion

    @property
    def extend_occlusion(self):
        return self._extend_occlusion

    @extend_occlusion.setter
    def extend_occlusion(self, occlusion):
        assert 0.0 <= occlusion <= 1.0, "occlusion ratio must be in range[0, 1], got {}".format(occlusion)
        self._extend_occlusion = occlusion

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, v):
        assert isinstance(v, Iterable) and len(v) == 2, 'velocity must be an iterable of two value'
        self._velocity = (v[0], v[1])

    @property
    def color_hist(self):
        return self._color_hist

    def overflow(self):
        if self._pos[1] + (self._pos[3] - self._pos[1]) * P['discard_head_shoulder_to_body_scale_y'] > P['frame_height']:
            return True
        else:
            return False

    def get_color_hist(self, frame, nb_bin=P['nb_color_hist_bin']):
        # calculate color histogram of a box given a frame
        assert frame is not None, "empty frame"
        assert nb_bin >= 3, "bin num is invalid"
        roi = frame[self._pos[1]:self._pos[3], self._pos[0]:self._pos[2]]
        if roi is None:
            self._color_hist = np.zeros(shape=(1, nb_bin), dtype=np.float32)
        else:
            self._color_hist = None
            for ch in range(3):
                h = cv2.calcHist([frame], [ch], None, [P['nb_color_hist_bin']], [0, 256])
                h = (h - h.min()) / h.max()   # normalization
                if self._color_hist is None:
                    self._color_hist = h
                else:
                    self._color_hist = np.vstack([self._color_hist, h])
            self._color_hist = self._color_hist.transpose()     # color hist will be a 1 x nb_bin array

    def iou(self, other):
        # compute iou between two boxes
        assert isinstance(other, Box), "bad argument, type must be Box"
        iw = max(0, (min(self._pos[2], other.pos[2]) - max(self._pos[0], other.pos[0])))
        ih = max(0, (min(self._pos[3], other.pos[3]) - max(self._pos[1], other.pos[1])))
        intersect = iw * ih
        union = self.size + other.size - intersect
        if union == 0:
            return 0
        else:
            return float(intersect) / union

    def find_gt_id(self, gt_list_per_frame):
        """
        gt_list_per_frame(ndarray):
            numpy array, ground truth of track label
        """
        max_iou = 0.5
        iou_list = [self.iou(Box(pos=(gt_label[2], gt_label[3], gt_label[2] + gt_label[4], gt_label[3] + gt_label[5]))) for gt_label in gt_list_per_frame]
        if len(iou_list) > 0:
            iou_arr = np.array(iou_list)
            if np.max(iou_arr) > 0.5:
                self._gt_track_id_list.append(gt_list_per_frame[np.argmax(iou_arr)][1])


    @property
    def gt_id(self):
        return self._gt_track_id_list

    @staticmethod
    def box_iou(box1, box2):
        assert isinstance(box1, Box), "bad argument, type must be Box"
        return box1.iou(box2)

    def dist(self, other, metric='euclidean', **kwargs):
        """
            Compute distance between two boxes with given metric.
            Available metric:
            ['euclidean']
                euclidean distance between centers of two boxes (default)
            ['bha']
                bha distance between histograms of two boxes
            ['gaussian_size']
                distance between sizes of two boxes,
                d = K * exp(-(size1 - size2)^2 / (2 * sigma^2))
                parameter K and sigma are defined in 'shared_parameters.py'
        """
        assert isinstance(other, Box), "bad argument, type must be Box"
        if metric == 'euclidean':
            assert kwargs['w_mat'] is not None and kwargs['h_mat'] is not None, 'must supply w_mat and h_mat as params'
            w_mat, h_mat = kwargs['w_mat'], kwargs['h_mat']
            cx1, cy1 = self.center
            cx2, cy2 = other.center
            nb_grid_h, nb_grid_w = w_mat.shape
            if self.velocity is None or other.velocity is None:
                # compute euclidean distance between two boxes
                # delta_dist = bresenham_line_integral(self.center, other.center, P['frame_width'], P['frame_height'],
                                                     # nb_grid_w, nb_grid_h, w_mat, h_mat)
                delta_dist = (cx1 - cx2)**2 + (cy1 - cy2) ** 2
                return math.exp(-float(delta_dist) / P['gaussian_dist_sigma'])
            else:
                # self.velocity is computed after calling Track.smooth_velocity
                # apply velocity direction check first
                # vx_other vy_other is just calculate by center point
                vx_other, vy_other = cx2 - cx1, cy2 - cy1
                vx_this, vy_this = self._velocity
                gx_other = max(0, min(int(math.floor(float(cx2)/P['frame_width']*nb_grid_w)), nb_grid_w-1))
                gy_other = max(0, min(int(math.floor(float(cy2)/P['frame_height']*nb_grid_h)), nb_grid_h-1))
                gx_this = max(0, min(int(math.floor(float(cx1)/P['frame_width']*nb_grid_w)), nb_grid_w-1))
                gy_this = max(0, min(int(math.floor(float(cy1)/P['frame_height']*nb_grid_h)), nb_grid_h-1))

                # rectify velocity according to calibration map
                vx_this, vy_this = float(vx_this)*w_mat[gy_this, gx_this], float(vy_this)*h_mat[gy_this, gx_this]
                vx_other, vy_other = float(vx_other)*w_mat[gy_other, gx_other], float(vy_other)*h_mat[gy_other, gx_other]

                # compute inner product of two velocities
                inner_product = vx_other*vx_this + vy_other*vy_this

                # if the inner product is too small, then it may be a bad association
                if inner_product < P['th_low_level_min_velocity_ip']:

                    return 0.0

                pred_cx1, pred_cy1 = cx1 + vx_this, cy1 + vy_this
                delta_dist = (pred_cx1 - cx2) ** 2 + (pred_cy1 - cy2) ** 2
                # delta_dist = bresenham_line_integral((pred_cx1, pred_cy1), other.center, P['frame_width'], P['frame_height'],
                #                                      nb_grid_w, nb_grid_h, w_mat, h_mat)
                return math.exp(-float(delta_dist) / P['gaussian_pred_sigma'])

        elif metric == 'bha':
            # compute bha distance between histograms of two boxes
            assert self.color_hist is not None, "must compute color hist before comparing"
            assert other.color_hist is not None, "must compute color hist before comparing"
            bha = 1.0 - cv2.compareHist(self.color_hist, other.color_hist, method=cv2.HISTCMP_BHATTACHARYYA)
            return bha

        elif metric == 'gaussian_size':
            delta_size = self._size - other.size
            return math.exp(-float(delta_size**2)/P['gaussian_size_sigma'])

        else:
            raise ValueError('Unknown metric in computing distance between two boxes: {}'.format(metric))

    @staticmethod
    def box_dist(box1, box2, metric='euclidean', **kwargs):
        assert isinstance(box1, Box) and isinstance(box2, Box), "bad argument, type must be Box"
        if box1.frame_id > box2.frame_id:
            return box2.dist(box1, metric, kwargs)
        else:
            return box1.dist(box2, metric, kwargs)

    @staticmethod
    def dump_to_file(box_list, save_name):
        # dump a list of boxes into file using cPickle
        assert isinstance(box_list, list), "bad argument, type must be list"
        assert all(isinstance(box, Box) for box in box_list), "entry must be Box"
        f = open(save_name, "wb")
        pickle.dump(box_list, f)
        print("Box data saved in {}".format(save_name))
        print("Total box number: {}".format(len(box_list)))

    @staticmethod
    def compute_box_affinity(box1, box2, w_mat, h_mat):
        """
            compute affinity between two box:
                A(b1, b2) = A_dist(b1, b2) * A_size(b1, b2) * A_app(b1, b2)
            where
                A_dist(b1, b2) = Gaussian(b1.pos - b2.pos; 0, sigma2)
                A_size(b1, b2) = Gaussian(b1.size - b2.size; 0, sigma2)
                A_app(b1, b2) = Bha(b1.color_hist, b2.color_hist)
        """
        assert isinstance(box1, Box) and isinstance(box2, Box), "Box objects expected"
        nb_grid_h, nb_grid_w = w_mat.shape
        d = bresenham_line_integral(box1.center, box2.center, P['frame_width'], P['frame_height'],
                                    nb_grid_w, nb_grid_h, w_mat, h_mat)


        if d > P['th_low_level_max_dist']:
            return 0.0
        else:
            aff_dist = Box.dist(box1, box2, metric='euclidean', w_mat=w_mat, h_mat=h_mat)
            aff_size = Box.dist(box1, box2, metric='gaussian_size')
            aff_bha = Box.dist(box1, box2, metric='bha')
            return aff_bha * aff_size * aff_dist * P['box_affinity_gamma']

