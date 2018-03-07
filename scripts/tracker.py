# coding:utf8
import cv2
import cPickle as pickle
import gc
import math
import numpy as np
import os
import shutil
import struct
import sys
sys.path.append("../tools")
sys.path.append("/home/xksj/workspace/lp/reid_caffe/caffe/reid/")
sys.path.append("/home/xksj/workspace/lp/reid_track/warper/")
from extract_feature import ExtractFeature
# from encoder import ReidNetEncoder
from visualizer import Visualizer
from box import Box
from track import Track
from shared_parameter import P
from tools import bresenham_line_integral
from scipy.optimize import linear_sum_assignment
# from cnnrnn import load_model
# from seq_cnn_rnn_nbw import load_test_model, load_encoder_model
from wandering import Wandering
from visualize_tracks import draw_tracks, get_track_frm, get_bbox_list_frm
from entry import EntryArea
import random
import commands

from event import WanderEvent
import matplotlib.pyplot as plt
####################################

class Tracker(object):
    """
        Tracker
        Init:
        [video_path]        path to video file
        [data_path]         .pkl file stores the detection results
                            assume each box in it is represented by (fid, x1, y1, x2, y2, confidence)
        [save_path]         path to saved tracking result
        [scl_path]          path to siamese cnn-lstm network pretrained on MOT16 & Trecvid dataset
        [mask_path]         path to entry-exit map
        [calib_path]        path to camera calibration map
    """

    def __init__(self, video_path, data_path, save_path, scl_path, mask_path=None, calib_path=None, eventspan_path=None, gt_path=None, entry_path=None):
        self._data_file = data_path
        self._save_dir = save_path
        self._mask_path = mask_path
        self._calib_path = None
        self._eventspan_path = eventspan_path
        self._frame_lists = {}  # dictionary stores <fid(relative), frame> pairs
        self._box_lists = {}  # dictionary stores <fid, list of Box objects> pairs
        self._total_frame_cnt = 0  # total frames count
        self._segment_cnt = 0  # number of segments to run
        self._segment_index = 0  # index of segment to be processed
        self._segment_start_fid = 0  # start fid of a segment
        self._segment_end_fid = 0  # end fid of a segment
        self._low_level_tracks = []  # list of low level tracks(Track Object)
        self._high_level_tracks = []  # list of high level tracks
        self._segments_path = []    # path to saved segments

        print(video_path)
        # open video
        self._cap = cv2.VideoCapture(video_path)  # video capture object
        assert self._cap.isOpened() is True, 'cannot open video'
        P["frame_height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        P["frame_width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_name = os.path.splitext(os.path.split(video_path)[1])[0]
        self._seq_name = self._video_name.split("-")[-1]
        self._total_frame_cnt = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._segment_cnt = \
            int(math.ceil(self._total_frame_cnt / (P['segment_length'] - P['overlap_length'])))

        # path to save the temp segment files
        subdir = os.path.join(self._save_dir, self._video_name)
        if os.path.exists(subdir) is False:
            os.mkdir(subdir)
        self._segment_dir = os.path.join(subdir, "segments")
        if os.path.exists(self._segment_dir) is False:
            os.mkdir(self._segment_dir)
        self._debug_img_dir = os.path.join(subdir, "debugging_img")
        if os.path.exists(self._debug_img_dir) is False:
            os.mkdir(self._debug_img_dir)
        # open mask
        self._mask = np.ones(shape=(P['frame_height'], P['frame_width']))
        """
        if self._mask_path is not None:
            print '[INIT] Mask found: {}'.format(self._mask_path)
            self._mask = cv2.imread(self._mask_path, flags=cv2.IMREAD_GRAYSCALE)
            cv2.threshold(self._mask, 128, 1, cv2.THRESH_BINARY, self._mask)
        else:
            print '[INIT] No mask found.'
        """

        # camera calibration previously computed by ../tools/calibration/calibration.py
        if self._calib_path is None or os.path.exists(self._calib_path) is False:
            # if None, set the calibration grid to be 1x1 with value 1
            print '[INIT] Calibration map not found.'
            self._calib_grid_w, self._calib_grid_h = 1, 1
            self._calib_w, self._calib_h = np.array([[1]]), np.array([[1]])
        else:
            # the larger the calib value is, the closer the grid to the camera
            print '[INIT] Calibration map found: {}'.format(self._calib_path)
            self._calib_w, self._calib_h = pickle.load(open(self._calib_path))
            self._calib_grid_h, self._calib_grid_w = self._calib_w.shape
            # normalize to [0, 1]
            self._calib_w = np.amin(self._calib_w) * (1. / self._calib_w)
            self._calib_h = np.amin(self._calib_h) * (1. / self._calib_h)
        # GT path
        if P['debug']:
            assert os.path.isfile(gt_path), "Ground truth file {} doesn't exist".format(gt_path)
            self._ground_truth_array = np.loadtxt(gt_path, delimiter=",", dtype=np.double)
        # load detection results into box_lists
        print '[INIT] Loading detection result from: {}'.format(self._data_file)
        self._load_data()


        # const values
        self._log_pinit = (P['log_regularizer'] + P['log_alpha'] * 8)
        self._log_pterm = self._log_pinit

        self.debug_track_list = []

        # visualize low tracks
        self._vis = Visualizer("track")

        # ReidNet
        self._reidnet = ExtractFeature()

        # Final track
        self._final_tracks = []
        self._merge_error = 0
        self._final_high_level_feature_list = []
        self._final_high_level_id_list = []

        self.pos_arr = None
        self.neg_arr = None

        # vertical edge list
        self.vertical_list = []
        self.edge_list = []
        # logistic
        coef_file = os.path.join(P['coef_path'], "MOT16-{}_logistic_weight.txt".format(P['mot_embedding_test2train'][self._seq_name]))
        assert os.path.isfile(coef_file), "{} doesn't exist".format(coef_file)
        self.coef = np.loadtxt(coef_file)

    def _load_segment(self):
        # Description: load frames of the segment to be processed
        # Detail:
        # step1: confirm the start fid and end fid in this segment
        # step2: load img to self._frame_lists[]

        self._segment_start_fid = \
            self._segment_index * (P['segment_length'] - P['overlap_length'])

        self._segment_end_fid = \
            min(self._segment_start_fid + P['segment_length'], self._total_frame_cnt)

        for i, f in enumerate(range(self._segment_start_fid, self._segment_end_fid)):
            if f < self._segment_start_fid + P['overlap_length'] and self._frame_lists.has_key(P['segment_length'] - P['overlap_length'] + i):
                ret, frame = True, self._frame_lists[P['segment_length'] - P['overlap_length'] + i]
            else:
                ret, frame = self._cap.read()
                # print("frame id is {}".format(f))
            assert ret, "load error: frame at {}".format(f)
            self._frame_lists[f - self._segment_start_fid] = frame

    def _load_data(self):
        # load detection results from detection result file
        ext = os.path.splitext(self._data_file)[1]
        append = list.append
        self._box_lists = {}
        if ext == '.pkl':
            with open(self._data_file) as f:
                raw = pickle.load(f)
                assert isinstance(raw, dict), "load error: dict expected"
                k = next(iter(raw))
                val = raw[k]
                if isinstance(val, np.ndarray):
                    for fid, det_list in raw.items():
                        self._box_lists[fid - 1] = []
                        l = self._box_lists[fid - 1]
                        for b in det_list:
                            w, h = b[2] - b[0], b[3] - b[1]
                            if b[4] > P['th_detection_confidence'] \
                                    and w > P['th_detection_min_width'] \
                                    and h > P['th_detection_min_height']:
                                append(l, Box(pos=(b[0], b[1], b[2], b[3]), frame_id=fid, confidence=b[4]))

                elif isinstance(val, list):
                    for fid, det_list in raw.items():
                        # print(det_list)
                        self._box_lists[fid - 1] = []
                        l = self._box_lists[fid - 1]
                        for b in det_list:
                            w, h = b[2] - b[0], b[3] - b[1]
                            if b[4] > P['th_detection_confidence'] \
                                    and w > P['th_detection_min_width'] \
                                    and h > P['th_detection_min_height']:
                                pos = b[2]
                                append(l, Box(pos=(pos[0], pos[1], pos[2], pos[3]), frame_id=fid, confidence=b[0]))
                else:
                    raise AssertionError

        elif ext == '.data' or ext == '.dat':
            box_count = 0
            gt_box_count = 0

            with open(self._data_file) as f:
                while True:
                    raw = f.read(8)
                    if not raw:
                        break
                    fid, num = struct.unpack('ii', raw)
                    # fid -= 1
                    self._box_lists[fid] = []
                    l = self._box_lists[fid]
                    # get gt trace in this fid
                    gt_idx = self._ground_truth_array[:, 0] == fid
                    gt_list_temp = self._ground_truth_array[gt_idx]
                    for i in range(num):
                        det = f.read(20)
                        x1, y1, x2, y2, c = struct.unpack('4if', det)
                        # print(x1, y1, x2, y2)
                        w, h = x2 - x1, y2 - y1
                        if c < P['th_detection_confidence'] or (w < P['th_detection_min_width'] and h < P['th_detection_min_height']):
                            continue
                        else:
                            if x1 >= P['frame_width'] or x1 <= 0 or x2 >= P['frame_width'] or x2 <= 0 or y1 >= P['frame_height'] or y1 <= 0 \
                                    or y2 >= P['frame_height'] or y2 <= 0:
                                continue
                            box = Box(pos=(x1, y1, x2, y2), frame_id=fid, confidence=c)
                            # box.find_gt_id(gt_list_temp)
                            box_count += 1
                            gt_box_count += len(box.gt_id)
                            append(l, box)
            print("{} and {}, scale is {}".format(box_count, gt_box_count, gt_box_count / float(box_count)))
        elif ext == ".txt":
            # mot format
            raw_data = np.loadtxt(self._data_file, delimiter=",")
            for row_data in raw_data:
                fid = int(row_data[0])
                if not self._box_lists.has_key(fid):
                    self._box_lists[fid] = []
                l = self._box_lists[fid]
                if P['debug']:
                    gt_idx = self._ground_truth_array[:, 0] == fid
                    gt_list_temp = self._ground_truth_array[gt_idx]
                    for i in range(gt_list_temp.shape[0]):
                        x1 = gt_list_temp[i][2]
                        y1 = gt_list_temp[i][3]
                        x2 = gt_list_temp[i][2] + gt_list_temp[i][4]
                        y2 = gt_list_temp[i][3] + gt_list_temp[i][5]
                        if x1 < 0:
                            x1 = 0
                        if y1 < 0:
                            y1 = 0
                        if x2 < 0:
                            x2 = 0
                        if y2 < 0:
                            y2 = 0
                        if x2 >= P['frame_width']:
                            x2 = P['frame_width'] - 1
                        if y2 >= P['frame_height']:
                            y2 = P['frame_height'] - 1
                        gt_list_temp[i][2], gt_list_temp[i][3], gt_list_temp[i][4], gt_list_temp[i][5] = \
                            x1, y1, x2 - x1, y2 - y1
                x1, y1 = int(row_data[2]), int(row_data[3])
                x2, y2 = int(row_data[2] + row_data[4]), int(row_data[3] + row_data[5])
                w, h = int(row_data[4]), int(row_data[5])
                c = float(row_data[6])
                if c > 1.0:
                    c = 1.0
                if c < P['th_detection_confidence'] or (
                        w < P['th_detection_min_width'] and h < P['th_detection_min_height']):
                    continue
                else:
                    if x1 >= P['frame_width'] or x1 <= 0 or x2 >= P['frame_width'] or x2 <= 0 or y1 >= P[
                        'frame_height'] or y1 <= 0 \
                            or y2 >= P['frame_height'] or y2 <= 0:
                        continue

                    box = Box(pos=(x1, y1, x2, y2), frame_id=fid, confidence=c)
                    if P['debug']:
                        box.find_gt_id(gt_list_temp)
                    append(l, box)

    def _preprocess_detections(self):
        # Description: compute box feature, occlusion and other things before association
        # Detail:
        # step1: calculate the overlap ratio, and delete the bbox that has high occulation retio
        # step2: calculate the color
        for fid in range(self._segment_start_fid, self._segment_end_fid):
            if self._box_lists.get(fid) is None:
                continue
            box_list = self._box_lists[fid]
            if not box_list:
                continue

            # box.track_id may be changed during tracklet association of last segment. reset to 0
            if fid < self._segment_start_fid + P['overlap_length']:
                for box in box_list:
                    box.track_id = 0

            frame = self._frame_lists[fid - self._segment_start_fid]

            # compute occlusion ratio for each box and abandon those with high occlusion ratio
            nb_box = len(box_list)
            box_to_be_discarded = {}
            for i in range(nb_box):
                box_to_be_discarded[i] = False
            append = list.append

            for i in range(0, nb_box):
                box_i = box_list[i]
                xi1, yi1, xi2, yi2 = box_i.pos
                occ_map = np.zeros(shape=(yi2 - yi1 + 1, xi2 - xi1 + 1))
                size = box_i.size
                xi1_extend, yi1_extend, xi2_extend, yi2_extend = box_i.extend_pos

                candidate = []
                for j in range(0, nb_box):
                    if j == i:
                        continue
                    box_j = box_list[j]
                    xj1, yj1, xj2, yj2 = box_j.pos
                    if xj1 > xi2 or xj2 < xi1 or yj1 > yi2 or yj2 < yi1:
                        continue
                    append(candidate, j)
                    occ_map[max(yi1, yj1)-yi1 : min(yi2, yj2)+1-yi1][max(xi1, xj1)-xi1 : min(xi2, xj2)+1-xi1] = 1
                occ_size = (occ_map != 0).sum()
                box_i.occlusion = float(occ_size)/size

            # filt some box, box_list redefine
            self._box_lists[fid] = [box for ind, box in enumerate(box_list) if box_to_be_discarded[ind] is False]
            # extend beyond frame height
            # print("fid and box num ", fid, len(self._box_lists[fid]))
            # raw_input()
            # calculate color hist
            for box in self._box_lists[fid]:
                box.get_color_hist(frame)

    def _view_low_level_tracklet(self):
        # view tracklet like _low_level_tracks
        for fid in range(self._segment_start_fid, self._segment_end_fid - 1):
            frame = self._frame_lists[fid - self._segment_start_fid]
            frame_clone = draw_tracks(frame, self._low_level_tracks, fid)
            cv2.imwrite(os.path.join(self._debug_img_dir, "low_level_tracklet_" + str(fid).zfill(5) + '.jpg'), frame_clone)
            # cv2.imshow("", frame_clone)
            cv2.waitKey(1)

    def _view_high_level_tracklet(self):
        # view tracklet like _high_level_tracks
        for fid in range(self._segment_start_fid, self._segment_end_fid - 1):
            frame = self._frame_lists[fid - self._segment_start_fid]
            frame_clone = draw_tracks(frame, self._high_level_tracks, fid)
            cv2.imwrite(os.path.join(self._debug_img_dir, "high_level_tracklet_" + str(fid).zfill(5) + '.jpg'), frame_clone)
            cv2.waitKey(1)

    def _view_low_level_tracklet_by_visdom(self):
        # view low level track on visdom
        # print("当前低级轨迹的个数为{}".format(len(self._low_level_tracks)))
        low_track_idx_list = [(track.start_fid, track.end_fid) for track in self._low_level_tracks]
        track_frm_dict = {i : [] for i in range(len(self._low_level_tracks))}
        for fid in range(self._segment_start_fid, self._segment_end_fid - 1):
            frame = self._frame_lists[fid - self._segment_start_fid]
            track_frm_list = get_track_frm(frame, self._low_level_tracks, fid)
            [track_frm_dict[i].append(frm) for i, frm in enumerate(track_frm_list) if frm is not None]
        cv2.waitKey(1)
        self._vis.img_seq(track_frm_dict, track=tuple(low_track_idx_list))

    def _view_high_level_tracklet_by_visdom(self):
        """
        view high level track on visdom
        """
        # print("当前高级轨迹的个数为{}".format(len(self._high_level_tracks)))
        high_track_idx_list = [(track.start_fid, track.end_fid, track.track_id) for track in self._high_level_tracks]
        track_frm_dict = {i: [] for i in range(len(self._high_level_tracks))}
        for fid in range(self._segment_start_fid, self._segment_end_fid - 1):
            frame = self._frame_lists[fid - self._segment_start_fid]
            track_frm_list = get_track_frm(frame, self._high_level_tracks, fid)
            [track_frm_dict[i].append(frm) for i, frm in enumerate(track_frm_list) if frm is not None]
        cv2.waitKey(1)
        self._vis.img_seq(track_frm_dict, track=tuple(high_track_idx_list))

    def _tracklet_init(self, fid):
        if self._box_lists.get(fid) is None:
            return
        curr_box_list = self._box_lists[fid]
        for i in range(len(curr_box_list)):
            new_box_list = [curr_box_list[i]]
            tid = len(self._low_level_tracks) + 1
            new_track = Track(new_box_list, track_id=tid)  # track id is initialized with num of _low_level_tracks
            self._low_level_tracks.append(new_track)

    def _neighboring_association(self, fid):
        # Description: associating detection boxes between neighboring frames (fid and fid + 1), i.e. low level association
        # Detail:
        # step1: get box list from two neighbor frames(fid and fid) ps: Been processed and color hist is got
        # step2: calculate the affinities between two boxlist and the affinity matrix is formed by size, distance and hist affinity
        #       you should carefully tune the variance meaning confidence, more variance, less confidence
        # step3: double thresh match method, the first matching should better much than second matching, match size should be noticed,
        #       you see, there are a lots of codes to deal with this situation.
        # step4：after match, we can see there is a lot of matching pairs, we link match pairs to track. new track will be generated, and old track will be
        #       updated(velocity)

        if self._box_lists.get(fid) is None or self._box_lists.get(fid+1) is None:
            return
        curr_box_list, next_box_list = self._box_lists[fid], self._box_lists[fid + 1]
        nb_curr, nb_next = len(curr_box_list), len(next_box_list)
        affinities = np.zeros(shape=(nb_curr, nb_next), dtype=np.float32)

        def compute_affinities(ind_ij):
            # compute the affinity of box[i] at current frame and box[j] at next frame
            i, j = ind_ij[0], ind_ij[1]
            affinities[i][j] = Box.compute_box_affinity(curr_box_list[i], next_box_list[j], self._calib_w, self._calib_h)

        # find indices of first and second max affinity values along each row
        def find_matching(matrix):
            # inds will be the column indices of the top two maximum values in the matrix
            inds = np.argpartition(-matrix, (0, 1), axis=-1)[:, 0:2]
            # aff will be the top two maximum values in the matrix
            aff = matrix[np.arange(matrix.shape[0])[:, None], inds]
            # candidate will be the rows that meet the double-threshold condition
            candidate = np.where((aff[:, 0] > P['th_low_level_db1']) &
                                 ((aff[:, 0] - aff[:, 1]) > P['th_low_level_db2']))[0]
            # match will be a K x 2 array in which each row represents a valid pair(association)
            match = np.transpose(np.vstack([candidate, inds[candidate, 0]]))
            return match
        # compute affinities of every possible pair of boxes
        map(compute_affinities, [(i, j) for i, _ in enumerate(curr_box_list) for j, _ in enumerate(next_box_list)])
        """
        curr_roi_list = map(lambda box:self._frame_lists[fid][box.pos[1]:box.pos[3], box.pos[0]:box.pos[2]], curr_box_list)
        next_roi_list = map(lambda box: self._frame_lists[fid][box.pos[1]:box.pos[3], box.pos[0]:box.pos[2]],
                            next_box_list)
        self._vis.img_seq({"curr":curr_roi_list, "next":next_roi_list})
        """
        # print("aff", affinities)

        if nb_next >= 2 and nb_curr >= 2:
            row_matching = find_matching(affinities)  # search along rows
            col_matching = find_matching(affinities.T)  # search along columns, by transposing the matrix
            col_matching[:, [0, 1]] = col_matching[:, [1, 0]]  # switch back the indexes
            row_matching = row_matching.tolist()
            col_matching = col_matching.tolist()
            # row_matching and col_matching now contain (r, c) pairs where 'double-threshold' condition are met
            # along rows and columns, respectively. A valid solution pair (r*, c*) must be included in both
            # row_matching and col_matching.
            # E.g.
            #     row_matching = [ (1, 3), (2, 4), ... ]
            #     col_matching = [ (1, 5), (2, 4), ... ]
            # (2, 4) is a solution while neither (1, 3) nor (1, 5) are valid solutions
            matching = [m for m in row_matching if m in col_matching]

        elif nb_next == 1 and nb_curr >= 2:
            # only one box in next frame, search along the first column
            matching = find_matching(affinities.T)
            matching[:, [0, 1]] = matching[:, [1, 0]]
        elif nb_curr == 1 and nb_next >= 2:
            # only one box in current frame, search along the first row
            matching = find_matching(affinities)
        elif nb_curr == 1 and nb_next == 1:
            # only one box in current frame and next frame, check if the affinity is greater than threshold1
            if affinities[0][0] < P['th_low_level_db1']:
                matching = []
            else:
                matching = [(0, 0)]
        else:
            # no box in current frame or next frame, matching is empty
            matching = []

        # associate boxes
        for pair in matching:
            # pair is a pair of indexes: <ind_curr, ind_next>, indicating the ind_curr-th box
            # is matched with the ind_next-th box
            if curr_box_list[pair[0]].track_id == 0:
                # the box hasn't been linked to any other boxes(tracks), generate a new track
                new_box_list = [curr_box_list[pair[0]], next_box_list[pair[1]]]
                tid = len(self._low_level_tracks) + 1
                new_track = Track(new_box_list, track_id=tid)  # track id is initialized with num of _low_level_tracks
                self._low_level_tracks.append(new_track)
                # print("track num increase and is {}".format(len(self._low_level_tracks)))
            else:
                # the box has been previously linked to a track
                tid = curr_box_list[pair[0]].track_id
                track = self._low_level_tracks[tid - 1]  # fetch the track
                track += next_box_list[pair[1]]  # append the newly matched box
                track.smooth_velocity()  # smooth velocity( the velocity is belong to box of this track)

    def _get_roi_list(self, t, specific_box_list=None):
        """
        :param t: track
        :param specific_box_list: box list given manually
        :return: 
        """
        append = list.append
        im_mean, im_stdvar = np.array(P['im_mean']), np.array(P['im_stdvar'])
        box_list = None
        rtn_roi_list = []
        if specific_box_list is None:
            box_list = t._box_list
        else:
            box_list = specific_box_list
        for box in box_list:
            frame = self._frame_lists[box.frame_id - self._segment_start_fid]
            x1, y1, x2, y2 = box.pos
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2 - x1, y2 - y1
            # expand to whole body
            if P['head_shoulder'] is True:
                width = int(float(x2 - x1) * P['head_shoulder_to_body_scale_x']) // 2
                cx, _ = box.center
                x1, x2 = max(0, cx - width), min(P['frame_width'], cx + width)
                height = int((y2 - y1) * P['head_shoulder_to_body_scale_y'])
                y2 = min(y1 + height, P['frame_height'])

            roi = np.copy(frame[y1:y2, x1:x2])[0:height, :]
            dst = cv2.resize(roi, (P['scl_input_width'], P['scl_input_height']))  # in cv2.resize, dSize = [dx, dy]
            dst = (dst - im_mean) / (im_stdvar + 1e-7)
            # roi = dst
            if specific_box_list is None:
                append(t.roi_list, roi)
            else:
                append(rtn_roi_list, roi)
        if specific_box_list is None:
            if len(t.roi_list) >= 10:
                t.signature.extend(random.sample(t.roi_list, 10))
            else:
                t.signature.extend(t.roi_list)

        else:
            if len(t.roi_list) >= 10:
                t.signature.extend(random.sample(t.roi_list, 10))
            else:
                t.signature.extend(t.roi_list)
        if specific_box_list is not None:
            return rtn_roi_list

    def _preprocess_tracks(self):
        # all the operations on tracks before high level association are done here
        # remove short tracks
        self._low_level_tracks = [track for track in self._low_level_tracks if len(track) >= P['th_track_len']]
        # just for debug
        if P['debug']:
            self._low_level_tracks = [track for track in self._low_level_tracks if track.gt_track_id != -1]
        for t in self._low_level_tracks:
            # smooth track's velocity using gaussian process regression
            t.gaussian_smooth()
            # rois means mat by roi
            # prepare rois for appearance comparison, maybe there is a more elegant way to iterate over protected member
            self._get_roi_list(t)

    def _preprocess_high_level_tracks(self, src_tracks):
        # all the operations on tracks before high level association are done here
        # remove short tracks
        tracks = [track for track in src_tracks if len(track) >= P['th_track_len']]
        for t in tracks:
            # smooth track's velocity using gaussian process regression
            t.gaussian_smooth()
            # rois means mat by roi
            # prepare rois for appearance comparison, maybe there is a more elegant way to iterate over protected member
            self._get_roi_list(t)

    def _compute_deep_feature(self, track_list):
        track_feature_list = []
        def construct_seq(ind_i):
            """
            Construct roi seq for input of network

            Parameters:
                ind_i: int
                The index of track
            Return:
                seq_roi: array
                The sequence of roi in image that has been shaped
            """
            track_i = track_list[ind_i]
            select_indices_i = track_i.sample_rois()
            seq_roi_list = [track_i.roi_list[i] for i in select_indices_i]
            return seq_roi_list

        def get_feature_list(roi_list):
            feature_list = self._reidnet.get_feature_list(roi_list)
            return feature_list

        def get_track_deep_feature(ind_i):
            roi_list = construct_seq(ind_i)
            feature = get_feature_list(roi_list)
            mean_feature = np.mean(feature, axis=0)
            norm_feature = mean_feature / np.linalg.norm(mean_feature)
            return norm_feature

        for idx, track in enumerate(track_list):
            deep_feature = get_track_deep_feature(idx)
            track_feature_list.append(deep_feature)
            track_list[idx].feature = deep_feature
        return track_feature_list

    def _compute_track_affinity(self, track_list):
        debug = P['debug']
        nb_track = len(track_list)
        from itertools import combinations
        combins = [c for c in combinations(range(nb_track), 2)]
        same_id_list = []
        diff_id_list = []
        same_img_pair = []
        self.edge_list = []
        self.vertical_list = []
        if debug:
            for idx1, idx2 in combins:
                size_affinity = Track.compute_track_size_affinity(track_list[idx1], track_list[idx2])
                motion_affinity = Track.compute_track_pos_affinity(track_list[idx1], track_list[idx2])
                appearance_affinity = Track.compute_track_appearance_affinity(track_list[idx1], track_list[idx2])
                gt_id1, gt_id2 = track_list[idx1].gt_track_id, track_list[idx2].gt_track_id
                if gt_id1 == gt_id2:
                    same_id_list.append((size_affinity, motion_affinity, appearance_affinity))
                    same_img_pair.append([track_list[idx1].roi_list[0], track_list[idx2].roi_list[0]])
                else:
                    diff_id_list.append((size_affinity, motion_affinity, appearance_affinity))
            # self._vis.img_seq(dict(zip(range(len(same_img_pair)), same_img_pair)))
            same_id_list = np.array(same_id_list)
            diff_id_list = np.array(diff_id_list)
            if self.pos_arr is not None:
                self.pos_arr = np.vstack((self.pos_arr, same_id_list))
                self.neg_arr = np.vstack((self.neg_arr, diff_id_list))
            else:
                self.pos_arr = same_id_list
                self.neg_arr = diff_id_list
            print(np.mean(same_id_list, axis=0), np.mean(diff_id_list, axis=0))
        else:
            for i, t in enumerate(self._low_level_tracks):
                t.track_id = i
                self.vertical_list.append(t.track_id)

            for idx1, idx2 in combins:
                # overlap
                if Track.overlap(track_list[idx1], track_list[idx2]) >= P['th_time_overlap']:
                    self.edge_list.append((track_list[idx1].track_id, track_list[idx2].track_id, -1e3))
                    continue
                size_affinity = Track.compute_track_size_affinity(track_list[idx1], track_list[idx2])
                motion_affinity = Track.compute_track_pos_affinity(track_list[idx1], track_list[idx2])
                appearance_affinity = Track.compute_track_appearance_affinity(track_list[idx1], track_list[idx2])
                edge_weight = np.matmul(self.coef, (size_affinity, motion_affinity, appearance_affinity, size_affinity
                                                    * motion_affinity,size_affinity * appearance_affinity,
                                                    motion_affinity * appearance_affinity, size_affinity *
                                                    motion_affinity * appearance_affinity))
                self.edge_list.append((track_list[idx1].track_id, track_list[idx2].track_id, edge_weight))
                """         
                if track_list[idx1].gt_track_id == track_list[idx2].gt_track_id:
                    print edge_weight
                else:
                    print "---->", edge_weight
                """


    def _process_multicut(self):
        def parse_result(res_file):
            with open(res_file, "rb") as f:
                raw = f.read(4)
                if not raw:
                    return
                track_len = struct.unpack("1i", raw)[0]
                for i in range(track_len):
                    raw = f.read(8)
                    track_label, tracklet_len = struct.unpack("2i", raw)
                    raw = f.read(tracklet_len * 4)
                    track_id_list = struct.unpack("{}i".format(tracklet_len), raw)
                    tmp_track_list = []
                    [tmp_track_list.append(self._low_level_tracks[idx]) for idx in track_id_list]
                    self._high_level_tracks.append(Track.merge_tracks(tmp_track_list))

        # save_file
        vertice_info_list = self.vertical_list
        edge_info_list = self.edge_list
        multicut_ver_file = os.path.join(P['mot_tmp_path'], "vertical.dat")
        multicut_weight_edge_file = os.path.join(P['mot_tmp_path'], "edge.dat")
        res_file = os.path.join(P['mot_tmp_path'], "res.dat")
        main = P['multicut_exe']
        with open(multicut_ver_file, "wb") as f:
            data = struct.pack("i", len(vertice_info_list))
            f.write(data)
            [f.write(struct.pack("i", vertex)) for vertex in vertice_info_list]
        with open(multicut_weight_edge_file, "wb") as f:
            data = struct.pack("i", len(edge_info_list))
            f.write(data)
            [f.write(struct.pack("2if", int(weight_edge[0]), int(weight_edge[1]),
                                 float(weight_edge[2]))) for weight_edge in edge_info_list]
        rc, out = commands.getstatusoutput("{} {} {} {}".format(main, multicut_ver_file, multicut_weight_edge_file, res_file))
        # print 'rc = %d, \nout = %s' % (rc, out)
        parse_result(res_file)
        os.remove(multicut_ver_file)
        os.remove(multicut_weight_edge_file)
        os.remove(res_file)


    def _fuse_high_level_tracks(self):
        """
        fuse higl level tracks with high level features
        First: fuse tracks in overlap windows
        Second: fuse tracks between same tracks
        after that any segment will be assigned an segment id(the tracks is present)
        """
        def _set_fused(track, seg_start):
            if track.start_fid - seg_start < P['overlap_length']:
                # There are a number of boxes lying in the overlap time window, may be fused with other tracks
                track.candidate_fused = False
            else:
                # No box lies in the overlap time window, won't be fused
                track.candidate_fused = True

        def _fuse(track1, track2):
            """
                First count the number of overlapping bounding boxes in track1 and track2.
                If the number is greater than a given threshold, the two tracks are considered
                to belong to the same person and we append track2 to the end of track1
                track1: prev_track
                track2: curr_track
            """
            if track1.fused or track2.fused:
                return False
            matched_box_num = 0
            if track1.end_fid - P['th_track_fuse_len'] + 1 < track2.start_fid or len(track1) < P['th_track_fuse_len']\
                    or track1.start_fid > track2.start_fid:
                # there is no chance that these two tracks are to be fused
                return False

            start_ind1 = max(0, len(track1) - (track1.end_fid - track2.start_fid + 1))
            start_ind2 = 0

            while start_ind1 < len(track1) and start_ind2 < len(track2):
                fid1, fid2 = track1[start_ind1].frame_id, track2[start_ind2].frame_id
                if fid1 < fid2:
                    start_ind1 += 1
                elif fid1 > fid2:
                    start_ind2 += 1
                else:
                    cx1, cy1 = track1[start_ind1].center
                    cx2, cy2 = track2[start_ind2].center
                    if abs(cx1 - cx2) < P['th_track_fuse_diff'] \
                            and abs(cy1 - cy2) < P['th_track_fuse_diff']:
                        matched_box_num += 1
                    start_ind1 += 1
                    start_ind2 += 1
                    if matched_box_num >= P['th_track_fuse_len']:
                        break

            if matched_box_num < P['th_track_fuse_len']:
                return False
            else:
                track1.fused = True
                track2.fused = True
                print "track1 {}-->{}   track2 {}-->{}".format(track1.start_fid, track1.end_fid, track2.start_fid,
                                                               track2.end_fid)
                track1.append(track2)
                track1.segment_id = self._segment_index # segment id is updated
            return True

        # firset assigned
        for track in self._high_level_tracks:
            _set_fused(track, self._segment_start_fid)
        nb_high_level_tracks = len(self._high_level_tracks)
        if nb_high_level_tracks == 0:
            return

        for i in range(len(self._final_tracks)):
            if self._final_tracks[i].segment_id == self._segment_index - 1:
                if self._final_tracks[i].end_fid - (self._segment_start_fid - P['overlap_length']) + 1 \
                        >= P['th_track_fuse_len']:
                    for ii in range(nb_high_level_tracks):
                        if not self._high_level_tracks[ii].candidate_fused:
                            _fuse(self._final_tracks[i], self._high_level_tracks[ii])

        # for all high level track left, append to final track
        for i in range(nb_high_level_tracks):
            if not self._high_level_tracks[i].fused:
                self._high_level_tracks[i].segment_id = self._segment_index
                self._high_level_tracks[i].track_id = len(self._final_tracks) + 1
                self._final_tracks.append(self._high_level_tracks[i])

        # for all final tracks, fused is False
        for i in range(len(self._final_tracks)):
            self._final_tracks[i].fused = False



    def _get_event(self):
        signature_list = self._event_judge.judge_wandering_event(self._final_tracks, self._segment_index)
        # for i, signature in enumerate(signature_list):
        #    self._vis.img_many(signature, str(i))
        stay_list = self._event_judge.judge_stay_event(self._high_level_tracks, self._frame_lists,
                                           self._segment_start_fid, self._calib_w, self._calib_h)
        for i, stay_frame in enumerate(stay_list):
            self._vis.img_many(stay_frame, str(i))



    def _run_segment(self):
        # run tracker on a single segment
        self._load_segment()  # load frames
        self._preprocess_detections()  # preprocess detection results
        self._low_level_tracks[:] = []  # clear low level tracks
        self._high_level_tracks[:] = []
        gc.collect() # Garbage Collector interface

        # low level association
        print "\t[LOW LE VEL ASSOCIATION]"
        # self._tracklet_init(self._segment_start_fid + 1)
        for i in range(self._segment_start_fid, self._segment_end_fid - 1):
            # Note that i is in range [start, end - 2] since neighboring association are done over fid (i, i+1)
            self._neighboring_association(i)

        # self._view_low_level_tracklet_by_visdom()
        # raw_input()
        # preprocess tracks
        print "\t[TRACKLET PREPROCESSING]"
        print("\t[BEFOR PROCESS LOW LEVEL TRACK NUM is {}]".format(len(self._low_level_tracks)))
        self._preprocess_tracks()
        print("\t[LOW LEVEL TRACK NUM is {}]".format(len(self._low_level_tracks)))
        print("\t[EXTRACT LOW LEVEL TRACK DEEP FEATURE]")
        self._compute_deep_feature(self._low_level_tracks)
        print "\t[HIGH LEVEL ASSOCIATION]"
        self._compute_track_affinity(self._low_level_tracks)
        print "\t[GEN MULTICUT FILE]"
        if not P['debug']:
            self._process_multicut()
        # self._tracklet_association()
        print("\t[HIGH LEVEL TRACK NUM is {}]".format(len(self._high_level_tracks)))
        self._fuse_high_level_tracks()
        # raw_input()
        i = None

    def run(self):
        """
            This is the main function of Tracker, which is called by exterior scripts.
            run() will call _run_segment iteratively to get tracking results of each
            segment. Finally, it will fuse tracks stored in different segments into complete
            tracks and dump the final tracks into file
        """
        self.track_len = []
        # debug
        while self._segment_index <= self._segment_cnt:
            if self._segment_index < 0:     # Uncomment this block to debug specific segment
                self._segment_index += 1
                continue
            # run association
            print "[Tracking]\tSegment index:\t{}  Total segment num:\t{}".format(self._segment_index, self._segment_cnt)
            start = cv2.getTickCount()
 
            self._run_segment()
            print "[Tracking]\tSegment start:\t{} Segment end\t{}".format(self._segment_start_fid,
                                                                          self._segment_end_fid)
            # dump into file
            """
            seg_name = 'segment_{}.track'.format(self._segment_index)
            seg_file = os.path.join(self._segment_dir, seg_name)
            self._segments_path.append(seg_file)
            Track.dump_to_track_file(self._high_level_tracks, save_name=seg_file)
            print "Track contains {} high level tracks".format(len(self._high_level_tracks))
            """
            self._segment_index += 1
            end = cv2.getTickCount()
            print "[Tracking]\tTime:\t{} seconds".format(float(end - start) / cv2.getTickFrequency())
        if P['debug']:
            pos_feature_num = self.pos_arr.shape[0]
            neg_feature_num = self.neg_arr.shape[0]
            pos_arr = np.hstack((self.pos_arr, np.ones(shape=(pos_feature_num, 1))))
            neg_arr = np.hstack((self.neg_arr, np.zeros(shape=(neg_feature_num, 1))))
            np.savetxt(os.path.join("../feature_classifier/", "{}_pos_feature.txt".format(self._video_name)), pos_arr)
            np.savetxt(os.path.join("../feature_classifier/", "{}_neg_feature.txt".format(self._video_name)), neg_arr)

        final_track_save_file = os.path.join(self._save_dir, self._video_name + "_final_merged.track")
        mot_track_save_file = os.path.join(self._save_dir, self._video_name + ".txt")
        Track.dump_to_track_file_no_feature(self._final_tracks, final_track_save_file, self._calib_w, self._calib_h)
        Track.dump_track_with_mot_format(self._final_tracks, mot_track_save_file,)
        print("there are {} tracklet in final merged track".format(len(self._final_tracks)))