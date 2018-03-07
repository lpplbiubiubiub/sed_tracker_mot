# Track class
import copy
import cPickle as pickle
import numpy as np
import math
import os
import struct

from collections import Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from box import Box
from shared_parameter import P
from tools import velocity_smoothing_kernel, bresenham_line_integral

from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine
import random

class Track(object):
    """
        A track is a list of boxes with an unified id denoting different object.
        The boxes in a track are stored in list, track._box_list

        Attributes:
        [start_fid]     The start frame id of a track
        [end_fid]       The end frame id of a track
        [track_id]      Identity of a track
        [box_num]       Number of boxes a track includes

        *   The frame id of the boxes may be not continuous but any two boxes occupying
            the same frame id is not allowed.
        *   Tracks can be linked using operator '+': T3 = T1 + T2.
        *   Get the i-th box in a track, one could use T[i], which will return a *deep copy*
            of the i-th box, preventing from modifying the i-th box.
        *   Slicing a track will return a copy of box list, e.g. track[0:4] ==> list of boxes
        *   One can split a track into two using track.split(split_index, track_id1, track_id2),
            which will return two new tracks whose id(s) are track_id1 and track_id2.
    """

    def __init__(self, box_list, track_id=0):
        assert isinstance(box_list, list), "box_list must be a list"
        if len(box_list) > 0:
            assert all(isinstance(box, Box) for box in box_list), "entries in box_list must Box"
            sorted(box_list, key=lambda box: box.frame_id)
            fid_list = [box.frame_id for box in box_list]
            fid_counter = Counter(fid_list)
            assert all(fid_counter[v] == 1 for v in fid_counter), "found boxes with same fid"
        self._box_list = box_list
        # box_list
        self._start_fid = 0 if len(box_list) == 0 else box_list[0].frame_id
        self._end_fid = 0 if len(box_list) == 0 else box_list[-1].frame_id
        self._box_num = len(box_list)
        self._track_id = 0
        self._track_id = None
        self.track_id = track_id if track_id else 0

        # maintain the gpr models
        self.__rbf_kernel = RBF(P['rbf_kernel_length_scale'])
        """
            The alpha value in GaussianProcessRegressor is an important hyper-param.
            Its default value is set to be 1e-10 which is too small and will cause
            over-fitting when smoothing tracks. Good value of alpha will be value near 1e-3
        """
        self._gpx = GaussianProcessRegressor(kernel=self.__rbf_kernel, alpha=P['gpr_alpha'],
                                             n_restarts_optimizer=P['gpr_restart_optimizer'])
        self._gpy = GaussianProcessRegressor(kernel=self.__rbf_kernel, alpha=P['gpr_alpha'],
                                             n_restarts_optimizer=P['gpr_restart_optimizer'])

        # list of ROIs, for computing appearance feature in high level association
        self._roi_list = []

        # if have been fused. if set to be True, won't be fused in track_fuse().
        self._fused = False
        self._candidate_fused = False

        # high level feature
        self._high_level_feature = None
        # track gt id
        self._gt_track_id = -1
        self._segment_id = -1

        self._feature_list = []
        self._signature_list = []
        self._max_len_signature = 10

    def size(self, is_head=True):
        w, h = 0., 0.
        box_list = []
        if is_head:
            if self.box_num >= 10:
                box_list = self[:10]
            else:
                box_list = self[:]
            box_num = len(box_list)
            w = sum([bbox.width for bbox in box_list]) / (box_num + 0.)
            h = sum([bbox.height for bbox in box_list]) / (box_num + 0.)
        else:
            if self.box_num >= 10:
                box_list = self[-10:]
            else:
                box_list = self[:]
            box_num = len(box_list)
            w = sum([bbox.width for bbox in box_list]) / (box_num + 0.)
            h = sum([bbox.height for bbox in box_list]) / (box_num + 0.)
        return w, h

    @property
    def start_fid(self):
        return self._start_fid

    @property
    def end_fid(self):
        return self._end_fid

    @property
    def box_num(self):
        return self._box_num

    @property
    def track_id(self):
        return self._track_id

    @track_id.setter
    def track_id(self, tid):
        assert tid >= 0, "id must be greater than 0"
        self._track_id = tid
        for box in self._box_list:
            box.track_id = tid

    @property
    def gpx(self):
        return self._gpx

    @property
    def gpy(self):
        return self._gpy

    @property
    def roi_list(self):
        return self._roi_list

    @property
    def fused(self):
        return self._fused
    @property
    def candidate_fused(self):
        return  self._candidate_fused
    @property
    def feature(self):
        return self._high_level_feature
    @property
    def gt_track_id(self):
        self.get_track_id()
        if self._gt_track_id == -1:
            return -1
        return int(self._gt_track_id[0][0])
    @property
    def segment_id(self):
        return self._segment_id
    @property
    def feature_list(self):
        if len(self._feature_list) == 0:
            return None
        return self._feature_list[:]

    @property
    def signature(self):
        return self._signature_list

    @property
    def start_id(self):
        return self._start_fid

    @feature.setter
    def feature(self, feature):
        self._high_level_feature = feature
        self._feature_list = [feature]

    @fused.setter
    def fused(self, fuse):
        if fuse:
            self._fused = True
        else:
            self._fused = False

    @candidate_fused.setter
    def candidate_fused(self, fused):
        if fused:
            self._candidate_fused = True
        else:
            self._candidate_fused = False

    @segment_id.setter
    def segment_id(self, segment_id):
        """
        segment id is assigned in higher level assigned
        """
        self._segment_id = segment_id

    @feature_list.setter
    def feature_list(self, feature):
        if type(feature) is list:
            self._feature_list.extend(feature)
            return
        self._feature_list.append(feature)

    def __repr__(self):
        return '[{}]  Id:{} | Start Frame:{} | End Frame:{} | Box Num:{} | [{} ...]'.format(
            self.__class__.__name__,
            self._track_id,
            self._start_fid,
            self._end_fid,
            self._box_num,
            " " if self._box_num == 0 else self._box_list[0]
        )

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self._box_num

    def __iadd__(self, other):
        # to support += operation
        self.append(other)
        return self

    def __getitem__(self, ind):
        assert isinstance(ind, int) or isinstance(ind, slice), "ind must be int or slice"
        if isinstance(ind, int):
            ind %= self._box_num
            return copy.deepcopy(self._box_list[ind])  # prevent from modifying the box
        else:
            return copy.deepcopy(self._box_list[ind])  # prevent from modifying the list

    def __lt__(self, other):
        return self.box_num < other.box_num  # compare two track according to their lengths

    def __gt__(self, other):
        return self.box_num > other.box_num  # compare two track according to their lengths

    def get_track_id(self):
        gt_map = {}
        for box in self._box_list:
            for track_id in box._gt_track_id_list:
                if not gt_map.has_key(track_id):
                    gt_map[track_id] = 0
                gt_map[track_id] += 1
        if len(gt_map.keys()) > 0:
            maximum = max(gt_map.values())
            self._gt_track_id = filter(lambda x: x[1] == maximum, gt_map.items())
        else:
            self._gt_track_id = [(-1, 0)]

    def overlap(self, other):
        assert isinstance(other, Track), "Track is accepted"
        if self.start_fid > other.end_fid or self.end_fid < other.start_fid:
            return 0
        else:
            s = max(self.start_fid, other.start_fid)
            e = min(self.end_fid, other.end_fid)
            overlap = max(0, e - s + 1)
            return overlap

    @staticmethod
    def overlap(track1, track2):
        assert isinstance(track1, Track), "Track1 is accept"
        assert isinstance(track2, Track), "Track2 is accept"
        if track1.start_fid > track2.end_fid or track1.end_fid < track2.start_fid:
            return 0
        else:
            s = max(track1.start_fid, track2.start_fid)
            e = min(track1.end_fid, track2.end_fid)
            overlap = max(0, e - s + 1)
            return overlap

    def append(self, other, high_level_merged=False):
        # append a box or a track to current track
        if isinstance(other, Box):
            assert other.frame_id > self._end_fid, "box's fid must be greater than track's end fid"
            new_box = other
            new_box.track_id = self._track_id
            self._box_list.append(new_box)
            self._end_fid = new_box.frame_id
            self._box_num += 1
        elif isinstance(other, Track):
            """
                Track B is allowed to be appended to the end of track A as long as B has an non-overlapping
                segment after the end of track A, for example:
                    1.  Non-overlap case:
                        A:  |-------|
                        B:              |*******|
                    this will gives:
                            |-------|*******|

                    2.  Overlap case:
                        A:  |-------|
                        B:      |----***|
                    this will gives:
                            |-------|***|
                where segment represented by '*' will be appended to the end of track A.
                This feature is useful if an overlap of 1 or 2 frame is allowed in high level  association since
                we may linked up two overlap tracks
            """
            if self.gt_track_id != other.gt_track_id and self.gt_track_id != -1 and other.gt_track_id != -1:
                print("------------->")
            if self._box_list:
                if self.end_fid > other.end_fid:
                    print "unable to append {}-{} with {}-{} because time overlap ".format(str(self.start_fid),
                                                        str(self.end_fid), str(other.start_fid), str(other.end_fid))
                    return
                append_pos = 0
                for i in range(other.box_num):
                    if other[i].frame_id > self.end_fid:
                        break
                    else:
                        append_pos += 1

                if append_pos == other.box_num:
                    return
                list_to_append = other[append_pos:]
            else:
                # if self._box_list is empty
                self._start_fid = other.start_fid
                list_to_append = other[:]

            # do not forget to assign track id
            for box in list_to_append:
                box.track_id = self.track_id

            self._box_list += list_to_append
            self._box_num += len(list_to_append)
            self._end_fid = other.end_fid
            # feature merge
            if self.feature is not None and other.feature is not None:
                # self.feature = self.feature + 0.1 * other.feature
                other_scale = other.box_num / float(self.box_num)
                other_scale = 0.1 if other_scale <= 0.1 else other_scale
                if not high_level_merged:
                    self.feature = self.feature + 1 * other.feature
                else:
                    self.feature = self.feature + 1 * other.feature
                self.feature /= norm(self.feature)

            elif self.feature is None:
                self.feature = other.feature
            self._feature_list.extend(other.feature_list)
            # ground truth id
            self._signature_list.extend(other.signature)
            if len(self._signature_list) > self._max_len_signature:
                self._signature_list = random.sample(self._signature_list, 10)
        else:
            raise AssertionError('Box or Track object is accepted')


    def insert_front(self, other, high_level_merged=False):
        if isinstance(other, Box):
            if other.frame_id > self.start_fid:
                return
            self._start_fid = other.frame_id
            self._box_list += 1
            other.track_id = self.track_id
            self._box_list.insert(0, other)
        elif isinstance(other, Track):
            if self._box_list:
                # same as append, insert will dismiss the overlapping segment of two tracks
                assert self.start_fid >= other.start_fid, "unable to insert"
                insert_end = other.box_num - 1
                for i in range(other.box_num - 1, -1, -1):
                    if other[i].frame_id < self.start_fid:
                        break
                    else:
                        insert_end -= 1
                if insert_end == -1:
                    return
                list_to_insert = other[0:insert_end + 1]
            else:
                # if self._box_list is empty
                list_to_insert = other[:]
                self._end_fid = other.end_fid

            for box in list_to_insert:
                box.track_id = self.track_id

            self._box_list = list_to_insert + self._box_list
            self._box_num += len(list_to_insert)
            self._start_fid = other.start_fid

            # feature merge
            if self.feature is not None and other.feature is not None:
                # self.feature = self.feature + 0.1 * other.feature
                other_scale = other.box_num / float(self.box_num)
                other_scale = 0.1 if other_scale <= 0.1 else other_scale
                if not high_level_merged:
                    self.feature = self.feature + 1 * other.feature
                else:
                    self.feature = self.feature + 1 * other.feature
            elif self.feature is None:
                self.feature = other.feature
            self._feature_list.extend(other.feature_list)
            self._signature_list.extend(other.signature)
            if len(self._signature_list) > self._max_len_signature:
                self._signature_list = random.sample(self._signature_list, 10)
        else:
            raise AssertionError('Box or Track object is accepted')

    @staticmethod
    def merge_tracks(track_list_src):
        track_list = track_list_src[:]
        assert all(isinstance(o, Track) for o in track_list), "all track should have Track type"
        sorted(track_list, key=lambda x: x.start_id)
        track_start = track_list[0]
        if len(track_list) > 1:
            for idx in range(1, len(track_list)):
                track_start.append(track_list[idx])
        return track_start

    def pop_back(self):
        # pop back a box
        assert self.box_num > 0, "track is empty"
        box = self._box_list[-1]
        self._box_num -= 1
        self._box_list.pop()
        if self._box_num == 0:
            self._start_fid = 0
            self._end_fid = 0
        else:
            self._end_fid = self._box_list[-1].frame_id
        return box

    def split_at(self, split_ind, track_id1, track_id2):
        """
            Split the current track at index split_ind into two tracks
            return two tracks. The former one in range [0, split_ind]
            and the latter one in range [split_ind + 1, self.box_num - 1]
        """
        assert 0 <= split_ind < self.box_num, "split index must in range [0, track.box_num]"
        if split_ind == self.box_num - 1:
            self.track_id = track_id1
            return self, None
        elif split_ind == 0:
            self.track_id = track_id2
            return None, self
        else:
            track1 = Track(box_list=self[:split_ind + 1], track_id=track_id1)
            track2 = Track(box_list=self[split_ind + 1:], track_id=track_id2)
            return track1, track2

    def list_all_boxes(self):
        # print all the boxes in a track
        print 'Listing all boxes of track:'
        print(self)
        for i in range(len(self._box_list)):
            print '    | %4d' % i, self._box_list[i]

    def get_rois(self, seg_start, frames, start, end):
        # return a list of numpy arrays (rois) from start to end
        assert isinstance(frames), 'list of frames is accpeted.'
        start = self.start_fid if start is None else start
        end = self.end_fid if end is None else end
        num_frames = len(frames)
        assert self.start_fid <= start <= self.end_fid, 'invalid start fid'
        assert self.start_fid <= end <= self.end_fid, 'invalid start fid'
        assert start <= end, 'start must be no more than end'
        assert num_frames >= end - start + 1, 'frames has less frames than required'

        roi_list = []
        append = list.append
        for box in self._box_list:
            if box.frame_id > end or box.frame_id < start:
                continue
            assert 0 <= box.frame_id - seg_start < num_frames, 'out of bound. wrong seg_start value'
            frame = frames[box.frame_id - seg_start]
            roi = frame[box.pos[1]:box.pos[3], box.pos[0]:box.pos[2]]
            append(roi_list, roi)
        return roi_list

    def sample_rois(self):
        # sample rois which are to be fed to the appearance model
        # sort boxes of this track based on their occlusion
        """
        conf_list = sorted([(ind, box.occlusion) for ind, box in enumerate(self._box_list)], key=lambda x:x[1])
        selected_indices, _ = zip(*conf_list)  # selected indices are the indices of boxes with lower occlusion ratio
        return selected_indices[:P['scl_max_compare_length']]
        """
        nb_box_list = len(self._box_list)
        low_occlusion_box_list = [idx for idx, box in enumerate(self._box_list) if box.occlusion < 0.2]
        nb_low_occ_box_list = len(low_occlusion_box_list)
        if nb_low_occ_box_list == 0:
            if nb_box_list >= P['scl_max_compare_length']:
                return random.sample(range(nb_box_list), P['scl_max_compare_length'])
            else:
                return range(nb_box_list)
        else:
            return [random.choice(low_occlusion_box_list) for i in range(P['scl_max_compare_length'])]

    # pkl file is extraordinarily big, use .track file instead
    @staticmethod
    def dump_to_pkl_file(track_list, save_name):
        # dump a list of tracks into file
        assert isinstance(track_list, list), "bad argument, type must be list"
        assert all(isinstance(track, Track) for track in track_list), "entry must be track"
        f = open(save_name, "wb")
        pickle.dump(track_list, f)
        print("Track data saved in {}".format(save_name))
        print("Total track number: {}".format(len(track_list)))

    """
        USE THIS FUNCTION TO SAVE TRACKS INSTEAD!
        This function uses struct to dumpy tracks onto disk which follows the format of C++ version
        tracker. This one saves much more space than dump_to_pkl_file
    """
    @staticmethod
    def dump_to_track_file(track_list, save_name):
        # dump a list of tracks into file
        assert isinstance(track_list, list), "bad argument, type must be list"
        assert all(isinstance(track, Track) for track in track_list), "entry must be track"
        with open(save_name, 'wb') as f:
            for track in track_list:
                data = struct.pack("4i", track.track_id, track.start_fid, track.end_fid, track.box_num)
                f.write(data)
                for ind, b in enumerate(track._box_list):
                    x1, y1, x2, y2 = b.pos
                    data = struct.pack("f7i", 1.0, b.frame_id, ind, x1, y1, x2, y2, 1)
                    f.write(data)
                data = struct.pack('=%sf' % P['feature_dim'], *(track.feature).flatten('F'))
                f.write(data)

    @staticmethod
    def dump_to_track_file_with_gt(track_list, save_name):
        assert isinstance(track_list, list), "bad argument, type must be list"
        assert all(isinstance(track, Track) for track in track_list), "entry must be track"
        with open(save_name, 'wb') as f:
            for track in track_list:
                print(track.track_id, track.start_fid, track.end_fid, track.gt_track_id)
                data = struct.pack("4i", track.track_id, track.start_fid, track.end_fid, track.gt_track_id)
                f.write(data)
                assert track.feature is not None, "feature is not none"
                data = struct.pack('=%sf' % P['feature_dim'], *(track.feature).flatten('F'))
                f.write(data)
        with open(save_name, 'rb') as f:
            while True:
                data = f.read(4 * 4)
                if not data:
                    break
                track_id, start_fid, end_fid, gt_track_id = struct.unpack("4i", data)
                raw = f.read(4 * P['feature_dim'])

    @staticmethod
    def dump_to_track_file_no_feature(track_list, save_name, w_mat=None, h_mat=None):
        # dump a list of tracks into file
        assert isinstance(track_list, list), "bad argument, type must be list"
        assert all(isinstance(track, Track) for track in track_list), "entry must be track"
        with open(save_name, 'wb') as f:
            for track in track_list:
                data = struct.pack("4i", track.track_id, track.start_fid, track.end_fid, track.box_num)
                f.write(data)
                for ind, b in enumerate(track._box_list):
                    x1, y1, x2, y2 = b.pos
                    data = struct.pack("f7i", 1.0, b.frame_id, ind, x1, y1, x2, y2, 1)
                    f.write(data)
                # data = struct.pack('=%sf' % P['feature_dim'], *(track.feature).flatten('F'))
                # f.write(data)

    #TODO: convert .track file to .txt mot format
    @staticmethod
    def convert_track_to_mot_format(track_file, mot_file):
        track_list = Track.load_from_track_file(track_file, has_feature=False)
        Track.dump_track_with_mot_format(track_list, mot_file)

    @staticmethod
    def dump_track_with_mot_format(track_list, save_name):
        final_box_list = []
        for track in track_list:
            track_id = track.track_id

            for ind, b in enumerate(track._box_list):
                x1, y1, x2, y2 = b.pos
                frame_id = b.frame_id
                final_box_list.append((frame_id + 1, track_id, x1, y1, x2 - x1, y2 - y1))

        final_box_arr = np.array(final_box_list)
        final_box_idx_arr = np.argsort(final_box_arr[:, 0]).flatten()
        final_box_arr = final_box_arr[final_box_idx_arr]
        with open(save_name, 'w') as f:
            for item in final_box_arr:
                str_item = ""
                for info in item:
                    str_item += str(info) + ", "
                str_item += "-1, -1, -1, -1\n"
                f.write(str_item)


    @staticmethod
    def load_from_track_file(track_file, has_feature=True):
        assert os.path.exists(track_file), "cannot open {}".format(track_file)
        track_list = []
        append = list.append
        with open(track_file, 'rb') as f:
            while True:
                raw = f.read(16)
                if not raw:
                    break
                tid, start_fid, end_fid, box_num = struct.unpack("4i", raw)
                box_list = []
                for i in range(box_num):
                    raw = f.read(32)
                    assert raw, "error reading file {}".format(track_file)
                    conf, fid, _, x1, y1, x2, y2, _ = struct.unpack("f7i", raw)
                    box = Box(pos=(x1, y1, x2, y2), frame_id=fid, track_id=tid, confidence=conf)
                    append(box_list, box)
                track = Track(box_list, tid)
                if has_feature:
                    raw = f.read(4 * P['feature_dim'])
                    feature = np.array(struct.unpack('=%sf' % P['feature_dim'], raw))
                    track.feature = feature

                append(track_list, track)
        return track_list

    def gaussian_smooth(self):
        """
            Apply Gaussian Process Regression to smooth track
            Here we use sklearn.GaussianProcessRegression package which requires
            input X as a N * M array while label Y as a 1-D array of size N.
            N is number of samples and M is dimension of each sample.

            In this case, X is frame id and Y is the center X/Y coordinates
        """
        assert self.box_num > 0, "track is empty"

        # prepare training data X
        fid_data = np.ndarray(shape=(self.box_num, 1))
        cx_data = np.ndarray(shape=(self.box_num,))
        cy_data = np.ndarray(shape=(self.box_num,))
        for ind, box in enumerate(self._box_list):
            fid_data[ind, 0] = box.frame_id
            cx_data[ind], cy_data[ind] = box.center

        # normalize data
        fid_max, fid_min = np.amax(fid_data), np.amin(fid_data)
        fid_data = (fid_data - fid_min) / (fid_max - fid_min)
        cx_data /= P['frame_width']
        cy_data /= P['frame_height']

        # fit data
        self._gpx.fit(fid_data, cx_data)
        self._gpy.fit(fid_data, cy_data)

        # prediction
        cx_pred = self._gpx.predict(fid_data)
        cy_pred = self._gpy.predict(fid_data)
        cx_pred *= P['frame_width']
        cy_pred *= P['frame_height']

        # only smooth velocity
        for i in range(self.box_num):
            cx, cy = cx_pred[i], cy_pred[i]
            if i != self.box_num - 1:
                self._box_list[i].velocity = (cx_pred[i+1]-cx, cy_pred[i+1]-cy)
            else:
                self._box_list[i].velocity = (cx - cx_pred[i-1], cy-cy_pred[i-1])

    def smooth_velocity(self):
        """
            Smooth velocity using gaussian kernel correlation (1D filtering)
            the length of the kernel is determined by P['velocity_smooth_range'].
            To get correct result, self.box_num must be greater than or equal to
            kernel length. The correlation operation is conducted on every data point
            of vx, vy by specifying parameter 'valid' (the length of the smoothed velocity
            will be self.box_num - kernel length + 1
            E.g.
                if vx = [0, 3, 6], kernel = [0.2, 0.6, 0.2]
                then vx_smoothed = [3]  ( 3 = 0.2*0 + 0.6*3 + 0.2*6 )

            Also, this function will be called every time a new box or a track is append to
            this one, assigning new values to the velocities of boxes at the front and back
            ends of the track, making the velocities being not None.
            When computing box affinities in low level association, if the velocity is not
            smoothed, the velocity properties of boxes will be None, the function will then
            use the positions of boxes to compute affinities. Otherwise velocity will be used
            instead of position.
        """
        cx, cy = zip(*[b.center for b in self._box_list]) # cx cy is a list axis is frame id
        cx, cy = np.array(cx), np.array(cy)
        vx, vy = np.diff(cx), np.diff(cy)
        if self.box_num < velocity_smoothing_kernel.shape[0]:
            self._box_list[0].velocity = (vx[0], vy[0])
            self._box_list[-1].velocity = (vx[-1], vy[-1])
        else:
            vx_smoothed = np.correlate(vx, velocity_smoothing_kernel, 'same')
            vy_smoothed = np.correlate(vy, velocity_smoothing_kernel, 'same')
            self._box_list[0].velocity = (vx_smoothed[0], vy_smoothed[0])
            self._box_list[-1].velocity = (vx_smoothed[-1], vy_smoothed[-1])

    def motion_and_time_affinity(self, other, w_mat, h_mat):
        """
            compute motion and time affinity together
            Time affinity:
                In the previous c++ version, we apply a hard time affinity computation:
                    time_aff = 0 or 1 if two tracklets do not overlap
                This is error-prone since we are not allowing mistakes made by low level
                association, i.e. two tracklets of the same person may have an overlap
                of 1 or 2 ambiguous boxes. As a result, this hard affinity will push
                Hungarian algorithm not to match these two tracklets.
                To allow mistakes made by low level association, here a peaking
                gaussian function is used instead of a hard one.
            Motion affinity:
                If the two tracklets do not overlap, use linear motion prediction
                in early c++ version. Else, apply box dist metric used in low
                level association
        """
        if self.start_fid > other.start_fid:
            # other must start after the ending of self
            return 0.
        elif self.end_fid < other.start_fid:
            gh, gw = w_mat.shape

            # head is the beginning box of the latter track while tail is the end box of the previous one
            tail, head = self[-1], other[0]

            delta_time = head.frame_id - tail.frame_id  # delta time
            v_tail_x, v_tail_y = tail.velocity  # velocity of tail
            v_head_x, v_head_y = head.velocity  # velocity of head
            cx_tail, cy_tail = tail.center  # center of tail
            cx_head, cy_head = head.center  # center of head

            # prediction
            pred_head_x = cx_tail + v_tail_x * delta_time
            pred_head_y = cy_tail + v_tail_y * delta_time
            pred_tail_x = cx_head - v_head_x * delta_time
            pred_tail_y = cy_head - v_head_y * delta_time

            # use precomputed calibration map to rectify distance
            d1 = bresenham_line_integral((pred_tail_x, pred_tail_y), (cx_tail, cy_tail), P['frame_width'],
                                         P['frame_height'], w_mat=w_mat, h_mat=h_mat, nb_grid_w=gw, nb_grid_h=gh)
            d2 = bresenham_line_integral((pred_head_x, pred_head_y), (cx_head, cy_head), P['frame_width'],
                                         P['frame_height'], w_mat=w_mat, h_mat=h_mat, nb_grid_w=gw, nb_grid_h=gh)
            d1 = -d1 * d1 / P['gaussian_pred_sigma']
            d2 = -d2 * d2 / P['gaussian_pred_sigma']

            d1, d2 = math.exp(d1), math.exp(d2)
            return (d1 + d2) / 2

        else:
            s, e = max(self.start_fid, other.start_fid), min(self.end_fid, other.end_fid)
            overlap = e - s + 1
            if overlap > P['th_time_overlap']:
                # if overlap is too big, there is no need to compute motion affinity
                return 0.
            time_aff = np.exp(-float(overlap) / P['time_affinity_gaussian_sigma'])

            # compute distance between overlapping boxes
            overlap_list1 = self[self.box_num - overlap:]
            overlap_list2 = other[:overlap]

            dist = 0.
            for i in range(overlap):
                box1, box2 = overlap_list1[i], overlap_list2[i]
                c1, c2 = box1.center, box2.center
                dx, dy = c1[0] - c2[0], c1[1] - c2[1]
                dx *= dx
                dy *= dy
                dist += math.exp(-float(dx + dy) / P['gaussian_dist_sigma'])

            dist /= overlap
            return time_aff * dist

    @staticmethod
    def compute_track_size_affinity(track1, track2):
        assert isinstance(track1, Track) and isinstance(track2, Track), "Track is expected"
        if Track.overlap(track1, track2):
            return 0.
        if track1.start_fid > track2.start_fid:
            return Track.compute_track_pos_affinity(track2, track1)
        if track1.end_fid < track2.end_fid:
            w1, h1 = track1.size(is_head=False)
            w2, h2 = track2.size(is_head=True)
            return math.exp(-P['size_affinity'] * (abs(w1 - w2) / abs(w1 + w2) + abs(h1 - h2) / abs(h1 + h2)))
        else:
            s, e = max(track1.start_fid, track2.start_fid), min(track1.end_fid, track2.end_fid)
            overlap = e - s + 1
            # compute distance between overlapping boxes
            overlap_list1 = track1[track1.box_num - overlap:]
            overlap_list2 = track2[:overlap]
            dist = 0.
            w1, h1 = 0., 0.
            w2, h2 = 0., 0.
            for i in range(overlap):
                box1, box2 = overlap_list1[i], overlap_list2[i]
                w1 += box1.width
                h1 += box1.height
                w2 += box2.width
                h2 += box2.height
            w1, h1, w2, h2 = w1 / overlap, h1 / overlap, w2 / overlap, h2 / overlap
            return math.exp(-P['size_affinity'] * (abs(w1 - w2) / abs(w1 + w2) + abs(h1 - h2) / abs(h1 + h2)))

    @staticmethod
    def compute_track_pos_affinity(track1, track2):
        assert isinstance(track1, Track) and isinstance(track2, Track), "Track is expected"
        if track1.start_fid > track2.start_fid:
            return Track.compute_track_pos_affinity(track2, track1)
        if Track.overlap(track1, track2):
            return 0.
        if track1.end_fid < track2.end_fid:
            w1, h1 = track1.size(is_head=False)
            w2, h2 = track2.size(is_head=True)
            # head is the beginning box of the latter track while tail is the end box of the previous one
            tail, head = track1[-1], track2[0]

            delta_time = head.frame_id - tail.frame_id  # delta time
            v_tail_x, v_tail_y = tail.velocity  # velocity of tail
            v_head_x, v_head_y = head.velocity  # velocity of head
            cx_tail, cy_tail = tail.center  # center of tail
            cx_head, cy_head = head.center  # center of head

            # prediction
            pred_head_x = cx_tail + v_tail_x * delta_time
            pred_head_y = cy_tail + v_tail_y * delta_time
            pred_tail_x = cx_head - v_head_x * delta_time
            pred_tail_y = cy_head - v_head_y * delta_time

            return math.exp(-P["motion_affinity"] * ((((pred_head_x - cx_head) / w1) ** 2) +
                (((pred_head_y - cy_head) / h1) ** 2))) * np.exp(-P["motion_affinity"] *
                    ((((pred_tail_x - cx_tail) / w2) ** 2) + (((pred_tail_y - cy_tail) / h2) ** 2)))
        else:
            s, e = max(track1.start_fid, track2.start_fid), min(track1.end_fid, track2.end_fid)
            overlap = e - s + 1
            # compute distance between overlapping boxes
            overlap_list1 = track1[track1.box_num - overlap:]
            overlap_list2 = track2[:overlap]
            dist = 0.
            for i in range(overlap):
                box1, box2 = overlap_list1[i], overlap_list2[i]
                c1, c2 = box1.center, box2.center
                dx, dy = (c1[0] - c2[0]) * 2 / (box1.width + box2.width), c1[1] - c2[1] / (box1.height + box2.height)
                dist += float(dx ** 2 + dy ** 2)
            dist /= overlap
            return math.exp(-P["motion_affinity"] * dist)

    @staticmethod
    def compute_track_appearance_affinity(track1, track2):
        assert isinstance(track1, Track) and isinstance(track2, Track), "Track is expected"
        if Track.overlap(track1, track2):
            return 0.
        return 1 - cosine(track1.feature, track2.feature)

    @staticmethod
    def compute_track_motion_affinity(track1, track2, w_mat, h_mat):
        assert isinstance(track1, Track) and isinstance(track2, Track), 'Track is expected'
        mo_aff = track1.motion_and_time_affinity(track2, w_mat, h_mat)
        return mo_aff

    @staticmethod
    def compute_track_affinity_by_feature_list(track1, track2):
        feature_list1 = track1.feature_list
        feature_list2 = track2.feature_list
        assert len(feature_list1) > 0 and len(feature_list2) > 0, "feature list should not be none"
        dist_mat = cdist(feature_list1, feature_list2)
        return np.min(dist_mat)


if __name__ == '__main__':
    # Run some test cases of Track class
    dummy_box_list = []
    for i in range(10):
        dummy_box_list.append(Box(frame_id=i, track_id=1))
    dummy_track = Track(dummy_box_list)
    print dummy_track
    print

    print 'Slicing'
    print dummy_track[::-1]
    print
