# parameter shared by all files
# Different cameras correspond to different parameter settings, which are saved in /configs/

import json
import math
import os

P = {}

def load_from_config(config_path):
    global P
    if os.path.exists(config_path) is False:
        P = {
            'frame_height': 720,
            'frame_width': 1280 ,
            'data_root': './dataset',  # root directory of videos and detection results

            'segment_length': 100,  # video frame buffer size, as a processing unit of tracker
            'overlap_length': 20,  # overlap length of two segments
            'th_detection_confidence': 0.3,  # boxes whose confidence are lower than this will be discarded
            'th_detection_min_width': 10,  # minimum width of detection box to accept
            'th_detection_min_height': 10,  # minimum width of detection box to accept
            'nb_color_hist_bin': 8,  # number of color hist bins
            'gaussian_size_sigma': 1e8,  # sigma used in the gaussian function when comparing sizes of two boxes
            'gaussian_dist_sigma': 200 ** 2,  # sigma used in the gaussian function when comparing distance
            'gaussian_pred_sigma': 75 ** 2,  # sigma used in the gaussian function when comparing distance
            'gaussian_dis_entry_sigma': 10 ** 2,
            'box_affinity_gamma': 1000,  # scale parameter of bha distance between two color histograms
            'velocity_smooth_sigma': 2,  # sigma of gaussian kernel used in velocity smoothing in low level
            'velocity_smooth_range': 7,  # length of 1D gaussian filter
            'head_shoulder': False,  # if detection is based on head-shoulder or whole body
            'head_shoulder_to_body_scale_x': 1,  # horizontal scale to expand head-shoulder box to whole body box
            'head_shoulder_to_body_scale_y': 3,  # vertical scale to expand head-shoulder box to whole body box
            'motion_affinity_scale': 3,  # scaling for motion affinity

            'time_affinity_gaussian_sigma': 2.804,  # sigma used in the gaussian function when computing time affinities
            'th_time_overlap': 2,  # if greater than this value, motion affinity will not be computed

            'th_occlusion_ratio': 0.75,  # boxes whose occlusion ratios are higher than this will be discarded
            'th_low_level_db1': 80,  # the first threshold in double-threshold strategy
            'th_low_level_db2': 10,  # the second threshold in double-threshold strategy
            'th_track_len': 2,  # low level track length threshold
            'th_low_level_max_dist_x': 30,  # if distance between boxes larger than this, affinity will be set to 0
            'th_low_level_max_dist_y': 30,  # if distance between boxes larger than this, affinity will be set to 0
            'th_low_level_max_dist': 30,  # if distance between boxes larger than this, affinity will be set to 0
            'th_low_level_min_velocity_ip': -25,
            'th_high_level_max_dist': 120,

            'log_alpha': math.log(0.27),

            'log_beta': math.log(0.7),  # log of beta (accuracy of detector)
            'log_minus_beta': math.log(0.1),  # log of 1 - beta
            'log_regularizer': math.log(0.9),  # regularizer
            'appearance_scale': 20,  # scale parameter of appearance affinity
            'entry_scale': 10,

            'max_batchsize': 16,  # maximum batchsize to use when computing appearance affinities
            'scl_input_width': 42,  # width of input roi to siamese cnn-lstm network
            'scl_input_height': 126,  # height of input roi to siamese cnn-lstm network
            'scl_max_compare_length': 10,  # max length of roi list to use when computing appearance feature
            'scl_rnn_output_dim': 320,  # output dimension of RNN

            'th_track_fuse_len': 5,  # least number of overlapping boxes if two track are to be fused
            'th_track_fuse_diff': 10,  # max distance between x/y of two boxes if they 'overlap'

            'rbf_kernel_length_scale': 10,  # length scale (lambda) in RBF kernel
            'gpr_alpha': 1e-3,  # alpha value in gaussian process regressor
            'gpr_restart_optimizer': 3,  # number of restart time of gaussian process model optimizer

            # image mean and stdvar
            'im_mean': [140.0684075, 134.53450038, 133.17845961],
            'im_stdvar': [63.05202265, 62.12885622, 58.22406321],
            'redis_db_name': 'mat_det_queue',

            #
            'feature_dim': 512,
            # segment fused
            'segment_fused_time_thresh': 0,
            'segment_fused_distance_thresh': 0.0,
            ############################################
            'mot_embedding_train2test': {"02": "01", "04": "03", "05": "06", "09": "07", "10": "08", "11": "12", "13": "14"},
            'mot_embedding_test2train': {"01": "02", "03": "04", "06": "05", "07": "09", "08": "10", "12": "11", "14": "13"},
            'size_affinity': 1.0,
            'motion_affinity': 1.0,
            'debug': False,
            'coef_path': "/home/xksj/workspace/lp/sed-tracker-mot/feature_classifier/trained_model",
            'mot_tmp_path': "/home/xksj/workspace/lp/sed-tracker-mot/tmp",
            'multicut_exe': "/home/xksj/workspace/lp/multi_cut/cpp/build/test-multicut-track",
        }
    else:
        print '[INIT] Load config from {}'.format(config_path)
        P = json.load(open(config_path, 'r'))

def save_config(save_name, config):
    with open(save_name, 'w') as f:
        json.dump(config, f,  sort_keys=True, indent=4, separators=(',', ': '))


# load_from_config('../configs/CAM3_config.json')
load_from_config('')

"""

'th_detection_confidence': 0.0,  # boxes whose confidence are lower than this will be discarded
'th_detection_min_width': 30,  # minimum width of detection box to accept
'th_detection_min_height': 30, # minimum width of detection box to accept
'nb_color_hist_bin': 8,  # number of color hist bins
'gaussian_size_sigma': 1e8,  # sigma used in the gaussian function when comparing sizes of two boxes
'gaussian_dist_sigma': 200 ** 2,  # sigma used in the gaussian function when comparing distance
'gaussian_pred_sigma': 100 ** 2,  # sigma used in the gaussian function when comparing distance
'box_affinity_gamma': 1000,  # scale parameter of bha distance between two color histograms
'velocity_smooth_sigma': 2,  # sigma of gaussian kernel used in velocity smoothing in low level
'velocity_smooth_range': 7,  # length of 1D gaussian filter
'head_shoulder': False,  # if detection is based on head-shoulder or whole body
'head_shoulder_to_body_scale_x': 1.5,  # horizontal scale to expand head-shoulder box to whole body box
'head_shoulder_to_body_scale_y': 3.6,  # vertical scale to expand head-shoulder box to whole body box
'motion_affinity_scale': 30,    # scaling for motion affinity

'time_affinity_gaussian_sigma': 2.804,  # sigma used in the gaussian function when computing time affinities
'th_time_overlap': 2,  # if greater than this value, motion affinity will not be computed

'th_occlusion_ratio': 1.0,  # boxes whose occlusion ratios are higher than this will be discarded
'th_low_level_db1': 80,  # the first threshold in double-threshold strategy
'th_low_level_db2': 30,  # the second threshold in double-threshold strategy
'th_track_len': 8,  # low level track length threshold
'th_low_level_max_dist_x': 100,  # if distance between boxes larger than this, affinity will be set to 0
'th_low_level_max_dist_y': 100,  # if distance between boxes larger than this, affinity will be set to 0
'th_low_level_max_dist': 200,  # if distance between boxes larger than this, affinity will be set to 0
'th_low_level_min_velocity_ip': -100, # if inner product of velocities of two bounding boxes smaller than this,
# association is banned

'th_high_level_max_dist': 400,    # if distance between tracks is larger than this, the two track wont be linked

'log_alpha': math.log(0.1),  # log of alpha (miss rate of detector)
'log_beta': math.log(0.9),  # log of beta (accuracy of detector)
'log_minus_beta': math.log(0.1),  # log of 1 - beta
'log_regularizer': math.log(0.1),  # regularizer
'appearance_scale': 5,  # scale parameter of appearance affinity

'max_batchsize': 16,  # maximum batchsize to use when computing appearance affinities
'scl_input_width': 42,  # width of input roi to siamese cnn-lstm network
'scl_input_height': 126,  # height of input roi to siamese cnn-lstm network
'scl_max_compare_length': 8,  # max length of roi list to use when computing appearance feature
'scl_rnn_output_dim': 128,  # output dimension of RNN

'th_track_fuse_len': 5,  # least number of overlapping boxes if two track are to be fused
'th_track_fuse_diff': 10,  # max distance between x/y of two boxes if they 'overlap'

'rbf_kernel_length_scale': 10,  # length scale (lambda) in RBF kernel
'gpr_alpha': 1e-4,  # alpha value in gaussian process regressor
'gpr_restart_optimizer': 3,  # number of restart time of gaussian process model optimizer

# image mean and stdvar
'im_mean': [140.0684075, 134.53450038, 133.17845961],
'im_stdvar': [63.05202265, 62.12885622, 58.22406321],
'redis_db_name':'mat_det_queue',

#
'feature_dim': 512,
# segment fused
'segment_fused_time_thresh': 0,
'segment_fused_distance_thresh': 0.0,
"""

"""
# parameter shared by all files
# Different cameras correspond to different parameter settings, which are saved in /configs/

import json
import math
import os

P = {}

def load_from_config(config_path):
    global P
    if os.path.exists(config_path) is False:
        P = {
            'base_path': '/home/xksj/workspace/lp/WJBPROJ/wandering',

            'frame_height': 720,
            'frame_width': 1280 ,
            'data_root': './dataset',  # root directory of videos and detection results

            'segment_length': 250,  # video frame buffer size, as a processing unit of tracker
            'overlap_length': 50,  # overlap length of two segments ps. change to 0 when online

            'th_detection_confidence': 0.8,  # boxes whose confidence are lower than this will be discarded
            'th_detection_min_width': 30,  # minimum width of detection box to accept
            'th_detection_min_height': 30, # minimum width of detection box to accept
            'nb_color_hist_bin': 8,  # number of color hist bins
            'gaussian_size_sigma': 1e8,  # sigma used in the gaussian function when comparing sizes of two boxes
            'gaussian_dist_sigma': 200 ** 2,  # sigma used in the gaussian function when comparing distance
            'gaussian_pred_sigma': 75 ** 2,  # sigma used in the gaussian function when comparing distance
            'gaussian_dis_entry_sigma': 10 ** 2,
            'box_affinity_gamma': 1000,  # scale parameter of bha distance between two color histograms
            'velocity_smooth_sigma': 2,  # sigma of gaussian kernel used in velocity smoothing in low level
            'velocity_smooth_range': 7,  # length of 1D gaussian filter
            'head_shoulder': True,  # if detection is based on head-shoulder or whole body
            'head_shoulder_to_body_scale_x': 1,  # horizontal scale to expand head-shoulder box to whole body box
            'head_shoulder_to_body_scale_y': 2.5,  # vertical scale to expand head-shoulder box to whole body box
            'motion_affinity_scale': 3,    # scaling for motion affinity

            'time_affinity_gaussian_sigma': 2.804,  # sigma used in the gaussian function when computing time affinities
            'th_time_overlap': 2,  # if greater than this value, motion affinity will not be computed

            'th_occlusion_ratio': 0.75,  # boxes whose occlusion ratios are higher than this will be discarded
            'th_low_level_db1': 80,  # the first threshold in double-threshold strategy
            'th_low_level_db2': 80,  # the second threshold in double-threshold strategy
            'th_track_len': 10,  # low level track length threshold
            'th_low_level_max_dist_x': 30,  # if distance between boxes larger than this, affinity will be set to 0
            'th_low_level_max_dist_y': 30,  # if distance between boxes larger than this, affinity will be set to 0
            'th_low_level_max_dist': 30,  # if distance between boxes larger than this, affinity will be set to 0
            'th_low_level_min_velocity_ip': -25, # if inner product of velocities of two bounding boxes smaller than this,
            # association is banned

            'th_high_level_max_dist': 120,    # if distance between tracks is larger than this, the two track wont be linked

            'log_alpha': math.log(0.27),  # log of alpha (miss rate of detector) lower log_alpha, lower track number, higher merge error
            'log_beta': math.log(0.7),  # log of beta (accuracy of detector)
            'log_minus_beta': math.log(0.1),  # log of 1 - beta
            'log_regularizer': math.log(0.9),  # regularizer
            'appearance_scale': 20,  # scale parameter of appearance affinity
            'entry_scale': 10,


            'max_batchsize': 16,  # maximum batchsize to use when computing appearance affinities
            'scl_input_width': 42,  # width of input roi to siamese cnn-lstm network
            'scl_input_height': 126,  # height of input roi to siamese cnn-lstm network
            'scl_max_compare_length': 10,  # max length of roi list to use when computing appearance feature
            'scl_rnn_output_dim': 320,  # output dimension of RNN

            'th_track_fuse_len': 5,  # least number of overlapping boxes if two track are to be fused
            'th_track_fuse_diff': 10,  # max distance between x/y of two boxes if they 'overlap'

            'rbf_kernel_length_scale': 10,  # length scale (lambda) in RBF kernel
            'gpr_alpha': 1e-3,  # alpha value in gaussian process regressor
            'gpr_restart_optimizer': 3,  # number of restart time of gaussian process model optimizer

            # image mean and stdvar
            'im_mean': [140.0684075, 134.53450038, 133.17845961],
            'im_stdvar': [63.05202265, 62.12885622, 58.22406321],
            'redis_db_name':'mat_det_queue',

            #
            'feature_dim': 512,
            # segment fused
            'segment_fused_time_thresh': 10,
            'segment_fused_distance_thresh': 0.6,
        }
    else:
        print '[INIT] Load config from {}'.format(config_path)
        P = json.load(open(config_path, 'r'))

def save_config(save_name, config):
    with open(save_name, 'w') as f:
        json.dump(config, f,  sort_keys=True, indent=4, separators=(',', ': '))


# load_from_config('../configs/CAM3_config.json')
load_from_config('')


"""