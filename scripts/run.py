import os
from tracker import Tracker
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_argse():
    arg_parser = argparse.ArgumentParser(description="For MOT tracker")
    arg_parser.add_argument("--mot_sequence", type=str, default="02", required=False, help="specific video sequence")
    return arg_parser.parse_args()



if __name__ == '__main__':

    batch_mode = False
    parsed_args = parse_argse()
    sequence_name = parsed_args.mot_sequence
    if batch_mode:
        data_root = '../data/sed-pd-17/video_test'
        scl_path = '../models/cnnrnn_0608.hdf5'
        save_dir = '../data/debugging'
        cam = 2     # *** DO NOT FORGET TO MODIFY THE CAM NUMBER IN share_parameter.py ***
        video_list_file = '../data/test_video_list_cam{}.txt'.format(cam)
        mask_path = '../data/mask/CAM{}_en_ex_map.bmp'.format(cam)
        calib_path = '../data/calib/CAM{}_calib.pkl'.format(cam)
        with open(video_list_file) as f:
            video_paths = f.readlines()
            for video_path in video_paths:
                video_path = video_path.strip('\n')
                if video_path[0] == '!':
                    continue
                # print(video_path)
                video_name = os.path.basename(video_path)
                data_path = os.path.join(data_root, video_name+'.data') # restore detection
                tracker = Tracker(video_path, data_path, save_dir, scl_path, mask_path, calib_path)
                tracker.run()

    else:
        # args = {'video_dir': '../data/video-data/VideoEval08/LGW_20071206_E1_CAM2.mpeg',
        #         'data_dir': '../data/sed-pd-17/video_train/LGW_20071206_E1_CAM2.mpeg.data',
        #         'save_dir': '../data/debugging',
        #         'scl_path': '../models/cnnrnn_0608.hdf5',
        #         'mask_path': '../data/mask/CAM2_en_ex_map.bmp',
        #         'calib_path': '../data/calib/CAM2_calib.pkl'}

        args = {'video_dir': '/home/xksj/Data/lp/MOT16/MOT16-{}.avi'.format(sequence_name, sequence_name),
            'data_dir': '/home/xksj/Data/lp/MOT16/high_level_det/MOT16-{}_det.txt'.format(sequence_name),
                'save_dir': '../data/debugging',
                'scl_path': '../models/seq_cnn_rnn_nbw_112209_126x42.hdf5',
                'mask_path': '../data/mask/all_white.bmp',
                'calib_path': '../mot-data/det_{}_calib_10x10.pkl'.format(sequence_name),
                'gt_path': '/home/xksj/Data/lp/MOT16/train/MOT16-{}/gt/gt.txt'.format(sequence_name),
                'entry_path':'../data/entry/20171107_193718.txt'}

        video_dir, data_file, save_dir, scl_path, mask_path, calib_path, gt_path, entry_path = \
            args['video_dir'], args['data_dir'], args['save_dir'], args['scl_path'], \
            args['mask_path'], args['calib_path'], args['gt_path'], args['entry_path']

        _, data_name = os.path.split(data_file)
        data_name, ext = os.path.splitext(data_name)

        _, video_name = os.path.split(video_dir)
        _, ext = os.path.splitext(video_name)

        # assert video_name == data_name, 'data file {} does not match video file {}'.format(video_name, data_name)

        tracker = Tracker(video_dir, data_file, save_dir, scl_path, mask_path, calib_path, gt_path=gt_path, entry_path=entry_path)
        tracker.run()

        # save_config(config_file, P)

