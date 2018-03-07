import sys
sys.path.append("/home/xksj/workspace/lp/multi_cut/tools")
from FeatureTrainer import BaseTrainer, LogisticTrainer, NNTrainer
import numpy as np
import os

def pre_process_dat(train_data):
    feature_num = train_data.shape[0]
    feature_arr = np.zeros(shape=(feature_num, 7))
    feature_arr[:, :3] = train_data
    feature_arr[:, 3] = train_data[:, 0] * train_data[:, 1]
    feature_arr[:, 4] = train_data[:, 0] * train_data[:, 2]
    feature_arr[:, 5] = train_data[:, 1] * train_data[:, 2]
    feature_arr[:, 6] = train_data[:, 0] * train_data[:, 1] * train_data[:, 2]
    return feature_arr

if __name__ == "__main__":
    train_seq_list = ["02", "04", "05", "09",  "10", "11", "13"]
    # train_seq_list = ["05", "13"]
    for seq in train_seq_list:
        seq_name = "MOT16-{}".format(seq)
        weight_file = os.path.join("trained_model", "MOT16-{}_logistic_weight.txt".format(seq))
        print(seq_name)
        pos_arr = np.loadtxt(seq_name + "_pos_feature.txt")
        neg_arr = np.loadtxt(seq_name + "_neg_feature.txt")
        train_pos_x = pos_arr[:, :-1]
        train_pos_y = pos_arr[:, -1]
        train_neg_x = neg_arr[:, :-1]
        train_neg_y = neg_arr[:, -1]

        X_arr = np.vstack((train_pos_x, train_neg_x))
        Y_arr = np.append(train_pos_y, train_neg_y)

        X_arr = pre_process_dat(X_arr)
        log_train = LogisticTrainer(X=X_arr, Y=Y_arr, iter=1000, test_size=0.2)
        log_train.fit(balance_sample=True)
        log_train.val()
        # log_train.save_model(weight_file)
        """
        nn_trainer = NNTrainer(X_arr, Y_arr, iter=10000, input_size=3, output=2)
        nn_trainer.fit(balance_sample=True)
        nn_trainer.val()
        """
