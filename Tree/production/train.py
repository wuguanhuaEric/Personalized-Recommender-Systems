#-*-coding:utf8-*-
"""
author:Eric
date:20190719
train gbdt model
"""

import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from scipy.sparse import coo_matrix


def get_train_data(train_file, feature_num_file):
    """
    get train data and label for training
    """
    total_feature_num = 32
    train_label = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=-1)
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols=feature_list)
    return train_feature, train_label

def train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate):
    """
    Args:
        train_mat: train data and label
        tree_depth:
        tree_num: total tree num
        learning_rate: step_size
    Return: Booster
    """
    para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent":1}
    bst = xgb.train(para_dict, train_mat, tree_num)
    print(xgb.cv(para_dict, train_mat, tree_num, nfold=2, metrics={"auc"}))
    return bst

def choose_parameter():
    """
    Return:
        list: such as [(tree_depth, tree_num, learning_rate)]
    """
    result_list = []
    tree_depth_list = [4, 5]
    tree_num_list = [10]
    learning_rate_list = [0.3, 0.5]
    for ele_tree_depth in tree_depth_list:
        for ele_tree_num in tree_num_list:
            for ele_learning_rate in learning_rate_list:
                result_list.append((ele_tree_depth, ele_tree_num, ele_learning_rate))
    return result_list

def grid_search(train_mat):
    """
    Args:
        train_mat: train_data and train_label
    select the best parameter for training model
    """
    para_list = choose_parameter()

    for ele in para_list:
        (tree_depth, tree_num, learning_rate) = ele
        para_dict = {"max_depth": tree_depth, "eta": learning_rate, "objective": "reg:linear", "silent": 1}
        res = xgb.cv(para_dict, train_mat, tree_num, nfold=2, metrics={"auc"})
        auc_score = res.loc[tree_num - 1, ['test-auc-mean']].values[0]
        print("tree_depth:%s, tree_num:%s, learning_rate:%s, auc:%f" % (tree_depth, tree_num, learning_rate, auc_score))

def train_tree_model(train_file, feature_num_file, tree_model_file):
    """
    Args:
        train_file: data for train model
        feature_num_file: file to record feature total num
        tree_model_file: file to store model
    Return
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    grid_search(train_mat)
    tree_num = 10
    tree_depth = 6
    learning_rate = 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(tree_model_file)

def get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth):
    """
    Args:
        tree_leaf: prediction of the tree model
        tree_num: total tree_num
        tree_depth: total_tree_depth
    Return:
        Sparse Matrix to record total train feature for lr part of mixed model
    """
    total_node_num = 2 ** (tree_depth + 1) - 1
    yezi_num =  2 ** tree_depth
    feiyezi_num = total_node_num - yezi_num
    total_col_num = yezi_num * tree_num
    total_row_num = len(tree_leaf)
    col = []
    row = []
    data = []
    base_row_index = 0
    for one_result in tree_leaf:
        base_col_index = 0
        for fix_index in one_result:
            yezi_index = fix_index - feiyezi_num
            yezi_index = yezi_index if yezi_index >= 0 else 0
            col.append(base_col_index + yezi_index)
            row.append(base_row_index)
            data.append(1)
            base_col_index += yezi_num
        base_row_index += 1
    total_feature_list = coo_matrix((data, (row, col)), shape = (total_row_num, total_col_num))
    return total_feature_list

def train_tree_and_lr_model(train_file, feature_num_file, mix_tree_model_file, mix_lr_model_file):
    """
    Args:
        train_file: file for training model
        feature_num_file: file to store total feature len
        mix_tree_model_file: tree part of the mix model
        mix_lr_model_file: lr part of the mix model
    """
    train_feature, train_label = get_train_data(train_file, feature_num_file)
    train_mat = xgb.DMatrix(train_feature, train_label)
    tree_num, tree_depth, learning_rate = 10, 6, 0.3
    bst = train_tree_model_core(train_mat, tree_depth, tree_num, learning_rate)
    bst.save_model(mix_tree_model_file)
    tree_leaf = bst.predict(train_mat, pred_leaf = True)
    print(tree_leaf[0])
    print(np.max(tree_leaf))
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf, tree_num, tree_depth)
    lr_cf = LRCV(Cs=[1, 10], penalty="l2", tol=0.0001, max_iter=500, cv=3, scoring='roc_auc').fit(total_feature_list, train_label)
    fw = open(mix_lr_model_file, "w+")


if __name__ == "__main__":
    train_tree_model("../data/train_file", "../data/feature_num_file", "../data/test.model")
    train_tree_and_lr_model("../data/train_file", "../data/feature_num", "../data/xgb_mix_model", "../data/lr_coef_mix_model")