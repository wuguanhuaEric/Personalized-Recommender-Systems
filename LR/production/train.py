#-*-coding:utf8-*-
"""
author:Eric
date:20190623
train lr model
"""
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.externals import joblib

def train_lr_model(train_file, model_coef, model_file):
    """
    Args:
        train_file: process file for lr train
        model_coef: w1 w2 ...
        model_file: pkl
    Return:
    """
    total_feature_num = 32
    train_label = np.genfromtxt(train_file,dtype=np.int32, delimiter = ",", usecols = -1)
    feature_list = range(total_feature_num)
    train_feature = np.genfromtxt(train_file, dtype=np.int32, delimiter=",", usecols= feature_list)
    lr_cf = LRCV(Cs=[1, 10], penalty="l2", tol = 0.0001, max_iter=500, cv=3, scoring='roc_auc').fit(train_feature, train_label)
    score = lr_cf.scores_
    print(train_feature)
    print(train_label)
    print(score)
    coef =(lr_cf.coef_[0])
    fw = open(model_coef, "w+")
    fw.write(",".join(str(ele) for ele in coef))
    fw.close()
    joblib.dump(lr_cf, model_file)
    #print("diff: %s" %(",".join([str(ele) for ele in score.mean(axis = 0)])))


if __name__ == "__main__":
    train_lr_model("../data/train_file", "../data/lr_coef", "../data/lr_model_file")