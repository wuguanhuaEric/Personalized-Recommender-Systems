#-*-coding:utf8-*-
"""
author:Eric
date:20190608
feature selection and data selection
"""

import pandas as pd
import numpy as np
import operator
import sys

def get_input(input_train_file, input_test_file):
    """
    Args:
        input_train_file:
        input_test_file:
    Return:
        pd.DataFrame train_data
        pd.DataFrame test_data
    """
    dtype_dict = {"age":np.int32,
                  "education-num":np.int32,
                  "capital-gain":np.int32,
                  "capital-loss":np.int32,
                  "hours-per-week":np.int32}
    use_list = [i for i in range(15)]
    use_list.remove(2)
    train_data_df = pd.read_csv(input_train_file, sep=",",header=0, dtype = dtype_dict, na_values = "?", usecols = use_list)
    print(train_data_df.shape)
    train_data_df = train_data_df.dropna(axis=0, how="any")
    print(train_data_df.shape)
    test_data_df = pd.read_csv(input_test_file, sep=",",header=0, dtype = dtype_dict, na_values = "?", usecols = use_list)
    test_data_df = test_data_df.dropna(axis=0, how="any")
    return train_data_df, test_data_df

def label_trans(x):
    """
    Args:
        x:each element in fix col of df
    """
    if x == "<=50K":
        return "0"
    if x == ">=50K":
        return "1"
    return "0"

def process_label_feature(label_feature_str, df_in):
    """
    Args:
        label_feature_str:"label"
        df_in: DataFrame
    """
    df_in.loc[:, label_feature_str] = df_in.loc[:, label_feature_str].apply(label_trans)

def dict_trans(dict_in):
    """
    Args:
        dict_in: key str, value int
    Return:
        a dict, key str value index for example 0, 1, 2
    """
    output_dict = {}
    index = 0
    for zuhe in sorted(dict_in.items(), key = operator.itemgetter(1), reverse = True):
        output_dict[zuhe[0]] = index
        index += 1
    return output_dict

def dis_to_feature(x, feature_dict):
    """
    Args:
        x: element
        feature_dict: pos dict
    Return:
        a str as [1, 0, 0]
    """
    output_list = [0] * len(feature_dict)
    if x not in feature_dict:
        return ",".join([str(ele) for ele in output_list])
    else:
        index = feature_dict[x]
        output_list[index] = 1
    return ",".join([str(ele) for ele in output_list])


def process_dis_feature(feature_str, df_train, df_test):
    """
    Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process dis feature for lr train
    """
    origin_dict = df_train.loc[:, feature_str].value_counts().to_dict()
    print(origin_dict)
    feature_dict = dict_trans(origin_dict)
    print("^^^")
    print(feature_dict)
    print("***")
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(dis_to_feature, args = (feature_dict,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(dis_to_feature, args=(feature_dict,))
    print(df_train.loc[:3, feature_str])
    print(feature_dict)
    return len(feature_dict)

def list_trans(input_dict):
    """
    Args:
        input_dict:
    Return:
        a list, [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    output_list = [0] * 5
    key_list = ["min", "25%", "50%", "75%", "max"]
    for index in range(len(key_list)):
        fix_key = key_list[index]
        if fix_key not in input_dict:
            print("error")
            sys.exit()
        else:
            output_list[index] = input_dict[fix_key]
    return output_list

def con_to_feature(x, feature_list):
    """
    Args:
        x: element
        feature_list: list for feature trans
    Return:
        str, "1_0_0_0"
    """
    feature_len = len(feature_list) - 1
    result = [0] * feature_len
    for index in range(feature_len):
        if x >= feature_list[index] and x <= feature_list[index + 1]:
            result[index] = 1
            return ",".join([str(ele) for ele in result])
    return ",".join([str(ele) for ele in result])


def process_con_feature(feature_str, df_train, df_test):
    """
    Args:
        feature_str: feature_str
        df_train: train_data_df
        df_test: test_data_df
    Return:
        the dim of the feature output
    process con feature for lr train
    """
    origin_dict = df_train.loc[:, feature_str].describe().to_dict()
    print(origin_dict)
    feature_list = list_trans(origin_dict)
    df_train.loc[:, feature_str] = df_train.loc[:, feature_str].apply(con_to_feature, args=(feature_list,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(con_to_feature, args=(feature_list,))
    print(df_train.loc[:3,feature_str])
    print(feature_list)
    return len(feature_list) - 1

def output_file(df_in, out_file):
    """

    write data of df_in to out_file
    """
    fw = open(out_file, "w+", encoding="utf-8")
    for row_index in df_in.index:
        outline = ",".join([str(ele) for ele in df_in.loc[row_index].values])
        fw.write(outline + "\n")
    fw.close()

def add(str_one, str_two):
    """
    Args:
        str_one: "0,0,1,0"
        str_two: "1,0,0,0"
    Return:
        "0,0,1,0,0"
    """
    list_one = str_one.split(",")
    list_two = str_two.split(",")
    list_one_len = len(list_one)
    list_two_len = len(list_two)
    return_list = [0] * (list_one_len * list_two_len)
    try:
        index_one = list_one.index("1")
    except:
        index_one = 0
    try:
        index_two = list_two.index("1")
    except:
        index_two = 0
    return_list[index_one * list_two_len + index_two] = 1
    return ",".join([str(ele) for ele in return_list])



def combine_feature(feature_one, feature_two, new_feature, train_data_df, test_data_df, feature_num_dict):
    """
    Args:
        feature_one:
        feature_two:
        new_feature: combine feature name
        train_data_df:
        test_data_df:
        feature_num_dict: ndim of every feature, key feature name value len of the dim
    Return:
        new_feature_num
    """
    train_data_df[new_feature] = train_data_df.apply(lambda row:add(row[feature_one], row[feature_two]), axis = 1)
    test_data_df[new_feature] = test_data_df.apply(lambda row: add(row[feature_one], row[feature_two]), axis=1)
    if feature_one not in feature_num_dict:
        print("error")
        sys.exit()
    if feature_two not in feature_num_dict:
        print("error")
        sys.exit()
    return feature_num_dict[feature_one] * feature_num_dict[feature_two]

def ana_train_data(input_train_data, input_test_data, out_train_file, out_test_file, feature_num_file):
    """
    Args:
        input_train_data:
        input_test_data:
        out_train_file:
        out_test_file:
        feature_num_file:
    """
    train_data_df, test_data_df = get_input(input_train_data,input_test_data)
    label_feature_str = "label"
    dis_feature_list = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    con_feature_list = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    process_label_feature(label_feature_str, train_data_df)
    process_label_feature(label_feature_str, test_data_df)
    process_dis_feature("workclass", train_data_df, test_data_df)
    dis_feature_num = 0
    con_feature_num = 0
    feature_num_dict = {}
    for dis_feature in dis_feature_list:
        tmp_feature_num = process_dis_feature(dis_feature, train_data_df, test_data_df)
        dis_feature_num += tmp_feature_num
        feature_num_dict[dis_feature] = tmp_feature_num
    for con_feature in con_feature_list:
        tmp_feature_num = process_con_feature(con_feature, train_data_df, test_data_df)
        con_feature_num += tmp_feature_num
        feature_num_dict[con_feature] = tmp_feature_num
    #process_con_feature("age", train_data_df, test_data_df)
    new_feature_len = combine_feature("age", "occupation", "age_co", train_data_df, test_data_df, feature_num_dict)
    print(new_feature_len)
    print(train_data_df["age"][:2])
    print(train_data_df["occupation"][:2])
    print(train_data_df["age_co"][:2])
    output_file(train_data_df, out_train_file)
    output_file(test_data_df, out_test_file)
    print(dis_feature_num)
    print(con_feature_num)
    fw = open(feature_num_file, "w+", encoding="utf-8")
    fw.write("feature_num=" + str(dis_feature_num + con_feature_num))




if __name__ == "__main__":
    ana_train_data("../data/train.txt", "../data/test.txt","../data/train_file", "../data/test_file", "../data/feature_num")


