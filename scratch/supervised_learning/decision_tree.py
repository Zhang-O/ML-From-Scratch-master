# -*- coding: utf8 -*-
"""
naive bayes
author: zhangyalong
ref: 李航 统计学习方法
"""

import numpy as np

dataSet = np.array([[u'青年', u'否', u'否', u'一般', u'拒绝'],
                    [u'青年', u'否', u'否', u'好', u'拒绝'],
                    [u'青年', u'是', u'否', u'好', u'同意'],
                    [u'青年', u'是', u'是', u'一般', u'同意'],
                    [u'青年', u'否', u'否', u'一般', u'拒绝'],
                    [u'中年', u'否', u'否', u'一般', u'拒绝'],
                    [u'中年', u'否', u'否', u'好', u'拒绝'],
                    [u'中年', u'是', u'是', u'好', u'同意'],
                    [u'中年', u'否', u'是', u'非常好', u'同意'],
                    [u'中年', u'否', u'是', u'非常好', u'同意'],
                    [u'老年', u'否', u'是', u'非常好', u'同意'],
                    [u'老年', u'否', u'是', u'好', u'同意'],
                    [u'老年', u'是', u'否', u'好', u'同意'],
                    [u'老年', u'是', u'否', u'非常好', u'同意'],
                    [u'老年', u'否', u'否', u'一般', u'拒绝'],
                    ])

def createDataSet():
    """
    创建数据集

    :return:
    """
    dataSet = np.array([[u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'青年', u'否', u'否', u'好', u'拒绝'],
               [u'青年', u'是', u'否', u'好', u'同意'],
               [u'青年', u'是', u'是', u'一般', u'同意'],
               [u'青年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'一般', u'拒绝'],
               [u'中年', u'否', u'否', u'好', u'拒绝'],
               [u'中年', u'是', u'是', u'好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'中年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'非常好', u'同意'],
               [u'老年', u'否', u'是', u'好', u'同意'],
               [u'老年', u'是', u'否', u'好', u'同意'],
               [u'老年', u'是', u'否', u'非常好', u'同意'],
               [u'老年', u'否', u'否', u'一般', u'拒绝'],
               ])
    labels = np.array([u'年龄', u'有工作', u'有房子', u'信贷情况'])
    # 返回数据集和每个维度的名称
    return dataSet, labels


X = dataSet[:,:-1]
Y = dataSet[:,-1]


def entropy(data, dim=-1):
    # data = np.append(x, y.reshape(y.shape[0], 1), axis=1)
    if data.shape[0] ==0:
        print("data have no elements.")
        return

    label_set = np.unique(data[:, dim])

    label_dict = {}
    for label in label_set:
        label_dict[label] = np.sum(data[:, dim] == label) / data.shape[0]

    shannonEnt = 0.0
    for key, value in label_dict.items():
        shannonEnt += -value * np.log2(value)

    return shannonEnt


def condition_entropy(data, dim):
    # data = np.append(x, y.reshape(y.shape[0], 1), axis=1)

    feature_set = np.unique(data[:, dim])

    feature_dict = {}
    for feature in feature_set:
        feature_dict[feature] = np.sum(data[:, dim] == feature) / data.shape[0]

    conditionEnt = 0.0
    for key, value in feature_dict.items():
        conditionEnt += value * entropy(data[data[:, dim] == key])

    return conditionEnt


def info_gain(data, dim):
    return entropy(data) - condition_entropy(data, dim)


def info_gain_ratio(data, dim):
    return info_gain(data, dim) / entropy(dataSet, dim)


def choose_best_feature():
    pass


xxx = info_gain(dataSet, 0)
xxxx = info_gain_ratio(dataSet, 0)

print(111)