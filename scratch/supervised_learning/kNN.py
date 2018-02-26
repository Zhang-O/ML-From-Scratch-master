# -*- coding: utf8 -*-
"""
knn
author: zhangyalong
ref: 李航 统计学习方法, hankers
"""

import numpy as np

T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]], dtype='float32')

class node:
    def __init__(self, point, depth):
        self.left = None
        self.right = None
        self.point = point
        self.depth = depth


def median(array):
    """

    :param array:
    :return: the median pointer and its index
    """
    m = int(len(array) / 2)
    return array[m], m


def build_kd_tree(data, depth):

    dim = depth % data.shape[1]  # accoding which dim to split data
    data = data[data[:, dim].argsort()]  # sort the data according to dim
    # data = sorted(data, key=lambda x: x[dim])

    # print(data)
    p, m = median(data)
    # print(p, m)

    tree = node(p, depth)

    if m > 0:
        print('left: m: {}; depth: {}'.format(m, depth))
        tree.left = build_kd_tree(data[:m, :], depth+1)

    if len(data) > 2:
        print('right: m: {}; depth: {}'.format(m, depth))
        tree.right = build_kd_tree(data[m+1:, :], depth+1)

    return tree


def distance(x, y, norm=2.0):
    return pow(np.sum((x - y)**norm), 1 / norm)
    # return np.sqrt(np.sum((x - y)**2))




result = build_kd_tree(T, 0)
print('ok')