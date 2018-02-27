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
        self.parent = None


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
    # data = sorted(data, key=lambda x: x[dim])  # not suitable to ndarray type

    # print(data)
    p, m = median(data)
    # print(p, m)

    tree = node(p, depth)
    # print('m: {}; depth: {} parent: {}'.format(m, depth, tree.parent))
    if m > 0:

        tree.left = build_kd_tree(data[:m, :], depth+1)
        tree.left.parent = tree
        # print('left: m: {}; depth: {} parent: {}'.format(m, tree.left.depth, tree.left.parent))
        # print('left: m: {}; depth: {} parent: {}'.format(m, depth, id(tree.parent)))

    if len(data) > 2:

        tree.right = build_kd_tree(data[m+1:, :], depth+1)
        tree.right.parent = tree
        # print('right: m: {}; depth: {} parent: {}'.format(m, tree.right.depth, tree.right.parent))
        # print('right: m: {}; depth:{}  parent: {}'.format(m, depth, id(tree.parent)))

    return tree


def distance(x, y, norm=2.0):
    print(x,y)
    return pow(np.sum((x - y)**norm), 1 / norm)
    # return np.sqrt(np.sum((x - y)**2))


def search_kd_tree(tree, d, target):
    """

    :param tree:
    :param d:
    :param target:
    :return:
    """
    dim = tree.depth % len(target)
    if target[dim] < tree.point[dim]:
        if tree.left is not None:
            return search_kd_tree(tree.left, d, target)
    else:
        if tree.right is not None:
            return search_kd_tree(tree.right, d, target)


    def update_best(t, _best):
        if t is None:
            return
        t = t.point
        dis = distance(t, target)
        if dis < _best[1]:
            _best[1] = dis
            _best[0] = t


    best = [tree.point, 100000.0]
    while tree.parent is not None:
        update_best(tree.parent.left, best)
        update_best(tree.parent.right, best)
        tree = tree.parent
    return best[0]



kd_tree = build_kd_tree(T, 0)
print(search_kd_tree(kd_tree, 0, [9, 4]))
print('ok')