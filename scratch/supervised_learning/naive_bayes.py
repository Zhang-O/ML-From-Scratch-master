# -*- coding: utf8 -*-
"""
naive bayes
author: zhangyalong
ref: 李航 统计学习方法, hankers
bayes estimate 使用的是 拉普拉斯平滑；
目前是把所有数据统一成str， numpy 貌似不允许不相同的数据在一起，当数值和str在一个数组里面，数值便自动转化为str
其实还有更快的一种办法，如果训练数据大，提前知道x的每一维度的特征种类数，可以create 一个矩阵，存储所有的P(Xi=aij|Y=ck)，
不足的用0补全就行，这样有新的分类任务时就直接取值计算后验概率，避免重复计算
"""

import numpy as np


# X = np.array([['short', 'flat', 'fast'], ['long', 'flat', 'slow'], ['long', 'uneven', 'slow']])
# Y = np.array(['bad', 'good', 'good'])
X = np.array([[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'], [1, 'S'], [2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'], [2, 'L'], [3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'], [3, 'L']])
Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])



class NaiveBayes:
    def __init__(self):
        self.X = None
        self.Y = None
        self.data = None
        self.classes = None

    def fit(self, x, y):
        self.X = x
        self.Y = y.astype(np.str)
        self.classes = np.unique(self.Y)
        # add X and Y to data is for the convenience of derive P(X|Y)
        self.data = np.append(X, Y.reshape(Y.shape[0], 1), axis=1)

    def predict(self, x):
        x = np.array(x).astype(np.str)
        posteriors = []
        for c in self.classes:
            posterior = self._calculate_priori(c)
            for index, value in enumerate(x):
                posterior *= self._cal_conditional_prob(c, index, value)

            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _calculate_priori(self, c):
        """
        Calculate the prior of class c
        (samples where class == c / total number of samples)
        """
        # return np.sum(self.Y == c) / self.Y.shape[0]
        return np.sum(self.data[:, -1] == c) / self.data.shape[0]

    def _cal_conditional_prob(self, c, index, value):
        """
        Laplace smoothing
        :return: P(X|Y) - Likelihood of data X given class distribution Y.
        """
        s = np.unique(self.data[:, index]).shape[0]
        return (np.sum(self.data[self.data[:, -1] == c][:, index] == value) + 1) / (np.sum(self.data[:, -1] == c) + s)


# bayes = NaiveBayes()
# bayes.fit(X, Y)
# print(bayes.predict([2,'S']))





