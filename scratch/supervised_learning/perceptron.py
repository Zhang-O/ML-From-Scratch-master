# -*- coding: utf8 -*-
"""
感知机对偶形式
author: zhangyalong
ref: 李航 统计学习方法, hankers
"""



import copy
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

tran_data = np.array([[3, 3], [4, 3], [1, 1]], dtype='float32')
tran_label = np.array([1, 1, -1], dtype='float32')

class Perceptron:
    def __init__(self, learning_rate=1, iteration=1000):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.alpha = None
        self.b = None
        self.W = None

        self.history = list()  # for plot

    def fit(self, X, Y):
        self.alpha = np.zeros(X.shape[0], dtype='float32')
        self.b = 0.0

        for i in range(self.iteration):
            for row in range(X.shape[0]):
                if (X[row] @ X.T @ (np.diag(self.alpha) @ Y) + self.b) * Y[row]  <= 0:
                    self.alpha[row] += self.learning_rate
                    self.b += self.learning_rate * Y[row]

                    self.history.append([np.diag(self.alpha) @ Y @ X, self.b])  # for plot
        self.W = np.diag(self.alpha) @ Y @ X


    def predict(self, x):
        return np.sign(self.W @ x + self.b)


model = Perceptron()
model.fit(tran_data, tran_label)
# print(model.predict(np.array([0,0])))



###  for animation
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))

line, = ax.plot([], [], 'g', lw=2)
label = ax.text([], [], '')

def init():

    line.set_data([], [])
    x = tran_data[tran_label >= 0]
    x_ = tran_data[tran_label < 0]

    # x, y, x_, y_ = [], [], [], []
    # for index, p in enumerate(tran_label):
    #     if p > 0:
    #         x.append(tran_data[index][0])
    #         y.append(tran_data[index][1])
    #     else:
    #         x_.append(tran_data[index][0])
    #         y_.append(tran_data[index][1])

    plt.plot(x[:, 0], x[:, 1], 'bo', x_[:, 0], x_[:, 1], 'rx')
    # plt.plot(x, y, 'bo', x_, y_, 'rx')
    plt.axis([-6, 6, -6, 6])
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Perceptron Algorithm 2 (www.hankcs.com)')
    return line, label


def animate(i):
    # state global variables
    # global ax, line, label

    w = model.history[i][0]
    b = model.history[i][1]
    if w[1] == 0: return line, label
    x1 = -7.0
    y1 = -(b + w[0] * x1) / w[1]
    x2 = 7.0
    y2 = -(b + w[0] * x2) / w[1]
    line.set_data([x1, x2], [y1, y2])
    x1 = 0.0
    y1 = -(b + w[0] * x1) / w[1]
    label.set_text(str(w) + ' ' + str(b))
    label.set_position([x1, y1])
    return line, label

# call the animator.  blit=true means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(model.history), interval=1000, repeat=True,
                               blit=True)
plt.show()
