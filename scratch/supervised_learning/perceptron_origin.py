# -*- coding: utf8 -*-
"""
感知机的原始形式，未使用numpy，纯 list
author: zhangyalong
ref: 李航 统计学习方法, hankers
"""


import copy
from matplotlib import pyplot as plt
from matplotlib import animation

import logging; logging.basicConfig(level=logging.INFO)



tran_data = [[3, 3], [4, 3], [1, 1]]
tran_label = [1, 1, -1]

class Perceptron:
    def __init__(self):
        self.name = 'perceptron'
        self.W = None
        self.b = None
        self.learning_rate = 1
        self.history = []


    def fit(self, X, Y):
        self.W = [0 for i in range(len(X[0]))]
        self.b = 0
        self.history.append([copy.copy(self.W), self.b])
        for i in range(1000):
            for index, item in enumerate(X):
                if self.cal(item) * Y[index] <= 0:
                    logging.info('a: {}'.format(self.cal(item) * Y[index]))
                    self.update(item, Y[index])


    def cal(self, x):
        a = 0
        for index, w in enumerate(self.W):
            a += w*x[index]
        a += self.b
        return a


    def predict(self, x):
        a = 0
        for index, w in enumerate(self.W):
            a += w*x[index]
        a += self.b
        if a >= 0:
            return 1
        else:
            return -1


    def update(self, x, y):
        """
        update parameters using stochastic gradient descent
        """
        for index, w in enumerate(self.W):
            self.W[index] += self.learning_rate * y * x[index]
        self.b += self.learning_rate * y
        self.history.append([copy.copy(self.W), self.b])

        logging.info('self.W, self.b :{}, {}'.format(self.W, self.b))


    def print_args(self):
        print(self.W, self.b)


model = Perceptron()
model.fit(tran_data, tran_label)
# model.print_args()
print(model.predict([-6, -4]))


###  the following which could be commented is for animation
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], 'g', lw=2)
label = ax.text([], [], '')
training_set = [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]

def init():
    line.set_data([], [])
    x, y, x_, y_ = [], [], [], []
    for p in training_set:
        if p[1] > 0:
            x.append(p[0][0])
            y.append(p[0][1])
        else:
            x_.append(p[0][0])
            y_.append(p[0][1])

    plt.plot(x, y, 'bo', x_, y_, 'rx')
    plt.axis([-6, 6, -6, 6])
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Perceptron Algorithm ')
    return line, label


def animate(i):
    global ax, line, label

    w = model.history[i][0]
    b = model.history[i][1]

    if w[1] ==0:
        return line, label

    # coordinate for two points of line
    x1 = -7
    y1 = -(b + w[0] * x1) / w[1]
    x2 = 7
    y2 = -(b + w[0] * x2) / w[1]
    line.set_data([x1, x2], [y1, y2])

    # coodinate for label
    x3 = 0
    y3 = -(b + w[0] * x3) / w[1]
    label.set_text(model.history[i])
    label.set_position([x3, y3])

    return line, label

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(model.history), interval=1000, blit=True, repeat=False)

plt.show()

