# -*- coding: utf8 -*-
"""
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

    def fit(self, X, Y):
        self.alpha = np.zeros(X.shape[0], dtype='float32')
        self.b = 0.0

        for i in range(self.iteration):
            for row in range(X.shape[0]):
                if (X[row] @ X.T @ (np.diag(self.alpha) @ Y) + self.b) * Y[row]  <= 0:
                    self.alpha[row] += self.learning_rate
                    self.b += self.learning_rate * Y[row]
        self.W = np.diag(self.alpha) @ Y @ X


    def predict(self, x):
        return np.sign(self.W @ x + self.b)


model = Perceptron()
model.fit(tran_data, tran_label)
print(model.predict(np.array([0,0])))


fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))

line, = ax.plot([], [], 'g', lw)


# call the animator.  blit=true means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=1000, repeat=True,
                               blit=True)
