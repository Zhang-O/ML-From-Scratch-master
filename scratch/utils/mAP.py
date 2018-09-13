import numpy as np
import matplotlib.pyplot as plt
"""
codes below are about computation of AP, mAP is merely the average of AP.
"""

table = np.array([
    [1, 0.23, 0],
    [2, 0.76, 1],
    [3, 0.01, 0],
    [4, 0.91, 1],
    [5, 0.13, 0],
    [6, 0.45, 0],
    [7, 0.12, 1],
    [8, 0.03, 0],
    [9, 0.38, 1],
    [10, 0.11, 0],
    [11, 0.03, 0],
    [12, 0.09, 0],
    [13, 0.65, 0],
    [14, 0.07, 0],
    [15, 0.12, 0],
    [16, 0.24, 1],
    [17, 0.1, 0],
    [18, 0.23, 0],
    [19, 0.46, 0],
    [20, 0.08, 1]
])


# def takeSecond(elem):
#     return elem[1]

# table.sort(key=takeSecond, reverse=True)
# T = sorted(table, key=takeSecond, reverse=True)
# print(table)

index = np.argsort(table[:,1])[::-1]
# print(index)
# print(table[index])
T = table[index]
length = T.shape[0]
print(length)

TP = len(T[T[:,-1] > 0])
print(TP)

result = []
for i in range(length):
    temp = np.sum(T[: i+1, -1] > 0)
    result.append([temp / TP, temp / (i + 1)])

# print(result)
result1 = np.array(result)
result2 = []

"""
N个样本中有M个正例，那么我们会得到M个recall值（1/M, 2/M, ..., M/M）,对于每个recall值r，我们可以计算出对应（r' > r）的最大precision，然后对这M个precision值取平均即得到最后的AP值"""


for i in range(TP):
    result2.append([(i + 1) / TP, np.max(result1[result1[:, 0] == (i + 1) / TP][:, 1])])

result2 = np.array(result2)


result3 = []
for i in range(result2.shape[0]):
    result3.append([result2[i, 0], np.max(result2[i:result2.shape[0], 1])])

# print(result3)
result3 = np.array(result3)
plt.plot(result3[:, 0], result3[:, 1])
plt.show()

AP = np.mean(result3[:, 1])
print(AP)









