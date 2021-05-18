import math
import numpy as np
from collections import Counter
 

movie_data = {"宝贝当家": [45, 2, 9, "喜剧片"],
              "美人鱼": [21, 17, 5, "喜剧片"],
              "澳门风云3": [54, 9, 11, "喜剧片"],
              "功夫熊猫3": [39, 0, 31, "喜剧片"],
              "谍影重重": [5, 2, 57, "动作片"],
              "叶问3": [3, 2, 65, "动作片"],
              "伦敦陷落": [2, 3, 55, "动作片"],
              "我的特工爷爷": [6, 4, 21, "动作片"],
              "奔爱": [7, 46, 4, "爱情片"],
              "夜孔雀": [9, 39, 8, "爱情片"],
              "代理情人": [9, 38, 2, "爱情片"],
              "新步步惊心": [8, 34, 17, "爱情片"]}
              
'''

# 测试样本  唐人街探案": [23, 3, 17, "？片"]
#下面为求与数据集中所有数据的距离代码：
x = [23, 3, 17]
KNN = []
for key, v in movie_data.items():
    d = math.sqrt((x[0] - v[0]) ** 2 + (x[1] - v[1]) ** 2 + (x[2] - v[2]) ** 2)
    KNN.append([key, round(d, 2)])

# 输出所用电影到 唐人街探案的距离
print(KNN)

#按照距离大小进行递增排序
KNN.sort(key=lambda dis: dis[1])

#选取距离最小的k个样本，这里取k=5；
KNN=KNN[:5]
print(KNN)

#确定前k个样本所在类别出现的频率，并输出出现频率最高的类别
labels = {"喜剧片":0,"动作片":0,"爱情片":0}
for s in KNN:
    label = movie_data[s[0]]
    labels[label[3]] += 1
labels =sorted(labels.items(),key=lambda l: l[1],reverse=True)
print(labels,labels[0][0],sep='\n')
'''
movie_data = [[45, 2, 9, 1],
             [21, 17, 5, 1],
             [54, 9, 11, 1],
             [39, 0, 31, 1],
             [5, 2, 57, 2],
             [3, 2, 65, 2],
             [2, 3, 55, 2],
             [6, 4, 21, 2],
             [7, 46, 4, 3],
             [9, 39, 8, 3],
             [9, 38, 2, 3],
             [8, 34, 17, 3]]

test =[ [23, 3, 17] ]

def KNN(X_train,X_test,k=5):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    # 遍历test_list,在train_list中计算距离
    y_test = [0]*len(X_test)
    for test_index in range(len(X_test)):
        distance_list = []
        for train_index in range(len(X_train)):
            distance_list.append([np.linalg.norm(X_test[test_index]-X_train[train_index][:-1]),X_train[train_index][-1]])
        distance_list.sort(key=lambda x: x[0])
        print(Counter(np.array(distance_list)[:k,1]).most_common())
        y_test[test_index] = Counter(np.array(distance_list)[:k,1]).most_common()[0][0]

    return y_test

y_test = KNN(movie_data,test)
print(y_test)
