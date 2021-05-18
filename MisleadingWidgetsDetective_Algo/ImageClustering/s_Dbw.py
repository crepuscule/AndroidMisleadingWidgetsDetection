# -*- coding: utf-8 -*-
import numpy as np
import math

class S_Dbw(object) :

    '''
    @:param data : np.array， n维的原始数据
    @:param dataclusterIds : np.array， data中每一个数据的聚类clusterId
    @:param centerIdxs : None或np.array，cluster center entry index， 对于
    k-means等聚类算法，能够算出来center entry，则直接输入；对于dbscan等算法，算法本身不能
    够算出来center entry，则输入None，算法中会找到一个cluster中和mean entry最近的entry作为
    center entry。
    '''

    def __init__(self, data, dataClusterIds, centerIdxs = None):
        self.data = data
        self.dataClusterIds = dataClusterIds

        if centerIdxs is not None :
            self.centerIdxs = centerIdxs
        else :
            self.__getCenterIdxs()

        #self.centerIdxs = centerIdxs

        self.clusterNum = len(self.centerIdxs)

        self.stdev = self.__getStdev()

        self.clusterDensity = []

        for i in range(self.clusterNum):
            self.clusterDensity.append(self.__density(self.data[self.centerIdxs[i]],i))

    #计算center index
    def __getCenterIdxs(self) :

        self.centerIdxs = []

        clusterDataMp = {}
        clusterDataIdxsMp = {}

        for i in range(len(self.data)) :
            entry = self.data[i]
            clusterId = self.dataClusterIds[i]
            clusterDataMp.setdefault(clusterId, []).append(entry)
            clusterDataIdxsMp.setdefault(clusterId, []).append(i)

        for clusterId in sorted(clusterDataMp.keys()) :
            subData = clusterDataMp[clusterId]
            subDataIdxs = clusterDataIdxsMp[clusterId]

            m = len(subData[0])
            n = len(subData)

            meanEntry = [0.0] * m

            for entry in subData :
                meanEntry += entry

            meanEntry = meanEntry / n

            minDist = float("inf")

            centerIdx = 0

            for i in range(len(subData)) :
                entry = subData[i]
                idx = subDataIdxs[i]
                dist = self.__dist(entry, meanEntry)
                if minDist > dist:
                    centerIdx = idx
                    minDist = dist

            self.centerIdxs.append(centerIdx)

    def __getStdev(self) :
        stdev = 0.0

        for i in range(self.clusterNum) :
            varMatrix = np.var(self.data[self.dataClusterIds == i], axis=0)
            stdev += math.sqrt(np.dot(varMatrix.T, varMatrix))

        stdev = math.sqrt(stdev) / self.clusterNum

        return stdev

    def __density(self, center, clusterIdx):

        density = 0

        clusterData = self.data[self.dataClusterIds == clusterIdx]
        for i in clusterData :
            if self.__dist(i, center) <= self.stdev:
                density += 1

        return density

    def __Dens_bw(self):
        print(self.clusterDensity)

        variance = 0

        for i in range(self.clusterNum):
            for j in range(self.clusterNum):
                if i == j:
                    continue
                center = self.data[self.centerIdxs[i]] + self.data[self.centerIdxs[j]] / 2
                interDensity = self.__density(center, i) + self.__density(center, j)
                #variance += interDensity / max(self.clusterDensity[i], self.clusterDensity[j])
                if max(self.clusterDensity[i], self.clusterDensity[j]) == 0:
                    print('what ??')
                variance += interDensity / ( max(self.clusterDensity[i], self.clusterDensity[j]) + 1e-9 )

        return variance / (self.clusterNum * (self.clusterNum - 1) + 1e-9)

    def __Scater(self):
        thetaC = np.var(self.data, axis=0)
        thetaCNorm = math.sqrt(np.dot(thetaC.T, thetaC))

        thetaSumNorm = 0

        for i in range(self.clusterNum):
            clusterData = self.data[self.dataClusterIds == i]
            theta_ = np.var(clusterData, axis=0)
            thetaNorm_ = math.sqrt(np.dot(theta_.T, theta_))
            thetaSumNorm += thetaNorm_

        return (1 / self.clusterNum) * (thetaSumNorm / thetaCNorm)

    #计算data entry的欧拉距离
    def __dist(self, entry1, entry2):
        return np.linalg.norm(entry1 - entry2)

    def result(self):
        return self.__Dens_bw() + self.__Scater()

# 默认是返回值越大认为评价越好,正和我们的意思
def accuracy_score(clusterer, X):
    from sklearn import metrics
    import numpy as np
    clusters_num = max(clusterer.labels_)+1
    # 聚类数(至少平均每个簇有三个吧？就是数量除以3)
    if clusters_num > (len(X) / 3.0):
        print('J</3',end='')
        return -8888
    # 所占比例
    percent_threshold = 0.5
    cluster_elements = [(clusterer.labels_==i).sum() for i in range(-1,clusters_num)]
    cluster_elements_percent = np.array(cluster_elements)/len(X)
    for item in cluster_elements_percent:
        if item >= percent_threshold:
            print('J>0.5',end='')
            return -7777

    if clusters_num <= 2:
        print('J<=2',end='')
        return -9999
    else:
        print('\n',clusterer,'\n')
        print('=>',clusters_num,'|',len(X),end='\t')
        s_score = metrics.silhouette_score(X,clusterer.labels_,metric='euclidean',sample_size=len(X))
        print('\n=',s_score)
        return s_score
        #s_dbw = S_Dbw(X,clusterer.labels_).result()
        #print('\n=',s_dbw)
        #return s_dbw


if __name__ == '__main__':
    data = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 2], [2, 2, 2], [2,2,3]])
    data_cluster = np.array([1, 0, 1, 2, 2])
    centers_index = np.array([1, 0, 3])

    #a = S_Dbw(data, data_cluster, centers_index)
    a = S_Dbw(data, data_cluster)
    print(a.result())







