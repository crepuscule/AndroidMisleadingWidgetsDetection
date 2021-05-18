# -*- coding: UTF-8 -*-
# S0 使用向导-------------------------------------------------------------
def guide(no=''):
    print('S0 guide>>>')
    functions = '''
    s1 loadPictureData(dataSource,labelSource='',prepocess='N')
    s2 runClusterer(clusterer_name,params,data,param_scale='',metricstring=''):
    s3 clustererEvaluationMetric(clusterer_container) 需加入有标签评价系统
    s4 searchParams(clusterer_name,ranges,data,param_scale,metricstring,filterFun,verbose='v')
    s5 searchManager(configuration,result='',verbose='v')
    s6 showClusterImages(clusterer_container,showCluster,showNum)

    f1 unPackClustererContainer(clusterer_container)
    f2 simplifyClusteringContainer(clusterer_container):
    f3 filterFun(clusterer_container)
    f4 def randomClustersData(clusterer,clusters_num,SX,size,showCluster='',showNum='')
    f5 getClusterData(clusterer_container,cluster_no)
    '''
    print(functions)
    s1 = '''
    e.g.: loadPictureData('/data/picutres/')
    '''
    s2 = '''
    e.g.: runClusterer('dbscan',(5,10,),(X,SX,y,224),(10,1,))
    o ok; t test; - not support;
    [o]kmeans 质心k
    [o]dbscan 密度
    [o]birch 层次k
    [t]optics 密度
    [t]mean-shift 密度
    [t]hierarch 层次k
    [-]spectral 谱k
    [-]GMM 高斯混合模型k
    Affinity 仿射传播聚类 适合高维、多类数据快速聚类
    '''

    s4 = '''
    e.g.: searchParams('dbscan',([5,100,10],[10,100,5]),(X,y,size),(10,1),filterFun)
    '''
    if no == 's2':
        print(s2)
    print('S0 done.<<<')


# S1 获取数据-------------------------------------------------------------
# 从地址或者图像列表读出数据
def loadPictureData(dataSource,labelSource='',prepocess='N'):
    print('S1 loadPictureData>>>')
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn import preprocessing
    from PIL import Image

    #------------------------------ s1读取Data
    # 如果是图像列表则等会直接遍历
    if type(dataSource) is list:
        files = dataSource
    else:
        # 遍历dataSource 载入图片
        print('S1 dataSource:',dataSource)
        files = ["%s/%s" % (dataSource, x) for x in os.listdir(dataSource)]
    imgs = []
    img = ''
    for imgfile in files:
        img = Image.open(imgfile).convert('RGB')
        img = np.array(img)
        #img = plt.imread(imgfile).convert('RGB')        
        imgs.append(img.flatten())
        #图片的高H为200，宽W为200，颜色通道C为3
        #print(img.shape,img.dtype)        
        #plt.imshow(img)
        #plt.show()
    X = np.array(imgs)
    print('S1 SX.shape',X.shape)
    if prepocess == 'Y':
        X=preprocessing.scale(X)

    #------------------------------ s2读取Label
    # 如果labelSource为空，说明没有label
    if type(labelSource) is list:
        y = labelSource
    # 否则读取字符串打开文件，暂时未实现
    else:
        if labelSource == '':
            #y = [-1 for _ in range(len(X))]
            y = np.zeros(len(X))
        else:
            pass
    
    #-------------------------s3 读取size和展示 
    size = img.shape
    #print('S1 示例图片:')    
    #plt.imshow(X[0].reshape(size))
    #plt.show()
    print('S1 done.<<<')
    return np.array(X),np.array(y),size

def loadPictureDataCV2(dataSource,labelSource='',prepocess='N'):
    print('S1 loadPictureData>>>')
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from sklearn import preprocessing
    from PIL import Image
    import cv2

    #------------------------------ s1读取Data
    # 如果是图像列表则等会直接遍历
    if type(dataSource) is list:
        files = dataSource
    else:
        # 遍历dataSource 载入图片
        print('S1 dataSource:',dataSource)
        files = ["%s/%s" % (dataSource, x) for x in os.listdir(dataSource)]
    imgs = []
    img = ''
    for imgfile in files:
        img = Image.open(imgfile).convert('RGB')
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2GRAY)  
        #img = np.array(img)
        #img = plt.imread(imgfile).convert('RGB')        
        #imgs.append(img.flatten())
        imgs.append(img)
    X = np.array(imgs)
    print('S1 SX.shape',X.shape)
    if prepocess == 'Y':
        X=preprocessing.scale(X)

    #------------------------------ s2读取Label
    # 如果labelSource为空，说明没有label
    if type(labelSource) is list:
        y = labelSource
    # 否则读取字符串打开文件，暂时未实现
    else:
        if labelSource == '':
            #y = [-1 for _ in range(len(X))]
            y = np.zeros(len(X))
        else:
            pass
    
    #-------------------------s3 读取size和展示 
    size = img.shape
    #print('S1 示例图片:')    
    #plt.imshow(X[0].reshape(size))
    #plt.show()
    print('S1 done.<<<')
    return np.array(X),np.array(y),size

# S2 运行聚类------------------------------------------------------------------------------------------------
# 聚类器名称str，聚类器参数list(str或num)，数据list(可字符串，可向量元祖）
def runClusterer(clusterer_name,params,data,param_scale='',metricstring=''):
    print('S2 runClusterer>>>')
    from time import time

    #----------------------------------s1 读取数据
    #如果data[0]存储的是字符串，则读出data[0],data[1]，即训练数据和标签位置
    if isinstance(data[0],str):
        X,y,size = loadPictureData(data[0],data[1],data[2])
        SX = X
    #如果存储的不是字符串，那就是直接能用的向量，直接存储就行,各分量自动存储
    else:
        X,SX,y,size = data
    print('S2 data load done')
    #----------------------------------s2 参数缩放 
    # params: (5,10,) param_scale: (1,100,)
    # ture params : (5,0.1,)
    # 建议meanshift ,dbsacan eps /10 
    if param_scale != '':
        params = list(params)
        for i in range(0,len(params)):
            params[i] /= param_scale[i]

    #s2 选择聚类器
    #kmeans 需指定k
    if clusterer_name == 'kmeans':
        from sklearn.cluster import KMeans
        clusterer = KMeans(init='k-means++', n_clusters=int(params[0]), n_init=10)
        ms = 'sc'
    elif clusterer_name == 'dbscan':        
        from sklearn.cluster import DBSCAN
        # 0.5,10 注意！！ eps 被缩小一个尺度!!!
        clusterer = DBSCAN(eps=params[0], min_samples=params[1])
        ms = 'sc'
    #birch 需指定k
    elif clusterer_name == 'birch':
        # None,0.5,50
        from sklearn.cluster import Birch
        clusterer = Birch(n_clusters = params[0], threshold = params[1], branching_factor = params[2])
        ms = 'sc'
    #optics 
    elif clusterer_name == 'optics':
        from sklearn.cluster import OPTICS
        clusterer = OPTICS(min_samples=int(params[0]))#,xi=params[1],min_cluster_size=params[2])
        #OPTICS(min_samples = 10, xi = 0.05, min_cluster_size = 0.05)
        ms = 'sc'
    #Spectral 需指定k
    elif clusterer_name == 'spectral':
        pass
        #clusterer = SpectralClustering(n_clusters = params[0], assign_labels = params[1], random_state = params[2])
    elif clusterer_name == 'hierarch':
        from sklearn.cluster import AgglomerativeClustering
        #clusterer = AgglomerativeClustering(n_clusters=params[0],affinity=params[1],linkage=params[2])#'canberra',linkage='complete')
        clusterer = AgglomerativeClustering(n_clusters=int(params[0]), affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='average')#, distance_threshold=None)
        ms = 'sc'
    elif clusterer_name == 'meanshift':
        from sklearn.cluster import MeanShift,estimate_bandwidth
        #0.2,500
        bandwidth = estimate_bandwidth(X, quantile=params[0], n_samples=params[1])
        clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True) 
        ms = 'sc'
    else:
        print('no cluster name specify')
        import sys
        sys.exit(0)

    if metricstring == '':
        metricstring = ms
    #s3 正式运行聚类
    t0 = time()
    clusterer.fit(X)
    t1 = time()
    
    infoDict = {'clusterer':clusterer,'clusterer_name':clusterer_name,'params':params,'metricstring':metricstring}
    # 聚类器，聚类器生成字符串，度量列表字符串
    dataDict = {'X':X,'SX':SX,'y':y,'size':size}
    # 存储数据的字典，三样全
    performanceDict = {'time':t1-t0,'clusters_num':max(clusterer.labels_)+1}
    # 存储表现的字典，先存储时间和聚类数量
    clusterer_container = {'info':infoDict ,'data':dataDict,'performance':performanceDict}    
    print('S4 done.<<<')
    return clusterer_container
# S2 -----------------------------------------------------------------------------------------------

# F1
def unPackClustererContainer(clusterer_container):
    clusterer = clusterer_container['info']['clusterer']
    X = clusterer_container['data']['X']
    SX = clusterer_container['data']['SX']
    y = clusterer_container['data']['y']
    size = clusterer_container['data']['size']
    clusters_num = clusterer_container['performance']['clusters_num']
    metricstring = clusterer_container['info']['metricstring']
    return clusterer,X,SX,y,size,clusters_num,metricstring

# S3 指标评估------------------------------------------------------------------------------------------------
# 评估器，用于评测算法结果
# 给定聚类器，给定聚类数量，给定数据，[给定标签]
def clustererEvaluationMetric(clusterer_container): 
    print('S3 clustererEvaluationMetric>>')
    import numpy as np
    from sklearn import metrics
    clusterer,X,SX,y,size,clusters_num,metricstring = unPackClustererContainer(clusterer_container)
    # 各集群百分比
    #performanceDict = {'time':t1-t0,'clusters_num':max(clusterer.labels_)+1,'cluster_elements_percent':[0.4,0.2,0.1],'silhouette':0.23,'calinski':342.9}
    cluster_elements = [(clusterer.labels_==i).sum() for i in range(-1,clusters_num)]
    cluster_elements_percent = np.array(cluster_elements)/len(X)
    clusterer_container['performance']['cluster_elements_percent'] = cluster_elements_percent
    # 数学指标
    choicenmetrics = {}
    if 'r' in metricstring:
        m = metrics.adjusted_rand_score(y, clusterer.labels_)
        clusterer_container['performance']['a-randscore'] = m
    if 'm' in metricstring:
        m = metrics.adjusted_mutual_info_score(y,  clusterer.labels_)
        clusterer_container['performance']['a-mutualinfo'] = m
    if 's' in metricstring:
        sample_size = len(X)
        # 聚类数量小于1是没有该指数的
        m = -9999 if clusters_num <= 1 else metrics.silhouette_score(X, clusterer.labels_,metric='euclidean',sample_size=sample_size)
        clusterer_container['performance']['silhouette'] = m
    if 'c' in metricstring:
        # 聚类数量小于1是没有该指数的
        #m = -9999 if clusters_num <= 1 else metrics.calinski_harabasz_score(X, clusterer.labels_)
        m = -9999 if clusters_num <= 1 else metrics.calinski_harabaz_score(X, clusterer.labels_)
        clusterer_container['performance']['calinski'] = m
    print('S3 done.<<<')
    return clusterer_container
# S3 ------------------------------------------------------------------------------------------------

# F2
def simplifyClusteringContainer(clusterer_container):
    import numpy as np
    simplifiedClusteringContainer = {}
    infoDict = dataDict = performanceDict = {}
    infoDict['clusterer_name'] = clusterer_container['info']['clusterer_name']
    #infoDict['inertia_'] = clusterer_container['info']['clusterer'].inertia_
    #print('原始即:',infoDict['inertia_'])
    infoDict['params']=clusterer_container['info']['params']
    infoDict['metricstring']=clusterer_container['info']['metricstring']

    dataDict['XSize'] = np.array(clusterer_container['data']['X']).shape
    dataDict['Xdtype'] = str(np.array(clusterer_container['data']['X']).dtype)
    #dataDict['SXSize'] = np.array(clusterer_container['data']['SX']).shape
    #dataDict['SXdtype'] = np.array(clusterer_container['data']['SX']).dtype
    dataDict['ySize'] = np.array(clusterer_container['data']['y']).shape
    dataDict['ydtype'] = str(np.array(clusterer_container['data']['y']).dtype)
    dataDict['size'] = clusterer_container['data']['size']
    import copy
    performanceDict = copy.deepcopy(clusterer_container['performance'])
    performanceDict['cluster_elements_percent'] =  performanceDict['cluster_elements_percent'][:10]
    simplifiedClusteringContainer['info']=infoDict
    simplifiedClusteringContainer['data']=dataDict
    simplifiedClusteringContainer['performance']=performanceDict
    return simplifiedClusteringContainer

# S4 参数搜索------------------------------------------------------------------------------------
def searchParams(clusterer_name,ranges,data,param_scale,metricstring,filterFun,verbose=''):
    print('S4 searchParams>>>')
    import numpy as np
    #或者使用列表推导式def cartesian_product2(a,b):return [[x,y]for x in a for y in b]
    performanceList = []
    clusterer_container_list = []
    productRanges = []
    for rangeList in ranges:
        productRanges.append(np.arange(rangeList[0],rangeList[1],rangeList[2]))

    def myProduct(args):
        pools = [tuple(pool) for pool in args]
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

    for param in myProduct(productRanges):
        #print(clusterer_name,'\n',param,'\n',data,'\n',param_scale,'\n',metricstring)
        clusterer_container = runClusterer(clusterer_name,param,data,param_scale,metricstring)
        clusterer_container = clustererEvaluationMetric(clusterer_container)
        if clusterer_container['performance']['clusters_num'] >= 2:
            clusterer_container_list.append(clusterer_container)
        if verbose == 'v': print('S4 聚类信息和表现',print(simplifyClusteringContainer(clusterer_container)))
    
    results = filter(filterFun,clusterer_container_list)
    resultsList = list(results)
    resultsList = sorted(resultsList,key=lambda keys:keys['performance']['silhouette'])#,reverse=True)
    print('S4 done.<<<')
    return resultsList
# S4 -----------------------------------------------------------------------------------

# F3 filter
def filterFun(clusterer_container):
    percent_threshold = 0.5
    cluster_num = [5,50]
    silhouette_threshold = 0.01
    calinski_threshold = 0
    # 首先不能有一项超过0.5,即一半，那个肯定有问题
    for item in clusterer_container['performance']['cluster_elements_percent']:
        if item >= percent_threshold:
            return False
    # 再者轮廓系数要大于0
    if clusterer_container['performance']['silhouette'] <= silhouette_threshold:
        return False
    if clusterer_container['performance']['calinski'] <= calinski_threshold:
        return False
    # 聚类的数量需要有限制
    if  clusterer_container['performance']['clusters_num'] > cluster_num[1] and clusterer_container['performance']['clusters_num'] < cluster_num[0]:
            return False
    return True

# S5 搜索管理-----------------------------------------------------------------------
def searchManager(configuration,result='',verbose='v'):
    print('S5 searchManager>>')
    # 直接字符串
    if configuration[0] == 'T':
        print('S5 reading config from Text...')
        configList=configuration[1:].split('\n')
    else:
        print('S5 reading config from File...')
        with open(configuration,'r') as config:
            configList = config.readlines()
            configList = [item[:-1] for item in configList]
        config.close()
    # 0行数据集位置
    # 1行标签位置
    # 2行size
    # 3行聚类算法名
    # 4行参数范围1
    # 5行参数范围2
    if verbose=='v': print(configList)

    ranges = []
    
    # 这里的data应该直接先取出
    X,y,size = loadPictureData(configList[0],configList[1])
    size = int(configList[2])
    data=(X,y,size)

    for paramItem in configList[4:]:
        if verbose=='v': print('S5 paramItem',paramItem)
        paramRange = paramItem.split(',')
        ranges.append([int(paramRange[0]),int(paramRange[1]),int(paramRange[2])])

    if verbose=='v': print(configList[3],ranges,data)
    resultsList = searchParams(configList[3],ranges,data,filterFun)
    
    import copy
    resultsListForWritten = copy.deepcopy(resultsList)
    for i in range(0,len(resultsListForWritten)):
         resultsListForWritten[i].pop('data')
    
    import json
    if result == '':
        print('S5 resultsList:\n',resultsListForWritten)
    else:
        f = open (result,'w')
        print(resultsListForWritten,file=f)
        f.close()
        print('S5 writing resultsList to ',result,'...')
    print('S5 done.<<<')
    return resultsList
# S5 -----------------------------------------------------------------------

# S6 真实聚类图像查看-------------------------------------------------------------------------------
# F4 对于outlier展示，传入整个聚类标签数组；传入聚类号，传入该号对应的label标识+-1
def showClusterImages(clusterer_container,showCluster='',showNum='',rawLabel=[],showScore=False,savePath='.'):
#def randomClustersData(clusters,clusters_num,SX,size,showCluster='',showNum='',rawLabel=[],showScore=False,savePath='.'):    
    print('S6 showClusterImages')
    import numpy as np
    from random import sample
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    clusterer,X,SX,y,size,clusters_num,metricstring = unPackClustererContainer(clusterer_container)
    clusters = clusterer.labels_
    clustersNumList = [i for i in range(0,clusters_num)]
    # 没有值显示所有簇
    if showCluster == '':
        choicen = clustersNumList
    # 小于0的值，随机显示多少簇
    elif showCluster < 0:
        showCluster = - showCluster
        choicen =  sample(clustersNumList, clusters_num if clusters_num<showCluster else showCluster)
        choicen.sort()
    # 剩余就是>=0的情况了，就是表示显示哪个簇
    else:
        choicen =  [showCluster]
        
    #print(clustersNumList, choicen)        
    for cluster in choicen:            
        plt.figure(figsize=(20, 20),facecolor='gray')
        alldata = SX[np.where(clusters == cluster)]
        if showNum == '':
            showNum = len(alldata )            
        showNumCopy = showNum
        rows = 0 
        cols = 10 
        while showNumCopy > 0:
            showNumCopy -= 10
            rows +=1
        start = 0
        #print('显示:',showNum,'\t',rows)
        choicendata = alldata[start:start+rows*cols]

        if rawLabel != []:
            label = rawLabel[cluster]
        else:
            label = []
        for i in range(-1,len(choicendata)):
            sub1 = plt.subplot(rows, cols, i+1)
            sub1.imshow(choicendata[i].reshape(size))
            autoAxis = sub1.axis()
            labelInfo = 'C'+str(cluster)        
            if label == []:
                plt.title(labelInfo)
                continue
            if showScore == True:
                labelInfo += str(label[i])[:5]
                if label[i] < -0.55:
                    labelInfo += ' O!'              
                    rec = patches.Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=4,color='red')
                    rec = sub1.add_patch(rec)
                    rec.set_clip_on(False)
            else:
                if label[i] == -1:
                    labelInfo += ' OUT!'                
            plt.title(labelInfo) #+str(choicendatalabel[i]))
        #plt.margins(0,0)
        #plt.axis('off')
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.savefig(savePath+'C'+str(cluster)+'.jpg',dpi=300,bbox_inches="tight", pad_inches=0.0)
        plt.clf()
    #plt.show()
    print('S6 done.')
        
def showClusterHistogram(clusterer_container,savePath='.'):
    import numpy as np
    import matplotlib.pyplot as plt
    clusterer,X,SX,y,size,clusters_num,metricstring = unPackClustererContainer(clusterer_container)
    plt.figure(figsize=(20,5))
    cluster_elements = [(clusterer.labels_==i).sum() for i in range(-1,clusters_num)]        
    plt.bar(range(-1,clusters_num), cluster_elements, width=1)   
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.savefig(savePath+'Histogram.jpg',dpi=300,bbox_inches="tight", pad_inches=0.0)
    plt.clf()

# 查看每个聚类的平均图像
def showClusterMean(clusterer_container,savePath='.'):
    import numpy as np
    import matplotlib.pyplot as plt
    clusterer,X,SX,y,size,clusters_num,metricstring = unPackClustererContainer(clusterer_container)
    average_clusters = []
    for i in range(clusters_num):
        average_clusters.append(SX[clusterer.labels_==i].mean(axis=0))
    average_clusters= np.asarray(average_clusters)
    print('S6 平均图像形状:\t',average_clusters.shape)
    plt.figure(figsize=(20,20))
    for i in range(clusters_num):
        plt.subplot(10, 10, i+1)
        #plt.imshow(average_clusters[i].reshape((8,8)))
        plt.imshow(average_clusters[i].reshape(size))
        plt.title('C%d'%(i))
        plt.axis('off')
    cluster_elements = [(clusterer.labels_==i).sum() for i in range(-1,clusters_num)]        
    plt.bar(range(-1,clusters_num), cluster_elements, width=1)   
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.savefig(savePath+'Mean.jpg',dpi=300,bbox_inches="tight", pad_inches=0.0)
    plt.clf()

# F5
def getClusterData(clusterer_container,cluster_no,show=False,showNum=''):
    import numpy as np
    clusterer,X,SX,y,size,clusters_num,metricstring = unPackClustererContainer(clusterer_container)
    clusterDataSX = SX[np.where(clusterer.labels_ == cluster_no)]
    clusterDataX = X[np.where(clusterer.labels_ == cluster_no)]
    
    if show == True:
        print('show')
        showClusterImages(clusterer,clusters_num,SX,size,cluster_no,showNum)
    return clusterDataX,clusterDataSX,size
        

def storeClustersVector(clusterer_container,storePath):
    # 取出info中的_labels
    # 是按传入的X的顺序标注的
    import numpy as np
    np.savetxt(storePath,clusterer_container['info']['clusterer'].labels_,fmt="%d")
    print(len(clusterer_container['info']['clusterer'].labels_),' records writen.')

# S7 存储聚类结果(标签和和数据存入JSON，图片聚类结果存在文件夹中)
def storeClustering(clusterer_container,storePath,dataInfoDict):
    # 首先存储一下本次聚类的基本外部信息
    storeDict = {}
    # 当前时间，和使用数据地址（需要手动指定，因为不好记录）
    import time
    storeDict['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # {'XPath':'/storage/staticResourceImages/beforeclustering/benign_pa_vectors.txt',
    # 'yPath':'','SXPath':'/storage/staticResourceImages/beforeclustering/benign_pa_vectors.txt'}
    storeDict.update(dataInfoDict)
    # 接着存储聚类的信息
    storeDict.update(simplifyClusteringContainer(clusterer_container))
    
    #import json
    #storeJSON = json.dumps(storeDict)
    f=open(storePath+'basicinfos.json','w',encoding='utf-8')
    print(storeDict,file=f)    
    f.close()
    
    # 接着将图片聚类结果存入文件系统中
    #聚类数量 
    import os
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    clusterer,X,SX,y,size,clusters_num,metricstring = unPackClustererContainer(clusterer_container)
    print('cd %s && mkdir Z && mkdir -p {%03d..%03d}'%(storePath,0,clusters_num))
    os.system('cd %s && mkdir Z && mkdir -p {%03d..%03d}'%(storePath,0,clusters_num))
    clustersNumList = [i for i in range(-1,clusters_num)]
    for cluster in clustersNumList:
        allData = SX[np.where(clusterer.labels_ == cluster)]
        if len(allData) == 0:
            continue
        for i in range(0,len(allData)):
            #print(allData[i])
            #print(allData[i].resize((224,224,3)))
            #print(allData[i].shape)
            plt.imshow(allData[i].reshape(size))
            #curImage = Image.fromarray(np.uint8(allData[i]))
            if cluster == -1:
                saveName = '%s/Z/%d.png' % (storePath ,i)
            else:
                saveName = '%s/%03d/%d.png' % (storePath ,cluster ,i)
            plt.savefig(saveName,dpi=500,bbox_inches = 'tight')
