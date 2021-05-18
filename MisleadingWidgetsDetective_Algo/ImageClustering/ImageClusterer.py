import sys, os
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor 
from bson import ObjectId
from PIL import Image
import numpy as np
import imagehash
import csv

class ImageClusterer(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(ImageClusterer, self).__init__()
        sys.path.append(os.path.abspath('..'))
        self.config = config
        self.DBP = DBP
        self.save_path = self.config['RAW_ROOT']+self.DBP.DBNAME+'/infos/'

    def checkDBP(self):
        DBP_info = self.DBP.project_name+'; '
        DBP_info +=self.DBP.subtask_name+'; '
        DBP_info +=self.DBP.configTableName+'; '
        DBP_info +=self.DBP.metaDataTableName+'; '
        DBP_info +=self.DBP.apkTreeTableName +'; '
        DBP_info +=self.DBP.rawApkForestName+'; '
        print(DBP_info)
    
    def info(self):
        info='''
    Example:
        if operator == 'prepare':
            self.preparePrecluster()                
        if operator  ==  'search':
            runLog = self.searchCluster(params, runLog)
        if operator  ==  'run':
            runLog = self.runCluster(params, runLog)
        if operator  ==  'auto':
            # A auto @hierarch(1,58,1,1,1)
        IC prepare preparerecylcetext
        IC prepare rico
        '''
        print(info)

    def getSearchParams(self):
        import os, sys
        sys.path.append(os.path.abspath(self.config['CLUSTER_COMMAND_DIR']+'image/'))
        import clusterParams 
        from importlib import reload
        reload(clusterParams)
        return clusterParams.getsearchParams()

    def getRunParams(self):
        import os, sys
        sys.path.append(os.path.abspath(self.config['CLUSTER_COMMAND_DIR']+'image/'))
        import clusterParams 
        from importlib import reload
        reload(clusterParams)
        return clusterParams.getRunParams()

    def transcription(self):
        import numpy as np
        idUpath = self.DBP.queryAllPath()
        path = np.array(idUpath)[:,1] # 取半条链作为mRNA
        path = [self.config['INPUT_DATA_DIR'] + x for x in path]
        return path #最后返回的是list型


    def ClusterSearchManager(self,X,method='hierarchy',params=[]):
        from sklearn.metrics import make_scorer
        from . import s_Dbw
        from importlib import reload
        reload(s_Dbw)
        from sklearn.cluster import KMeans
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.model_selection import GridSearchCV#RandomizedSearchCV
        from scipy.cluster import hierarchy
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')

        if method == 'kmeans':
            clusterer = KMeans(random_state=9)
            searchParam = params
        elif method == 'hierarchy':
            # 层次聚类的预先查看
            Z = hierarchy.linkage(X,method='ward',metric='euclidean')
            dendrogram = hierarchy.dendrogram(Z)

            plt.savefig(self.save_path+"dendrogram.png")
            nexttodo = input('what next to do?')

                #clusterer = DBSCAN(metric='wminkowski',p=2,metric_params={"w":idf})
            if nexttodo == 'b':
                return runLog
            elif nexttodo == 'c':
                #label = cluster.hierarchy.cut_tree(Z, n_clusters=2, height=None) 
                labels = hierarchy.cut_tree(Z,height=30) 
                print('聚类簇数:',max(labels)+1)
            else:
                clusterer = AgglomerativeClustering(affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='average', distance_threshold=None)
                searchParam = params
            return runLog
            # 层次聚类的预先查看
        else:
            pass

        # 全部的图片共有:11380 / 3 = 3700
        param_dist = {
                'n_clusters': searchParam#range(1500,3000,50),
        }
        ''' 用于随机搜索
        param_dist = {                              param_dist = {
                'eps': np.linspace(0.01,100,100),           'eps': np.linspace(0.01,100,10),
                'min_samples': range(1,1000,100)            'min_samples': np.linspace(0.01,100,100),#range(1,100,5)
                }                                           }
        '''
        print('搜索空间:1500,3000,50 正在搜索...')
        '''
        self, estimator, param_grid, scoring=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score=False
        '''
        clusterer_grid = GridSearchCV(clusterer,param_dist,cv=None,scoring=s_Dbw.accuracy_score,n_jobs = 1)
        clusterer_grid.fit(X)
        print('搜索完成，正在进行选择')
        best_clusterer = clusterer_grid.best_estimator_
        print('clusterer:',best_clusterer, clusterer_grid.best_params_)
        print('clusters:',max(best_clusterer.labels_)+1,'\nscore:',clusterer_grid.best_score_)
        return best_clusterer,clusterer_grid.best_params_
           

    # 聚类
    def searchCluster_old(self, params, runLog):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        X = self.readVector(self.save_path+'precluster.vec', dtype='')
        SX = y = size = np.array([])
        best_clusterer = self.ClusterSearchManager(X,'hierarchy',range(50,100,10))
        '''
        params = self.getSearchParams()
        params['data'] = (X, SX, y, size)

        resultList =  ClusteringCore.searchParams(clusterer_name = params['clusterer_name'], 
                ranges = params['ranges'], 
                data = params['data'], 
                param_scale = params['param_scale'], 
                metricstring = params['metricstring'], 
                filterFun = self.filterFun, 
                verbose = params['verbose'])
        #print('聚类完成后List:\n')
        [print(self.simplifyClusteringContainer(result), '\n-------------------------------') for result in resultList]
        '''
    

    def searchCluster(self, params, runLog, topn='top1'):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        # 填入参数,如果没有在调用时给定则从文件读取
        X = self.readVector(self.save_path+'precluster.vec', dtype='')
        SX = y=size = np.array([])
        if params == '':
            params = self.getSearchParams()
        params['data'] = (X, SX, y, size)

        resultList =  ClusteringCore.searchParams(clusterer_name = params['clusterer_name'],
                ranges = params['ranges'],
                data = params['data'],
                param_scale = params['param_scale'],
                metricstring = params['metricstring'],
                filterFun = self.filterFun,
                verbose = params['verbose'])
        #print('聚类完成后List:\n')
        #[print(self.simplifyClusteringContainer(result), '\n-------------------------------') for result in resultList]
        simplified_result_list = []
        for result in resultList:
            simplified_result = self.simplifyClusteringContainer(result)
            simplified_result_list.append(simplified_result)
            print(simplified_result,'\n-------------------------------')

        simplified_result_list = sorted(simplified_result_list,key=lambda x:x['performance']['silhouette'],reverse=True)
        # 为了显示时方便
        params.pop('data')
        runLog.append('X.shape %s,  SX.shape %s,  y.shape %s,  size %s'% (X.shape, SX.shape, y.shape, size))
        runLog.append('params %s' % params)
        runLog.append('resultList len %d' % len(resultList))

        # 开始选择列表
        return_simplified_result_list = simplified_result_list[:3]
        return_simplified_result_list = sorted(return_simplified_result_list,key=lambda x:x['info']['params'][0])
        return return_simplified_result_list

    def writeIntoInfos(self,labels):
        self.writeVector(labels, self.save_path+'cluster.txt', fmt="%d")

        # 预聚类需将信息存入.csv文件 [app,path,cluster_no]
        #f = open(self.config['EXTRACTED_FEATURE_PATH']+'-info.csv','r')
        f = open(self.save_path+'info.csv','r')
        lines = csv.reader(f)
        new_lines = []
        i = 0
        for row in lines:
        	new_lines.append(row + [labels[i]])
        	i += 1
        f.close()
        
        #f = open(self.config['EXTRACTED_FEATURE_PATH']+'-info.csv','w')
        f = open(self.save_path+'info.csv','w')
        f_csv = csv.writer(f)
        f_csv.writerows(new_lines)
        f.close()
    
    def runCluster(self,params):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        X = self.readVector(self.save_path+'precluster.vec', dtype='')
        SX = y = size = np.array([])
        params['data'] = (X,SX,y,size)

        clusterer_container = ClusteringCore.runClusterer(params['clusterer_name'],
                params['params'],
                params['data'],
                params['param_scale'],
                params['metricstring'])
        clusterer_container = ClusteringCore.clustererEvaluationMetric(clusterer_container)
        labels = clusterer_container['info']['clusterer'].labels_
        self.writeIntoInfos(labels)
        print('All Done.')
    
    # 这个run已经和普通的VC run不同了，这里面是只针对预聚类的
    def runCluster_old(self, params, runLog):
        from . import ClusteringCore
        from importlib import reload
        reload(ClusteringCore)
        import matplotlib.pyplot as plt
        from scipy.cluster import hierarchy
        plt.switch_backend('agg')

        X = self.readVector(self.save_path+'precluster.vec', dtype='')

        #--------------- 层次聚类-------------------
        # 0.5,10 注意！！ eps 被缩小一个尺度!!!
        # 设定聚类器
        #clusterer = DBSCAN()#eps=params[0], min_samples=params[1])

        # 层次聚类的预先查看
        Z = hierarchy.linkage(X,method='ward',metric='euclidean')
        dendrogram = hierarchy.dendrogram(Z)

        plt.savefig(self.save_path+"dendrogram.png")
        nexttodo = input('what next to do?')

        if nexttodo == 'b':
            return runLog
        elif nexttodo == 'c':
            #label = cluster.hierarchy.cut_tree(Z, n_clusters=2, height=None) 
            labels = hierarchy.cut_tree(Z,height=30) 
            print('聚类簇数:',max(labels)+1)
        else:
            pass
        #if want open this ',below shuld ->
        '''
        #fclusterdata(X,t=0.99,criterion='inconsistent',metric='euclidean',method='average',R=None)
        params = self.getRunParams()
        SX = y = size = np.array([])
        params['data'] = (X, SX, y, size)
        #print(params)
        clusterer_container = ClusteringCore.runClusterer(params['clusterer_name'], 
                params['params'], 
                params['data'], 
                params['param_scale'], 
                params['metricstring'])
        clusterer_container = ClusteringCore.clustererEvaluationMetric(clusterer_container)
        labels = clusterer_container['info']['clusterer'].labels_
        '''
        self.writeVector(labels, self.save_path+'cluster.txt', fmt="%d")

        # 预聚类需将信息存入.csv文件 [app,path,cluster_no]
        #f = open(self.config['EXTRACTED_FEATURE_PATH']+'-info.csv','r')
        f = open(self.save_path+'info.csv','r')
        lines = csv.reader(f)
        new_lines = []
        i = 0
        for row in lines:
        	new_lines.append(row + [labels[i]])
        	i += 1
        f.close()
        
        #f = open(self.config['EXTRACTED_FEATURE_PATH']+'-info.csv','w')
        f = open(self.save_path+'info.csv','w')
        f_csv = csv.writer(f)
        f_csv.writerows(new_lines)
        f.close()
        
        # 为了显示时方便
        #runLog.append('X.shape %s,  SX.shape %s,  y.shape %s,  size %s'% (X.shape, SX.shape, y.shape, size))
        #runLog.append('cluster_result:%s'% ClusteringCore.simplifyClusteringContainer(clusterer_container))
        return runLog


            
    #evalCluster:评价聚类效果，用图形展示参数和示例图片
    def evalClustering(self, params, runLog):
        # 不用加载簇的标签，直接调用图就行
        from . import ClusteringCore
        from importlib import reload
        reload(ClusteringCore)
        # 获取数据进行评测
        showCluster = showNum = ''
        paramList = params.split(', ')
        # 如果参数只有一个且为空，pass
        if len(paramList) == 1 and paramList[0] == '': 
            pass
        else:
            if paramList[0] != '':
                showCluster = int(paramList[0])
            if paramList[1] != '':
                showNum = int(paramList[1])
        showScore = True

        # 读取container
        params = self.getRunParams()
        #paramsStr = '-'+params['clusterer_name'] +'-'+  ", ".join([str(i) for i in params['params']])
        clusterer_container = self.loadObject(self.config['CLUSTERER_CONTAINER_PATH'])#+clusterSuffix)

        # 保存图片
        self.makeSureExists(self.config['CLUSTER_PICTURE_RESULT_DIR'])
        print('writing images...')
        ClusteringCore.showClusterImages(clusterer_container, showCluster, showNum, [], showScore, self.config['CLUSTER_PICTURE_RESULT_DIR'])
        ClusteringCore.showClusterHistogram(clusterer_container, savePath=self.config['CLUSTER_PICTURE_RESULT_DIR'])#+clusterSuffix)
        return runLog

    def analysisParams(self, paramString):
        if paramString  ==  '':
            return {}
        params = dict()
        paramPairs = paramString.split(';')
        for item in paramPairs:
            item = item.split(":")
            params[item[0]] = item[1]
        return params

    def simplifyClusteringContainer(self, clusterer_container):
        import numpy as np
        simplifiedClusteringContainer = {}
        infoDict = dataDict = performanceDict = {}
        infoDict['clusterer_name'] = clusterer_container['info']['clusterer_name']
        infoDict['params']=clusterer_container['info']['params']
        #infoDict['inertia_'] = clusterer_container['info']['clusterer'].inertia_
        #print('原始即:', infoDict['inertia_'] )
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

    def filterFun(self, clusterer_container):
        percent_threshold = 0.5
        cluster_num = [0, 9999]
        silhouette_threshold = 0.01
        calinski_threshold = 0
        # 首先不能有一项超过0.5, 即一半，那个肯定有问题
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
    def visualizeClustering(self, params ,runLog):
        # coordinates 二维的坐标
        # images 图像，且缩放
        vector = self.readImage(self.config['API_VECTORS_PATH'].split('-')[0]+'-visualize.txt')
        cluster_data = self.DBP.queryAllCluster()

        pictureList = self.transcription()
        # 图像原始值
        SX,size = self.loadPictureData(pictureList,method='PIL',resize=(8,8))

        coordinates = vector
        print('coordinates shape:',coordinates.shape)
        images = SX
        print('images shape:',images.shape)
        self.plot_embedding_scatter(coordinates,images,figsize=(1,1),frameon=False,title=None,xticks=[],yticks=[],min_dist=4e-6)
        return runLog

    
    def getImageMixHash(self,image):
        shape_hash = imagehash.average_hash(image)
        color_hash = imagehash.colorhash(image)
        return shape_hash,color_hash

    def getDiff(self,image_a,image_b):
        import numpy as np
        image_a_hash = self.getImageMixHash(image_a)
        image_b_hash = self.getImageMixHash(image_b)
        x = np.array(image_a_hash[0]-image_b_hash[0])
        y = np.array(image_a_hash[1]-image_b_hash[1])
        print("HASH: %s , %s =norm=> %s =New=> %s" % (x,y,np.linalg.norm(x-y),(x<=COMPARE_THRESHOLD and y<=COMPARE_THRESHOLD)))
        return x,y


    # ------------------------------------------------------
    def extractImageFeaturebyVGG(self,image_list):
        from ImageFeatureExtract.cnn import testVGGFeatureExtractor
        from importlib import reload
        reload(testVGGFeatureExtractor)
        return testVGGFeatureExtractor.runExtract(image_list)

    def extractImageFeaturebySPM(self,image_list):
        from ImageFeatureExtract.spm import spm
        from importlib import reload
        reload(spm)
        featureList = []
        for picture_path in image_list:
            picture_feature = Image.open(picture_path).convert('L').convert('RGB')
            featureList.append(np.array(picture_feature))
        return spm.getSPM(np.array(featureList),self.config['SPM_CODE_BOOK_PATH'])

    def extractImageFeaturebyHOG(self,pictureList):
        size = (224,224,3)
        # S3 获取图片的方向梯度直方图
        def getHOG(picture):
            from skimage import feature as ft
            from skimage import io
            features,hogImage = ft.hog(picture, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
            #io.imshow(hogImage)
            #io.show()
            return features

        featureList = []
        for picture_path in pictureList:
            picture_feature = Image.open(picture_path).convert('L').convert('RGB')
            featureList.append(getHOG(picture_feature))

        return featureList
        
    # ------------------------------------------------------
    def loadPictureFeature(self,dataSource,labelSource='',prepocess='N'):
        # 在这里，需要从文件列表中将图像特征提取出来，然后返回
        print('S1 loadPictureData>>>')
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from sklearn import preprocessing
        from PIL import Image
        import cv2
 
        #------------------------------ s1读取Data
        # 如果是图像列表则等会直接遍历
        if type(dataSource) is str:
            # 遍历dataSource 载入图片
            print('S1 dataSource:',dataSource)
            files = ["%s/%s" % (dataSource, x) for x in os.listdir(dataSource)]
        else:
            files = dataSource
        '''
        imgs = []
        img = ''
        # 这里要取特征，参考IF
        for imgfile in files:
            img = Image.open(self.config['INPUT_DATA_DIR']+imgfile).convert('RGB')
            img = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2GRAY)
            from skimage import feature as ft
            from skimage import io
            size = img.size
            features,hogImage = ft.hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)

            #img = np.array(img)
            #img = plt.imread(imgfile).convert('RGB')        
            #imgs.append(img.flatten())
           imgs.append(features)
        '''
        absolute_path_Source = []
        for img in dataSource:
            #icons/27360_icons/icon_6.jpg'
            absolute_path_Source.append(self.config['INPUT_DATA_DIR']+img)
        imgs = self.extractImageFeaturebyVGG(absolute_path_Source)
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
        #size = img.shape
        #print('S1 示例图片:')    
        #plt.imshow(X[0].reshape(size))
        #plt.show()
        print('S1 done.<<<')
        return np.array(X)#,size

    def preparePrecluster(self,params):
        from PIL import Image
        # 遍历输入目录，这个目录是json和图片混合存储的
        IMAGE_SUFFIX = '.png'
        IMAGE_SUFFIX1 = '.jpg'
        DROP_SUFFIX = '.abandon'
        cluster_list = []

        for root, dirs, files in os.walk(self.config['INPUT_DATA_DIR']):
            for name in files:
                # 得到具体文件了，判断其后缀
                if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                    print('---------------------------------------------')
                    absolute_path = root+os.path.sep+name
                    try:
                        image = Image.open(absolute_path, 'r').convert('RGB')
                    except:
                        print('Read', absolute_path ,' Error! Dropping it...')
                        #self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'broken/')
                        print('Bad Read remove: ', absolute_path, ' w*h: ', image.size)
                        continue

                    if 'preparerecylcetext' in params:
                        cluster_list.append(['images','images/'+name])
                    elif 'rico' in params:
                        # /data/wangruifeng/datasets/DroidBot_Epoch/raw_data/rico_dataset/input_data/icons/1111_icons/1.png
                        # ['icons', '1111_icons', '1.png'] -3
                        cluster_list.append(['icons','/'.join(absolute_path.split('/')[-3:])])
                    else:
                        # 将可以访问的图片收集起来以便聚类
                        relative_path = '/'.join(absolute_path.split('/')[-4:])
                        app = relative_path.split('/')[0]
                        cluster_list.append([app,relative_path])
                    

        # 获得图片特征，然后就可以进行聚类了
        import numpy as np
        X = self.loadPictureFeature(np.array(cluster_list)[:,1])
        #self.writeVector(X,self.config['EXTRACTED_FEATURE_PATH']+'.precluster')
        self.writeVector(X,self.save_path+'precluster.vec')
        import csv
        #f = open(self.config['EXTRACTED_FEATURE_PATH']+'-info.csv','w')
        f = open(self.save_path+'info.csv','w')
        f_csv = csv.writer(f)
        f_csv.writerows(cluster_list)
        f.close()

    def searchVideoCluster(self, params):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        resultList =  ClusteringCore.searchParams(clusterer_name = params['clusterer_name'],
                ranges = params['ranges'],
                data = params['data'],
                param_scale = params['param_scale'],
                metricstring = params['metricstring'],
                filterFun = self.filterFun,
                verbose = params['verbose'])
    
        simplified_result_list = []
        for result in resultList:
            simplified_result = self.simplifyClusteringContainer(result)
            simplified_result_list.append(simplified_result)
            print(simplified_result,'\n-------------------------------')

        simplified_result_list = sorted(simplified_result_list,key=lambda x:x['performance']['silhouette'],reverse=True)
        # 为了显示时方便
        params.pop('data')
        runLog.append('X.shape %s,  SX.shape %s,  y.shape %s,  size %s'% (X.shape, SX.shape, y.shape, size))
        runLog.append('params %s' % params)
        runLog.append('resultList len %d' % len(resultList))

        # 开始选择列表
        return_simplified_result_list = simplified_result_list[:3]
        return_simplified_result_list = sorted(return_simplified_result_list,key=lambda x:x['info']['params'][0])
        return return_simplified_result_list

    def runVideoCluster(self,params):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        clusterer_container = ClusteringCore.runClusterer(params['clusterer_name'],
                params['params'],
                params['data'],
                params['param_scale'],
                params['metricstring'])
        clusterer_container = ClusteringCore.clustererEvaluationMetric(clusterer_container)
        labels = clusterer_container['info']['clusterer'].labels_
        return labels


    def videoClustering(self,feature_path):
        X = self.readVector(feature_path)#+suffix)#, dtype='int32')
        video_nums = X.shape[0]
        SX = y = size = np.array([])
        searchparam = {
            "clusterer_name":"hierarch",
            "ranges":[[int(video_nums/5),video_nums,500]],
            "data":(X, SX, y, size),
            "param_scale":[1],
            "metricstring":"",
            "filterFun":"",
            "verbose":""
        }
        # 第一次搜索
        print('第一次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
        top_list = self.searchVideoCluster(searchparam)

        # 第二次搜索
        searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,100]]
        print('第二次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
        top_list = self.searchVideoCluster(searchparam)

        # 第三次搜索
        searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,50]]
        print('第三次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
        top = self.searchVideoCluster(searchparam)

        # 正式执行
        runparam = {
            "clusterer_name":"hierarch",
            "params":[int(top[0]['info']['params'][0])],
            "data":(X, SX, y, size),
            "param_scale":[1],
            "metricstring":""
        }
        print('正式执行,参数为:',runparam['params'])
        labels = self.runVideoCluster(runparam)

    def videoCluster(self,video_info_path,feature_path):#,video_dir
        '''给予一个视频文件列表csv(存储着视频的完整path信息)和一个对应相同顺序的特征列表
        最后将在给予的视频文件csv中填写每个视频的簇信息，可以在web系统中展示
        '''
        '''
        # 首先读取相应的video文件夹，存入info.csv
        VIDEO_SUFFIX = '.mp4'
        video_list = []
        for video_name in os.listdir(video_dir):
            video_path = video_dir+video_name
            if (VIDEO_SUFFIX in video_path) and (DROP_SUFFIX not in video_path):
                video_list.append(video_path)

        f = open(video_dir+'info.csv','w')
        f_csv = csv.writer(f)
        f_csv.writerows(video_list)
        f.close()
        '''

        # 进行聚类
        labels = self.videoClustering(feature_path)

        # 读取视频信息csv
        f = open(video_info_path,'r') #(video_dir+'info.csv','r')
        lines = csv.reader(f)
        new_lines = []
        i = 0
        for row in lines:
        	new_lines.append(row + [labels[i]])
        	i += 1
        f.close()
        
        # 往视频信息csv中写入簇信息
        f = open(video_info_path,'w') #(video_dir+'info.csv','w')
        f_csv = csv.writer(f)
        f_csv.writerows(new_lines)
        f.close()

    def runClustering(self, operator=[], params=[]):
        # 首先需要指定项目中的图片地址，接着用cluster_core读取出来，再参考ImageExt提取出的东西
        runLog = []
        runLog.append(operator)
        print('目标项目:',self.config['CLUSTER_COMMAND_DIR']+'image/'+'clusterParams.py')
        if not self.isExist(self.config['CLUSTER_COMMAND_DIR']+'image/'+'clusterParams.py'):
            self.makeSureExists(self.config['CLUSTER_COMMAND_DIR']+'image/')
            self.copyFile(self.config['SCRIPT_ROOT']+'VectorClustering/config/clusterParams.py', self.config['CLUSTER_COMMAND_DIR']+'image/'+'clusterParams.py')
        if operator == 'info' or operator == '':
            self.info()
        if operator  ==  'hash':
            image_hash_list = []
            # 遍历输入目录，这个目录是json和图片混合存储的
            for root, dirs, files in os.walk(self.config['INPUT_DATA_DIR']):
                for name in files:
                    # 得到具体文件了，判断其后缀
                    if IMAGE_SUFFIX in name:            
                        print('---------------------------------------------')
                        absolute_path = root+os.path.sep+name
                        relative_path = absolute_path.replace(r'/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/','')
                        try:
                            image = Image.open(absolute_path, 'r')
                        except:
                            print('Read', absolute_path ,' Error! Dropping it...')
                            continue
                        # 开始处理,# 首先对每个图像使用image hash提取出特征
                        image_hash_value = self.getImageMixHash(absolute_path)
                        image_hash_list_temp = [relative_path,image_hash_value]
                        image_hash_list.append(image_hash_list_temp)
        if operator == 'prepare':
            self.preparePrecluster(params)                
        if operator  ==  'search':
            runLog = self.searchCluster(params, runLog)
        if operator == 'videocluster':
            self.videoCluster()
        if operator  ==  'run':
            # 防止重复运行
            '''
            cluster_data = self.DBP.queryAllCluster()
            if len(cluster_data) > 0:
                print('/\/\cluster data exists, deleteing it?')                                               
                return runLog
                curApkTree = self.DBP.getApkForestDB()                                                  
                curApkTree.drop()
            '''
            # 正式执行
            runparam = {
                "clusterer_name":method,
                "params":[120],
                "data":"",
                "param_scale":[1],
                "metricstring":""
            }
            print('正式执行,参数为:',runparam['params'])
            runLog = self.runCluster(runparam)

        if operator == 'auto':
            ##### Initial Params #####
            method = 'hierarch'

            X = self.readVector(self.save_path+'precluster.vec', dtype='')
            widget_nums = X.shape[0]

            start = int(widget_nums/5)
            end = widget_nums
            step1 = 500
            step2 = 100
            step3 = 50
            multiplier = 0.1
            ##### Given Params #####
            # A auto @optics(1,7,1,1,1)
            if params != '':
                if '(' in params:
                    params,searchrange = params.rstrip(')').split('(')
                    start,end,step1,step2,step3 = searchrange.split(',')
                    start = int(start)
                    end = int(end)
                    step1 = int(step1)
                    step2 = int(step2)
                    step3 = int(step3)

                # 首先解析聚类参数 
                if '@'  in params:
                    params,method = params.split('@')

            ###### Build the search Params #######

            searchparam = {
                "clusterer_name":method,
                "ranges":[[start,end,step1]],
                "data":"",
                "param_scale":[1],
                "metricstring":"",
                "filterFun":"",
                "verbose":""
            }
            # 第一次搜索
            print('第一次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top_list = self.searchCluster(searchparam, runLog,  'top3')

            # 第二次搜索
            searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,step2]]
            print('第二次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top_list = self.searchCluster(searchparam, runLog,  'top3')

            # 第三次搜索
            searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,step3]]
            print('第三次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top = self.searchCluster(searchparam, runLog,  'top1')

            # 正式执行
            runparam = {
                "clusterer_name":method,
                "params":[int(top[0]['info']['params'][0])],
                "data":"",
                "param_scale":[1],
                "metricstring":""
            }
            print('正式执行,参数为:',runparam['params'])
            runLog = self.runCluster(runparam)


        print('\n---------------------------runLog:----------------------------\n', runLog)
        return runLog
