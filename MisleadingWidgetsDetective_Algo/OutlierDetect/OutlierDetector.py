import sys, os
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor
class OutlierDetector(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(OutlierDetector, self).__init__()
        sys.path.append(os.path.abspath('..'))
        self.config = config
        self.DBP = DBP      
        
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
    Desc:
        进行异常检测
    transform csv records to dicts

    feed meanful record
    write dicts file
    show dicts length

    methods list:
        def runTransform(self, operator='', params=[]):
            if operator == 'info':
            if operator == 'runo':
    Example:
        O runo -spm
        O evalo 簇号, 每个簇的实例显示数量, 1显示分数0不显示分数
        '''
        print(info)

    def transcription(self):
        import numpy as np
        idUcluster = self.DBP.queryAllCluster()
        clusters = np.array(idUcluster)[:, 1] # 取半条链作为mRNA 
        ids = np.array(idUcluster)[:, 0] # 取半条链作为mRNA 
        clusters = [int(x) for x in clusters]
        return ids, clusters #最后返回的是list型

    def outiler(self, data, method='iforest', showScore=False):
        if method == 'iforest':
            from sklearn.ensemble import IsolationForest
            picker = IsolationForest(n_estimators=100, n_jobs=-1)#, contamination=0.1)
        elif method=='knn':
            from sklearn.neighbors import NearestNeighbors        
            picker = NearestNeighbors(2,  0.4)
        
        picker.fit(data)
        pred = picker.predict(data, showScore)
        return list(pred)

    #runOutlier:运行outlier算法
    def runOutlier(self, ids, clusters, extractedFetures, method='iforest', showScore=False):
        import numpy as np
        clusters = np.array(clusters)
        ids = np.array(ids)
        cluster_num = max(clusters)+1
        outlierVectorId = []
        outlierVector = []
        for i in range(-1, cluster_num):
            # 首先取出clusters列表中==i的簇所有成员
            ithClusterIndex = np.where(clusters == i)
            # 该簇对应位置的id，便于后面的提交
            thisclusterId = ids[ithClusterIndex]
            # 如果这个簇没有，一般可能是kmeans没有-1，则不再继续
            if len(thisclusterId)==0:continue
            # 某个簇的特征矩阵，需要对其实行outlier算法
            thiscluster = extractedFetures[ithClusterIndex]
            # 全部记录下来，记录在两个结构中
            outlierVectorId.append(thisclusterId)
            if i == -1:
                outlierVector.append([0]*len(thisclusterId))
            else:
                outlierVector.append(self.outiler(thiscluster, method, showScore))
        return np.array(outlierVectorId), np.array(outlierVector)

    def runOutlierInOutlier(self, outlier_vector,method='iforest',showScore=False):
        import numpy as np
        # for every cluster in outlier_vector
        outlier_vector_return = []
        for cluster in outlier_vector:
            print('cluster ready to outlier:',cluster,len(cluster))
            if len(cluster) == 1:
                outlier_vector_return.append([np.nan])
            else:
                cluster = np.array([cluster]).transpose()
                result = self.outiler(cluster, method, showScore)
                outlier_vector_return.append(result)

        return np.array(outlier_vector_return)
            
    #evlaOutlier:评价outlier算法
    def evalOutlier(self, params):
        # 读取上述数据，包括SX，outlierVector，可以根据聚类号查看某一簇内的信息
        import sys, os
        sys.path.append(os.path.abspath('..'))
        from VectorClustering import ClusteringCore
        from importlib import reload
        reload(ClusteringCore)
        #pictureLists = self.getPictureListsFromTree(self.config['API_VECTORS_NAME_PATH'], self.readDict(self.config['APK_PATH_TREE_PATH']), self.config['PICTURES_DIR'])
        # 获取图片数据
        #SX, y, size = ClusteringCore.loadPictureData(pictureLists)
        # 获取outlier标签
        # 获取簇信息
        #clusters = self.readVector(self.config['CLUSTER_RESULT'], dtype='int32')
        # 先获取命令行参数信息
        showCluster = showNum = ''
        paramList = params.split(', ')
        if len(paramList) == 1 and paramList[0] == '':                      
            pass                                                                             
        else:                                                                                
            if paramList[0] != '':                                                           
                showCluster = int(paramList[0])                                              
            if paramList[1] != '':                                                           
                showNum = int(paramList[1])                                                  
        showScore = True        

        # 再获取读取的container信息
        params = self.getRunParams()
        clusterSuffix = self.DBP.subtask_name
        clusterer_container = self.loadObject(self.config['CLUSTERER_CONTAINER_PATH'])
        outlierVector = self.DBP.queryAllOutlier()
        #outlierVector = self.readCSV(self.config['OUTILER_VECTOR_PATH'], dtype='float')
        #ClusteringCore.randomClustersData(clusters, max(clusters)+1, SX, size, showCluster=params[0], showNum=params[1], label=outlierVector[params[0]], showScore=params[2])
        self.makeSureExists(self.config['OUTLIER_PICTURE_RESULT_DIR'])
        print('writing images...')
        ClusteringCore.showClusterImages(clusterer_container, showCluster, showNum, outlierVector, showScore, self.config['OUTLIER_PICTURE_RESULT_DIR'])

    def addClusterInfo(self, outlierVectorId, outlierVector):
        idUoutlier = []
        for i in range(len(outlierVector)):
            for j in range(len(outlierVector[i])):
                idUoutlier.append( [outlierVectorId[i][j], {"$set":{"outlier_score":str(outlierVector[i][j])}} ])
        self.DBP.updateAPKForest(idUoutlier)

    def MutlieMetic(self,params):
        # 取出图像特征，计算每个图标的异常分数
        # 取出API特征，计算每个API特征的异常分数
        # 根据计算公式排序，要求后者小，前者大,孤立森林的分数均在0-1之间，可以直接用倒数
        import numpy as np

        # 根据id顺序来标的
        ids, clusters = self.transcription()
        print('ids and cluster:',ids,'\n\n',clusters) # 3209 list 3209 list
        '''
        [ObjectId('6073b123f36d3615ab14ed7e') ObjectId('6073b123f36d3615ab14ed80') ObjectId('6073b123f36d3615ab14ed82') ... ObjectId('6073b12bf36d3615ab151bc6') ObjectId('6073b12bf36d3615ab151bc8') ObjectId('6073b12bf36d3615ab151bca')]
        [0, 154, 155, 18, -1, -1, -1, -1, -1, -1, -1, 347, -1, -1, -1, -1, 8, 2, 347, -1, -1 ... ]
        '''

        # 图像特征异常检测
        extracted_features = self.readVector(self.config['EXTRACTED_FEATURE_PATH'], dtype='')
        image_outlier_vector_id, image_outlier_vector = self.runOutlier(ids, clusters, extracted_features, 'iforest', True)
        print('image_outlier_vector_id.shape, image_outlier_vector.shape',image_outlier_vector_id.shape, image_outlier_vector.shape)
        print(image_outlier_vector_id[:1][:10], image_outlier_vector[:1][:10])
        '''
        image_outlier_vector_id.shape, image_outlier_vector.shape (425,) (425,)
        [array([ObjectId('6073b123f36d3615ab14ed86'),ObjectId('6073b123f36d3615ab14ed88'),ObjectId('6073b123f36d3615ab14ed8a'), ...,ObjectId('6073b12bf36d3615ab151bc6'),ObjectId('6073b12bf36d3615ab151bc8'), ObjectId('6073b12bf36d3615ab151bca')], dtype=object)] 
        [list([0, 0 ,0 ,0])]
        '''
        
        # API特征异常检测
        api_features = self.readVector(self.config['API_VECTORS_PATH'])
        api_outlier_vector_id, api_outlier_vector = self.runOutlier(ids, clusters, api_features , 'iforest', True)
        print('api_outlier_vector_id.shape, api_outlier_vector.shape',api_outlier_vector_id.shape, api_outlier_vector.shape)
        print(api_outlier_vector_id[:1][:10], api_outlier_vector[:1][:10])
        '''
        api_outlier_vector_id.shape, api_outlier_vector.shape (425,) (425,)
        [array([ObjectId('6073b123f36d3615ab14ed86'),ObjectId('6073b123f36d3615ab14ed88'),ObjectId('6073b123f36d3615ab14ed8a'), ...,ObjectId('6073b12bf36d3615ab151bc6'),ObjectId('6073b12bf36d3615ab151bc8')ObjectId('6073b12bf36d3615ab151bca')], dtype=object)] 
        [list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0			, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
        '''
        
        mm_outlier_vector = list() #np.zeros((image_outlier_vector.shape[0],len(image_outlier_vector[0])))
        for i in range(api_outlier_vector_id.shape[0]):
            print(len(image_outlier_vector[i]) ,len( api_outlier_vector[i])) # 3489
            print(image_outlier_vector[i], api_outlier_vector[i]) # [0,0,0] [0,0,0] 
            if params == 'mm':
                # 注意，这里取了负数，以便后面在web页面好排序
                temp_array = np.array(api_outlier_vector[i]) - np.array(image_outlier_vector[i])  
            elif params == 'mmdao':
                # 注意，这里取了负数，以便后面在web页面好排序
                temp_array = 1/np.array(image_outlier_vector[i]) + np.array(api_outlier_vector[i])
            print('i=',i,'temp_array.shape:',temp_array.shape)
            mm_outlier_vector.append(temp_array)
        mm_outlier_vector = np.array(mm_outlier_vector)
        #self.addClusterInfo(outlier_vector_id, mm_outlier_vector)    
        print('mm_outlier_vector.shape',mm_outlier_vector.shape)
        self.addClusterInfo(image_outlier_vector_id, mm_outlier_vector)    

    #runCluster:运行最佳参数的聚类
    def runDetector(self, operator=[], params=[]):
        if operator == 'info' or operator == '':
            self.info()
        if operator == 'myod':
            #两种实现方式，第一种聚类，将两个聚类定义为联合距离，被标为-1的认定是噪声
            # 第二种孤立森林，要不把api和图像数据一起？
            # 占：找到聚类中心，然后按聚类中心，越是靠近聚类中心且其异常分数越大的，就越有可能是
            pass
        if operator == 'run':
            extractedFeatures = self.readVector(self.config['EXTRACTED_FEATURE_PATH'], dtype='')
            # 根据id顺序来标的
            ids, clusters = self.transcription()
            # 运行outlier
            outlierVectorId, outlierVector = self.runOutlier(ids, clusters, extractedFeatures, 'iforest', True)
            self.addClusterInfo(outlierVectorId, outlierVector)    

        # Multimodality OD，注意，该方法只在真正的聚类VC情况下使用，sim方法不可使用
        if operator == 'mm':
            self.MutlieMetic('mm')
        if operator == 'mmdao':
            self.MutlieMetic('mmdao')

        if operator == 'oio':
            extracted_features = self.readVector(self.config['EXTRACTED_FEATURE_PATH'], dtype='')
            # 根据id顺序来标的
            ids, clusters = self.transcription()
            # 运行outlier
            outlier_vector_id, outlier_vector = self.runOutlier(ids, clusters, extracted_features, 'iforest', True)
            print('extracted_features:',extracted_features.shape) # N,1000
            print('outlier_vector:',outlier_vector.shape) # 
            outlier_vector = self.runOutlierInOutlier(outlier_vector, 'iforest', True)
            print('oio_outlier_vector.shape',outlier_vector.shape)
            self.addClusterInfo(outlier_vector_id,outlier_vector)    

        if operator == 'cluster':
            extracted_features = self.readVector(self.config['EXTRACTED_FEATURE_PATH'], dtype='')
            # 根据id顺序来标的
            ids, clusters = self.transcription()
            # 运行outlier
            outlier_vector_id, outlier_vector = self.runOutlier(ids, clusters, extracted_features, 'iforest', True)
            print('extracted_features:',extracted_features.shape) # N,1000
            print('outlier_vector:',outlier_vector.shape) # 
            outlier_vector = self.runOutlierInOutlier(outlier_vector, 'iforest', True)
            print('oio_outlier_vector.shape',outlier_vector.shape)

            self.addClusterInfo(outlier_vector_id,outlier_vector)    
