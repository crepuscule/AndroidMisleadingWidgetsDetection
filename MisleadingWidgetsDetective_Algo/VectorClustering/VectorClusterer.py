import sys, os
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor 
from bson import ObjectId
import numpy as np

class VectorClusterer(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(VectorClusterer, self).__init__()
        sys.path.append(os.path.abspath('..'))
        self.config = config
        self.DBP = DBP      
        self.save_path = self.config['RAW_ROOT']+self.DBP.DBNAME+'/'
        
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
    transform csv records to dicts

    feed meanful record
    write dicts file
    show dicts length

    methods list:
        def setConfig(self, config);
        def feature(self):
        def cutDimension(self, blackListPath, apiSet, widgetAPIVectors):
        def vector(self, apkTree):
        def saveList2Txt(self, content, path):
        def saveAPIVector(self, apiSet, saveAPIColumnPath, widgetAPIVectors, savePath, widgetNames, saveAPINamePath):
        def readVector(self, savePath):
        def writeDict(self, content, path):
        def readDict(self, path): def runTransform(self, operator='', params=[]):
            if operator  ==  'info':
            if operator  ==  'feature':
            if operator  ==  'vector':
            if operator  ==  'search':
            if operator  ==  'run':
            if operator  ==  'eval':
            if operator == 'agg':
            if operator == 'circ':
                self.circulateCluster()
            if operator == 'od':
                self.outlierDetectingController()
    Example:
        VC search 500,7000,1000 # start,end,step
        VC runc -kpca-300
        # 与翻译后的pro无关，因为pro通过调控重新映射到了DNA中
        VC evalc 54, 10 
        VC evalc , 10
    Problem:
        1. getPictureListsFromTree in 59 modified !Note!
        2. change write, read to Base Standard
        3. cutDimension wait for rewrite
        4. update ClusteringCore
        '''
        print(info)

    def transcription(self): 
        import numpy as np 
        idUpath = self.DBP.queryAllPath() 
        path = np.array(idUpath)[:, 1] # 取半条链作为mRNA 
        path = [self.config['INPUT_DATA_DIR'] + x for x in path] 
        return path #最后返回的是list型 

    def transcription_ids(self):
        import numpy as np
        idUcluster = self.DBP.queryAllCluster()
        clusters = np.array(idUcluster)[:, 1] # 取半条链作为mRNA
        ids = np.array(idUcluster)[:, 0] # 取半条链作为mRNA
        clusters = [int(x) for x in clusters]
        return ids, clusters #最后返回的是list型
 

    def api_transcription(self):
        import numpy as np
        idUapi = self.DBP.queryAllApis()
        package_apis = np.array(idUapi)[:,1] # 取半条链作为mRNA 
        class_apis = np.array(idUapi)[:,2] # 取半条链作为mRNA 
        method_apis = np.array(idUapi)[:,3] # 取半条链作为mRNA 
        return list(package_apis),list(class_apis),list(method_apis)

    def getSearchParams(self):
        import os, sys
        sys.path.append(os.path.abspath(self.config['CLUSTER_COMMAND_DIR']))
        import clusterParams 
        from importlib import reload
        reload(clusterParams)
        return clusterParams.getsearchParams()

    def getRunParams(self):
        import os, sys
        sys.path.append(os.path.abspath(self.config['CLUSTER_COMMAND_DIR']))
        import clusterParams 
        from importlib import reload
        reload(clusterParams)
        return clusterParams.getRunParams()

    def saveAPIVector(self, apiSet, saveAPIColumnPath, widgetAPIVectors, savePath, widgetNames, saveAPINamePath):
        self.makeSureExists(saveAPIColumnPath)
        self.makeSureExists(saveAPINamePath)
        self.makeSureExists(savePath)
        import numpy
        self.writeTxt(apiSet, saveAPIColumnPath)
        numpy.savetxt(savePath, widgetAPIVectors, fmt="%d") 
        self.writeTxt(widgetNames, saveAPINamePath)

    def searchCluster(self, params, runLog, ratio=[0,0,1,0], topn='top1', outlierdection=False):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        if outlierdection == False:
            #------------------准备数据-----------------#
            #pictureLists = self.getPictureListsFromTree(self.readDict(self.config['APK_TREE_PATH']), self.config['INPUT_DATA_DIR'])
            X = self.readVector(self.config['API_VECTORS_PATH'])#+suffix)#, dtype='int32')
            pictureList = self.transcription()
            # 图像原始值
            #SX, y, size = ClusteringCore.loadPictureData(pictureList)
            SX = y = size = np.array([])
            #best_clusterer = self.ClusterSearchManager(X,'kmeans',range(50,100,10))

        try:
            result = self.DBP.queryMetaData(queryField={'encodermethod':1})
            print('result:',result)
            mode = result[0]['encodermethod'] 
        except:
            mode = 'apimethod-Features'
        if mode == 'apimethod-Features':
            print('in apimethod-Features mode.')
        elif mode == 'multi-Features':
            print('in multi-Features mode.')
            #------------------X的重构加权-----------------#
            print('X.shape:',np.array(X).shape) #=>(1010, 500)
            print('SX.shape:',np.array(SX).shape)
            X = np.array(X)
            result = self.DBP.queryMetaData(queryField={'package_api_len':1,'class_api_len':1,'method_api_len':1,'doc_api_len':1})
            print('api_len_distribute:',result)
            pl = int(result[0]['package_api_len'])
            cl = int(result[0]['class_api_len'])
            ml = int(result[0]['method_api_len'])
            dl = int(result[0]['doc_api_len'])
            #  90     683       3392   100
            #[    ;       ;           ;   ]
            #0    pl     pl+cl      -dl   
            #print('before:',X[0])
            X[:,:pl] = X[:,:pl] * ratio[0]
            X[:,pl:pl+cl] = X[:,pl:pl+cl] * ratio[1]
            X[:,pl+cl:-dl] = X[:,pl+cl:-dl] * ratio[2]
            X[:,-dl:] = X[:,-dl:] * ratio[3]

        # 填入参数,如果没有在调用时给定则从文件读取
        print('print params before ["data"]:',params)
        if params == '':
            params = self.getSearchParams()
        elif ('data' not in params) or (params['data'] == ''):
            params['data'] = (X, SX, y, size)

        
        '''
        if ratio == 0:
            # 0.7 0.1 0.1 0.1 pure_big_ResearchRatio.hog_complimit5fuse7111_raw_hierarch_iforest
            X[:,:pl] = X[:,:pl] * 0.1
            X[:,pl:pl+cl] = X[:,pl:pl+cl] * 0.1
            X[:,pl+cl:-dl] = X[:,pl+cl:-dl] * 0.1
            X[:,-dl:] = X[:,-dl:] * 0.7
        elif ratio == 1:
            # pure_big_ResearchRatio.hog_complimit5fuse1333_raw_hierarch_iforest
            X[:,:pl] = X[:,:pl] * 0.3
            X[:,pl:pl+cl] = X[:,pl:pl+cl] * 0.3
            X[:,pl+cl:-dl] = X[:,pl+cl:-dl] * 0.3
            X[:,-dl:] = X[:,-dl:] * 0.1
        else:
            X[:,:pl] = X[:,:pl] * 0.2
            X[:,pl:pl+cl] = X[:,pl:pl+cl] * 0.3
            X[:,pl+cl:-dl] = X[:,pl+cl:-dl] * 0.1
            X[:,-dl:] = X[:,-dl:] * 0.4
        #print('after:',X[0])
        '''
        '''
        data = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 2], [2, 2, 2], [2,2,3]])
        data_cluster = np.array([1, 0, 1, 2, 2]) 
        centers_index = np.array([1, 0, 3]) 
       
        #a = S_Dbw(data, data_cluster, centers_index)
        a = S_Dbw(data, data_cluster)
        print(a.result())
        '''

        resultList =  ClusteringCore.searchParams(clusterer_name = params['clusterer_name'], 
                ranges = params['ranges'], 
                data = params['data'], 
                param_scale = params['param_scale'], 
                metricstring = params['metricstring'], 
                filterFun = self.OutlierDetectionfilterFun if outlierdection == True else self.filterFun, 
                verbose = params['verbose'])
        #print('聚类完成后List:\n')
        #[print(self.simplifyClusteringContainer(result), '\n-------------------------------') for result in resultList]
        simplified_result_list = []
        for result in resultList:
            simplified_result = self.simplifyClusteringContainer(result)
            simplified_result_list.append(simplified_result)
            #print(simplified_result,'\n-------------------------------')

        simplified_result_list = sorted(simplified_result_list,key=lambda x:x['performance']['silhouette'],reverse=True)
        # 为了显示时方便
        params.pop('data')
        if outlierdection == False:
            runLog.append('X.shape %s,  SX.shape %s,  y.shape %s,  size %s'% (X.shape, SX.shape, y.shape, size))
        runLog.append('params %s' % params)
        runLog.append('resultList len %d' % len(resultList))

        # 开始选择列表
        return_simplified_result_list = simplified_result_list[:3]
        print('simplified_result:\n---------------------------',return_simplified_result_list)
        return_simplified_result_list = sorted(return_simplified_result_list,key=lambda x:x['info']['params'][0])
        return return_simplified_result_list 



    def addClusterInfo(self, clusterLabels):
        '''将簇信息加入到subtask库中，从这里建立数据库G'''
        idUpath = self.DBP.queryAllPath() 
        for i in range(len(idUpath)):
            #idUpath[i][1] = {"$set":{"cluster":str(clusterLabels[i])}}
            idUpath[i][1] = str(clusterLabels[i])
        self.DBP.saveAPKForest(idUpath)

    # 目前必须得是mvector才能使用
    def apiFilter(self):
        import numpy as np
        # 根据api的idf来判断api的重要程度,一次性返回需要考虑的api
        _,_,method_apis = self.api_transcription() #获取method api，即完整api
        #api_idf = self.readVector(self.config['API_IDF_PATH'].split('-')[0]+'.txt')
        #api_names = np.array(self.readTxt(self.config['API_VECTORS_PATH']+'-'+'apinames.txt'))
        if self.config['VERSION'] == '2':
            api_idf = self.readVector(self.config['API_VECTORS_PATH'].replace('APIVector-','APIIDF-')+'.txt')
            api_names = np.array(self.readTxt(self.config['API_VECTORS_PATH'].replace('.txt','apinames.txt')))
        else:
            api_idf = self.readVector(self.config['API_IDF_PATH'].split('-')[0]+'.txt')
            api_names = np.array(self.readTxt(self.config['API_VECTORS_PATH']+'-'+'apinames.txt'))
        exceptList = np.array(['android.app.Activity.startActivity',
        'android.app.Activity.finish',
        'android.app.Activity.onOptionsItemSelected',
        'android.net.Uri.parse',
        'android.support.v4.view.ViewPager.setCurrentItem',
        'android.app.Activity.onPrepareOptionsMenu',
        'android.view.View.getHeight'])
        # > 12??
        api_index = np.where((api_idf < 20) & (api_idf > 1))[0]
        print(api_index,'\n',len(api_index))
        # 返回这种api名字集合
        return np.concatenate((api_names[api_index],exceptList),axis=0)

    # Fsim 聚类方法中的api过滤法
    def doFilter(self,apiwhitelist,api):
        if len(api) <= 2:
            return api
        return_api = []
        for item in api:
            if item in apiwhitelist:
                return_api.append(item)
        return return_api


    # 主要关注项目: db_pure_big: pure_big_ImportSimAPI.hog_complimit5Fsim_raw_hierarch_iforest   4964 clusters total
    # 思路2: 对于每两个控件，去除公有的，查看不一样的那些是不是都是低idf的。如果不一样的都是低idf那就不看这些低idf的了
    # 注意：直接抛弃这些api: android.util.Log.v
    def runSimCluster(self,params , runLog):
        # 完全sim
        import hashlib
        import numpy as np  
        isAddFilter = params.strip(' ')
        
        cluster_data = self.DBP.queryAllCluster()
        if len(cluster_data) > 0:
            print('/\/\cluster data exists, deleteing it?')                                               
            curApkTree = self.DBP.getApkForestDB()                                                  
            curApkTree.drop()
        # 首先获取所有API信息
        idUapi = self.DBP.queryAllApi()
        apis = np.array(idUapi)[:,1] # 取半条链作为mRNA 
        # 将每个控件的API信息进行hash
        apihash2widgets_dict = dict()
        
        # 将hash变为dict的key，value中存储序号
        count = 0
        apiwhitelist = self.apiFilter()
        for api in apis:
            # api实际上是一个api列表
            # 首先进行api过滤(需要定义合适的阈值)
            if isAddFilter == 'F':
                api = self.doFilter(apiwhitelist,api)
            # 接着比较过滤后的api
            seed = ('\n'.join(api)).encode("utf8")
            api_hash = hashlib.md5(seed).hexdigest()
            if api_hash in apihash2widgets_dict:
                apihash2widgets_dict[api_hash].append(count)
            else:
                apihash2widgets_dict[api_hash] = [count]
            count += 1
        # 由于序号和id顺序一致，按照和addClusterInfo的相同逻辑即可完成信息提交
        cluster_no = 0
        clusterLabels = [0] * len(apis)

        # 遍历dict中所有元素，然后按照dict中记录的顺序
        for key,value in apihash2widgets_dict.items():
            # 给数据库中的控件标注簇号，如果所在簇数量太少<5，标为-1
            if len(value) < 3:
                for i in value:
                    clusterLabels[i] = -1
            else:
                for i in value:
                    clusterLabels[i]  = cluster_no
            cluster_no += 1
        self.addClusterInfo(clusterLabels)
        self.DBP.updateMetaData({"$set":{'clustermethod':'sim', 'clusters':str(max(clusterLabels)+1), 'silhouette':'NAN', 'calinski':'NAN'}})

    def runCluster(self,params,runLog, ratio=[0,0,1,0],outlierdection=False):
        import numpy as np
        from . import ClusteringCore
        from importlib import reload
        import matplotlib.pyplot as plt
        reload(ClusteringCore)

        #------------------准备数据-----------------#
        #pictureLists = self.getPictureListsFromTree(self.readDict(self.config['APK_TREE_PATH']), self.config['INPUT_DATA_DIR'])
        if outlierdection == False:
            X = self.readVector(self.config['API_VECTORS_PATH'])#+suffix)#, dtype='int32')
            pictureList = self.transcription()
            # 图像原始值
            #SX, y, size = ClusteringCore.loadPictureData(pictureList)
            SX = y = size = np.array([])
            #best_clusterer = self.ClusterSearchManager(X,'kmeans',range(50,100,10))
        try:
            result = self.DBP.queryMetaData(queryField={'encodermethod':1})
            print('result:',result)
            mode = result[0]['encodermethod'] 
        except:
            mode = 'apimethod-Features'
        if mode == 'apimethod-Features':
            print('in apimethod-Features mode.')
        elif mode == 'multi-Features':
            print('in multi-Features mode.')
            #------------------X的重构加权-----------------#
            print('X.shape:',np.array(X).shape) #=>(1010, 500)
            print('SX.shape:',np.array(SX).shape)
            X = np.array(X)
            result = self.DBP.queryMetaData(queryField={'package_api_len':1,'class_api_len':1,'method_api_len':1,'doc_api_len':1})
            print('api_len_distribute:',result)
            pl = int(result[0]['package_api_len'])
            cl = int(result[0]['class_api_len'])
            ml = int(result[0]['method_api_len'])
            dl = int(result[0]['doc_api_len'])
            #  90     683       3392   100
            #[    ;       ;           ;   ]
            #0    pl     pl+cl      -dl   
            #print('before:',X[0])
            X[:,:pl] = X[:,:pl] * ratio[0]
            X[:,pl:pl+cl] = X[:,pl:pl+cl] * ratio[1]
            X[:,pl+cl:-dl] = X[:,pl+cl:-dl] * ratio[2]
            X[:,-dl:] = X[:,-dl:] * ratio[3]

        # 如果data未指明,则填入参数
        if 'data' not in params or params['data'] == '':
            params = self.getRunParams()
            params['data'] = (X, SX, y, size)
        #print(params)
        clusterer_container = ClusteringCore.runClusterer(params['clusterer_name'], 
                params['params'], 
                params['data'], 
                params['param_scale'], 
                params['metricstring'])
        labels = clusterer_container['info']['clusterer'].labels_
        clusterer_container = ClusteringCore.clustererEvaluationMetric(clusterer_container)

        if outlierdection == True:
            return labels
        else:
            self.saveObject(clusterer_container, self.config['CLUSTERER_CONTAINER_PATH'])#+clusterSuffix)
            self.DBP.updateMetaData({"$set":{'clustermethod':self.DBP.subtask_name, 'clusters':str(clusterer_container['performance']['clusters_num']), 'silhouette':clusterer_container['performance']['silhouette'], 'calinski':clusterer_container['performance']['calinski']}})
            #print('clustering done:\n', ClusteringCore.simplifyClusteringContainer(clusterer_container))
            #ClusteringCore.storeClustersVector(clusterer_container, self.config['CLUSTER_RESULT'])
            self.writeVector(labels, self.save_path+'vector-cluster.txt', fmt="%d")
            # 需要更新APKTree，可以根据获取的path列表，毕竟都是一样的
            self.addClusterInfo(labels)
            # 插入metadata信息
            #suffix2 = '-'+self.subtask_name.split('_')[2]
            #paramsStr = suffix
            #paramsStr = '-'+params['clusterer_name'] +'-'+  ", ".join([str(i) for i in params['params']])
            #self.writeDict(newApkTree, self.config['APK_TREE_PATH'])

        # 为了显示时方便
        params.pop('data')
        if outlierdection == False:
            runLog.append('X.shape %s,  SX.shape %s,  y.shape %s,  size %s'% (X.shape, SX.shape, y.shape, size))
        #runLog.append('params %s' % params)
        #runLog.append('cluster_result:%s'% ClusteringCore.simplifyClusteringContainer(clusterer_container))
        return runLog
# -------------------------------------------------------循环聚类------------------------------------------------
    def autoCluster(self,X):
        runLog = []
        ##### Initial Params #####
        method = 'hierarch'
        widget_nums = X.shape[0]
        start = int(widget_nums/5)
        if start <3:start = 3
        end = widget_nums
        step1 = int((end-start)/10)
        if step1 <= 0:step1 = 1
        step2 = int((end-start)/50)
        step3 = int((end-start)/100)

        print('....New Cluster Task....,widget_nums/start/end',widget_nums,start,end,step1,step2,step3)

        SX = y = size = np.array([])
        ###### Build the search Params #######
            
        searchparam = {
            "clusterer_name":method,
            "ranges":[[start,end,step1]],
            "data":(X,SX,y,size),
            "param_scale":[1],
            "metricstring":"",
            "filterFun":"",
            "verbose":""
        }
        if step1 > 0:
            # 第一次搜索
            print('第一次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top_list = self.searchCluster(searchparam, runLog, [], 'top3',True)
        if step1 > 0 and step2 > 0:
            # 第二次搜索
            searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,step2]]
            searchparam['data'] = (X,SX,y,size)
            print('第二次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])),'\nsearchparam:',searchparam)
            top_list = self.searchCluster(searchparam, runLog, [], 'top3',True)
        if step1 > 0 and step2 > 0 and step3 > 0:
            # 第三次搜索
            searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,step3]]
            searchparam['data'] = (X,SX,y,size)
            print('第三次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top_list = self.searchCluster(searchparam, runLog, [], 'top1',True)

        # 正式执行
        print('top_list===>',top_list)
        runparam = {
            "clusterer_name":method,
            "params":[int(top_list[0]['info']['params'][0])],
            "data":(X,SX,y,size),
            "param_scale":[1],
            "metricstring":""
        }
        print('正式执行,参数为:',runparam['params'])
        return self.runCluster(runparam, runLog, [], True)

        
    def circulateClusterSub(self,index,X,start,end,lazy=False):
        # 对于Sub而言，每次聚类都是在找齐一个需要聚类的向量，布置好搜索参数，交给聚类方法聚类
        #id
        return_index = np.array([0]*index.shape[0])
        # 遍历index中的每个值，实际就是对每个簇再进行聚类，当然第一次使对所有的聚类，因为第一次均为0
        for i in range(0,max(index)+1):
            # 当遍历到index中的i时 index: [0,0,0,0,0] X: [[xxxx],[xxxxx],[xxxxx]]
            X_sub = X[np.where(index == i)] # 拿去聚类,第一次index中的序号均为-1，所以都给了
            # 进行搜索,给予单纯的X返回标签即可
            if len(X_sub) <= 3:
                return_index[np.where(index == i)] = [0]*len(X_sub)
            else:
                index_sub = self.autoCluster(X_sub[:,start:end])
                # 将索引写回index
                return_index[np.where(index == i)] = index_sub

        if lazy == True:
            return return_index
        else:
            count = 0
            for root_cluster_no in range(0,max(index)+1):
                # 在这个簇中重新对class_index进行编码
                child_clusters = return_index[np.where(index == root_cluster_no)]
                for child_cluster_no in range(0,max(child_clusters)+1):
                    indexs = np.where((return_index == child_cluster_no) & (index == root_cluster_no))
                    return_index[indexs]= [count]*len(indexs)# 拿去聚类,第一次index中的序号均为-1，所以都给了
                    count += 1
            return return_index


    def circulateClusterController(self,params):
        # 向量都是一个向量，不过先拿出前...聚类
        # 聚类完成后会有labels, 根据这个labels取出同簇的再次聚类，循环多次
        X = self.readVector(self.config['API_VECTORS_PATH'])#)+apiSuffix)#, dtype='int32')
        SX = y = size = np.array([])
        # 用于给向量指示簇大小,1. 整个向量聚类，一次聚完；2. 部分向量分几次聚完
        index = np.array([0]*X.shape[0])

        # 得到每次聚类需要的向量
        result = self.DBP.queryMetaData(queryField={'package_api_len':1,'class_api_len':1,'method_api_len':1,'doc_api_len':1})
        print('api_len_distribute:',result)
        pl = int(result[0]['package_api_len'])
        cl = int(result[0]['class_api_len'])
        ml = int(result[0]['method_api_len'])
        dl = int(result[0]['doc_api_len'])
        #  90     683       3392   100
        #[    ;       ;           ;   ]
        #0    pl     pl+cl      -dl   
        #print('before:',X[0])
        #X[:,:pl] = X[:,:pl] * ratio[0]
        #X[:,pl:pl+cl] = X[:,pl:pl+cl] * ratio[1]
        #X[:,pl+cl:-dl] = X[:,pl+cl:-dl] * ratio[2]
        #X[:,-dl:] = X[:,-dl:] * ratio[3]
        package_index = self.circulateClusterSub(index,X,0,pl,False) #index:[0,0,1,1,2,1,3,3]
        class_index = self.circulateClusterSub(package_index,X,pl,pl+cl) #index:[0,0,1,2,3,1,4,4]
        method_index = self.circulateClusterSub(class_index,X,pl+cl,-dl) #index:[0,0,1,2,3,1,4,4]
        print('package_index',list(package_index))
        print('class_index',list(class_index))
        print('method_index',list(method_index))
        print('waiting for add..........................')
        

# -------------------------------------------------------抛弃------------------------------------------------
    def runCluster_hierarchy(self, params, runLog):
        from . import ClusteringCore
        from importlib import reload
        reload(ClusteringCore)
        pictureList = self.transcription()
        # 图像原始值
        SX, y, size = ClusteringCore.loadPictureData(pictureList)
        '''
        if params  ==  '':
            suffix = '-raw'
            suffix = '-raw'
        else:
            suffix = params
        '''
        #apiSuffix = '-'+self.subtask_name.split('_')[1]
        #))clusterSuffix = self.subtask_name

        # api特征, 为了放置降维算法重新映射维度，改为dtype='float32'
        X = self.readVector(self.config['API_VECTORS_PATH'])#)+apiSuffix)#, dtype='int32')

        from scipy.cluster import hierarchy
        plt.switch_backend('agg')
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
            labels = cluster.hierarchy.cut_tree(Z,height=100)
            self.DBP.updateMetaData({"$set":{'clustermethod':self.DBP.subtask_name, 'clusters':str(max(labels)+1)}})
        else:
            params = self.getRunParams()
            params['data'] = (X, SX, y, size)
            #print(params)
            clusterer_container = ClusteringCore.runClusterer(params['clusterer_name'], 
                    params['params'], 
                    params['data'], 
                    params['param_scale'], 
                    params['metricstring'])
            clusterer_container = ClusteringCore.clustererEvaluationMetric(clusterer_container)
            labels = clusterer_container['info']['clusterer'].labels_
            self.saveObject(clusterer_container, self.config['CLUSTERER_CONTAINER_PATH'])#+clusterSuffix)
            self.DBP.updateMetaData({"$set":{'clustermethod':self.DBP.subtask_name, 'clusters':str(clusterer_container['performance']['clusters_num']), 'silhouette':clusterer_container['performance']['silhouette'], 'calinski':clusterer_container['performance']['calinski']}})
        #print('clustering done:\n', ClusteringCore.simplifyClusteringContainer(clusterer_container))
        #ClusteringCore.storeClustersVector(clusterer_container, self.config['CLUSTER_RESULT'])
        self.writeVector(labels, '/home/dl/users/wangruifeng/05MisleadingWidgets/androidwidgetclustering/cluster.txt', fmt="%d")
        # 需要更新APKTree，可以根据获取的path列表，毕竟都是一样的
        self.addClusterInfo(labels)
        # 插入metadata信息
        #suffix2 = '-'+self.subtask_name.split('_')[2]
        #paramsStr = suffix
        #paramsStr = '-'+params['clusterer_name'] +'-'+  ", ".join([str(i) for i in params['params']])
        #self.writeDict(newApkTree, self.config['APK_TREE_PATH'])

        # 为了显示时方便
        params.pop('data')
        runLog.append('X.shape %s,  SX.shape %s,  y.shape %s,  size %s'% (X.shape, SX.shape, y.shape, size))
        #runLog.append('params %s' % params)
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
        percent_threshold = 0.4
        cluster_num = [5, 5000]
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
        vector = self.readVector(self.config['API_VECTORS_PATH'].split('-')[0]+'-visualize.txt')
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
# -------------------------------------------------------异常检测------------------------------------------------
    def OutlierDetectionfilterFun(self, clusterer_container):
        '''
        percent_threshold = 1
        cluster_num = [1, 5000]
        silhouette_threshold = 0
        calinski_threshold = 0
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
        '''
        return True
    def outlierDetecting(self,thiscluster,method='hierarch',showScore=True):
        runLog = []
        '''
        start1 = 1
        end1 = 100
        step1 = 5
        start2 = 1 #0.1
        end2 = 100 # 10
        step2 =  5
        '''

        widget_nums = thiscluster.shape[0]
        start = 2
        end = widget_nums+1
        step = 1

        ###### Build the search Params #######
        SX = y = size = np.array([])
        searchparam = {
            "clusterer_name":method,
            #"ranges":[[start1,end1,step1],[start2,end2,step2]],
            "ranges":[[start,end,step]],
            "data":(thiscluster,SX,y,size),
            "param_scale":[1],
            "metricstring":"",
            "filterFun":"",
            "verbose":""
        }
        # 搜索
        print('搜索中,参数为:',searchparam['ranges'])
        top = self.searchCluster(searchparam, runLog, [], 'top1',True)

        # 正式执行
        runparam = {
            "clusterer_name":method,
            #"params":[int(top[0]['info']['params'][0]*10),int(top[0]['info']['params'][1])],
            "params":[int(top[0]['info']['params'][0])],
            "data":(thiscluster,SX,y,size),
            "param_scale":[1],
            "metricstring":""
        }
        print('正式执行,参数为:',runparam['params'])
        return self.runCluster(runparam, runLog, [], True)
                
    def addClusterOutilerInfo(self, outlierVectorId, outlierVector,outlierAdviseVector):
        idUoutlier = []
        for i in range(len(outlierVector)):
            for j in range(len(outlierVector[i])):
                idUoutlier.append( [outlierVectorId[i][j], {"$set":{"outlier_score":str(outlierVector[i][j]) , "outlier_advise":str(outlierAdviseVector[i][j])}} ])
        self.DBP.updateAPKForest(idUoutlier)
    '''
    def sorted_dict(self,container, keys, reverse):
        """返回 keys 的列表,根据container中对应的值排序"""
        aux = [ (container[k], k) for k in keys]
        aux.sort()
        if reverse: aux.reverse()
        return [(k,v) for v,k in aux]
    
    def seekOutlierFromClusterOV(self,cluster_outlier_vector):
        # 首先统计outlier_vector中的簇 cluster_outlier_vector: [0 0 1 2 3 4 1 5 0 0 6 1 7 0 0 8 9 0 0]
        cluster_dict = dict()
        for cluster_no in range(-1,max(cluster_outlier_vector)+1):
            if -1 not in cluster_outlier_vector:
                continue
            cluster_dict[cluster_no] = len(np.where(cluster_outlier_vector == cluster_no))
            print('in seekOutlierFromClusterOV"s for: cluster_dict[cluster_no]',cluster_dict[cluster_no])
        # 取值最小的那些键 
        sorted_cluster_dict_Keys = self.sorted_dict(cluster_dict,cluster_dict.keys(),False)
        print('-------------sorted_cluster_dict_Keys:',sorted_cluster_dict_Keys)
        min_indexs = []
        min_value = 9999
        for k,v in sorted_cluster_dict_Keys:
            if v > min_value:
                break
            min_value = v
            min_indexs.append(k)

        print('=========',min_indexs)
        # 将这些键对应的均置为0,其他置为-1
        cluster_outlier_advise_vector = np.array([0]*len(cluster_outlier_vector))
        for k in min_indexs:
            cluster_outlier_advise_vector[np.where(cluster_outlier_vector == k)] = -1
        return cluster_outlier_advise_vector
        '''
    def sorted_dict(self,container, keys, reverse, by='value'):
        """返回 keys 的列表,根据container中对应的值排序"""
        if by == 'value':
            aux = [ (container[k], k) for k in keys]
            aux.sort()
            if reverse: aux.reverse()
            return [(k,v) for v,k in aux]
        else:
            aux = [ (k , container[k]) for k in keys]
            aux.sort()
            if reverse: aux.reverse()
            return [(v,k) for v,k in aux]

    def seekOutlierFromClusterOV(self,cluster_outlier_vector):
        # 首先统计outlier_vector中的簇 cluster_outlier_vector: [0 0 1 2 3 4 1 5 0 0 6 1 7 0 0 8 9 0 0]
        cluster_dict = dict()
        cluster_num_dict = dict()
        for cluster_no in range(-1,max(cluster_outlier_vector)+1):
            if (cluster_no == -1) and (-1 not in cluster_outlier_vector):
                continue
            cluster_num = len(np.where(cluster_outlier_vector == cluster_no)[0])
            cluster_dict[cluster_no] = cluster_num
            if cluster_num in cluster_num_dict:
                cluster_num_dict[cluster_num].append(cluster_no)
            else:
                cluster_num_dict[cluster_num] = [cluster_no]
        # 取值最小的那些键 
        sorted_cluster_dict = self.sorted_dict(cluster_dict,cluster_dict.keys(),False)
        sorted_cluster_num_dict = self.sorted_dict(cluster_num_dict,cluster_num_dict.keys(),False,'key')
        # 标上数量最小的，依次遍历，如果全是最小的，不再标(如果sorted_cluster_num_dict.min2[0] == sorted_cluster_dict.min)
        # 只有一个长度，都一样，肯定全0
        min_indexs = []
        if len(sorted_cluster_num_dict) == 1:
            pass
        else:
            for k,v in sorted_cluster_dict:
                # 如果簇内数量大于等于第二小的簇内数量
                if v >= sorted_cluster_num_dict[1][0]:
                    break
                min_value = v
                min_indexs.append(k)
        # 将这些键对应的均置为0,其他置为-1
        cluster_outlier_advise_vector = np.array([0]*len(cluster_outlier_vector))
        for k in min_indexs:
            cluster_outlier_advise_vector[np.where(cluster_outlier_vector == k)] = -1
        return cluster_outlier_advise_vector

    def outlierDetectingController(self):
        # 这种聚类数量不会太大，但是必须自动
        ids, clusters = self.transcription_ids()
        print('读取特征:',self.config['EXTRACTED_FEATURE_PATH'])
        extracted_features = self.readVector(self.config['EXTRACTED_FEATURE_PATH'], dtype='')

        clusters = np.array(clusters)
        ids = np.array(ids)
        cluster_num = max(clusters)+1
        outlierVectorId = []
        outlierVector = []
        outlierAdviseVector = []
        for i in range(-1, cluster_num):
            # 首先取出clusters列表中==i的簇所有成员
            ithClusterIndex = np.where(clusters == i)
            # 该簇对应位置的id，便于后面的提交
            thisclusterId = ids[ithClusterIndex]
            # 如果这个簇没有，一般可能是kmeans没有-1，则不再继续
            if len(thisclusterId)==0:continue
            # 某个簇的特征矩阵，需要对其实行outlier算法
            thiscluster = extracted_features[ithClusterIndex]
            print('in outlierDetectingController:',thiscluster.shape)
            print('=============>this cluster::',thiscluster)
            # 全部记录下来，记录在两个结构中
            outlierVectorId.append(thisclusterId)
            if i == -1 or len(thiscluster) == 1 or len(thiscluster) == 2:
                outlierVector.append([0]*len(thiscluster))
                outlierAdviseVector.append([0]*len(thiscluster))
            else:
                cluster_outlier_vector = self.outlierDetecting(thiscluster, 'hierarch', True)
                print('cluster_outlier_vector',cluster_outlier_vector)
                cluster_outlier_advise_vector = self.seekOutlierFromClusterOV(cluster_outlier_vector)
                print('cluster_outlier_advise_vector',cluster_outlier_advise_vector)
                outlierVector.append(cluster_outlier_vector)
                outlierAdviseVector.append(cluster_outlier_advise_vector)
        self.addClusterOutilerInfo(np.array(outlierVectorId),np.array(outlierVector),np.array(outlierAdviseVector))


    #runCluster:运行最佳参数的聚类
    def runClustering(self, operator=[], params=[]):
        runLog = []
        runLog.append(operator)
        if not self.isExist(self.config['CLUSTER_COMMAND_DIR']+'clusterParams.py'):
            self.makeSureExists(self.config['CLUSTER_COMMAND_DIR'])
            self.copyFile(self.config['SCRIPT_ROOT']+'VectorClustering/config/clusterParams.py', self.config['CLUSTER_COMMAND_DIR']+'clusterParams.py')
        if operator == 'info' or operator == '':
            self.info()
        if operator  ==  'search':
            
            #ratio = [0.4,0.4,0.2,0]
            #ratio = [0.1,0.2,0.6,0.1]
            ratio = [int(params[0])*0.1,int(params[1])*0.1,int(params[2])*0.1,int(params[3])*0.1]
            params = ''
            #ratio = [0.2,0.3,0.5,0]
            runLog = self.searchCluster(params, runLog, ratio)
        if operator  ==  'run':
            # 放置重复运行
            cluster_data = self.DBP.queryAllCluster()
            if len(cluster_data) > 0:
                print('/\/\cluster data exists, deleteing it?')                                               
                curApkTree = self.DBP.getApkForestDB()                                                  
                curApkTree.drop()
            ratio = [0.1,0.2,0.6,0.1]
            #ratio = [int(params[0])*0.1,int(params[1])*0.1,int(params[2])*0.1,int(params[3])*0.1]
            params = ''

            runLog = self.runCluster(params, runLog, ratio)
        if operator == 'auto':
            cluster_data = self.DBP.queryAllCluster()
            if len(cluster_data) > 0:
                print('/\/\cluster data exists, deleteing it?')                                               
                curApkTree = self.DBP.getApkForestDB()                                                  
                curApkTree.drop()

            ##### Initial Params #####
            ratio = [0,0,1,0]
            method = 'hierarch'
            try:
                metadata = self.DBP.queryMetaData({'record_len':1})
                widget_nums = int(metadata[0]['record_len'])
            except:
                X = self.readVector(self.config['API_VECTORS_PATH'])#+suffix)#, dtype='int32')
                widget_nums = X.shape[0]

            start = int(widget_nums/5)
            if start <3:start = 3
            end = widget_nums
            step1 = int((end-start)/10)
            if step1 <= 0:step1 = 1
            step2 = int((end-start)/50)
            step3 = int((end-start)/100)

            multiplier = 0.1
            ##### Given Params #####
            # VC auto 3331|0.1@optics(1,7,1,1,1)
            if params != '':
                if '(' in params:
                    params,searchrange = params.rstrip(')').split('(')
                    start,end,step1,step2,step3 = search.split(',')
                    start = int(start)
                    end = int(end)
                    step1 = int(step1)
                    step2 = int(step2)
                    step3 = int(step3)

                # 首先解析聚类参数 
                if '@'  in params:
                    params,method = params.split('@')

                if params == '':
                    pass
                else:
                    # next the wigets: 3341|0.1
                    if '|' in params:
                        multiplier = float(params.split('|')[-1])

                    ratio = [int(params[0])*multiplier,int(params[1])*multiplier,int(params[2])*multiplier,int(params[3])*multiplier]

                    if '|' not in params and ratio.count(0.1) + ratio.count(0.0) == len(ratio):
                        for i in range(len(ratio)):
                            ratio[i] = ratio[i]*10
                    params = ''

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
            top_list = self.searchCluster(searchparam, runLog, ratio, 'top3')

            # 第二次搜索
            searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,step2]]
            print('第二次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top_list = self.searchCluster(searchparam, runLog, ratio, 'top3')

            # 第三次搜索
            searchparam['ranges'] = [[int(top_list[0]['info']['params'][0]),int(top_list[-1]['info']['params'][0])+1,step3]]
            print('第三次搜索,参数为:',searchparam['ranges'],'=>',list(range(*searchparam['ranges'][0])))
            top = self.searchCluster(searchparam, runLog, ratio, 'top1')

            # 正式执行
            runparam = {
                "clusterer_name":method,
                "params":[int(top[0]['info']['params'][0])],
                "data":"",
                "param_scale":[1],
                "metricstring":""
            }
            print('正式执行,参数为:',runparam['params'])
            runLog = self.runCluster(runparam, runLog, ratio)
        

        if operator == 'circ':
            self.circulateClusterController(params)
        if operator == 'od':
            self.outlierDetectingController()

        # 相同聚类
        if operator == 'sim':
            #runLog = self.runSimCluster('F',runLog)
            runLog = self.runSimCluster('P',runLog)
            '''
            import hashlib
            import numpy as np  
            
            cluster_data = self.DBP.queryAllCluster()
            if len(cluster_data) > 0:
                print('/\/\cluster data exists, deleteing it?')                                               
                curApkTree = self.DBP.getApkForestDB()                                                  
                curApkTree.drop()
            # 首先获取所有API信息
            idUapi = self.DBP.queryAllApi()
            apis = np.array(idUapi)[:,1] # 取半条链作为mRNA 
            # 将每个控件的API信息进行hash
            apihash2widgets_dict = dict()
            
            # 将hash变为dict的key，value中存储序号
            count = 0
            for api in apis:
                seed = ('\n'.join(api)).encode("utf8")
                api_hash = hashlib.md5(seed).hexdigest()
                if api_hash in apihash2widgets_dict:
                    apihash2widgets_dict[api_hash].append(count)
                else:
                    apihash2widgets_dict[api_hash] = [count]
                count += 1
            # 由于序号和id顺序一致，按照和addClusterInfo的相同逻辑即可完成信息提交
            cluster_no = 0
            clusterLabels = [0] * len(apis)

            # 遍历dict中所有元素，然后按照dict中记录的顺序
            for key,value in apihash2widgets_dict.items():
                # 给数据库中的控件标注簇号，如果所在簇数量太少<5，标为-1
                if len(value) < 5:
                    for i in value:
                        clusterLabels[i] = -1
                else:
                    for i in value:
                        clusterLabels[i]  = cluster_no
                cluster_no += 1
            self.addClusterInfo(clusterLabels)
            self.DBP.updateMetaData({"$set":{'clustermethod':'sim', 'clusters':str(max(clusterLabels)+1), 'silhouette':'NAN', 'calinski':'NAN'}})
            '''

'''旧的使用Sklearn中搜索的功能
        from sklearn.metrics import make_scorer
        from . import s_Dbw
        from importlib import reload
        reload(s_Dbw)
        from sklearn.cluster import DBSCAN
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.model_selection import GridSearchCV#RandomizedSearchCV
        # 0.5,10 注意！！ eps 被缩小一个尺度!!!
        # 设定聚类器
        #clusterer = DBSCAN()#eps=params[0], min_samples=params[1])
        clusterer = AgglomerativeClustering(affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='average', distance_threshold=None)
        #clusterer = DBSCAN(metric='wminkowski',p=2,metric_params={"w":idf})

#eps=params[0], min_samples=params[1])
        # 用于随机搜索
        #param_dist = {
        #        'eps': np.linspace(0.01,100,100),
        #        'min_samples': range(1,1000,100)
        #        }
        #param_dist = {
        #        'eps': np.linspace(0.01,100,10),
        #        'min_samples': np.linspace(0.01,100,100),#range(1,100,5)
        #        }
        #
        param_dist = {
                #'n_clusters': range(500,7000,1000),
                'n_clusters': range(int(start),int(end),int(step)),
        }
        print('正在搜索...')
        #
        #self, estimator, param_grid, scoring=None,
        #         n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
        #         pre_dispatch='2*n_jobs', error_score='raise-deprecating',
        #         return_train_score=False
        #
        clusterer_grid = GridSearchCV(clusterer,param_dist,cv=None,scoring=s_Dbw.accuracy_score,n_jobs = -1)
        clusterer_grid.fit(X)
        print('搜索完成，正在进行选择')
        #best_clusterer = clusterer_grid.best_estimator_
        print('clusterer:',best_clusterer, clusterer_grid.best_params_)
        print('clusters:',max(best_clusterer.labels_)+1,'\nscore:',clusterer_grid.best_score_)
        return runLog
'''
