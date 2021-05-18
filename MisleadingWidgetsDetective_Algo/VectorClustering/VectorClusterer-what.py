import sys, os
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor 
from bson import ObjectId

class VectorClusterer(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(VectorClusterer, self).__init__()
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

    def searchcluster(self, params, runlog, ratio=0):
        from . import clusteringcore
        from importlib import reload
        reload(clusteringcore)
        #picturelists = self.getpicturelistsfromtree(self.readdict(self.config['apk_tree_path']), self.config['input_data_dir'])
        picturelist = self.transcription()
        # 图像原始值
        sx, y, size = clusteringcore.loadpicturedata(picturelist)
        '''
        if params  ==  '':
            suffix = '-raw'
        else:
            suffix = params
        '''
        #suffix = '-'+self.subtask_name.split('_')[1]
        # api特征, 为了放置降维算法重新映射维度，改为dtype='float32'
        x = self.readvector(self.config['api_vectors_path'])#+suffix)#, dtype='int32')
        #idf = self.readvector(self.config['api_idf_path'].split('-')[0]+'.txt')
        #print('===>idf:',idf)
        
        start,end,step = params.split(',')
        params = self.getsearchparams()
        params['data'] = (x, sx, y, size)

        #################这里更新使用新的随机搜索###############################
        import numpy as np
        print('x.shape:',np.array(x).shape) #=>(1010, 500)
        print('sx.shape:',np.array(sx).shape)
        x = np.array(x)
        result = self.dbp.querymetadata(queryfield={'package_api_len':1,'class_api_len':1,'method_api_len':1,'doc_api_len':1})
        print(result)
        pl = int(result[0]['package_api_len'])
        cl = int(result[0]['class_api_len'])
        ml = int(result[0]['method_api_len'])
        dl = int(result[0]['doc_api_len'])
        #  90     683       3392   100
        #[    ;       ;           ;   ]
        #0    pl     pl+cl      -dl   
        #print('before:',x[0])
        if ratio == 0:
            # 0.7 0.1 0.1 0.1 pure_big_researchratio.hog_complimit5fuse7111_raw_hierarch_iforest
            x[:,:pl] = x[:,:pl] * 0.1
            x[:,pl:pl+cl] = x[:,pl:pl+cl] * 0.1
            x[:,pl+cl:-dl] = x[:,pl+cl:-dl] * 0.1
            x[:,-dl:] = x[:,-dl:] * 0.7
        elif ratio == 1:
            # pure_big_researchratio.hog_complimit5fuse1333_raw_hierarch_iforest
            x[:,:pl] = x[:,:pl] * 0.3
            x[:,pl:pl+cl] = x[:,pl:pl+cl] * 0.3
            x[:,pl+cl:-dl] = x[:,pl+cl:-dl] * 0.3
            x[:,-dl:] = x[:,-dl:] * 0.1
        else:
            x[:,:pl] = x[:,:pl] * 0.2
            x[:,pl:pl+cl] = x[:,pl:pl+cl] * 0.3
            x[:,pl+cl:-dl] = x[:,pl+cl:-dl] * 0.1
            x[:,-dl:] = x[:,-dl:] * 0.4
        #print('after:',x[0])
        '''
        data = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 2], [2, 2, 2], [2,2,3]])
        data_cluster = np.array([1, 0, 1, 2, 2]) 
        centers_index = np.array([1, 0, 3]) 
       
        #a = s_dbw(data, data_cluster, centers_index)
        a = s_dbw(data, data_cluster)
        print(a.result())
        '''
        from sklearn.metrics import make_scorer
        from . import s_dbw
        from importlib import reload
        reload(s_dbw)
        from sklearn.cluster import dbscan
        from sklearn.cluster import agglomerativeclustering
        from sklearn.model_selection import gridsearchcv#randomizedsearchcv
        # 0.5,10 注意！！ eps 被缩小一个尺度!!!
        # 设定聚类器
        #clusterer = dbscan()#eps=params[0], min_samples=params[1])
        clusterer = agglomerativeclustering(affinity='euclidean', memory=none, connectivity=none, compute_full_tree='auto', linkage='average', distance_threshold=none)
        #clusterer = dbscan(metric='wminkowski',p=2,metric_params={"w":idf})

#eps=params[0], min_samples=params[1])
        ''' 用于随机搜索
        param_dist = {
                'eps': np.linspace(0.01,100,100),
                'min_samples': range(1,1000,100)
                }
        param_dist = {
                'eps': np.linspace(0.01,100,10),
                'min_samples': np.linspace(0.01,100,100),#range(1,100,5)
                }
        '''
        param_dist = {
                #'n_clusters': range(500,7000,1000),
                'n_clusters': range(int(start),int(end),int(step)),
        }
        print('正在搜索...')
        '''
        self, estimator, param_grid, scoring=none,
                 n_jobs=none, iid='warn', refit=true, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score=false
        '''
        clusterer_grid = gridsearchcv(clusterer,param_dist,cv=none,scoring=s_dbw.accuracy_score,n_jobs = -1)
        clusterer_grid.fit(x)
        print('搜索完成，正在进行选择')
        best_clusterer = clusterer_grid.best_estimator_
        print('clusterer:',best_clusterer, clusterer_grid.best_params_)
        print('clusters:',max(best_clusterer.labels_)+1,'\nscore:',clusterer_grid.best_score_)
        return runlog

        resultlist =  clusteringcore.searchparams(clusterer_name = params['clusterer_name'], 
                ranges = params['ranges'], 
                data = params['data'], 
                param_scale = params['param_scale'], 
                metricstring = params['metricstring'], 
                filterfun = self.filterfun, 
                verbose = params['verbose'])
        #print('聚类完成后list:\n')
        [print(self.simplifyclusteringcontainer(result), '\n-------------------------------') for result in resultlist]
        # 用于选择参数
        self.choiceparams(resultlist)
        # 为了显示时方便
        params.pop('data')
        runlog.append('x.shape %s,  sx.shape %s,  y.shape %s,  size %s'% (x.shape, sx.shape, y.shape, size))
        runlog.append('params %s' % params)
        runlog.append('resultlist len %d' % len(resultlist))

    def addclusterinfo(self, clusterlabels):
        '''将簇信息加入到subtask库中，从这里建立数据库g'''
        idupath = self.dbp.queryallpath() 
        for i in range(len(idupath)):
            #idupath[i][1] = {"$set":{"cluster":str(clusterlabels[i])}}
            idupath[i][1] = str(clusterlabels[i])
        self.dbp.saveapkforest(idupath)

    # 目前必须得是mvector才能使用
    def apifilter(self):
        import numpy as np
        # 根据api的idf来判断api的重要程度,一次性返回需要考虑的api
        _,_,method_apis = self.api_transcription() #获取method api，即完整api
        api_idf = self.readvector(self.config['api_idf_path'].split('-')[0]+'.txt')
        api_names = np.array(self.readtxt(self.config['api_vectors_path']+'-'+'apinames.txt'))
        exceptlist = ['android.app.activity.startactivity',
        'android.app.activity.finish',
        'android.app.activity.onoptionsitemselected',
        'android.net.uri.parse',
        'android.support.v4.view.viewpager.setcurrentitem',
        'android.app.activity.onprepareoptionsmenu',
        'android.view.view.getheight']
        # > 12??
        api_index = np.where((api_idf < 20) & (api_idf > 1) & (api_names not in exceptlist))[0]
        print(api_index,'\n',len(api_index))
        # 返回这种api名字集合
        return api_names[api_index]

    # fsim 聚类方法中的api过滤法
    def dofilter(self,apiwhitelist,api):
        if len(api) <= 2:
            return api
        return_api = []
        for item in api:
            if item in apiwhitelist:
                return_api.append(item)
        return return_api


    # 主要关注项目: db_pure_big: pure_big_importsimapi.hog_complimit5fsim_raw_hierarch_iforest   4964 clusters total
    # 思路2: 对于每两个控件，去除公有的，查看不一样的那些是不是都是低idf的。如果不一样的都是低idf那就不看这些低idf的了
    # 注意：直接抛弃这些api: android.util.log.v
    def runSimCluster(self,params , runlog):
        # 完全sim
        import hashlib
        import numpy as np  
        isaddfilter = params.strip(' ')
        
        cluster_data = self.dbp.queryallcluster()
        if len(cluster_data) > 0:
            print('/\/\cluster data Exists, deleteing it?')                                               
            curapktree = self.dbp.getapkforestdb()                                                  
            curapktree.drop()
        # 首先获取所有api信息
        iduapi = self.dbp.queryallapi()
        apis = np.array(iduapi)[:,1] # 取半条链作为mrna 
        # 将每个控件的api信息进行hash
        apihash2widgets_dict = dict()
        
        # 将hash变为dict的key，value中存储序号
        count = 0
        apiwhitelist = self.apifilter()
        for api in apis:
            # api实际上是一个api列表
            # 首先进行api过滤(需要定义合适的阈值)
            if isaddfilter == 'f':
                api = self.dofilter(apiwhitelist,api)
            # 接着比较过滤后的api
            seed = ('\n'.join(api)).encode("utf8")
            api_hash = hashlib.md5(seed).hexdigest()
            if api_hash in apihash2widgets_dict:
                apihash2widgets_dict[api_hash].append(count)
            else:
                apihash2widgets_dict[api_hash] = [count]
            count += 1
        # 由于序号和id顺序一致，按照和addclusterinfo的相同逻辑即可完成信息提交
        cluster_no = 0
        clusterlabels = [0] * len(apis)

        # 遍历dict中所有元素，然后按照dict中记录的顺序
        for key,value in apihash2widgets_dict.items():
            # 给数据库中的控件标注簇号，如果所在簇数量太少<5，标为-1
            if len(value) < 3:
                for i in value:
                    clusterlabels[i] = -1
            else:
                for i in value:
                    clusterlabels[i]  = cluster_no
            cluster_no += 1
        self.addclusterinfo(clusterlabels)
        self.dbp.updatemetadata({"$set":{'clustermethod':'sim', 'clusters':str(max(clusterlabels)+1), 'silhouette':'nan', 'calinski':'nan'}})

    def runcluster(self, params, runlog):
        from . import clusteringcore
        from importlib import reload
        reload(clusteringcore)
        picturelist = self.transcription()
        # 图像原始值
        sx, y, size = clusteringcore.loadpicturedata(picturelist)
        '''
        if params  ==  '':
            suffix = '-raw'
            suffix = '-raw'
        else:
            suffix = params
        '''
        #apisuffix = '-'+self.subtask_name.split('_')[1]
        #))clustersuffix = self.subtask_name

        # api特征, 为了放置降维算法重新映射维度，改为dtype='float32'
        x = self.readvector(self.config['api_vectors_path'])#)+apisuffix)#, dtype='int32')

        params = self.getrunparams()
        params['data'] = (x, sx, y, size)
        #print(params)
        clusterer_container = clusteringcore.runclusterer(params['clusterer_name'], 
                params['params'], 
                params['data'], 
                params['param_scale'], 
                params['metricstring'])
        clusterer_container = clusteringcore.clustererevaluationmetric(clusterer_container)
        #print('clustering done:\n', clusteringcore.simplifyclusteringcontainer(clusterer_container))
        #clusteringcore.storeclustersvector(clusterer_container, self.config['cluster_result'])
        self.writevector(clusterer_container['info']['clusterer'].labels_, '/home/dl/users/wangruifeng/05misleadingwidgets/androidwidgetclustering/cluster.txt', fmt="%d")
        # 需要更新apktree，可以根据获取的path列表，毕竟都是一样的
        self.addclusterinfo(clusterer_container['info']['clusterer'].labels_)
        # 插入metadata信息
        #suffix2 = '-'+self.subtask_name.split('_')[2]
        #paramsstr = suffix
        #paramsstr = '-'+params['clusterer_name'] +'-'+  ", ".join([str(i) for i in params['params']])
        self.dbp.updatemetadata({"$set":{'clustermethod':self.dbp.subtask_name, 'clusters':str(clusterer_container['performance']['clusters_num']), 'silhouette':clusterer_container['performance']['silhouette'], 'calinski':clusterer_container['performance']['calinski']}})
        #self.writedict(newapktree, self.config['apk_tree_path'])
        self.saveobject(clusterer_container, self.config['clusterer_container_path'])#+clustersuffix)

        # 为了显示时方便
        params.pop('data')
        runlog.append('x.shape %s,  sx.shape %s,  y.shape %s,  size %s'% (x.shape, sx.shape, y.shape, size))
        runlog.append('params %s' % params)
        runlog.append('cluster_result:%s'% clusteringcore.simplifyclusteringcontainer(clusterer_container))
        return runlog
    
    #evalcluster:评价聚类效果，用图形展示参数和示例图片
    def evalclustering(self, params, runlog):
        # 不用加载簇的标签，直接调用图就行
        from . import clusteringcore
        from importlib import reload
        reload(clusteringcore)
        # 获取数据进行评测
        showcluster = shownum = ''
        paramlist = params.split(', ')
        # 如果参数只有一个且为空，pass
        if len(paramlist) == 1 and paramlist[0] == '': 
            pass
        else:
            if paramlist[0] != '':
                showcluster = int(paramlist[0])
            if paramlist[1] != '':
                shownum = int(paramlist[1])
        showscore = true

        # 读取container
        params = self.getrunparams()
        #paramsstr = '-'+params['clusterer_name'] +'-'+  ", ".join([str(i) for i in params['params']])
        clusterer_container = self.loadobject(self.config['clusterer_container_path'])#+clustersuffix)

        # 保存图片
        self.makesureExists(self.config['cluster_picture_result_dir'])
        print('writing images...')
        clusteringcore.showclusterimages(clusterer_container, showcluster, shownum, [], showscore, self.config['cluster_picture_result_dir'])
        clusteringcore.showclusterhistogram(clusterer_container, savepath=self.config['cluster_picture_result_dir'])#+clustersuffix)
        return runlog

    def analysisparams(self, paramstring):
        if paramstring  ==  '':
            return {}
        params = dict()
        parampairs = paramstring.split(';')
        for item in parampairs:
            item = item.split(":")
            params[item[0]] = item[1]
        return params

    def simplifyclusteringcontainer(self, clusterer_container):
        import numpy as np
        simplifiedclusteringcontainer = {}
        infodict = datadict = performancedict = {}
        infodict['clusterer_name'] = clusterer_container['info']['clusterer_name']
        infodict['params']=clusterer_container['info']['params']
        #infodict['inertia_'] = clusterer_container['info']['clusterer'].inertia_
        #print('原始即:', infodict['inertia_'] )
        infodict['metricstring']=clusterer_container['info']['metricstring']

        datadict['xsize'] = np.array(clusterer_container['data']['x']).shape
        datadict['xdtype'] = str(np.array(clusterer_container['data']['x']).dtype)
        #datadict['sxsize'] = np.array(clusterer_container['data']['sx']).shape
        #datadict['sxdtype'] = np.array(clusterer_container['data']['sx']).dtype
        datadict['ysize'] = np.array(clusterer_container['data']['y']).shape
        datadict['ydtype'] = str(np.array(clusterer_container['data']['y']).dtype)
        datadict['size'] = clusterer_container['data']['size']
        import copy
        performancedict = copy.deepcopy(clusterer_container['performance'])
        performancedict['cluster_elements_percent'] =  performancedict['cluster_elements_percent'][:10]
        simplifiedclusteringcontainer['info']=infodict
        simplifiedclusteringcontainer['data']=datadict
        simplifiedclusteringcontainer['performance']=performancedict
        return simplifiedclusteringcontainer

    def filterfun(self, clusterer_container):
        percent_threshold = 0.4
        cluster_num = [5, 50]
        silhouette_threshold = 0.01
        calinski_threshold = 0
        # 首先不能有一项超过0.5, 即一半，那个肯定有问题
        for item in clusterer_container['performance']['cluster_elements_percent']:
            if item >= percent_threshold:
                return false
        # 再者轮廓系数要大于0
        if clusterer_container['performance']['silhouette'] <= silhouette_threshold:
            return false
        if clusterer_container['performance']['calinski'] <= calinski_threshold:
            return false
        # 聚类的数量需要有限制
        if  clusterer_container['performance']['clusters_num'] > cluster_num[1] and clusterer_container['performance']['clusters_num'] < cluster_num[0]:
                return false
        return true

    def visualizeclustering(self, params ,runlog):
        # coordinates 二维的坐标
        # images 图像，且缩放
        vector = self.readvector(self.config['api_vectors_path'].split('-')[0]+'-visualize.txt')
        cluster_data = self.dbp.queryallcluster()

        picturelist = self.transcription()
        # 图像原始值
        sx,size = self.loadpicturedata(picturelist,method='pil',resize=(8,8))

        coordinates = vector
        print('coordinates shape:',coordinates.shape)
        images = sx
        print('images shape:',images.shape)
        self.plot_embedding_scatter(coordinates,images,figsize=(1,1),frameon=false,title=none,xticks=[],yticks=[],min_dist=4e-6)
        return runlog
    
    #runcluster:运行最佳参数的聚类
    def runClustering(self, operator=[], params=[]):
        runLog = []
        runlog.append(operator)
        if not self.isExist(self.config['CLUSTER_COMMAND_DIR']+'clusterparams.py'):
            self.makeSureExists(self.config['CLUSTER_COMMAND_DIR'])
            self.copyFile(self.config['SCRIPT_ROOT']+'VectorClustering/config/clusterParams.py', self.config['CLUSTER_COMMAND_DIR']+'clusterparams.py')
        if operator == 'info' or operator == '':
            self.info()
        if operator  ==  'search':
            runLog = self.searchCluster(params, runLog)
        if operator  ==  'run':
            # 放置重复运行
            cluster_data = self.DBP.queryAllCluster()
            if len(cluster_data) > 0:
                print('/\/\cluster data Exists, deleteing it?')                                               
                curApkTree = self.DBP.getApkForestDB()                                                  
                curApkTree.drop()
            runLog = self.runCluster(params, runLog)
        # 相同聚类
        if operator == 'sim':
            runLog = self.runSimCluster('F',runLog)
            '''
            import hashlib
            import numpy as np  
            
            cluster_data = self.DBP.queryAllCluster()
            if len(cluster_data) > 0:
                print('/\/\cluster data Exists, deleteing it?')                                               
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
