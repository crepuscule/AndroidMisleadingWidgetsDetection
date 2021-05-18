# 这个类需要重构
# 将本地IO写在一个文件中
# 再将数据库IO写在另一个文件中
# 或者将暂时用不到的本地方法移动到另一个文件中

from pymongo import MongoClient
from bson import ObjectId

class DataBaseProcessor():
    def __init__(self):
        #self.IP = '47.95.196.25'
        #self.IP = '172.17.9.10'
        self.IP = '127.0.0.1'
        self.PORT=27017
        self.db_projects_index = 'db_projects_index'
        # 根据自己的数据集来，数据集不同，DBNAME不同

        self.DBNAME = 'db_pure_big' # 12.29日用于统计API集合大小分布、从而API数量少的icons进行聚类=>对应数据库'apinumberstatistics'
        self.rawApkForestName = 'compatibleapilimit5'
        
        self.project_name = 'project_name'
        self.subtask_name = 'subtask_name'
        self.configTableName = self.project_name+'_config'
        self.metaDataTableName = self.project_name + '_metadata'
        self.apkTreeTableName = self.project_name + self.subtask_name+'_apktree'

    def info(self):
        info = ''
        info += self.DBNAME+':'
        info += self.rawApkForestName 

        print(info)
    ###########        Base DB IO  ##############
    #get
    def getConnection(self):
        connection = MongoClient(self.IP, self.PORT)
        db = connection[self.DBNAME]  #连接mydb数据库，没有则自动创建
        return db

    def getIndexDB(self):
        connection = MongoClient(self.IP, self.PORT)
        db = connection[self.db_projects_index]  #连接mydb数据库，没有则自动创建
        collect_index = db['projects_index']
        #print('=> use %s projects_index')
        return collect_index

    def getConfigDB(self):
        db = self.getConnection()
        collect_config = db[self.configTableName]
        #print('=> use %s' % self.configTableName)
        return collect_config

    def getRawApkForestDB(self):
        db = self.getConnection()
        collect_rawapkforest = db[self.rawApkForestName]
        #print('=> use %s' % self.rawApkForestName)
        return collect_rawapkforest

    def getApkForestDB(self):
        db = self.getConnection()
        collect_apktree = db[self.apkTreeTableName]
        #print('=> use %s' % self.apkTreeTableName)
        return collect_apktree

    def getMetaDataDB(self):
        db = self.getConnection()
        collect_metadata = db[self.metaDataTableName]
        #print('=> use %s' % self.metaDataTableName)
        return collect_metadata

    def getUserDataDB(self):
        db = self.getConnection()
        collect_users = db['users']
        #print('=> use %s' % 'users')
        return collect_users

    def getEvaluationDataDB(self):
        db = self.getConnection()
        collect_evaluations = db['evaluations']
        #print('=> use %s' % 'evaluations')
        return collect_evaluations

    # ------------------collect_index--------------------
    def queryIndex(self,queryFilter={},queryField={}):
        # 根据条件筛选,_id包括在findFilter之中即可
        collect_index = self.getIndexDB()
        findResult = collect_index.find(queryFilter,queryField).sort("_id",1)
        #findResult = collect_index.find(queryFilter).sort("raw_id",1)
        findResultList = list(findResult)
        print(len(findResultList),' docs queried.')
        return findResultList

    def getIndexs(self):
        idUprojects = self.queryIndex({},{'_id':1,'project_name':1,'project_database':1,'project_rawapkforest':1,'project_desc':1})
        for i in range(len(idUprojects)):
            idUprojects[i] = [idUprojects[i]['_id'],idUprojects[i]['project_name'],idUprojects[i]['project_database'],idUprojects[i]['project_rawapkforest'],idUprojects[i]['project_desc']]
        print(len(idUprojects),' docs queried.')
        return idUprojects

    # ------------------collect_config-----------------
    def initConfig(self,_id=None):
        import sys,os                                                                                                                                         
        from importlib import reload                                                                                                                          
        sys.path.append(os.path.abspath('.')) 
        sys.path.append(os.path.abspath('../Assistant/')) 
        import DefaultConfiguration
        reload(DefaultConfiguration) 
        # use 1,2 to use different config file 
        #self.config = configuration.initConfig(2) 
        defaultConfig = DefaultConfiguration.initConfig(self.project_name,1)
        # 用于更新
        if _id != None:
            defaultConfig["_id"] = _id
        collect_config = self.getConfigDB()
        collect_config.save(defaultConfig)
        print('config init done.')

    def updateConfig(self,updateValue):
        # content: [id,{"$set":{'xxx':xx}}] , [...]
        collect_config = self.getConfigDB()
        collect_config.update_one({"INSTANCE_NAME":self.project_name},updateValue)
        print('metadata updated.')

    def getConfig(self,update=False):
        collect_config = self.getConfigDB()
        config =  collect_config.find_one()
        if config == None:
            print('config not found, turn to init it...')
            self.initConfig()
            return self.getConfig()
        elif update == True:
            print('config need to update, turn to init it...')
            self.initConfig(config["_id"])
            return self.getConfig()
        else:
            return config
    # ------------------collect_metadata-----------------
    # 一次是save一个控件，如何被读出的_id未被更改，重新插入则是更新
    def saveMetaData(self,content,_id=None):
        # 这就是创建一个新项目
        collect_metadata = self.getMetaDataDB()
        if _id !=None:
            content['_id'] = _id
        insertResult = collect_metadata.save(content)
        #print(len(content.keys()),' keys in the doc.')
        return insertResult

    def updateMetaDataRemark(self,newRemark):
        # content: [id,{"$set":{'xxx':xx}}] , [...]
        collect_metadata = self.getMetaDataDB()
        print('self.subtask_name',self.subtask_name)
        collect_metadata.update_one({"subtask_name":self.subtask_name},{"$set":{"remark":newRemark}})
        print('metadata updated.')

    def updateMetaData(self,updateValue):
        # content: [id,{"$set":{'xxx':xx}}] , [...]
        collect_metadata = self.getMetaDataDB()
        print('self.subtask_name',self.subtask_name)
        collect_metadata.update_one({"subtask_name":self.subtask_name}, updateValue)
        print('metadata updated.')

    def queryMetaData(self,queryField={}):
        # 根据条件筛选,_id包括在findFiltr之中即可
        collect_metadata = self.getMetaDataDB()
        findResult = collect_metadata.find({"subtask_name":self.subtask_name},queryField).sort("_id",1)
        #findResult = collect_metadata.find({"subtask_name":self.subtask_name}).sort("_id",1)
        findResultList = list(findResult)
        print(len(findResultList),' docs queried.')
        return findResultList 

    def queryMetaDataByName(self,subtask_name):
        return self.queryMetaData({"subtask_name":subtask_name})

    # {'username':''}
    def queryUsers(self,filed={},judage={}):
        collect_users = self.getUserDataDB()
        findResult = collect_users.find(judage,filed)
        findResultList = list(findResult)
        print(len(findResultList),' docs queried.')
        return findResultList 

    # {'username':''}
    def queryEvaluations(self,queryField={}):
        collect_evaluations = self.getEvaluationDataDB()
        findResult = collect_evaluations.find({},queryField)
        findResultList = list(findResult)
        print(len(findResultList),' docs queried.')
        return findResultList 
    #-------------- update -------------------
    def updateRawAPKForest_Cluster(self,content):
        # content: [id,{"$set":{'xxx':xx}}] , [...]
        count = 0
        collect_rawapktree = self.getRawApkForestDB()
        for line in content:
            queryFilter = {"_id":line[0]}
            updateValue = {"$set":{"image_cluster_no":line[1]}}
            collect_rawapktree.update_one(queryFilter,updateValue)
            count += 1
        print(count,' docs updated.')
                
    def updateRawAPKForest_Suspect(self,content):
        # content: [id,{"$set":{'xxx':xx}}] , [...]
        count = 0
        collect_rawapktree = self.getRawApkForestDB()
        for line in content:
            queryFilter = {"_id":line[0]}
            updateValue = {"$set":{"suspect":line[1]}}
            collect_rawapktree.update_one(queryFilter,updateValue)
            count += 1
        print(count,' docs updated.')
    # ------------------collect_apkforest-----------------
    # 一次是save一个控件，如何被读出的_id未被更改，重新插入则是更新
    def saveRawAPKTree(self,content,_id=None):
        collect_rawapkforest = self.getRawApkForestDB()
        if _id !=None:
            content['_id'] = _id
        insertResult = collect_rawapkforest.save(content)
        #print(len(content.keys()),' keys in the doc.')
        return insertResult

    def saveRawAPKForest(self,content):
        count = 0
        for app,widget in content.items():
            for widgetName,widgetProperty in widget.items():
                widgetProperty['app'] = app
                widgetProperty['widget'] = widgetName
                self.saveRawAPKTree(widgetProperty)
                count += 1
        print(count,' docs saved.')

    def saveRawAPKForest_Cluster(self,content):
        count = 0
        collect_rawapktree = self.getRawApkForestDB()
        for line in content:
            #由于获得的信息都是从raw中得到的，所以id也是raw_id
            tree = {'_id':line[0],'image_cluster_no':line[1]}
            self.saveRawAPKTree(tree)
            count += 1
        print(count,' docs saved.')

    # 最新的(2020.9.29)直接调用这个插入每一条json就可以了
    def saveAPKTree(self,content,_id=None):
        collect_apkforest = self.getApkForestDB()
        if _id !=None:
            content['_id'] = _id
        insertResult = collect_apkforest.save(content)
        #print(len(content.keys()),' keys in the doc.')
        return insertResult

    def saveAPKForest(self,content):
        count = 0
        collect_apktree = self.getApkForestDB()
        for line in content:
            #由于获得的信息都是从raw中得到的，所以id也是raw_id
            tree = {'raw_id':line[0],'cluster_no':line[1]}
            self.saveAPKTree(tree)
            count += 1
        print(count,' docs saved.')

    def saveUsers(self,content):
        collect_users = self.getUserDataDB()
        insertResult = collect_users.save(content)
        print(len(content.keys()),' keys in the doc.')
        return insertResult

    def saveEvaluations(self,content):
        collect_users = self.getUserDataDB()
        insertResult = collect_users.save(content)
        print(len(content.keys()),' keys in the doc.')
        return insertResult

    #-------------- update -------------------
    def updateAPKForest(self,content):
        # content: [id,{"$set":{'xxx':xx}}] , [...]
        count = 0
        collect_apktree = self.getApkForestDB()
        for line in content:
            #由于获得的信息都是从raw中得到的，所以id也是raw_id
            queryFilter = {"raw_id":line[0]}
            updateValue = line[1]
            collect_apktree.update_one(queryFilter,updateValue)
            count += 1
        print(count,' docs saved.')

    def queryAPKTree(self,queryFilter={},queryField={}):
        # 根据条件筛选,_id包括在findFilter之中即可
        collect_apktree = self.getApkForestDB()
        findResult = collect_apktree.find(queryFilter,queryField).sort("raw_id",1)
        #findResult = collect_apktree.find(queryFilter).sort("raw_id",1)
        findResultList = list(findResult)
        print(len(findResultList),' docs queried.')
        return findResultList 

    def queryRawAPKTree(self,queryFilter={},queryField={}):
        # 根据条件筛选,_id包括在findFilter之中即可
        collect_rawapkforest = self.getRawApkForestDB()
        findResult = collect_rawapkforest.find(queryFilter,queryField).sort("_id",1)
        #findResult = collect_rawapkforest.find(queryFilter).sort("_id",1)
        findResultList = list(findResult)
        print(len(findResultList),' docs queried.')
        return findResultList 

    def queryAllPath(self):
        # 查询所有数据的部分列
        # field = 'api'
        # field = 'path'
        idUPath =  self.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}},{'path':1})
        for i in range(len(idUPath)):
            idUPath[i] = [idUPath[i]['_id'],idUPath[i]['path']]
        #print(len(idUPath),' docs queried.')
        return idUPath

    def queryAllApis(self):
        # 查询所有数据的部分列
        # field = 'api'
        # field = 'path'
        # 使用频率
        #idUApi = self.queryRawAPKTree({},{'api':1,'api_freq':1})
        idUApi = self.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}},{'package_api':1,'class_api':1,'method_api':1})
        for i in range(len(idUApi)):
            #idUApi[i] = [idUApi[i]['_id'],idUApi[i]['api'],idUApi[i]['api_freq']]
            idUApi[i] = [idUApi[i]['_id'],idUApi[i]['package_api'],idUApi[i]['class_api'],idUApi[i]['method_api']]
        #print(len(idUApi),' docs queried.')
        return idUApi

    def queryAllApi(self):
        # 查询所有数据的部分列
        # field = 'api'
        # field = 'path'
        # 使用
        #idUApi = self.queryRawAPKTree({},{'api':1,'api_freq':1})
        idUApi = self.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}},{'method_api':1})
        for i in range(len(idUApi)):
            #idUApi[i] = [idUApi[i]['_id'],idUApi[i]['api'],idUApi[i]['api_freq']]
            idUApi[i] = [idUApi[i]['_id'],idUApi[i]['method_api']]
        #print(len(idUApi),' docs queried.')
        return idUApi

    def queryAllApiNums(self):
        idUApi = self.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}},{'raw_api_len':1,'filtered_api_len':1})
        for i in range(len(idUApi)):
            #idUApi[i] = [idUApi[i]['_id'],idUApi[i]['api'],idUApi[i]['api_freq']]
            idUApi[i] = [idUApi[i]['_id'],idUApi[i]['raw_api_len'],idUApi[i]['filtered_api_len']]
        #print(len(idUApi),' docs queried.')
        return idUApi

    def queryAllDoc(self):
        # 查询所有数据的部分列
        # field = 'api'
        # field = 'path'
        # 使用频率
        #idUApi = self.queryRawAPKTree({},{'api':1,'api_freq':1})
        idUDoc = self.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}},{'doc':1})
        for i in range(len(idUDoc)):
            #idUDoc[i] = [idUDoc[i]['_id'],idUDoc[i]['api'],idUDoc[i]['api_freq']]
            idUDoc[i] = [idUDoc[i]['_id'],idUDoc[i]['doc']]
        #print(len(idUDoc),' docs queried.')
        return idUDoc

    def queryAllCluster(self):
        # 查询所有数据的部分列
        # field = 'api'
        # field = 'path'
        idUcluster = self.queryAPKTree({},{'raw_id':1,'cluster_no':1})
        for i in range(len(idUcluster)):
            idUcluster[i] = [idUcluster[i]['raw_id'],idUcluster[i]['cluster_no']]
        print(len(idUcluster),' docs queried.')
        return idUcluster

    def queryAllOutlier(self):
        # 查询所有数据的部分列
        # field = 'api'
        # field = 'path'
        idUcluster = self.queryAPKTree({},{'cluster_no':1,'outlier_score':1})

        queryResult = self.queryMetaData({'clusters':1})
        clusters = int(queryResult[0]['clusters'])
        outlierList = []
        # 遍历簇号，逐个生成列表
        for cluster in range(clusters):
            curOutlierList = []
            for item in idUcluster:
                # 如果返回的簇信息中符合当前簇号
                if int(item['cluster_no']) == cluster:
                    curOutlierList.append(float(item['outlier_score']))
            #idUcluster[i] = [idUcluster[i]['_id'],int(idUcluster[i]['cluster']),float(idUcluster[i]['outlier'])]
            # 以簇为索引,添加outlier，按id排序好
            outlierList.append(curOutlierList)
        return outlierList

    #------------------DELETE-------------------------
    def deleteProject(self):
        # delete metedata中的记录
        collect_metadata = self.getMetaDataDB()
        deleteResult = collect_metadata.delete_one({"subtask_name":self.subtask_name})
        print('删除metdata结果:',deleteResult.deleted_count)
        print('删除的项目所属项目:',self.project_name)
        print('删除的项目subtask名:',self.subtask_name)

        # drop subtask对应的apktree
        client = self.getConnection()
        collist = client.collection_names()
        apkTreeTableName =self.project_name+'__'+self.subtask_name+'_apktree'
        print('删除表格:',self.apkTreeTableName , client[apkTreeTableName])
        deleteResult2 = client[apkTreeTableName].drop()
        print('drop apkTree结果:',deleteResult2)

        return (deleteResult.deleted_count,deleteResult2)

    def deleteUser(self,_id):
        collect_users = self.getUserDataDB()
        deleteResult = collect_users.delete_one({'_id':_id})
        return deleteResult
