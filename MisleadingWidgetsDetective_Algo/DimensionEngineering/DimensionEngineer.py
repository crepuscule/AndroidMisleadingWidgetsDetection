import sys,os
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor 

W2V_MODEL = None

class DimensionEngineer(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(DimensionEngineer, self).__init__()
        sys.path.append(os.path.abspath('..'))
        self.config = config
        self.DBP = DBP
    def info(self):
        info='''
        if operator == 'vector':
        if operator == 'cutdim':
            def dimensionReduction(self,method,vector,ratio='0.5'):
                if method == 'pca':
                elif method == 'lle':
        if operator == 'statistics':

        e.g:
            DE vector
            DE cutdim pca,0.5
            DE cutdim raw
        '''
        print(info)

    def transcription(self): 
        import numpy as np 
        idUapi = self.DBP.queryAllApis() 
        package_apis = np.array(idUapi)[:,1] # 取半条链作为mRNA 
        class_apis = np.array(idUapi)[:,2] # 取半条链作为mRNA 
        method_apis = np.array(idUapi)[:,3] # 取半条链作为mRNA 
        return list(package_apis),list(class_apis),list(method_apis)

    def doc_transcription(self):
        import numpy as np 
        idUdoc = self.DBP.queryAllDoc() 
        docs = np.array(idUdoc)[:,1] # 取半条链作为mRNA 
        return list(docs)

    def get_word_vector(self,word):
        global W2V_MODEL
        w2v_model = W2V_MODEL
        if word in w2v_model.index2word:
            vec = w2v_model[word]
            return vec
        else:
            return []

    # 接下来直接累加求平均/求最大来构造段落向量
    def doc_vector(self,docs):
        import math
        import numpy as np
        # 首先拿过来的东西是这样的:
        #idUdoc ('view view','net uri',...,'view find view by id')
        #[['location location manager', 'os handler', 'text format date utils', 'view view', 'widget text view'], 
        # ['app shared preferences impl', 'graphics bitmap', 'graphics color', , 'widget toast']]
        print(docs)
        widgetAPIDocVectors = list()
        for line in docs:
            # line => ['location location manager', 'os handler',... ]
            global W2V_MODEL
            lineDocVectors = np.zeros( (len(W2V_MODEL['id']),) )
            count = 0
            for api in line:
                # api => 'location location manager'
                words = api.split(' ')
                for word in words:
                   # word => 'location'
                   word_vector = self.get_word_vector(word)
                   # print(word_vector,'\n\n',type(word_vector),'\n\n',len(word_vector))
                   if word_vector != []:
                       lineDocVectors += word_vector
                       count += 1
            lineDocVectors /= count
            widgetAPIDocVectors.append(list(lineDocVectors))

        return widgetAPIDocVectors
        
    '''
    def transcription(self): 
        import numpy as np 
        idUapi = self.DBP.queryAllApi() 
        apis = np.array(idUapi)[:,1] # 取半条链作为mRNA 
        #api_freqs = np.array(idUapi)[:,2] # 取半条链作为mRNA 
        #return list(apis),list(api_freqs)
        return list(apis)
    '''
    def generalEncoder(self,raw_list):
        # 假设任意一种字符串，需要进行编码
        # 首先拿过来的东西是这样的:
        #idUdoc ('view view','net uri',...,'view find view by id')
        #[['location location manager', 'os handler', 'text format date utils', 'view view', 'widget text view'], 
        # ['app shared preferences impl', 'graphics bitmap', 'graphics color', , 'widget toast']]
        pass        

        
    #def vector(self,apis,api_freqs):
    # 只需要将分别的包，类，方法进行编码，注意，类和方法需要全，前面包不需要全
    def generalvector(self,apis):
        import math
        import numpy as np
        apiSet = set()
        for line in apis:
            apiSet.update(line)

        apiSet = list(apiSet)
        apiSet.sort()
        apiSetLen = len(apiSet)
        apiNo = [i for i in range(len(apiSet))]
        #apiDicts => 'api1':1,'api2':2
        apiDicts = dict(zip(apiSet,apiNo))
        print('apiSet len',apiSetLen)

        widgetAPIVectors = []
        '''
        # 每一行就是一个控件，所以用i就可以定位到相应的api和freq
        for i in range(len(apis)):
            widgetAPIVector = [0]*apiSetLen
            # 仅用于计数
            # 这个api就是每个控件中的api的位置
            for j in range(len(apis[i])):
                api = apis[i][j]
                # one-hot 编码
                # 在api所在位置上(即apiDicts的值),插入1
                # 现在则插入对应控件对应的其tf*idf值??
                widgetAPIVector[apiDicts[api]] = api_freqs[i][j]
            # 一行检查完毕之后写入
            widgetAPIVectors.append(widgetAPIVector) 
        # 顺序与之前完全相同
        '''
        for line in apis:
            widgetAPIVector = [0]*apiSetLen
            # 仅用于计数
            for api in line:
                # one-hot 编码
                # 在api所在位置上(即apiDicts的值),插入1
                widgetAPIVector[apiDicts[api]] = 1
            # 一行检查完毕之后写入
            widgetAPIVectors.append(widgetAPIVector) 
	
	# 由于vector已经生成好了，所以直接就按列加起来就能计算IDF值了，有几个就是几不就行了嘛？
        api_appear_idf = np.sum(widgetAPIVectors,axis=0,dtype='float32')
        print('before api_appear_idf:',api_appear_idf)
	# widget_num => |D| 
        widget_num = len(widgetAPIVectors)
        print('widget_num:',widget_num)
        # 1 + all_api_dict[i] => 1+|j: t_i in d_j|  包含词t_i的文档数量
        for i in range(len(api_appear_idf)):
            #api_appear_idf[i]  = math.log(( 1 + widget_num )/(1 + api_appear_idf[i])) + 1
            api_appear_idf[i]  = (( 1 + widget_num )/(1 + api_appear_idf[i])+ 1 ) * 0.01
	# 然后这个东西同样保存起来和那个向量一起，用得时候读取就行了,顺序也和这个一样
        # 顺序与之前完全相同
        print('after api_appear_idf:',api_appear_idf)
        # 存储api的文本信息，好对比
        self.writeTxt(apiSet,self.config['API_VECTORS_PATH'].replace('.txt','apinames.txt'))
        #self.writeTxt(apiSet,self.config['API_VECTORS_PATH'].replace('.txt','-apinames.txt'))
        if self.config['VERSION'] == '2':
            self.writeVector(api_appear_idf,self.config['API_VECTORS_PATH'].replace('APIVector-','APIIDF-')+'.txt',fmt="%f") # 原先是d
        else:
            self.writeVector(api_appear_idf,self.config['API_IDF_PATH']+'.txt')
        return widgetAPIVectors,api_appear_idf

    #vector:根据apkTree将所有数据进行0-1 向量化或带权向量化
    def vectorOld(self,apkTree):
        apiSet = set()
        for app,widgets in apkTree.items():
            for widgetName in widgets.keys():
                apiSet.update(apkTree[app][widgetName]['api'])
        
        apiSet = list(apiSet)
        apiSet.sort()
        apiNo = [i for i in range(0,len(apiSet))]
        print('apiSet len',len(apiSet))
        
        apiDicts = dict(zip(apiSet,apiNo))
        widgetNames = []
        widgetAPIVectors = []
        for app,widgets in apkTree.items():
            # 这是一个控件
            for widgetName in widgets.keys():
                widgetAPIVector = [0]*len(apiDicts)
                for api in apkTree[app][widgetName]['api']:                
                    widgetAPIVector[apiDicts[api]] = 1
                widgetAPIVectors.append(widgetAPIVector) 
                widgetNames.append(app+','+widgetName)
                            
        return apiSet,widgetAPIVectors,widgetNames

    def cutDimension(self,blackListPath,apiSet,widgetAPIVectors):
        if blackListPath=='':
            return apiSet,widgetAPIVectors

        #增加筛选功能
        import numpy as np
        
        blackListFile = open(blackListPath,'r')
        blackList = blackListFile.readlines()
        blackListFile.close()
        
        matchAPISet = []
        for item in apiSet:
            item = item.strip('<').strip('>')
            classitem = item.split(": ")[0]
            methoditem = item.split(": ")[1]
            methoditem = methoditem.split(" ")[-1]
            matchAPISet.append(classitem+'.'+methoditem)
        
        whiteIndex = [i for i in range(0,len(matchAPISet))]
        blackIndex = []
        for item in blackList:
            item = item.strip('\n')
            if item in matchAPISet:
                blackIndex.append(matchAPISet.index(item))
                print('black:',item)        
        
        # 这个是白名单，按照它来找需要的维度
        whiteIndex = list(set(whiteIndex) ^ set(blackIndex))
        
        # 这个是新维度名
        newAPISet = np.array(apiSet)[whiteIndex]
        
        # 这是是新维度下的向量
        widgetAPIVectors = np.array(widgetAPIVectors)
        newAPIVector = np.array([widgetAPIVectors[0][whiteIndex]])
        for i in range(1, len(widgetAPIVectors)): 
            newAPIVector = np.concatenate((newAPIVector, [widgetAPIVectors[i][whiteIndex]]), axis = 0)
        
        return newAPISet,newAPIVector

    def dimensionReduction(self,vector,method,ratio=0.5):
        import numpy as np
        vector = np.array(vector)
        print(vector.shape)
        print(len(vector))
        # 如果是0.x，则是比率
        if ratio < 1:
            originDim = vector.shape[1]
            reducedDim = int(originDim * ratio)
        # 如果不是，则就是具体多少维了
        else:
            reducedDim = int(ratio)
        if reducedDim > vector.shape[0]:
            reducedDim = vector.shape[0]
        if method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=reducedDim)
            pca.fit(vector)
            reducedVector = pca.transform(vector)
        elif method == 'lle':
            from sklearn import manifold 
            reducedVector =  manifold.LocallyLinearEmbedding(n_neighbors = reducedDim//2, n_components = reducedDim,method='standard').fit_transform(vector)
        elif method == 'kpca':
            from sklearn.decomposition import KernelPCA
            #pca = KernelPCA(kernel='cosine',n_components=reducedDim)
            pca = KernelPCA(kernel='rbf',n_components=reducedDim)
            pca.fit(vector)
            reducedVector = pca.transform(vector)
        elif method == 'svd':
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=reducedDim, n_iter=7, random_state=42)
            svd.fit(vector)
            # svd贡献率
            print(svd.explained_variance_ratio_)
            reducedVector = svd.transform(vector)
        elif method == 'iso':
            from sklearn import manifold 
            isomap = manifold.Isomap( n_components=reducedDim, n_neighbors=reducedDim//2)
            reducedVector = isomap.fit_transform(vector)
        elif method == 'tsne':
            from sklearn import manifold 
            tsne = manifold.TSNE(n_components=reducedDim, learning_rate=100,random_state=42,method='exact')
            reducedVector = tsne.fit_transform(vector)
        elif method == 'spe':
            from sklearn import manifold 
            embedding = manifold.SpectralEmbedding(n_components =reducedDim,random_state=42)
            reducedVector = embedding.fit_transform(vector)

        print('after dim cut:',reducedVector.shape[1])
        return reducedVector

    def saveAPIVector(self,apiSet,saveAPIColumnPath,widgetAPIVectors,savePath,widgetNames,saveAPINamePath):
        self.makeSureExists(saveAPIColumnPath)
        self.makeSureExists(saveAPINamePath)
        self.makeSureExists(savePath)
        import numpy
        self.writeTxt(apiSet,saveAPIColumnPath)
        numpy.savetxt(savePath,widgetAPIVectors,fmt="%d")
        print(len(widgetAPIVectors),' records writen.')
        self.writeTxt(widgetNames,saveAPINamePath)

    
    def multiLevelAPIVecotr():
        # 首先需要对三个层次的进行分别编码，然后拼在一起
        pass

    
    #runCluster:运行最佳参数的聚类
    def runEngineer(self,operator=[],params=[]):
        runLog = []
        runLog.append(operator)
        global W2V_MODEL

        if operator == 'test':
            #api = self.transcription()
            #print(api)
            api,api_freqs = self.transcription()
            print(api,len(api),'\n',api_freqs,len(api_freqs))
        if operator == 'info' or operator=='':
            self.info()
        if operator == 'statistics':
            import numpy as np
            idUapi = self.DBP.queryAllApiNums()
            raw_api_nums = np.array(idUapi)[:,1] # 取出所有原始API数量
            filtered_api_nums = np.array(idUapi)[:,2] # 取所有过滤后的API数量
            self.writeVector(raw_api_nums,self.config['API_VECTORS_PATH'].replace('.txt','_raw_api_nums.txt'))#+'-'+params[0]+params[1])
            self.writeVector(filtered_api_nums,self.config['API_VECTORS_PATH'].replace('.txt','_filtered_api_nums.txt'))#+'-'+params[0]+params[1])
        # 仅有method方法的api，具有配套的apiidf计算`
        if operator == 'mvector':
            #apis,api_freqs = self.transcription()
            _,_,apis = self.transcription()
            #widgetAPIVectors = self.vector(apis,api_freqs)
            widgetAPIVectors,api_appear_idf = self.generalvector(apis)
            #/data/wangruifeng/datasets/DroidBot_Epoch/generated_data/dbUniversalSetPureapkforest_EvaluateMethods/DimensionEngineering/APIIDF
            #/data/wangruifeng/datasets/DroidBot_Epoch/generated_data/dbUniversalSetPureapkforest_EvaluateMethods/DimensionEngineering/APIVector-fwbgbw1spm_puremethodapi.txt
            if self.config['VERSION'] == '2':
                self.writeVector(widgetAPIVectors,self.config['API_VECTORS_PATH'].replace('.txt','-raw.txt'),fmt="%f") # 原先是d
                self.writeVector(api_appear_idf,self.config['API_VECTORS_PATH'].replace('APIVector-','APIIDF-')+'.txt',fmt="%f") # 原先是d
            else:
                self.writeVector(widgetAPIVectors,self.config['API_VECTORS_PATH'].split('-')[0]+'-raw.txt',fmt="%f") # 原先是d
                self.writeVector(api_appear_idf,self.config['API_IDF_PATH'].split('-')[0]+'.txt',fmt="%f") # 原先是d

            #reducedVector = self.dimensionReduction(widgetAPIVectors,'tsne',2)
            #self.writeVector(reducedVector,self.config['API_VECTORS_PATH'].split('-')[0]+'-visualize.txt',fmt="%f") # 原先是d
            '''
            apkTree = self.readDict(self.config['APK_TREE_PATH'])
            apiSet,widgetAPIVectors,widgetNames = self.vector(apkTree)
            self.saveAPIVector(apiSet,self.config['API_VECTORS_COLUMN_PATH'],
                            widgetAPIVectors,self.config['API_VECTORS_PATH'],
                            widgetNames,self.config['API_VECTORS_NAME_PATH'])
            '''
            #update({"projectname":''},{'widgets':},{'rawdims'},{'cutdimway'},{'reducedims'},{'ifmethod'},{'ifdim/ifsize'},{'clusterm|~                             
                #ethod'},{'clusters'},{'clusterindex'},{'outliermethod'},{})
            self.DBP.updateMetaData({"$set":{"encodermethod":'apimethod-Features',"rawdims":len(widgetAPIVectors[0]),"record_len":len(widgetAPIVectors)}})
            runLog.append('widgetAPIVectors len: %d ' % (len(widgetAPIVectors)))            
        if operator == 'vector':
            #apis,api_freqs = self.transcription()
            #from gensim.models import word2vec
            #W2V_MODEL = word2vec.Word2Vec.load('/data/wangruifeng/datasets/corpus/Word2Vec_Android-master/data/android.word2vec.model')
            from gensim.models import  KeyedVectors
            #W2V_MODEL = KeyedVectors.load_word2vec_format('/data/wangruifeng/datasets/corpus/Word2Vec_Android-master/data/android.en.text.vector',binary=False)
            W2V_MODEL = KeyedVectors.load_word2vec_format('/data/wangruifeng/datasets/android.en.text.vector',binary=False)
            # 调用word2vec模型将doc字段向量化并存储
            docs = self.doc_transcription()
            package_apis,class_apis,method_apis = self.transcription()
            #widgetAPIVectors = self.vector(apis,api_freqs)
            package_widgetAPIVectors,package_api_appear_idf = self.generalvector(package_apis)
            class_widgetAPIVectors,class_api_appear_idf = self.generalvector(class_apis)
            method_widgetAPIVectors,method_api_appear_idf = self.generalvector(method_apis)
            doc_widgetAPIVectors = self.doc_vector(docs)
            print(len(package_widgetAPIVectors[-1]))
            print(len(class_widgetAPIVectors[-1]))
            print(len(method_widgetAPIVectors[-1]))
            print(len(doc_widgetAPIVectors[-1]))

            #[[p1,p2,...pn,c1,c2...,cn,m1,m2...mn,s1,s2...,sn],
            # [p1,p2,...pn,c1,c2...,cn,m1,m2...mn,s1,s2...,sn]]
            widgetAPIVectors = list()
            for line in range(len(package_widgetAPIVectors)):
                tempVectors = list()
                tempVectors.extend(package_widgetAPIVectors[line])
                tempVectors.extend(class_widgetAPIVectors[line])
                tempVectors.extend(method_widgetAPIVectors[line])
                tempVectors.extend(doc_widgetAPIVectors[line])
                widgetAPIVectors.append(tempVectors)
            print(type(widgetAPIVectors),'\n',len(widgetAPIVectors),'\n',len(widgetAPIVectors[0]),'\n',len(widgetAPIVectors[-1]))
            #self.writeVector(widgetAPIVectors,self.config['API_VECTORS_PATH'])
            #/data/wangruifeng/datasets/DroidBot_Epoch/generated_data/dbUniversalSetPureapkforest_EvaluateMethods/DimensionEngineering/APIVector-fwbgbw1spm_puremethodapi.txt
            # APIVector-spm_FuseAPI2350.txt
            if self.config['VERSION'] == '2':
                self.writeVector(widgetAPIVectors,self.config['API_VECTORS_PATH'].replace('.txt','-raw.txt'),fmt="%f") # 原先是d
            else:
                self.writeVector(widgetAPIVectors,self.config['API_VECTORS_PATH'].split('-')[0]+'-raw.txt',fmt="%f") # 原先是d
            #先不考虑idf了，先直接拿这些特征进行聚类
            #self.writeVector(api_appear_idf,self.config['API_IDF_PATH'].split('-')[0]+'.txt',fmt="%f") # 原先是d

            #reducedVector = self.dimensionReduction(widgetAPIVectors,'tsne',2)
            #self.writeVector(reducedVector,self.config['API_VECTORS_PATH'].split('-')[0]+'-visualize.txt',fmt="%f") # 原先是d
            '''
            apkTree = self.readDict(self.config['APK_TREE_PATH'])
            apiSet,widgetAPIVectors,widgetNames = self.vector(apkTree)
            self.saveAPIVector(apiSet,self.config['API_VECTORS_COLUMN_PATH'],
                            widgetAPIVectors,self.config['API_VECTORS_PATH'],
                            widgetNames,self.config['API_VECTORS_NAME_PATH'])
            '''
            #update({"projectname":''},{'widgets':},{'rawdims'},{'cutdimway'},{'reducedims'},{'ifmethod'},{'ifdim/ifsize'},{'clustermethod'},{'clusters'},{'clusterindex'},{'outliermethod'},{})
            self.DBP.updateMetaData({"$set":{"encodermethod":'multi-Features',"rawdims":len(widgetAPIVectors[0]),"package_api_len":len(package_widgetAPIVectors[0]),"class_api_len":len(class_widgetAPIVectors[0]),"method_api_len":len(method_widgetAPIVectors[0]),"doc_api_len":len(doc_widgetAPIVectors[0]),"record_len":len(widgetAPIVectors)}})
            runLog.append('widgetAPIVectors len: %d ' % (len(widgetAPIVectors)))
        # 输入示例: DE cutdim pca,0.5
        if operator == 'cutdim':
            # 用于新式项目
            if self.config['VERSION'] == '2':
                vector = self.readVector(self.config['API_VECTORS_PATH'].replace('.txt','-raw.txt'))
            else:
                vector = self.readVector(self.config['API_VECTORS_PATH'].split('-')[0]+'-raw.txt')
            if params == 'raw':
                reducedVector = vector
            else:
                params = params.split(',')
                # 如果要降维，则需要先乘上去
                if self.config['VERSION'] == '2':
                    idf = self.readVector(self.config['API_VECTORS_PATH'].replace('APIVector-','APIIDF-')+'.txt') # 原先是d
                else:
                    idf = self.readVector(self.config['API_IDF_PATH'].split('-')[0]+'.txt') # 原先是d
                vector = vector * idf
                reducedVector = self.dimensionReduction(vector,params[0],float(params[1]))
            '''
            # 如果原向量不存在，则创建一个
            if not self.isExist(self.config['API_VECTORS_PATH']+'.total'):
                self.moveFile(self.config['API_VECTORS_PATH'],self.config['API_VECTORS_PATH']+'.total')
            '''
            self.writeVector(reducedVector,self.config['API_VECTORS_PATH'])#+'-'+params[0]+params[1])
            self.DBP.updateMetaData({"$set":{"cutdimway":params[0],"reducedims":params[1]}})
            runLog.append('after cutdim: %d' % len(reducedVector))
        if operator == 'pickdim':
            vector = self.readVector(self.config['API_VECTORS_PATH'])
            params = params.split(',')

        if operator == 'w2v':
            #from gensim.models import word2vec
            #W2V_MODEL = word2vec.Word2Vec.load('/data/wangruifeng/datasets/corpus/Word2Vec_Android-master/data/android.word2vec.model')
            from gensim.models import  KeyedVectors
            W2V_MODEL = KeyedVectors.load_word2vec_format('/data/wangruifeng/datasets/corpus/Word2Vec_Android-master/data/android.en.text.vector',binary=False)
            # 调用word2vec模型将doc字段向量化并存储
            docs = self.doc_transcription()
            widgetAPIDocVectors = self.doc_vector(docs)
            #self.writeVector(,self.config['API_VECTORS_PATH'].split('-')[0]+'-w2v.txt',fmt="%f") # 原先是d
            self.writeVector(widgetAPIDocVectors,self.config['API_VECTORS_PATH'])#+'-'+params[0]+params[1])
            self.DBP.updateMetaData({"$set":{"encodermethod":'w2v-Features',"cutdimway":'w2v',"reducedims":100}})
            runLog.append('after cutdim: %d' % len(widgetAPIDocVectors))
        if operator == 'copy':
            #APIVector-lim20fuse2350_raw.txt
            #lim20fuse3340_raw_hierarch_vgg_if
            copy_from_vector_name = "_".join(params.split('_')[:2]) 
            this_vector_name = "_".join(self.DBP.subtask_name.split('_')[:2]) 
            self.linkFile(self.config['API_VECTORS_PATH'].replace(this_vector_name,copy_from_vector_name),self.config['API_VECTORS_PATH'])
    

        return runLog
