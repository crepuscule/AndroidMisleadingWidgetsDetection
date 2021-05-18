import sys,os
sys.path.append(os.path.abspath('..'))                                                                                         
from Base import BaseProcessor  

WRITE = True
DEBUG = False

# 适用于csvs格式数据
class CSVSetsTransformer(BaseProcessor.BaseProcessor):
    def __init__(self,config,project_name,subtask_name):
        super(CSVSetsTransformer,self).__init__()
        sys.path.append(os.path.abspath('..'))                                                                                         
        from Base import BaseProcessor, DataBaseProcessor
        self.DBP = DataBaseProcessor.DataBaseProcessor()

        self.config = config
        self.whiteList = self.readTxt(self.config['WHITE_LIST_PATH'])
        self.DBP.project_name = project_name
        self.DBP.subtask_name = subtask_name
        self.DBP.configTableName = self.DBP.project_name+'_config'                                                       
        self.DBP.metaDataTableName = self.DBP.project_name + '_metadata'                                                 
        self.DBP.apkTreeTableName = self.DBP.project_name +'__'+ self.DBP.subtask_name+'_apktree'

    def info(self):
        info='''
    Describe:
        (当前的数据集替代Transformer的模块)
        功能: 将csv s记录转化为dict树，存储更方便
        transform csv records to dicts
 
        输入：有意义记录集合
        feed: csv dirs , pic dirs
 
        输出：apkTree,apkPathTree
        output: apkTree,apkPathTree

        更新：目前使用开发集DEV_SET做集合
 
    Methods list:
            def setConfig(self,config);
            @def transformToapkTree(self,csvPath):
            def writeDict(path,content):
            @def runTransform(self,operator='',params=[]):
                if operator == 'info':
                if operator == 'apktree':
    Example:
        T apktree dev
        T apktree train
        T apktree csvsets
    Problem:
        1. 应该也有能够选择数据集的功能，这里绕过去了(具体数据未知，暂不开发)
        2. 已经实现apkTree和apkPathTree合成一体
        '''
        print(info)
    # 思路1：直接读取各csv，添加到一个apktree中和apkpathtree中
    # 思路2：将其转化为之前的csv形式，然后利用之前已经成熟的方式提取

    # 思路1:
    # # 难题1: handler - api 和 widget - handler 是两棵树\
    # 指定目录后从中提取文件列表用于比对
    #super.getPictureListsFromDir
    def getPictureLists(self,path):
        import os
        pictureLists = []
        pathLists = []
        x = os.walk(path)
        for path,d,filelist in x:
            for filename in filelist:
                apkName = path.split('/')[-1]
                # 为了防止图片有多个'.'
                picName = filename.split('.')[:-1]
                picName = '.'.join(picName)
                pictureLists.append([apkName,picName])
               
                # 该路径从图像path的根目录开始
                relativepath = apkName+'/'+filename
                pathLists.append(relativepath)
        return pictureLists,pathLists

    def getPicturePath(self,pictureLists,pathLists,widgetInfo):
        picNO = pictureLists.index(widgetInfo)
        return pathLists[picNO]

    def APIFilter(self,api,filterType='white'):
        # 选择白名单
        if filterType == 'white':
            apiList = self.readTxt( self.config['WHITE_API_LIST_PATH'] )
            if api in apiList:
                return True
        # 选择黑名单
        elif filterType == 'black':
            apiList = self.readTxt( self.config['BLACK_API_LIST_PATH'] )
            if api not in apiList:
                return True
        elif filterType == 'simply':
            if api[:9] == '<android.':
                return True
        # 选择
        return False

    def transformToapkTreeFromNewCSV(self,csvPath,pictureLists,pathLists): 
        apiSet = set()
        apkTree = dict()     
        handlerTree = dict()

        global flag
        flag = ''
        # 读入一个csv文件，即一个apk
        with open(csvPath) as f:
            # 分析每一行
            for line in f:
                if line == 'APK	Image	WID	WID Name	Layout	Handler	Method	Lines	Permissions\n':
                    continue
                elif line == '---- line to handler ----\n':
                    flag = 'widget'
                    continue
                elif line == '---- handler to API ----\n':
                    flag = 'api'
                    continue

                # 开始处理控件，一行是一个控件,也就是说这一次就是一个控件,只是有可能重复（因为名字重复,而控件号不同的原因）
                if flag == 'widget':
                    # 提取widget的信息列表和它对应的handler列表
                    widgetInfo , handlerList = line.split('->')
                    widgetInfo = (widgetInfo.strip(' ').strip('\t')).split('\t')
                    # 首先它不能是已被淘汰图片的控件
                    if widgetInfo[:2] not in pictureLists:
                        continue
                    # 没被淘汰的话，继续处理
                    handlerList = (handlerList.strip('\n').strip(' ').strip(']').strip('[')).split(', ')
                    #print('widgetInfo -> handlerList',widgetInfo,'\n',handlerList)
                    # 这样一个csv或者一个apk中的所有控件列表都在这里了
                    # 只需要按apk名称填入dict即可
                    if widgetInfo[0] not in apkTree:
                        apkTree[widgetInfo[0]] = dict()
                        #apkPathTree[widgetInfo[0]]=dict()
                    # 存储对应图像的地址,图像地址直接到rawdata中找就行了,主要就为了一个后缀名
                    nameno = widgetInfo[1]+','+widgetInfo[2]
                    #nameno = widgetInfo[1]
                    #apkPathTree[widgetInfo[0]][nameno]=self.getPicturePath(pictureLists,pathLists,widgetInfo[:2])

                    # 此时apk一定已经注册，看image有无注册,这个handlerlist赋值只发生一次，一个控件的handler都在这个列表里面了
                    if nameno not in apkTree[widgetInfo[0]]:
                        apkTree[widgetInfo[0]][nameno] = dict()
                        apkTree[widgetInfo[0]][nameno]['handler'] = set()
                        apkTree[widgetInfo[0]][nameno]['api'] = set()
                        apkTree[widgetInfo[0]][nameno]['handler'].update(handlerList) 
                        apkTree[widgetInfo[0]][nameno]['path'] = self.getPicturePath(pictureLists,pathLists,widgetInfo[:2])
         
                elif flag == 'api':
                    handler , api = line.split('->')
                    handler = handler.strip(' ')
                    api = api.strip('\n').strip(' ') 

                    if self.APIFilter(api,'simply') == False:
                        continue
                    #print('handler -> api',handler,'\n',api)
                    # 将handler当作一个树，内部api全部无重复地存入其中
                    if handler in handlerTree:
                        handlerTree[handler].add(api)
                    else:
                        handlerTree[handler] = set()
                        handlerTree[handler].add(api)

        #return apkTree,apkPathTree,handlerTree
        return self.concatTree(apkTree,handlerTree)

    def concatTree(self,apkTree,handlerTree):
        #print(handlerTree.keys())
        for key,value in apkTree.items():
            for widgetName,handlerValue in value.items():
                for handler in handlerValue['handler']:
                    if handler in handlerTree.keys():
                        apkTree[key][widgetName]['api'].update(handlerTree[handler])
                    else:
                        # 有些控件的handler没有给出？？
                        pass
        for key,value in list(apkTree.items()):
            for widgetName in list(value.keys()):
                apis = list(apkTree[key][widgetName]['api'])
                if len(apis) == 0:
                    apkTree[key].pop(widgetName)
                    #apkPathTree[key].pop(widgetName)
                    if len(apkTree[key])==0:
                        apkTree.pop(key)
                        #apkPathTree.pop(key)
                    continue
                apkTree[key][widgetName].pop('handler')
                apkTree[key][widgetName]['api'] = apis
                apkTree[key][widgetName]['api'].sort()
        return apkTree

    def treeStruct(self,tree,level=1):
        global DEBUG
        widgetNum = 0
        print('level 1')
        count= 0
        for key,value in tree.items():
            count += 1
            if DEBUG: print(key,':',type(value))
        print('level 1 total: ',count)
        print('--------------------------\n')

        if level >= 2:
            apiSet = set()
            handlerSet = set()
            count= 0
            print('level 2')
            for key,value in tree.items():
                if DEBUG: print(key,'::',type(value))
                for k,v in value.items():
                    count += 1
                    handlerSet.add(k)
                    apiSet.update(tree[key][k])
                    if DEBUG: print('%3d'%count,end='')
                    if DEBUG: print('...',k,":",type(v))
            print('level 2 total: ',count)
            widgetNum = count
            print('apiSet len:',len(apiSet))
            print('handlerSet len:',len(handlerSet))
            print('-------------lv2-------------\n')
        
        if level >= 3:
            count= 0
            print('level 3')
            for key,value in tree.items():
                if DEBUG: print(key,':::',type(value),'\n')
                for k,v in value.items():
                    if DEBUG: print('...',k,"::",type(v))
                    if DEBUG: print(v)
                    #for i,j in v.items():
                    #    count += 1
                    #    print('%4d'%count,end='')
                    #    print('......',i,":",type(j))
                    if DEBUG: print('')
            print('level 3 total: ',count)
            print('---------------lv3-----------\n')
        return widgetNum

    def transformToapkTree(self,csvDir,picDir):
        import os
        from collections import OrderedDict
        apkTree = dict()
        apiSet = set()
        #apkPathTree = dict()

        pictureLists,pathLists = self.getPictureListsFromDir(picDir)
        csvdirs = os.listdir(csvDir)
        count = 0
        for csvPath in csvdirs:
            count += 1
            #oneapkTree,oneapkPathTree,handlerTree = self.transformToapkTreeFromNewCSV(csvDir+csvPath,pictureLists,pathLists)
            #newoneapkTree = self.concatTree(oneapkTree,handlerTree)
            oneapkTree = self.transformToapkTreeFromNewCSV(csvDir+csvPath,pictureLists,pathLists)
            #apkTree.update(newoneapkTree)
            apkTree.update(oneapkTree)
            #apkPathTree.update(oneapkPathTree)
        return OrderedDict(apkTree)

    '''
    def writeDict(self,path,content):
        import os
        pathDir = "/".join(path.split('/')[:-1])
        if not os.path.exists(pathDir):
            print('making dir:',pathDir)
            os.makedirs(pathDir)
        writefile = open(path,mode='w')
        import json
        writefile.write(json.dumps(content))
        writefile.close()

    def readDict(self,path):
        readfile = open(path,mode='r')
        import json
        content = readfile.read()
        readfile.close()
        return json.loads(content)
    '''

    def runTransformer(self,operator='',params=[],config=[]):
        global WRITE
        global DEBUG
        runLog = []
        if operator == 'info' or operator == '':
            self.info()
        if operator == 'apktree':
            # 如果已经有rawapktree了，那就不再运行
            rawdata = self.DBP.queryRawAPKTree()
            if len(rawdata) > 0:
                print('/\/\ rawdata exists, skipping.')
                return runLog
            #?# 使用前注意检查有没有！！！
            if params == 'dev':
                self.config['CURRENT_SET_PATH'] = self.config['DEV_SET_PATH']
            elif params == 'train':
                self.config['CURRENT_SET_PATH'] = self.config['TRAIN_SET_PATH']
            elif params == 'val':
                self.config['CURRENT_SET_PATH'] = self.config['VAL_SET_PATH']
            elif params == 'test':
                self.config['CURRENT_SET_PATH'] = self.config['TEST_SET_PATH']
            elif params == 'csvsets':
                self.config['CURRENT_SET_PATH'] = self.config['UNIVERSAL_RECORDS_DIR']
            else:
                self.config['CURRENT_SET_PATH'] = self.config['MEANINGFUL_RECORDS_PATH']
            runLog.append(params)

            apkTree = self.transformToapkTree(self.config['CURRENT_SET_PATH'],self.config['PICTURES_DIR'])
            runLog.append('CURRENT_SET_PATH:%s' % self.config['CURRENT_SET_PATH'] )
            runLog.append('PICTURES_DIR:%s' % self.config['PICTURES_DIR'] )

            #self.treeStruct(apkPathTree,2)
            widgetNum = self.treeStruct(apkTree,2)
            if WRITE == True:
                #self.writeDict(apkTree,self.config['APK_TREE_PATH'])
                self.DBP.saveRawAPKForest(apkTree)
                self.DBP.updateMetaData({"$set":{"widgets":widgetNum}})
                #self.writeDict(apkPathTree,self.config['APK_PATH_TREE_PATH'])
                runLog.append('apkTree length: %d' % len(apkTree))
                #runLog.append('apkPathTree length: %d' % len(apkPathTree))
                runLog.append('widgetNum: %d' % widgetNum)
            #print('-------')
            #newapkTree = readDict('apkTree.json')
            #newapkPathTree = readDict('apkPathTree.json')
            #treeStruct(newapkTree,2)
            #treeStruct(newapkPathTree,2)
        print('\n---------------------------runLog:----------------------------\n',runLog)
        return runLog
