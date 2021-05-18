#!/usr/bin/python
import sys, os                   
import shutil
import traceback
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from Base import BaseProcessor

RAW_ROOT = '/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/'
class Interpreter(BaseProcessor.BaseProcessor):
    # 加载配置文件
    def loadConfiguration(self, update=False):
        for key, value in self.config.items():            
            if key=='_id' or key == 'INSTANCE_NAME':
                continue
            if key == 'WHITE_LIST_PATH':
                self.config[key] = self.BP.nominate((key, value), '', self.DBP.subtask_name)
            if key == 'SPM_CODE_BOOK_PATH':
                self.config[key] = self.BP.nominate(
                    ('CLUSTERER_CONTAINER_PATH', value), 
                    '', 
                    self.DBP.subtask_name)
            if key == 'EXTRACTED_FEATURE_PATH':
                self.config[key] = self.BP.nominate((key, value), 'IF', self.DBP.subtask_name)
            if key == 'API_VECTORS_PATH':
                self.config[key] = self.BP.nominate((key, value), 'DE', self.DBP.subtask_name)
            if key == 'CLUSTERER_CONTAINER_PATH':
                self.config[key] = self.BP.nominate((key, value), 'VC', self.DBP.subtask_name)
            if key == 'CLUSTER_PICTURE_RESULT_DIR':
                self.config[key] = self.BP.nominate((key, value), 'VC', self.DBP.subtask_name)
            if key == 'CLUSTER_COMMAND_DIR':
                self.config[key] = self.BP.nominate((key, value), 'VC', self.DBP.subtask_name)
            if key == 'OUTLIER_PICTURE_RESULT_DIR':
                self.config[key] = self.BP.nominate((key, value), 'OD', self.DBP.subtask_name)


    # 打印dict
    def printDict(self, dictionary):
        for key, value in dictionary.items():
            print(key, ' : ', value)
            print('---------------------------------------')
        print()
    
    # 解析项目文件状态
    def ParsingConfig(self):
        self.configStatus = dict()
        no = 0
        for key, value in self.config.items():            
            if key=='_id' or key == 'INSTANCE_NAME':
                continue
            from pathlib import Path
            dataFile = Path(value)
            self.configStatus[key] = dict()            
            #  对于所有目录或文件：小写项目名称，是否存在/没有配置，创建日期
            no += 1
            self.configStatus[key]['no'] = no
            self.configStatus[key]['name'] = key[:key.rfind('_')].lower()
            if value == '' or dataFile.exists() == False: 
                self.configStatus[key]['isExist'] = False
                self.configStatus[key]['type'] = None
                continue
            else:
                self.configStatus[key]['isExist'] = True
            import time 
            import datetime
            t = dataFile.stat().st_ctime            
            self.configStatus[key]['createTime'] = datetime.datetime.fromtimestamp(t).strftime("%Y.%m.%d %H:%M")  
            # 对于目录：含有多少子目录
            if dataFile.is_dir():
                self.configStatus[key]['type'] = 'dir'
                self.configStatus[key]['subdirs'] = len([i for i in dataFile.iterdir()])
                self.configStatus[key]['subfiles'] = len([i for i in dataFile.glob('**/*')])

            # 对于文件：文件大小，行数，第一行内容，
            else:
                import os
                self.configStatus[key]['type'] = 'file'
                fileSize = dataFile.stat().st_size / 1024
                sizeUnit = 'K'
                if fileSize > 1024:
                    fileSize = fileSize / 1024
                    sizeUnit = 'M'
                if fileSize > 1024:
                    fileSize = fileSize / 1024
                    sizeUnit = 'G'
                self.configStatus[key]['size'] = str(round(fileSize, 2))+sizeUnit
                # 判断是否可以读出内容预览
                if dataFile.suffix not in ['.txt', '.json']:
                    self.configStatus[key]['lines'] = self.configStatus[key]['head'] =self.configStatus[key]['tail'] = '-'
                    self.configStatus[key]['type'] = 'bytefile'
                    continue
                result = os.popen('wc -l '+value).read()
                self.configStatus[key]['lines'] = int(result.split(' ')[0])
                result = os.popen('head -1 '+value).read()                
                self.configStatus[key]['head'] = result[:100]#.decode(encoding='UTF-8', errors='strict')
                result = os.popen('tail -1 '+value).read()
                self.configStatus[key]['tail'] = result[:100]#.decode(encoding='UTF-8', errors='strict')
        return self.configStatus

    # Dict操作
    def writeDict(self, content, path): 
        import json
        writefile = open(path, mode='a')
        writefile.write(json.dumps(content)) 
        writefile.close()
        print(len(content), ' records writen.')                                                          

    def readDict(self, path): 
        import json
        readfile = open(path, mode='r') 
        content = readfile.read() 
        readfile.close()
        print(len(content), ' records read.') 
        return json.loads(content)
        
    # 函数信息
    def info(self):
        info='''
        There are 7 steps,  1 assistant in this algo:
            [A]<building> Assistant(Interpreter.py)

            [DP]<building> DataPreprocess(DataPreprocessor.py)
                handle the pictures, remove meaningless records, extract Records 
            [DT]<building> DataTransform(DataTransformer.py)
                transform the raw picutre and records pair to trees
            [IF]<building> ImageFeatureExtract(ImageFeatureExtractor.py)
                extract the feature of images
            [DE]<building> DimensionEngineering(DimensionEngineer.py)
                Feature Engineering,  extract the meaningful feature of pictures
            [VC]<restruct> VectorCluster(VectorClusterer.py)
                use api vector to cluster
            [IC]<building> ImageCluster(ImageClusterer.py)
                use image to cluster
            [OD]<building> OutlierDetect(OutlierDetector.py)
                use outlier detection algo to pick outlier image
        '''
        print(info)
        print(self.DBP.project_name, '\n', self.DBP.metaDataTableName, '\n', self.DBP.configTableName, '\n', self.DBP.apkTreeTableName)
        print(self.DBP.queryMetaData({"projectname":self.DBP.project_name}))

    def checkPassport(self, step):
        #if step == '':
        #    return
        #if step not in self.status:
        #    print('not enough data to %s!' % step)
        pass

    # 保证工作目录均存在
    def checkWorkingDir(self):
        self.makeSureExists(self.config['RAW_ROOT'])
        self.makeSureExists(self.config['GENERATED_ROOT'])
        self.makeSureExists(self.config['INPUT_DATA_DIR'])

    def __init__(self, project_task_info=''):
        ''' 可以通过project_task_info传入项目信息和子任务信息'''
        super(Interpreter, self).__init__()
        sys.path.append(os.path.abspath('.'))
        sys.path.append(os.path.abspath('..'))
        # 使用组合来获取Base类功能
        from Base import BaseProcessor, DataBaseProcessor
        self.BP = BaseProcessor.BaseProcessor()
        self.DBP = DataBaseProcessor.DataBaseProcessor()
        # 运行日志
        self.run_log = {'DP':[], 'DT':[], 'IF':[], 'DE':[], 'VC':[], 'IC':[], 'OD':[]}
        # 如果项目和子任务信息不为空则立即自动打开
        if project_task_info !='':
            self.openProject(project_task_info)

    def openSubTask(self, project_name, subtask_name, remark, copy_from=''):
        # 由于配置重复更新的问题，需要在subtask开启时自动重置config
        self.config = self.DBP.getConfig() 
        # 更新config,这个config上文已经定义好了，在DBP内部,config是整个项目共有的，不能写入子任务
        # 最新逻辑12.22日 为了将每个项目都隔离开来
        if copy_from != '':
            # 复制相应subtask的数据库等结构,一般来讲直接复制所有数据库，复制所有文件即可
            # 首先复制copy_from的metadata,config等记录
            copy_from_project_name,copy_from_subtask_name = copy_from.split('.')
            self.DBP.project_name = copy_from_project_name
            self.DBP.subtask_name = copy_from_subtask_name
            self.DBP.configTableName = self.DBP.project_name+'_config'
            self.DBP.metaDataTableName = self.DBP.project_name + '_metadata'
            self.DBP.apkTreeTableName = self.DBP.project_name+'__'+self.DBP.subtask_name+'_apktree'
            copy_from_config = self.DBP.getConfig() 
            self.loadConfiguration()
            copy_from_GENERATED_ROOT = self.config['GENERATED_ROOT']
            copy_from_config_loaded = self.config 
            copy_from_metadata = self.DBP.queryMetaData({'project_name':1, 'subtask_name':1, 'updatetime':1, "ifdim":1, "ifmethod":1, "imagesize":1, "rawdims":1, "cutdimway":1, "reducedims":1, "calinski" :1, "clustermethod":1, "clusters":1, "silhouette":1,"rawapkforestname":1,"remark":1})
            copy_from_apkTree = self.DBP.queryAPKTree({}, {"raw_id" : 1, "cluster_no" :1, "outlier_score" :1})
            # ----
            self.DBP.project_name = project_name
            self.DBP.subtask_name = subtask_name
            self.DBP.configTableName = self.DBP.project_name+'_config'
            self.DBP.metaDataTableName = self.DBP.project_name + '_metadata'
            self.DBP.apkTreeTableName = self.DBP.project_name+'__'+self.DBP.subtask_name+'_apktree'
            print('==>new apktree tabl:',self.DBP.apkTreeTableName)
            # 先复制config，由于config刚才已经创建，所以直接更新即可||另外，由于还没有调用loadConfiguration所以可以存入数据库,如果projectname相同 就不需要更改config，都一样
            # "_id" : ObjectId("605ed8c61fac64c8cc28923b")去除
            # 注意，暂不支持跨库
            if copy_from_project_name != project_name:
                del copy_from_config['_id']
                copy_from_config['INSTANCE_NAME'] = project_name
                copy_from_config['UPDATETIME'] = self.getFormatedTime()
                print('==>self.DBP.updateConfig({"$set":',copy_from_config)
                self.DBP.updateConfig({"$set":copy_from_config})

            metadata = copy_from_metadata[0]
            del metadata['_id']
            metadata['project_name'] = project_name
            metadata['subtask_name'] = subtask_name
            metadata['updatetime'] = self.getFormatedTime()
            metadata['remark'] = remark
            print('==>self.DBP.saveMetaData(',metadata)
            self.DBP.saveMetaData(metadata)

            #----------------------------复制apktree-----------------------
            if len(copy_from_apkTree) == 0:
                print('原apktree无内容,不再复制')
            else:
                for tree in copy_from_apkTree:
                    self.DBP.saveAPKTree(tree)
                newapktree = self.DBP.queryAPKTree({},{'_id'})
                print('==>after copy apktree length:',len(newapktree))

            #-----------------------------复制GENERATED_ROOT文件夹-------------
            # 最后再复制文件夹?复制项目 注意：由于目前已经实现完全的尽量共享，所以不必复制其文件，只是不同项目复制其整个文件夹而已
            self.config = self.DBP.getConfig() 
            self.loadConfiguration()

            print('正在构建',self.config['GENERATED_ROOT']+project_name)
            if copy_from_project_name != project_name and  (not os.path.exists(self.config['GENERATED_ROOT']+project_name)):
                shutil.copytree(copy_from_GENERATED_ROOT+copy_from_project_name,self.config['GENERATED_ROOT']+project_name)
            self.lastOperator = ''
            print('rawapkforest: ', self.DBP.rawApkForestName, ' opened.')
            print('subtask: ', subtask_name, ' opened.')
            print('Create from %s done.' % copy_from)
            return ''
        if '#' in self.config['INPUT_DATA_DIR']:
            #self.listDataSet()
            #database_name = input('choose your database:')
            database_name = self.DBP.DBNAME
            self.config['INPUT_DATA_DIR'] = self.config['INPUT_DATA_DIR'].replace('#',database_name+'/')
            self.config['PICTURES_TRASH_DIR'] = self.config['PICTURES_TRASH_DIR'].replace('#',database_name+'/')
            self.config['VERSION'] = '2'
            self.DBP.updateConfig({"$set":{"UPDATETIME":self.getFormatedTime(),"INPUT_DATA_DIR":self.config['INPUT_DATA_DIR'],"PICTURES_TRASH_DIR":self.config['PICTURES_TRASH_DIR'],'VERSION':self.config['VERSION']}})
        else:
            self.DBP.updateConfig({"$set":{"UPDATETIME":self.getFormatedTime()}})

        # 切换subTask
        self.DBP.subtask_name = subtask_name
        self.DBP.apkTreeTableName = self.DBP.project_name +'__'+ self.DBP.subtask_name+'_apktree'
        # 查询已有的子任务
        result = self.DBP.queryMetaData(queryField={'rawapkforestname':1})
        # 如果没有找到需要新创建一个,有的话就先算了(可以更新时间)
        if result == []:
            #self.listDatabase()
            #rawapkforestname = input('choose your rawapkforestname:')
            rawapkforestname = self.DBP.rawApkForestName
            self.DBP.saveMetaData({"project_name":project_name, "subtask_name":subtask_name, "updatetime":self.getFormatedTime(), "configdocname":self.DBP.configTableName, "apktreedocname":self.DBP.apkTreeTableName,"rawapkforestname":rawapkforestname,"remark":remark,"version":'2'})
            self.DBP.rawApkForestName = rawapkforestname
        else:
            print(result)
            self.DBP.rawApkForestName = result[0]['rawapkforestname']

        self.loadConfiguration()
        self.printDict(self.config)
        self.lastOperator = ''
        print('rawapkforest: ', self.DBP.rawApkForestName, ' opened.')
        print('subtask: ', subtask_name, ' opened.')

    def openProject(self, project_task_info):
        '''打开项目，即访问数据库创建config，metadata和相应子任务数据库
            示例输入：ExampleProject.hog_pca150_optics5
        '''
        remark = ''
        copy_from = ''
        if '--m' in project_task_info:
            project_task_info,remark = project_task_info.split('--m')
            if '--c' in remark:
                remark,copy_from = remark.split('--c')
        elif '--c' in project_task_info:
                project_task_info,copy_from = project_task_info.split('--c')
        if '.' in project_task_info:
            project_name, subtask_name = project_task_info.split('.')
        else:
            project_name = project_task_info
            subtask_name = ''
        # 首先定义好DBP内部的参数
        self.DBP.project_name = project_name
        self.DBP.configTableName = self.DBP.project_name+'_config'
        self.DBP.metaDataTableName = self.DBP.project_name + '_metadata'
        # 首先需要按项目名称寻找，找到则更新时间，找不到则创建新的(这是函数自带功能)

        # 打开子任务，这时需要更新metadata
        if subtask_name != '':
            self.openSubTask(project_name, subtask_name, remark, copy_from)
        print('project: ', project_name, ' opened.')
        print('database: ', self.DBP.DBNAME, ' opened.')
        
    def deleteProject(self, project_name):
        if project_name == '':
            project_name = self.DBP.project_name
        ans = input('Are Your Sure To Delete This Project? \nInput the project_name to ensure:%s\n' % project_name)
        if ans == project_name:
            apktrees = self.DBP.getApkForestDB()
            apktrees.drop()
            metadata = self.DBP.getMetaDataDB()
            metadata.drop()
            config = self.DBP.getConfigDB()
            config.drop()
            print('Delete Project :', project_name, ' done.')
            self.DBP.project_name = None
            self.DBP.projectMetaDataName = None
            self.DBP.projectConfigName = None
            self.DBP.projectApkTreeName = None
            print('To continue use this system, please open a new project.')

    # 列出各种信息，数据集，数据库，项目等
    def listDataSet(self):
        print('===Local DataSets:')
        global RAW_ROOT
        output = os.popen("find %s  -maxdepth 1 -name 'db_*'" % (RAW_ROOT)).read()
        print(output)

    def listDatabase(self):
        print('===Mongo DataBases:')
        print('in database: '+self.DBP.DBNAME+':'+self.DBP.rawApkForestName)
        client = self.DBP.getConnection()
        collist = client.collection_names()
        for line in collist:
            print(line)

    def listProject(self):
        print('===Mongo Projects:')
        print('in database: '+self.DBP.DBNAME+':'+self.DBP.rawApkForestName)
        client = self.DBP.getConnection()
        collist = client.collection_names()
        for line in collist:
            if 'metadata' in line:
                collect_metadata = self.DBP.getConnection()[line]
                findResult = collect_metadata.find({},{'project_name':1,'subtask_name':1,'updatetime':1,'remark':1}).sort("_id",1)
                for item in findResult:
                    if 'remark' in item:
                        print(item['updatetime']+'\t'+item['project_name']+'.'+item['subtask_name']+'\t'+item['remark'][:30])
                    else:
                        print(item['updatetime']+'\t'+item['project_name']+'.'+item['subtask_name'])
    def listIndexedProject(self):
        print('===Indexed Projects:')
        indexs = self.DBP.getIndexs()
        for i in indexs:
            print(i)

    def chooseDataBase(self,database):
        if database == '':
            self.listDataSet()
            self.DBP.DBNAME = input('choose your DataBase.')
        else:
            print('choose DataBase:',database)
            self.DBP.DBNAME = database
            
    def chooseApkForest(self,apkforest):
        if apkforest == '':
            self.listDatabase()
            self.DBP.rawApkForestName = input('choose your ApkForest.')
        else:
            print('choose ApkForest:',apkforest)
            self.DBP.rawApkForestName = apkforest
        
    def remarksubtask(self,remark):
        self.DBP.updateMetaDataRemark(remark)

    def showCMDs(self):
        cmds='''
        if 'cmds':
            showCMDs()
        if 'cb' :
            chooseDataBase()
        if 'l' :
            listProject()
        if 'ls' :
            listDataSet()
        if 'lb' :
            listDatabase()
        if 'li' :
            listIndexedProject()
        if 'dbinfo' :
            DBP.info()
        if 'o' :
            print('opening project...')
            openProject(params)
        if 's' :
            print('opening subtask...')
            #A sub spm_spe300_optics3
            openSubTask(DBP.project_name,params)
        if 'delete' :
            deleteProject(params)
        if 'cfg' :
            printDict(config)
        if 'more' :
            ParsingConfig()
        if 'log' :
            for key, value in run_log.items():
                print('===', key)
                for item in value:
                    print(end='') if item == [] or item == [''] else print(item)
        if 'reload' :
            loadConfiguration(True)
        if 'info' :
            info()
        '''
        print(cmds)
        

    # `3最主要的命令解析
    def cmd(self, step=[], operator=[], params=[]):
        import sys, os
        from importlib import reload
        sys.path.append(os.path.abspath('.'))
        sys.path.append(os.path.abspath('..'))
        self.checkPassport(step)
        step = step.strip(' ')
        # 进入cmd之后首先选择数据集
        # 助手，显示配置等
        if 'A' == step:
            if 'cmds'==operator:
                self.showCMDs()
            if 'cb' == operator:
                self.chooseDataBase(params)
            if 'cf' == operator:
                self.chooseApkForest(params)
            if 'l' == operator:
                self.listProject()
            if 'ls' == operator:
                self.listDataSet()
            if 'lb' == operator:
                self.listDatabase()
            if 'li' == operator:
                self.listIndexedProject()
            if 'curinfo' == operator:
                if self.DBP.project_name=='project_name' or self.DBP.subtask_name == 'subtask_name':
                    print('No project or subtask has been selected, use `A o xx.xx` or `A s xx` to open.')
                    return ''
                print('===current database&apkforest:')
                self.DBP.info()

                print('===current project&subtask:')
                print(self.DBP.project_name,'.',self.DBP.subtask_name)

                print('===current progress:')
                print(os.system('tree '+self.config['GENERATED_ROOT']+self.DBP.project_name))
                
                print('===remark:')
                print(self.DBP.queryMetaData(queryField={'remark':1}))
                remark = input('==>can remark this subtask:(n to skip)')
                if remark != 'n' and remark != 'N':
                    self.remarksubtask(remark)
            if 'o' == operator:
                print('opening project...')
                self.openProject(params)
            if 's' == operator:
                print('opening subtask...')
                #A sub spm_spe300_optics3
                self.openSubTask(self.DBP.project_name,params)
            if 'delete' == operator:
                self.deleteProject(params)
            if 'cfg' == operator:
                self.printDict(self.config)
            if 'more' == operator:
                self.ParsingConfig()
            if 'log' == operator:
                for key, value in self.run_log.items():
                    print('===', key)
                    for item in value:
                        print(end='') if item == [] or item == [''] else print(item)
            if 'reload' == operator:
                self.loadConfiguration(True)
            if 'info' == operator:
                self.info()

        # 数据预处理 
        if 'DP' == step:
            from DataPreprocess import DataPreprocessor
            reload(DataPreprocessor)
            preprocessor = DataPreprocessor.DataPreprocessor(self.config, self.DBP)
            self.run_log['DP'].append(preprocessor.runDataPreprocess(operator, params))

        # 转化(包括将csvs直接变为apktree)
        if 'DT' == step:
            from DataTransform import Transformer
            reload(Transformer)
            print('extracting.')
            transformer = Transformer.Transformer(self.config, self.DBP)
            self.run_log['DT'].append(transformer.runTransformer(operator, params, self.config))

        # --------------------------8.5 线----------------------------------------------
        # 特征工程，降维
        if 'DE' == step:
            from DimensionEngineering import DimensionEngineer
            reload(DimensionEngineer)
            dimensionEngineer = DimensionEngineer.DimensionEngineer(self.config, self.DBP)
            self.run_log['DE'].append(dimensionEngineer.runEngineer(operator, params))
        # 图像特征提取
        if 'IF' == step:
            from ImageFeatureExtract import ImageFeatureExtractor
            reload(ImageFeatureExtractor)
            imageFeatureExtractor = ImageFeatureExtractor.ImageFeatureExtractor(self.config, self.DBP)
            self.run_log['IF'].append(imageFeatureExtractor.runExtractor(operator, params))
        # 向量聚类
        if 'VC' == step:
            # 首先需要判断是否具有这个subtask的数据库，如果有那是需要删除的
            from VectorClustering import VectorClusterer
            reload(VectorClusterer)
            clusterer = VectorClusterer.VectorClusterer(self.config, self.DBP)
            self.run_log['VC'].append(clusterer.runClustering(operator, params))
        if 'IC' == step:
            from ImageClustering import ImageClusterer
            reload(ImageClusterer)
            imageClusterer = ImageClusterer.ImageClusterer(self.config, self.DBP)
            self.run_log['IC'].append(imageClusterer.runClustering(operator, params))
        if 'ICC' == step:
            from IConClassifing import IConClassifier
            reload(IConClassifier)
            iconClassifier = IConClassifier.IConClassifier(self.config, self.DBP)
            self.run_log['IC'].append(iconClassifier.runClassifing(operator, params))
        # 异常检测
        if 'OD' == step:
            from OutlierDetect import OutlierDetector
            reload(OutlierDetector)
            outlierDetector = OutlierDetector.OutlierDetector(self.config, self.DBP)
            self.run_log['OD'].append(outlierDetector.runDetector(operator, params))
        # 退出系统
        if 'exit' == step:
            #if params.split(', ')[0] != 'droplog':
            #    self.writeDict(self.run_log, 'run.log')
            sys.exit()

    # `1命令文件中的命令读取
    def sequenceCMD(self, cmdListPath,inputType='F'):
        if inputType == 'F':
            cmdListFile = open(cmdListPath, 'r')
            cmdList = cmdListFile.readlines()
        else:
            cmdList = cmdListPath.split("\r\n")
            print('in inter cmdList:',cmdList)
        for cmd in cmdList:
            if cmd[0] == '#':
                continue 
            commands = cmd.strip('\n').split(' ')
            while len(commands) < 3:
                commands += ['']
            self.cmd(commands[0], commands[1], commands[2])

    # `0入口函数
    def main(self):
        import os, sys
        # 一般都使用cmd文件，则使用这个函数解析
        if len(sys.argv) == 2:
            self.sequenceCMD(sys.argv[1])
            return
        
        # 在最一开始选择
        #self.chooseDataBase()
        while True:
            inputText = input(">>> ")
            if inputText == '.':
                inputText = self.lastOperator
                print('use ', inputText)
            self.lastOperator = inputText
            commands = inputText.split(' ')
            # 防止简写命令省略操作部分和参数部分
            while len(commands) < 3:
                commands += ['']
            try:
                self.cmd(commands[0], commands[1], commands[2])
            except Exception as e:
                print('Except! Try agin')
                traceback.print_exc()

if __name__ == '__main__':
    interpreter = Interpreter()
    interpreter.main()


