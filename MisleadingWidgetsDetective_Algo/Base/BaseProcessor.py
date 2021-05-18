# 这个类需要重构
# 将本地IO写在一个文件中
# 再将数据库IO写在另一个文件中
# 或者将暂时用不到的本地方法移动到另一个文件中

from pymongo import MongoClient
from bson import ObjectId

class BaseProcessor():
    def __init__(self):
        pass

    def info(self):
        info = '''
        '''
        print(info)


    ###########        Base Info ##############
    def getFormatedTime(self):
        import datetime
        now_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S') 
        return time_str

    #get PicutreLists from Picture Path
    def getPictureListsFromDir(self,rootDir): 
        import os                                                                                                      
        pictureLists = [] 
        pathLists = []                                                                                                   
        x = os.walk(rootDir)                                                                                              
        for rootDir,d,filelist in x:                                                                                   
            for filename in filelist:                                                                               
                apkName = rootDir.split('/')[-1]                                                                                
                # 为了防止图片有多个'.'                                                                                        
                picName = filename.split('.')[:-1] 
                picName = '.'.join(picName)                                                                         
                pictureLists.append([apkName,picName])                                                                         
                                                                                                                               
                # 该路径从图像rootDir的根目录开始 
                relativepath = apkName+'/'+filename 
                pathLists.append(relativepath)           
        return pictureLists,pathLists
    
    # 图片数据，依照aptTreee中路径的排序顺序来
    # 再依照apk+iamge到apkpathtree当中读
    def getPictureListsFromTree(self,apkTree,rootDir,treeVersion='4'): # 新接口，按照apktree生成picutrelists
        if treeVersion == '4':
            pictureLists = []
            for app,widgets in apkTree.items():
                for widgetName in widgets.keys():
                   pictureLists.append(rootDir + apkTree[app][widgetName]['path'])            
            return pictureLists
        if treeVersion == '3':
            pictureLists = []
            for app,widgets in apkTree.items():
                for widgetName in widgets.keys():
                   pictureLists.append(rootDir + apkTree[app][widgetName]['path'])            
            return pictureLists
        
        import csv
        pictureLists = []
        csv_reader = csv.reader(open(apiVectorNamePath))                                                                       
        for line in csv_reader:
            if treeVersion == '1':
                pictureLists.append( rootDir + apkPathTree[line[0]][line[1]])                                        
            else:
                #new sturct 
                pictureLists.append( rootDir + apkPathTree[line[0]][line[1]+','+line[2]]) 

        return pictureLists 

    # 将图片读为矩阵
    def loadPictureData(self,picutreList,method='cv2',resize=(0,0)):
        import numpy as np
        import os
        from PIL import Image
        import cv2

        imgs = []
        img = ''
        for imgfile in picutreList:
            if not os.path.exists(imgfile):
                print('img ',imgfile,' not exists.')
                continue
            img = Image.open(imgfile).convert('RGB')
            if method == 'cv2':
                img = cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2GRAY)  
            else:
                if resize != (0,0):
                    img = img.resize(resize,Image.ANTIALIAS)
                img = np.array(img)
            imgs.append(img)
        X = np.array(imgs)
        size = img.shape
        print('Image Martix shape',X.shape)
        print('Image size',size)
        #-------------------------s3 读取size和展示 
        return np.array(X),size

    def plot_embedding_scatter(self,coordinates,images,figsize=(4,3),frameon=False,title=None,xticks=[],yticks=[],min_dist=4e-4):
        import matplotlib.pyplot as plt
        from matplotlib import offsetbox
        import numpy as np

        X = coordinates
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        # 将X进行归一化，便于显示
        X = (X - x_min) / (x_max - x_min)
        
        '''
        fig,ax = plt.subplots()的意思是，建立一个fig对象，建立一个axis对象。不然要用更复杂的方式来建如下：
        
        fig=plt.figure()
        
        ax=fig.add_subplot(111)
        '''
        # figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
        #plt.figure(figsize=figsize)#,dpi=1000)#,frameon=True)
        # 将图像边框去除
        ax = plt.subplot(111,frameon=frameon)
        
        '''
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
             color=plt.cm.Set1(y[i] / 10.),
             fontdict={'weight': 'bold', 'size': 9})
        '''
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            # 遍历digits数据，shape[0]是其所有数据的数量
            for i in range(X.shape[0]):
                # 距离？ [(45.9419847,10.60655811) - (1,1)]^2 + 1 
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                # 如果距离过小不再显示该图片
                #if np.min(dist) < 4e-3:
                if np.min(dist) < min_dist:
                    # don't show points that are too close
                    continue
                # 显示的图片被赋值为：[[1,1],[45,0.6],...]
                # np.r_延长行；np.c_延长列
                shown_images = np.r_[shown_images, [X[i]]]
                # 还可以使用offsetbox模块中提供的AnnotationBbox和OffsetImage实现相同的功能。
                # AnnotationBbox是一个标注框，其中可以放置任何Artist对象，我们在其中放置一个OffsetImage对象，
                # 它能按照指定的比例显示图像，缺省比例为1。关于这两个对象的各种参数，请读者查看相关文档及源代码。
                # 参数为: 图像，位置
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),X[i])
            ax.add_artist(imagebox)
        # 设置x，y坐标，实际上可以加上
        #plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        plt.show()

    ###########        base IO process      #######
    # 新命名文件
    def newNominate(self,sourceConfig,phase,subtask_name):
        # ------------------------New------------------------
        # UniversalSet_PureApk_EvaluateMethods__puremapi_raw_sim_vgg_if
        #                                          FE     DE  VC  IF OD
        # ...t_PureApk_EvaluateClusters.puremethodapi_kpca1000_kmeans_vgg_if
        subtask_suffix = subtask_name.split('_') 
        trueFileName = sourceConfig[1].rstrip('/')+'-'  # rstrip为了去除Path最后的/
        phaseSuffix = ''
        fileSuffix = ''

        if phase in ['DE']: 
            phaseSuffix = "_".join(subtask_suffix[:2]) 
        elif phase in ['VC']:
            phaseSuffix = "_".join(subtask_suffix[:3]) 
        elif phase in ['IF']:
            phaseSuffix = subtask_suffix[3] 
        elif phase in ['OD']:
            phaseSuffix = "_".join(subtask_suffix) 
        else:
            # 如果不在这里可能并不需要phase上的suffix
            trueFileName = trueFileName.rstrip('-')

        if sourceConfig[0] == 'CLUSTERER_CONTAINER_PATH': 
            fileSuffix = '.pkl'
        elif '_DIR' in sourceConfig[0]: 
            fileSuffix = '/' 
        else:
            fileSuffix = '.txt' 
  
        return trueFileName + phaseSuffix + fileSuffix
    # 命名文件
    def nominate(self,sourceConfig,phase,subtask_name):
        # projectName: 整个项目外层文件夹，对内层没有关系，所以不会参与到命名中来                         
        # phase: ['DP','DT','IF','DE','VC','IC','OD']                                                     
        # subtask_name1: ExampleProject__spm_lle300_optics3_iForest
        #                                IF    DE      VC     OD
        # subtask_name2: ExampleProject__hog_kpca1000_dbscan_iForest 
        # ------------------------New------------------------
        # UniversalSet_PureApk_EvaluateMethods__puremapi_raw_sim_vgg_if
        #                                          FE     DE  VC  IF OD
        # ...t_PureApk_EvaluateClusters.puremethodapi_kpca1000_kmeans_vgg_if
        # ------------------------New------------------------
        #subtask_name = subtask_name.split('__')[1]
        subtask_suffix = subtask_name.split('_') 
        if len(subtask_suffix) == 5:
            return self.newNominate(sourceConfig,phase,subtask_name)
        trueFileName = sourceConfig[1].rstrip('/')+'-'  # rstrip为了去除Path最后的/
        phaseSuffix = ''
        fileSuffix = ''

        if phase in ['IF']: 
            phaseSuffix = subtask_suffix[0] 
        elif phase in ['DE']:
            phaseSuffix = "_".join(subtask_suffix[:2]) 
        elif phase in ['VC']:
            phaseSuffix = "_".join(subtask_suffix[:3]) 
        elif phase in ['OD']:
            phaseSuffix = "_".join(subtask_suffix) 
        else:
            # 如果不在这里可能并不需要phase上的suffix
            trueFileName = trueFileName.rstrip('-')

        if sourceConfig[0] == 'CLUSTERER_CONTAINER_PATH': 
            fileSuffix = '.pkl'
        elif '_DIR' in sourceConfig[0]: 
            fileSuffix = '/' 
        else:
            fileSuffix = '.txt' 
  
        return trueFileName + phaseSuffix + fileSuffix

    # Txt
    # 将正常的list一行行存储在txt中
    def writeTxt(self,content,path):
        #path += '.txt'
        self.makeSureExists(path)
        file_handle=open(path,mode='w')
        file_handle.write("\n".join(content))
        file_handle.close()
        print(path,',',len(content),' records writen.')

    def readTxt(self,path):
        #path += '.txt'
        file_handle=open(path,mode='r')
        lines = file_handle.readlines()
        file_handle.close()
        content = []
        for i in lines:
            content.append(i.strip('\n'))
        print(path,',',len(content),' records read.')
        return content


    # Dict
    def writeDict(self,content,path):
        #path += '.json'
        import json
        self.makeSureExists(path)
        writefile = open(path,mode='w')
        writefile.write(json.dumps(content))
        writefile.close()
        print(path,',',len(content.keys()),' keys read.')

    
    def readDict(self,path,readType="ordered"):
        #path += '.json'
        import json
        from collections import OrderedDict
        readfile = open(path,mode='r')
        if readType == "ordered":
            content = json.loads(readfile.read(), object_pairs_hook=OrderedDict)
        else:
            content = json.loads(readfile.read())
        readfile.close()
        print(path,',',len(content),' keys read.')
        return content

   
    # Vector
    def writeVector(self,content,path,fmt="%f"):
        #path += '.txt'
        import numpy 
        self.makeSureExists(path)
        numpy.savetxt(path,content,fmt=fmt)
        print(path,',',len(content),' records writen.')
        
    def readVector(self,path,dtype=''):
        #path += '.txt'
        import numpy
        if dtype=='int32':            
            content = numpy.loadtxt(path,dtype=numpy.int32)
        elif dtype == '':
            content = numpy.loadtxt(path)
        print(path,',',len(content),' records read.')
        return content
    
        
    # CSV
    def writeCSV(self,content,path):
        #path += '.csv'
        self.makeSureExists(path)
        import csv
        with open(path,'w',newline='') as t_file:
            csv_writer = csv.writer(t_file)
            for l in content:
                csv_writer.writerow(l)
        print(path,',',len(content),' records writen.')

    def readCSV(self,path,dtype='int'):
        #path += '.csv'
        import csv
        csv_reader = csv.reader(open(path))
        content = []
        for curRec in csv_reader:
            if dtype == 'float':
                content.append([float(item) for item in curRec])
            else:
                content.append([int(item) for item in curRec])
        print(path,',',len(content),' records read.')
        return content
                
    def saveObject(self,content,path):
        #path += '.pkl'
        self.makeSureExists(path)
        import pickle
        with open(path, "wb") as f:
            pickle.dump(content, f)
        print(path,',',len(content),' records writen.')
        
    def loadObject(self,path):
        #path += '.pkl'
        import pickle
        with open(path, "rb") as f:
            content = pickle.load(f)
        print(path,',',len(content),' records read.')
        return content

    ###########        Base File Operator ##############
    #get PicutreLists from Picture Path
    # make sure path exists
    def makeSureExists(self,path):
        import os
        pathDir = "/".join(path.split('/')[:-1])
        if not os.path.exists(pathDir):
            print('making dir:',pathDir)
            os.makedirs(pathDir)

    def isExist(self,filePath):
        import os
        if not os.path.exists(filePath):
            return False
        return True

    def moveFile(self,sourcepath,destpath):                                                   
        import os,shutil                                                                                
        import sys                                                                                      
        destdir = '/'.join(destpath.split('/')[:-1])
        self.makeSureExists(destdir)
        shutil.move(sourcepath,destpath)  

    def copyFile(self,sourcepath,destpath):                                                   
        import os,shutil                                                                                
        import sys                                                                                      
        destdir = '/'.join(destpath.split('/')[:-1])+'/'
        self.makeSureExists(destdir)
        shutil.copy(sourcepath,destpath)  

    def linkFile(self,sourcepath,destpath):                                                   
        import os
        destdir = '/'.join(destpath.split('/')[:-1])+'/'
        self.makeSureExists(destdir)
        os.system('ln '+sourcepath+' '+destpath)

    def resolveAPI(self,api):
        import copy
        raw_api = copy.deepcopy(api)
        import re
        package = class_name = method = ""
        # 先找包名
        package_regex = re.compile('^[a-z\.0-9]+')
        package_regex_result = package_regex.match(api)
        if package_regex_result == None:
            print('无法获取包名:',raw_api)
            return '','',''
        else:
            package = package_regex_result.group(0).rstrip('.')
        # 再找类名(v7那几个不行)
        api = api[len(package)+1:]
        #print(' api in class:',api)
        class_regex = re.compile('^[A-Za-z\$]+')
        class_regex_result = class_regex.match(api)
        if class_regex_result == None:
            print('无法获取类名:',raw_api)
            return package,'',''
        else:
            class_name = class_regex_result.group(0).rstrip('.')
        # 再找方法
        api = api[len(class_name)+1:]
        method_regex = re.compile('^[A-Za-z<>]+')
        method_regex_result = method_regex.match(api)
        if method_regex_result == None:
            print('无法获取method名:',raw_api)
            return package,class_name,''
        else:
            method = method_regex_result.group(0).rstrip('.')
        #print('success:',package,class_name,method)
        return package,class_name,method
