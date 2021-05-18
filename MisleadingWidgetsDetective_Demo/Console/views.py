'''                                                                                               # Projects
    # Basic                                                                                       ## Console/projects.html
    path('',views.homepage),                                                                      path('projects',views.projects),
    path('home',views.homepage),                                                                  ## Console/subtask.html
                                                                                                  path('subtask/<str:project_name>/<str:subtask_name>',views.subtask),
    # Console                                                                                     ## Console/
    ## Console/console.html                                                                       path('remarksubtask',views.remarkSubtask),
    path('console',views.console),                                                                path('resovlecoverlist',views.resovleCoverList),
    path('remarkdatabase',views.remarkDataBase),                                                  path('viewclusters/<str:project_name>/<str:subtask_name>',views.viewClusters),
    path('changedb/<str:db_name>',views.chooseDataBase),                                          
    path('changeapkforest/<str:apkforest>',views.chooseRawApkForest),                             # Gallerys
    ## 用于检查有多少算法程序在本机执行                                                           ## Console/gallerys.html
    path('checkRun',views.checkRun),                                                              path('gallerys',views.gallerys),
    ## Console/submitRun.html                                                                     
    path('submitRun',views.submitRun),                                                            ## Console/datasets.html 外链✈
    ## 进行数据集的预聚类，标注,去噪                                                              path('datasets',views.dataSets),
    path('tagrawdatabase/<str:database>',views.tagRawDataBase),                                   
    path('submitrawdatabasetag/<str:database>',views.submitRawDataBaseTag),                       # Tags
    path('submitdropproject/<str:project_name>/<str:subtask_name>',views.submitDropProject),      #Console/tags.html
                                                                                                  path('tags',views.tags), 
    path('apkforestlist',views.apkforestlist),                                                    # Console/album_outlier.html
    path('apkforest/<str:apkforestName>',views.apkforest),                                        path('album/<str:projectName>/<str:subtask_name>/<str:galleryId>/ \
                                                                                                  <str:hightlightClusterId>/<int:cluster_no>',views.album), 
    path('apkForestSubmit',views.apkforestsubmit),                                                # Console/album_tagging.html
    ## 将app检查列表化简去重                                                                      path('tagging/<str:projectName>/<str:subtask_name>/<str:widget_id>',views.tagging),
    path('checkapp',views.checkAPPs),                                                             # Console/evaluate.html
    path('savecoverlist',views.saveCoverList),                                                    path('evaluateHome',views.evaluateHome),
                                                                                                  path('evaluate/<str:projectName>/<str:subtask_name>',views.evaluate),
    # Users                                                                                       path('evaluateSubmit',views.evaluateSubmit),
    path('login',views.login),                                                                    # 实际上不需要，但是先这样
    path('users',views.users),                                                                    path('submitSuspects',views.submitSuspects),
    path('addUser',views.addUser),                                                                
    path('deleteUser',views.deleteUser),                                                          #### videos
                                                                                                  path('videos',views.videos),
                                                                                                  path('hashtags',views.hashtags),
                                                                                               
                                                                                                  
                                                                                                  ### bug-show
                                                                                                  path('bugshow',views.bugShow),
'''
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from threading import Thread
from bson import ObjectId
import re
import time
import datetime
import bcrypt
import math
import csv
import os
import numpy as np
from . import HelpMeDownload

DBIP = '127.0.0.1'
DBPORT = 27017
#DBNAME = 'db_universal_set_evaluation'
DBNAME = 'db_universal_set'
RAWAPKFOREST = 'pureapkforest'
#HOST_NAME = 'http://219.216.64.42:8000/'
HOST_NAME = 'http://219.216.64.127:8000/'
#HOST_NAME = 'http://uimanager.crepuscule.site:8601/'
STATIC_PATH = HOST_NAME + 'static'
PICTURE_PATH = 'http://219.216.64.127'
#PICTURE_PATH = 'http://127.0.0.1'
#PICTURE_PATH = 'http://uimanager.crepuscule.site:8601/static'
CONFIG_PATHS = '/home/dl/users/wangruifeng/05MisleadingWidgets/androidwidgetclustering'
CURRENT_INSPECTED_APP_SET_PATH = '/home/dl/users/wangruifeng/05MisleadingWidgets/MisleadingWidgetsDetective/static/inspected.txt'

# 应该可以避免而从算法系统中读出的:
RAW_ROOT = '/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/'
ZIP_ROOT = '/data/wangruifeng/datasets/DroidBot_Epoch/zips/'
COVER_ICON_TXT_PATH = '/data2/user_codes/wangruifeng/05MisleadingWidgets/MisleadingWidgetsDetective/static/needcover.txt'
COVER_ICON_CSV_PATH = '/data2/user_codes/wangruifeng/05MisleadingWidgets/MisleadingWidgetsDetective/static/needcover-'

# #>>Basic  ---------------------------------------------------------------------------------------------------------------
# path('',views.homepage),
# path('home',views.homepage),

def isAtPackageList(api,package_list):
    import re
    # 这么做的目的是，不让android直接就匹配，要不然误进很多api
    # 正则表达式，从开头开始的小写字母或者\.，遇到大写就结束了，这样可以匹配出包名
    package_regex = re.compile('^[a-z\.]+')
    package_regex_result = package_regex.match(api)
    if package_regex_result == None:
        print('无法获取包名！')
        return '',False # 无法获取包名！
    else:
        package = package_regex_result.group(0).rstrip('.')

    print('searching >>%s<< in package_list...' % package)
    if package in package_list:
        return package,True
    return package,False

def readTxt(path):                                                                                                   
    #path += '.txt'                                                                                                       
    file_handle=open(path,mode='r')                                                                                       
    lines = file_handle.readlines()                                                                                       
    file_handle.close()                                                                                                   
    content = []                                                                                                          
    for i in lines:                                                                                                       
        content.append(i.strip('\n').strip(' '))                                                                                     
    print(path,',',len(content),' records read.')                                                                         
    return content

'''
def writeTxt(content,path):                                                                            
    #path += '.txt'                                                                                         
    file_handle=open(path,mode='w')                                                                         
    file_handle.write(content)                                                                   
    file_handle.close()                                                                                     
    print(path,',',len(content),' records writen.') 
'''

def writeTxt(content,path,mode='w'):
    #path += '.txt'
    file_handle=open(path,mode=mode)
    file_handle.write("\n".join(content))
    file_handle.close()
    print(path,',',len(content),' records writen.')

def readDict(path,readType="ordered"):
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

def writeDict(content,path):
    #path += '.json'
    import json
    writefile = open(path,mode='w')
    writefile.write(json.dumps(content))
    writefile.close()
    print(path,',',len(content.keys()),' keys read.')
    


# Basic
def homepage(request):
    #根据需要选择HttpRespone或者HttpResponse
    # 获取数据库信息
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    db = dataBaseProcessor.getConnection()
    collist = db.collection_names()

    no = 1
    subtaskDictList = list()
    # 遍历数据表
    for col in collist:
        # filter all _metadata
        if '_metadata' not in col :
            continue
        # 这条分支是能获取到信息的
        # 对于每个项目数据表遍历其内容
        for subtask in db[col].find():
            subtaskDict = dict()
            subtaskDict['no'] = no
            subtaskDict['projectName'] = col.replace('_metadata', '')
            subtaskDict['subtaskName'] = subtask['subtask_name']
            subtaskDict['updateTime'] = subtask['updatetime']
            subtaskDict['link'] = '/Console/console/%s/%s' % (subtaskDict['projectName'], subtaskDict['subtaskName'])
            subtaskDictList.append(subtaskDict)
            no +=1
    context = {'subtaskDictList':subtaskDictList, 'staticPath':STATIC_PATH}
    return render(request, 'Console/homepage.html', context)

def loadInterpreter(configPaths, projectName="", subtask_name=""):
    import sys, os
    sys.path.append(os.path.abspath(configPaths)) 
    sys.path.append(os.path.abspath(configPaths)+'/Assistant/') 
    import Interpreter
    from importlib import reload
    reload(Interpreter)
    # use 1, 2 to use different config file
    #self.config = configuration.initConfig(2)
    #return configuration.initConfig(1)
    if projectName == "" or subtask_name == "":
        interpreter = Interpreter.Interpreter()
        return interpreter
    interpreter = Interpreter.Interpreter(projectName+'.'+subtask_name)
    return interpreter#.ParsingConfig()

# Basic
def loadDataBase(configPaths):
    import sys, os
    sys.path.append(os.path.abspath(configPaths)+'/Base/') 
    import DataBaseProcessor
    from importlib import reload
    reload(DataBaseProcessor)
    # use 1, 2 to use different config file
    #self.config = configuration.initConfig(2)
    #return configuration.initConfig(1)
    dataBaseProcessor = DataBaseProcessor.DataBaseProcessor()
    dataBaseProcessor.DBNAME = DBNAME
    dataBaseProcessor.rawApkForestName = RAWAPKFOREST
    return dataBaseProcessor

def writeEvaluation(content):
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    findResult = dataBaseProcessor.queryUsers({'username':username})

def objectId2Time(objectid):
    timestamp = time.mktime(objectid.generation_time.timetuple())
    #dateArray = datetime.datetime.fromtimestamp(timestamp)
    #otherStyleTime = dateArray.strftime("%Y.%m.%d %H:%M")
    time_str = datetime.datetime.strftime(timestamp, '%Y-%m-%d %H:%M:%S')
    return time_str

def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper


def index(request):
    #根据需要选择HttpRespone或者HttpResponse
    return JsonResponse({'message':'Hello world'})
    #return HttpResponse("Hello world")
# #<<Basic  ---------------------------------------------------------------------------------------------------------------

# #>>Users  ---------------------------------------------------------------------------------------------------------------
# path('login',views.login),
# path('users',views.users),
# path('addUser',views.addUser),
# path('deleteUser',views.deleteUser),

def login(request):
    username = request.GET.get('username','')
    password = request.GET.get('password','')

    password = password.encode('utf-8')
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    findResult = dataBaseProcessor.queryUsers({"password":1,'job':1,'email':1},{'username':username})
    print('findResult',findResult)
    true_password = findResult[0]['password']
    if bcrypt.checkpw(password, true_password):
        print("match")
        return JsonResponse({'msg':'True','username':username})
    return JsonResponse({'msg':'False'})

def users(request):
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    findResult = dataBaseProcessor.queryUsers({'_id':1,'username':1,'job':1,'email':1})
    context = {'userList':findResult,'staticPath':STATIC_PATH}
    print('in users',findResult)
    for i in range(len(findResult)):
        findResult[i]['no'] = findResult[i]['_id']
    return render(request, 'Console/users.html', context)

# Uesrs
def addUser(request):
    username = request.GET.get('username','user001')
    password = request.GET.get('password','user001')
    email = request.GET.get('email')
    job = request.GET.get('job')
    password = password.encode('utf-8')

    salt = bcrypt.gensalt(rounds=8)
    hashed = bcrypt.hashpw(password, salt)
    print('addUser:',{"username":username,"password":password,"job":job,'email':email})
    # db operator
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    insertResult = dataBaseProcessor.saveUsers({"username":username,"password":hashed,"job":job,'email':email})
    print(insertResult)
    context = {'userList':insertResult,'staticPath':STATIC_PATH, 'msg':'Add User %s Successfully!' % username}
    return JsonResponse({'msg':'Add User Successfully!'})

def deleteUser(request):
    user_id = request.GET.get('user_id')
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    print('deleteing id=',user_id)
    deleteResult = dataBaseProcessor.deleteUser(ObjectId(user_id))
    context = {'userList':deleteResult,'staticPath':STATIC_PATH}
    print('deleteResult',deleteResult)
    return JsonResponse({'msg':'Delete Successfully!'})

# #<<Users---------------------------------------------------------------------------------------------------------------

# #>>Console ---------------------------------------------------------------------------------------------------------------
## Console/console.html
# path('console',views.console),
# path('changedb/<str:db_name>',views.chooseDataBase),
# path('changeapkforest/<str:apkforest>',views.chooseRawApkForest),
# ## 用于检查有多少算法程序在本机执行
# path('checkRun',views.checkRun),
# ## Console/submitRun.html
# path('submitRun',views.submitRun),
# ## 进行数据集的预聚类，标注,去噪
# path('rawdatabase/<str:database>',views.tagRawDataBase),
# path('tagrawdatabase',views.submitRawDataBaseTag),
# path('apkforestlist',views.apkforestlist),
# path('apkforest/<str:apkforestName>',views.apkforest),
# path('apkForestSubmit',views.apkforestsubmit),
# ## 将app检查列表化简去重
# path('checkapp',views.checkAPPs),

def console(request):
    from pymongo import MongoClient
    default_cmd = '''
#尝试kpca降维到700,300,150的效果对比
A o kpcaExampleProject
######################### DP
# 已经提前处理过

A s spm_kpca700_optics
######################### DT
DT apktree csvsets

######################### IF
#IF extif spm
IF usedefault

######################### DE
DE vector

DE cutdim kpca,700
######################### VC
VC run
#VC eval
######################### OD
OD run
#OD eval

A s spm_kpca300_optics
DE cutdim kpca,300
######################### VC
VC run
#VC eval
######################### OD
OD run
#OD eval
    '''
    connection = MongoClient(DBIP, DBPORT)
    db_list = connection.database_names()
    db_list.remove('local')
    db_list.remove('admin')

    dataBaseList = []
    for db in db_list:
        dataBaseListItem = dict()
        dataBaseListItem['name'] = db
        save_path = RAW_ROOT + db + '/infos/'
        try:
            remark =  '\n'.join(readTxt(save_path+'remark.txt'))
        except:
            remark = 'Click to Remark'
        dataBaseListItem['remark'] = remark
        dataBaseListItem['remarkLink'] = '/Console/remarkdatabase/%s' % db
        dataBaseListItem['tagLink'] = '/Console/tagrawdatabase/%s' % db
        dataBaseListItem['chooseLink'] = '/Console/changedb/%s' % db
        dataBaseList.append(dataBaseListItem)

    print(db_list)

    # apkforestlist
    global DBNAME
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    client = dataBaseProcessor.getConnection()
    collist = client.collection_names()
    apkForestList = []
    for line in collist:
        if '.' not in line and '_' not in line:
            # 作为列表之一
            apkForestListItem = dict()
            apkForestListItem['name'] = line
            apkForestListItem['tagLink'] = '/Console/apkforest/%s'  % line
            apkForestListItem['chooseLink'] = '/Console/changeapkforest/%s' % line
            apkForestList.append(apkForestListItem)

    context = {'default_cmd':default_cmd,'apkForestList':apkForestList,'staticPath':STATIC_PATH,'dataBaseList':dataBaseList,'current_db':DBNAME,'current_apkforest':RAWAPKFOREST}
    return render(request, 'Console/console.html', context)

def saveCoverList(request):
    coverListFile = request.FILES.get('coverListFile')
    if not coverListFile:
        #文件没有上传，需要将默认文件名写入数据库，拒绝请求并将错误信息写入message（失败了也需要记录）
        coverListFile_name = 'Null'
        coverListFile_uuid = 'Null'
        coverListFile_path = 'Null'           
        coverListFile_size = 0
        return JsonResponse({'coverListFile_id':'0'})
    else:
        coverListFile_path = COVER_ICON_TXT_PATH 
        os.rename(coverListFile_path,coverListFile_path.replace('.txt','-old.txt'))
        destination = open(coverListFile_path,'wb')
        for chunk in coverListFile.chunks():
            destination.write(chunk)
        destination.close()
        return JsonResponse({'msg':'success'})

# Console
def chooseDataBase(request,db_name=''):
    global DBNAME
    if db_name != '':
        DBNAME = db_name 
    return console(request)

def allDataBase(request):
    connection = MongoClient(DBIP, DBPORT)
    db_list = connection.database_names()
    res = {'db_list':db_list}
    return JsonResponse(res,json_dumps_params={'ensure_ascii': False})

# Console
def chooseRawApkForest(request,apkforest=''):
    global RAWAPKFOREST
    if apkforest != '':
        RAWAPKFOREST = apkforest 
    return console(request)
    

# Console
@async
def callFun(cmd):
    import os
    from subprocess import PIPE
    import psutil
    writeTxt(cmd,CONFIG_PATHS+'Assistant/cmdline.txt')
    # 使用linux命令运行
    exectuePath = 'cd '+CONFIG_PATHS
    exectuor = '/home/crepuscule/anaconda3/envs/python3.6/bin/python3.6 '
    print(exectuePath + ' && '+exectuor + 'Assistant/Interpreter.py '+CONFIG_PATHS+'Assistant/cmdline.txt')
    os.system(exectuePath + ' && '+exectuor + 'Assistant/Interpreter.py '+CONFIG_PATHS+'Assistant/cmdline.txt')
    #interpreter = loadInterpreter(CONFIG_PATHS)
    #interpreter.sequenceCMD(cmd,'S')
    # 使用ps命令看一下有没有相关程序运行就行了，然后通过回调ajax返回就可以了

# Console
def checkRun(request):
    import os
    import psutil
    returnValue = os.popen('ps aux | grep "/home/crepuscule/anaconda3/envs/python3.6/bin/python3.6 Assistant/Interpreter.py" | wc -l')
    #returnValue = os.popen('ps aux | grep firefox | wc -l')
    returnValue = int(returnValue.read())
    if returnValue < 2:
        returnValue = 0
    else:
        returnValue -= 2
    res = {'running_num':returnValue,'cpu_percent':psutil.cpu_percent(),'mem_percent':psutil.virtual_memory().percent}
    return JsonResponse(res,json_dumps_params={'ensure_ascii': False})

# Console
def print_object(obj):
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))

# Console
def submitRun(request):
    print("async submitRun start.")
    # 获取命令
    cmd = request.POST.get("cmd")
    callFun(cmd)
    # 返回主页//需要了解有几个脚本正在运行
    print("async submitRun done.")
    print('cmd:',cmd)
    res = {'msg':'success!'}
    #return JsonResponse(res,json_dumps_params={'ensure_ascii': False})
    return console(request)

# Console
# 给数据库进行评论
def remarkDataBase(request):
    global RAW_ROOT
    database = request.GET.get('database')
    remark = request.GET.get('remark')

    save_path = RAW_ROOT + database + '/infos/'
    print(database,'=>',remark)
    if remark != '':
        writeTxt(remark,save_path+'remark.txt','w')
    #    updateMetaData({'$set':{'remark':remark}})
    res = {'msg':'success'}
    return JsonResponse(res,json_dumps_params={'ensure_ascii': False})
        

# Console
# 读取预聚类的csv文件，构造成一个json然后显示
# 算法上应该直接可以执行，存储在db_pure_big的目录下
''' return apkforest.html
'''
def tagRawDataBase(request, database=DBNAME):
    # 读取csv文件
    global RAW_ROOT
    save_path = RAW_ROOT + database + '/infos/'
    if not os.path.exists(save_path+'info.csv'):
        return HttpResponse('尚未进行聚类，无法人工审核该数据集.使用IC search搜索参数,IC run运行聚类')

    f = open(save_path+'info.csv','r') # 专门有用于构建的项目？
    lines = csv.reader(f)


    if os.path.exists(save_path+'abandon.txt'):
        abandon_list = readTxt(save_path+'abandon.txt') # 专门有用于构建的项目？
        print('raw abandon_list:',abandon_list)
    else:
        abandon_list = []


    #rawapkTree = collect_rawapkforest.find({},{'path':1,"app" : 1, "widget" : 1,'image_cluster_no':1}).sort("_id",1)
    rawapkTreeList = []
    max_cluster = 0
    for row in lines:
        #  如果path在抛弃清单里面，不加入
        if row[1] in abandon_list:
            print('abandon_list:',row[1])
            continue
        rawapkTreetemp = {'path':row[1],'app':row[0],"widget":'---','image_cluster_no':row[2]}
        # ??????临时临时
        #rawapkTreetemp = {'path':row[0],'app':'',"widget":'---','image_cluster_no':row[1]}
        # path为id，用path引导软删除
        rawapkTreeList.append(rawapkTreetemp)
        if int(rawapkTreetemp['image_cluster_no']) > max_cluster:
            max_cluster = int(rawapkTreetemp['image_cluster_no'])
    f.close()
    
    # 异常退出，没有进行聚类
    if len(rawapkTreeList) == 0:
        return HttpResponse('尚未进行聚类，无法人工审核该数据集.使用IC search搜索参数,IC run运行聚类')

    # 在线标注
    BaseDir= RAW_ROOT + '%s/input_data/' % database
    context = {'picturePath':PICTURE_PATH,'BaseDir':BaseDir,'rawapkTree':rawapkTreeList,'staticPath':STATIC_PATH,'current_db':database,'current_apkforest':'PRECLUSTER','new_max_cluster':max_cluster,'tagType':'rawdatabase'}
    return render(request,'Console/apkforest.html',context)
    
# Console
def submitRawDataBaseTag(request,database=DBNAME):
    '''
    tagResult = request.GET.get('tagResult')
    # 对删除的文件直接执行更名命令
    '''
    global RAW_ROOT
    BaseDir= RAW_ROOT + '%s/input_data/' % database
    save_path = RAW_ROOT + database + '/infos/'

    widgets = request.POST.get('widgets')
    print(request.POST,'---------')
    widgets = widgets.rstrip(';')

    abandon_list = []
    for widget_path in widgets.split(';'):
        # 将传过来的文件进行改名
        abandon_list.append(widget_path)
        try:
            os.rename(BaseDir+widget_path,BaseDir+widget_path+'.abandon')
        except:
            print('not found '+BaseDir+widget_path)
    writeTxt(abandon_list,save_path+'abandon.txt','a')
    res = {'msg':'success'}
    return JsonResponse(res,json_dumps_params={'ensure_ascii': False})


def submitDropProject(request,project_name,subtask_name):
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    dataBaseProcessor.project_name = project_name
    dataBaseProcessor.subtask_name = subtask_name
    dataBaseProcessor.configTableName = dataBaseProcessor.project_name+'_config'
    dataBaseProcessor.metaDataTableName = dataBaseProcessor.project_name + '_metadata'        
    dataBaseProcessor.apkTreeTableName = dataBaseProcessor.subtask_name+'_apktree'         
    
    res = dataBaseProcessor.deleteProject()
    #return JsonResponse(res,json_dumps_params={'ensure_ascii': False})
    return projects(request)
    

# Console
def apkforestlist(request):
    global DBNAME
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    client = dataBaseProcessor.getConnection()
    collist = client.collection_names()
    apkforestlist = []
    for line in collist:
        if '.' not in line and '_' not in line:
            # 作为列表之一
            apkforestlist.append((line,'/Console/apkforest/%s'  % line ))
    context = {'apkForestList':apkforestlist,'staticPath':STATIC_PATH}
    return render(request,'Console/apkforests.html',context)

# Console
def apkforest(request, apkforestName='MW_lle'):
    global DBNAME
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    db = dataBaseProcessor.getConnection()
    collect_rawapkforest = db[apkforestName]
    rawapkTree = collect_rawapkforest.find({},{'path':1,"app" : 1, "widget" : 1,'image_cluster_no':1}).sort("_id",1)
    rawapkTreeList = []
    max_cluster = 0
    for i in rawapkTree:
        rawapkTreetemp = dict(i)
        rawapkTreetemp['id'] = rawapkTreetemp['_id']
        rawapkTreeList.append(rawapkTreetemp)
        if int(rawapkTreetemp['image_cluster_no']) > max_cluster:
            max_cluster = int(rawapkTreetemp['image_cluster_no'])
    context = {'picturePath':PICTURE_PATH,'BaseDir':'/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/%s/input_data/' % DBNAME,'rawapkTree':rawapkTreeList,'staticPath':STATIC_PATH,'current_db':DBNAME,'current_apkforest':apkforestName,'new_max_cluster':max_cluster,'tagType':'apkforest'}
    return render(request,'Console/apkforest.html',context)

# Console
def apkforestsubmit(request):
    global DBNAME
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    dataBaseProcessor.DBNAME = DBNAME
    dataBaseProcessor.rawApkForestName = request.POST.get('apkforestName')
    widgets = request.POST.get('widgets')
    print(request.POST,'---------')
    widgets = widgets.rstrip(';')
    idUpath = []
    for widget in widgets.split(';'):
        idUpath.append([ObjectId(widget),'-2'])
    dataBaseProcessor.updateRawAPKForest_Cluster(idUpath)
    print(idUpath)
    res = {'msg':'success'}
    return JsonResponse(res,json_dumps_params={'ensure_ascii': False})

# Console
def checkAPPs(request):
    global CURRENT_INSPECTED_APP_SET_PATH
    applist = request.GET.get('applist')
    applist = applist.rstrip('\n')
    needapplist = []
    current_app_set = set(open(CURRENT_INSPECTED_APP_SET_PATH,'r').read().rstrip('\n').split('\n'))
    for app in applist.split('\n'):
        print('=>',app)
        if app not in current_app_set:
            needapplist.append(app)
            current_app_set.add(app)
        else:
            print(app,'hased!')
    current_app_file = open(CURRENT_INSPECTED_APP_SET_PATH,'w')
    current_app_file.write('\n'.join(list(current_app_set)))
    
    print('needapplist:',"\n".join(needapplist))
    return JsonResponse({'needapplist':"\n".join(needapplist)})

# #<<Console---------------------------------------------------------------------------------------------------------------

# #>>Projects---------------------------------------------------------------------------------------------------------------
## Console/projects.html
# path('projects',views.projects),
## Console/subtask.html
# path('subtask/<str:projectName>/<str:subtask_name>',views.subtask),
def projects(request):
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    db = dataBaseProcessor.getConnection()
    collist = db.collection_names()

    no = 1
    subtaskDictList = list()
    # 遍历数据表
    for col in collist:
        # filter all _metadata
        if '_metadata' not in col :
            continue
        # 这条分支是能获取到信息的
        # 对于每个项目数据表遍历其内容
        for subtask in db[col].find():
            subtaskDict = dict()
            subtaskDict['no'] = no
            subtaskDict['project_name'] = col.replace('_metadata', '')
            subtaskDict['subtask_name'] = subtask['subtask_name']

            try:
                dataBaseProcessor.project_name = subtaskDict['project_name']
                dataBaseProcessor.subtask_name = subtaskDict['subtask_name']
                dataBaseProcessor.metaDataTableName = dataBaseProcessor.project_name + '_metadata'        
                metadata = dataBaseProcessor.queryMetaData({'remark':1}) 
                metadata = metadata[0]['remark']
            except:
                metadata = 'Nothing'
            subtaskDict['remarkLink'] = '/Console/remarksubtask/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name']) 
            experiments = re.findall(r'\[[\s\S][^\[\]]*\]',metadata)
            subtaskDict['Experiment'] = ''.join(experiments)
            subtaskDict['remark'] = metadata.replace(subtaskDict['Experiment'],'')

            subtaskDict['Eval1'],subtaskDict['Eval2'],subtaskDict['Eval3'],subtaskDict['EvalSum'],subtaskDict['EvalSum_link']  = readCoverIndexStatistics(subtaskDict['project_name'], subtaskDict['subtask_name'])
            subtaskDict['top10p'],subtaskDict['top20p'],subtaskDict['top30p']= readGlobalCoverIndexStatistics(subtaskDict['project_name'], subtaskDict['subtask_name'])
            subtaskDict['update_time'] = subtask['updatetime']
            subtaskDict['link'] = '/Console/subtask/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name'])
            # viewClusters()
            subtaskDict['view_link'] = '/Console/viewclusters/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name'])
            subtaskDict['delete_link'] = '/Console/submitdropproject/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name'])
            subtaskDictList.append(subtaskDict)
            no +=1
    context = {'subtaskDictList':subtaskDictList, 'staticPath':STATIC_PATH}
    return render(request, 'Console/projects.html', context)

def resovleCoverList(request):
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    db = dataBaseProcessor.getConnection()
    collist = db.collection_names()

    fail_nums = 0
    # 遍历数据表
    for col in collist:
        # filter all _metadata
        if '_metadata' not in col :
            continue
        # 这条分支是能获取到信息的
        # 对于每个项目数据表遍历其内容
        for subtask in db[col].find():
            project_name = col.replace('_metadata','')
            subtask_name = subtask['subtask_name']
            # 遍历所有subtask，判断csv是否存在，不存在调用函数构造之，就可以了
            coverlist_csv = COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'.csv'
            if not os.path.exists(coverlist_csv):
                try:
                    loadfromApkTrees(project_name,subtask_name)
                except:
                    print(project_name,subtask_name,'解析失败!')
                    fail_nums += 1
                    if project_name == 'dbUniversalSetPureapkforest_EvaluateMethods' and subtask_name=='spm_puremethodapi_Sim_iforest':
                        loadfromApkTrees(project_name,subtask_name)

    if fail_nums == 0:
        return JsonResponse({'msg':'解析成功，即将刷新页面'})
    else:
        return JsonResponse({'msg':str(fail_nums)+'个解析失败'})
    
def loadfromApkTrees(projectName='MW_lle', subtask_name='spm_lle150_optics3'):
    global DBNAME
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    dataBaseProcessor.projectName = projectName
    dataBaseProcessor.subtask_name = subtask_name
    dataBaseProcessor.configTableName = dataBaseProcessor.projectName+'_config'
    dataBaseProcessor.metaDataTableName = dataBaseProcessor.projectName + '_metadata'        
    dataBaseProcessor.apkTreeTableName = dataBaseProcessor.projectName+'__'+dataBaseProcessor.subtask_name+'_apktree'         
    config = dataBaseProcessor.getConfig()
    metadata = dataBaseProcessor.queryMetaData({'project_name':1, 'subtask_name':1, 'updatetime':1, "ifdim":1, "ifmethod":1, "imagesize":1, "rawdims":1, "cutdimway":1, "reducedims":1, "calinski" :1, "clustermethod":1, "clusters":1, "silhouette":1,"rawapkforestname":1,"remark":1}) 
    dataBaseProcessor.rawApkForestName = metadata[0]['rawapkforestname']
    rawapkTree = dataBaseProcessor.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}}, {"path" : 1, "app" : 1, "widget" : 1,"method_api":1, "suspect":1})
    apkTree = dataBaseProcessor.queryAPKTree({}, {"raw_id" : 1, "cluster_no" :1, "outlier_score" :1, "outlier_advise":1})
    print('what metadata?',metadata)
    print('what rawapkTree?',rawapkTree[:2])
    print('what apkTree?',apkTree[:2])
    print('----------------------------------------------')

    apkInfoTree = []
    rawapkTree[0]['api_type'] = []
    # 将簇信息存入一个字典，这样可以统计出每个簇有多少个控件，分属于多少应用
    cluster_info_dict = dict()
    cluster_show_dict = dict()
    app_statisic_dict = dict()
    # 将rawAPITree中的信息补充到apkTree当中，最后存入到apkInfoTree中
    for i in range(len(rawapkTree)):
        # 用于统计各簇的数量>>>>
        # 会有各簇下各应用数量的统计,每次更新时都检查状态，如果一个簇内数量>5个，表为可以显示can_show
        # 同时统计每个簇内应用纯度(熵)，低于某个纯度则标为low_app_abundance.即熵越小，不确定性越小，丰度越低，越是不要显示
        if apkTree[i]['cluster_no'] not in cluster_info_dict:
            cluster_info_dict[apkTree[i]['cluster_no']] = dict()
            cluster_info_dict[apkTree[i]['cluster_no']][rawapkTree[i]['app']] = 1
        else:
            if rawapkTree[i]['app'] not in cluster_info_dict[apkTree[i]['cluster_no']]:
                cluster_info_dict[apkTree[i]['cluster_no']][rawapkTree[i]['app']] = 1
            else:
                cluster_info_dict[apkTree[i]['cluster_no']][rawapkTree[i]['app']] +=1
        cluster_show_dict[apkTree[i]['cluster_no']] = isCanShow(cluster_info_dict[apkTree[i]['cluster_no']])
        # 统计各簇中控件数量结束<<<<
        if rawapkTree[i]['_id'] != apkTree[i]['raw_id']:
            print('Wrong!!')
        apkTree[i]['isOutlier'] = ''
        apkTree[i]['outlier_score'] = apkTree[i]['outlier_score'][:5]
        #if float(apkTree[i]['outlier_score']) <= -0.55:
        if float(apkTree[i]['outlier_score']) <= 1:
        #if float(apkTree[i]['outlier_score']) > 3:
            apkTree[i]['isOutlier'] = 'outlier'
        rawapkTree[i]['widget'] = ''.join(rawapkTree[i]['widget'].split(', '))
        apkTree[i]['id'] = str(apkTree[i]['raw_id'])

        rawapkTree[i]['api_type'] = rawapkTree[i]['method_api']#resoluteAPI(rawapkTree[i]['api'],stanard_android_api,'simple')#
        rawapkTree[i]['api_string'] = '  '.join(rawapkTree[i]['method_api'])
        apkInfoTree.append((rawapkTree[i], apkTree[i]))

    apkInfoTree_good = []
    good_cluster_no_dict = dict()
    good_cluster_no_count = 0
    apkInfoTree_pass = []
    pass_cluster_no_dict = dict()
    pass_cluster_no_count = 0
    for i in range(len(rawapkTree)):
        if cluster_show_dict[apkTree[i]['cluster_no']] == 'good':
            if apkInfoTree[i][1]['cluster_no'] not in good_cluster_no_dict:
                apkInfoTree[i][1]['good_cluster_no'] = good_cluster_no_count
                good_cluster_no_dict[apkInfoTree[i][1]['cluster_no']] = good_cluster_no_count
                good_cluster_no_count += 1
            else:
                apkInfoTree[i][1]['good_cluster_no'] = good_cluster_no_dict[apkInfoTree[i][1]['cluster_no']]
            apkInfoTree_good.append(apkInfoTree[i])
        elif cluster_show_dict[apkTree[i]['cluster_no']] == 'pass':
            if apkInfoTree[i][1]['cluster_no'] not in pass_cluster_no_dict:
                apkInfoTree[i][1]['good_cluster_no'] = pass_cluster_no_count
                pass_cluster_no_dict[apkInfoTree[i][1]['cluster_no']] = pass_cluster_no_count
                pass_cluster_no_count += 1
            else:
                apkInfoTree[i][1]['good_cluster_no'] = pass_cluster_no_dict[apkInfoTree[i][1]['cluster_no']]
            apkInfoTree_pass.append(apkInfoTree[i])

    
        
    # 给异常控件排序，使用函数,p给异常控件加入报表的信息
    cluster_info_dict = displayInfoInjetor(DBNAME,apkInfoTree_good,projectName,subtask_name,False)
    # 统计命中信息
    coverIndexStatistics(apkInfoTree_good,cluster_info_dict,projectName, subtask_name)

    for i in range(0,len(apkInfoTree_pass)):
        apkInfoTree_pass[i][1]['good_cluster_no'] += good_cluster_no_count

    apkInfoTree_good.extend(apkInfoTree_pass)
    
    context = {'BaseDir':config['INPUT_DATA_DIR'],'staticPath':STATIC_PATH,'picturePath':PICTURE_PATH, 'language':'zh','current_db':DBNAME,'cover_icon_csv_path':COVER_ICON_CSV_PATH,\
    'metadata':metadata[0], 'apkInfoTree_good':apkInfoTree_good, 'origin_cluster_nums':metadata[0]['clusters'],'show_cluster_nums':good_cluster_no_count,'averAPIs':len(rawapkTree[0]['api_type']),'cluster_show_dict':cluster_show_dict,'apkInfoTree_pass':apkInfoTree_pass,'pass_cluster_nums':pass_cluster_no_count,'total_cluster_nums':good_cluster_no_count+pass_cluster_no_count}
    return context

def find(path,file_name,findloc='local'):
    if findloc == 'local':
        import os
        result = os.popen('find %s -name %s' % (path,file_name))
        return result.read()
    elif findloc == 'db':
        path 
        global DBNAME
        dataBaseProcessor = loadDataBase(CONFIG_PATHS)
        db = dataBaseProcessor.getConnection()
        collect_rawapkforest = db[path]
        rawapkTree = collect_rawapkforest.find({'path':re.compile(file_name)},{'path':1}).sort("_id",1)
        count = 0
        for item in rawapkTree:
            print('*********:',item)
            count += 1
        if count > 0:
            return True
        else:
            return False
    #os.system('find /data/wangruifeng/datasets/DroidBot_Epoch/raw_data/db_universal_set/input_data_origin/ -name view_83a89454cfb3eaad6b69218a01725efe.png')

def isIconInDataSet(icon_path,icon_name):
    # 返回两个值: 只有在本簇不cover时时，在pureapkforest可找到,在rawapkforest可找到,才在db_universal_set 可找到,在origin可找到，将地址呈出
    in_pureapkforest = find('pureapkforest',icon_name,'db')
    in_rawapkforest = find('rawapkforest',icon_name,'db')
    in_universal_set = find('/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/db_universal_set/input_data_origin/',icon_name,'local')
    in_origin_set = find('/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/db_universal_set-origin/',icon_name,'local')
    return [in_pureapkforest, in_rawapkforest, in_universal_set, in_origin_set]

def subtask(request, project_name='MW_pca', subtask_name='spm_pca300_optics3'):
    #根据需要选择HttpRespone或者HttpResponse
    #dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    #dataBaseProcessor.project_name = project_name
    #dataBaseProcessor.subtask_name = subtask_name
    #dataBaseProcessor.configTableName = dataBaseProcessor.project_name+'_config'
    #dataBaseProcessor.metaDataTableName = dataBaseProcessor.project_name + '_metadata'        
    #dataBaseProcessor.apkTreeTableName = dataBaseProcessor.subtask_name+'_apktree'         
    #config = dataBaseProcessor.getConfig({'INSTANCE_NAME':project_name})
    #metadata = dataBaseProcessor.getConfig({'subtask_name':subtask_name})
    #context = {'configLists': dataBaseProcessor.getConfig()}
    interpreter = loadInterpreter(CONFIG_PATHS, project_name, subtask_name)
    config = interpreter.config
    parsingConfig = interpreter.ParsingConfig()
    metadata = interpreter.DBP.queryMetaData({'project_name':1, 'subtask_name':1, 'updatetime':1, "ifdim":1, "ifmethod":1, "imagesize":1, "rawdims":1, "cutdimway":1, "reducedims":1, "calinski" :1, "clustermethod":1, "clusters":1, "silhouette":1}) 
    metadata[0]['projectName'] = project_name
    print('metadata', metadata)
    print('config', config)

    canPreview = parsingConfig['CLUSTER_PICTURE_RESULT_DIR']['isExist']
    if canPreview == False: canPreview = ' disabled="disabled"'
    else: canPreview = ''

    # 20210424 New Add ,View Cover
    # Read total Cover txt

    # if cover , mark as Green , if in meaningful cluster , mark as  , if not in meaningful cluster ,mark as red
    # 遍历all_coverlist的同时检查是否在coverlist_csv中，，是true是false
    # 最后返回给页面的
    total_count = covered_count = to_cover_count =0
    covered_ratio = to_cover_ratio = 0

    show_cover_list = []

    # coverlist的构建
    coverlist_csv = COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'.csv'
    all_coverlist_csv = COVER_ICON_TXT_PATH
    if os.path.exists(coverlist_csv) and os.path.exists(all_coverlist_csv):
        coverdict = dict()
        with open(coverlist_csv,'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                coverdict[row[0]] = row

        need_cover_icon = readTxt(all_coverlist_csv)
        for item in need_cover_icon:
            icon_path,replay = item.split(',')
            icon_name = icon_path.split('/')[-1]
            # 原始簇为-1，不纳入
            if icon_name in coverdict and coverdict[icon_name][8] == '-1':
                #coverdict[icon_name][2]=coverdict[icon_name][5]=coverdict[icon_name][6]=coverdict[icon_name][7]='/'
                show_cover_list.append(['red',icon_path] + coverdict[icon_name] + [replay] + isIconInDataSet(icon_path,icon_name))
            elif icon_name in coverdict and coverdict[icon_name][2] == 'True': 
                show_cover_list.append(['green',icon_path] + coverdict[icon_name] + [replay])
                covered_count += 1
            elif icon_name in coverdict and coverdict[icon_name][2] == 'False': 
                show_cover_list.append(['blue',icon_path] + coverdict[icon_name] + [replay])
                to_cover_count += 1
            else:
                show_cover_list.append(['red',icon_path,icon_name]+['/']*8 + [replay] + isIconInDataSet(icon_path,icon_name))
            total_count += 1
        covered_ratio = covered_count/total_count
        to_cover_ratio = to_cover_count/total_count
        cover_staticis_dict = {'covered_count':covered_count,'to_cover_count':to_cover_count,'covered_ratio':covered_ratio,'to_cover_ratio':to_cover_ratio,'total_count':total_count}
        #print(show_cover_list)
    else:
        #  尝试构建
        print('尚未进行cover统计')

    context = {'metadata':metadata[0], 'configLists':parsingConfig, 'staticPath':STATIC_PATH, 'language':'zh', 'clusterPictureDir':config['CLUSTER_PICTURE_RESULT_DIR'], 'canPreview':canPreview,'coverlist':show_cover_list,'PICTURE_PATH':PICTURE_PATH,"cover_staticis_dict":cover_staticis_dict}
    #return JsonResponse({'message':'Hello world'})
    return render(request, 'Console/subtask.html', context)
    #return HttpResponse("Hello world")

def remarkSubtask(request):
    # 通过config获取metadata即可,在展示subtaskremark时给出remark的链接
    remark = request.GET.get('remark')
    project_name = request.GET.get('project_name')
    subtask_name = request.GET.get('subtask_name')
    print('remark,project_name,subtask_name',remark,project_name,subtask_name)
    
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    dataBaseProcessor.projectName = project_name
    dataBaseProcessor.subtask_name = subtask_name
    dataBaseProcessor.configTableName = dataBaseProcessor.projectName+'_config'
    dataBaseProcessor.metaDataTableName = dataBaseProcessor.projectName + '_metadata'        
    dataBaseProcessor.apkTreeTableName = dataBaseProcessor.subtask_name+'_apktree'         
    
    print('=>',remark)
    if remark != '':
        dataBaseProcessor.updateMetaDataRemark(remark)
    #    updateMetaData({'$set':{'remark':remark}})
    res = {'msg':'success','remark':remark}
    return JsonResponse(res,json_dumps_params={'ensure_ascii': False})
    
# #<<Projects---------------------------------------------------------------------------------------------------------------

# #>>DataSets---------------------------------------------------------------------------------------------------------------
def dataSets(request):
    from . import views_datasets
    from importlib import reload
    reload(views_datasets)
    classification_Video_Dataset = views_datasets.ClassificationDataSet()
    return classification_Video_Dataset.showDataSetHome(request)

def createDataSetByChoicen(request,database=DBNAME):
    from . import views_datasets
    from importlib import reload
    reload(views_datasets)
    classification_Video_Dataset = views_datasets.ClassificationDataSet()
    return classification_Video_Dataset.createDataSetByChoicen(request,database)

def submitDataSetChoicen(request,database=DBNAME):
    from . import views_datasets
    from importlib import reload
    reload(views_datasets)
    classification_Video_Dataset = views_datasets.ClassificationDataSet()
    return classification_Video_Dataset.submitDataSetChoicen(request,database)

# #>>DataSets---------------------------------------------------------------------------------------------------------------

# #>>Gallerys---------------------------------------------------------------------------------------------------------------
## Console/gallerys.html
# path('gallerys',views.gallerys),
def gallerys(request):
    # 获取所有 parsingConfig['CLUSTER_PICTURE_RESULT_DIR']['isExist'] 为True的项目子任务
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    db = dataBaseProcessor.getConnection()
    collist = db.collection_names()

    no = 1
    subtaskDictList = list()
    # 遍历数据表
    for col in collist:
        # filter all _metadata
        if '_metadata' not in col :
            continue
        # 这条分支是能获取到信息的
        # 对于每个项目数据表遍历其内容
        for subtask in db[col].find():
            subtaskDict = dict()
            subtaskDict['no'] = no
            subtaskDict['projectName'] = col.replace('_metadata', '')
            subtaskDict['subtaskName'] = subtask['subtask_name']
            subtaskDict['updateTime'] = subtask['updatetime']
            subtaskDict['cluster_link'] = '/Console/album/%s/%s/cluster/none' % (subtaskDict['projectName'], subtaskDict['subtaskName'])
            subtaskDict['outlier_link'] = '/Console/album/%s/%s/outlier/none' % (subtaskDict['projectName'], subtaskDict['subtaskName'])
            subtaskDictList.append(subtaskDict)
            no +=1
    context = {'subtaskDictList':subtaskDictList, 'staticPath':STATIC_PATH}
    return render(request, 'Console/gallerys.html', context)

# #<<Gallerys---------------------------------------------------------------------------------------------------------------

# #>>Tags---------------------------------------------------------------------------------------------------------------
# path('tags',views.tags),
# Console/album_outlier.html
# path('album/<str:projectName>/<str:subtask_name>/<str:galleryId>/<str:hightlightClusterId>/<int:cluster_no>',views.album),
# path('tagging/<str:projectName>/<str:subtask_name>/<str:widget_id>',views.tagging),
# # Console/evaluate.html
# path('evaluateHome',views.evaluateHome),
# path('evaluate/<str:projectName>/<str:subtask_name>',views.evaluate),
# path('evaluateSubmit',views.evaluateSubmit),
# 实际上不需要，但是先这样
# path('submitSuspects',views.submitSuspects),

def resoluteAPI(apis,stanard_android_api,method='simple'):
    if method == 'custom':
        api_type_set = set()
        for api in apis:
            package,status = isAtPackageList(api,stanard_android_api)
            if status == False:
                api_type_set.add('other')
                continue
            print('----->',package.split('.'))
            print('----->',package.split('.')[1:2])
            api_type = '.'.join(package.split('.')[1:2])#'.'.join(api.split('.')[1:3])
            # 将来可以对包建立一个map库对应起来
            api_type_set.add(api_type[:10])
        return list(api_type_set)
    else:
        api_type_set = set()
        for api in apis:
            api_type = '.'.join(api.split('.')[1:3])
            # 将来可以对包建立一个map库对应起来
            api_type_set.add(api_type[:21])
        return list(api_type_set)

def isCanShow(apps_dict):
    # apps_dict: {'app1':2,'app2':3,'app3'1}
    #### app数量必须大于等于3  小于5时需要求任意app占比不超过66%(0 1 1 ,0 0 1 1)
    app_num = len(list(apps_dict.keys()))
    #if app_num < 2:
    #    return 'bad'
    if app_num < 2:
        return 'pass'

    ### 总控件数量必须大于等于5
    widget_num_list = list(apps_dict.values())
    widget_num = sum(widget_num_list)
    #if widget_num < 3:
    #    return 'bad'
    if widget_num < 3:
        return 'pass'

    ### 控件的占比应该小于33%(最坏情况，只有三个，每个占比33.33%)
    '''
    widget_num_list = [i/widget_num for i in widget_num_list]
    for item in widget_num_list: 
        #if item > 0.34:
        #    return 'bad'
        if item > 0.5:
            return 'pass' 
    '''
    '''
    result=0; 
    for x in c: 
        result+=(-x)*math.log(x,2)
    if result < 0.91:
        return False
    '''
    return 'good'

def fetchAppWebsiteInfo(curdbname,app):
    # 首先尝试获取存储的json文件
    global RAW_ROOT
    app_json_path = RAW_ROOT + 'app_json.json'
    try:
        app_json = readDict(app_json_path,'noordered') # 专门有用于构建的项目？
    except:
        app_json = dict()
        global ZIP_ROOT
        # 如果没有，直接读取一个当前DBNAME对应的系统文件夹，生成这种json文件
        zip_json_path = RAW_ROOT + 'app_json.json'
        # 应当从zip中读取
        raw_zip_paths = ['views_info_f_droid_1538/','views_info_google_play_13k_12011736/','views_info_google_play_13k_12021120/','views_info_google_play_13k_12071659/','views_info_google_play_1435_apps_rerun/']
        for raw_folder_path in raw_zip_paths:
            for app_folder_name in os.listdir(ZIP_ROOT+raw_folder_path):
                if 'f_droid' in raw_folder_path:
                    website = 'https://f-droid.org/en/packages/'
                else:
                    website = 'https://play.google.com/store/apps/details?id='
                if app_folder_name not in app_json:
                    app_json[app_folder_name] = dict()
                app_json[app_folder_name]['website'] = website
        writeDict(app_json,app_json_path) # 专门有用于构建的项目？

    # 正常的流程
    app_search_name = app.replace('.apk','')
    app_search_name = re.sub(r"_\d+","",app_search_name)
    website = app_json[app]['website']+app_search_name
    print('fetchAppWebsiteInfo:',website)
    return website

    # jon文件中，各个app对应的库均在其中，得知库后会发送web请求，如果请求无效则更换地址，实在不行在原地址上加上标识
def createGlobalSort(cluster_no_dict,cluster_value_max_num):
    print('[1]:len(cluster_no_dict),cluster_value_max_num:',len(cluster_no_dict),cluster_value_max_num)
    #pointers = [0]*len(cluster_no_dict)
    # 获取到最长的那个
    cluster_no_list = np.full((len(cluster_no_dict),cluster_value_max_num+2),0,dtype='float64')
    count = 0
    for cluster_key,cluster_value in cluster_no_dict.items():
        cluster_no_list_item = np.full(cluster_value_max_num+2,0,dtype='float64')
        cluster_no_list_item[0] = len(cluster_value['content'])
        cluster_no_list_item[1] = int(cluster_key)
        for i in range(0,len(cluster_value['content'])):
            cluster_no_list_item[i+2] = -float(cluster_value['content'][i][1])
        print('[1]/count',count)
        cluster_no_list[count] = cluster_no_list_item
        count+=1

    cluster_no_list = np.array(cluster_no_list)
    rows = np.array(range(0,len(cluster_no_list)))
    # 由于0，1存储了非数据，所以从2号开始
    pointers = np.full(len(cluster_no_dict),2,dtype='int32')
    print('[2]:rows,pointers:\n',rows,'\n',pointers)

    global_sort = 0
    # 遍历所有的元素一遍？
    while True:
    #for cluster_key,cluster_value in cluster_no_dict.items():
        # 首先取出pointers所指示的元素
        # 行可以限定,列用pointers动态地限定
        # 将每个pointer指向的元素取出
        current_scores = cluster_no_list[rows,pointers]
        # 找到最大的值
        max_score = max(current_scores)
        print('[3]:max_score:',max_score)
        # 寻找等于该最大值的下标
        cur_row = np.where(current_scores == max_score)[0]
        #[-0.51,-0.42,-0.55...]
        #                ^
        # 给其标注排序(用np.where下标和pointer赋值)
        # np.where : 3，即第3行，第三个簇；pointer[3] 第3行目前是第pointer[3]个元素参与当前比较
        # 最后取出content的0号元素访问dict回填排序
        # 为了删除某些已经完成的row
        down_row = []
        for row in cur_row:
            row = int(row)
            print('[4]:外层循环',row,'||',cur_row)
            # 回填sort
            # 每个簇的簇号
            cluster_no = int(cluster_no_list[rows[row]][1])
            # 每个簇的控件数量
            cluster_num = cluster_no_list[rows[row]][0]
            # 将对应的簇中控件加上global_sort排序
            while True:
                widget_col = pointers[row]-2 
                if -float(cluster_no_dict[cluster_no]['content'][widget_col][1]) == max_score:
                    cluster_no_dict[cluster_no]['content'][widget_col].append(global_sort)
                    # pointer指向下一个
                    if widget_col+1 == cluster_num:
                        # 这个不行了，rows去除即可
                        down_row.append(row)
                        break
                    else:
                        pointers[row] += 1
                else:
                    break
            #                某行              |    该列          | 某
        # 上面是一次结束，应该重新统计sort和rows
        #rows = np.setdiff1d(rows,down_row)
        down_row.reverse()
        for row in down_row:
            rows = np.delete(rows,row,0)
            pointers = np.delete(pointers,row,0)
        if len(rows) == 0:
            break;
        global_sort += 1
    max_global_sort = global_sort
    return cluster_no_dict,max_global_sort
def displayInfoInjetor(curdbname,apkInfoTree,project_name,subtask_name,checkwebsite=True):
    # rawapkTree: {'image_cluster_no':{"$ne":'-2'}}, {"path" : 1, "app" : 1, "widget" : 1,"method_api":1, "suspect":1}
    # apkTree: {"raw_id" : 1, "cluster_no" :1, "outlier_score" :1}
    # apkInfoTree.append((rawapkTree[i], apkTree[i])
    # cluster_no_dict: {'0':[(tree_item[0]['widget'],tree_item[1]['outlier_score'],icon_name,tree_item[1]['new_cluster_no']),
    #                        (tree_item[0]['widget'],tree_item[1]['outlier_score'],icon_name,tree_item[1]['new_cluster_no'])]}
    #                  {'1':[(tree_item[0]['widget'],tree_item[1]['outlier_score'],icon_name,tree_item[1]['new_cluster_no']),
    #                        (tree_item[0]['widget'],tree_item[1]['outlier_score'],icon_name,tree_item[1]['new_cluster_no'])]}
    ##### 用于给异常排序，对于每一个簇号相同的一起排序
    # 首先将所有cluster_no相同的找出，然后逐步的给他们动态排序
    cluster_no_dict = dict()
    for tree_item in apkInfoTree:
        # 当前控件的cluster_no
        # ????????????
        cur_cluster_no = tree_item[1]['good_cluster_no']
        icon_name = tree_item[0]['path'].split('/')[-1]
        if tree_item[1]['cluster_no'] == '-1':
            continue
        if cur_cluster_no in cluster_no_dict:
            cluster_no_dict[cur_cluster_no]['indexes'].append(tree_item[0]['widget'])
            cluster_no_dict[cur_cluster_no]['content'].append([tree_item[0]['widget'],tree_item[1]['outlier_score'],icon_name,tree_item[1]['good_cluster_no'],tree_item[1]['cluster_no'],tree_item[0]['app']])
        else:
            cluster_no_dict[cur_cluster_no] = dict()
            cluster_no_dict[cur_cluster_no]['indexes'] = [tree_item[0]['widget']]
            cluster_no_dict[cur_cluster_no]['content'] = [[tree_item[0]['widget'],tree_item[1]['outlier_score'],icon_name,tree_item[1]['good_cluster_no'],tree_item[1]['cluster_no'],tree_item[0]['app']]]
        
    cluster_no_dict_threholds = dict()

    cluster_no_list = []
    cluster_value_max_num = 0
    for cluster_key,cluster_value in cluster_no_dict.items():
        cluster_no_dict[cluster_key]['content'].sort(key=lambda x:float(x[1]))
        if cluster_value_max_num < len(cluster_no_dict[cluster_key]['content']):
            cluster_value_max_num = len(cluster_no_dict[cluster_key]['content'])
        cluster_no_list.append([cluster_key]+cluster_no_dict[cluster_key]['content'])
        # 给控件上色
        color_dict = dict()
        color_index = 0
        cluster_outlier_list = []
        for i in range(len(cluster_no_dict[cluster_key]['content'])):
            item = cluster_no_dict[cluster_key]['content'][i] 
            if item[4] == '-1':
                continue
            # Create the indexes
            cluster_no_dict[cluster_key]['indexes'][i] = item[0]

            cluster_outlier_list.append(item)
            if item[5] in color_dict:
                cluster_no_dict[cluster_key]['content'][i][5] = color_dict[item[5]]
            else:
                color_dict[item[5]] = color_index
                cluster_no_dict[cluster_key]['content'][i][5] = color_index
                color_index += 1
        cluster_no_dict_threholds[cluster_key] = getCheckoutIndex(cluster_no_dict[cluster_key]['content'])

                
    cluster_no_dict,max_global_sort = createGlobalSort(cluster_no_dict,cluster_value_max_num)

    ####### 添加的功能:已经被cover的控件显示颜色
    coverdict = dict()
    inspected_app_set = set()
    

    coverlist_csv = COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'.csv'
    current_inspected_app_set_path = CURRENT_INSPECTED_APP_SET_PATH

    if os.path.exists(coverlist_csv):
        with open(coverlist_csv,'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                coverdict[row[0]] = row

    if os.path.exists(current_inspected_app_set_path):
        inspected_app_set = set(open(current_inspected_app_set_path,'r').read().rstrip('\n').split('\n'))
        
    #coverlist = []
    #if os.path.exists(coverlist_csv):
    #    with open(coverlist_csv,'r') as f:
    #        f_csv = csv.reader(f)
    #        [coverlist.append(row[0]) for row in f_csv]
    ##### 异常排序部署阶段；插入报表信息
    for i in range(len(apkInfoTree)):
        tree_item = apkInfoTree[i]
        # 当前控件的cluster_no
        cur_cluster_no = tree_item[1]['good_cluster_no']
        if tree_item[1]['cluster_no'] == '-1':
            continue
        icon_name = tree_item[0]['path'].split('/')[-1]
        # cluster_no_dict[cur_cluster_no] => [('2021_034342','-0.45'),('2021_034342','-0.45')]
        cur_widget_index = cluster_no_dict[cur_cluster_no]['indexes'].index(tree_item[0]['widget'])
        #,tree_item[1]['outlier_score'],icon_name,tree_item[1]['new_cluster_no'],tree_item[1]['cluster_no'],tree_item[0]['app']))
        apkInfoTree[i][1]['color_index_in_cluster'] = cluster_no_dict[cur_cluster_no]['content'][cur_widget_index][5]
        apkInfoTree[i][1]['global_outlier_sort'] = cluster_no_dict[cur_cluster_no]['content'][cur_widget_index][6]
        apkInfoTree[i][1]['global_outlier_sort_ratio'] = cluster_no_dict[cur_cluster_no]['content'][cur_widget_index][6]/max_global_sort
        cluster_no_dict[cur_cluster_no]['content'][cur_widget_index][6] = apkInfoTree[i][1]['global_outlier_sort_ratio'] 
        print('==>global_sort:',apkInfoTree[i][1]['global_outlier_sort'])
        # content , indexs has same length
        cluster_len = len(cluster_no_dict[cur_cluster_no]['content'])
        apkInfoTree[i][1]['outlier_sort'] = cur_widget_index
        apkInfoTree[i][1]['outlier_percent_sort'] = (cur_widget_index+1)/cluster_len

        if cur_widget_index <= cluster_no_dict_threholds[cluster_key][1]:
            apkInfoTree[i][0]['top20p'] = 'top20p'
        # 如果在整个需cover中出现
        if (icon_name in coverdict) and (coverdict[icon_name][8] == '-1'):
            pass
        elif (icon_name in coverdict) and coverdict[icon_name][2] == 'True': 
            apkInfoTree[i][0]['top20p'] = ''
            apkInfoTree[i][0]['covered'] = 'covered'
        elif (icon_name in coverdict) and coverdict[icon_name][2] == 'False': 
            apkInfoTree[i][0]['top20p'] = ''
            apkInfoTree[i][0]['covered'] = 'to_cover'

        if tree_item[0]['app'] in inspected_app_set:
            apkInfoTree[i][0]['inspected'] = 'inspected'
        else:
            apkInfoTree[i][0]['inspected'] = 'button_text'
        
        # 一些用于报表的属性
        if checkwebsite == True:
            apkInfoTree[i][1]['appwebsite'] = fetchAppWebsiteInfo(DBNAME,tree_item[0]['app'])
    return cluster_no_dict

def getCheckoutIndex(cluster_list):
    # 给定一个簇列表，得出其Top1/3/5,Top20%,therhold>4.9,therhold_outlier>0.55的临界下标
    # ['-0.66', '-0.44', '-0.34', '-0.34']
    top_n_index = 0
    top_20percent_index = 0
    threshold = 0.49
    threshold_index = 0
    threshold_outlier_threshold = 0.55
    threshold_outlier_threshold_index = 0
    
    # 先按数学序数计算，最后统一减去1，注意本身就是0
    cluster_length = len(cluster_list)
    if cluster_length <= 5:
        top_n_index = 1
    elif cluster_length <= 10:
        top_n_index = 3
    else:
        top_n_index = 5
    top_20percent_index = int(cluster_length * 0.2)
    # top20至少要有1个
    if top_20percent_index < 1: top_20percent_index = 1
    # 这是里面最后一个大于0.49的后一个项的下标，所以减1之后是最后一个大于0.49的
    # 如果没有大于0.49的，threshold_index=-1
    for index in range(cluster_length):
        #print(cluster_list,'\n',cluster_list[index],'\n',threshold)
        if -float(cluster_list[index][1]) < threshold:
            threshold_index = index-1
            break

    return top_n_index-1,top_20percent_index-1,threshold_index

# 读取统计信息
def readCoverIndexStatistics(project_name,subtask_name):
    coverlist_csv = COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'.csv'
    if os.path.exists(coverlist_csv):
        with open(coverlist_csv,'r') as f:
            f_csv = csv.reader(f)
            is_top_n_hit_count =  is_top_20percent_hit_count = is_threshold_hit_count =  is_hit_count = 0
            for row in f_csv:
            #=> ['控件icon名', '包含簇号', '是否命中', '簇内排序', '簇内控件总数', 'top-1,3,5命中', 'top-20%命中', 'Threshold0.49命中']
            #=> ['view_ba27a1df2a40cc2f63697dfc71f86c79.png', '6', 'True', '1', '10', 'True', 'True', 'True']
                #print('=>',row)
                if row[1] == '包含簇号' or row[8] == '-1':
                    continue
                if row[2]=='True':is_hit_count+=1
                if row[5]=='True':is_top_n_hit_count+=1
                if row[6]=='True':is_top_20percent_hit_count+=1
                if row[7]=='True':is_threshold_hit_count+=1
        return is_top_n_hit_count ,  is_top_20percent_hit_count , is_threshold_hit_count ,  is_hit_count , STATIC_PATH+coverlist_csv
    else:
        return ['-']*5

def readGlobalCoverIndexStatistics(project_name,subtask_name):
    global_coverlist_csv = COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'global.csv'
    if os.path.exists(global_coverlist_csv):
        with open(global_coverlist_csv,'r') as f:
            f_csv = csv.reader(f)
            top10p = top20p = top30p = 0
            for row in f_csv:
                if row[1] == '包含簇号' or row[5] == '-1':
                    continue
                if float(row[2]) <= 0.1:top10p+=1
                elif float(row[2]) <= 0.2:top20p+=1
                elif float(row[2]) <= 0.3:top30p+=1
        return top10p,top20p,top30p
    else:
        return ['-']*3
                

# 对当前控件cover的好坏
def coverIndexStatistics(apkInfoTree,cluster_no_dict,project_name,subtask_name):
    print('^^^^^^^正在统计命中信息^^^^^^^')
    # 如果这个项目的已经有了，不需要了
    if os.path.exists(COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'.csv') and os.path.exists(COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'global.csv'):
        return '' 
    # 首先读出needcover的控件名，注意只需要最后.png即可
    need_cover_icon = readTxt(COVER_ICON_TXT_PATH)
    for i in range(len(need_cover_icon)):
        icon_path = need_cover_icon[i].split(',')[0]
        need_cover_icon[i] = icon_path.split('/')[-1]
    cover_icon_csv = []
    global_cover_icon_csv = []
    #is_top_n_hit_count =  is_top_20percent_hit_count = is_threshold_hit_count =  is_hit_count = 0
    # 对于每一个簇，检查是否具有列表中的icon
    # cluster_no_dict[cur_cluster_no].append((tree_item[0]['widget'],tree_item[1]['outlier_score'],tree_item[0]['path'].split('/')[-1]))
    for cluster_key,cluster_value in cluster_no_dict.items():
        # 每个cluster_value 均是一个簇，簇由列表构成，按顺序排列
        # 可以开发一个函数专门获取每个数组中top几的下标，比较need_cover_icon中出现的下标与这下下标即可
        for icon_item_index in range(len(cluster_no_dict[cluster_key]['content'])):
            cluster_list = cluster_no_dict[cluster_key]['content']
            # 在need_cover_icon内则统计: 如有在其中，统计它是否在Top1/3/5,Top20%,therhold>4.9,therhold_outlier>0.55中，记录在csv中
            # 注意：原始簇号为-1的不统计
            # 首先统计global
            if (cluster_list[icon_item_index][2] in need_cover_icon):
                # 如果不是无意义簇的话
                if (cluster_list[icon_item_index][4] != '-1'): 
                    # ratio < 10%?
                    global_cover_icon_csv.append([cluster_list[icon_item_index][2],cluster_list[icon_item_index][3],cluster_list[icon_item_index][6],icon_item_index,len(cluster_no_dict[cluster_key]['content']),cluster_list[icon_item_index][4]])
                    '''
                    if cluster_list[icon_item_index][6] <= 0.1:
                    elif cluster_list[icon_item_index][6] <= 0.2:
                        global_cover_icon_csv.append([cluster_list[icon_item_index][2],cluster_list[icon_item_index][3],'20%',icon_item_index,len(cluster_no_dict[cluster_key]['content']),cluster_list[icon_item_index][4]])
                    elif cluster_list[icon_item_index][6] <= 0.3:
                        global_cover_icon_csv.append([cluster_list[icon_item_index][2],cluster_list[icon_item_index][3],'30%',icon_item_index,len(cluster_no_dict[cluster_key]['content']),cluster_list[icon_item_index][4]])
                    '''
            if (cluster_list[icon_item_index][2] in need_cover_icon):
                if (cluster_list[icon_item_index][4] == '-1'): 
                    is_hit = is_top_n_hit = is_top_20percent_hit = is_threshold_hit = '-'
                    cover_icon_csv.append([cluster_list[icon_item_index][2],cluster_list[icon_item_index][3],is_hit,icon_item_index,len(cluster_no_dict[cluster_key]['content']),is_top_n_hit,is_top_20percent_hit,is_threshold_hit,cluster_list[icon_item_index][4]])
                else:
                    top_n_index, top_20percent_index,threshold_index = getCheckoutIndex(cluster_list)
                    is_top_n_hit = icon_item_index <= top_n_index
                    is_top_20percent_hit = icon_item_index <= top_20percent_index
                    is_threshold_hit = icon_item_index <= threshold_index
                    '''
                    if is_hit_count:is_hit_count+=1
                    if is_top_n_hit:is_top_n_hit+=1
                    if is_top_20percent_hit:is_top_20percent_hit+=1
                    if is_threshold_hit:is_threshold_hit_count+=1
                    '''
                    is_hit = is_top_n_hit or is_top_20percent_hit or is_threshold_hit
                    cover_icon_csv.append([cluster_list[icon_item_index][2],cluster_list[icon_item_index][3],is_hit,icon_item_index,len(cluster_no_dict[cluster_key]['content']),is_top_n_hit,is_top_20percent_hit,is_threshold_hit,cluster_list[icon_item_index][4]])
            
    headers = ['控件icon名','包含簇号','是否命中','簇内排序','簇内控件总数','top-1,3,5命中','top-20%命中','Threshold0.49命中','原始簇号']
    with open(COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(cover_icon_csv)

    headers = ['控件icon名','包含簇号','命中百分比','簇内排序','簇内控件总数','原始簇号']
    with open(COVER_ICON_CSV_PATH+project_name+'.'+subtask_name+'global.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(global_cover_icon_csv)
    #return is_top_n_hit_count ,  is_top_20percent_hit_count , is_threshold_hit_count ,  is_hit_count

#### view_clusters
def viewClusters(request,project_name='',subtask_name=''):

    context = loadfromApkTrees(project_name,subtask_name)
    return render(request, 'Console/evaluateCluster.html', context)


# 评测目前主要看的函数#$#$#$,簇视图
# 1. 需要将控件的异常排名搞出来
def album(request, projectName='MW_lle', subtask_name='spm_lle150_optics3', galleryId='',hightlightClusterId='',cluster_no=0):
    global DBNAME
    if hightlightClusterId == 'none':
        hightlightCluster = hightlightId = ''
    else:
        hightlightCluster,hightlightId = hightlightClusterId.split('.')
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    dataBaseProcessor.projectName = projectName
    dataBaseProcessor.subtask_name = subtask_name
    dataBaseProcessor.configTableName = dataBaseProcessor.projectName+'_config'
    dataBaseProcessor.metaDataTableName = dataBaseProcessor.projectName + '_metadata'        
    dataBaseProcessor.apkTreeTableName = dataBaseProcessor.projectName+'__'+dataBaseProcessor.subtask_name+'_apktree'         
    #print('what??', dataBaseProcessor.configTableName , dataBaseProcessor.metaDataTableName, dataBaseProcessor.apkTreeTableName)
    config = dataBaseProcessor.getConfig()
    metadata = dataBaseProcessor.queryMetaData({'project_name':1, 'subtask_name':1, 'updatetime':1, "ifdim":1, "ifmethod":1, "imagesize":1, "rawdims":1, "cutdimway":1, "reducedims":1, "calinski" :1, "clustermethod":1, "clusters":1, "silhouette":1,"rawapkforestname":1,"remark":1}) 
    dataBaseProcessor.rawApkForestName = metadata[0]['rawapkforestname']
    print('1',dataBaseProcessor.rawApkForestName)
    rawapkTree = dataBaseProcessor.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}}, {"path" : 1, "app" : 1, "widget" : 1,"method_api":1, "suspect":1})
    print('2',dataBaseProcessor.rawApkForestName)
    apkTree = dataBaseProcessor.queryAPKTree({}, {"raw_id" : 1, "cluster_no" :1, "outlier_score" :1})
    print('3',dataBaseProcessor.rawApkForestName)
    print('4',len(rawapkTree),len(apkTree))
    #print(rawapkTree)
    #print(apkTree)
    #print('what?',metadata)
    #stanard_android_api = readTxt('/home/dl/users/wangruifeng/05MisleadingWidgets/MisleadingWidgetsDetective/MisleadingWidgetsDetective/type.txt')
    #stanard_android_api = readTxt('/core/kernel/01work/02project/system/AndroidUIUnderstanding/05MisleadingWidget/process/MisleadingWidgetsDetective/MisleadingWidgetsDetective/type.txt')

    apkInfoTree = []
    rawapkTree[0]['api_type'] = []
    # 将簇信息存入一个字典，这样可以统计出每个簇有多少个控件，分属于多少应用
    cluster_info_dict = dict()
    cluster_show_dict = dict()
    app_statisic_dict = dict()
    # 将rawAPITree中的信息补充到apkTree当中，最后存入到apkInfoTree中
    for i in range(len(rawapkTree)):
        # 用于统计各簇的数量>>>>
        # 会有各簇下各应用数量的统计,每次更新时都检查状态，如果一个簇内数量>5个，表为可以显示can_show
        # 同时统计每个簇内应用纯度(熵)，低于某个纯度则标为low_app_abundance.即熵越小，不确定性越小，丰度越低，越是不要显示
        if apkTree[i]['cluster_no'] not in cluster_info_dict:
            cluster_info_dict[apkTree[i]['cluster_no']] = dict()
            cluster_info_dict[apkTree[i]['cluster_no']][rawapkTree[i]['app']] = 1
        else:
            if rawapkTree[i]['app'] not in cluster_info_dict[apkTree[i]['cluster_no']]:
                cluster_info_dict[apkTree[i]['cluster_no']][rawapkTree[i]['app']] = 1
            else:
                cluster_info_dict[apkTree[i]['cluster_no']][rawapkTree[i]['app']] +=1
        cluster_show_dict[apkTree[i]['cluster_no']] = isCanShow(cluster_info_dict[apkTree[i]['cluster_no']])
        # 
        # 统计各簇中控件数量结束<<<<
        if rawapkTree[i]['_id'] != apkTree[i]['raw_id']:
            print('Wrong!!')
        apkTree[i]['isOutlier'] = ''
        apkTree[i]['outlier_score'] = apkTree[i]['outlier_score'][:5]
        #if float(apkTree[i]['outlier_score']) <= -0.55:
        if float(apkTree[i]['outlier_score']) <= 1:
        #if float(apkTree[i]['outlier_score']) > 3:
            apkTree[i]['isOutlier'] = 'outlier'
        rawapkTree[i]['widget'] = ''.join(rawapkTree[i]['widget'].split(', '))
        apkTree[i]['id'] = str(apkTree[i]['raw_id'])

        if apkTree[i]['id'] == hightlightId and apkTree[i]['cluster_no'] == hightlightCluster:
            apkTree[i]['hightlight'] = 'hightlight'
        if galleryId == 'evaluate':
            rawapkTree[i]['api_type'] = rawapkTree[i]['method_api']#resoluteAPI(rawapkTree[i]['api'],stanard_android_api,'simple')#
            rawapkTree[i]['api_string'] = '  '.join(rawapkTree[i]['method_api'])
            apkInfoTree.append((rawapkTree[i], apkTree[i]))


    if galleryId == 'evaluate':
        apkInfoTree_good = []
        good_cluster_no_dict = dict()
        good_cluster_no_count = 0
        apkInfoTree_pass = []
        pass_cluster_no_dict = dict()
        pass_cluster_no_count = 0
        for i in range(len(rawapkTree)):
            if cluster_show_dict[apkTree[i]['cluster_no']] == 'good':
                if apkInfoTree[i][1]['cluster_no'] not in good_cluster_no_dict:
                    apkInfoTree[i][1]['good_cluster_no'] = good_cluster_no_count
                    good_cluster_no_dict[apkInfoTree[i][1]['cluster_no']] = good_cluster_no_count
                    good_cluster_no_count += 1
                else:
                    apkInfoTree[i][1]['good_cluster_no'] = good_cluster_no_dict[apkInfoTree[i][1]['cluster_no']]
                apkInfoTree_good.append(apkInfoTree[i])
            elif cluster_show_dict[apkTree[i]['cluster_no']] == 'pass':
                if apkInfoTree[i][1]['cluster_no'] not in pass_cluster_no_dict:
                    apkInfoTree[i][1]['pass_cluster_no'] = pass_cluster_no_count
                    pass_cluster_no_dict[apkInfoTree[i][1]['cluster_no']] = pass_cluster_no_count
                    pass_cluster_no_count += 1
                else:
                    apkInfoTree[i][1]['pass_cluster_no'] = pass_cluster_no_dict[apkInfoTree[i][1]['cluster_no']]
                apkInfoTree_pass.append(apkInfoTree[i])
    else:
        apkInfoTree_good = apkInfoTree
        good_cluster_no_count = 0

    # 给异常控件排序，使用函数,p给异常控件加入报表的信息
    cluster_info_dict = displayInfoInjetor(DBNAME,apkInfoTree_good,projectName,subtask_name)
    # 统计命中信息
    coverIndexStatistics(apkInfoTree_good,cluster_info_dict,projectName, subtask_name)

    context = {'cluster_no':cluster_no,'BaseDir':config['INPUT_DATA_DIR'], 'metadata':metadata[0], 'apkInfoTree':apkInfoTree_good, 'apkTree':apkTree, 'max_cluster':metadata[0]['clusters'], 'cluster_num':range(int(metadata[0]['clusters'])),'new_max_cluster':good_cluster_no_count, 'staticPath':STATIC_PATH,'picturePath':PICTURE_PATH, 'language':'zh','hightlightCluster':hightlightCluster,'averAPIs':len(rawapkTree[0]['api_type']),'current_db':DBNAME,'cluster_show_dict':cluster_show_dict,'cover_icon_csv_path':COVER_ICON_CSV_PATH,'passApkInfoTree':apkInfoTree_pass,'pass_max_cluster':pass_cluster_no_count}
    #print(context)

    if galleryId == 'outlier':
        return render(request, 'Console/album_outlier.html', context)
    elif galleryId == 'tagging':
        return render(request, 'Console/album_tagging.html', context)
    elif galleryId == 'evaluate':
        return render(request, 'Console/evaluateCluster.html', context)
    else:
        return render(request, 'Console/album_cluster.html', context)


def tags(request):
    # 获取所有 parsingConfig['CLUSTER_PICTURE_RESULT_DIR']['isExist'] 为True的项目子任务
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    db = dataBaseProcessor.getConnection()
    collist = db.collection_names()

    no = 1
    subtaskDictList = list()
    # 遍历数据表
    for col in collist:
        # filter all _metadata
        if '_metadata' not in col :
            continue
        # 这条分支是能获取到信息的
        # 对于每个项目数据表遍历其内容
        for subtask in db[col].find():
            subtaskDict = dict()
            subtaskDict['no'] = no
            subtaskDict['project_name'] = col.replace('_metadata', '')
            subtaskDict['subtask_name'] = subtask['subtask_name']
            subtaskDict['update_time'] = subtask['updatetime']

            try:
                dataBaseProcessor.project_name = subtaskDict['project_name']
                dataBaseProcessor.subtask_name = subtaskDict['subtask_name']
                dataBaseProcessor.metaDataTableName = dataBaseProcessor.project_name + '_metadata'        
                metadata = dataBaseProcessor.queryMetaData({'remark':1}) 
                metadata = metadata[0]['remark']
            except:
                metadata = 'Nothing'
            subtaskDict['remark'] = metadata
            subtaskDict['remarkLink'] = '/Console/remarksubtask/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name']) 
            # 由于evaluate加入，更改
            #subtaskDict['tagging_link'] = '/Console/tagging/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name'])
            subtaskDict['tagging_link'] = '/Console/evaluate/%s/%s' % (subtaskDict['project_name'], subtaskDict['subtask_name'])
            subtaskDictList.append(subtaskDict)
            no +=1
    context = {'subtaskDictList':subtaskDictList, 'staticPath':STATIC_PATH}
    return render(request, 'Console/tags.html', context)

def tagging(request, projectName='MW_lle', subtask_name='spm_lle150_optics3', widget_id=''):
    return album(request, projectName, subtask_name, 'tagging')

    # 首先要求登录
def evaluateHome(request):
    username_session = request.session.get('username',None)
    if username_session == None: 
        needLogin = True
    else:
        needLogin = False
    context = { 'staticPath':STATIC_PATH, 'needLogin': needLogin, 'username':username_session}
    return render(request,'Console/evaluateHome.html',context)

# 异常,可抛弃
def evaluate(request,projectName='MW_pca',subtask_name='spm_pca300_optics3'):
    # 获取所有outlier的控件
    dataBaseProcessor = loadDataBase(CONFIG_PATHS)
    dataBaseProcessor.projectName = projectName
    dataBaseProcessor.subtask_name = subtask_name
    dataBaseProcessor.configTableName = dataBaseProcessor.projectName+'_config'
    dataBaseProcessor.metaDataTableName = dataBaseProcessor.projectName + '_metadata'        
    dataBaseProcessor.apkTreeTableName =dataBaseProcessor.projectName+'__'+dataBaseProcessor.subtask_name+'_apktree'         
    #print('what??', dataBaseProcessor.configTableName , dataBaseProcessor.metaDataTableName, dataBaseProcessor.apkTreeTableName)
    config = dataBaseProcessor.getConfig()
    #metadata = dataBaseProcessor.queryMetaData({'projectName':1, 'subtask_name':1, 'updatetime':1, "ifdim":1, "ifmethod":1, "imagesize":1, "rawdims":1, "cutdimway":1, "reducedims":1, "calinski" :1, "clustermethod":1, "clusters":1, "silhouette":1}) 
    metadata = dataBaseProcessor.queryMetaData({'project_name':1, 'subtask_name':1, 'updatetime':1, "ifdim":1, "ifmethod":1, "imagesize":1, "rawdims":1, "cutdimway":1, "reducedims":1, "calinski" :1, "clustermethod":1, "clusters":1, "silhouette":1,"rawapkforestname":1}) 
    dataBaseProcessor.rawApkForestName = metadata[0]['rawapkforestname']
    rawapkTree = dataBaseProcessor.queryRawAPKTree({'image_cluster_no':{"$ne":'-2'}}, {"path" : 1, "app" : 1, "widget" : 1})
    apkTree = dataBaseProcessor.queryAPKTree({}, {"raw_id" : 1, "cluster_no" :1, "outlier_score" :1})
    print('tag evaluate(评测) starting...')
    metadata[0]['projectName'] = projectName

    apkInfoTree = []
    if apkTree == []:
        return HttpResponse('该项目下没有已运行完成的项目.')
    if 'outlier_score' not in apkTree[0]:
        return HttpResponse('该项目还未进行异常检测.')

    #pageSize = 12
    #curPageSize = 12
    pageSize = 1
    curPageSize = 1
    page = 0
    maxPage = 0
    for i in range(len(rawapkTree)):
        if rawapkTree[i]['_id'] != apkTree[i]['raw_id']:
            print('Wrong!!')
        apkTree[i]['isOutlier'] = ''
        apkTree[i]['outlier_score'] = apkTree[i]['outlier_score'][:5]
        #if float(apkTree[i]['outlier_score']) <= -0.55:
        if float(apkTree[i]['outlier_score']) <= 1:
        #if float(apkTree[i]['outlier_score']) > :
            apkTree[i]['isOutlier'] = 'outlier'
            # 现在只有是outlier的才被加入
            rawapkTree[i]['widget'] = ''.join(rawapkTree[i]['widget'].split(', '))
            apkTree[i]['id'] = str(apkTree[i]['raw_id'])
            curPageSize -= 1
            if curPageSize == 0:
                curPageSize = 12
                page += 1
                maxPage = page
            apkTree[i]['page'] = str(page)
            apkInfoTree.append((rawapkTree[i], apkTree[i]))

    context = {'BaseDir':config['INPUT_DATA_DIR'], 'metadata':metadata[0], 'outlierapkInfoTree':apkInfoTree, 'apkTree':apkTree, 'max_cluster':metadata[0]['clusters'], 'cluster_num':range(int(metadata[0]['clusters'])), 'staticPath':STATIC_PATH,'picturePath':PICTURE_PATH ,'language':'zh','max_page':maxPage}
    #print(context)

    return render(request, 'Console/evaluate.html', context)

def evaluateSubmit(request):
    #username = request.POST.get('username')
    score = request.POST.get('score')
    #writeEvaluation({'username':username,'score':score})
    context = {'staticPath':STATIC_PATH}
    return render(request, 'Console/evaluateDone.html', context)


def getWebCachefromVPS(website,platform=''):
    print('getWebCachefromVPS>>>')
    global RAW_ROOT
    if 'f-droid' in website:
        cache_html_name = website.replace('https://f-droid.org/en/packages/','')+'-'+platform+'.html'
    else:
        cache_html_name = website.split('=')[-1]+'-'+platform+'.html'
    cache_path = RAW_ROOT+'cachehtmls'
    if not os.path.exists(cache_path+'/'+cache_html_name):
        print('使用远程服务器下载')
        HelpMeDownload.downloadExecuter(website,cache_html_name,cache_path,'wget')
    else:
        print('使用缓存文件.')
    print('<<<getWebCachefromVPS')
    print('getWebCachefromVPS:', cache_path+'/'+cache_html_name)
    return cache_path+'/'+cache_html_name

def getInfofromGooglePlay(website):
    # logo: class="xSyT2c"
    # appname: h1 class="AHFaub"
    # contact: span "hrTbp euBY6b"
    from bs4 import BeautifulSoup

    cache_html_path = getWebCachefromVPS(website,'googleplay')
    if not os.path.exists(cache_html_path):
        print('visist error! Return None')
        return None,None,None,None

    #html = requests.get(website,headers=headers,proxies=proxies)
    #soup = BeautifulSoup(html.content,'lxml')
    soup = BeautifulSoup(open(cache_html_path),'html.parser')

    try:
        logo = soup.find('img', {'class', 'T75of sHb2Xb'}).get('src')
        appname = soup.find('h1', {'class', 'AHFaub'}).span.string
        contact = soup.find('a', {'class', 'hrTbp euBY6b'}).string
    except:
        print('this page might not exist.')
        return None,None,None,None

    return logo,appname,contact,''

    
def getInfofromFdroid(website):
    # logo: "article-area" > class "package-icon"
    # appname: "article-area" > class "package-name"
    # contact: "package-links" > Issue Tracker
    from bs4 import BeautifulSoup
    '''
    from fake_useragent import UserAgent
    import requests
    ua=UserAgent()
    headers={"User-Agent":ua.random}

    html = requests.get(website,headers=headers)
    if html.status_code != 200:
        print('status_code:',html.status_code,' error!')
        return None,None,None,None
    '''
    cache_html_path = getWebCachefromVPS(website,'fdroid')
    if not os.path.exists(cache_html_path):
        print('visist error! Return None')
        return None,None,None,None
    soup = BeautifulSoup(open(cache_html_path),'html.parser')

    try:
        main = soup.find('div', {'class', 'article-area'})
        logo = main.article.header.img.get('src')
        appname = main.article.header.div.h3.string
        # 去除空白字符
        appname = appname.strip()

        contacts = soup.select('.package-links > li >  a')
        contact = ''
        for item in contacts:
            if item.string == 'Issue Tracker':
                contact = item.get('href')
                break
        download = soup.find('p',{'class','package-version-download'}).b.a.get('href')
    except:
        print('this page might not exist.')
        return None,None,None,None

    return logo,appname,contact,download

def getInfofromApkfab(website):
    # https://apkfab.com/free-apk-download?q=
    # logo: "article-area" > class "package-icon"
    # appname: "article-area" > class "package-name"
    # contact: "package-links" > Issue Tracker
    from bs4 import BeautifulSoup

    # 首先进行网址转化:
    website = 'https://apkfab.com/free-apk-download?q='+website.split('=')[-1]
    cache_html_path = getWebCachefromVPS(website,'apkfab')
    if not os.path.exists(cache_html_path):
        print('visist error! Return None')
        return None,None,None,None

    #html = requests.get(website,headers=headers,proxies=proxies)
    #soup = BeautifulSoup(html.content,'lxml')
    soup = BeautifulSoup(open(cache_html_path),'html.parser')

    
    logos = soup.select('.packageInfo > a > img')
    if len(logos) == 0:
        logos = soup.select('.packageInfo > div > img')
    logo = logos[0].get('src')
    # = main.find('a[class="title"]')
    appnames = soup.select('.packageInfo > div > a')
    appnames.extend(soup.select('.packageInfo > div > div'))
    appname = download = ''
    for item in appnames:
        #print('=>',item)
        if 'title' in item.get('class'):
            appname = item.string
            if 'href' in item.attrs:
                download = item.get('href')
            break
            '''
    except:
        print('this page might not exist.')
        return None,None,None,None
        '''
    if download == '':
        download = website
    return logo,appname,'',download

def submitSuspects(request):
    suspects = request.GET.get('suspects')
    submit = request.GET.get('submit')
    suspects = suspects.rstrip(';').split(';')
    print('---->submit suspects:',suspects)

    if submit != 0:
        # 连接数据库，提交标为suspects的状态
        dataBaseProcessor = loadDataBase(CONFIG_PATHS)
        print('prepare to submit suspects:')

        #对idUpath更改从而更新数据库记录是否suspect
        idUpath = dataBaseProcessor.queryAllPath() # 这里就是从原apktree获取的 
        for i in range(len(idUpath)):
            #idUpath[i][1] = {"$set":{"cluster":str(clusterLabels[i])}}
            # _id字段保留，对1字段进行修改改为image_cluster_no
            if str(idUpath[i][0]) in suspects:
                idUpath[i][1] = 'suspect btn-warning'
                print(str(idUpath[i][0]),idUpath[i][1])
            else:
                idUpath[i][1] = 'notsuspect btn-default'
        dataBaseProcessor.updateRawAPKForest_Suspect(idUpath)

    # 数据库完成，，，接下来获取更多信息
    suspects_app = request.GET.get('suspects_app')
    suspects_app = suspects_app.rstrip(';').split(';')
    print('---->suspects_app:',suspects_app)
    return_website = []
    for app in suspects_app:
        website = fetchAppWebsiteInfo(DBNAME,app)
        if 'f-droid' in website:
            return_website.append( getInfofromFdroid(website) )
        else:
            google_logo = google_appname = google_contact = google_download = ''
            result2 = getInfofromApkfab(website) 
            #print('#>result2',result2)
            if result2 != (None,None,None,None):
                google_logo = result2[0]
                google_appname = result2[1]
                google_download = result2[3]
            result1 = getInfofromGooglePlay(website)
            #print('#result1>',result1)
            if result1 != (None,None,None,None):
                if google_logo != '':google_logo = result1[0]
                if google_appname != '':google_appname = result1[1]
                google_contact = result1[2]
            result = (google_logo , google_appname , google_contact , google_download  )
            return_website.append(result)
    print('======>return_website:',return_website)
    return JsonResponse({'msg':'success!','data':return_website})











    
def videos(request, database=DBNAME):
    # 读取csv文件
    video_info_path = '/data/kongcancan/Classification_Video_Dataset/Videos/info.csv'
    f = open(video_info_path,'r') # 专门有用于构建的项目？
    lines = csv.reader(f)


    #video = collect_rawapkforest.find({},{'path':1,"app" : 1, "widget" : 1,'image_cluster_no':1}).sort("_id",1)
    videoList = []
    max_cluster = 0
    for row in lines:
        videotemp = {'path':row[0],'app':row[0],"widget":'---','image_cluster_no':row[1]}
        # path为id，用path引导软删除
        videoList.append(videotemp)
        if int(videotemp['image_cluster_no']) > max_cluster:
            max_cluster = int(videotemp['image_cluster_no'])
    f.close()
    
    # 异常退出，没有进行聚类
    if len(videoList) == 0:
        return HttpRespone('尚未进行聚类')

    # 在线标注
    BaseDir= '/'.join(video_info_path.split('/')[:-1])+'/'
    context = {'picturePath':PICTURE_PATH,'BaseDir':BaseDir,'rawapkTree':videoList,'staticPath':STATIC_PATH,'current_db':'videoDatabase','current_apkforest':'PRECLUSTER','new_max_cluster':max_cluster,'tagType':'rawdatabase'}
    return render(request,'Console/videos.html',context)

def hashtags(request):
    # 读取csv文件
    hashtag_info_path = '/data/kongcancan/Classification_Video_Dataset/HashTag/hasgtag_clustering_result.csv'
    f = open(hashtag_info_path,'r') # 专门有用于构建的项目？
    lines = csv.reader(f)


    #hashtag = collect_rawapkforest.find({},{'path':1,"app" : 1, "widget" : 1,'image_cluster_no':1}).sort("_id",1)
    hashtagList = []
    max_cluster = 0
    for row in lines:
        hashtagtemp = {'content':row[0],'path':'','app':'',"widget":'','image_cluster_no':row[1]}
        # path为id，用path引导软删除
        hashtagList.append(hashtagtemp)
        if int(hashtagtemp['image_cluster_no']) > max_cluster:
            max_cluster = int(hashtagtemp['image_cluster_no'])
    f.close()
    
    # 异常退出，没有进行聚类
    if len(hashtagList) == 0:
        return HttpRespone('尚未进行聚类')

    # 在线标注
    context = {'picturePath':PICTURE_PATH,'rawapkTree':hashtagList,'staticPath':STATIC_PATH,'current_db':'videoDatabase','current_apkforest':'PRECLUSTER','new_max_cluster':max_cluster,'tagType':'rawdatabase'}
    return render(request,'Console/hashtags.html',context)

def bugShow(request):
    context = {'staticPath':STATIC_PATH}
    return render(request,'Console/bugshow2.html',context)
