from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from bson import ObjectId
import json
import csv
import os

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

# #>>DataSets---------------------------------------------------------------------------------------------------------------
class ClassificationDataSet:
    def __init__(self):
        pass
        
    def writeTxt(self,content,path,mode='w'):
        #path += '.txt'
        file_handle=open(path,mode=mode)
        file_handle.write("\n".join(content))
        file_handle.close()
        print(path,',',len(content),' records writen.')
        
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

    def writeCSV(self,content,path):
        #path += '.csv'
        import csv
        with open(path,'w',newline='') as t_file:
            csv_writer = csv.writer(t_file)
            for l in content:
                csv_writer.writerow(l)
        print(path,',',len(content),' records writen.')        
        
    def loadInterpreter(self,configPaths, projectName="", subtask_name=""):
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
    def loadDataBase(self,configPaths):
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

    def showDataSetHome(self,request):
        '''说明:
            返回选择页面，整合数据库的标注，数据库的选择
        '''
        from pymongo import MongoClient
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
            dataBaseListItem['chooseLink'] = '/Console/createdataset/%s' % db
            dataBaseList.append(dataBaseListItem)

        print(db_list)

        # apkforestlist
        global DBNAME
        dataBaseProcessor = self.loadDataBase(CONFIG_PATHS)
        client = dataBaseProcessor.getConnection()
        collist = client.collection_names()
        apkForestList = []
        for line in collist:
            if '.' not in line and '_' not in line:
                # 作为列表之一
                apkForestListItem = dict()
                apkForestListItem['name'] = line
                apkForestListItem['tagLink'] = '/Console/apkforest/%s'  % line
                apkForestListItem['chooseLink'] = '/Console/createdataset/%s' % line

                apkForestList.append(apkForestListItem)

        context = {'apkForestList':apkForestList,'staticPath':STATIC_PATH,'dataBaseList':dataBaseList,'current_db':DBNAME,'current_apkforest':RAWAPKFOREST}
        return render(request, 'Console/DataSetHome.html', context)
        
    def createDataSetByChoicen(self,request,database):
        # 读取csv文件
        global RAW_ROOT
        save_path =  RAW_ROOT + database + '/derived_dataset/'
        info_csv_path = save_path + 'seed_icons_dataset_info.csv'
        label_type_map_path = save_path + 'seed_icons_dataset_label_type_map.json'
        if not os.path.exists(save_path+'seed_icons_dataset_info.csv'):
            return HttpResponse('尚未进行聚类，无法人工审核该数据集.使用IC search搜索参数,IC run运行聚类')

        f = open(info_csv_path,'r') # 专门有用于构建的项目？
        label_type = self.readDict(label_type_map_path,"noordered") # 专门有用于构建的项目？
        label_type_map = dict()
        for key,value in label_type.items():
            label_type_map[int(key)] = value
        label_type_map[-1] = 'No Type'
        lines = csv.reader(f)
        print('csv 是否存在?',lines)


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
            rawapkTreetemp = {'path':row[1],'app':row[0],"widget":'---','image_cluster_no':row[2],'type':label_type_map[int(row[2])]}
            # path为id，用path引导软删除
            rawapkTreeList.append(rawapkTreetemp)
            if int(rawapkTreetemp['image_cluster_no']) > max_cluster:
                max_cluster = int(rawapkTreetemp['image_cluster_no'])
        f.close()
        
        print('rawapkTreeList构建正常？',rawapkTreeList)
        # 异常退出，没有进行聚类
        if len(rawapkTreeList) == 0:
            return HttpResponse('尚未进行聚类，无法从该数据集构建新数据集.使用IC search搜索参数,IC run运行聚类')

        # 在线标注
        import html
        BaseDir= RAW_ROOT + '%s/input_data/' % database
        context = {'picturePath':PICTURE_PATH,'BaseDir':BaseDir,'rawapkTree':rawapkTreeList,'staticPath':STATIC_PATH,'current_db':database,'current_apkforest':'PRECLUSTER','new_max_cluster':max_cluster,'tagType':'rawdatabase','label_type_map':html.unescape(json.dumps(label_type_map,ensure_ascii=False)),'label_type':label_type_map}
        print('context::',context)
        return render(request,'Console/DataSetChoicen.html',context)
        
    #Choicen
    #DataSetChoicen.html
    def submitDataSetDelete(self,request):
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

    def submitDataSetChoicen(self,request,database=DBNAME):
        '''
        tagResult = request.GET.get('tagResult')
        # 对删除的文件直接执行更名命令
        '''
        global RAW_ROOT
        save_path =  RAW_ROOT + database + '/derived_dataset/'
        info_csv_path = save_path + 'seed_icons_dataset_info.csv'
        train_dataset_path = save_path + 'train_dataset.csv'

        selected_icons = request.POST.get('selected_icons')
        print(request.POST,'---------')

        selected_icons_csv_list = []
        selected_icons = selected_icons.split('|')
        for one_type_icons in selected_icons:
            # 将传过来的文件进行改名
            print('in selected_icons for',one_type_icons)
            if one_type_icons != "":
                for icon in one_type_icons.split(';'):
                    print('icon',icon)
                    selected_icons_csv_list.append(icon)

        self.writeTxt(selected_icons_csv_list,train_dataset_path)
        res = {'msg':'success'}
        return JsonResponse(res,json_dumps_params={'ensure_ascii': False})


    # #>>DataSets---------------------------------------------------------------------------------------------------------------

    def tagRawDataBase(request, database=DBNAME):
        pass
        
    # Console
    def submitRawDataBaseTag(request,database=DBNAME):
        pass
