import sys, os
import csv
sys.path.append(os.path.abspath('..')) 
from importlib import reload
reload(sys)
from Base import BaseProcessor
import re
from string import digits
import hashlib

DEBUG = False
WHITE_API_LIST = []
RELATIVEPATH = 'views_info'
#RELATIVEPATH = ''
APIFILTERTYPE_LEVEL = 'simply'
#APIFILTERTYPE_DOC = 'no' # 专门用于Doc的
APIFILTERTYPE_DOC = 'simply' # 专门用于Doc的
API_NUM_LIMIT = 999999

APIIDF_FILTER_TYPE = 'smaller'

API_IDF_CSV_DICT = None
API_IDF_CSV_PATH = '/data/wangruifeng/datasets/DroidBot_Epoch/raw_data/dimension_data/Sorted_APIIDF.csv'
EXCEPTAPILIST = ["android.view.View.getId",
"android.support.v7.app.AppCompatActivity.findViewById",
"android.app.Activity.findViewById",
"android.graphics.Canvas.rotate ",
"android.graphics.Canvas.translate",
"android.app.Activity.setTitle",
"android.widget.TextView.setText",
"android.view.View.getTag",
"android.support.v4.widget.DrawerLayout.isDrawerOpen",
"android.content.Context.getString",
"android.util.Log.d",
"android.util.Log.e",
"android.util.Log.i",
"android.util.Log.v",
"android.util.Log.w",
"android.view.View.getContext",
"android.view.View.setOnClickListener",
"org.json.JSON.toInteger",
"android.content.Intent.setData",
"android.util.SparseArray.size",
"android.content.Intent.setClass",
"android.content.ContextWrapper.getPackageName",
"android.support.v7.util.SortedList.size"]
'''
"android.support.v7.view.menu.MenuItemImpl.getItemId",
"android.database",
"android.view.RenderNodeAnimator.setDuration",
"android.graphics",
"androidx.swiperefreshlayout.widget.SwipeRefreshLayout.isRefreshing",
]
'''

class Transformer(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(Transformer, self).__init__()
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
    Describe:
        功能: 读取json文件，审查其各方面要求，可以的写入数据库，不行的则将其图片移动到相应文件夹
        check widgets and write into db
 
        输入：有意义记录集合
        feed:
 
        输出:
        output:

        更新：目前使用开发集DEV_SET做集合
 
    Methods list:
        if operator == 'info' or operator == '':
        if operator == 'run':
            if ' apiidf smaller' in params:
                APIIDF_FILTER_TYPE = 'smaller'
                params = params.replace(' apiidf smaller','')
            elif ' apiidf center' in params:
                APIIDF_FILTER_TYPE = 'center'
                params = params.replace(' apiidf center','')

    Example:
        DT run lim10
        DT run apiidf-smaller
        '''
        print(info)


    # ----------------------------Transform Operator---------------------------
    def readJsons(self):
        jsons = dict()
        # 首先获取路径
        input_data_path = self.config['INPUT_DATA_DIR']
        for name in os.listdir(input_data_path):
            print('name:',name)
            try:
                # 获取到文件夹之后读取json文件，然后json文件里面有图片的地址就好说了
                json_temp = self.readDict(os.path.join(input_data_path,name,RELATIVEPATH, 'views_info.json'), "noordered")
            except:
                print('Read',os.path.join(input_data_path,name,RELATIVEPATH, 'views_info.json'),' Error! skipping it.')
                continue
            jsons[name] = json_temp
        print(len(jsons),' jsons')
        return jsons

    def isAtPackageList(self,package,package_list):
        '''
        import re
        # 这么做的目的是，不让android直接就匹配，要不然误进很多api
        # 正则表达式，从开头开始的小写字母或者\.，遇到大写就结束了，这样可以匹配出包名
        package_regex = re.compile('^[a-z\.]+')
        package_regex_result = package_regex.match(api)
        if package_regex_result == None:
            print('无法获取包名！')
            return False # 无法获取包名！
        else:
            package = package_regex_result.group(0).rstrip('.')
    
        print('searching >>%s<< in package_list...' % package)
        '''
        if package in package_list:
            return True
        return False

    def APIFilter(self,package,filterType='white'):                                                                               
        # 选择白名单
        if filterType == 'white':
            apiList = WHITE_API_LIST#self.readTxt( self.config['WHITE_API_LIST_PATH'] )                                                      
            return self.isAtPackageList(package,apiList)
        # 选择黑名单
        elif filterType == 'black': 
            apiList = self.readTxt( self.config['BLACK_API_LIST_PATH'] )                                                      
            if api not in apiList:                                                                                            
                return True
        elif filterType == 'simply':
            if package[:8] == 'android.' or package[:9] == 'androidx.' or package[:11] == 'com.android.': 
                return True                                                                                                   
            else: 
                apiList = WHITE_API_LIST#self.readTxt( self.config['WHITE_API_LIST_PATH'] ) 
                judge1 = self.isAtPackageList(package,apiList)
                #judge2 = not self.isAtPackageList(package,EXCEPTAPILIST)
                if judge1: #and judge2:
                    return True
        elif filterType == 'no':
            return True
        # 选择 
        return False

    def APIIDFFilter(self,api,filterType='center'):
        global API_IDF_CSV_DICT
        if API_IDF_CSV_DICT == None:
            api_idf_dict = dict()
            with open(API_IDF_CSV_PATH,'r') as f:
                f_csv = csv.reader(f)
                for row in f_csv:
                    if row[1] == 'idf':
                        continue
                    # 前者api，，后者idf
                    api_idf_dict[row[0]] = float(row[1])
                API_IDF_CSV_DICT = api_idf_dict
                print(API_IDF_CSV_DICT.keys())
        
        if filterType == 'smaller':
            # 开始判别 < 12的可行
            if API_IDF_CSV_DICT[api] < 12.14:
                return True
            else:
                return False
        elif filterType == 'center':
            # 开始判别 < 12的可行
            if 0.115326 < API_IDF_CSV_DICT[api] < 12.14:
                return True
            else:
                return False
        elif filterType == 'verycenter':
            if 0.291004 < API_IDF_CSV_DICT[api] < 12.14:
                return True
            else:
                return False
        elif filterType == 'except':
            if api in EXCEPTAPILIST:
                return False
            else:
                return True




    # 一个控件的所有apis
    def checkAPI(self, widget_apis,ApiFilterType,simply=False):
        import numpy as np
        # 考虑入口的问题，简单的，直接忽略，将所有APIs都统计进来
        # 先过滤掉一些常用的
        # 可以存一个list单独统计api_list的出现次数
        # 用于统计筛选前的数量`
        api_set = set()
        # =============计算TF===================
        package_api = set()
        class_api = set()
        method_api = set()
        #api_dict = dict()
        # 改为处理所有的线程
        for key in widget_apis.keys():
            for api_json in widget_apis[key]:
                for api in api_json['APIs']:
                    api_set.add(api)
                    # 首先解析api，分离出package,class,method
                    package,class_name,method = self.resolveAPI(api)
                    if package == "" or class_name == "" or method == "":
                        # 如果为空，这个api不处理即可
                        continue

                    method_for_judge = package+'.'+class_name+'.'+method
                    #if method_for_judge in EXCEPTAPILIST:
                    #    print('#######',method_for_judge)
                    #    continue
                    # 首先筛选白名单
                    #if method_for_judge not in EXCEPTAPILIST and self.APIFilter(package,ApiFilterType):
                    if self.APIFilter(package,ApiFilterType):
                        #之前是往集合中加入，现在需要统计数量了
                        #api_set.add(api_json['APIs'])
                        if simply == False:
                           package_api.add(package)
                           class_api.add(package+'.'+class_name)

                        # 只有经过idffilter才可加入
                        if self.APIIDFFilter(method_for_judge,APIIDF_FILTER_TYPE) and self.APIIDFFilter(method_for_judge,'except'):
                            method_api.add(method_for_judge)
                        '''
                        key = package+'.'+class_name+'.'+method
                        if key not in api_dict:
                            api_dict[key] = 1
                        else:
                            api_dict[key] += 1
                        '''
                        ''' 如果要恢复完全API，启用下面这段
                        if api not in api_dict:
                            api_dict[api] = 1
                        else:
                            api_dict[api] += 1
                        '''
            if simply == False:
            # 采用了白名单的方式肯定会导致有些控件没有api，不符合标准，则直接返回两个空值
            #if len(api_dict) == 0:
                if len(method_api) == 0:
                    #return None,None
                    return None,None,None,0
            # https://docs.python.org/zh-cn/3.6/library/functions.html#zip
            #api_list, api_freq = zip(*sorted(api_dict.items(), key=lambda t: t[0]))
            #api_list, _ = zip(*sorted(api_dict.items(), key=lambda t: t[0]))
            #api_freq = list(api_freq / np.sum(api_freq))
            #api_list = list(api_set)
            #api_list.sort()
                package_api = list(package_api)
                class_api = list(class_api)
                method_api = list(method_api)
                package_api.sort()
                class_api.sort()
                method_api.sort()
                #print('return :',package_api,class_api,method_api,len(api_set))
                return package_api,class_api,method_api,len(api_set)
            if len(method_api) == 0:
                return None
            method_api = list(method_api)
            method_api.sort()
            #print('simply return :',method_api.sort())
            return method_api

    def splitCamelCase(self,camel_case_name):
        doc = ""
        # 需要查证！！！！
        # 驼峰法命名分割，寻找大写字母，按大写字母分开
        split_camel_case_regex = re.compile('[A-Z]')
        split_camel_case_regex_result = split_camel_case_regex.finditer(camel_case_name)
        it = iter(split_camel_case_regex_result)
        try:
            current = next(it)
        except StopIteration:
            return doc

        while True:
            before = current
            try:
                # 如果能获取下一个，就获取，否则将后面的全部放进doc，之后就break
                current = next(it)
                doc += (" " + camel_case_name[int(before.span()[0]) : int(current.span()[0])])
            except StopIteration:
                doc += (" " + camel_case_name[int(before.span()[0]) :])
                break
        return doc
 

    def makeDoc(self,apis,api2doc_dict):
        # 3.16 Change for unique Doc
        api_docs = set()
        for api in apis:
            # 使用api2doc_dict存储已经解析过的，节省时间
            if api in api2doc_dict:
                api_docs.add( api2doc_dict[api] )
            else:
                # 首先需要将API分三块，调用之间写过的函数
                package,class_name,method = self.resolveAPI(api)
                # 接下来再去掉停用词，分割class和method中的大小写组成句子
                doc = package.replace('androidx.','').replace('android.','').replace('java.','').replace('org.','').replace('.',' ').replace('support v','support')
                remove_digits = str.maketrans('', '', digits)
                doc = doc.translate(remove_digits)
                class_name = class_name.replace('SQLite','Sqlite')
                class_name = class_name.replace('JSON','Json')
                class_name = class_name.replace('$','')
                method = method.replace('<','').replace('>','')
                # 进行类名拆分
                doc += self.splitCamelCase(class_name)
                # 进行方法名拆分          
                doc += self.splitCamelCase(method[0:1].upper() + method[1:])
                # doc 去重复
                docs = set(doc.split(' '))
                doc = ' '.join(docs)
                api_docs.add( doc.lower() )
                api2doc_dict[api] = doc.lower()
        api_docs = list(api_docs)
        return tuple(api_docs),api2doc_dict

    def TF_IDF(self,apk_widget_forest):
        import math
        # apk forest中每个控件中都有对应的API列表和频率
        # 遍历每个API列表，加入到字典中，统计每个API在所有文档中出现的频率

        # ================计算IDF=============
        # 总文档数|D|
        widget_num = len(apk_widget_forest)
        # 包含每个词的文档总数,最后将存储逆文档频率
        all_api_dict = dict()

        # 对于每一个控件，如果该控件所包含的某个api在all_api_dict中有，则为其加1，表示该api的总文档+1
        # 如果没有，则令其=1，目前有1个文档有次api
        for widget in apk_widget_forest:
            # 每个api考察一遍，因为每个都不一样
            for api in widget['api']:
                if api in all_api_dict:
                    all_api_dict[api] += 1
                else:
                    all_api_dict[api] = 1
        for key in all_api_dict.keys():
            # widget_num => |D| 
            # 1 + all_api_dict[i] => 1+|j: t_i in d_j|  包含词t_i的文档数量
            # 这里的log，底数为e
            all_api_dict[key] = math.log(( 1 + widget_num )/(1 + all_api_dict[key])) + 1
        
        # ===============计算TF * IDF =============
        '''
        for widget in apk_widget_forest:
            for i in range(len(widget['api'])):
                api = widget['api'][i]
                print("widget['api_freq'][i] * all_api_dict[api]:\n%f*\t%f" % (widget['api_freq'][i], all_api_dict[api]))
                widget['api_freq'][i] = widget['api_freq'][i] * all_api_dict[api]
                print("\n=%f" % (widget['api_freq'][i]))
        '''
        return apk_widget_forest


    # 只是检查好不好打开而已
    def checkPicutre(self, image_path, image_size, image_text):
        abs_image_path = self.config['INPUT_DATA_DIR'] + image_path
        #abs_image_path =  image_path
        print('$=========>processing ',abs_image_path)

        # 一个是如果有image_size本身有问题的，直接不要，如: -216*-89
        if '-' in image_size:
            return False

        # 尝试图片是否可以打开
        try:
            from PIL import Image
            _ = Image.open(abs_image_path, 'r')
            return True
        except:
            print('Read', abs_image_path ,' Error! Dropping it.')
            return False

    def writeIntoDB(self,apk_widget_forest):
        #self.DBP.save...
        #self.DBP.saveRawAPKForest(apk_widget_forest)                                
        for apk_widget in apk_widget_forest:
            # 保存每一条json数据就可以
            self.DBP.saveRawAPKTree(apk_widget)
        self.DBP.updateMetaData({"$set":{"widgets":len(apk_widget_forest)}}) 
        print(apk_widget_forest,'\n',len(apk_widget_forest))

    def calcMD5(self,file_name):
        with open(file_name, 'rb') as fp:
            data = fp.read()
        file_md5= hashlib.md5(data).hexdigest()
        return file_md5

    def runTransform(self):
        global API_NUM_LIMIT
        global APIFILTERTYPE_LEVEL
        global APIFILTERTYPE_DOC
        api_forest = list()
        jsons = self.readJsons()
        hash_set = set()
        api2doc_dict = dict()
        for app_name, widget_jsons in jsons.items():
            # 遍历每一个widget
            for widget in widget_jsons:
                api_tree = dict()
                api_tree['app'] = app_name
                api_tree['widget'] = widget['view_tag']
                api_tree['package'] = widget['info']['package']
                api_tree['path'] = os.path.join(app_name,RELATIVEPATH,'images', widget['image_filename'])
                #api_tree['suspect'] = 'suspect btn-default'
                # 注意这个是原始的图像尺寸，最后都是224*224
                api_tree['raw_image_size'] = widget['info']['size']
                api_tree['text'] = widget['info']['text']
                if not self.checkPicutre(api_tree['path'], api_tree['raw_image_size'], api_tree['text']):
                    continue
                # entrance对这个有关系嘛？content_description? bounds?
                api_tree['package_api'],api_tree['class_api'],api_tree['method_api'],api_tree['raw_api_len'] = self.checkAPI(widget['APIs'],APIFILTERTYPE_LEVEL) # Filter啥意思?
                # 如果api没有符合要求的，也得删除
                if api_tree['method_api'] == None:
                    print('api 均不在白名单内')
                    continue
                if len(api_tree['method_api']) > API_NUM_LIMIT:
                    print('api 数量过多！')
                    continue
                api_tree['filtered_api_len'] = len(api_tree['method_api'])
                # 直接用检查过的api就行了,是一个元祖，取出来就行 X
                # 如果doc使用全部api的话，那就只能重新分析了
                api_tree['doc'],api2doc_dict = self.makeDoc(self.checkAPI(widget['APIs'],APIFILTERTYPE_DOC,True),api2doc_dict) # Filter啥意思?

                seed = (self.calcMD5(self.config['INPUT_DATA_DIR'] + api_tree['path'])+'\n'.join(api_tree['method_api'])).encode("utf8")
                path_api_hash = hashlib.md5(seed)
                if path_api_hash.hexdigest() not in hash_set:
                    api_forest.append(api_tree)
                    hash_set.add(path_api_hash.hexdigest())
                else:
                    print('重复的控件')
        # 计算TF-IDF
        #api_forest = self.TF_IDF(api_forest)
        if DEBUG == False:
            self.writeIntoDB(api_forest) # 或者直接用self.DBP函数也可以
        else:
            print(api_forest,len(api_forest))
        
    def runTransformer(self, operator='', params=[], config=[]):
        run_log = [] 
        global WHITE_API_LIST
        # 事先一次性读取省事
        WHITE_API_LIST = self.readTxt( self.config['WHITE_API_LIST_PATH'] ) 
        self.checkDBP()
        if operator == 'info' or operator == '':
            self.info()

        if operator == 'run':
            if DEBUG == False:
            #?# 使用前注意检查有没有！！！
                rawdata = self.DBP.queryRawAPKTree()
                if rawdata:
                    print('/\/\ rawdata exists, skipping.')
                    return run_log
            if 'apiidf-smaller' in params:
                APIIDF_FILTER_TYPE = 'smaller'
                params = params.replace('apiidf-smaller','')
            elif 'apiidf-center' in params:
                APIIDF_FILTER_TYPE = 'center'
                params = params.replace('apiidf-center','')
            elif 'apiidf-verycenter' in params:
                APIIDF_FILTER_TYPE = 'verycenter'
                params = params.replace('apiidf-verycenter','')

            api_num_limit = params.replace('lim','')
            global API_NUM_LIMIT
            if api_num_limit == '':
                API_NUM_LIMIT = 9999
            else:
                API_NUM_LIMIT = int(api_num_limit)
            print(API_NUM_LIMIT)
            self.runTransform()
