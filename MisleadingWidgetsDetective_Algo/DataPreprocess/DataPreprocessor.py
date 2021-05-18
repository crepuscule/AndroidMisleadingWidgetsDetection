import sys,os
import shutil
from PIL import Image
from operator import itemgetter
import time
import easyocr
import numpy as np
import requests
import base64
import difflib
import imagehash
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor 
from paddleocr import PaddleOCR, draw_ocr
import cv2
from cv2 import *
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import scipy.misc
import random
# 改进方向：
# 性能记录：12.16 googleplay_13k
# [OK]size: 大的图片其实感觉加入也没问题；300->400 小的图片应该删除(20像素?)； OK
# text：少有误删(只是有些字很小也被删了，有些可惜)；阿拉伯文字没有被删除(其实还是很好认的，不会引起更多误删)；
# [等待]unconcerned: 误删多=>各个数据集考虑各自的unconcerned即可
# color: 少有误删除()
# 复杂意义:照片很多，



# 数据预处理分为以下几个部分：
# 首先去除尺寸不合适的控件，即宽高任意一个大小>300去除，纵横比> 4 或者 < 0.25去除
# 接着去除纯色图片，他们在人眼看起来为纯色
# 接着去除某些人为指定的图片，这通过计算图片相似度来实现
# 最后去除包含文字的图片，将先尝试使用easyocr，如果判不出文字则交给百度ocr识别
# 最后的图片将再需要经过人工审查确定没有其他异常的，即可进行下一步Transform

# 另外，由于预处理可能经过几个步骤，所以应该有对于数据集重置的功能，通过os.system执行linux命令实现

DEBUG = True
TEMP_LOG = '/data/wangruifeng/datasets/DroidBot_Epoch/recogize.txt'

# 预处理相关配置
IMAGE_SUFFIX = '.png'
IMAGE_SUFFIX1 = '.jpg'
DROP_SUFFIX = '.abandon'
EXTEND_SIZE_THRESHOLD = 500 
SIZE_THRESHOLD = 224
SIZE_THRESHOLD_SMALL = 20
#WH_RATIO_THRESHOLD = (4,0.25)
WH_RATIO_THRESHOLD = (3.5,0.29)

COLOR_SIMILAR_THRESHOLD = 30

# 2稍有区别，0几乎差不多
#COMPARE_THRESHOLD = 2
COMPARE_THRESHOLD = 2
BG_COLOR = (255,255,255)

ROOT_DIR = '/data/wangruifeng/datasets/'

UNCONCERNED_IMAGE_DIR = ROOT_DIR + 'DroidBot_Epoch/raw_data/unconcerned_image/'
# 防止出现某些图片漏筛的问题,基本上筛除一遍之后就可以了
CONCERNED_IMAGE_LIST_PATH = None
CONCERNED_IMAGE_LIST = None
NOT_CONCERNED_IMAGE_LIST_PATH = None
NOT_CONCERNED_IMAGE_LIST = None
# OCR相关配置
OCR_API_URL = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic'
ACCESS_TOKEN_POOL = ['24.559ca4c7f10a50da300dd61a5e28213c.2592000.1617762744.282335-22846592',
'24.d2a747904704222d496cd506b7f74c58.2592000.1617762744.282335-22891462',
'24.90ace3dd6df85588560fede08a7d7366.2592000.1617762745.282335-22891477',
'24.3a0ce9bd7cc3b818a904283c58fe0a7d.2592000.1617762745.282335-22891506',
'24.708a2f229d758bc0a79269cb6d149d0f.2592000.1617762745.282335-22891512',
'24.92413d2f6c109d0c6d8ad12236d23ee7.2592000.1617762746.282335-22897930'] #3.8

QPS = 0
REST_TIME = 0
CHOOSE = 0
EASYOCR_READER = None
PADDLEOCR_READER = None

# recycleTextTrash Config
TEXT_CONFIDENCE_THRESHOLD = 0.7 # 0.8
# 图片高度占比阈值
TEXT_HEIGHT_RATIO_THRESHOLD = 0.4 #0.3
# 图片宽度占比阈值
#TEXT_WIDTH_RATIO_THRESHOLD = 0.3
#图片面积占比阈值
TEXT_AREA_RATIO_THRESHOLD = 0.3 #0.15
# 图片距中心偏移位置域
#def TEXT_CENTER_OFFSET(TOTAL_HEIGHT):
#    return ((TOTAL_HEIGHT/5)*3,(TOTAL_HEIGHT/5)*4)

# 其他配置
#INPUT_DATA_BACKUP_PATH = ROOT_DIR + 'DroidBot_Epoch/raw_data/input_data_backup/'

class DataPreprocessor(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(DataPreprocessor, self).__init__()
        sys.path.append(os.path.abspath('..'))
        self.config = config
        self.DBP = DBP      
        self.recogize_log = []
        
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
        # 数据预处理分为以下几个部分：
        # 首先去除尺寸不合适的控件，即宽高任意一个大小>300去除，纵横比> 4 或者 < 0.25去除
        # 接着去除纯色图片，他们在人眼看起来为纯色
        # 接着去除某些人为指定的图片，这通过计算图片相似度来实现
        # 最后去除包含文字的图片，将先尝试使用easyocr，如果判不出文字则交给百度ocr识别
        # 最后的图片将再需要经过人工审查确定没有其他异e的，即可进行下一步Transform

        # 另外，由于预处理可能经过几个步骤，所以应该有对于数据集重置的功能，通过os.system执行linux命令实现
     
    Methods list:
        if operator == 'recover':
            recover_dir = input('Recover which ?')
        if operator == 'testsets':
            if make_sure == 'yes':
        if operator == 'backup':
            print('begin backup')
        if operator == 'run':
            if 'resize' in params:
                use_resize = True
            if 'nouse' in params:
                use_unconcerned_image = True
        if operator == 'unistrategy':
            params = params.split(',')
            dataset_name = params[0] 
            strategy = params[1] # resize fillwithbg
            refence_dataset_name = params[2] # 
    Example:
        DP run resize,nouse
        DP unistrategy db_universal_set_evaluation,resize,db_universal_set_evaluation-origin

        =>最新recycletext流程:
        首先创建准备数据集
        DP unistrategy db_universal_set_evaluation,preparerecycletext,任意
        再创建真的数据集(自动将trash_data变为新的prepare)
        DP unistrategy db_universal_set_evaluation,recycletext,db_universal_set_evaluation-origin

        最后再merge(省略input_data_)
        DP merge db_universal_set_evaluation,extenedwithtext,origin
        DP merge db_universal_set_evaluation,extenedwithtext,recycletext
        '''
        print(info)

    #---------------------------- 0. basic file operator--------------------------------
    def copyPictureToTrash(self, sourcepath, destpath):
        import os, shutil
        if not os.path.exists(destpath):
            os.makedirs(destpath)
        shutil.copy(sourcepath, destpath)

    def movePictureToTrash(self, sourcepath, destpath):
        self.copyPictureToTrash(sourcepath, destpath)
        os.remove(sourcepath)

    #---------------------------- 1. check size--------------------------------
    # 检查尺寸和尺寸比例，通过返回True，不通过返回Flase
    def check_size(self, image , size_threshold,size_threshold_small):
        top_img_w, top_img_h = image.size
        if (top_img_w > size_threshold) or (top_img_h > size_threshold):
            return False
        if (top_img_w < size_threshold_small) or (top_img_h < size_threshold_small):
            return False
        return True

    def check_wh_ratio(self, image, wh_ratio_threshold):
        top_img_w, top_img_h = image.size
        wh_ratio =  top_img_w / top_img_h
        if wh_ratio > wh_ratio_threshold[0] or wh_ratio < wh_ratio_threshold[1]:
            return False
        return True

    # --------------------------- 2. check pure color image-------------------------
    # 根据实验Distance=33左右人眼能够分辨出来
    # 32.99 人眼可以分辨，两个颜色是不同的
    # 所以认为30一下，两个颜色是相同的，30以上两个颜色是不同的
    # 所以能分辨出的阈值应该设定在
    def calcColourDistance(self,rgb_1, rgb_2): 
        import math
        R_1,G_1,B_1 = rgb_1 
        R_2,G_2,B_2 = rgb_2 
        rmean = (R_1 +R_2 ) / 2 
        R = R_1 - R_2 
        G = G_1 -G_2 
        B = B_1 - B_2 
        return math.sqrt((2+rmean/256)*(R**2)+4*(G**2)+(2+(255-rmean)/256)*(B**2))
        
    def isBadScreenshot(self,image_path):
        # 如果有大量的白色块，占据全左，全右边，全上，全下方，超过20% 
        pass

    def check_color(self, image, color_similar_threshold):
        top_img_w, top_img_h = image.size
        # 获取颜色列表
        clrs = image.getcolors(top_img_w * top_img_h)
        # 如果clrs == 1 则肯定是纯色可以直接删除
        if len(clrs) == 1:
            return False
        # 否则要颜色计算相似度
        else:
            sortedclrs = sorted(clrs, key=itemgetter(0), reverse=True)
            #print('sortedclrs',sortedclrs)
            # 存储颜色占比
            rate_list = list()
            # 存储颜色
            clr_list = list()
            for clr_item in sortedclrs:
                # 拿出每个颜色进行对比，当rate_list的和超过80%时就停止，后面的不要了
                rate_list.append(clr_item[0]/(top_img_w * top_img_h))
                clr_list.append(clr_item[1][:3])
                # 为了防止里面只有一个颜色，就只允许rate_list长度大于1时才break
                if sum(rate_list) >= 0.99 and len(rate_list) > 1:
                    break
            #print('两个数组\n',rate_list,'\n',clr_list)
            # 以第一个为基准，计算后面几个与第一个相似度就行了
            # 还要判断rate_list长度
            # 如果0.8 0.1 0.1 这样子？=>非纯色
            color_diff_num = 0
            for i in range(1,len(clr_list)):
                if self.calcColourDistance(clr_list[0],clr_list[i]) > color_similar_threshold:
                    color_diff_num += 1
            # 只要有颜色不同的，那就可以不用删除,颜色都相同，那就删除
            if color_diff_num == 0:
                return False
        return True
    # --------------------------- 3. check unconcerned image using image similarity-------------------------
    def is_concerned_images(self,image_path):
        global CONCERNED_IMAGE_LIST
        if CONCERNED_IMAGE_LIST == None:
            global CONCERNED_IMAGE_LIST_PATH
            CONCERNED_IMAGE_LIST = self.readTxt(CONCERNED_IMAGE_LIST_PATH)
        if image_path.split('/')[-1] in CONCERNED_IMAGE_LIST:
            return True
        return False
    def is_not_concerned_images(self,image_path):
        global NOT_CONCERNED_IMAGE_LIST
        if NOT_CONCERNED_IMAGE_LIST == None:
            global NOT_CONCERNED_IMAGE_LIST_PATH
            NOT_CONCERNED_IMAGE_LIST = self.readTxt(NOT_CONCERNED_IMAGE_LIST_PATH)
        if image_path.split('/')[-1] in NOT_CONCERNED_IMAGE_LIST:
            return True
        return False

    def get_unconcerned_images(self,unconcerned_image_dir):
        unconcerned_images = []
        for root,dirs,files in os.walk(unconcerned_image_dir):
            for name in files:
                unconcerned_images.append(os.path.join(root,name))
        return unconcerned_images

    def getImageMixHash(self,image):
        shape_hash = imagehash.average_hash(image)
        color_hash = imagehash.colorhash(image)
        return shape_hash,color_hash

    def getDiff(self,image_a,image_b):
        import numpy as np
        image_a_hash = self.getImageMixHash(image_a)
        image_b_hash = self.getImageMixHash(image_b)
        x = np.array(image_a_hash[0]-image_b_hash[0])
        y = np.array(image_a_hash[1]-image_b_hash[1])
        print("HASH: %s , %s =norm=> %s =New=> %s" % (x,y,np.linalg.norm(x-y),(x<=COMPARE_THRESHOLD and y<=COMPARE_THRESHOLD)))
        return x,y

    # 返回值表示是否正常，正常为True，不正常为False
    def check_unconcerned_image(self, image, compare_threshold, unconcerned_image):
        compare1,compare2 = self.getDiff(image, unconcerned_image)
        if (compare1 <= compare_threshold and compare2 <= compare_threshold):
            return False
        return True

    # -----------------------4. check have text new----------------------
    # 基本思路：不再为了争取百度ocr的识别次数，直接每次均分别用两个ocr工具，两者结果进行结合决策：两个分别的识别结果和置信度，放进csv文件中||
    # 两者识别都有文字，且符合Meaningful则有文字，一个识别有文字不算
    def isMeaningfulAlpha(self,characters,ocr_type):                                                                                   
        total_num = len(characters)
        if total_num == 0:
            return False
        digit_num = 0
        alpha_num = 0
        other_num = 0
        for i in characters:
            if i.isdigit():
                digit_num += 1
            elif i.isalpha():
                alpha_num += 1
            else:
                other_num += 1
        other_num_ratio = other_num / total_num
        # 单个的字母不认为是文字，单个的字母认为是文字
        # 所以如果只有1个字母，认为不是文字
        if total_num == 1 and alpha_num == 1:
            return False
        # 其他字符比例大于0.5,不是文字
        if other_num_ratio > 0.5:
            return False
        return True

    # 先搞简单的，如果两者均检测出东西，则认为确实有文字，否则不认
    def check_has_text(self,image_path,ocr_type):
        #try:
        global EASYOCR_READER
        global PADDLEOCR_READER
        result1 = EASYOCR_READER.readtext(image_path,detail=1)
        # [] or [([[28, 24], [98, 24], [98, 102], [28, 102]], '', 0.27468618750572205)] 
        if ocr_type != 'ar':
            result2 = PADDLEOCR_READER.ocr(image_path, cls=True)
            # [] or [[[[224.0, 47.0], [493.0, 33.0], [493.0, 217.0], [233.0, 230.0]], ('A7', 0.57441366)]]
            self.recogize_log.append([image_path,result1,result2])
            print([image_path,result1,result2])
        else:
            result2 = []
        #except:
        #    print('the image has problem, skipping.')
        #    return False
        if len(result1) ==0 and len(result2) ==0: 
            return False
        elif len(result1) !=0 and len(result2) == 0:
            print('result1检出文字')
            result1_text = ""
            result1_ratio = 0
            for line in result1:
                result1_text += line[1]
                if result1_ratio < line[2]:
                    result1_ratio = line[2]
            # 根据这个判断，如果一半以上是则是
            print(result1_text,'||',self.isMeaningfulAlpha(result1_text,ocr_type))
            return self.isMeaningfulAlpha(result1_text,ocr_type)
        elif len(result1) ==0 and len(result2) != 0:
            print('result2检出文字')
            result2_text = ""
            result2_ratio = 0
            for line in result2:
                result2_text += line[1][0]
                if result2_ratio < line[1][1]:
                    result2_ratio = line[1][1]
            # 根据这个判断，如果一半以上是则是
            print(result2_text,'||',self.isMeaningfulAlpha(result2_text,ocr_type))
            return self.isMeaningfulAlpha(result2_text,ocr_type)
        else:            
            return True 
            '''
            lowest_probability = min(np.array(result)[:,2])
            if ocr_type == 'ar':
                return True,result[0][1],result[0][2],'阿拉伯通过'
            vote_mode = False
            # 如果最低低0.5请求baidu ocr
            if lowest_probability < 0.5:
                baidu_ocr_result = self.check_by_Baidu_OCR(image_path,OCR_API_URL)
                if len(baidu_ocr_result) == len(result):
                    vote_mode = True
                else:# len(baidu_ocr_result) == 0:
                    return False,'',lowest_probability,'识别不同len Diff'

            returns = []
            for index in range(len(result)):
                if result[index][2] < 0.5 and vote_mode == True:
                    print('vote!')
                    if difflib.SequenceMatcher(None,result[index][1],baidu_ocr_result[index][1]).quick_ratio() < 0.5:
                        returns.append( (False,result[index][1]+'|'+baidu_ocr_result[index][1],result[index][2],'识别内容不同Diff') )
                        continue
                if self.isMeaningfulAlpha(result[index][1]):
                    returns.append( ( True,result[index][1],result[index][2],'全部符合All Pass') )
                else:
                    returns.append( ( False,result[index][1],result[index][2],'无意义字符Not Meaningful') )
            return returns
            '''
    # -----------------------4. check have text----------------------
    # 判断是不是有意义的文字(True将被删除，False保留)

    def next_access_token(self):
        global CHOOSE
        global ACCESS_TOKEN_POOL
        CHOOSE += 1
        if CHOOSE == 6:
            CHOOSE = 0
        return ACCESS_TOKEN_POOL[CHOOSE]

    def check_by_Baidu_OCR(self,image_path, ocr_api_url):
        #global QPS
        #if QPS % 6 == 0:
        #    time.sleep(REST_TIME)
        #QPS += 1

        # 二进制方式打开图片文件
        f = open(image_path, 'rb')
        img = base64.b64encode(f.read())
        params = {"image":img,"language_type":'ENG',"probability":"true"}
        access_token = self.next_access_token()
        request_url = ocr_api_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        #{'words_result': [{'probability': {'average': 0.720593, 'min': 0.720593, 'variance': 0.0}, 'words': 'P'}, {'probability': {'average': 0.946776, 'min': 0.946776, 'variance': 0.0}, 'words': 'Parkins'}], 'log_id': 1318464131694592000, 'words_result_num': 2}
        try:
            characters = response.json()
        except:
            try:
                print('调用接口发生错误,重试中.')
                response = requests.post(request_url, data=params, headers=headers)
                characters = response.json()
            except:
                print('调用接口失败，判为无字!')
                return 'noText'

        if 'error_code' in characters.keys():
            print('调用接口过于频繁,重试中.')
            time.sleep(1)
            response = requests.post(request_url, data=params, headers=headers)
            characters = response.json()
            print('调用接口失败，判为无字!')
            if 'error_code' in characters.keys():
                return 'noText'

        print('百度api识别:',characters)
        if characters['words_result_num'] == 0:
            return []
        else:
            result = []
            for character in characters['words_result']:
                result.append(([],character['words'],character['probability']['min']))
            return result

    def check_has_text_backup(self,image_path,ocr_type):
        try:
            global EASYOCR_READER
            result = EASYOCR_READER.readtext(image_path,detail=1)
        except:
            print('the image has problem, skipping.')
            return False,'','','读取问题Problem'
        if len(result) == 0:
            return False,'','','识别空EO Empty'
        else:            
            lowest_probability = min(np.array(result)[:,2])
            if ocr_type == 'ar':
                return True,result[0][1],result[0][2],'阿拉伯通过'
            vote_mode = False
            # 如果最低低0.5请求baidu ocr
            if lowest_probability < 0.5:
                baidu_ocr_result = self.check_by_Baidu_OCR(image_path,OCR_API_URL)
                if len(baidu_ocr_result) == len(result):
                    vote_mode = True
                else:# len(baidu_ocr_result) == 0:
                    return False,'',lowest_probability,'识别不同len Diff'

            returns = []
            for index in range(len(result)):
                if result[index][2] < 0.5 and vote_mode == True:
                    print('vote!')
                    if difflib.SequenceMatcher(None,result[index][1],baidu_ocr_result[index][1]).quick_ratio() < 0.5:
                        returns.append( (False,result[index][1]+'|'+baidu_ocr_result[index][1],result[index][2],'识别内容不同Diff') )
                        continue
                if self.isMeaningfulAlpha(result[index][1]):
                    returns.append( ( True,result[index][1],result[index][2],'全部符合All Pass') )
                else:
                    returns.append( ( False,result[index][1],result[index][2],'无意义字符Not Meaningful') )
            return returns
    # -----------------------5. add backgrund----------------------
    def add_background(self, image_path, size_threshold, bg_color, use_resize=False):
        # 先读出
        image = Image.open(image_path, 'r')
        maxw = maxh = 0
        # 原图的宽，高
        top_img_w, top_img_h = image.size
        # 如果图片不是正方形
        if (top_img_w > size_threshold or top_img_h > size_threshold) and use_resize == True:
            # 如果是比较宽的图片，w缩为224，h按比例缩小
            if top_img_w > top_img_h:
                new_img_h = int(top_img_h/(top_img_w/size_threshold))
                image = image.resize((size_threshold, new_img_h),Image.ANTIALIAS)
                top_img_w = size_threshold
                top_img_h = new_img_h
            else:
                new_img_w = int(top_img_w/(top_img_h/size_threshold))
                image = image.resize((new_img_w, size_threshold),Image.ANTIALIAS)
                top_img_w = new_img_w
                top_img_h = size_threshold


        # 如果这张图片尺寸没问题，那要检查颜色，是全白变成全黑
        #top_img = self.handleBlackandWhite(picture, top_img, trashPath)
        bottom_img = Image.new('RGB', (size_threshold, size_threshold), bg_color)
        # get the size or use 150x150 if it's constant
        bottom_img_w, bottom_img_h = bottom_img.size
        # offset the top image so it's placed in the middle of the bottom image
        offset = ((bottom_img_w - top_img_w) // 2, (bottom_img_h - top_img_h) // 2)
        # embed top_img on top of bottom_img
        bottom_img.paste(image, offset)
        output_name = image_path
        self.movePictureToTrash(image_path, self.config['PICTURES_TRASH_DIR']+'input_origin/')
        bottom_img.save(image_path)


    # ----------------------control----------------------
    def preprocess(self, use_resize=True , use_unconcerned_image=False, ocr_type='en', runcheck='run'):
        from PIL import Image
        # 遍历输入目录，这个目录是json和图片混合存储的
        for root, dirs, files in os.walk(self.config['INPUT_DATA_DIR']):
            for name in files:
                # 得到具体文件了，判断其后缀
                if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                    print('---------------------------------------------')
                    absolute_path = root+os.path.sep+name
                    try:
                        image = Image.open(absolute_path, 'r')
                    except:
                        print('Read', absolute_path ,' Error! Dropping it...')
                        self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'broken/')
                        print('Bad Read remove: ', absolute_path, ' w*h: ', image.size)
                        continue

                   # 首先去除尺寸不合适的控件，即宽高任意一个大小>300去除，纵横比> 4 或者 < 0.25去除
                    size_threshold_small = SIZE_THRESHOLD_SMALL
                    if use_resize == True:
                        size_threshold = EXTEND_SIZE_THRESHOLD
                    else:
                        size_threshold = SIZE_THRESHOLD
                    if not self.check_size(image, size_threshold,size_threshold_small):
                        self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'size/')
                        print('check_size remove: ', absolute_path, ' w*h: ', image.size)
                        continue
                    if not self.check_wh_ratio(image, WH_RATIO_THRESHOLD):
                        self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'wh/')
                        print('check_wh_ratio remove: ', absolute_path, ' w*h: ', image.size)
                        continue
                    # 接着去除纯色图片，他们在人眼看起来为纯色
                    if not self.check_color(image, COLOR_SIMILAR_THRESHOLD):
                        self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'color/')
                        print('check_color remove: ', absolute_path, ' w*h: ', image.size)
                        continue
                    # 接着去除某些人为指定的图片，这通过计算图片相似度来实现
                    if use_unconcerned_image == True:
                        # 如果和unconcerned_image_paths中的任意一个图像相似的话，就会直接break，然后continue下一个
                        # 如果是已经知道了会分错的，就直接不让它进行相似比较
                        if self.is_concerned_images(absolute_path):
                            continue
                        if self.is_not_concerned_images(absolute_path):
                            self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'man_unconcerned/')
                            print('check_unconcerned_image remove: ', absolute_path, ' w*h: ', image.size)
                            continue
                        unconcerned_images = self.get_unconcerned_images(UNCONCERNED_IMAGE_DIR)
                        continue_flag = False
                        for unconcerned_image_path in unconcerned_images:
                            try:
                                unconcerned_image = Image.open(unconcerned_image_path, 'r')
                            except:
                                print('Read unconcerned image ', unconcerned_image_path ,' Error! Omitting...')
                                break
                            # 返回True表示没有问题，False表示有问题
                            if not self.check_unconcerned_image(image, COMPARE_THRESHOLD, unconcerned_image ):
                                self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'unconcerned/')
                                print('check_unconcerned_image remove: ', absolute_path, ' w*h: ', image.size)
                                continue_flag = True
                                break
                        if continue_flag == True:
                            continue
                    '''
                    # 最后去除包含文字的图片，将先尝试使用easyocr，如果判不出文字则交给百度ocr识别
                    returns = self.check_has_text(absolute_path,ocr_type)
                    if isinstance(returns,list):
                        if 'True' in np.array(returns)[:,0]:
                            new_name = 'True ' + returns[-1][1] +":"+ str(returns[-1][2])[:4] +":"+returns[-1][-1]+' =>'+name
                        else:
                            new_name = 'False ' + returns[-1][1] + ":"+str(returns[-1][2])[:4] +":"+returns[-1][-1]+' =>'+name
                    else:
                        if returns[0] == True:
                            new_name = 'True ' + returns[1] + ":"+str(returns[2])[:4]+":"+returns[-1] + ' =>'+name
                        else:
                            new_name = 'False ' + returns[1] + ":"+str(returns[2])[:4]+":"+returns[-1] + ' =>'+name
                    print(new_name)
                    if new_name[0] == 'T':
                        self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'text/')
                        print('check_has_text remove: ', absolute_path, ' w*h: ', image.size)
                        continue
                    '''
                    if self.check_has_text(absolute_path,ocr_type):
                        self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'text/')
                        print('check_has_text remove: ', absolute_path, ' w*h: ', image.size)
                        continue
                    # 这里添加bg这个size阈值就是真的了，不像一开始那样还要考虑稍大于SIZE_THRESHOLD的
                    if runcheck == 'done':
                        self.add_background(absolute_path, SIZE_THRESHOLD, BG_COLOR, use_resize)
                        # 最后的图片将再需要经过人工审查确定没有其他异常的，即可进行下一步Transform

    # ----------------------统一背景--------------


    def get_contrasted(self,image, type="dark", level=3):
        maxIntensity = 255.0 # depends on dtype of image data
        phi = 1
        theta = 1

        if type == "light":
            newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
            newImage0 = np.array(newImage0,dtype='uint8')
            return newImage0
        elif type == "dark":
            newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**level
            newImage1 = np.array(newImage1,dtype='uint8')

            return newImage1

    def sharp(self,image, level=3):
        f = cv2.GaussianBlur(image, (level,level), level)
        f = cv2.addWeighted(image, 1.5, f, -0.5, 0)
        return f

    def colorVote(self,colors):
       '''根据传入的colors数组获取其中排名最高的颜色
           colors:  [0.48235294, 0.5372549 , 0. ],[0.48235294, 0.5372549 , 0.]
       '''
       color_dict = dict()
       for color in colors:
           if color.tobytes() in color_dict:
               color_dict[color.tobytes()] += 1
           else:
               color_dict[color.tobytes()] = 1
       sorted_dict = sorted(color_dict.items(),key=lambda item:item[1])
       last_color = np.frombuffer(sorted_dict[-1][0],dtype=np.uint8) 
       return last_color

    def getBackgroundColor(self,image_path):
        original_image = cv2.imread(image_path)
        # 1 Convert to gray & Normalize
        if original_image.shape[0] == original_image.shape[1] == 224:
            return (0,0,0)
        if len(original_image.shape) == 2:
            gray_img = original_image
        else:
            # 有可能不是RGB?
            try:
                gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            except:
                print('image_path:',image_path)
                print('original_image shape:',original_image.shape,'original_image[0]:',original_image[0][0])
        gray_img = self.sharp(self.get_contrasted(gray_img))
        gray_img = cv2.normalize(gray_img, None, 0, 255, NORM_MINMAX, CV_8UC1)
 
        # 2 Find Threshold
        #-------
        #gray_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        gray_blur = cv2.GaussianBlur(gray_img, (7, 7), 0)
        adapt_thresh_im = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
        max_thresh, thresh_im = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh = cv2.bitwise_or(adapt_thresh_im, thresh_im)
        
        high_thresh_val = max_thresh
        lower_thresh_val = max_thresh * 0.5;
 
        # 3 Dilate
        #-------
        gray = cv2.Canny(thresh, lower_thresh_val, high_thresh_val, apertureSize=3)
        #gray = cv2.Canny(thresh, 88, 400, apertureSize=3)
        gray = cv2.dilate(gray, None, iterations=8)
        gray = cv2.erode(gray, None, iterations=8)
 
        # 4 Flood
        #-------
        _, contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(original_image,contours,-1,(0,0,255),3)
        contour_info = []
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))
        #::: contour_info :::
        # [(array([[[81,61]],[[60,82]],[[60,85]],[[81,106]],[[84,106]],[[86,104]],[[86,87]],[[87,86]],[[106,86]],[[106,81]],[[105,80]],[[87,80]],[[86,79]],[[86,63]],[[84,61]]],dtype=int32),
        #False, 检查是否曲线是凸面
        #845.5)]  轮廓的面积
        
        if len(contour_info) == 0:
            colors = list()
            #
            #for original_image_i in original_image:
            #    for original_image_i_j in original_image_i:            
            #        colors.extend(list(background_i_j_d))
 
            # 策略2，随机取颜色矩阵的1%个像素点的颜色
            sample_num = int(original_image.shape[0]*original_image.shape[1]*0.01)
            for i in range(10 if sample_num < 10 else sample_num):
                i = random.randint(0,original_image.shape[0]-1)
                j = random.randint(0,original_image.shape[1]-1)
                #print('i,j',i,j,'=>',original_image[i][j])
                colors.append(np.array(original_image[i][j]))
                    
            last_color = self.colorVote(colors)
            return (last_color[-1],last_color[-2],last_color[-3])
        
        # contour_info[0][0].shape (15, 1, 2) 即由15个点组成的轮廓,每个点是一个二维的，存在一个单独的仅有1个元素的数组中
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        # 获取轮廓面积最大的那个轮廓
        max_contour = contour_info[0]  # 这句话出错是因为提取不出图片的边框，可能是纯色图片，给予删除最好
        holes = np.zeros(gray_img.shape, np.uint8)
        drawContours(holes, max_contour, 0, 255, -1)
        # holes应该是将图片轮廓全部绘制出来了，-1就是填充模式
 
        #-------
        mask = cv2.GaussianBlur(holes, (15, 15), 0)
        mask = np.dstack([mask] * 3)  # Create 3-channel alpha mask
 
        mask = mask.astype('float32') / 255.0  # Use float matrices,
        # mask.shape => (168, 167, 3)
        img = original_image.astype('float32') / 255.0  # for easy blending
        # original_image.shape (168, 167, 3)
        # original_image[102,84,:] => array([255, 255, 255], dtype=uint8)
        background = img[1-mask.astype('uint8')]
        # (168, 167, 3, 167, 3)
        # background[102][84][0] => ([ ... [0.48235294, 0.5372549 , 0.        ],      [0.48235294, 0.5372549 , 0.        ]], dtype=float32)
        background = (background * 255).astype('uint8')        
        colors = list()
        # 策略1，最保险，最慢，全部遍历所有颜色 run time: 14.90
        ''' 
        for background_i in background:
            for background_i_j in background_i:
                for background_i_j_d in background_i_j:
                    colors.extend(background_i_j_d)
        '''
        # 策略2，随机取颜色矩阵的1%个像素点的颜色 run time: 0.38        
        sample_num = int(background.shape[0]*background.shape[1]*0.01)
        for i in range(10 if sample_num < 10 else sample_num):
            i = random.randint(0,background.shape[0]-1)
            j = random.randint(0,background.shape[1]-1)
            k = random.randint(0,background.shape[2]-1)
            d = random.randint(0,background.shape[3]-1)
            #print('i,j,d:',i,j,d,'=>',background[i][j][k][d])
            colors.append(background[i][j][k][d])
        
        # 策略3，只取头尾 run time: 1.56
        #colors = [background[0][0][0][0],background[1][1][1][1],background[-2][-2][-2][-2],background[-1][-1][-1][-1]]
                    
        last_color = self.colorVote(colors)
        return (last_color[-1],last_color[-2],last_color[-3])

    def myMin(self,a,b,small_threshold):
        if a < b:
            return a
        else:
            return b

    def myMax(self,a,b,big_threshold):
        if a > b:
            return a
        else:
            return b

    def samplingInMartix(self,martix,sampling_ratio):
        sample_num = int(martix.shape[0]*martix.shape[1]*sampling_ratio)
        colors = []
        for x in range(10 if sample_num < 10 else sample_num):
            # 选取第i外拓行/列
            i = random.randint(0,martix.shape[0]-1)
            j = random.randint(0,martix.shape[1]-1)
            # 选取第j固有列/行
            #print('i,j,d:',i,j,d,'=>',background[i][j][k][d])
            colors.append(martix[i][j])
        return colors

    def getAdjacentBackground(self,raw_image,text_info_dict):
        # 向外10个像素，随机取
        #start: row col 
        #end: row col 
        raw_image_array = np.array(raw_image)
        raw_image_width = raw_image.size[0]
        raw_image_height = raw_image.size[1]
        # 上面 => for start.col -> end.col: for start.raw-10 -> start.raw
        top_dome = text_info_dict['start']['row'] - 10
        if top_dome < 0: top_dome = 0 
        top_extend_area = 10* text_info_dict['width']
        print(top_dome,text_info_dict['start']['row'], text_info_dict['start']['col'],text_info_dict['end']['col'])
        top_extend_martix = raw_image_array[top_dome:text_info_dict['start']['row'], text_info_dict['start']['col']:text_info_dict['end']['col']]
        # 下面 => for start.col -> end.col: for end.raw -> end.raw+10
        bottom_dome = text_info_dict['end']['row'] + 10
        if bottom_dome > raw_image_height: bottom_dome =  raw_image_height
        bottom_extend_area = 10* text_info_dict['width']
        bottom_extend_martix = raw_image_array[text_info_dict['end']['row']:bottom_dome, text_info_dict['start']['col']:text_info_dict['end']['col']]
        # 左面 => for start.row -> end.row: for start.col-10 -> start.col
        left_dome = text_info_dict['start']['col'] - 10
        if left_dome < 0: left_dome = 0 
        left_extend_area = 10* text_info_dict['height']
        left_extend_martix = raw_image_array[text_info_dict['start']['row']:text_info_dict['end']['row'], left_dome:text_info_dict['start']['col']]
        # 右面 => for start.row -> end.row: for end.col -> end.col+10
        right_dome = text_info_dict['end']['col'] + 10
        if right_dome > raw_image_width: right_dome =  raw_image_width
        right_extend_area = 10* text_info_dict['height']
        right_extend_martix = raw_image_array[text_info_dict['start']['row']:text_info_dict['end']['row'],text_info_dict['end']['col']:right_dome]


        colors = []
        if top_dome < text_info_dict['start']['row']:
            colors1 = self.samplingInMartix(top_extend_martix,0.01)
            colors += colors1
        if text_info_dict['end']['row'] < bottom_dome:
            colors2 = self.samplingInMartix(bottom_extend_martix,0.01)
            colors += colors2
        if left_dome < text_info_dict['start']['col']:
            colors3 = self.samplingInMartix(left_extend_martix,0.01)
            colors += colors3
        if text_info_dict['end']['col'] < right_dome:
            colors4 = self.samplingInMartix(right_extend_martix,0.01)
            colors += colors4

        last_color = self.colorVote(colors)
        return last_color

    def isValuableTextTrash(self,image_path,step='full'):
        # 首先用ocr检测获取文本情况
        global PADDLEOCR_READER
        global TEXT_HEIGHT_RATIO_THRESHOLD 
        global TEXT_AREA_RATIO_THRESHOLD 
        result = PADDLEOCR_READER.ocr(image_path, cls=True)
        if len(result) == 0 :
            print('================>No Text')
            return None

        # 图片基本信息,用于计算总面积，总长-高等
        raw_image = Image.open(image_path,'r').convert('RGB')
        raw_image_width = raw_image.size[0]
        raw_image_height = raw_image.size[1]
        raw_image_area = raw_image_width*raw_image_height

        text_confidence_threshold = TEXT_CONFIDENCE_THRESHOLD
        text_height_ratio_threshold = TEXT_HEIGHT_RATIO_THRESHOLD
        #text_width_ratio_threshold = TEXT_WIDTH_RATIO_THRESHOLD
        text_area_ratio_threshold = TEXT_AREA_RATIO_THRESHOLD
        #text_center_offset = TEXT_CENTER_OFFSET(raw_image_height)
        text_center_row = raw_image_height / 2
        
        text_info_dict_list = list()
        top_list = list()
        bottom_list = list()
        # [    #10Col,211Row
        #   [ [[10.0, 211.0], [198.0, 209.0], [198.0, 237.0], [10.0, 238.0]], ('MOUNTAIN', 0.9483581)],
        #   [ [[202.0, 211.0], [337.0, 212.0], [337.0, 238.0], [202.0, 236.0]], ('PASSES', 0.9953794)]
        #       left-top        right-top       right-bottom    lef-bottom
        # ]
        for item in result:
            text_info_dict = dict()

            text_info_dict['text-content'] = item[1][0]
            # 文字置信度应该较高 > 0.80
            text_info_dict['confidence'] = item[1][1]
            if text_info_dict['confidence'] < text_confidence_threshold:
                print('================>confidence low',text_info_dict['confidence'])
                return None
            if step == 'prepare':
                return True

            text_info_dict['left-top'] = dict()
            text_info_dict['right-top'] = dict()
            text_info_dict['right-bottom'] = dict()
            text_info_dict['left-bottom'] = dict()
            text_info_dict['left-top']['col'] = item[0][0][0]
            text_info_dict['left-top']['row'] = item[0][0][1]

            text_info_dict['right-top']['col'] = item[0][1][0]
            text_info_dict['right-top']['row'] = item[0][1][1]

            text_info_dict['right-bottom']['col'] = item[0][2][0]
            text_info_dict['right-bottom']['row'] = item[0][2][1]
            
            text_info_dict['left-bottom']['col'] = item[0][3][0]
            text_info_dict['left-bottom']['row'] = item[0][3][1]

            text_info_dict['start'] = dict()
            text_info_dict['end'] = dict()
            # 从左上角开始，列开始位置以小为准
            #print('***********************What???',text_info_dict['left-top']['col'],'\n',text_info_dict['left-bottom']['col'],'\n',self.myMin(text_info_dict['left-top']['col'],text_info_dict['left-bottom']['col']))
            text_info_dict['start']['col'] = int(self.myMin(text_info_dict['left-top']['col'],text_info_dict['left-bottom']['col'],raw_image_width))
            # 从左上角开始，行开始位置以小为准
            text_info_dict['start']['row'] = int(self.myMin(text_info_dict['left-top']['row'],text_info_dict['right-top']['row'],raw_image_width))
            # 到右下角开始,列结束位置以大为准
            text_info_dict['end']['col'] = int(self.myMax(text_info_dict['right-top']['col'],text_info_dict['right-bottom']['col'],raw_image_height))
            # 到右下角开始,行结束位置以大为准
            text_info_dict['end']['row'] = int(self.myMax(text_info_dict['left-bottom']['row'],text_info_dict['right-bottom']['row'],raw_image_height))

            top_list.append(text_info_dict['left-top']['row'])
            sorted(top_list) # 升序
            bottom_list.append(text_info_dict['right-bottom']['row'])
            sorted(bottom_list,reverse=True) # 降序
            if top_list[0] >= bottom_list[0]:
                print('================>many line',text_info_dict,top_list,bottom_list)
                return None

            text_info_dict['width'] = text_info_dict['right-top']['col'] - text_info_dict['left-top']['col']
            text_info_dict['height'] = text_info_dict['right-bottom']['row'] - text_info_dict['right-top']['row']
            # 过滤策略，，通过外部函数或者参数定义
            # 文字长宽应该符合一定限制(height < 30%)TEXT_HEIGHT()
            #print('================>high',text_info_dict['height'],raw_image_height)
            if text_info_dict['height']/raw_image_height > text_height_ratio_threshold:
                print('================>too high',text_info_dict['height'],raw_image_height)
                return None
            text_info_dict['area'] = text_info_dict['width'] * text_info_dict['height']
            #print('================>big',text_info_dict['area'],raw_image_area)
            # 文字所占总面积比例不超过15% TEXT_AREA
            if text_info_dict['area']/raw_image_area > text_area_ratio_threshold:
                print('================>too big',text_info_dict['area'],raw_image_area)
                return None
            # 文字的位置不能在正中间,不能跨越中心线 #出现在图片的中心位置 TEXT_CENTER_OFFSET=(center_top,center_bottom)
            top_cross = text_info_dict['left-top']['row'] - text_center_row
            bottom_cross = text_info_dict['right-bottom']['row'] - text_center_row

            #if (text_center_offset[0] < text_info_dict['left-top']['row'] < text_center_offset[1]) or \
            #   (text_center_offset[0] < text_info_dict['right-bottom']['row'] < text_center_offset[1]):
            if top_cross * bottom_cross <= 0:
                print('================>in center',text_info_dict)
                return None
            text_info_dict_list.append(text_info_dict)

        # 全部运行下来，所有item均加入后，再次检查
        # 文字只有一行 任何text_item的top不得大于任何其他item的bottom 
        #=>top数组保持从低到高[top1,top2] |  bottom数组，保持从高到底[bottom1,bottom2]  | 最低的top也应该比最高的bottom高
        '''
        if top_list[0] >= bottom_list[0]:
            print('================>finally many line',text_info_dict_list,top_list,bottom_list)
            return None
        '''
        print('success!',text_info_dict_list,'\n-----------------------------------------')
        return raw_image,text_info_dict_list

    def removeTextandReturn(self,image_path):
        judge_result = self.isValuableTextTrash(image_path)
        if judge_result == None:
            return None
        else:
            raw_image,text_info_dict_list = judge_result

        #background_color = self.getBackgroundColor(image_path)
        raw_image_array = np.array(raw_image)

        for text_info_dict in text_info_dict_list:
            background_color = self.getAdjacentBackground(raw_image,text_info_dict) 
            for i in range(int(text_info_dict['start']['row']),int(text_info_dict['end']['row'])):
                for j in range(int(text_info_dict['start']['col']),int(text_info_dict['end']['col'])):
                    raw_image_array[i][j] = background_color 

        removed_image = Image.fromarray(raw_image_array)

        # 最后必须经过聚类审核(最好单独审核）
        return removed_image

    # ----------------------Uitily----------------------
    def createset(self, params):
        # 创建的数据集是从本数据集中随机抽取得来，所以需要注意打开的数据集
        # 创建数据集,抽取1500张图片，期望最后有1000个控件可以进入数据库
        
        params = params.split(',')
        sample_nums = int(params[0])
        # 没有数量检查，注意数量不要超过本数据集数量
        new_dataset_name = params[1]
        
        from PIL import Image
        # 运行一次循环，将所有文件路径保存，当然仅保存相对路径
        picture_list = []
        json_set = set()
        for root, dirs, files in os.walk(self.config['INPUT_DATA_DIR']):
            for name in files:
                absolute_path = root+os.path.sep+name
                # 得到具体文件了，判断其后缀
                if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                    print('---------------------------------------------')
                    try:
                        image = Image.open(absolute_path, 'r')
                    except:
                        print('Read', absolute_path ,' Error! Dropping it...')
                        continue
                    # 将图片移动到指定数据集文件夹内
                    picture_list.append(absolute_path.replace(self.config['INPUT_DATA_DIR'],''))
                if ('.json' in name):
                    json_set.add(absolute_path.replace(self.config['INPUT_DATA_DIR'],''))
        
        choicen_picture_list = np.random.choice(picture_list,sample_nums,replace=False,p=None)
        print(len(choicen_picture_list))
        
        for picture in choicen_picture_list:
            origin_path =self.config['INPUT_DATA_DIR']+picture
            new_path = self.config['RAW_ROOT']+new_dataset_name+'/input_data'+os.path.sep+picture
            print(origin_path,'=>',new_path)
            self.copyFile(origin_path,new_path)
        
        for json_file in json_set:
            print(self.config['INPUT_DATA_DIR']+json_file,'=>',self.config['RAW_ROOT']+new_dataset_name+'/input_data'+os.path.sep+json_file)
            self.copyFile(self.config['INPUT_DATA_DIR']+json_file,self.config['RAW_ROOT']+new_dataset_name+'/input_data'+os.path.sep+json_file)
        
        print('创建成功，数据集名称为:',new_dataset_name)

    def unistrategy(self, params):
        '''对于每个数据集，重新架构input_data部分：
        每个strategy的input_data部分都有自己的名字，然后用软链接连接
        '''
        params = params.split(',')
        dataset_name = params[0]
        # 没有数量检查，注意数量不要超过本数据集数量
        strategy = params[1]
        refence_dataset_name = params[2]

        # 两个形式化链接文件
        input_data_path = self.config['INPUT_DATA_DIR']
        input_data_infos_path = self.config['INPUT_DATA_DIR'].replace('input_data/','infos/')

        # 文本垃圾文件
        input_trash_path  = self.config['PICTURES_TRASH_DIR'] + 'text/'
        input_prepare_trash_path  = self.config['INPUT_DATA_DIR'].rstrip('/')+'_'+'preparerecycletext/'
        # 仅仅用于备份
        input_data_origin_path = self.config['INPUT_DATA_DIR'].rstrip('/')+'_origin/'
        input_data_strategy_path = self.config['INPUT_DATA_DIR'].rstrip('/')+'_'+strategy+'/'
        input_data_strategy_swap_path = self.config['INPUT_DATA_DIR'].rstrip('/')+'_'+strategy+'_swap/'
        input_data_strategy_trash_path = self.config['INPUT_DATA_DIR'].rstrip('/')+'_trash/'
        # 这个才用于参考
        input_data_refence_path = self.config['RAW_ROOT']+refence_dataset_name+'/input_data/'

        # 真正的infos文件夹,在input_data内部
        input_data_strategy_infos_path = input_data_strategy_path + 'infos/'
        print('input_data_path,input_trash_path,input_data_origin_path,input_data_strategy_path,input_data_refence_path')
        print(input_data_path,'\n',input_trash_path,'\n',input_data_origin_path,'\n',input_data_strategy_path,'\n',input_data_refence_path)

        # 首先检查是否有这个strategy
        if os.path.exists(input_data_strategy_path):
            print(input_data_strategy_path,' exists, directly to link.')
            # 设置input_data链接
            # 直接切换即可
            if os.path.islink(input_data_path.rstrip('/')):
                print('removing old link file.')
                os.remove(input_data_path.rstrip('/'))
            print('ln -s '+ input_data_strategy_path + ' ' +input_data_path.rstrip('/'))
            os.system('ln -s '+ input_data_strategy_path + ' ' +input_data_path.rstrip('/'))

            # 设置infos目录链接
            if not os.path.exists(input_data_strategy_infos_path):
                os.makedirs(input_data_strategy_infos_path)
            if os.path.islink(input_data_infos_path.rstrip('/')):
                print('removing old link file.')
                os.remove(input_data_infos_path.rstrip('/'))
            print('ln -s '+ input_data_strategy_infos_path + ' ' +input_data_infos_path.rstrip('/'))
            os.system('ln -s '+ input_data_strategy_infos_path + ' ' +input_data_infos_path.rstrip('/'))

            print('链接成功，数据子集名称为:',input_data_strategy_path.split('/')[-2])
            return ''
        elif os.path.exists(input_data_strategy_swap_path):
            print('find swap dir, turn to continue') 
        elif strategy == 'preparerecycletext':
            print('no need to copy origin_path.') 
        elif strategy == 'recycletext':
            print('no need to copy origin_path.') 
        else:
            # 开始构建,构建中不需要origin_data
            if os.path.exists(input_data_origin_path):
                print(input_data_origin_path,' exists, copy to ',input_data_strategy_swap_path)
                shutil.copytree(input_data_origin_path, input_data_strategy_swap_path)
            else:
                # 首先复制一个已有数据集中的input_data部分为input_data_origin,再将原名更改
                print(input_data_origin_path,' not exists, copy from ', input_data_path ,'to ',input_data_strategy_swap_path)
                shutil.copytree(input_data_path, input_data_origin_path)
                shutil.copytree(input_data_origin_path, input_data_strategy_swap_path)

        # 添加包含文字的话，需要遍历origin文件夹
        if strategy == 'preparerecycletext':
            print('strategy == ',strategy)
            '''
            首先使用大小，颜色等重新检查一遍
            用ocr检查文本高不高
            然后保存一个集合为preparerecycletext
            使用IC prepare IC auto方法进行聚类，审核完成后，保存了比较好的图片
            '''
            choicen_trash_data_list = []
            for root,dirs,files in os.walk(input_trash_path):
                for name in files:
                    absolute_path = root+os.path.sep+name
                    if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                        print('---------------------------------------------')
                        absolute_path = root+os.path.sep+name
                        try:
                            image = Image.open(absolute_path, 'r')
                        except:
                            print('Read', absolute_path ,' Error! Dropping it...')
                            self.movePictureToTrash(absolute_path, self.config['PICTURES_TRASH_DIR']+'broken/')
                            print('Bad Read remove: ', absolute_path, ' w*h: ', image.size)
                            continue
                       # 首先去除尺寸不合适的控件，即宽高任意一个大小>300去除，纵横比> 4 或者 < 0.25去除
                        size_threshold_small = SIZE_THRESHOLD_SMALL
                        use_resize = True
                        if use_resize == True:
                            size_threshold = EXTEND_SIZE_THRESHOLD
                        else:
                            size_threshold = SIZE_THRESHOLD
                        if not self.check_size(image, size_threshold,size_threshold_small):
                            print('check_size remove: ', absolute_path, ' w*h: ', image.size)
                            continue
                        if not self.check_wh_ratio(image, WH_RATIO_THRESHOLD):
                            print('check_wh_ratio remove: ', absolute_path, ' w*h: ', image.size)
                            continue
                        # 接着去除纯色图片，他们在人眼看起来为纯色
                        if not self.check_color(image, COLOR_SIMILAR_THRESHOLD):
                            print('check_color remove: ', absolute_path, ' w*h: ', image.size)
                            continue
                        if self.isValuableTextTrash(absolute_path,'prepare') == None:
                            print('low text confidence remove: ', absolute_path, ' w*h: ', image.size)
                            continue

                        # 将其加入prepare数据集,不需要保持目录结构(这种数据集只会预先聚类，不需要json文件)
                        print('reading ',absolute_path,' from trash_data.')
                        print('will saving to :',input_prepare_trash_path+'images/'+name)
                        self.copyFile(absolute_path,input_prepare_trash_path+'images/'+name)
                        self.add_background(input_prepare_trash_path+'images/'+name, SIZE_THRESHOLD, BG_COLOR , use_resize=True)

            print('preparerecycletext done',input_prepare_trash_path)
        elif strategy == 'recycletext':
            print('strategy == ',strategy)
            #首先构建trashdata的目录数组
            trash_data_list = []
            for root,dirs,files in os.walk(input_prepare_trash_path):
                for name in files:
                    absolute_path = root+os.path.sep+name
                    if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                        trash_data_list.append(name)
            print('trash_data_list created.',trash_data_list[:3])
            # 现在trash_data中的名单已经获取，，接下来，直接从refence_path中获取指定的图片尽心分析即可!
            for root,dirs,files in os.walk(input_data_refence_path):
                for name in files:
                    absolute_path = root+os.path.sep+name
                    # 只有满足三个条件，，其中还需要在trash_data_list中才能进行操作
                    if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name) and (name in trash_data_list):
                        print('reading ',absolute_path,' from refence_path.')
                        path_prefix = root.split('input_data/')[-1]
                        recycled_img_path = input_data_strategy_swap_path+path_prefix+os.path.sep+name
                        trash_img_path = input_data_strategy_trash_path+os.path.sep+name
                        print('will saving to :',recycled_img_path)

                        recycled_image = self.removeTextandReturn(absolute_path)
                        if recycled_image != None:
                            print('pass,saving.')
                            self.makeSureExists(recycled_img_path)
                            recycled_image.save(recycled_img_path)
                            self.add_background(recycled_img_path, SIZE_THRESHOLD, BG_COLOR , use_resize=True)
                        else:
                            # fail, to trash ,for look
                            #self.copyFile(absolute_path,trash_img_path)
                            self.makeSureExists(trash_img_path)
                            shutil.copy(absolute_path,trash_img_path)
            os.rename(input_data_strategy_swap_path,input_data_strategy_path)
        # 其他的情况，遍历最后要生成的文件夹，相当于替换
        else:
            print('strategy == ',strategy)
            for root,dirs,files in os.walk(input_data_strategy_swap_path):
                for name in files:
                    absolute_path = root+os.path.sep+name
                    if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                        print('be Replacing:',absolute_path)
                        path_prefix = root.split('input_data/')[-1]
                        refence_img_path = input_data_refence_path+path_prefix+os.path.sep+name
                        print('Replace with:',refence_img_path)
                        # strategy1: 直接resize
                        if strategy == 'resize':
                            refence_img = Image.open(refence_img_path).convert('RGB')
                            resize_img = refence_img.resize((SIZE_THRESHOLD,SIZE_THRESHOLD),Image.ANTIALIAS)
                            resize_img.save(absolute_path)
                        # strategy1: 直接resize
                        elif strategy == 'fillwithbg':
                            # 然后根据在数据集内的所有图片对的名字，找到对应原图片
                            # 然后对原图片进行resize，再替换本文件
                            # 对于本身就是接近正方形的图片，直接resize
                            background_color = self.getBackgroundColor(refence_img_path)
                            print('background_color:',background_color)
                            refence_img = Image.open(refence_img_path).convert('RGB')
                            refence_img.save(absolute_path)
                            self.add_background(absolute_path, SIZE_THRESHOLD, background_color , use_resize=True)
            # all done ,rename the dir ,and print
            os.rename(input_data_strategy_swap_path,input_data_strategy_path)
        # 设置input_data链接
        if os.path.islink(input_data_path.rstrip('/')):
            print('removing old link file.')
            os.remove(input_data_path.rstrip('/'))
        print('ln -s '+ input_data_strategy_path + ' ' +input_data_path.rstrip('/'))
        os.system('ln -s '+ input_data_strategy_path + ' ' +input_data_path.rstrip('/'))

        # 设置infos目录链接
        if not os.path.exists(input_data_strategy_infos_path):
            os.makedirs(input_data_strategy_infos_path)
        if os.path.islink(input_data_infos_path.rstrip('/')):
            print('removing old link file.')
            os.remove(input_data_infos_path.rstrip('/'))
        print('ln -s '+ input_data_strategy_infos_path + ' ' +input_data_infos_path.rstrip('/'))
        os.system('ln -s '+ input_data_strategy_infos_path + ' ' +input_data_infos_path.rstrip('/'))

        print('链接成功，数据子集名称为:',input_data_strategy_path.split('/')[-2])
        print('创建成功，数据子集名称为:',input_data_strategy_path.split('/')[-2])
        return ''

    def merge(self, params):
        '''合并数据集，前面是主，后面是辅
        '''
        params = params.split(',')
        dataset_name = params[0]
        master_input_data_name = params[1]
        slave_input_data_name = params[2]

        master_input_data_path = self.config['INPUT_DATA_DIR'].rstrip('/')+'_'+master_input_data_name+'/'
        slave_input_data_path = self.config['INPUT_DATA_DIR'].rstrip('/')+'_'+slave_input_data_name+'/'


        if not os.path.exists(master_input_data_path):
            ans = input('master数据集不存在!是否创建?')
            if ans == 'y': os.makedirs(master_input_data_path)
        if not os.path.exists(slave_input_data_path):
            print('slave数据集不存在,退出!')
            return ''

        for root,dirs,files in os.walk(slave_input_data_path):
            for name in files:
                absolute_path = root+os.path.sep+name
                if (IMAGE_SUFFIX in name or IMAGE_SUFFIX1 in name) and (DROP_SUFFIX not in name):
                    print('merging:',absolute_path)
                    path_prefix = '/'.join(root.split('/')[-3:])
                    master_img_path = master_input_data_path+path_prefix+os.path.sep+name
                    print('copy to:',master_img_path)
                    self.copyFile(absolute_path,master_img_path)
                if ('.json' in name):
                    path_prefix = '/'.join(root.split('/')[-2:])
                    master_img_path = master_input_data_path+path_prefix+os.path.sep+name
                    self.copyFile(absolute_path,master_img_path)
        print('合并成功,主数据集为:',master_input_data_path.split('/')[-2])


    def runDataPreprocess(self, operator='', params=[], config=[]):
        run_log = []
        global CONCERNED_IMAGE_LIST_PATH
        global NOT_CONCERNED_IMAGE_LIST_PATH
        CONCERNED_IMAGE_LIST_PATH = self.config['PICTURES_TRASH_DIR'] + 'concerned_image.txt'
        NOT_CONCERNED_IMAGE_LIST_PATH = self.config['PICTURES_TRASH_DIR'] + 'not_concerned_image.txt'
        if operator == 'info' or operator == '':
            self.info()
        if operator == 'createset':
            self.createset(params)
        if operator == 'ls':
            # 了解数据集，读取每个数据集的信息，如果recover该数据集，将在数据库中定义该数据库
            output = os.popen("find %s  -maxdepth 1 -name 'db_*'" % (self.config['RAW_ROOT'])).read()
            print(output) 
            watch_dir = 'input_data'
            while watch_dir != 'none':
                watch_dir = input('watch which ?')
                if watch_dir == 'none':
                    print('done.')
                    return run_log
                else:
                    print('Current=>',self.config['INPUT_DATA_DIR'])
                    # 2020.12.28已更改为读取统一的datasetinfo.txt
                    datasets_info_path = self.config['RAW_ROOT'] + 'datasets_info.txt'
                    if not os.path.exists(datasets_info_path):
                        print('no description file.')
                        return run_log
                    f = open(datasets_info_path)
                    # 输出说明文件内容
                    content = f.read()
                    f.close()
                    start_index = content.find('#'+watch_dir+'#')
                    if start_index == -1:
                        print('no related description in file.')
                        return run_log
                    else:
                        print('===Text Content===\n','\n'.join(content[start_index:].split('\n')[:13]))
                        print('===Text Content===\n')

        global EASYOCR_READER
        global PADDLEOCR_READER
        if operator == 'unistrategy':
            PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en",use_gpu=True)
            self.unistrategy(params)
        if operator == 'merge':
            self.merge(params)
        if operator == 'run':
            if DEBUG == False:
            #?# 使用前注意检查有没有！！！
                rawdata = self.DBP.queryRawAPKTree()
                if rawdata:
                    print('/\/\ rawdata exists, skipping.')
                    return run_log

            use_resize = use_unconcerned_image = False
            use_resize = True
            ocr_type = 'en'
            runcheck = 'run'
            #os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            EASYOCR_READER = easyocr.Reader(['en'],gpu=True)
            PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en",use_gpu=True)
            #EASYOCR_READER = easyocr.Reader(['ar'],gpu=False)
            #PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en",use_gpu=False)

            self.preprocess(use_resize,use_unconcerned_image,ocr_type,runcheck)
            for i in self.recogize_log:
                print(i)
            print('PICUTRE ALL DONE.')
            return run_log

        # 二次跑，这一次主要是筛掉阿拉伯文字，uncorenimage,等等（先不考虑中文）
        if operator == 'check':
            if DEBUG == False:
            #?# 使用前注意检查有没有！！！
                rawdata = self.DBP.queryRawAPKTree()
                if rawdata:
                    print('/\/\ rawdata exists, skipping.')
                    return run_log
            
            use_resize = use_unconcerned_image = True
            ocr_type = 'ar'
            runcheck = 'done'
            #runcheck = 'done'
            #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            EASYOCR_READER = easyocr.Reader(['ar'],gpu=True)
            PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en",use_gpu=True)
            #EASYOCR_READER = easyocr.Reader(['ar'],gpu=False)
            #PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en",use_gpu=False)
            self.preprocess(use_resize,use_unconcerned_image,ocr_type,runcheck)
            print('PICUTRE ALL DONE.')
            return run_log




