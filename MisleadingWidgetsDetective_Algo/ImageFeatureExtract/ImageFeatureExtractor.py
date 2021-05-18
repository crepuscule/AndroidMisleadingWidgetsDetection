import sys,os
sys.path.append(os.path.abspath('..')) 
from Base import BaseProcessor
class ImageFeatureExtractor(BaseProcessor.BaseProcessor):
    def __init__(self, config, DBP):
        super(ImageFeatureExtractor, self).__init__()
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
    Desc:
        提取特征模块,从生成好的应用树中得到图片路径对其提取特征
        transform csv records to dicts
 
        feed meanful record
        write dicts file
        show dicts length
 
    methods list:
        def runTransform(self,operator='',params=[]):
            if operator == 'extif':
    Example:
        IF extif spm
        IF extif hog

    Update:
        使用queryAllPath中获取图片的id-path对，该id-path对相当于mRNA
        依据该mRNA对该id-path对进行翻译，生成pictureList相当于pro.
        最后对pictureList进行特征提取，转存到文件中
        '''
        print(info)
    def transcription(self):
        import numpy as np
        idUpath = self.DBP.queryAllPath()
        path = np.array(idUpath)[:,1] # 取半条链作为mRNA
        path = [self.config['INPUT_DATA_DIR'] + x for x in path]
        return path #最后返回的是list型


    def extractHOG(self,pictureList,size):
        # S3 获取图片的方向梯度直方图
        def getHOG(picture):
            from skimage import feature as ft
            from skimage import io
            features,hogImage = ft.hog(picture, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
            #io.imshow(hogImage)
            #io.show()
            return features

        featureList = []
        for picture in pictureList:
            featureList.append(getHOG(picture.reshape(size)))

        return featureList
    
    #extractIF:提取图像特征
    def extractIF(self,method,pictureList,size):                
        extractIFFun = ''
        if method == 'hog':
            return self.extractHOG(pictureList,size)
        elif method =='hsv':
            # S4 获取图片的颜色直方图
            pass
        elif method == 'spm':
            #from . import spm
            from ImageFeatureExtract.spm import spm
            from importlib import reload
            reload(spm)
            return spm.getSPM(pictureList,self.config['SPM_CODE_BOOK_PATH'])         
        elif method == 'scspm':
            from scspm import scspm
            from importlib import reload
            reload(scspm)
            return scspm.getSCSPM(pictureList,self.config['SPM_CODE_BOOK_PATH'])         
        elif method == 'cnn-resnet':
            from ImageFeatureExtract.cnn import testResnetFeatureExtractor
            from importlib import reload
            reload(testResnetFeatureExtractor)
            return testResnetFeatureExtractor.runExtract(pictureList)
        elif method == 'cnn-densnet':
            from ImageFeatureExtract.cnn import testDensnetFeatureExtractor
            from importlib import reload
            reload(testDensnetFeatureExtractor)
            return testDensnetFeatureExtractor.runExtract(pictureList)
        elif method == 'cnn-vgg':
            from ImageFeatureExtract.cnn import testVGGFeatureExtractor
            from importlib import reload
            reload(testVGGFeatureExtractor)
            return testVGGFeatureExtractor.runExtract(pictureList)
        elif method == 'ae':
            from ImageFeatureExtract.ae import testAutoEncoder
            from importlib import reload
            reload(testAutoEncoder)
            return testAutoEncoder.runExtract(pictureList)
        else:
            print('no method selected!')
            
    #runCluster:运行最佳参数的聚类
    def runExtractor(self,operator=[],params=[]):
        runLog = []
        runLog.append(operator)
        if operator == 'info' or operator=='':
            self.info()
        if operator == 'usedefault':
            self.makeSureExists(self.config['EXTRACTED_FEATURE_PATH']) 
            self.copyFile(self.config['RAW_ROOT']+'pretrain_data/ExtractedFeature-default.txt',
                    self.config['EXTRACTED_FEATURE_PATH'])
        if operator == 'extif':
            # 获得mRNA
            pictureList = self.transcription()
            imageMatrix,imageSize = self.loadPictureData(pictureList,'cv2')
            #提取全部图像的特征
            if 'cnn' in params:
                # cnn专用
                extractedFeatures = self.extractIF(params,pictureList,imageSize)
            else:
                extractedFeatures = self.extractIF(params,imageMatrix,imageSize)
            self.writeVector(extractedFeatures,self.config['EXTRACTED_FEATURE_PATH'])#+'-'+params)
            self.DBP.updateMetaData({"$set":{"imagesize":str(imageSize[0])+'x'+str(imageSize[1]),"ifmethod":params,"ifdim":len(extractedFeatures[0])}})
            runLog.append('len extractedFeatures:%d' % len(extractedFeatures))

        print('\n---------------------------runLog:----------------------------\n',runLog)
        return runLog
