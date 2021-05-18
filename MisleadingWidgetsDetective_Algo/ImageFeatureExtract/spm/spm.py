from . import utils
import cv2
import numpy as np

# 构建空间金字塔 
# 输入使用图像，描述子，层级
def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    重建描述子
    """
    assert 0 <= level <= 2, "Level Error"
    DSIFT_STEP_SIZE = 4
    step_size = DSIFT_STEP_SIZE    
    s = utils.DSIFT_STEP_SIZE 
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
                            
    # 通过step_size将图片分成若干份
    h = image.shape[0] // step_size
    w = image.shape[1] // step_size
    # 通过h,w的
    idx_crop = np.array(range(len(descriptor))).reshape(h,w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height//bh, width//bw, bh, bw)
    strides = size * np.array([width*bh, bw, width, 1])
    crops = np.lib.stride_tricks.as_strided(
            idx_crop, shape=shape, strides=strides)
    des_idxs = [col_block.flatten().tolist() for row_block in crops
                for col_block in row_block]
    pyramid = []
    for idxs in des_idxs:
        pyramid.append(np.asarray([descriptor[idx] for idx in idxs]))
    return pyramid

# 使用spm提取特征
def spatial_pyramid_matching(image, descriptor, codebook, level):
    pyramid = []
    if level == 0:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        code = [utils.input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [utils.input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        # 相当于运行一次BOW了，这是对于第0层次提取直方图，按图像划分区域，然后统计不同特征计入pyramid
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        # 相当于对于pyramid中的每个层次统计了特征
        code = [utils.input_vector_encoder(crop, codebook) for crop in pyramid]
        # 不同的尺度下的match应赋予不同权重，显然大尺度的权重小，而小尺度的权重大
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        # 将这三个特征拼接在一起作为特征返回
        return np.concatenate((code_level_0, code_level_1, code_level_2))

def loadPictureData(pictureList, size,test=None):
    X = []
    for each in pictureList:
        print(each)
        img = cv2.imread(each)
        if img.all() == None:
            print('None')
        if img.shape[:2] != (size,size):
            img = cv2.resize(img, (size,size))
        X.append(img)
    return X

def saveVector(saveContent,savePath,fmt="%f"):
    import numpy 
    numpy.savetxt(savePath,saveContent,fmt=fmt)
    print(len(saveContent),' records writen.')

def convert2CV2(SX):
    SX = [cv2.cvtColor(np.asarray(img),cv2.COLOR_BGR2GRAY) for img in SX]
    return SX

# SX是已经取到的图像数据
def getSPM(SX,codeBookPath):
    #SX = convert2CV2(SX)
    # 码本数量100
    VOC_SIZE = 100
    # 金字塔等级 1
    PYRAMID_LEVEL = 2

    DSIFT_STEP_SIZE = 4
    # DSIFT_STEP_SIZE is related to the function
    # extract_DenseSift_descriptors in utils.py
    # and build_spatial_pyramid in spm.py      
    
    # 密集SIFT特征提取
    print( "Dense SIFT feature extraction" )
    # 对于x_train中的每一副图片，提取密集SIFT特征
    x_train_feature = [utils.extract_DenseSift_descriptors(img) for img in SX]
    # 函数在调用多个参数时，在列表、元组、集合、字典及其他可迭代对象作为实参，并在前面加 * ，因为x_train_feature本身就是一个两个元素的元祖
    # 解包成关键点和描述符
    x_train_kp, x_train_des = zip(*x_train_feature)    
    
    # 训练测试集划分
    #print( "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
    # 码本大小
    print( "Codebook Size: {:d}".format(VOC_SIZE))
    # 金字塔等级
    print( "Pyramid level: {:d}".format(PYRAMID_LEVEL))
    
    import os

    if os.path.exists(codeBookPath) == True:
        import pickle
        print( "loading the codebook...")
        read_file = open(codeBookPath, 'rb')
        codebook = pickle.load(read_file)
        read_file.close()  
    else:
        # 构建码本
        print( "Building the codebook, it will take some time")
        #首先确保该文件所在的目录都存在，这样即使构建码本时也可以创建路径
        pathDir = "/".join(codeBookPath.split('/')[:-1])
        if not os.path.exists(pathDir):
            print('making dir:',pathDir)
            os.makedirs(pathDir)

        # 使用x_train的描述符以及码本大小来构建码本
        codebook = utils.build_codebook(x_train_des, VOC_SIZE)
        # 存储起来
        import pickle
        try:
            write_file = open(codeBookPath, 'wb')
        except:
            print('codeBookPath not exist,using default path.')
            write_file = open('codeBook.pkl', 'wb')
        pickle.dump(codebook, write_file)
        write_file.close()


    # SPM编码开始
    print( "Spatial Pyramid Matching encoding")
    # 将每个图像使用SPM函数进行编码，输入x_train，原数据,x_train_des，描述符，码本，层级
    SX = [spatial_pyramid_matching(SX[i],
                                        x_train_des[i],
                                        codebook,
                                        level=PYRAMID_LEVEL)
                                        for i in range(len(SX))]
    SX = np.asarray(SX)
    #print(SX)
    #print(SX.shape)
    return SX
    #saveVector(SX,'/data2/wangruifeng/datasets/WidgetClustering/staticWidgetResources/intermediate_data/VectorClustering/extracetd/spm-extFeature.txt')
