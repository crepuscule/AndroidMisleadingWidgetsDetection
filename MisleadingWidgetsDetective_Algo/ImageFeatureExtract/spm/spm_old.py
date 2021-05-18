from utils import load_cifar10_data
from utils import extract_DenseSift_descriptors
from utils import build_codebook
from utils import input_vector_encoder
from classifier import svm_classifier

import numpy as np

# 构建空间金字塔 
# 输入使用图像，描述子，层级
def build_spatial_pyramid(image, descriptor, level):
    """
    Rebuild the descriptors according to the level of pyramid
    重建描述子
    """
    assert 0 <= level <= 2, "Level Error"
    step_size = DSIFT_STEP_SIZE
    from utils import DSIFT_STEP_SIZE as s
    assert s == step_size, "step_size must equal to DSIFT_STEP_SIZE\
                            in utils.extract_DenseSift_descriptors()"
                            
    # 通过step_size将图片分成若干份
    h = image.shape[0] / step_size
    w = image.shape[1] / step_size
    # 通过h,w的
    idx_crop = np.array(range(len(descriptor))).reshape(h,w)
    size = idx_crop.itemsize
    height, width = idx_crop.shape
    bh, bw = 2**(3-level), 2**(3-level)
    shape = (height/bh, width/bw, bh, bw)
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
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        return np.asarray(code).flatten()
    if level == 1:
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        code_level_0 = 0.5 * np.asarray(code[0]).flatten()
        code_level_1 = 0.5 * np.asarray(code[1:]).flatten()
        return np.concatenate((code_level_0, code_level_1))
    if level == 2:
        # 相当于运行一次BOW了，这是对于第0层次提取直方图，按图像划分区域，然后统计不同特征计入pyramid
        pyramid += build_spatial_pyramid(image, descriptor, level=0)
        pyramid += build_spatial_pyramid(image, descriptor, level=1)
        pyramid += build_spatial_pyramid(image, descriptor, level=2)
        # 相当于对于pyramid中的每个层次统计了特征
        code = [input_vector_encoder(crop, codebook) for crop in pyramid]
        # 不同的尺度下的match应赋予不同权重，显然大尺度的权重小，而小尺度的权重大
        code_level_0 = 0.25 * np.asarray(code[0]).flatten()
        code_level_1 = 0.25 * np.asarray(code[1:5]).flatten()
        code_level_2 = 0.5 * np.asarray(code[5:]).flatten()
        # 将这三个特征拼接在一起作为特征返回
        return np.concatenate((code_level_0, code_level_1, code_level_2))

# 码本数量100
VOC_SIZE = 100
# 金字塔等级 1
PYRAMID_LEVEL = 1

DSIFT_STEP_SIZE = 4
# DSIFT_STEP_SIZE is related to the function
# extract_DenseSift_descriptors in utils.py
# and build_spatial_pyramid in spm.py


if __name__ == '__main__':
    # 引入数据
    x_train, y_train = load_cifar10_data(dataset='train')
    x_test, y_test = load_cifar10_data(dataset='test')
    
    # 密集SIFT特征提取
    print( "Dense SIFT feature extraction" )
    # 对于x_train中的每一副图片，提取密集SIFT特征
    x_train_feature = [extract_DenseSift_descriptors(img) for img in x_train]
    x_test_feature = [extract_DenseSift_descriptors(img) for img in x_test]
    # 函数在调用多个参数时，在列表、元组、集合、字典及其他可迭代对象作为实参，并在前面加 * ，因为x_train_feature本身就是一个两个元素的元祖
    # 解包成关键点和描述符
    x_train_kp, x_train_des = zip(*x_train_feature)
    x_test_kp, x_test_des = zip(*x_test_feature)
    
    # 训练测试集划分
    print( "Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))
    # 码本大小
    print( "Codebook Size: {:d}".format(VOC_SIZE))
    # 金字塔等级
    print( "Pyramid level: {:d}".format(PYRAMID_LEVEL))
    # 构建码本
    print( "Building the codebook, it will take some time")
    # 使用x_train的描述符以及码本大小来构建码本
    codebook = build_codebook(x_train_des, VOC_SIZE)
    # 存储起来
    import cPickle
    with open('./spm_lv1_codebook.pkl','w') as f:
        cPickle.dump(codebook, f)

    # SPM编码开始
    print( "Spatial Pyramid Matching encoding")
    # 将每个图像使用SPM函数进行编码，输入x_train，原数据,x_train_des，描述符，码本，层级
    x_train = [spatial_pyramid_matching(x_train[i],
                                        x_train_des[i],
                                        codebook,
                                        level=PYRAMID_LEVEL)
                                        for i in xrange(len(x_train))]

    x_test = [spatial_pyramid_matching(x_test[i],
                                       x_test_des[i],
                                       codebook,
                                       level=PYRAMID_LEVEL) for i in xrange(len(x_test))]

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    # 再使用svm进行分类
    svm_classifier(x_train, y_train, x_test, y_test)
