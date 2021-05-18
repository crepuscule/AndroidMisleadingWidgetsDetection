def getLLC(SX,codeBookPath):
    #SX = convert2CV2(SX)
    # 码本数量100
    VOC_SIZE = 50
    # 金字塔等级 1
    @PYRAMID_LEVEL = 1
    pyramid = [1, 2]
    knn = 5
    img_width=224
    img_height=224
    #DSIFT_STEP_SIZE = 4
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
        # 使用x_train的描述符以及码本大小来构建码本
        codebook = utils.build_codebook(x_train_des, VOC_SIZE)
        # 存储起来
        import pickle
        write_file = open(codeBookPath, 'wb')
        pickle.dump(codebook, write_file)
        write_file.close()
    
    print( "LLC encoding")
    # llc编码
    fea = LLC_pooling(codebook,x_train_des[i],pyramid,knn,img_width,img_height,X1,Y)
    # SPM编码开始
    
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
