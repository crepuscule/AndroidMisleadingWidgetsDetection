# Android Widget Clustering Project Configuration
# new standard
def initConfig(instanceName='',no=1):
    config1 = {
    # ================================基本设置 ================================
    # 实例名称
    'INSTANCE_NAME':'MisleadingWidgetProject',
    # 脚本所在地
    'SCRIPT_ROOT':'/core/kernel/01work/02project/system/AndroidUIUnderstanding/03WidgetClustering/process/androidwidgetclustering/',
    # 源数据根目录
    #'RAW_ROOT':'/data/NewStaticWidgetData/raw_data/',
    'RAW_ROOT':'/data/DroidBot_Epoch/raw_data/',
    # 生成数据根目录
    'GENERATED_ROOT':'/data/DroidBot_Epoch/generated_data/',
    
    # ==========================raw data设置[将在前面增加RAW_ROOT前缀]========================
    # 全集记录地址(1/2)
    # 新记录的存储地点
    'INPUT_DATA_DIR':'input_data/',
    #'UNIVERSAL_RECORDS_PATH':'record_data/universal.csv',
    'UNIVERSAL_RECORDS_DIR':'record_data/',
    # 图像数据地址(必须以/结尾)
    'PICTURES_DIR':'image_data/',
    # 图像预处理后丢弃的图像数据地址
    'PICTURES_TRASH_DIR':'trash_data/',
    # API白名单，避免第三方API混入
    'WHITE_LIST_PATH':'dimension_data/androidPlatformAPI226',#TTT
    # SPM训练的码本
    'SPM_CODE_BOOK_PATH':'pretrain_data/SPMCodeBook-l1-c500',
    # 根据idf选中的维度map地址
    'CHOICEN_IDF_MAP_PATH':'', 

    # ==========================generated_data设置[将在前面增加GENERATED_ROOT前缀]==================
    # DataPreprocess ------------------------------------          #KKKKKK直接保存到apktree db

    # DataTransform ------------------------------------
    
    # FeatureExtract -------------------------------
    'EXTRACTED_FEATURE_PATH':'ImageFeatureExtract/ExtractedFeature', #multi-supported 必须保存在本地，是pro之一,特征之一

    # DimensionEngineering-----------------------------
    # api0-1向量地址
    'API_VECTORS_PATH':'DimensionEngineering/APIVector', #multi-supported 必须保存在本地，是pro之一，特征之一

    # VectorClustering--------------------------------
    # 聚类结果
    'CLUSTERER_CONTAINER_PATH':'VectorClustering/clusterer_container',#???
    'CLUSTER_PICTURE_RESULT_DIR':'VectorClustering/picutures/', #暂时保留作为命令行版本展示
    # 暂时停用
    'CLUSTER_COMMAND_DIR':'VectorClustering/config/',#TTT可能转到项目元数据 或者不转也挺好，放在本地
    
    # OutlierDetect-----------------------------------
    'OUTLIER_PICTURE_RESULT_DIR':'OutlierDetect/picutures/',#暂时保留作为命令行版本展示
    }
    
    config2 = {
    # ================================基本设置 ================================
    # 实例名称
    'INSTANCE_NAME':'MisleadingWidget-kpca',
    # 脚本所在地
    'SCRIPT_ROOT':'/core/kernel/01work/02project/system/AndroidUIUnderstanding/03WidgetClustering/process/androidwidgetclustering/',
    # 源数据根目录
    'RAW_ROOT':'/data/NewStaticWidgetData/raw_data/',
    # 生成数据根目录
    'GENERATED_ROOT':'/data/NewStaticWidgetData/generated_data/',
    
    # ==========================raw data设置[将在前面增加RAW_ROOT前缀]========================
    # 全集记录地址(1/2)
    #'UNIVERSAL_RECORDS_PATH':'record_data/universal.csv',
    'UNIVERSAL_RECORDS_DIR':'record_data/',
    # 图像数据地址(必须以/结尾)
    'PICTURES_DIR':'image_data/',
    # 图像预处理后丢弃的图像数据地址
    'PICTURES_TRASH_DIR':'trash_data/',
    # API白名单，避免第三方API混入
    'WHITE_LIST_PATH':'dimension_data/androidPlatformAPI226.txt',
    # SPM训练的码本
    'SPM_CODE_BOOK_PATH':'pretrain_data/SPMCodeBook-l1-c500.pkl',

    # ==========================generated_data设置[将在前面增加GENERATED_ROOT前缀]==================
    # DataPreprocess ------------------------------------
    # 后期将RGB等配置也写在这里，搞一个纯配置和数据库文件配置
    # 简化全集记录地址
    'SIMPLIFIED_RECORDS_PATH':'FeatureExtract/simplified.csv',
    # 有图像对应的有意义记录地址16266
    'MEANINGFUL_RECORDS_PATH':'FeatureExtract/meaningful.csv',
    'CURRENT_SET_PATH':'',
    'TRAIN_SET_PATH':'FeatureExtract/train_set.csv',
    'VAL_SET_PATH':'FeatureExtract/val_set.csv',
    'TEST_SET_PATH':'FeatureExtract/test_set.csv',
    'DEV_SET_PATH':'FeatureExtract/dev_set.csv',

    # DataTransform ------------------------------------
    # apktree地址
    'APK_TREE_PATH':'DataTransform/APKTree.json',
    ### apkpathtree地址
    ##'APK_PATH_TREE_PATH':'DataTransform/APKPathTree.json',
    # 根据idf选中的维度map地址
    'CHOICEN_IDF_MAP_PATH':'',
    
    # FeatureExtract -------------------------------
    'EXTRACTED_FEATURE_PATH':'FeatureExtract/ExtractedFeature.txt',

    # DimensionEngineering-----------------------------
    # api0-1向量地址
    'API_VECTORS_PATH':'DimensionEngineering/APIVector.txt',
    # api0-1向量列名
    'API_VECTORS_COLUMN_PATH':'DimensionEngineering/APIVectorColumn.txt',
    # api0-1向量行名
    'API_VECTORS_NAME_PATH':'DimensionEngineering/APIVectorName.txt',  

    # VectorClustering--------------------------------
    # 聚类结果
    'CLUSTER_RESULT_PATH':'VectorClustering/clusterResult.txt',
    'CLUSTERER_CONTAINER_PATH':'VectorClustering/clusterer_container.pkl',
    'CLUSTER_PICTURE_RESULT_DIR':'VectorClustering/picutures/',
    # 暂时停用
    'CLUSTER_COMMAND_DIR':'VectorClustering/config/',

    
    # OutlierDetect-----------------------------------
    'OUTILER_VECTOR_PATH':'OutlierDetect/outlierVector.txt',
    'OUTLIER_PICTURE_RESULT_DIR':'OutlierDetect/picutures/',
    }
    
    if no == 1:
        config = config1
    elif no == 2:
        config = config2
    else:
        config = config1

    if instanceName != '': 
        config['INSTANCE_NAME'] = instanceName

    for key,value in config.items():
        # 配置为空的不改
        if value == '':
            continue
        # 基本配置不改
        elif key == 'SCRIPT_ROOT' or key == 'RAW_ROOT' or key == 'GENERATED_ROOT' or key == 'INSTANCE_NAME':
            continue
        # raw 数据，添加raw root
        elif key == 'INPUT_DATA_DIR' or key == 'UNIVERSAL_RECORDS_PATH' or key == 'UNIVERSAL_RECORDS_DIR' or key == 'WHITE_LIST_PATH' or key == 'PICTURES_DIR' or key == 'PICTURES_TRASH_DIR' or key == 'SPM_CODE_BOOK_PATH':
            config[key] = config['RAW_ROOT'] + value
        # generated 数据，添加generated root
        else:
            config[key] = config['GENERATED_ROOT'] + config['INSTANCE_NAME'] + '/' + value

    return config
