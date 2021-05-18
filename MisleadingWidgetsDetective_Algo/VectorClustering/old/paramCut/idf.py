# 读取矩阵，以序号为api名称
# 计算各个api的idf
# 作出箱形图，看看阈值
def initConfig():
    config = {
    # Datas ---------
    # 8701313
    'UNIVERSAL_RECORDS_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/raw_data/record_data/origin/traditional/benign_pa.csv',
    'SIMPLIFIED_RECORDS_PATH':'',
    'PICTURES_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/raw_data/image_data/',
    # 16266
    'MEANINGFUL_RECORDS_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/raw_data/record_data/origin/benign_pa_meaningful.csv',
    'TEMP_MEANINGFUL_RECORDS_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/raw_data/record_data/origin/benign_pa_unisim_static.csv',


    #Transform
    'APK_TREE_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/intermediate_data/VectorClustering/apktree/apkTree.json',
    'APK_PATH_TREE_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/intermediate_data/VectorClustering/apktree/apkPathTree.json',

    #VectorCluster
    'API_VECTORS_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/intermediate_data/VectorClustering/vectors/apiVector.txt',
    'API_VECTORS_COLUMN_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/intermediate_data/VectorClustering/vectors/apiVectorColumn.txt',
    'API_VECTORS_NAME_PATH':'/data/wangruifeng/datasets/WidgetClustering/staticWidgetResources/intermediate_data/VectorClustering/vectors/apiVectoName.txt',
    }
    return config
    
#API_VECTORS_PATH
def readVectors(savePath):
    import numpy
    widgetAPIVectors = numpy.loadtxt(savePath,dtype=numpy.int32)
    return widgetAPIVectors
    
def calcIDF(apiVector):
    import numpy as np
    frequent = apiVector.sum(axis = 0)
    print(frequent)
    idfs = np.log10(1+ apiVector.shape[0]/frequent)    
    return idfs

def readAPIColumn(apiVectorsColumnPath):
    file_handle = open(apiVectorsColumnPath,mode='r')
    apiColumns=file_handle.readlines()
    apiColumns=[x.strip() for x in apiColumns]
    return apiColumns

def chooseAbove(idfs,apiColumns,aboveValue=0,aboveRatio=0):
    import numpy as np
    if aboveRatio<0:
        aboveValue = -idfs[ int(-aboveRatio * len(idfs))]
    if aboveRatio > 0:
        aboveValue = idfs[ int(aboveRatio * len(idfs))]
        #print(aboveValue)
    if aboveValue <0:        
        indexs = np.argwhere(idfs <= -aboveValue)
    elif aboveValue > 0 :        
        indexs = np.argwhere(idfs > aboveValue)
    else:        
        indexs = np.array([i for i in range(0,len(idfs))])
        return indexs,np.array(idfs),np.array(apiColumns)    
    indexs = indexs.flatten()
    return indexs,np.array(idfs)[indexs],np.array(apiColumns)[indexs]

def show(config):
    X = readVectors(config['API_VECTORS_PATH'])
    print(X.shape)
    idfs = calcIDF(X)
    print(idfs.shape)
    #print(idfs)

    apiColumns = readAPIColumn(config['API_VECTORS_COLUMN_PATH'])
    #print(apiNames)
    import numpy as  np

    indexs,partIdfs,partAPIColumns = chooseAbove(idfs,apiColumns,0,0)

    show = np.dstack((partIdfs,partAPIColumns))
    show = list(show[0])
    show.sort(key=(lambda x:x[0]),reverse=True)
    #np.sort(show, axis=-1)
    #print(show)
    for i in range(0,len(partIdfs)):
        print('%.2f | %s' % (float(show[i][0]),show[i][1]))
    
def createNewVector(widgetAPIVectors,indexs,newSavePath):
    import numpy as np
    newAPIVector = np.array([widgetAPIVectors[0][indexs]])
    for i in range(1, len(widgetAPIVectors)): 
        newAPIVector = np.concatenate((newAPIVector, [widgetAPIVectors[i][indexs]]), axis = 0)
    
    np.savetxt(newSavePath,newAPIVector,fmt="%d") 
    return newAPIVector
    
if __name__=='__main__':
    config = initConfig()
    show(config)
    
    
# savePath = '/storage/staticResourceImages/beforeclustering/oldversion/total_benign_pa_vectors.txt'
# dataSourcePath = '/storage/staticResourceImages/beforeclustering/oldversion/total_benign_pa_dicts.pth'
#savePath = '/storage/staticResourceImages/beforeclustering/benign_pa_vectors.txt'
#dataSourcePath = '/storage/staticResourceImages/beforeclustering/benign_pa_dicts.pth'
#saveAPINamePath = '/storage/staticResourceImages/beforeclustering/benign_pa_vectornames.txt'
#newSavePath='/storage/staticResourceImages/beforeclustering/benign_pa_vectors_idf.txt'




#newAPIVector = createNewVector(X,indexs,newSavePath)
#print(newAPIVector.shape)
