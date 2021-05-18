import cv2
import sklearn
from sklearn.cluster import KMeans
import scipy.cluster.vq as vq
import numpy as np



# 这个就是关键点采集的像素步长，步长大，采集地越是粗略
DSIFT_STEP_SIZE = 4
#DSIFT_STEP_SIZE = 8


def load_cifar10_data(dataset):
    if dataset == 'train':
        with open('./cifar10/train/train.txt','r') as f:
            paths = f.readlines()
    if dataset == 'test':
        with open('./cifar10/test/test.txt','r') as f:
            paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        path, label = each.split(' ')
        img = cv2.imread(path)
        x.append(img)
        y.append(label)
    return [x, y]

def load_my_data(path, size,test=None):
    with open(path, 'r') as f:
        paths = f.readlines()
    x, y = [], []
    for each in paths:
        each = each.strip()
        label, path = each.split(' ')
        img = cv2.imread(path)
        if img.shape[:2] != (size,size):
            img = cv2.resize(img, (size,size))
        x.append(img)
        y.append(label)
    return [x, y]

def extract_sift_descriptors(img):
    """
    Input BGR numpy array
    Return SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# 提取密集SIFT特征描述子
def extract_DenseSift_descriptors(img):
    """
    Input BGR numpy array
    Return Dense SIFT descriptors for an image
    Return None if there's no descriptor detected
    """
    # 转化为灰度图
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    # opencv3.4.3.16 版本后，这个功能就不能用
    # https://blog.csdn.net/weixin_43772533/article/details/103242845
    sift = cv2.xfeatures2d.SIFT_create()

    # opencv docs DenseFeatureDetector
    # opencv 2.x code
    #dense.setInt('initXyStep',8) # each step is 8 pixel
    #dense.setInt('initImgBound',8)
    #dense.setInt('initFeatureScale',16) # each grid is 16*16
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, gray.shape[0], disft_step_size)
                for x in range(0, gray.shape[1], disft_step_size)]

    keypoints, descriptors = sift.compute(gray, keypoints)

    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    # 最后返回关键点和描述子
    # kps是关键点。它所包含的信息有：
    #angle：角度，表示关键点的方向，为了保证方向不变形，SIFT算法通过对关键点周围邻域进行梯度运算，求得该点方向。-1为初值。
    #class_id：当要对图片进行分类时，我们可以用class_id对每个特征点进行区分，未设定时为-1，需要靠自己设定
    #octave：代表是从金字塔哪一层提取的得到的数据。
    #pt：关键点点的坐标
    #response：响应程度，代表该点强壮大小，更确切的说，是该点角点的程度。
    #size：该点直径的大小
    return [keypoints, descriptors]

# 构建码本，使用描述符和码本大小
def build_codebook(X, voc_size):
    """
    Inupt a list of feature descriptors
    voc_size is the "K" in K-means, k is also called vocabulary size
    Return the codebook/dictionary
    码本大小就是聚类数目
    """
    # 将X中的描述符堆叠成列向量 np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    features = np.vstack((descriptor for descriptor in X))
    kmeans = KMeans(n_clusters=voc_size, n_jobs=-2)
    kmeans.fit(features)
    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    # array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]]) >>> np.squeeze(a) >>> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    codebook = kmeans.cluster_centers_.squeeze()
    # 相当于最后返回的是一个个的聚类中心
    return codebook

# 输入向量编码，将特征对应码本进行编码
def input_vector_encoder(feature, codebook):
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    # 根据聚类中心将所有数据进行分类.obs为数据,code_book则是kmeans产生的聚类中心. 
    # 输出同样有两个:第一个是各个数据属于哪一类的label,第二个和kmeans的第二个输出是一样的,都是distortion
    code, _ = vq.vq(feature, codebook)
    # np.histogram()是一个生成直方图的函数
    # np.histogram() 默认地使用bin=10个相同大小的区间（箱），然后返回一个元组（频数，分箱的边界），如上所示。要注意的是：这个边界的数量是要比分箱数多一个的，因为边界永远会比区域多一个值。可以简单通过下面代码证实。
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    # 只返回频数了，因为桶是一样的
    return word_hist
