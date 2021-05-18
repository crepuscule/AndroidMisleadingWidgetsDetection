#clusterer_name: hierarch,kmeans,dbscan,optics

def getsearchParams():
    searchparam = {
        "clusterer_name":"hierarch",
        "ranges":[[100, 8000, 1000]],
        "data":"",
        "param_scale":[1],
        "metricstring":"",
        "filterFun":"",
        "verbose":""
    }
    '''
    searchparam = {
        "clusterer_name":"dbscan",
        "ranges":[[1, 100, 5],[1,100,5]],
        "data":"",
        "param_scale":[10,1],
        "metricstring":"",
        "filterFun":"",
        "verbose":""
    }
    '''
    return searchparam

def getRunParams():
    runparam = {
        "clusterer_name":"optics",
        "params":[3],
        "data":"",
        "param_scale":[1],
        "metricstring":""
    }
    return runparam
'''
def getsearchParams():
    searchparam = {
        "clusterer_name":"dbscan",
        "ranges":[[1, 50, 5], [1, 30, 5]],
        "data":"",
        "param_scale":[10, 1],
        "metricstring":"",
        "filterFun":"",
        "verbose":""
    }
    return searchparam

def getRunParams():
    runparam = {
        "clusterer_name":"dbscan",
        "params":[71, 2],
        "data":"",
        "param_scale":[100, 1],
        "metricstring":""
    }
    return runparam

'''
'''
{'info': {'clusterer_name': 'dbscan',  'params': [2.1,  1.0],  'metricstring': 'sc',  'XSize': (781,  781),  'Xdtype': 'int32',  'ySize': (781, ),  'ydtype': 'float64',  'size': (224,  224,  3)},  'data': {'clusterer_name': 'dbscan',  'params': [2.1,  1.0],  'metricstring': 'sc',  'XSize': (781,  781),  'Xdtype': 'int32',  'ySize': (781, ),  'ydtype': 'float64',  'size': (224,  224,  3)},  'performance': {'time': 0.4232370853424072,  'clusters_num': 133,  'cluster_elements_percent': array([0.        ,  0.48911652,  0.00512164,  0.00256082,  0.00256082,
           0.00768246,  0.00768246,  0.00128041,  0.00256082,  0.00384123]),  'silhouette': 0.7592103666448573,  'calinski': 2424.7517288738068}} 
'''

