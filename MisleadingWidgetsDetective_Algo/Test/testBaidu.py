import time
def checkOCRByBaiduAPI(image_path):                                                                                                
    import requests                                                                                                                     
    import base64                                                                                                                       
    import sys                                                                                                                          
                                                                                                                                        
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"                                                              
    # 二进制方式打开图片文件                                                                                                            
    f = open(image_path, 'rb')                                                                                                          
    img = base64.b64encode(f.read())                                                                                                    
    params = {"image":img,"language_type":'ENG',"probability":"true"}                                                                   
    access_token = '24.b49cc7014edccd03e39ea6f1b84a3b61.2592000.1605754922.282335-22846592'                                             
    request_url = request_url + "?access_token=" + access_token                                                                         
    headers = {'content-type': 'application/x-www-form-urlencoded'}                                                                     
    response = requests.post(request_url, data=params, headers=headers)                                                                 
    #{'words_result': [{'probability': {'average': 0.720593, 'min': 0.720593, 'variance': 0.0}, 'words': 'P'}, {'probability': {'average': 0.946776, 'min': 0.946776, 'variance': 0.0}, 'words': 'Parkins'}], 'log_id': 1318464131694592000, 'words_result_num': 2}
    characters = response.json()                                                                                                        
    print('百度api识别:',characters)                                                                                                    
    if characters['words_result_num'] == 0:                                                                                             
        return 'noText'                                                                                                                 
    else:                                                                                                                               
        noText = False                                                                                                                  
        hasChar = False                                                                                                                 
        for character in characters['words_result']:                                                                                    
            if character['probability']['min'] > 0.8:                                                                                   
                return 'hasText'                                                                                                        
            elif character['probability']['min'] < 0.7:                                                                                 
                noText = True                                                                                                           
            else:                                                                                                                       
                hasChar = True                                                                                                          
        if hasChar:                                                                                                                     
            return 'hasChar'                                                                                                            
        else:                                                                                                                           
            return 'noText' 

print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print('5s后:')
time.sleep(5)
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print('5s后:')
time.sleep(5)
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
print( checkOCRByBaiduAPI('/data/DroidBot_Epoch/raw_data/input_data/name.gdr.acastus_photon_18.apk/images/view_64fe1a775a72cb1026e7097caed3dee9.png') )
