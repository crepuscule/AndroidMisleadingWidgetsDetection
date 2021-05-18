def movePictureToTrash(sourcepath,destpath):
    import os,shutil
    import sys
    from PIL import Image
    sourcename = sourcepath.split('/')[-1]
    sourcedir = sourcepath.split('/')[-2]
    destdir = destpath + sourcedir 
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    shutil.move(sourcepath,destdir)

#---------------------------- 1. check size--------------------------------
def checkSize(pictureLists,trashPath,SIZE):
    import os,shutil
    import sys
    from PIL import Image
    for picture in pictureLists:
        picture = self.config['PICTURES_DIR'] + picture
        # load in the top image
        top_img = Image.open(picture, 'r')
        top_img_w, top_img_h = top_img.size
        if maxw < top_img_w: maxw = top_img_w
        if maxh < top_img_h: maxh = top_img_h
        count += 1
        if (top_img_w > SIZE) or (top_img_h > SIZE):
            movePictureToTrash(picture,trashPath+'size/')
            print('removed: ',picture,' w,h: ',top_img_w,' * ',top_img_h)
    print('max width:',maxw,'\nmax hight:',maxh,'\ntotal numbers:',count) 
    
# --------------------------- 2. move simmilar image-------------------------
def getGray(image_file):
   tmpls=[]
   for h in range(0,  image_file.size[1]):#h
      for w in range(0, image_file.size[0]):#w
         tmpls.append( image_file.getpixel((w,h))  )
   return tmpls
def getAvg(ls):#获取平均灰度值
   return sum(ls)/len(ls)
def getMH(a,b):#比较100个字符有几个字符相同
   dist = 0;
   for i in range(0,len(a)):
      if a[i]==b[i]:
         dist=dist+1
   return dist
def getImgHash(fne):
   image_file = Image.open(fne) # 打开
   image_file=image_file.resize((12, 12))#重置图片大小我12px X 12px
   image_file=image_file.convert("L")#转256灰度图
   Grayls=getGray(image_file)#灰度集合
   avg=getAvg(Grayls)#灰度平均值
   bitls=''#接收获取0或1
   #除去变宽1px遍历像素
   for h in range(1,  image_file.size[1]-1):#h
      for w in range(1, image_file.size[0]-1):#w
         if image_file.getpixel((w,h))>=avg:#像素的值比较平均值 大于记为1 小于记为0
            bitls=bitls+'1'
         else:
            bitls=bitls+'0'
   return bitls
  # m2 = hashlib.md5()  
  # m2.update(bitls)
  # print m2.hexdigest(),bitls
  # return m2.hexdigest()
def deleteSimilar(folder,trashPath,COMPARE):
    picturePath = folder
    outterdirs = getPictureLists(picturePath)
    #外部dirs可能因为remove而失去，所以要判断
    for judgeImage in outterdirs:
        if os.path.exists(judgeImage):
            a=getImgHash(judgeImage)#原点图片
        else:
            continue
        #每次完成后再次更新，这样肯定不会出错
        innerdirs = getPictureLists(picturePath)
        for otherImage in innerdirs:
            if otherImage == judgeImage:
                continue
            b=getImgHash(otherImage)#被审图片
            compare=getMH(a,b)
            # 这里定义相似度为多少时给予删除
            if (compare >= COMPARE) and (otherImage != judgeImage):
                print('remove ',otherImage,u'相似度',str(compare)+'%')
                movePictureToTrash(otherImage,trashPath+'similar/')
'''
def runPreprocess(operator=[],params=[]):
    if operator == 'info' or operator == '':
        self.info()    
    # only choose needed columns , store to SIMPLIFIED_SET_PATH
    if operator == 'sim':
        print('It takes much time...')
        simplifiedRecords = self.simplify(self.config['UNIVERSAL_RECORDS_PATH'])
        self.writeCSV(simplifiedSet,self.config['SIMPLIFIED_RECORDS_PATH'])
        print(len(simplifiedRecords),' records.')
    # choose record in picutreLists, store to MEANINGFUL_RECORDS_PATH
    if operator == 'ext':
        print('It takes much time...')
        if ',' not in params: params+=','
        setName = params.split(',')[0]
        num = params.split(',')[-1]
        
        #?# 这里注意需要先有meaningful集合才能进行dev等，但是没有这个约束
        current_records_path = self.config['MEANINGFUL_RECORDS_PATH']
        if setName == 'dev':
            self.config['CURRENT_SET_PATH'] = self.config['DEV_SET_PATH']
        elif setName == 'train':
            self.config['CURRENT_SET_PATH'] = self.config['TRAIN_SET_PATH']
        elif setName == 'val':
            self.config['CURRENT_SET_PATH'] = self.config['VAL_SET_PATH']
        elif setName == 'test':
            self.config['CURRENT_SET_PATH'] = self.config['TEST_SET_PATH']
        else:
            self.config['CURRENT_SET_PATH'] = self.config['MEANINGFUL_RECORDS_PATH']
            current_records_path = self.config['SIMPLIFIED_RECORDS_PATH']

        pictureLists,pathLists = self.getPictureLists(self.config['PICTURES_DIR'])            
        # 如果已经有有意义集合了，何不直接用呢？如果mode为lazy则在次函数中直接从meanful中抽取
        # 如果mode为normal，则从simplifed慢慢重新构建,注意如果构建全集，则模式必定为normal
        extractedRecords = self.partitionDataset(pictureLists,pathLists,current_records_path,num)
        #extractedRecords = self.extractSimCsvWithPictureLists(pictureLists,pathLists,self.config['SIMPLIFIED_RECORDS_PATH'])
        # 写入到当前操作数据集位置中
        self.writeCSV(extractedRecords,self.config['CURRENT_SET_PATH'])
        print(len(extractedRecords),' records.')

if __name__ == "__main__":
    SIZE = 224
    COMPARE = 99
    backgroundR = 255
    backgroundG = 255
    backgroundB = 255
    folder = '/data/DroidBot_Epoch/raw_data/zips/try_input_data/'
    trashPath = '/data/DroidBot_Epoch/raw_data/zips/try_input_data_trash'
    operator = 'all'
    if operator == 'all':
        _,pictureLists = self.getPictureLists(folder)
        self.moveBad(pictureLists,trashPath)
        pictureLists = self.getPictureLists(folder)
        self.checkSizeandAddBackGround(pictureLists,trashPath,SIZE,(backgroundR,backgroundG,backgroundB))
    if operator == 'bkg':
        _,pictureLists = self.getPictureLists(folder)
        self.checkSizeandAddBackGround(pictureLists,trashPath,SIZE,(backgroundR,backgroundG,backgroundB))
    if operator == 'mvb':
        _,pictureLists = self.getPictureLists(folder)
        self.moveBad(pictureLists,trashPath)
