def vector(apis,api_freqs):                                                                                                                          
    apiSet = set()
    for line in apis:
        apiSet.update(line)                                                                                                                               
    
    apiSet = list(apiSet)                                                                                                                                 
    apiSet.sort()
    print('=>',apiSet)
    apiSetLen = len(apiSet)
    apiNo = [i for i in range(len(apiSet))]                                                                                                               
    #apiDicts => 'api1':1,'api2':2
    apiDicts = dict(zip(apiSet,apiNo))                                                                                                                    
    print('apiSet len',apiSetLen)                                                                                                                         
    
    widgetAPIVectors = []
    # 每一行就是一个控件，所以用i就可以定位到相应的api和freq                                                                                              
    for i in range(len(apis)):
        widgetAPIVector = [0]*apiSetLen                                                                                                                   
        # 仅用于计数
        # 这个api就是每个控件中的api的位置                                                                                                                
        for j in range(len(apis[i])):                                                                                                                     
            api = apis[i][j]                                                                                                                              
            # one-hot 编码
            # 在api所在位置上(即apiDicts的值),插入1                                                                                                       
            # 现在则插入对应控件对应的其tf*idf值??
            widgetAPIVector[apiDicts[api]] = api_freqs[i][j]                                                                                              
        # 一行检查完毕之后写入
        widgetAPIVectors.append(widgetAPIVector)                                                                                                          
    # 顺序与之前完全相同
    return widgetAPIVectors


apis = [[ 'com.android.internal.view.menu.MenuItemImpl.getSubMenu ()Landroid/view/SubMenu;', 'com.android.internal.view.menu.MenuItemImpl.hasSubMenu ()Z', 'java.lang.Double.isNaN (D)Z', 'java.lang.Math.abs (I)I', 'java.lang.Math.round (F)I'],
        ['android.widget.ViewAnimator.setDisplayedChild (I)V', 'java.lang.Object.equals (Ljava/lang/Object;)Z'], 
        ['dalvik.system.CloseGuard.close ()V', 'libcore.io.IoUtils.closeQuietly (Ljava/io/FileDescriptor;)V'], 
        ['dalvik.system.CloseGuard.close ()V', 'libcore.io.IoUtils.closeQuietly (Ljava/io/FileDescriptor;)V'], 
        ['dalvik.system.CloseGuard.close ()V', 'libcore.io.IoUtils.closeQuietly (Ljava/io/FileDescriptor;)V']]
api_freqs = [[0.6,1.3,0.0,0.1,0.1],[0.3,0.7],[0.1,0.9],[0.21,1],[0.0,0.35]]
print(vector(apis,api_freqs))
