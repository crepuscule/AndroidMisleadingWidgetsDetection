def TF_IDF(apk_widget_forest):
    import math
    # apk forest中每个控件中都有对应的API列表和频率
    # 遍历每个API列表，加入到字典中，统计每个API在所有文档中出现的频率

    # ================计算IDF=============
    # 总文档数|D|
    widget_num = len(apk_widget_forest)
    # 包含每个词的文档总数
    all_api_dict = dict()

    # 对于每一个控件，如果该控件所包含的某个api在all_api_dict中有，则为其加1，表示该api的总文档+1
    # 如果没有，则令其=1，目前有1个文档有次api
    for widget in apk_widget_forest:
        # 每个api考察一遍，因为每个都不一样
        for api in widget['api']:
            if api in all_api_dict:
                all_api_dict[api] += 1
            else:
                all_api_dict[api] = 1
    for key in all_api_dict.keys():
        # widget_num => |D| 
        # 1 + all_api_dict[i] => 1+|j: t_i in d_j|  包含词t_i的文档数量
        # 这里的log，底数为e
        all_api_dict[key] = math.log(( 1 + widget_num )/(1 + all_api_dict[key])) + 1

    # ===============计算TF * IDF =============
    for widget in apk_widget_forest:
        for i in range(len(widget['api'])):
            api = widget['api'][i]
            print("widget['api_freq'][i] * all_api_dict[api]:\n%f*\t%f" % (widget['api_freq'][i], all_api_dict[api]))
            widget['api_freq'][i] = widget['api_freq'][i] * all_api_dict[api]
            print("\n=%f" % (widget['api_freq'][i]))
    return apk_widget_forest

apk_widget_forest = [
{  "app" : "org.asdtm.goodweather_13.apk", "widget" : "2020-09-17_205803", "package" : "org.asdtm.goodweather", "path" : "org.asdtm.goodweather_13.apk/images/view_59c0433eba49fb297ffe2224f795fdfd.png", "raw_image_size" : "144*144", "text" : None, "api" : [ "android.util.Log.e (Ljava/lang/String;Ljava/lang/String;Ljava/lang/Throwable;)I", "java.lang.Boolean.booleanValue ()Z", "java.lang.Boolean.valueOf (Z)Ljava/lang/Boolean;", "java.lang.Double.parseDouble (Ljava/lang/String;)D", "java.lang.Double.valueOf (D)Ljava/lang/Double;", "java.lang.Object.<init> ()V", "java.lang.String.equals (Ljava/lang/Object;)Z", "java.lang.String.format (Ljava/lang/String;[Ljava/lang/Object;)Ljava/lang/String;", "java.lang.String.hashCode ()I", "java.lang.String.replace (Ljava/lang/CharSequence;Ljava/lang/CharSequence;)Ljava/lang/String;", "java.util.ArrayList.contains (Ljava/lang/Object;)Z", "java.util.Locale.getDefault ()Ljava/util/Locale;" ],"api_freq":[0.1,0.1,0.15,0.1,0.1,0.1,0.1,0.00,0.05,0.05,0.05,0.05,0.05] },
{  "app" : "org.asdtm.goodweather_13.apk", "widget" : "2020-09-17_210049", "package" : "org.asdtm.goodweather", "path" : "org.asdtm.goodweather_13.apk/images/view_e018c5ebedd7ffceb4654384caefed98.png", "raw_image_size" : "168*168", "text" : None, "api" : [ "android.app.Activity.startActivity (Landroid/content/Intent;)V", "android.app.SharedPreferencesImpl.getBoolean (Ljava/lang/String;Z)Z", "android.app.SharedPreferencesImpl.getFloat (Ljava/lang/String;F)F", "android.app.SharedPreferencesImpl.getLong (Ljava/lang/String;J)J", "android.app.SharedPreferencesImpl.getString (Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;", "android.content.Context.getString (I)Ljava/lang/String;", "android.content.ContextWrapper.getSharedPreferences (Ljava/lang/String;I)Landroid/content/SharedPreferences;", "android.content.Intent.<init> (Ljava/lang/String;)V"],"api_freq":[0.1,0.1,0.1,0.2,0.1,0.1,0.2,0.1] },
{  "app" : "org.asdtm.goodweather_13.apk", "widget" : "2020-09-17_205729", "package" : "org.asdtm.goodweather", "path" : "org.asdtm.goodweather_13.apk/images/view_b14e61618e50cf7e135556c6dcf50591.png", "raw_image_size" : "168*168", "text" : None, "api" : [ "android.support.v4.widget.DrawerLayout.e (I)V","android.app.SharedPreferencesImpl.getString (Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;" ] ,"api_freq":[0.65,0.35]},
{  "app" : "de.pinyto.exalteddicer_4.apk", "widget" : "2020-09-18_150255", "package" : "de.pinyto.exalteddicer", "path" : "de.pinyto.exalteddicer_4.apk/images/view_fda3121121c976f10d6f5b55ddf87ebd.png", "raw_image_size" : "101*240", "text" : "1", "api" : [ "android.util.FloatMath.sqrt (F)F" ,"android.util.Log.e (Ljava/lang/String;Ljava/lang/String;Ljava/lang/Throwable;)I"] ,"api_freq":[0.6,0.2,0.2]},
{  "app" : "de.pinyto.exalteddicer_4.apk", "widget" : "2020-09-18_150105", "package" : "de.pinyto.exalteddicer", "path" : "de.pinyto.exalteddicer_4.apk/images/view_4ac4fb94b91cb827a94fadba86501aa1.png", "raw_image_size" : "120*144", "text" : None, "api" : [ "android.util.FloatMath.sqrt (F)F" ],"api_freq":[1] }]
for item in (TF_IDF(apk_widget_forest)):
    print(item)
