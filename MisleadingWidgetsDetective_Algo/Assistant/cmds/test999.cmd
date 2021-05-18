# 尝试使用1k的实验集深度探究聚类结果
# 类型		Test_(共Formal_,Test_,Onetime_,Inter_?)
# 数据集	test999
# 图片特征	SPMCodeBook-l2-c1000
# 降维		none,kpca1000,kpca500,kpca100
# 聚类		dbscan+sklearn(固定kpc500)=>具体是按照config中来的，这里面只需要写清楚即可，或者notknown
# 异常检测	iforest
# 可视化	二维平面可视化

######################### DP
# 一般都是提前准备好,Inter_可以先写好再进行DP，完了以后二次打开项目

A o Onetime_test999.spm_kpca500_notknown_iforest

######################### DT
DT run

######################### IF
IF extif spm

######################### DE
DE vector

DE cutdim kpca,500
######################### VC
VC run
#VC eval

######################### OD
OD run
#OD eval

