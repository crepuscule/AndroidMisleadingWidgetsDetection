# 首先检查参数是否存在，存在就直接拿来使用
# 不存在再继续训练
from . import AutoEncoderTrain
from importlib import reload
reload(AutoEncoderTrain)

def runExtract(X_train):
    AutoEncoderTrain.runTrain(X_train)
