import numpy as np
from tensorflow import keras
from numpy import genfromtxt
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score,accuracy_score
import time
import os
from Process import data_angle
mean = genfromtxt('Dataset/xmean.csv', delimiter=',')
testdata = genfromtxt('Dataset/x.csv', delimiter=',')
model = keras.models.load_model('model_v8.h5')
transformer = Normalizer().fit(testdata)
def ComparatorNet(x_new,id):
    x_new = np.array(x_new)
    x = np.concatenate((mean[id-1], x_new)).reshape(1, 114)
    x = transformer.transform(x)
    y_pred = model.predict(x)
    return y_pred[0][0]
def ComBinePred(img,id):
    newx = data_angle(img)
    rs = ComparatorNet(newx,id)
    if(rs > 0.5):
        return id
    else:
        return 0
