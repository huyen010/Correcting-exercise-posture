import pickle
import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import time
from Process import data_angle
from statistics import mean
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from ComparatorNet_Predict import ComBinePred
import Process
import pymysql
from PIL import Image
import shutil
connection = pymysql.connect(host="localhost", port=3306, user="root", passwd="", database="pose_database")
cursor = connection.cursor()
dataset_path = 'Dataset/yogaTrain1.csv'
data_train = pd.read_csv(dataset_path)
data_label_train = data_train["id"]
droplist = ['img','id']
data_features_train = data_train.drop(droplist,axis=1)
dataset_path2 = 'Dataset/yogaTest1.csv'
data_test = pd.read_csv(dataset_path2)
data_label_test = data_test["id"]
data_features_test = data_test.drop(droplist,axis=1)
model = pickle.load(open("trained_mode.pickle", "rb"))
listlabel = []
listlabel.append('No name')
for folder_name in os.listdir('Data/TEST'):
    listlabel.append(folder_name)
def train_model():
    data_features_minmax_train = (data_features_train - data_features_train.min()) / (
                data_features_train.max() - data_features_train.min())
    data_features_minmax_test = (data_features_test - data_features_train.min()) / (
                data_features_train.max() - data_features_train.min())
    rfc = RandomForestClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)
     # define search space
    space = dict()
    # so luong cay toi da
    space['bootstrap'] = [True, False]
    # space['n_estimators'] = [100, 200, 500, 1000, 5000]
    # space['max_features'] = []
    search = GridSearchCV(rfc, space, n_jobs=4, cv=cv)
    result = search.fit(data_features_minmax_train, data_label_train)
    print(result.best_score_)
    print('Best n_estimator:', result.best_estimator_.get_params()['n_estimators'])
    print('Best max_features:', result.best_estimator_.get_params()['max_features'])
    print('Best max_depth:', result.best_estimator_.get_params()['max_depth'])
    print('Best min_samples_split:', result.best_estimator_.get_params()['min_samples_split'])
    print('Best min_samples_leaf:', result.best_estimator_.get_params()['min_samples_leaf'])
    print('Best bootstrap:', result.best_estimator_.get_params()['bootstrap'])
    print(result.score(data_features_minmax_test, data_label_test))
    y_pred = result.predict(data_features_minmax_test)
    # print(y_pred)
    # pickle.dump(result, open("trained_mode.pickle", 'wb'))
    # cm = confusion_matrix(data_label_test, y_pred)
    # sns.heatmap(cm, annot=True, fmt="d")
def predict(image):
    listXYZ = Process.data_angle(image)
    i = 0
    for col in data_features_train.columns:
       listXYZ[i] = (listXYZ[i] - data_features_train.min()[col]) / (data_features_train.max()[col] - data_features_train.min()[col])
       i = i + 1
    new_row = {'x1': listXYZ[0], 'y1': listXYZ[1], 'z1': listXYZ[2], 'x2': listXYZ[3], 'y2': listXYZ[4], 'z2': listXYZ[5], 'x3': listXYZ[6], 'y3': listXYZ[7], 'z3': listXYZ[8],
               'x4': listXYZ[9], 'y4': listXYZ[10], 'z4': listXYZ[11], 'x5': listXYZ[12], 'y5': listXYZ[13], 'z5': listXYZ[14], 'x6': listXYZ[15], 'y6': listXYZ[16], 'z6': listXYZ[17],
               'x7': listXYZ[18], 'y7': listXYZ[19], 'z7': listXYZ[20], 'x8': listXYZ[21], 'y8': listXYZ[22], 'z8': listXYZ[23], 'x9': listXYZ[24], 'y9': listXYZ[25], 'z9': listXYZ[26],
               'x10': listXYZ[27], 'y10': listXYZ[28], 'z10': listXYZ[29], 'x11': listXYZ[30], 'y11': listXYZ[31], 'z11': listXYZ[32], 'x12': listXYZ[33], 'y12': listXYZ[34], 'z12': listXYZ[35],
               'x13': listXYZ[36], 'y13': listXYZ[37], 'z13': listXYZ[38], 'x14': listXYZ[39], 'y14': listXYZ[40], 'z14': listXYZ[41], 'x15': listXYZ[42], 'y15': listXYZ[43], 'z15': listXYZ[44],
               'x16': listXYZ[45], 'y16': listXYZ[46], 'z16': listXYZ[47], 'x17': listXYZ[48], 'y17': listXYZ[49], 'z17': listXYZ[50], 'x18': listXYZ[51], 'y18': listXYZ[52], 'z18': listXYZ[53],
               'x19': listXYZ[54], 'y19': listXYZ[55], 'z19': listXYZ[56]}
    data_features_train.loc[len(data_features_train.index)] = new_row
    l = len(data_features_train)
    row = pd.DataFrame(data_features_train.iloc[l - 1]).T
    pred = model.predict_proba(row)
    print(pred)
    id = model.predict(row)[0]
    if pred[0][id-1] >= 0.4:
        return ComBinePred(image,id)
    return 0
def TestCombine():
    data_folder = "Data/TEST"
    listRS = []
    listCheck = []
    id = 0
    for folder_name in os.listdir(data_folder):
        id = id + 1
        if os.path.isdir(os.path.join(data_folder, folder_name)):
            folder = data_folder + "/" + folder_name
            for file in os.listdir(folder):
                listRS.append(id)
                listCheck.append(predict(folder + "/" + file))
    acc = accuracy_score(listRS,listCheck)
    print(acc)
def TestNonPose():
    path = 'IMG_WRONG/'
    dir_list = os.listdir(path)
    i = 0
    listRS = []
    listcheck = []
    for file in dir_list:
        i = i + 1
        print(file)
        path1 = path + file
        listRS.append(predict(path1))
        listcheck.append(0)
    # print(ComBinePred('IMG/w.jpg'))
    print(accuracy_score(listRS,listcheck))
    print(len(listRS))
# st = time.time()
# print(predict('p.png'))
# end = time.time()
# print(end-st)
# name = 'plank'
# filename = 'p.png'
# #
# timee = str(time.time())
# name = timee+'.jpg'
# os.rename('p.png', name)
# path = 'D/CNWEB/Poses/Pose/Pose/media/images/'
# img = Image.open(name)
# shutil.move(name , path + name)
