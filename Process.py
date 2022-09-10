import pickle
from os import listdir
import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os
from statistics import mean
import openpyxl as openpyxl
from Point import Point
from Angle import Angle


point = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
number_angle = 15
def IMGprocess(img):
    #print(img)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    # cap = cv2.VideoCapture(0)
    output = cv2.imread(img)
    output = cv2.resize(output, (600,400))
    # cv2.imshow("before", output)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.9) as pose:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output.flags.writeable = False
        results_check = pose.process(output)
        output.flags.writeable = True
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        if results_check.pose_landmarks.landmark[0].x > results_check.pose_landmarks.landmark[32].x:
            output = cv2.flip(output, 1)
    return output

def normalizedVector(listRS):
    X = []
    Y = []
    Z = []
    listPoint = []
    for rs in listRS:
        X.append(rs.x)
        Y.append(rs.y)
        Z.append(rs.z)
    width = max(X)-min(X)
    height = max(Y) - min(Y)
    deep = max(Z) - min(Z)
    scale_factor = max(width,height,deep)
    for i in range(len(X)):
        new_x = (X[i] - mean(X))/scale_factor
        new_y = (Y[i] - mean(Y)) / scale_factor
        new_z = (Z[i] - mean(Z)) / scale_factor
        p = Point(new_x,new_y,new_z)
        listPoint.append(p)
    return listPoint

def data_angle(image):
    img = IMGprocess(image)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.9) as pose:
        output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output.flags.writeable = False
        results = pose.process(output)
        output.flags.writeable = True
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        # print(results.pose_landmarks)
        listRS = []
        for i in point:
            listRS.append(results.pose_landmarks.landmark[i])
        # listPoint = normalizedVector(listRS)
        # listG = Angle(listPoint)
        # listG.save_angle()
        listXYZ = []
        for p in listRS:
            listXYZ.append(p.x)
            listXYZ.append(p.y)
            listXYZ.append(p.z)
        # mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # cv2.imshow("img", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return listXYZ

# data_angle("Data/Train/plank/00000313.jpg")
# data_angle("IMG/istock.jpg")



















