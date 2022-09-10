import pickle
from os import listdir

import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os
import csv
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

from predict import predict

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

class Angle:
    ListGoc = []
    ListToaDo = []

    def __init__(self, listRs):
        self.ListToaDo = listRs
        self.ListGoc = []

    @staticmethod
    def find_angle(point1, point2, point3):
        point21 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
        point23 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]
        cosa = (point21[0] * point23[0] + point21[1] * point23[1] + point21[2] * point23[2]) / (
                math.sqrt(point21[0] * point21[0] + point21[1] * point21[1] + point21[2] * point21[2]) * math.sqrt(
                point23[0] * point23[0] + point23[1] * point23[1] + point23[2] * point23[2]))
        return round(180 * np.arccos(cosa) / np.pi, 5)

    @staticmethod
    def find_average_point(point1, point2):
        point_average = Point((point1.x + point2.x) / 2, (point1.y + point2.y) / 2, (point1.z + point2.z) / 2)
        return point_average

    def save_angle(self):
        average = self.find_average_point(self.ListToaDo[1], self.ListToaDo[2])
        self.ListGoc.append(self.find_angle(self.ListToaDo[0], average, self.ListToaDo[12]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[1], self.ListToaDo[2], self.ListToaDo[4]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[2], self.ListToaDo[1], self.ListToaDo[3]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[6], self.ListToaDo[4], self.ListToaDo[2]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[5], self.ListToaDo[3], self.ListToaDo[1]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[8], self.ListToaDo[6], self.ListToaDo[4]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[7], self.ListToaDo[5], self.ListToaDo[3]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[4], self.ListToaDo[2], self.ListToaDo[10]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[3], self.ListToaDo[1], self.ListToaDo[9]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[9], self.ListToaDo[10], self.ListToaDo[12]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[10], self.ListToaDo[9], self.ListToaDo[11]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[10], self.ListToaDo[12], self.ListToaDo[14]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[9], self.ListToaDo[11], self.ListToaDo[13]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[18], self.ListToaDo[14], self.ListToaDo[16]))
        self.ListGoc.append(self.find_angle(self.ListToaDo[15], self.ListToaDo[13], self.ListToaDo[17]))


point = [0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
number_angle = 15
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
def process(img):
    output = cv2.imread(img)
    weight, height, c = output.shape
    # height = int(output.shape[0] * 60 / 100)
    # width = int(output.shape[1] * 60 / 100)
    # output = cv2.resize(output, (width, height))
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
    return output, weight, height


def data_angle(img):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic
    list_landmark = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.9) as pose:
        output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output.flags.writeable = False
        results = pose.process(output)
        output.flags.writeable = True
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        for p in point:
            list_landmark.append(results.pose_landmarks.landmark[p])
        # mp_drawing.draw_landmarks(output, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # cv2.imshow("img", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return list_landmark


def bounding_box(landmarks):
    min_x = 10000
    min_y = 10000
    min_z = 10000
    max_x = -10000
    max_y = -10000
    max_z = -10000
    for landmark in landmarks:
        if landmark[0] > max_x:
            max_x = landmark[0]
        if landmark[0] < min_x:
            min_x = landmark[0]
        if landmark[1] > max_y:
            max_y = landmark[1]
        if landmark[1] < min_y:
            min_y = landmark[1]
    return min_x, min_y, max_x, max_y


def get_new_pose(pose, bound1, bound2):
    pose[:, 0] = pose[:, 0] - bound1
    pose[:, 1] = pose[:, 1] - bound2
    #pose[:, 2] = pose[:, 2] - bound3
    return pose


def L2_normalize(pose):
    new_pose = []
    pose = np.array(pose)
    # for i in range(19):
    #     if pose[i][0] != float(0) and pose[i][1] != float(0):
    #         l2 = float(math.sqrt(pose[i][0]**2 + pose[i][1]**2))
    #         pose[i][0] = pose[i][0] / l2
    #         pose[i][1] = pose[i][1] / l2
    #         #pose[i][2] = pose[i][2] / l2
    pose[:, 0] = pose[:, 0] / max(pose[:, 0])
    # pose_1[:,1] = pose_1[:,1] / max(pose_1[:,1])
    for i in range(19):
        new_pose.append(pose[i][0])
        new_pose.append(pose[i][1])
        #new_pose.append(pose[i][2])
    return np.array(new_pose)

def new_pose_img(img):
    output, w, h = process(img)
    landmarks = data_angle(output)
    pose = []
    for i in landmarks:
        pose.append((i.x * w, i.y * h))
    pose = np.array(pose)
    min_x, min_y, max_x, max_y = bounding_box(pose)
    new_pose = get_new_pose(pose, min_x, min_y)
    return new_pose


def compare_pose(img1, img2):
    pose1 = new_pose_img(img1)
    pose2 = new_pose_img(img2)
    resize_x = max(pose2[:, 0])/max(pose1[:, 0])
    resize_y = max(pose2[:, 1])/max(pose1[:, 1])
    #resize_z = max(pose2[:, 2]) / max(pose1[:, 2])
    pose1[:, 0] = pose1[:, 0] * resize_x
    pose1[:, 1] = pose1[:, 1] * resize_y
   # pose1[:, 2] = pose1[:, 2] * resize_z
    pose1 = L2_normalize(pose1)
    pose2 = L2_normalize(pose2)
    score = spatial.distance.cosine(pose1, pose2)
    print("Cosine Distance:", score)
    return score

def get_img_true(name):
    switcher = {
        'downdog': 'Data/TRAIN/downdog/00000003.jpg',
        'plank': 'Data/TRAIN/plank/2.jpg',
        'warrior2': 'Data/TRAIN/warrior2/00000076.jpg',
        'goddess': 'Data/TRAIN/goddess/00000106.jpg',
        'dandasana':'Data/TRAIN/dandasana/47.jpg',
        'hanumanasana':'Data/TRAIN/hanumanasana/r.jpg',
        'bitilasana': 'Data/TRAIN/bitilasana/78-0.png',
        'purvottanasana': 'Data/TRAIN/purvottanasana/b.jpg',
        'virabhadrasana': 'Data/TRAIN/virabhadrasana/13.jpg'
    }
    return switcher.get(name, 'nothing')

def check(img, name):
    img_true = get_img_true(name)
    check = ""
    if compare_pose(img, img_true) <= 0.0045:
         check = "True"
    else:
         check = "False"
    return check
# print(check('t.png','warrior2'))


