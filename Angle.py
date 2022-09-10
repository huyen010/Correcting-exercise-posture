import numpy as np
import math
from Point import  Point

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
        # point21 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
        # point23 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]
        # cosa = (point21[0] * point23[0] + point21[1] * point23[1] ) / (
        #         math.sqrt(point21[0] * point21[0] + point21[1] * point21[1] ) * math.sqrt(
        #         point23[0] * point23[0] + point23[1] * point23[1] ))
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
