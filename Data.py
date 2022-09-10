import os
from os import listdir
from Point import  Point
from Angle import Angle
import openpyxl as openpyxl
from Process import data_angle
import truefalse
number_angle = 15
def save_data1():
    labels = []
    angles = []
    data_folder = "Data/TRAIN"
    id = 0
    for folder_name in os.listdir(data_folder):
        id = id + 1
        if os.path.isdir(os.path.join(data_folder, folder_name)):
            with open("Dataset/data.txt", 'at') as fileopen:
                folder = data_folder + "/" + folder_name
                for file in os.listdir(folder):
                    fileopen.write(file + '   ')
                    print(file)
                    listAngle = data_angle(folder + "/" + file)
                    for i in range(number_angle):
                        fileopen.write(str(listAngle[i]) + "   ")
                    fileopen.write(str(id))
                    fileopen.write("\n")
def save_data2():
    labels = []
    angles = []
    data_folder = "Data/TRAIN"
    id = 0
    for folder_name in os.listdir(data_folder):
        id = id + 1
        if os.path.isdir(os.path.join(data_folder, folder_name)):
            with open("Dataset/data.txt", 'at') as fileopen:
                folder = data_folder + "/" + folder_name
                for file in os.listdir(folder):
                    fileopen.write(file + '   ')
                    print(file)
                    listRS = data_angle(folder + "/" + file)
                    for i in range(len(listRS)):
                        fileopen.write(str(listRS[i].x) + "   ")
                        fileopen.write(str(listRS[i].y) + "   ")
                        fileopen.write(str(listRS[i].z) + "   ")
                    fileopen.write(str(id))
                    fileopen.write("\n")
def GetData():
    listData = []
    with open('Dataset/data.txt', 'r') as f:
        lines = f.readlines()
        for i in range(0,len(lines)):
            addList = []
            line = lines[i]
            listG = line.split()
            for g in listG:
                addList.append(g)
            listData.append(addList)
    return listData

def output_Excel(input_detail, output_excel_path):
    # Xác định số hàng và cột lớn nhất trong file excel cần tạo
    row = len(input_detail)
    column = len(input_detail[0])

    # Tạo một workbook mới và active nó
    wb = openpyxl.Workbook()
    ws = wb.active

    # Dùng vòng lặp for để ghi nội dung từ input_detail vào file Excel
    for i in range(0, row):
        for j in range(0, column):
            v = input_detail[i][j]
            ws.cell(column=j + 1, row=i + 1, value=v)

    # Lưu lại file Excel
    wb.save(output_excel_path)
# save_data2()
input_detail = GetData()
output_excel_path= 'Dataset/yogaTrain.xlsx'
output_Excel(input_detail,output_excel_path)