#!/usr/bin/env python3

import rospy
import time
import numpy as np
from grcnn_rgb.msg import GGCNN_Grasp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
clf = KMeans(n_clusters=2)
import cv2
import csv
import os
path = os.path.dirname(__file__) #獲取當前路徑

grasp = GGCNN_Grasp()

grasp_area = 0
Aspect_ratio = 0
bounding_area = 0
compare = 0
data_list = [] 

camera_height = 38.5 #cm

def data_read():
    global data_list
    with open(path+'/kmeans.csv', "r", newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        # 以迴圈輸出指定欄位
        for row in rows:
            data_list.append([row['width'], row['bounding_area'], row['Aspect_ratio'], row['compare']])
        data_list = [[float(x) for x in row] for row in data_list] 
        data_list = np.array(data_list)
        print(data_list)
def data_append():
    global grasp, grasp_area, Aspect_ratio, compare, bounding_area
    with open(path+'/kmeans.csv', "a", newline='') as csvfile:
    # 定義欄位
        fieldnames = ['width', 'bounding_area', 'Aspect_ratio','compare']
        # 將 dictionary 寫入 CSV 檔
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # # 寫入第一列的欄位名稱
        # writer.writeheader()
        # 寫入資料
        writer.writerow({'width': round(grasp.width,2) , 'bounding_area': round(bounding_area,2), 'Aspect_ratio': round(Aspect_ratio,2), 'compare': round(compare,2)})
    
def get_Grasp(msg):
    global grasp
    grasp = msg

def kmeans_fit():
    global clf, data_list
    #開始訓練！
    clf.fit(data_list)
    clf.fit_predict(data_list)
    # 儲存模型
    joblib.dump(clf,path+'/grasp_kmeans.pkl')
def grasp_collation():
    global grasp, grasp_area, Aspect_ratio, compare, bounding_area
    # grasp.length
    # grasp.width
    # grasp.yolo_bbox_min_x
    # grasp.yolo_bbox_max_x
    # grasp.yolo_bbox_min_y
    # grasp.yolo_bbox_max_y
    # print("grasp_width:",grasp.width)
    bounding_area = (grasp.yolo_bbox_max_x-grasp.yolo_bbox_min_x)*(grasp.yolo_bbox_max_y-grasp.yolo_bbox_min_y)
    # print("bounding_area:",bounding_area)
    
    grasp_area = grasp.length*grasp.width
    # print("grasp_area:",grasp_area)

    if (grasp.yolo_bbox_max_y-grasp.yolo_bbox_min_y) != 0.0:
        Aspect_ratio = (grasp.yolo_bbox_max_x-grasp.yolo_bbox_min_x)/(grasp.yolo_bbox_max_y-grasp.yolo_bbox_min_y)
    else: 
        Aspect_ratio = 0
    # print("Aspect_ratio:",Aspect_ratio)

    if grasp_area != 0.0:
        compare = bounding_area/grasp_area
    else: 
        compare = 0
    # print("compare:",compare)

def kmeans_predict():
    global clf, data_list
    global grasp, grasp_area, Aspect_ratio, compare, bounding_area
    
    # 載入模型
    clf=joblib.load(path+'/grasp_kmeans.pkl')
    #這樣就可以取得預測結果了！
    # clf.labels_
    # print(clf.labels_)
    # print(kmeans.cluster_centers_)
    X = np.array([[grasp.width,bounding_area,Aspect_ratio,compare]])
    # print(X)
    # print("tool:",clf.predict(X))
    if clf.predict(X)==1:
        print("tool:suck")
    elif clf.predict(X)==0:
        print("tool:grasp")

if __name__ == '__main__':
    rospy.init_node('kmeans_node', anonymous=True)
    rospy.Subscriber("/object/Grasp_Detect", GGCNN_Grasp, get_Grasp)
    data_read()
    # kmeans_fit()# 訓練並儲存模型
    while not rospy.is_shutdown():
        grasp_collation()
        kmeans_predict()#預測
        time.sleep(0.5)
    