#!/usr/bin/env python3

import rospy
import time
from grcnn_rgb.msg import GGCNN_Grasp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv
import os
path = os.path.dirname(__file__) #獲取當前路徑


grasp = GGCNN_Grasp()
clf = KMeans(n_clusters=2)

with open(path+'/kmeans.csv', "r", newline='') as csvfile:
    rows = csv.DictReader(csvfile)

    print(rows)
    # 以迴圈輸出指定欄位
    for row in rows:
        print(row['width'], row['bounding_area'])

# with open(path+'/kmeans.csv', "w", newline='') as csvfile:
#     writer = csv.writer(csvfile)

#     # 寫入一列資料
#     writer.writerow(['姓名', '身高', '體重'])



def get_Grasp(msg):
    global grasp
    grasp = msg

def define_data():
    X = np.random.rand(100,2)

def kmeans_fit():
    global clf
    #開始訓練！
    clf.fit(X)
def grasp_collation():
    global grasp
    # grasp.length
    # grasp.width
    # grasp.yolo_bbox_min_x
    # grasp.yolo_bbox_max_x
    # grasp.yolo_bbox_min_y
    # grasp.yolo_bbox_max_y
    print("grasp_width:",grasp.width)
    bounding_area = (grasp.yolo_bbox_max_x-grasp.yolo_bbox_min_x)*(grasp.yolo_bbox_max_y-grasp.yolo_bbox_min_y)
    print("bounding_area:",bounding_area)
    
    grasp_area = grasp.length*grasp.width
    print("grasp_area:",grasp_area)

    if (grasp.yolo_bbox_max_y-grasp.yolo_bbox_min_y) != 0.0:
        Aspect_ratio = (grasp.yolo_bbox_max_x-grasp.yolo_bbox_min_x)/(grasp.yolo_bbox_max_y-grasp.yolo_bbox_min_y)
    else: 
        Aspect_ratio = 0
    print("Aspect_ratio:",Aspect_ratio)

    if grasp_area != 0.0:
        compare = bounding_area/grasp_area
    else: 
        compare = 0
    print("compare:",compare)

def kmeans_predict(data):

    #這樣就可以取得預測結果了！
    clf.labels_

    #最後畫出來看看
    #真的分成三類！無意義的資料也能分～
    plt.scatter(X[:,0],X[:,1], c=clf.labels_)
    #KMeans的使用時機就在於～你根本不知道測試的資料有什麼特性的時候
    #就是用他的時候了，我稱KMeans為盲劍客 XD
    plt.show()

if __name__ == '__main__':
    rospy.init_node('kmeans_node', anonymous=True)
    rospy.Subscriber("/object/Grasp_Detect", GGCNN_Grasp, get_Grasp)
    # kmeans_fit()
    while not rospy.is_shutdown():
        # kmeans_predict(grasp)
        grasp_collation()
        time.sleep(1)