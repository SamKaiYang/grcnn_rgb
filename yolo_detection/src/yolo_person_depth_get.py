#!/usr/bin/env python3
import sys

# depth information
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import cv2
from get_rs_image import Get_image
from cv_bridge import CvBridge, CvBridgeError

import rospy
import os
import threading
import numpy as np
import math
import enum
from darknet_ros_msgs.msg import BoundingBox
from darknet_ros_msgs.msg import BoundingBoxes
from yolo_detection.msg import ROI_array
from yolo_detection.msg import ROI
from yolo_detection.msg import depth_alert
import time
#import cv2
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
# 子執行序工作函數
import threading
import time

obj_num = 0
center_x = 0
center_y = 0
person_distance = 0
time_cnt = 0
Depth_level = 0
Depth_level = depth_alert()
alert_flag = False

class bounding_boxes():
    def __init__(self,probability,xmin,ymin,xmax,ymax,id_name,Class_name):
        self.probability = probability
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.id_name = str(id_name)
        self.Class_name = str(Class_name)

# boxes = bounding_boxes(0,0,0,0,0,0,0)
# YOLO V4 輸入
i = 0
def Yolo_callback(data):
    global obj_num,center_x,center_y,Depth_level,person_distance

    person_detect_flag = False
    boxes = bounding_boxes(0,0,0,0,0,0,0)
    #global probability,xmin,ymin,xmax,ymax,id_name,Class_name
    obj_num = len((data.bounding_boxes))
    if obj_num == 0:
            print("No Object Found!")
            print("change method to Realsense!")
    else:
        #i = obj_num-1
        List = []
        for i in range(len(data.bounding_boxes)):
            if data.bounding_boxes[i].Class == "person":
                person_detect_flag = True
                boxes.probability = data.bounding_boxes[i].probability
                boxes.xmin = data.bounding_boxes[i].xmin
                boxes.ymin = data.bounding_boxes[i].ymin
                boxes.xmax = data.bounding_boxes[i].xmax
                boxes.ymax = data.bounding_boxes[i].ymax
                boxes.id_name = data.bounding_boxes[i].id
                boxes.Class_name = data.bounding_boxes[i].Class
                
                center_x  = (boxes.xmax+boxes.xmin)/2
                center_y  = (boxes.ymax+boxes.ymin)/2

                ROI_data = ROI()
                ROI_data.probability = boxes.probability
                ROI_data.object_name= boxes.Class_name
                ROI_data.id = boxes.id_name
                ROI_data.x = center_x
                ROI_data.y = center_y
                ROIarray = ROI_array()
                List.append(ROI_data)
                ROIarray.ROI_list = List

                if (type(listener.cv_image) is np.ndarray)and(type(listener.cv_depth) is np.ndarray):
                    #cv2.imshow("rgb module image", listener.cv_image)
                    img_color = np.asanyarray(listener.cv_image)  # 把图像像素转化为数组
                    img_depth = np.asanyarray(listener.cv_depth)  # 把图像像素转化为数组
                    if center_x != None and center_y != None:
                        cv2.circle(img_color, (round(center_x), round(center_y)), 8, [255, 0, 255], thickness=-1)
                        # cv2.putText(img_color, "Distance/mm:"+str(img_depth[round(center_x), round(center_y)]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
                        person_distance = img_depth[round(center_y), round(center_x)]
                        
            else:
                center_x = None
                center_y = None
        if person_detect_flag == True:
            pub.publish(ROIarray)
            person_detect_flag = False

def alert_level_cal():
    global person_distance, alert_flag, Depth_level
    print("Distance: %d mm"%person_distance)
    if person_distance < 1000 and alert_flag == False:
        Depth_level = 1
    elif person_distance < 1000 and alert_flag == True:
        Depth_level = 2
    else :
        Depth_level = 0

    pub_alert.publish(Depth_level)

def thread_time_cal():
    global Depth_level, alert_flag, person_distance
    count = 0
    while True: 
        try:
            if person_distance < 1000 and count < 3:
                count += 1
                alert_flag = False
                time.sleep(1)
            elif person_distance < 1000 and count == 3:
                alert_flag = True
            elif person_distance >= 1000:
                count = 0
                alert_flag = False
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        except:
            print("程式出現其它異常")



if __name__ == '__main__':
    #global boxes
    argv = rospy.myargv()
    rospy.init_node('yolo_boundingboxes', anonymous=True) #
    listener = Get_image()
    rate = rospy.Rate(10) # 10hz
    rospy.Subscriber("/darknet_ros/bounding_boxes",BoundingBoxes,Yolo_callback)
    pub = rospy.Publisher("obj_position", ROI_array, queue_size=10)
    pub_alert = rospy.Publisher("alert_level", depth_alert, queue_size=10)
    # 建立一個子執行緒
    t = threading.Thread(target = thread_time_cal)
    # 執行該子執行緒
    t.start()
    while not rospy.is_shutdown():
        listener.display_mode = 'depth'
        alert_level_cal()
        cv2.waitKey(1)

        os.system("clear")
        rate.sleep()
    # 等待 t 這個子執行緒結束
    t.join()
    rospy.spin()
    
