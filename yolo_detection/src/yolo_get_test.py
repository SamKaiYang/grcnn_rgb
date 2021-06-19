#!/usr/bin/env python3
import sys
# sys.path.insert(1, "/home/iclab/.local/lib/python3.6/site-packages/")
# sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import rospy
import os
import threading
import numpy as np
#from sensor_msgs.msg import Image
# from geometry_msgs.msg import Point
# from geometry_msgs.msg import PoseStamped
# from cv_bridge import CvBridge, CvBridgeError
import math
import enum
from darknet_ros_msgs.msg import BoundingBox
from darknet_ros_msgs.msg import BoundingBoxes
from yolo_detection.msg import ROI_array
from yolo_detection.msg import ROI
import time
#import cv2
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
#from get_rs_image import Get_image
#from get_rs_image.srv import *

obj_num = 0

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
    global obj_num
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
        
        print("ROI_array:",ROIarray)
        pub.publish(ROIarray)

if __name__ == '__main__':
    #global boxes
    argv = rospy.myargv()
    rospy.init_node('yolo_boundingboxes', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    rospy.Subscriber("/darknet_ros/bounding_boxes",BoundingBoxes,Yolo_callback)
    #pub = rospy.Publisher("obj_position", ROI, queue_size=10)
    pub = rospy.Publisher("obj_position", ROI_array, queue_size=10)
    while not rospy.is_shutdown():
        # print("ID:",boxes.id_name)
        # print("信心值:",boxes.probability)
        # print("Class:",boxes.Class_name)
        # print("xmin:",boxes.xmin)
        # print("ymin:",boxes.ymin)
        # print("xmnax",boxes.xmax)
        # print("ymax:",boxes.ymax)
        os.system("clear")
        rate.sleep()
    rospy.spin()
