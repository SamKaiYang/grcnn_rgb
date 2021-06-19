#!/usr/bin/env python3

# import python module
import sys
import cv2
import math
import torch
import numpy as np
from os import path
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
sys.path.insert(1, '/usr/local/lib/python3.6/dist-packages/cv2')
# import ROS module
import rospy
from cv_bridge import CvBridge, CvBridgeError

# import ROS message
from sensor_msgs.msg import Image
from ggcnn.msg import ROI
from ggcnn.msg import ROI_array
from ggcnn.msg import GGCNN_Grasp
from ggcnn.msg import GGCNN_Grasp_array

# import user module
from utils.dataset_processing import grasp, image

# current path in ROS 
here = path.dirname(path.abspath(__file__))

bridge = CvBridge()

curr_yolo = ROI() 
curr_ggcnn = GGCNN_Grasp()
curr_rgb_img = np.zeros((0,0,3), np.float32)

Grasp_Detect_pub = rospy.Publisher("/object/Grasp_Detect", GGCNN_Grasp, queue_size=100)

vis = True

def get_RegionYOLO(ROI_array):
    if (len(ROI_array.ROI_list)>0):

        curr_ROI = ROI_array.ROI_list[0]

        curr_yolo.min_x = curr_ROI.min_x
        curr_yolo.Max_x = curr_ROI.Max_x
        curr_yolo.min_y = curr_ROI.min_y
        curr_yolo.Max_y = curr_ROI.Max_y
        curr_yolo.score = curr_ROI.score
        curr_yolo.object_name = curr_ROI.object_name
        # print(curr_yolo)

def get_GraspGGCNN(GGCNN_Grasp_array):
    if (len(GGCNN_Grasp_array.Grasp_list)>0):
        curr_grasp = GGCNN_Grasp_array.Grasp_list[0]

        # in 300*300 ggcnn pixel-size  
        curr_ggcnn.x = curr_grasp.x
        curr_ggcnn.y = curr_grasp.y
        curr_ggcnn.angle = curr_grasp.angle
        curr_ggcnn.length = curr_grasp.length
        curr_ggcnn.width = curr_grasp.width

        # in origin image(yolo, maybe 640*480) pixel-size 
        curr_ggcnn.yolo_bbox_center_x = curr_grasp.yolo_bbox_center_x
        curr_ggcnn.yolo_bbox_center_y = curr_grasp.yolo_bbox_center_y
        curr_ggcnn.yolo_bbox_s = curr_grasp.yolo_bbox_s

        # print(curr_ggcnn)

def get_RGBImage(Image):

    curr_rgb_img = bridge.imgmsg_to_cv2(Image, "bgr8")
    
    pub_yolo = curr_yolo
    pub_ggcnn = curr_ggcnn
    
    print("pub_ggcnn.length : ", pub_ggcnn.length)
    
    bbox_shift_x = (pub_ggcnn.x -150)/300*pub_ggcnn.yolo_bbox_s*2
    bbox_shift_y = (pub_ggcnn.y -150)/300*pub_ggcnn.yolo_bbox_s*2

    Grasp_center_x = int(pub_ggcnn.yolo_bbox_center_x + bbox_shift_x)
    Grasp_center_y = int(pub_ggcnn.yolo_bbox_center_y + bbox_shift_y)

    # set grasp line length
    S = pub_ggcnn.length/2

    # * -1 for image Coordinate 
    dy_1 = -1 * math.ceil(np.sin(pub_ggcnn.angle/(np.pi/2)) * S)
    dx_1 =  math.ceil(np.cos(pub_ggcnn.angle/(np.pi/2)) * S)

    # diagonal coordinates
    dy_2 = -1 * dy_1
    dx_2 = -1 * dx_1

    x_center = int((pub_yolo.min_x + pub_yolo.Max_x)/2)
    y_center = int((pub_yolo.min_y + pub_yolo.Max_y)/2)

    cv2.line(curr_rgb_img, (pub_yolo.min_x, pub_yolo.min_y), (pub_yolo.Max_x, pub_yolo.min_y), (255, 0, 0), 2)
    cv2.line(curr_rgb_img, (pub_yolo.Max_x, pub_yolo.min_y), (pub_yolo.Max_x, pub_yolo.Max_y), (255, 0, 0), 2)
    cv2.line(curr_rgb_img, (pub_yolo.Max_x, pub_yolo.Max_y), (pub_yolo.min_x, pub_yolo.Max_y), (255, 0, 0), 2)
    cv2.line(curr_rgb_img, (pub_yolo.min_x, pub_yolo.Max_y), (pub_yolo.min_x, pub_yolo.min_y), (255, 0, 0), 2)


#========

    cv2.circle(curr_rgb_img, (Grasp_center_x, Grasp_center_y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.line(curr_rgb_img, (Grasp_center_x + dx_2, Grasp_center_y + dy_2), (Grasp_center_x + dx_1, Grasp_center_y + dy_1), (0, 255, 255), 5)

    Grasp = GGCNN_Grasp()
    Grasp.x = Grasp_center_x
    Grasp.y = Grasp_center_y
    Grasp.angle = pub_ggcnn.angle
    Grasp.length = pub_ggcnn.length
    Grasp_Detect_pub.publish(Grasp)

    # # draw angle to image 
    object_name_text = str(pub_yolo.object_name)
    score_text = str(round(pub_yolo.score, 3))

    cv2.putText(curr_rgb_img, object_name_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(curr_rgb_img, score_text, (100, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
    
    if vis:
        cv2.namedWindow("Grasp_result", cv2.WINDOW_NORMAL)
        cv2.imshow("Grasp_result", curr_rgb_img)
        cv2.waitKey(1)

if __name__=="__main__":

    rospy.init_node('plot_grasp_detect_result')

    rospy.Subscriber("/object/GGCNN_Grasp", GGCNN_Grasp_array, get_GraspGGCNN)

    rospy.Subscriber("/object/ROI_array", ROI_array, get_RegionYOLO)

    rospy.Subscriber("/camera/color/image_raw", Image, get_RGBImage)

    rospy.spin()