#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
sys.path.insert(1, '/usr/local/lib/python3.6/dist-packages/cv2')
import os
import datetime 
import cv2
import rospy
import time 
import argparse
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

parser = argparse.ArgumentParser()
parser.add_argument('--Object_Name', type=str, default='My_Object', help='Class name of training object.')
FLAGS = parser.parse_args()

day = str(datetime.datetime.now()).split(" ")[0]
time = str(datetime.datetime.now()).split(" ")[1]
time = time.split(":")[0] + "_" + time.split(":")[1] + "_" + time.split(":")[2].split(".")[0]

current_time = day + "_" + time + "_"

Object_Name = FLAGS.Object_Name

Train_Data_Dir = os.path.dirname(os.path.realpath(__file__)) + '/Training_Data/' + \
    current_time + '_' + Object_Name + '/'

vis = True

rgb_brdige = CvBridge()
depth_brdige = CvBridge()
rgb_image = np.zeros((0,0,3), np.float32)
depth_image = np.zeros((0,0,1), np.float32)
temp_depth_img = np.zeros((0,0,1), np.float32)

def rgb_callback(Image):
    global rgb_image
    try:
        rgb_image = rgb_brdige.imgmsg_to_cv2(Image, "bgr8")
    except CvBridgeError as e:
        print(e)

def depth_callback(Image):
    global depth_image
    try:
        temp_depth_img = depth_brdige.imgmsg_to_cv2(Image, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)

    temp_depth_img = np.array(temp_depth_img, dtype=np.float32)
        
    cv2.normalize(temp_depth_img, temp_depth_img, 0, 1, cv2.NORM_MINMAX)

    depth_image = temp_depth_img.copy()
    

if __name__ == '__main__':

    rospy.init_node('get_image_cornell', anonymous=True)

    rospy.Subscriber("/camera/color/image_raw", Image, rgb_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)

    take_picture_counter = 0

    while not rospy.is_shutdown():

        if rgb_image.shape != (0, 0, 3) and depth_image.shape != (0, 0, 1):

            if vis:
                cv2.namedWindow("rgb_result", cv2.WINDOW_NORMAL)
                cv2.imshow("rgb_result", rgb_image)
                cv2.waitKey(1)

                cv2.namedWindow("depth_result", cv2.WINDOW_NORMAL)
                cv2.imshow("depth_result", depth_image)
                cv2.waitKey(1)

            if cv2.waitKey(33) & 0xFF == ord('s'):

                if not os.path.exists(Train_Data_Dir):
                    os.makedirs(Train_Data_Dir)
                
                print("============================")
                
                objct_number = "9"

                if take_picture_counter<10:
                    serial_number = objct_number + "10" + str(take_picture_counter)

                elif take_picture_counter<100:
                    serial_number = objct_number +"1" + str(take_picture_counter)

                else:
                    serial_number = objct_number + str(take_picture_counter+100)
                print("take_picture_counter ", take_picture_counter)
                
                rgb_name = str(Train_Data_Dir + "pcd" + serial_number + "r" + ".png")
                cv2.imwrite(rgb_name, rgb_image)
                print("[Save] ", rgb_name)

                depth_name = str(Train_Data_Dir + "pcd" + serial_number + "d" + ".png")
                #相机采集的深度图是 32 位的浮点数。经bridge.imgmsg_to_cv2(msg, '32FC1')转换得到是以米为单位的深度数值。
                #而cv2.imwrite()写入的则是 0-255 的数值，因此深度值都被取整了，导致直接保存的图片全黑了
                cv2.imwrite(depth_name, depth_image*255)
                print("[Save] ", depth_name)

                #save .tiff
                depth_name_tiff = str(Train_Data_Dir + "pcd" + serial_number + "d" + ".tiff")
                cv2.imwrite(depth_name_tiff, depth_image)
                print("[Save] ", depth_name_tiff)

                print("============================")

                take_picture_counter += 1
            
            else:
                pass

    cv2.destroyAllWindows()