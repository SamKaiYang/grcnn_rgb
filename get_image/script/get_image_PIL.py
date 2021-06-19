#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROS_Image
from PIL import Image as PIL_Image

import cv2
import numpy as np


class Get_image():
    def __init__(self):
        rospy.init_node('get_image_from_Realsense_D435i', anonymous=True)
        
        self.image = np.zeros((0,0,3), np.uint8)

        rospy.Subscriber("/camera/color/image_raw", ROS_Image, self.callback)
        
        rospy.spin()

    def callback(self, image):
        # self.image = PIL_Image.fromarray(np.asarray(image.data))
        print((image.data))


if __name__ == '__main__':
    listener = Get_image()
