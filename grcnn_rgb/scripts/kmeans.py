#!/usr/bin/env python3

import rospy
import time
from grcnn_rgb.msg import GGCNN_Grasp
from sklearn.cluster import KMeans
# from grcnn_rgb.msg import GGCNN_Grasp_array


grasp = GGCNN_Grasp()
# grasp_list = GGCNN_Grasp_array()

def get_Grasp(msg):
    global grasp
    grasp = msg

if __name__ == '__main__':
    rospy.init_node('kmeans_node', anonymous=True)
    rospy.Subscriber("/object/Grasp_Detect", GGCNN_Grasp, get_Grasp)

    while not rospy.is_shutdown():
        print(grasp)
        time.sleep(1)