#!/usr/bin/env python3
import sys
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
sys.path.insert(1, '/usr/local/lib/python3.6/dist-packages/cv2')
import argparse
from os import path
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import save_results, plot_results, plot_results_lose

import rospy
from grcnn_rgb.msg import GGCNN_Grasp
from grcnn_rgb.msg import GGCNN_Grasp_array
from ggcnn.msg import ROI
from ggcnn.msg import ROI_array

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# current path in ROS 
here = path.dirname(path.abspath(__file__))

rgb_image_raw = np.zeros((0,0,3), np.float32)
depth_image_raw = np.zeros((0,0,1), np.float32)

rgb_bridge = CvBridge()
depth_bridge = CvBridge()

curr_yolo = ROI() 
curr_yolo_list = []
# curr_yolo_list = ROI_array()

def rgb_image_raw_callback(msg):
    global rgb_image_raw
    try:
        rgb_image_raw = rgb_bridge.imgmsg_to_cv2(msg)
    except CvBridgeError as e:
        print (e)

def depth_img_callback(msg):
    global depth_image_raw
    try:
        depth_image_raw = depth_bridge.imgmsg_to_cv2(msg)
    except CvBridgeError as e:
        print (e)
    depth_image_raw = depth_image_raw[..., np.newaxis]

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str, default=here + '/logs/210621_2154_training_cornell_0_12/epoch_20_iou_1.00',
                        help='Path to saved network to evaluate')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args

def get_RegionYOLO(ROI_array):
    global curr_yolo_list
    if (len(ROI_array.ROI_list)>0):
        curr_yolo_list = ROI_array.ROI_list
    else:
        curr_yolo_list = []

if __name__ == '__main__':

    rospy.init_node('GRCNN_GRBD_node', anonymous=True)
    
    ggcnn_pub = rospy.Publisher("/object/Grasp_Detect", GGCNN_Grasp, queue_size=100)
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_image_raw_callback, queue_size=100)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_img_callback, queue_size=100)
    rospy.Subscriber("/object/ROI_array", ROI_array, get_RegionYOLO)

    args = parse_args()

    cam_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb)

    # Load Network
    print('Loading model...')
    net = torch.load(args.network)
    print('Loading model ok')

    # Get the compute device
    device = get_device(args.force_cpu)
    print("device ", device)
    fig = plt.figure(figsize=(10, 5))
    
    while not rospy.is_shutdown():
        depth_image_raw = np.asarray(depth_image_raw, dtype=np.float32)
        # Determine depth scale
        depth_image_raw *= 0.0010000000474974513

        x, depth_img, rgb_img = cam_data.get_data(rgb=rgb_image_raw, depth=depth_image_raw)
 
        if rgb_img.size != 0 and depth_img.size != 0:
            with torch.no_grad():
                xc = x.to(device)

                pred = net.predict(xc)

                q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
                
                gs = plot_results_lose(fig=fig,
                                rgb_img=cam_data.get_rgb(rgb_image_raw, False),
                                depth_img=np.squeeze(cam_data.get_depth(depth_image_raw)),
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                no_grasps=args.n_grasps,
                                grasp_width_img=width_img,
                                vis=True)

                if len(gs) != 0: 

                    grasp_x = gs[0].center[1]
                    grasp_y = gs[0].center[0]
                    grasp_angle = gs[0].angle
                    grasp_length = gs[0].length
                    grasp_width = gs[0].width

                    print("center: {} , angle: {} , length: {} , width: {} "\
                        .format((grasp_x, grasp_y), grasp_angle, grasp_length, grasp_width))
                    
                    pub_Grasp = GGCNN_Grasp()
                    pub_Grasp_array = GGCNN_Grasp_array()
                    pub_Grasp.object_name = "no match"
                    # print("curr_yolo_list ", curr_yolo_list)
                    if len(curr_yolo_list)> 0:
                        for i in range(len(curr_yolo_list)):
                            curr_yolo.min_x = curr_yolo_list[i].min_x
                            curr_yolo.Max_x = curr_yolo_list[i].Max_x
                            curr_yolo.min_y = curr_yolo_list[i].min_y
                            curr_yolo.Max_y = curr_yolo_list[i].Max_y
                            curr_yolo.score = curr_yolo_list[i].score
                            curr_yolo.object_name = curr_yolo_list[i].object_name

                            if curr_yolo.min_x <= grasp_x <=curr_yolo.Max_x and curr_yolo.min_y <= grasp_y <= curr_yolo.Max_y:
                                pub_Grasp.object_name = curr_yolo.object_name
                                print("get match!!")
                                pub_Grasp.x = grasp_x
                                pub_Grasp.y = grasp_y
                                pub_Grasp.angle = grasp_angle
                                pub_Grasp.length = grasp_length
                                pub_Grasp.width = grasp_width
                                pub_Grasp.yolo_bbox_min_x = curr_yolo.min_x
                                pub_Grasp.yolo_bbox_max_x = curr_yolo.Max_x
                                pub_Grasp.yolo_bbox_min_y = curr_yolo.min_y
                                pub_Grasp.yolo_bbox_max_y = curr_yolo.Max_y
                                ggcnn_pub.publish(pub_Grasp)
                                break

                    
