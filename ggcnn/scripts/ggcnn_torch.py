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
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.feature import peak_local_max

# import user module
from utils.dataset_processing import grasp, image
from dougsm_helpers.timeit import TimeIt
from grasp import GraspRectangles, detect_grasps

# import ROS module
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ggcnn.msg import ROI
from ggcnn.msg import ROI_array
from ggcnn.msg import GGCNN_Grasp
from ggcnn.msg import GGCNN_Grasp_array

# current path in ROS 
here = path.dirname(path.abspath(__file__))

class GGCNN_predict():
    def __init__(self, MODEL_FILE, vis=False):
        
        # self.test_image = "0110"
        
        self.img_crop_size = (300, 300)   
        self.out_size = (300, 300)    
 
        self.zoom = 1
        self.rot = 0

        # need to be modified!!!!!
        self.center = [240, 320]
        
        # x
        self.left = max(0, min(self.center[1] - self.out_size[1] // 2, 640 - self.out_size[1]))
        
        # y
        self.top = max(0, min(self.center[0] - self.out_size[0] // 2, 480 - self.out_size[0]))
        
        # creat image 
        self.origin_rgb_img = np.zeros((0,0,3), np.float32)
        self.curr_rgb_img = np.zeros((0,0,3), np.float32)
        self.temp_rgb_img = np.zeros((0,0,3), np.float32)
        self.curr_depth_img = np.zeros((0,0,3), np.float32)
        self.temp_depth_img = np.zeros((0,0,3), np.float32)

        # load ggcnn model
        self.model = torch.load(MODEL_FILE)
        self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)
        self.vis = vis
        print("model: ", self.model)

        # creat cvbridge
        self.bridge = CvBridge()

        # YOLO 
        self.yolo_bbox_min_x = 0
        self.yolo_bbox_min_y = 0
        self.yolo_bbox_max_x = 0
        self.yolo_bbox_max_y = 0
        self.yolo_bbox_score = 0
        self.yolo_bbox_object_name = "none"
        self.yolo_overstep = 10

        # define ros topic output
        self.Depth_crop_pub = rospy.Publisher('/Depth_crop', Image, queue_size=10)
        self.Grasp_Q_pub = rospy.Publisher('/Grasp_Q', Image, queue_size=10)
        self.Grasp_Angle_pub = rospy.Publisher('/Grasp_Angle', Image, queue_size=10)
        self.RGB_pub = rospy.Publisher('/RGB_output', Image, queue_size=10)
        rospy.Subscriber("/object/ROI_array", ROI_array, self.get_RegionYOLO)

        # GG-CNN publisher
        self.ggcnn_pub = rospy.Publisher("/object/GGCNN_Grasp", GGCNN_Grasp_array, queue_size=100)

        # define ros subscribe input
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_img_callback, queue_size=1)
        rospy.Subscriber("/camera/color/image_raw", Image, self.rgb_img_callback, queue_size=1)



    
    def get_RegionYOLO(self, ROI_array):
        if (len(ROI_array.ROI_list)>0):
            curr_ROI = ROI_array.ROI_list[0]
            self.yolo_bbox_min_x = curr_ROI.min_x
            self.yolo_bbox_min_y = curr_ROI.min_y
            self.yolo_bbox_max_x = curr_ROI.Max_x
            self.yolo_bbox_max_y = curr_ROI.Max_y
            self.yolo_bbox_score = curr_ROI.score
            self.yolo_bbox_object_name = curr_ROI.object_name

    def rgb_img_callback(self, msg):
        self.origin_rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.temp_rgb_img = image.Image(self.origin_rgb_img)

        self.temp_rgb_img.rotate(self.rot, self.center)
        self.temp_rgb_img.crop((self.top, self.left), (min(480, self.top + self.out_size[0]), min(640, self.left + self.out_size[1])))
        self.temp_rgb_img.zoom(self.zoom)
        self.temp_rgb_img.resize((self.out_size[0], self.out_size[1]))
        self.curr_rgb_img = self.temp_rgb_img.img

        # print("self.curr_rgb_img.shape ", self.curr_rgb_img.shape)

        # print("(self.top, self.left) ", (self.top, self.left))
        # print("(min(480, self.top + self.out_size[0]), min(640, self.left + self.out_size[1])) ", (min(480, self.top + self.out_size[0]), min(640, self.left + self.out_size[1])))

        self.RGB_pub.publish(self.bridge.cv2_to_imgmsg(self.curr_rgb_img))

    def depth_img_callback(self, msg):

        try:
            self.temp_depth_img = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            print (e)

        self.temp_depth_img = np.array(self.temp_depth_img, dtype=np.float32)
        
        cv2.normalize(self.temp_depth_img, self.temp_depth_img, 0, 1, cv2.NORM_MINMAX)

        self.curr_depth_img = self.temp_depth_img.copy()

        # self.curr_depth_img = cv2.resize(self.temp_depth_img, (300, 300), interpolation=cv2.INTER_AREA)


        # self.detectGrasp()

    def process_depth_image(self, depth, crop_size, out_size=300, return_mask=False, crop_y_offset=0):

        imh, imw = depth.shape
        
        # print("depth.shape ", depth.shape)

        # with TimeIt('1'):
        #     # Crop.
        #     depth_crop = depth[(imh - crop_size[0]) // 2 - crop_y_offset:(imh - crop_size[0]) // 2 + crop_size[0] - crop_y_offset,
        #                     (imw - crop_size[1]) // 2:(imw - crop_size[1]) // 2 + crop_size[1]]
        
        depth_crop = depth
        
        # Inpaint
        # OpenCV inpainting does weird things at the border.
        with TimeIt('2'):
            depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
            depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

        with TimeIt('3'):
            depth_crop[depth_nan_mask==1] = 0

        with TimeIt('4'):
            # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
            depth_scale = np.abs(depth_crop).max()
            depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.

            with TimeIt('Inpainting'):
                depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

            # Back to original size and value range.
            depth_crop = depth_crop[1:-1, 1:-1]
            depth_crop = depth_crop * depth_scale

        with TimeIt('5'):
            # Resize
            depth_crop = cv2.resize(depth_crop, (out_size[0], out_size[1]), cv2.INTER_AREA)

        if return_mask:
            with TimeIt('6'):
                depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
                depth_nan_mask = cv2.resize(depth_nan_mask, (out_size[0], out_size[1]), cv2.INTER_NEAREST)
            return depth_crop, depth_nan_mask
        else:
            return depth_crop

    def predict(self, depth, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
        if process_depth:
            depth, depth_nan_mask = self.process_depth_image(depth, crop_size, out_size=out_size, return_mask=True, crop_y_offset=crop_y_offset)

        # Inference
        depth = np.clip((depth - depth.mean()), -1, 1)
        depthT = torch.from_numpy(depth.reshape(1, 1, out_size[0], out_size[1]).astype(np.float32)).to(self.device)
        with torch.no_grad():
            pred_out = self.model(depthT)

        points_out = pred_out[0].cpu().numpy().squeeze()
        points_out[depth_nan_mask] = 0

        # Calculate the angle map.
        cos_out = pred_out[1].cpu().numpy().squeeze()
        sin_out = pred_out[2].cpu().numpy().squeeze()
        ang_out = np.arctan2(sin_out, cos_out) / 2.0

        width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

        # Filter the outputs.
        if filters[0]:
            points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
        if filters[1]:
            ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
        if filters[2]:
            width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

        points_out = np.clip(points_out, 0.0, 1.0-1e-3)

        # SM
        # temp = 0.15
        # ep = np.exp(points_out / temp)
        # points_out = ep / ep.sum()

        # points_out = (points_out - points_out.min())/(points_out.max() - points_out.min())

        return points_out, ang_out, width_out, depth.squeeze()

    def detectGrasp(self):

        if (self.curr_depth_img.shape != (0, 0, 1) and self.curr_rgb_img.shape != (0, 0, 3)):
            
            # initial mean value
            total_gs_center = [0, 0]
            total_gs_angle = 0
            total_gs_length = 0
            total_gs_width = 0
            mean_range = 1
            q_x = 0
            q_y = 0

            yolo_bbox_dx = (self.yolo_bbox_max_x - self.yolo_bbox_min_x)
            yolo_bbox_dy = (self.yolo_bbox_max_y - self.yolo_bbox_min_y)

            yolo_bbox_center_x = int((self.yolo_bbox_max_x + self.yolo_bbox_min_x)/2)
            yolo_bbox_center_y = int((self.yolo_bbox_max_y + self.yolo_bbox_min_y)/2)

            # 0.5 yolo bbox Side length
            yolo_bbox_s = yolo_bbox_dx if yolo_bbox_dx > yolo_bbox_dy else yolo_bbox_dy

            yolo_bbox_s = int(yolo_bbox_s/1.5)

            # do "mean_range" time mean
            for i in range(mean_range):
                
                # depth imapg post process
                # depth_crop, depth_nan_mask = self.process_depth_image(self.curr_depth_img, self.img_crop_size, out_size=self.out_size, return_mask=True)
                
                # print("self.curr_depth_img.shape ", self.curr_depth_img.shape)

                # crop from yolo v4
                depth_crop = self.curr_depth_img[max(0, (yolo_bbox_center_y - yolo_bbox_s) - self.yolo_overstep ) : min(self.curr_depth_img.shape[0],  (yolo_bbox_center_y + yolo_bbox_s) + self.yolo_overstep), \
                                                max(0, (yolo_bbox_center_x - yolo_bbox_s) - self.yolo_overstep) : min(self.curr_depth_img.shape[1], (yolo_bbox_center_x + yolo_bbox_s) + self.yolo_overstep)]

                # print("depth_crop.shape ", depth_crop.shape)
                # resize to 300*300 for GG-CNN
                resize_depth_image = cv2.resize(depth_crop, (300, 300), interpolation=cv2.INTER_AREA)

                # predict by GG-CNN
                points, angle, width_img, _ = self.predict(resize_depth_image, crop_size=self.img_crop_size, process_depth=True, out_size=self.out_size)
                
                # # predict by GG-CNN
                # points, angle, width_img, _ = self.predict(depth_crop, process_depth=True, out_size=self.out_size, depth_nan_mask=depth_nan_mask)
                
                # print("points.shape ", points.shape)
                # ROS Publisher for Depth_Crop & Q image & Angle image
                self.Depth_crop_pub.publish(self.bridge.cv2_to_imgmsg(depth_crop))
                self.Grasp_Q_pub.publish(self.bridge.cv2_to_imgmsg(points))
                self.Grasp_Angle_pub.publish(self.bridge.cv2_to_imgmsg(angle))
                    
                # choose best grasp
                gs = detect_grasps(points, angle, width_img=width_img, no_grasps=1)

                # If there is a grasp box
                if len(gs) > 0:
                    # print("center:{}, angle:{}, length:{}, width:{}".format(gs[0].center, gs[0].angle, gs[0].length, gs[0].width))
                    
                    # gs.center = (y, x), so need to do a simple transform
                    total_gs_center[0] = total_gs_center[0] + gs[0].center[1]
                    total_gs_center[1] = total_gs_center[1] + gs[0].center[0]

                    total_gs_angle = total_gs_angle + gs[0].angle
                    total_gs_length = total_gs_length + gs[0].length
                    total_gs_width = total_gs_width + gs[0].width

                    # final time mean calculation
                    if i == (mean_range-1):
                    
                        # calculate mean of (q_x, q_y) & angle & length & width
                        q_x = math.ceil(total_gs_center[0]/mean_range)
                        q_y = math.ceil(total_gs_center[1]/mean_range)
                        total_gs_angle = total_gs_angle/mean_range
                        total_gs_length = total_gs_length/mean_range
                        total_gs_width = total_gs_width/mean_range

                        pub_Grasp_array = GGCNN_Grasp_array()
                        pub_Grasp = GGCNN_Grasp()
                        
                        pub_Grasp.x = q_x
                        pub_Grasp.y = q_y
                        pub_Grasp.angle = total_gs_angle
                        pub_Grasp.length = total_gs_length
                        pub_Grasp.width = total_gs_width

                        pub_Grasp.yolo_bbox_center_x = yolo_bbox_center_x
                        pub_Grasp.yolo_bbox_center_y = yolo_bbox_center_y
                        pub_Grasp.yolo_bbox_s = int(yolo_bbox_s)

                        pub_Grasp_array.Grasp_list.append(pub_Grasp)

                        self.ggcnn_pub.publish(pub_Grasp_array)
                        # print("center:{}, angle:{}, length:{}, width:{}".format((q_x, q_y), total_gs_angle, total_gs_length, total_gs_width))
                       
            # if choose to visualize
            if self.vis:
                
                # set grasp line length
                S = total_gs_length/2

                # * -1 for image Coordinate 
                dy_1 = -1 * math.ceil(np.sin(total_gs_angle/(np.pi/2)) * S)
                dx_1 =  math.ceil(np.cos(total_gs_angle/(np.pi/2)) * S)
                
                # diagonal coordinates
                dy_2 = -1 * dy_1
                dx_2 = -1 * dx_1

                # set q center to red
                # print("q_x, q_y ", (q_x, q_y))
                
                # draw grasp line                
                cv2.line(self.curr_rgb_img, (q_x + dx_2, q_y + dy_2), (q_x + dx_1, q_y + dy_1), (0, 255, 255), 5)
                cv2.circle(self.curr_rgb_img, (q_x, q_y), radius=5, color=(0, 0, 255), thickness=-1)

                # # draw angle to image 
                # text = str(total_gs_angle)
                # text_sin = str(dy_1)
                # text_cos = str(dx_1)

                # cv2.putText(self.curr_rgb_img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(self.curr_rgb_img, text_sin, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                # cv2.putText(self.curr_rgb_img, text_cos, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)

                # show cv window
                cv2.namedWindow("ggcnn_torch_result", cv2.WINDOW_NORMAL)
                cv2.imshow("ggcnn_torch_result", self.curr_rgb_img)
                cv2.waitKey(1)

    def plot_output(self, rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
        """
        Plot the output of a GG-CNN
        :param rgb_img: RGB Image
        :param depth_img: Depth Image
        :param grasp_q_img: Q output of GG-CNN
        :param grasp_angle_img: Angle output of GG-CNN
        :param no_grasps: Maximum number of grasps to plot
        :param grasp_width_img: (optional) Width output of GG-CNN
        :return:
        """
        gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

        for g in gs:
            print("center:{}, angle:{}, length:{}, width:{}".format(g.center, g.angle, g.length, g.width))

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow(self.curr_rgb_img)
        for g in gs:
            g.plot(ax)
        ax.set_title('RGB')
        ax.axis('off')

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(depth_img, cmap='gray')
        for g in gs:
            g.plot(ax)
        ax.set_title('Depth')
        ax.axis('off')

        ax = fig.add_subplot(2, 2, 3)
        plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
        ax.set_title('Q')
        ax.axis('off')
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 2, 4)
        plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
        ax.set_title('Angle')
        ax.axis('off')
        plt.colorbar(plot)
        plt.show()
        plt.pause(10)
        plt.close()


if __name__=="__main__":
    
    rospy.init_node('ggcnn_predict')

    MODEL_FILE = here + '/models/20210505/epoch_34_iou_0.87'

    GraspDetect = GGCNN_predict(MODEL_FILE, vis=False)

    while not rospy.is_shutdown():
        GraspDetect.detectGrasp()
    
    # rospy.spin()