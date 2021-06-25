# open all Tool selection strategy
```bash
source venv/bin/activate
roslaunch grcnn_rgb grcnn_all.launch
```

<!-- # realsense d435i and yolo v4
terminal 1
```bash
roslaunch realsense2_camera rs_aligned_depth.launch 
```
terminal 2
```bash
roslaunch yolo_detection yolo_get_wrs.launch
```
(can't use)roslaunch grcnn_rgb grcnn_grasp.launch

# GRCNN
terminal 3
```bash
source venv/bin/activate
rosrun grcnn_rgb run_realtime.py 
``` -->
# K-means dataset
```bash
rosrun grcnn_rgb kmeans_dataset.py 
```

# Environment build
```bash
python3.6 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install imagecodecs
```

Download Cornell Grasping Dataset

[https://www.kaggle.com/oneoneliu/cornell-grasp](https://www.kaggle.com/oneoneliu/cornell-grasp)
### Download YOLO package
```bash
cd <catkin_workspace>/src
#git clone https://github.com/leggedrobotics/darknet_ros # up to v3
git clone https://github.com/tom13133/darknet_ros # up to v4
cd darknet_ros/ && git submodule update --init --recursive
cd ~/catkin_workspace
# before build, check (-O3 -gencode arch=compute_<version>,code=sm_<version>) part in darknet_ros/darknet_ros/CMakeLists.txt if you use CUDA
# ex) 75 for GTX1650 and RTX2060, 86 for RTX3080, 62 for Xavier 
catkin_make

#download weight
cd <catkin_workspace>/src/darknet_ros/darknet_ros/yolo_network_config/weights
# yolo v4
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
# yolo v4 tiny
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```
### Use realsense 
```bash
cd darknet_ros/config
sudo gedit ros.yaml
# Found->
  camera_reading:
    topic: /usb_cam/image_raw
# Modify the following to->
  camera_reading:
    topic: /camera/color/image_raw

cd darknet_ros/launch
sudo gedit darknet_ros.launch
# Found line 6.23->
"/camera/rgb/image_raw"
# Modify the following to->
"/camera/color/image_raw"
```
