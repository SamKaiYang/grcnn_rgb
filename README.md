# open all 
```bash
source venv/bin/activate
roslaunch grcnn_rgb grcnn_all.launch
```

# realsense d435i and yolo v4
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
```
# K-means 夾取工具選擇
```bash
rosrun grcnn_rgb kmeans.py 
```