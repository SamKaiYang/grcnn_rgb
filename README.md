# realsense d435i and yolo v4
terminal 1
```bash
roslaunch realsense2_camera rs_rgbd.launch 
or
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
