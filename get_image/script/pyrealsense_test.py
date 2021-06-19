import pyrealsense2 as rs
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
pipeline.start(config)


while True:
  frames = pipeline.wait_for_frames()
  depth_frame = frames.get_depth_frame()
  color_frame = frames.get_color_frame()
  motion_frame = frames.as_motion_frame()
if not depth_frame or not color_frame:
  continue