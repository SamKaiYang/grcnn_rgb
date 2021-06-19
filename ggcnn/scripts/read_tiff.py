from utils.dataset_processing import grasp, image
import cv2

depth_img = image.DepthImage.from_tiff("pcd2100d.tiff")


print((depth_img.shape))

cv2.namedWindow("tiff_result", cv2.WINDOW_NORMAL)
cv2.imshow("tiff_result", depth_img.img)
cv2.waitKey(30000)
cv2.destroyAllWindows()