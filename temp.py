
import cv2
from PIL import Image
from headpose.detect import PoseEstimator
import numpy as np
import math

est = PoseEstimator()

img1 = Image.open('/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f20_to_f19_faceVid2Vid/S_cropped.png')
img1 = np.array(img1)

img2 = Image.open('/apdcephfs_cq2/share_1290939/branchwang/data/e4s_vis/f20_to_f19_faceVid2Vid/T_cropped.png')
img2 = np.array(img2)

# est.detect_landmarks(img)
pose1 = est.pose_from_image(image=img1)
pose2 = est.pose_from_image(image=img2)

diff = (pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2 + (pose1[2] - pose2[2]) ** 2
print(math.sqrt(diff))