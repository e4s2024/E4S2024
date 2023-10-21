import os
import cv2
import numpy as np
import glob


filelist = sorted(glob.glob('/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/swap_face_video_res/swap_scarlett_to_target3/pasted/*.jpg'))[:200]

fps = 23.976023976023978  # 视频每秒24帧
size = (1920, 1080)  # 需要转为视频的图片的尺寸
# 可以使用cv2.resize()进行修改

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

video = cv2.VideoWriter("/apdcephfs_cq2/share_1290939/branchwang/projects/E4S/res.mp4", fourcc, fps, size)
# 视频保存在当前目录下

print(filelist)

for item in filelist:
    img = cv2.imread(item)
    # img = cv2.resize(img, size)
    video.write(img)

video.release()
