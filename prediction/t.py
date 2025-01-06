import cv2
import os

import torch
import numpy as np

# 이미지 파일 경로 (순서대로)
image_files = ['/data/onebom/data/ucf101/ucf101_jpg/train/v_ApplyEyeMakeup_g01_c01/0000.jpg',
               '/data/onebom/data/ucf101/ucf101_jpg/train/v_ApplyEyeMakeup_g01_c01/0001.jpg',
               '/data/onebom/data/ucf101/ucf101_jpg/train/v_ApplyEyeMakeup_g01_c01/0002.jpg',
               '/data/onebom/data/ucf101/ucf101_jpg/train/v_ApplyEyeMakeup_g01_c01/0003.jpg']
            #    '/data/onebom/data/ucf101/ucf101_jpg/train/v_ApplyEyeMakeup_g01_c01/0004.jpg']

# 동영상 저장 경로 및 설정
output_video = './output_video.avi'

# 이미지들을 동영상으로 저장
frames = []
for image_file in image_files:
    frame = cv2.imread(image_file)
    frames.append(frame)  # 프레임 추가

video = np.expand_dims(np.array(frames), axis=0)
video = np.transpose(video, (0, 1, 4, 2, 3))

motion_cond_predictor = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
pred_tracks, pred_visibility = motion_cond_predictor(torch.tensor(video),grid_size=3, grid_query_frame=0) 

print(pred_tracks)
print(pred_visibility)