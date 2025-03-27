import numpy as np
import torch
import os
from glob import glob

import pickle
from tqdm import tqdm

# 데이터 경로 설정
base_path = "/data/onebom/data/Cityscapes/leftImg8bit_sequence_trainvaltest/motion_condition2/point_track/train"

# 모든 npy 파일을 찾기
npy_files = sorted(glob(os.path.join(base_path, "*/*/*.npy")))

# 로드된 텐서를 저장할 리스트
sum_mean = None
sum_std = None
count = 0

# 모든 npy 파일 로드
for npy_file in tqdm(npy_files):
    track = np.load(npy_file)  # (C, T, PN) 형태의 npy 파일 로드
    track = torch.tensor(track, dtype=torch.float32)[:2]

    mean = track.mean(dim=(1,2))
    std = track.std(dim=(1,2))

    if sum_mean is None:
        sum_mean = mean
        sum_std = std
    else:
        sum_mean += mean
        sum_std += std
    count += 1

final_mean = sum_mean / count
final_std = sum_std / count
print(f"Final Mean Shape: {final_mean.shape}")  # (C,)
print(f"Final Std Shape: {final_std.shape}")    # (C,)
print(f"Final Mean: {final_mean}")
print(f"Final Std: {final_std}")

data = {"mean":final_mean, "std": final_std}
with open(os.path.join(base_path,"mean_std.pkl"), "wb") as f:
    pickle.dump(data, f)

print("done")

# import os
# import pickle

# root_path = "/data/onebom/data/Cityscapes/leftImg8bit_sequence_trainvaltest/motion_condition2/point_track/train"

# with open(os.path.join(root_path,"mean_std.pkl"), "rb") as f:
#     mean_std = pickle.load(f)

# print(mean_std)