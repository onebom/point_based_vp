import numpy as np
path = "/data/onebom/data/Cityscapes/leftImg8bit_sequence_trainvaltest/motion_condition2/point_track/train/aachen/aachen_000000_000000_leftImg8bit/from_000000.npy"

array = np.load(path)
print(array.shape)