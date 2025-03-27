import torch
from accelerate import Accelerator
# from model.motionPredictor import TrackMotionModel

from safetensors.torch import load_file

# track_predictor = TrackMotionModel(cfg.model).to(accelerator.device)

checkpoint_path = "/data/onebom/result/ongoing/Trajectory_Prediction/ver2_13_attn_gruE_odeD_adaptiveTrack/checkpoints/vdm_steps_16000/model.safetensors"  # 실제 경로로 바꿔주세요
# checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = load_file(checkpoint_path)

print(state_dict.keys)