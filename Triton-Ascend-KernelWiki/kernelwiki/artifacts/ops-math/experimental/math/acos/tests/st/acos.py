import torch, torch_npu
a = torch.rand(100, 10000, dtype=torch.float32).npu()
torch.acos(a)