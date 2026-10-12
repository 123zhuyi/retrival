"""P2.5 simulator 采集用例 — grid_sample 2D bilinear 主线（fp16, C=32, H=W=64）"""
import sys, torch, torch_npu
sys.path.insert(0, "/home/w00836812/shixi_test02/zmm_source_retrieval/ascendc_to_okf_workflow/output/grid_sample")
from triton_skeleton import grid_sample_triton
torch.manual_seed(0)
dev = torch.device("npu")
x = (torch.rand(1, 32, 64, 64, device=dev) * 2 - 1).to(torch.float16)
g = torch.rand(1, 64, 64, 2, device=dev) * 2.4 - 1.2
out = grid_sample_triton(x, g, "bilinear", "zeros", False)
torch.npu.synchronize()
print("DONE")
