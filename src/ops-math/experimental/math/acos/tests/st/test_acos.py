import torch, torch_npu

a = torch.rand(12801, 10241, dtype=torch.float32).npu()
a_cpu = a.cpu()
res_npu = torch.acos(a)
res_cpu = torch.acos(a_cpu)
print("fp32 result: ", torch.allclose(res_npu.cpu(), res_cpu, 1e-4, 1e-4))

res_npu_fp16 = torch.acos(a.to(torch.float16))
res_cpu_fp16 = torch.acos(a_cpu.to(torch.float16))
print("fp16 result: ", torch.allclose(res_npu_fp16.cpu(), res_cpu_fp16, 1e-3,1e-3))

res_npu_bf16 = torch.acos(a.to(torch.bfloat16))
res_cpu_bf16 = torch.acos(a_cpu.to(torch.bfloat16))
print("bf16 result: ", torch.allclose(res_npu_bf16.cpu(), res_cpu_bf16, 4e-3,4e-3))