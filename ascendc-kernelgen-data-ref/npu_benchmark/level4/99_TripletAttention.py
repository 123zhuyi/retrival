import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [_to_device(item, device) for item in obj]
    return obj


class Model(nn.Module):
    """
    Triplet Attention Module.

    三个分支分别捕获 C-H、C-W、H-W 维度的交叉交互；
    每个分支：ZPool -> Conv2d(2,1) -> BN(batch 统计) -> ReLU -> Sigmoid -> 乘回输入。
    所有 Conv/BN 在模块定义内，权重传入 forward。

    说明：BN 显式固定 training=True（batch 统计），等价于原始 nn 模块
    默认状态（构造即 training=True）直接运行的行为，且不受外部 .eval() 影响。

    输入约束（由 case 保证）：
      x: (B, C, H, W)
      kernel_size: 奇数
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, kernel_size, gate_weights):
        gate_weights = _to_device(gate_weights, x.device)
        (conv_ch_w, conv_ch_b, bn_ch_w, bn_ch_b), \
        (conv_cw_w, conv_cw_b, bn_cw_w, bn_cw_b), \
        (conv_hw_w, conv_hw_b, bn_hw_w, bn_hw_b) = gate_weights

        padding = (kernel_size - 1) // 2

        def gate_fn(t, conv_w, conv_b, bn_w, bn_b, permute=None, permute_back=None):
            if permute is not None:
                t = t.permute(permute)
            z = torch.cat([t.mean(dim=1, keepdim=True),
                           t.max(dim=1, keepdim=True)[0]], dim=1)
            y = F.conv2d(z, conv_w, conv_b, padding=padding)
            y = F.batch_norm(
                y, torch.zeros_like(bn_w), torch.ones_like(bn_w),
                bn_w, bn_b, training=True, momentum=0.1, eps=1e-5
            )
            y = F.relu(y, inplace=True)
            attn = torch.sigmoid(y)
            out = t * attn
            if permute_back is not None:
                out = out.permute(permute_back)
            return out

        x_ch = gate_fn(
            x, conv_ch_w, conv_ch_b, bn_ch_w, bn_ch_b,
            permute=(0, 3, 1, 2), permute_back=(0, 2, 3, 1)
        )
        x_cw = gate_fn(
            x, conv_cw_w, conv_cw_b, bn_cw_w, bn_cw_b,
            permute=(0, 2, 1, 3), permute_back=(0, 2, 1, 3)
        )
        x_hw = gate_fn(x, conv_hw_w, conv_hw_b, bn_hw_w, bn_hw_b)

        return (x_ch + x_cw + x_hw) / 3.0


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _random_tensor(shape, dtype, seed):
    # 修复：所有随机调用都挂传入的 generator，per-case 种子真正生效、case 间解耦
    g = torch.Generator()
    g.manual_seed(seed)
    if torch.rand(1, generator=g).item() < 0.5:
        mu = float(torch.empty(1).uniform_(-5.0, 5.0, generator=g).item())
        sigma = float(torch.empty(1).uniform_(0.1, 2.0, generator=g).item())
        return torch.normal(mu, sigma, shape, dtype=torch.float32, generator=g).to(dtype)
    else:
        return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0, generator=g)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "99_TripletAttention.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    input_groups = []
    for case_index, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = _DTYPE_MAP[x_info["dtype"]]
        x = _random_tensor(x_info["shape"], dtype, 42 + case_index)

        kernel_size = next(
            (inp["value"] for inp in inputs if inp["name"] == "kernel_size"), 7
        )

        torch.manual_seed(1000 + case_index)
        gate_weights = []
        for _ in range(3):
            conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1,
                             padding=(kernel_size - 1) // 2)
            bn = nn.BatchNorm2d(1)
            bn.weight.data.mul_(1.0 + 0.1 * torch.randn(1))
            bn.bias.data.copy_(0.1 * torch.randn(1))
            gate_weights.append((
                conv.weight.detach().to(dtype),
                conv.bias.detach().to(dtype),
                bn.weight.detach().to(dtype),
                bn.bias.detach().to(dtype),
            ))

        input_groups.append([x, kernel_size, gate_weights])
    return input_groups


def get_init_inputs():
    return []