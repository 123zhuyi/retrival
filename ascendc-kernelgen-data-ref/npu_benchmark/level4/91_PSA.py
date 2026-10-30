import torch, torch_npu
torch.npu.conv.allow_hf32 = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os


class Model(nn.Module):
    """
    Pyramid Squeeze Attention (PSA) Module.

    Splits channels into S groups, applies convolutions of increasing kernel
    sizes (SPC module), then uses SE-style attention with softmax (SPA module)
    to re-weight each group before recombining.
    """

    def __init__(self):
        super(Model, self).__init__()
        self._cache = {}

    def forward(self, x: torch.Tensor, channel: int, reduction: int = 4, S: int = 4) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            channel: Number of input channels.
            reduction: Reduction factor for SE blocks.
            S: Number of groups (also number of parallel convolutions).

        Returns:
            Attention-refined tensor of same shape as input.
        """
        torch.manual_seed(42)
        b, c, h, w = x.shape
        # [修复1] 缓存键带上 device/dtype：同一 (channel,reduction,S) 在不同设备/精度下
        # 各自缓存一份已搬移好的层；权重种子仍只由 (channel,reduction,S) 决定，
        # 因此不同 dtype 下权重数值一致（只是精度转换）
        key = (channel, reduction, S, x.device, x.dtype)
        seed_key = (channel, reduction, S)
        if key not in self._cache:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(seed_key) & 0xFFFFFFFF)

            ci = channel // S
            # Build convs: each with kernel_size = 2*(i+1)+1, padding = i+1
            convs = nn.ModuleList()
            for i in range(S):
                ks = 2 * (i + 1) + 1
                pad = i + 1
                convs.append(nn.Conv2d(ci, ci, kernel_size=ks, padding=pad))

            # Build SE blocks
            se_blocks = nn.ModuleList()
            for _ in range(S):
                se_blocks.append(nn.Sequential(
                    nn.Conv2d(ci, ci // reduction, kernel_size=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ci // reduction, ci, kernel_size=1, bias=False),
                    nn.Sigmoid()
                ))

            # [修复1] 创建后立即搬到与 x 相同的 device/dtype，再入缓存
            convs = convs.to(device=x.device, dtype=x.dtype)
            se_blocks = se_blocks.to(device=x.device, dtype=x.dtype)
            self._cache[key] = (convs, se_blocks)
            torch.set_rng_state(rng_state)

        convs, se_blocks = self._cache[key]

        # Step1: SPC module
        # [修复2] view 后加 clone，切断与输入 x 的共享存储，
        # 否则下面的 slice 赋值会原地篡改调用方的输入张量
        SPC_out = x.view(b, S, -1, h, w).clone()  # (b, S, ci, h, w), ci = channel // S
        for idx, conv in enumerate(convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step2: SE weight for each group
        se_outs = []
        for idx, se in enumerate(se_blocks):
            # Squeeze: global avg pool, then MLP, then sigmoid
            group = SPC_out[:, idx, :, :, :]          # (b, ci, h, w)
            pooled = F.adaptive_avg_pool2d(group, 1)  # (b, ci, 1, 1)
            se_out = se(pooled)                       # (b, ci, 1, 1)
            se_outs.append(se_out)
        SE_out = torch.stack(se_outs, dim=1)          # (b, S, ci, 1, 1)
        SE_out = SE_out.expand_as(SPC_out)            # (b, S, ci, h, w)

        # Step3: Softmax across groups (dim=1)
        softmax_out = F.softmax(SE_out, dim=1)        # (b, S, ci, h, w)

        # Step4: SPA (weighted sum)
        PSA_out = SPC_out * softmax_out               # (b, S, ci, h, w)
        PSA_out = PSA_out.view(b, -1, h, w)           # (b, c, h, w)
        return PSA_out


def get_input_groups():
    torch.manual_seed(42)
    json_path = os.path.join(os.path.dirname(__file__), "91_PSA.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    def random_tensor(shape, dtype):
        if torch.rand(1).item() < 0.5:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            return torch.normal(mu, sigma, shape, dtype=dtype)
        else:
            return torch.empty(shape, dtype=dtype).uniform_(-5.0, 5.0)

    input_groups = []
    for case in cases:
        inputs = case["inputs"]
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        x_info = next(inp for inp in inputs if inp["name"] == "x")
        dtype = dtype_map[x_info["dtype"]]
        x = random_tensor(x_info["shape"], dtype)

        channel = next(inp for inp in inputs if inp["name"] == "channel")["value"]
        reduction = next((inp["value"] for inp in inputs if inp["name"] == "reduction"), 4)
        S = next((inp["value"] for inp in inputs if inp["name"] == "S"), 4)

        input_groups.append([x, channel, reduction, S])
    return input_groups


def get_init_inputs():
    return []