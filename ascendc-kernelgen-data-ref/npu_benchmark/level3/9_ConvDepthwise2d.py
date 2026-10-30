import torch
import torch.nn as nn
import json
import os
import torch_npu

torch_npu.npu.conv.allow_hf32 = False


class Model(nn.Module):
    """
    Simple model that performs depthwise 2D convolution.

    Torch 原生小算子拼接实现（等价于 nn.Conv2d with groups=in_channels）：
    方式一：函数式 API
        import torch.nn.functional as F
        output = F.conv2d(
            x,
            self.conv.weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups  # groups = in_channels
        )
    方式二：unfold + matmul 显式小算子拼接
        import torch.nn.functional as F
        batch_size, C, H, W = x.shape
        kh, kw = self.conv.kernel_size
        # 1. 提取滑动窗口
        x_unfold = F.unfold(
            x,
            kernel_size=(kh, kw),
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation
        )  # shape: (batch, C * kh * kw, H_out * W_out)
        # 2. reshape 为 (batch, C, kh*kw, H_out*W_out)
        x_unfold = x_unfold.view(batch_size, C, kh * kw, -1)
        # 3. weight reshape 为 (C, kh*kw)
        weight = self.conv.weight.view(C, kh * kw)
        # 4. 逐通道矩阵乘法: (batch, C, kh*kw, N) @ (C, kh*kw) -> (batch, C, N)
        output = torch.einsum('bckn,ck->bcn', x_unfold, weight)
        # 5. reshape 回空间维度
        H_out = (H + 2 * self.conv.padding[0] - self.conv.dilation[0] * (kh - 1) - 1) // self.conv.stride[0] + 1
        W_out = (W + 2 * self.conv.padding[1] - self.conv.dilation[1] * (kw - 1) - 1) // self.conv.stride[1] + 1
        output = output.view(batch_size, C, H_out, W_out)
        # 6. 加 bias
        if self.conv.bias is not None:
            output = output + self.conv.bias.view(1, C, 1, 1)
    """

    def __init__(self):
        super(Model, self).__init__()
        self._convs = {}

    def forward(self, inputs) -> torch.Tensor:
        """
        Applies depthwise 2D convolution to the input tensor.

        Args:
            inputs (tuple): (x, in_channels, kernel_size, stride, padding, bias)

        Returns:
            torch.Tensor: Output tensor after performing depthwise nn.Conv2d.
        """
        x, in_channels, kernel_size, stride, padding, bias = inputs

        key = (in_channels, kernel_size, stride, padding, bias)
        conv = self._convs.get(key)
        if conv is None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=bias
            )
            torch.set_rng_state(rng_state)
            conv = conv.to(x.device)
            self._convs[key] = conv

        return conv(x)


def _load_cases():
    json_path = os.path.join(os.path.dirname(__file__), "9_ConvDepthwise2d.json")
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_init_inputs():
    return []


def get_input_groups():
    cases = _load_cases()
    input_groups = []
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        tensor_inputs = [inp for inp in inputs if inp.get("type") == "tensor"]
        attr_inputs = {inp["name"]: inp["value"] for inp in inputs if inp.get("type") == "attr"}

        x_info = tensor_inputs[0]
        dtype = dtype_map.get(x_info["dtype"], torch.float32)

        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            sigma = float(torch.empty(1).uniform_(0.1, 2.0).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5.0, 5.0)

        in_channels = attr_inputs.get("in_channels")
        kernel_size = attr_inputs.get("kernel_size")
        stride = attr_inputs.get("stride", 1)
        padding = attr_inputs.get("padding", 0)
        bias = attr_inputs.get("bias", True)

        input_groups.append([(x, in_channels, kernel_size, stride, padding, bias)])

    return input_groups