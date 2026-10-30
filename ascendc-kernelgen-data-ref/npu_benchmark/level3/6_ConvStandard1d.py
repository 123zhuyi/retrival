import torch
import torch.nn as nn
import json
import os
import torch_npu

torch_npu.npu.conv.allow_hf32 = False


class Model(nn.Module):
    r"""
    Simple model that performs standard 1D convolution.
        torch.nn.Conv1d = class Conv1d(_ConvNd)
        torch.nn.Conv1d(
            in_channels: int,
            out_channels: int,
            kernel_size: _size_1_t,
            stride: _size_1_t = 1,
            padding: str | _size_1_t = 0,
            dilation: _size_1_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
            device=None,
            dtype=None
        ) -> None

        Applies a 1D convolution over an input signal composed of several input
        planes.

        In the simplest case, the output value of the layer with input size
        :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
        precisely described as:

        .. math::
            \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
            \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
            \star \text{input}(N_i, k)

        where :math:`\star` is the valid `cross-correlation`_ operator,
        :math:`N` is a batch size, :math:`C` denotes a number of channels,
        :math:`L` is a length of signal sequence.

        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

        On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

        * :attr:`stride` controls the stride for the cross-correlation, a single
        number or a one-element tuple.

        * :attr:`padding` controls the amount of padding applied to the input. It
        can be either a string {'valid', 'same'} or a tuple of ints giving the
        amount of implicit padding applied on both sides.

        * :attr:`dilation` controls the spacing between the kernel points; also
        known as the à trous algorithm. It is harder to describe, but this `link`_
        has a nice visualization of what :attr:`dilation` does.

        * :attr:`groups` controls the connections between inputs and outputs.
        :attr:`in_channels` and :attr:`out_channels` must both be divisible by
        :attr:`groups`. For example,

            * At groups=1, all inputs are convolved to all outputs.
            * At groups=2, the operation becomes equivalent to having two conv
            layers side by side, each seeing half the input channels
            and producing half the output channels, and both subsequently
            concatenated.
            * At groups= :attr:`in_channels`, each input channel is convolved with
            its own set of filters (of size
            :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).

        Note:
            When `groups == in_channels` and `out_channels == K * in_channels`,
            where `K` is a positive integer, this operation is also known as a "depthwise convolution".

            In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
            a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
            :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
        Note:

        Note:
            ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
            the input so the output has the shape as the input. However, this mode
            doesn't support any stride values other than 1.

        Note:
            This module supports complex data types i.e. ``complex32, complex64, complex128``.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to both sides of
                the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`
            - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where

            .. math::
                L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                            \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                :math:`(\text{out\_channels},
                \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
                The values of these weights are sampled from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
            bias (Tensor):   the learnable bias of the module of shape
                (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
                sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`

        Examples::

            >>> m = nn.Conv1d(16, 33, 3, stride=2)
            >>> input = torch.randn(20, 16, 50)
            >>> output = m(input)

        Torch 原生小算子拼接实现（等价于 nn.Conv1d）：
        方式一：函数式 API
            import torch.nn.functional as F
            output = F.conv1d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                dilation=self.conv.dilation,
                groups=self.conv.groups
            )
        方式二：unfold + matmul 显式小算子拼接（groups=1 时）
            import torch.nn.functional as F
            batch_size, C_in, L = x.shape
            C_out = self.conv.out_channels
            k = self.conv.kernel_size[0]
            s = self.conv.stride[0]
            p = self.conv.padding[0]
            d = self.conv.dilation[0]

            # 1. 对 L 维度做 padding
            x_pad = F.pad(x, (p, p))

            # 2. 使用 tensor.unfold 提取滑动窗口
            x_unfold = x_pad.unfold(2, k, s)  # (batch, C_in, L_out, k)

            # 3. weight reshape: (C_out, C_in, k) -> (C_out, C_in, k)
            weight = self.conv.weight

            # 4. 矩阵乘法: (batch, C_in, L_out, k) 与 (C_out, C_in, k) 做 einsum
            #    输出 (batch, C_out, L_out)
            output = torch.einsum('bilk,oik->bol', x_unfold, weight)

            # 5. 加 bias
            if self.conv.bias is not None:
                output = output + self.conv.bias.view(1, C_out, 1)
    """

    def __init__(self):
        super(Model, self).__init__()
        self._convs = {}

    def forward(self, inputs) -> torch.Tensor:
        """
        Applies standard 1D convolution to the input tensor.

        Args:
            inputs (tuple): (x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        Returns:
            torch.Tensor: Output tensor after performing nn.Conv1d.
        """
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        conv = self._convs.get(key)
        if conv is None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
            torch.set_rng_state(rng_state)
            conv = conv.to(x.device)
            self._convs[key] = conv

        return conv(x)


def _load_cases():
    json_path = os.path.join(os.path.dirname(__file__), "6_ConvStandard1d.json")
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
        out_channels = attr_inputs.get("out_channels")
        kernel_size = attr_inputs.get("kernel_size")
        stride = attr_inputs.get("stride", 1)
        padding = attr_inputs.get("padding", 0)
        dilation = attr_inputs.get("dilation", 1)
        groups = attr_inputs.get("groups", 1)
        bias = attr_inputs.get("bias", True)

        input_groups.append([(x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)])

    return input_groups