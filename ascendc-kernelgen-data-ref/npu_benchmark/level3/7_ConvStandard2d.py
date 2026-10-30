import torch
import torch.nn as nn
import json
import os
import torch_npu

torch_npu.npu.conv.allow_hf32 = False


class Model(nn.Module):
    r"""
    Simple model that performs standard 2D convolution.
        torch.nn.Conv2d = class Conv2d(_ConvNd)
        torch.nn.Conv2d(
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: str | _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
            device=None,
            dtype=None
        ) -> None

        Applies a 2D convolution over an input signal composed of several input
        planes.

        In the simplest case, the output value of the layer with input size
        :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
        can be precisely described as:

        .. math::
            \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
            \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

        where :math:`\star` is the valid 2D `cross-correlation`_ operator,
        :math:`N` is a batch size, :math:`C_{\text{in}}` and :math:`C_{\text{out}}` correspond to
        :attr:`in_channels` and :attr:`out_channels` respectively,
        :math:`H` and :math:`W` are the input height and width in pixels.
        See the Shape section below for how :math:`H_{\text{out}}` and :math:`W_{\text{out}}`
        are derived from :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, and :attr:`dilation`.

        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

        On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

        * :attr:`stride` controls the stride for the cross-correlation, a single
        number or a tuple.

        * :attr:`padding` controls the amount of padding applied to the input. It
        can be either a string {'valid', 'same'} or an int / a tuple of ints giving the
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

        The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

            - a single ``int`` -- in which case the same value is used for the height and width dimension
            - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
            and the second `int` for the width dimension

        Note:
            When `groups == in_channels` and `out_channels == K * in_channels`,
            where `K` is a positive integer, this operation is also known as a "depthwise convolution".

            In other words, for an input of size :math:`(N, C_{in}, H, W)`,
            a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
            :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

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
            padding (int, tuple or str, optional): Padding added to all four sides of
                the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Default: ``True``
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

            .. math::
                H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                            \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

            .. math::
                W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                            \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                The values of these weights are sampled from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
            bias (Tensor):   the learnable bias of the module of shape
                (out_channels). If :attr:`bias` is ``True``,
                then the values of these weights are
                sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

        Examples:

            >>> # With square kernels and equal stride
            >>> m = nn.Conv2d(16, 33, 3, stride=2)
            >>> # non-square kernels and unequal stride and with padding
            >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
            >>> # non-square kernels and unequal stride and with padding and dilation
            >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
            >>> input = torch.randn(20, 16, 50, 100)
            >>> output = m(input)

        Torch 原生小算子拼接实现（等价于 nn.Conv2d）：
        方式一：函数式 API
            import torch.nn.functional as F
            output = F.conv2d(
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
            batch_size, C_in, H, W = x.shape
            C_out = self.conv.out_channels
            kh, kw = self.conv.kernel_size
            sh, sw = self.conv.stride
            ph, pw = self.conv.padding
            dh, dw = self.conv.dilation

            # 1. 对四维输入做 padding
            x_pad = F.pad(x, (pw, pw, ph, ph))

            # 2. 使用 tensor.unfold 在 H/W 维度分别提取滑动窗口
            x_unfold = x_pad.unfold(2, kh, sh)  # (batch, C_in, H_out, W_pad, kh)
            x_unfold = x_unfold.unfold(3, kw, sw)  # (batch, C_in, H_out, W_out, kh, kw)

            # 3. reshape 为 (batch, C_in, H_out*W_out, kh*kw)
            x_unfold = x_unfold.contiguous().view(batch_size, C_in, -1, kh * kw)

            # 4. weight reshape: (C_out, C_in, kh, kw) -> (C_out, C_in, kh*kw)
            weight = self.conv.weight.view(C_out, C_in, kh * kw)

            # 5. 矩阵乘法: (batch, C_in, N, K) @ (C_out, C_in, K) -> (batch, C_out, N)
            output = torch.einsum('bink,oik->bon', x_unfold, weight)

            # 6. reshape 回空间维度
            H_out = (H + 2*ph - dh*(kh-1) - 1) // sh + 1
            W_out = (W + 2*pw - dw*(kw-1) - 1) // sw + 1
            output = output.view(batch_size, C_out, H_out, W_out)

            # 7. 加 bias
            if self.conv.bias is not None:
                output = output + self.conv.bias.view(1, C_out, 1, 1)
    """

    def __init__(self):
        super(Model, self).__init__()
        self._convs = {}

    def forward(self, inputs) -> torch.Tensor:
        """
        Applies standard 2D convolution to the input tensor.

        Args:
            inputs (tuple): (x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        Returns:
            torch.Tensor: Output tensor after performing nn.Conv2d.
        """
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        conv = self._convs.get(key)
        if conv is None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
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
    json_path = os.path.join(os.path.dirname(__file__), "7_ConvStandard2d.json")
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