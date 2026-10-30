import torch
import torch.nn as nn
import json
import os
import torch_npu

torch_npu.npu.conv.allow_hf32 = False


class Model(nn.Module):
    r"""
    Simple model that performs transpose 2D convolution.
        torch.nn.ConvTranspose2d = class ConvTranspose2d(_ConvTransposeNd)
        torch.nn.ConvTranspose2d(
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            output_padding: _size_2_t = 0,
            groups: int = 1,
            bias: bool = True,
            dilation: _size_2_t = 1,
            padding_mode: Literal['zeros', 'reflect', 'replicate', 'circular'] = 'zeros',
            device=None,
            dtype=None
        ) -> None

        Applies a 2D transposed convolution operator over an input image
        composed of several input planes.

        This module can be seen as the gradient of Conv2d with respect to its input.
        It is also known as a fractionally-strided convolution or
        a deconvolution (although it is not an actual deconvolution operation as it does
        not compute a true inverse of convolution). For more information, see the visualizations
        `here`_ and the `Deconvolutional Networks`_ paper.

        This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

        On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

        * :attr:`stride` controls the stride for the cross-correlation. When stride > 1, ConvTranspose2d inserts zeros between input
        elements along the spatial dimensions before applying the convolution kernel. This zero-insertion operation is the standard
        behavior of transposed convolutions, which can increase the spatial resolution and is equivalent to a learnable
        upsampling operation.

        * :attr:`padding` controls the amount of implicit zero padding on both
        sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
        below for details.

        * :attr:`output_padding` controls the additional size added to one side
        of the output shape. See note below for details.

        * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
        It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

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

        The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
        can either be:

            - a single ``int`` -- in which case the same value is used for the height and width dimensions
            - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
            and the second `int` for the width dimension

        Note:
            The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
            amount of zero padding to both sizes of the input. This is set so that
            when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`
            are initialized with same parameters, they are inverses of each other in
            regard to the input and output shapes. However, when ``stride > 1``,
            :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output
            shape. :attr:`output_padding` is provided to resolve this ambiguity by
            effectively increasing the calculated output shape on one side. Note
            that :attr:`output_padding` is only used to find output shape, but does
            not actually add zero-padding to output.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding
                will be added to both sides of each dimension in the input. Default: 0
            output_padding (int or tuple, optional): Additional size added to one side
                of each dimension in the output shape. Default: 0
            groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1

        Shape:
            - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`
            - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where

            .. math::
                    H_{out} = (H_{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                            \times (\text{kernel\_size}[0] - 1) + \text{output\_padding}[0] + 1
            .. math::
                    W_{out} = (W_{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                            \times (\text{kernel\_size}[1] - 1) + \text{output\_padding}[1] + 1

        Attributes:
            weight (Tensor): the learnable weights of the module of shape
                            :math:`(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}},`
                            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                            The values of these weights are sampled from
                            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                            :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
            bias (Tensor):   the learnable bias of the module of shape (out_channels)
                            If :attr:`bias` is ``True``, then the values of these weights are
                            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                            :math:`k = \frac{groups}{C_\text{out} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

        Examples::

            >>> # With square kernels and equal stride
            >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
            >>> # non-square kernels and unequal stride and with padding
            >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
            >>> input = torch.randn(20, 16, 50, 100)
            >>> output = m(input)
            >>> # exact output size can be also specified as an argument
            >>> input = torch.randn(1, 16, 12, 12)
            >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
            >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
            >>> h = downsample(input)
            >>> h.size()
            torch.Size([1, 16, 6, 6])
            >>> output = upsample(h, output_size=input.size())
            >>> output.size()
            torch.Size([1, 16, 12, 12])

        Torch 原生小算子拼接实现（等价于 nn.ConvTranspose2d）：
        方式一：函数式 API
            import torch.nn.functional as F
            output = F.conv_transpose2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=self.conv.stride,
                padding=self.conv.padding,
                output_padding=self.conv.output_padding,
                groups=self.conv.groups,
                dilation=self.conv.dilation
            )
        方式二：插零 + 标准卷积（更底层的小算子拼接）
            import torch.nn.functional as F
            # 1. 在输入空间维度之间插入 (stride-1) 个零
            if any(s > 1 for s in self.conv.stride):
                x = F.interpolate(x, scale_factor=self.conv.stride, mode='nearest')
            # 2. 计算等效 padding: dilation * (kernel_size - 1) - padding
            pad_h = self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - self.conv.padding[0]
            pad_w = self.conv.dilation[1] * (self.conv.kernel_size[1] - 1) - self.conv.padding[1]
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
            # 3. 标准卷积
            output = F.conv2d(
                x,
                self.conv.weight,
                self.conv.bias,
                stride=1,
                padding=0,
                groups=self.conv.groups,
                dilation=self.conv.dilation
            )
            # 4. output_padding 处理（在输出一侧裁剪或补充）
            if any(op > 0 for op in self.conv.output_padding):
                output = output[:, :, :output.shape[2]-self.conv.output_padding[0] or None,
                                    :output.shape[3]-self.conv.output_padding[1] or None]
    """

    def __init__(self):
        super(Model, self).__init__()
        self._convs = {}
        self._rng_states = {}

    def forward(self, inputs) -> torch.Tensor:
        """
        Args:
            inputs (tuple): (x, in_channels, out_channels, kernel_size, stride, padding, bias)
        """
        x, in_channels, out_channels, kernel_size, stride, padding, bias = inputs

        key = (in_channels, out_channels, kernel_size, stride, padding, bias)
        conv = self._convs.get(key)

        if conv is None:
            rng_state = torch.get_rng_state()
            torch.manual_seed(hash(key) & 0xFFFFFFFF)
            conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
            torch.set_rng_state(rng_state)
            conv = conv.to(x.device)
            self._convs[key] = conv

        return conv(x)


def _load_cases():
    json_path = os.path.join(os.path.dirname(__file__), "10_ConvTranspose2d.json")
    with open(json_path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_init_inputs():
    return []


def get_input_groups():
    """
    每组返回一个单元素列表: [(x, in_channels, out_channels, kernel_size, stride, padding, bias)]
    validate_task.py 调用 model(*inputs) 即 model( (x, ...) )，forward 只接收一个参数。
    """
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
        bias = attr_inputs.get("bias", True)
        input_groups.append([(x, in_channels, out_channels, kernel_size, stride, padding, bias)])

    return input_groups