import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    r"""
    torch_npu.npu_dequant_swiglu_quant = npu_dequant_swiglu_quant(
        x, *, weight_scale=None, activation_scale=None, bias=None,
        quant_scale=None, quant_offset=None, group_index=None,
        activate_left=False, quant_mode=0, swiglu_mode=0,
        clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0) -> (Tensor, Tensor)

    Dequantization + SwiGLU + Quantization.

    Equivalent PyTorch native implementation:
    .. code-block:: python
        def dequant_swiglu_quant_native(
            x, weight_scale, activation_scale, bias,
            quant_scale, quant_offset, group_index,
            activate_left, quant_mode, swiglu_mode,
            clamp_limit, glu_alpha, glu_bias
        ):
            # 1. Dequantization (fp32)
            x_f = x.float()
            if weight_scale is not None:
                x_f = x_f * weight_scale.float()
            if activation_scale is not None:
                x_f = x_f * activation_scale.float()
            if bias is not None:
                x_f = x_f + bias.float()

            # 2. SwiGLU
            gate, up = torch.chunk(x_f, 2, dim=-1)
            if not activate_left:
                gate, up = up, gate
            if swiglu_mode == 0:
                output = F.silu(gate) * up
            else:
                gate = torch.clamp(gate, -clamp_limit, clamp_limit)
                up = torch.clamp(up, -clamp_limit, clamp_limit)
                activated = (gate + glu_bias) * torch.sigmoid(glu_alpha * (gate + glu_bias))
                output = activated * up

            # 3. Pre-quant scale
            if quant_scale is not None:
                output = output * quant_scale.float()

            # 4. Quantization
            prefix_dims = 1
            for s in x.shape[:-1]:
                prefix_dims *= s

            if quant_mode == 1:  # Dynamic
                max_val = torch.amax(torch.abs(output), dim=-1)
                scale = max_val / 127.0
                scale = torch.clamp(scale, min=1e-10)
                output = output / scale.unsqueeze(-1)
                output = torch.clamp(output, -128, 127)
                quant = torch.round(output).to(torch.int8)
                return quant, scale.reshape(prefix_dims)
            else:  # Static
                if quant_offset is not None:
                    output = output + quant_offset.float()
                output = torch.clamp(output, -128, 127)
                quant = torch.round(output).to(torch.int8)
                return quant, torch.zeros(prefix_dims, dtype=torch.float32)

    Args:
        x (Tensor): Input tensor, dtype int32, shape [TokensNum, 2H].
        weight_scale (Tensor, optional): Weight dequant scale, dtype float32.
        activation_scale (Tensor, optional): Per-token activation dequant scale, dtype float32.
        bias (Tensor, optional): Bias, dtype int32.
        quant_scale (Tensor, optional): Pre-quant smooth scale, dtype float32.
        quant_offset (Tensor, optional): Quant offset for static mode, dtype float32.
        group_index (Tensor, optional): Group index, dtype int64.
        activate_left (bool, optional): Activate left half. Default: False.
        quant_mode (int, optional): 0=static, 1=dynamic. Default: 0.
        swiglu_mode (int, optional): 0=standard silu, 1=clamp+swish. Default: 0.
        clamp_limit (float, optional): Clamp limit for swiglu_mode=1. Default: 7.0.
        glu_alpha (float, optional): Swish alpha. Default: 1.702.
        glu_bias (float, optional): Swish bias. Default: 1.0.

    Returns:
        tuple: (quantized_output, quant_scales)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, weight_scale: torch.Tensor = None,
                activation_scale: torch.Tensor = None, bias: torch.Tensor = None,
                quant_scale: torch.Tensor = None, quant_offset: torch.Tensor = None,
                group_index: torch.Tensor = None, activate_left: bool = False,
                quant_mode: int = 0, swiglu_mode: int = 0,
                clamp_limit: float = 7.0, glu_alpha: float = 1.702,
                glu_bias: float = 1.0) -> tuple:
        """
        Dequantization + SwiGLU + Quantization (PyTorch native golden reference).

        Args:
            x (torch.Tensor): Input tensor.
            weight_scale (torch.Tensor, optional): Weight scale.
            activation_scale (torch.Tensor, optional): Activation scale.
            bias (torch.Tensor, optional): Bias.
            quant_scale (torch.Tensor, optional): Quant scale.
            quant_offset (torch.Tensor, optional): Quant offset.
            group_index (torch.Tensor, optional): Group index.
            activate_left (bool, optional): Activate left.
            quant_mode (int, optional): Quant mode.
            swiglu_mode (int, optional): SwiGLU mode.
            clamp_limit (float, optional): Clamp limit.
            glu_alpha (float, optional): GLU alpha.
            glu_bias (float, optional): GLU bias.

        Returns:
            tuple: (quantized_output, quant_scales).
        """
        # 1. Dequantization (fp32)
        x_float = x.float()

        if weight_scale is not None:
            x_float = x_float * weight_scale.float()

        if activation_scale is not None:
            x_float = x_float * activation_scale.float()

        if bias is not None:
            x_float = x_float + bias.float()

        # 2. SwiGLU
        gate, up = torch.chunk(x_float, 2, dim=-1)

        if not activate_left:
            gate, up = up, gate

        if swiglu_mode == 0:
            output = F.silu(gate) * up
        elif swiglu_mode == 1:
            gate = torch.clamp(gate, -clamp_limit, clamp_limit)
            up = torch.clamp(up, -clamp_limit, clamp_limit)
            activated = (gate + glu_bias) * torch.sigmoid(glu_alpha * (gate + glu_bias))
            output = activated * up
        else:
            raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}")

        # 3. Pre-quant scale
        if quant_scale is not None:
            output = output * quant_scale.float()

        # 4. Quantization
        prefix_dims = 1
        for s in x.shape[:-1]:
            prefix_dims *= s

        if quant_mode == 1:  # Dynamic
            max_val = torch.amax(torch.abs(output), dim=-1)
            scale = max_val / 127.0
            scale = torch.clamp(scale, min=1e-10)
            output = output / scale.unsqueeze(-1)
            output = torch.clamp(output, -128, 127)
            quantized_output = torch.round(output).to(torch.int8)
            quant_scales = scale.reshape(prefix_dims)
            return quantized_output, quant_scales
        else:  # Static
            if quant_offset is not None:
                output = output + quant_offset.float()
            output = torch.clamp(output, -128, 127)
            quantized_output = torch.round(output).to(torch.int8)
            quant_scales = torch.zeros(prefix_dims, dtype=torch.float32, device=x.device)
            return quantized_output, quant_scales


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), '11_DequantSwigluQuant.json')
    input_groups = []

    fill_map = {
        'inf': float('inf'),
        '-inf': float('-inf'),
        'nan': float('nan'),
        '-nan': -float('nan'),
        'zero': 0.0,
    }

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'int32': torch.int32,
        'int64': torch.int64,
        'int8': torch.int8,
        'bool': torch.bool,
    }

    with open(json_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            attrs = {}

            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    fill = inp.get('fill', None)

                    if shape is None:
                        tensors[name] = None
                        continue

                    # Priority 1: fill special values
                    if fill in fill_map:
                        fill_val = fill_map[fill]
                        if dtype_str == 'bool':
                            tensors[name] = torch.full(shape, fill_val != 0.0, dtype=torch.bool)
                        elif dtype_str in ('int32', 'int64', 'int8'):
                            dtype = dtype_map.get(dtype_str, torch.int32)
                            if fill_val in (float('inf'), float('-inf'), float('nan')):
                                tensors[name] = torch.zeros(shape, dtype=dtype)
                            else:
                                tensors[name] = torch.full(shape, int(fill_val), dtype=dtype)
                        else:
                            dtype = dtype_map.get(dtype_str, torch.float32)
                            tensors[name] = torch.full(shape, fill_val, dtype=dtype)
                        continue

                    # Priority 2: bool
                    if dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                        continue

                    # Priority 3: integer (including x which is int32)
                    if dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = dtype_map.get(dtype_str, torch.int32)
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                        continue

                    # Priority 4: float (50% normal, 50% uniform [-5, 5])
                    dtype = dtype_map.get(dtype_str, torch.float32)
                    if idx % 2 == 0:
                        mu = float(torch.empty(1).uniform_(-100, 100).item())
                        sigma = float(torch.empty(1).uniform_(1, 25).item())
                        raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
                        tensors[name] = raw.to(dtype)
                    else:
                        raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
                        tensors[name] = raw.to(dtype)

                elif inp['type'] == 'attr':
                    attrs[inp['name']] = inp['value']

            # Adjust shapes and values for dequant/quant params
            x_shape = None
            x_dtype = None
            for inp in inputs:
                if inp['name'] == 'x' and inp['type'] == 'tensor':
                    x_shape = inp.get('shape')
                    x_dtype = inp.get('dtype', 'int32')
                    break

            if x_shape is not None:
                tokens_num = x_shape[0]
                hidden_size = x_shape[1]
                half_h = hidden_size // 2

                quant_mode = attrs.get('quant_mode', 0)
                group_index_val = tensors.get('group_index')
                has_group = group_index_val is not None

                if has_group:
                    num_groups = group_index_val.shape[0]
                    if quant_mode != 1:
                        quant_mode = 1
                    tensors['bias'] = None
                    tensors['quant_offset'] = None
                    # count mode: each group has tokens_num // num_groups
                    tensors['group_index'] = torch.tensor([tokens_num // num_groups] * num_groups, dtype=torch.int64)
                else:
                    if x_dtype == 'int32':
                        if tensors.get('weight_scale') is not None:
                            ws_shape = tensors['weight_scale'].shape
                            tensors['weight_scale'] = _make_tensor(ws_shape, torch.float32, idx)
                        else:
                            tensors['weight_scale'] = torch.randn((1, hidden_size), dtype=torch.float32)
                        if tensors.get('activation_scale') is not None:
                            as_shape = tensors['activation_scale'].shape
                            tensors['activation_scale'] = _make_tensor(as_shape, torch.float32, idx)
                        else:
                            tensors['activation_scale'] = torch.randn((tokens_num, 1), dtype=torch.float32)
                    else:
                        tensors['weight_scale'] = None
                        tensors['activation_scale'] = None
                        tensors['bias'] = None

                if tensors.get('quant_scale') is not None:
                    qs_shape = tensors['quant_scale'].shape
                    num_groups_qs = qs_shape[0] if len(qs_shape) > 0 else 1
                    tensors['quant_scale'] = torch.randn(num_groups_qs, half_h, dtype=torch.float32)

                if has_group:
                    tensors['quant_offset'] = None
                elif tensors.get('quant_offset') is not None and quant_mode == 0:
                    qs_tensor = tensors.get('quant_scale')
                    if qs_tensor is not None:
                        tensors['quant_offset'] = torch.randn(qs_tensor.shape, dtype=torch.float32)
                    else:
                        tensors['quant_offset'] = None
                else:
                    tensors['quant_offset'] = None

            group = [
                tensors['x'],
                tensors.get('weight_scale'),
                tensors.get('activation_scale'),
                tensors.get('bias'),
                tensors.get('quant_scale'),
                tensors.get('quant_offset'),
                tensors.get('group_index'),
                attrs.get('activate_left', False),
                quant_mode,
                attrs.get('swiglu_mode', 0),
                attrs.get('clamp_limit', 7.0),
                attrs.get('glu_alpha', 1.702),
                attrs.get('glu_bias', 1.0),
            ]
            input_groups.append(group)

    return input_groups


def _make_tensor(shape, dtype, idx):
    """Helper to generate tensor with proper distribution."""
    fill_map = {
        'inf': float('inf'),
        '-inf': float('-inf'),
        'nan': float('nan'),
        '-nan': -float('nan'),
        'zero': 0.0,
    }
    if idx % 2 == 0:
        mu = float(torch.empty(1).uniform_(-100, 100).item())
        sigma = float(torch.empty(1).uniform_(1, 25).item())
        raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
        return raw.to(dtype)
    else:
        raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
        return raw.to(dtype)


def get_init_inputs():
    return []