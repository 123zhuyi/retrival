import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    r"""
        Simple model that performs padding on a tensor.
        torch.nn.functional.pad = pad(
        input: Tensor,
        pad: list[int],
        mode: str = 'constant',
        value: float | None = None
    ) -> Tensor
        pad(input, pad, mode="constant", value=None) -> Tensor

        Pads tensor.

        Padding size:
            The padding size by which to pad some dimensions of :attr:`input`
            are described starting from the last dimension and moving forward.
            :math:`\left\lfloor\frac{\text{len(pad)}}{2}\right\rfloor` dimensions
            of ``input`` will be padded.
            For example, to pad only the last dimension of the input tensor, then
            :attr:`pad` has the form
            :math:`(\text{padding\_left}, \text{padding\_right})`;
            to pad the last 2 dimensions of the input tensor, then use
            :math:`(\text{padding\_left}, \text{padding\_right},`
            :math:`\text{padding\_top}, \text{padding\_bottom})`;
            to pad the last 3 dimensions, use
            :math:`(\text{padding\_left}, \text{padding\_right},`
            :math:`\text{padding\_top}, \text{padding\_bottom}`
            :math:`\text{padding\_front}, \text{padding\_back})`.

        Padding mode:
            See :class:`torch.nn.CircularPad2d`, :class:`torch.nn.ConstantPad2d`,
            :class:`torch.nn.ReflectionPad2d`, and :class:`torch.nn.ReplicationPad2d`
            for concrete examples on how each of the padding modes works. Constant
            padding is implemented for arbitrary dimensions. Circular, replicate and
            reflection padding are implemented for padding the last 3 dimensions of a
            4D or 5D input tensor, the last 2 dimensions of a 3D or 4D input tensor,
            or the last dimension of a 2D or 3D input tensor.

        Args:
            input (Tensor): N-dimensional tensor
            pad (tuple): m-elements tuple, where
                :math:`\frac{m}{2} \leq` input dimensions and :math:`m` is even.
            mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'constant'``
            value: fill value for ``'constant'`` padding. Default: ``0``

        Examples::

            >>> t4d = torch.empty(3, 3, 4, 2)
            >>> p1d = (1, 1) # pad last dim by 1 on each side
            >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
            >>> print(out.size())
            torch.Size([3, 3, 4, 4])
            >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
            >>> out = F.pad(t4d, p2d, "constant", 0)
            >>> print(out.size())
            torch.Size([3, 3, 8, 4])
            >>> t4d = torch.empty(3, 3, 4, 2)
            >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
            >>> out = F.pad(t4d, p3d, "constant", 0)
            >>> print(out.size())
            torch.Size([3, 9, 7, 3])
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, pad: tuple, mode: str = 'constant', value: float = None) -> torch.Tensor:
        """
        Pads tensor with specified padding mode.

        Args:
            x (torch.Tensor): Input tensor.
            pad (tuple): Padding sizes in the form (pad_left, pad_right, pad_top, pad_bottom, ...).
            mode (str, optional): Padding mode: 'constant', 'reflect', 'replicate', 'circular'.
            value (float, optional): Fill value for 'constant' padding.

        Returns:
            torch.Tensor: Padded tensor.
        """
        return torch.nn.functional.pad(x, pad, mode=mode, value=value)


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "15_Pad.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        
        x_info = inputs[0]
        dtype = dtype_map[x_info["dtype"]]
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5)
        
        pad = None
        mode = "constant"
        value = None
        
        for inp in inputs[1:]:
            if inp["name"] == "pad":
                pad = tuple(inp["value"])
            elif inp["name"] == "mode":
                mode = inp["value"]
            elif inp["name"] == "value":
                value = inp["value"]
        
        input_groups.append([x, pad, mode, value])
    return input_groups


def get_init_inputs():
    return []
