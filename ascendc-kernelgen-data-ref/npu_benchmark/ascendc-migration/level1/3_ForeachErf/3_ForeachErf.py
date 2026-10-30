import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    torch._foreach_erf = _foreach_erf(self: List[Tensor]) -> List[Tensor]

    Applies the error function (erf) element-wise to each tensor in the input list.
    This processes all tensors simultaneously, which is more efficient than calling
    torch.erf on each tensor individually.

    Args:
        tensors (List[Tensor]): A list of input tensors. All tensors must be on the
            same device and have the same dtype.
    Returns:
        List[Tensor]: A list of output tensors, each with erf applied element-wise.
    Example::

        >>> x = [torch.tensor([0.00, 1.0]), torch.tensor([-1.0, 0.0, 1.0])]
        >>> torch._foreach_erf(x)
        = [tensor([0.0000, 0.5205, 0.8427]), tensor([-0.8427, 0.0000, 0.8427])]
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensors):
        return torch._foreach_erf(tensors)


def _make_fill_tensor(shape, fill_val, dtype):
    if fill_val == 'inf':
        return torch.full(shape, float('inf'), dtype=dtype)
    elif fill_val == '-inf':
        return torch.full(shape, float('-inf'), dtype=dtype)
    elif fill_val == 'nan':
        return torch.full(shape, float('nan'), dtype=dtype)
    elif fill_val == 'zero':
        return torch.zeros(shape, dtype=dtype)
    else:
        return torch.zeros(shape, dtype=dtype)


def get_input_groups(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "3_ForeachErf.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    input_groups = []

    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        scene = case.get("scene", "")
        tensors_info = inputs[0]
        items = tensors_info["items"]
        tensors = []
        for item in items:
            dtype = dtype_map[item["dtype"]]
            shape = item["shape"]
            fill = item.get("fill", None)
            non_continuous = item.get("non_continuous", False)

            if fill is not None:
                if fill == 'inf':
                    tensor = torch.full(shape, float('inf'), dtype=dtype)
                elif fill == '-inf':
                    tensor = torch.full(shape, float('-inf'), dtype=dtype)
                elif fill == 'nan':
                    tensor = torch.full(shape, float('nan'), dtype=dtype)
                else:
                    tensor = torch.zeros(shape, dtype=dtype)
            elif 'empty_tensor' in scene and len(shape) == 1 and shape[0] == 0:
                tensor = torch.empty(0, dtype=dtype)
            elif 'scalar' in scene and len(shape) == 0:
                if idx % 2 == 0:
                    mu = float(torch.empty(1).uniform_(-100, 100).item())
                    raw = torch.normal(mu, 1.0, (), dtype=torch.float32)
                    tensor = raw.to(dtype)
                else:
                    raw = torch.empty((), dtype=torch.float32).uniform_(-5, 5)
                    tensor = raw.to(dtype)
            elif non_continuous and 'transposed' in scene and len(shape) == 2:
                tensor = torch.empty(shape[0], shape[1], dtype=dtype).uniform_(-5, 5).t()
            elif non_continuous and 'sliced' in scene and len(shape) == 1:
                tensor = torch.empty(shape[0] * 2, dtype=dtype).uniform_(-5, 5)[::2]
            else:
                if idx % 2 == 0:
                    mu = float(torch.empty(1).uniform_(-100, 100).item())
                    sigma = float(torch.empty(1).uniform_(1, 25).item())
                    raw = torch.normal(mu, sigma, shape, dtype=torch.float32)
                    tensor = raw.to(dtype)
                else:
                    raw = torch.empty(shape, dtype=torch.float32).uniform_(-5, 5)
                    tensor = raw.to(dtype)
            tensors.append(tensor)

        input_groups.append([tensors])
    return input_groups


def get_init_inputs():
    return []
