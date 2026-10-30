import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that puts values into a tensor at specified indices (1D case).
    For a 1D tensor x, uses a single index tensor to put values.
    torch.Tensor.index_put_ = index_put_(...) unbound torch._C.TensorBase method
    index_put_(indices, values, accumulate=False) -> Tensor

    Puts values from the tensor :attr:`values` into the tensor :attr:`self` using
    the indices specified in :attr:`indices` (which is a tuple of Tensors). The
    expression ``tensor.index_put_(indices, values)`` is equivalent to
    ``tensor[indices] = values``. Returns :attr:`self`.

    If :attr:`accumulate` is ``True``, the elements in :attr:`values` are added to
    :attr:`self`. If accumulate is ``False``, the behavior is undefined if indices
    contain duplicate elements.

    Args:
        indices (tuple of LongTensor): tensors used to index into `self`.
        values (Tensor): tensor of same dtype as `self`.
        accumulate (bool): whether to accumulate into self
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, values: torch.Tensor, accumulate: bool = False) -> torch.Tensor:
        """
        Puts values into the tensor at the specified indices.

        Args:
            x (torch.Tensor): Input tensor.
            index (torch.Tensor): 1-D index tensor for the first dimension.
            values (torch.Tensor): Values to put at the specified indices.
            accumulate (bool, optional): Whether to accumulate values at the indices.

        Returns:
            torch.Tensor: Tensor with values put at specified indices.
        """
        x.index_put_((index,), values, accumulate=accumulate)
        return x


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "19_IndexPut.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        index_info = inputs[1]
        values_info = inputs[2]
        accumulate_info = inputs[3]
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[x_info["dtype"]]
        
        if idx % 2 == 0:
            mu = float(torch.empty(1).uniform_(-100, 100).item())
            sigma = float(torch.empty(1).uniform_(1, 25).item())
            x = torch.normal(mu, sigma, x_info["shape"], dtype=dtype)
        else:
            x = torch.empty(x_info["shape"], dtype=dtype).uniform_(-5, 5)
        
        index_range = index_info.get("range", [0, x_info["shape"][0] - 1])
        index_size = index_info["shape"][0]
        index = torch.randint(index_range[0], index_range[1] + 1, (index_size,), dtype=torch.int64)
        
        values = torch.randn(values_info["shape"], dtype=dtype)
        accumulate = accumulate_info["value"]
        
        input_groups.append([x, index, values, accumulate])
    return input_groups


def get_init_inputs():
    return []
