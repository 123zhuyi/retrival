import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Simple model that splits a tensor into chunks.
    torch.split = split(tensor: Tensor, split_size_or_sections: int | list[int], dim: int = 0) -> tuple[Tensor, ...]
    Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim` is not divisible by
    :attr:`split_size`.

    If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
    to :attr:`split_size_or_sections`.

    Args:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
            list of sizes for each chunk
        dim (int): dimension along which to split the tensor.

    Example::

        >>> a = torch.arange(10).reshape(5, 2)
        >>> a
        tensor([[0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9]])
        >>> torch.split(a, 2)
        (tensor([[0, 1],
                 [2, 3]]),
         tensor([[4, 5],
                 [6, 7]]),
         tensor([[8, 9]]))
        >>> torch.split(a, [1, 4])
        (tensor([[0, 1]]),
         tensor([[2, 3],
                 [4, 5],
                 [6, 7],
                 [8, 9]]))
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, split_size_or_sections, dim: int = 0):
        """
        Splits the tensor into chunks.

        Args:
            x (torch.Tensor): Input tensor to split.
            split_size_or_sections (int or list): If int, size of each chunk. If list, sizes of each chunk.
            dim (int, optional): Dimension along which to split the tensor.

        Returns:
            tuple: Tuple of tensors resulting from the split.
        """
        if x.device.type == "cpu":
            return torch.split(x, split_size_or_sections, dim=dim)
        else:
            out_list = []
            for s in split_size_or_sections:
                shape = list(x.shape)
                shape[dim] = s
                out_t = torch.empty(shape, dtype=x.dtype, device=x.device)
                out_list.append(out_t)
            torch.split_with_sizes_copy(x, split_size_or_sections, dim=dim, out=out_list)
            return out_list


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "14_Split.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        split_info = inputs[1]
        dim_info = inputs[2]
        
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
        split_size_or_sections = split_info["value"]
        dim = dim_info["value"]
        input_groups.append([x, split_size_or_sections, dim])
    return input_groups


def get_init_inputs():
    return []
