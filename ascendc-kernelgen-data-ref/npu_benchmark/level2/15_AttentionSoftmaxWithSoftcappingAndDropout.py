import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Attention Softmax with Softcapping and Dropout Module.

    Applies Gemma3's softcapping transformation followed by softmax normalization.
    Softcapping: tanh(logits / 30.0) * 30.0
    This clamps effective logit range to approximately [-30, +30].
        torch.nn.functional.softmax = softmax(
        input: Tensor,
        dim: int | None = None,
        _stacklevel: int = 3,
        dtype: DType | None = None
    ) -> Tensor
        Apply a softmax function.

        Softmax is defined as:

        :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

        It is applied to all slices along dim, and will re-scale them so that the elements
        lie in the range `[0, 1]` and sum to 1.

        See :class:`~torch.nn.Softmax` for more details.

        Args:
            input (Tensor): input
            dim (int): A dimension along which softmax will be computed.
            dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            If specified, the input tensor is casted to :attr:`dtype` before the operation
            is performed. This is useful for preventing data type overflows. Default: None.

        .. note::
            This function doesn't work directly with NLLLoss,
            which expects the Log to be computed between the Softmax and itself.
            Use log_softmax instead (it's faster and has better numerical properties).
    """

    def __init__(self):
        super(Model, self).__init__()
        self.SOFTCAP = 30.0

    def forward(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply softcapping transformation followed by softmax.

        Args:
            attn_weights: Attention logits of shape (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns:
            Normalized attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        scaled = attn_weights / self.SOFTCAP
        clamped = torch.tanh(scaled)
        softcapped = clamped * self.SOFTCAP
        output = F.softmax(softcapped, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
        return output


def get_input_groups():
    """Generate input groups from JSON test cases."""
    json_path = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0] + '.json')
    input_groups = []
    with open(json_path, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case['inputs']
            tensors = {}
            for inp in inputs:
                if inp['type'] == 'tensor':
                    name = inp['name']
                    dtype_str = inp.get('dtype', 'float32')
                    shape = inp.get('shape')
                    if shape is None:
                        tensors[name] = None
                    elif dtype_str == 'bool':
                        tensors[name] = (torch.rand(shape) > 0.5).to(torch.bool)
                    elif dtype_str in ('int32', 'int64', 'int8'):
                        max_val = {'int32': 1000, 'int64': 10000, 'int8': 127}.get(dtype_str, 100)
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}[dtype_str]
                        tensors[name] = torch.randint(0, max_val, shape, dtype=dtype)
                    else:
                        dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16, 'int32': torch.int32, 'int64': torch.int64, 'int8': torch.int8, 'bool': torch.bool}.get(dtype_str, torch.float32)
                        if idx % 2 == 0:
                            mu = float(torch.empty(1).uniform_(-100, 100).item())
                            sigma = float(torch.empty(1).uniform_(1, 25).item())
                            tensors[name] = torch.normal(mu, sigma, shape, dtype=dtype)
                        else:
                            tensors[name] = torch.empty(shape, dtype=dtype).uniform_(-5, 5)
                elif inp['type'] == 'attr':
                    tensors[inp['name']] = inp['value']

            # Build input list in order matching forward signature
            group = []
            for inp in inputs:
                group.append(tensors[inp['name']])
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
