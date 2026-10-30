import torch
import torch.nn as nn
import json
import os


class Model(nn.Module):
    r"""
    torch.nn.CTCLoss = class CTCLoss(_Loss)
        torch.nn.CTCLoss(blank: int = 0, reduction: str = 'mean', zero_infinity: bool = False) -> None

        The Connectionist Temporal Classification loss.

        Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
        probability of possible alignments of input to target, producing a loss value which is differentiable
        with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
        limits the length of the target sequence such that it must be :math:`\leq` the input length.

        Args:
            blank (int, optional): blank label. Default :math:`0`.
            reduction (str, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the output losses will be divided by the target lengths and
                then the mean over the batch is taken, ``'sum'``: the output losses will be summed.
                Default: ``'mean'``
            zero_infinity (bool, optional):
                Whether to zero infinite losses and the associated gradients.
                Default: ``False``
                Infinite losses mainly occur when the inputs are too short
                to be aligned to the targets.

        Shape:
            - Log_probs: Tensor of size :math:`(T, N, C)` or :math:`(T, C)`,
            where :math:`T = \text{input length}`,
            :math:`N = \text{batch size}`, and
            :math:`C = \text{number of classes (including blank)}`.
            The logarithmized probabilities of the outputs (e.g. obtained with
            :func:`torch.nn.functional.log_softmax`).
            - Targets: Tensor of size :math:`(N, S)` or
            :math:`(\operatorname{sum}(\text{target\_lengths}))`,
            where :math:`N = \text{batch size}` and
            :math:`S = \text{max target length, if shape is } (N, S)`.
            It represents the target sequences. Each element in the target
            sequence is a class index. And the target index cannot be blank (default=0).
            In the :math:`(N, S)` form, targets are padded to the
            length of the longest sequence, and stacked.
            In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
            the targets are assumed to be un-padded and
            concatenated within 1 dimension.
            - Input_lengths: Tuple or tensor of size :math:`(N)` or :math:`()`,
            where :math:`N = \text{batch size}`. It represents the lengths of the
            inputs (must each be :math:`\leq T`). And the lengths are specified
            for each sequence to achieve masking under the assumption that sequences
            are padded to equal lengths.
            - Target_lengths: Tuple or tensor of size :math:`(N)` or :math:`()`,
            where :math:`N = \text{batch size}`. It represents lengths of the targets.
            Lengths are specified for each sequence to achieve masking under the
            assumption that sequences are padded to equal lengths. If target shape is
            :math:`(N,S)`, target_lengths are effectively the stop index
            :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
            each target in a batch. Lengths must each be :math:`\leq S`
            If the targets are given as a 1d tensor that is the concatenation of individual
            targets, the target_lengths must add up to the total length of the tensor.
            - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
            ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N)` if input is batched or
            :math:`()` if input is unbatched, where :math:`N = \text{batch size}`.

        Examples:

            >>> # Target are to be padded
            >>> T = 50  # Input sequence length
            >>> C = 20  # Number of classes (including blank)
            >>> N = 16  # Batch size
            >>> S = 30  # Target sequence length of longest target in batch (padding length)
            >>> S_min = 10  # Minimum target length, for demonstration purposes
            >>>
            >>> # Initialize random batch of input vectors, for *size = (T,N,C)
            >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
            >>>
            >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
            >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
            >>>
            >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
            >>> target_lengths = torch.randint(
            ...     low=S_min,
            ...     high=S,
            ...     size=(N,),
            ...     dtype=torch.long,
            ... )
            >>> ctc_loss = nn.CTCLoss()
            >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
            >>> loss.backward()
            >>>
            >>>
            >>> # Target are to be un-padded
            >>> T = 50  # Input sequence length
            >>> C = 20  # Number of classes (including blank)
            >>> N = 16  # Batch size
            >>>
            >>> # Initialize random batch of input vectors, for *size = (T,N,C)
            >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
            >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
            >>>
            >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
            >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
            >>> target = torch.randint(
            ...     low=1,
            ...     high=C,
            ...     size=(sum(target_lengths),),
            ...     dtype=torch.long,
            ... )
            >>> ctc_loss = nn.CTCLoss()
            >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
            >>> loss.backward()
            >>>
            >>>
            >>> # Target are to be un-padded and unbatched (effectively N=1)
            >>> T = 50  # Input sequence length
            >>> C = 20  # Number of classes (including blank)
            >>>
            >>> # Initialize random batch of input vectors, for *size = (T,C)
            >>> # xdoctest: +SKIP("FIXME: error in doctest")
            >>> input = torch.randn(T, C).log_softmax(1).detach().requires_grad_()
            >>> input_lengths = torch.tensor(T, dtype=torch.long)
            >>>
            >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
            >>> target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
            >>> target = torch.randint(
            ...     low=1,
            ...     high=C,
            ...     size=(target_lengths,),
            ...     dtype=torch.long,
            ... )
            >>> ctc_loss = nn.CTCLoss()
            >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
            >>> loss.backward()
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, log_probs, targets, input_lengths, target_lengths,
                blank=0, zero_infinity=False):
        """
        Computes the CTC loss.

        Args:
            log_probs (torch.Tensor): Log probabilities of shape (T, N, C) or (T, C).
            targets (torch.Tensor): Target sequences.
            input_lengths (torch.Tensor): Input sequence lengths.
            target_lengths (torch.Tensor): Target sequence lengths.
            blank (int): Blank label index. Default: 0.
            zero_infinity (bool): Zero out infinite losses. Default: False.

        Returns:
            torch.Tensor: CTC loss (negLogLikelihood).
        """
        result = torch.nn.functional.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=blank, reduction='none', zero_infinity=zero_infinity
        )
        return result


def get_input_groups(json_path=None):
    """从JSON文件读取测试用例并生成输入数据"""
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "2_CTClossV3.json")
    
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_dtype_map = {
        "int64": torch.int64,
        "int32": torch.int32,
    }

    input_groups = []
    
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        log_probs_info = inputs[0]
        targets_info = inputs[1]

        lp_shape = log_probs_info["shape"]
        dtype = dtype_map[log_probs_info["dtype"]]
        target_dtype = target_dtype_map[targets_info["dtype"]]
        tgt_shape = targets_info["shape"]
        fill = log_probs_info.get("fill", None)

        # 获取参数
        blank = 0
        zero_infinity = False
        for inp in inputs[2:]:
            if inp["name"] == "blank":
                blank = inp["value"]
            elif inp["name"] == "zero_infinity":
                zero_infinity = inp["value"]

        # 判断维度
        is_2d = len(lp_shape) == 2

        if is_2d:
            T, C = lp_shape
            N = 1
            S = tgt_shape[0] if len(tgt_shape) == 1 else tgt_shape[1]
        else:
            T, N, C = lp_shape
            S = tgt_shape[1]

        # 生成 log_probs（保持原始精度）
        if fill == 'inf':
            log_probs = torch.full(lp_shape, float('inf'), dtype=dtype)
        elif fill == '-inf':
            log_probs = torch.full(lp_shape, float('-inf'), dtype=dtype)
        elif fill == 'nan':
            log_probs = torch.full(lp_shape, float('nan'), dtype=dtype)
        elif fill == 'zero':
            # 全零输入经过log_softmax
            raw = torch.zeros(lp_shape, dtype=torch.float32)
            dim = -1 if len(lp_shape) == 3 else 1
            log_probs = torch.nn.functional.log_softmax(raw, dim=dim).to(dtype)
        else:
            # 随机输入：根据索引奇偶性选择不同的生成方式
            if idx % 2 == 0:
                # 偶数索引：使用正态分布
                mu = float(torch.empty(1).uniform_(-100, 100).item())
                sigma = float(torch.empty(1).uniform_(1, 25).item())
                # 先生成正态分布的原始数据
                raw = torch.normal(mu, sigma, lp_shape, dtype=torch.float32)
            else:
                # 奇数索引：使用均匀分布
                raw = torch.empty(lp_shape, dtype=torch.float32).uniform_(-5, 5)
            
            # 应用log_softmax
            dim = -1 if len(lp_shape) == 3 else 1
            log_probs = torch.nn.functional.log_softmax(raw, dim=dim).to(dtype)

        # 生成 targets
        all_labels = [i for i in range(C) if i != blank]
        if not all_labels:  # 如果所有标签都是blank
            all_labels = [0]
        
        label_tensor = torch.tensor(all_labels, dtype=target_dtype)
        indices = torch.randint(0, len(all_labels), (N, S))
        targets = label_tensor[indices]

        # 生成 lengths
        input_lengths = torch.full((N,), T, dtype=torch.int64)
        target_lengths = torch.full((N,), min(S, T//2), dtype=torch.int64)

        # 处理2D情况
        if is_2d:
            log_probs = log_probs.squeeze(1) if len(log_probs.shape) == 3 else log_probs
            targets = targets.squeeze(0)
            input_lengths = input_lengths.squeeze(0)
            target_lengths = target_lengths.squeeze(0)

        input_groups.append([log_probs, targets, input_lengths, target_lengths, blank, zero_infinity])
    
    return input_groups


def get_init_inputs():
    return []