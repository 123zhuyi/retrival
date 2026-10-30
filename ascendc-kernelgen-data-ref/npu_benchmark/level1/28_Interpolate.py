import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
        Simple model that performs interpolation (resizing) of tensors.
        torch.nn.functional.interpolate = interpolate(
        input: Tensor,
        size: int | None = None,
        scale_factor: list[float] | None = None,
        mode: str = 'nearest',
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
        antialias: bool = False
    ) -> Tensor
        Down/up samples the input.

        Tensor interpolated to either the given :attr:`size` or the given
        :attr:`scale_factor`

        The algorithm used for interpolation is determined by :attr:`mode`.

        Currently temporal, spatial and volumetric sampling are supported, i.e.
        expected inputs are 3-D, 4-D or 5-D in shape.

        The input dimensions are interpreted in the form:
        `mini-batch x channels x [optional depth] x [optional height] x width`.

        The modes available for resizing are: `nearest`, `linear` (3D-only),
        `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `lanczos` (4D-only, CPU only), `area`, `nearest-exact`

        Args:
            input (Tensor): the input tensor
            size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
                output spatial size.
            scale_factor (float or Tuple[float]): multiplier for spatial size. If `scale_factor` is a tuple,
                its length has to match the number of spatial dimensions; `input.dim() - 2`.
            mode (str): algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'`` | ``'lanczos'`` | ``'area'`` | ``'nearest-exact'``. Default: ``'nearest'``
            align_corners (bool, optional): Geometrically, we consider the pixels of the
                input and output as squares rather than points.
                If set to ``True``, the input and output tensors are aligned by the
                center points of their corner pixels, preserving the values at the corner pixels.
                If set to ``False``, the input and output tensors are aligned by the corner
                points of their corner pixels, and the interpolation uses edge value padding
                for out-of-boundary values, making this operation *independent* of input size
                when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
                is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
                Default: ``None``. When ``None`` and :attr:`mode` is one of the linear modes,
                it is treated as ``False``.
            recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
                interpolation calculation. If `recompute_scale_factor` is ``True``, then
                `scale_factor` must be passed in and `scale_factor` is used to compute the
                output `size`. The computed output `size` will be used to infer new scales for
                the interpolation. Note that when `scale_factor` is floating-point, it may differ
                from the recomputed `scale_factor` due to rounding and precision issues.
                If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will
                be used directly for interpolation. Default: ``None``.
            antialias (bool, optional): flag to apply anti-aliasing. Default: ``False``. Using anti-alias
                option together with ``align_corners=False``, interpolation result would match Pillow
                result for downsampling operation. Supported modes: ``'bilinear'``, ``'bicubic'``, ``'lanczos'``.

        .. note::
            With ``mode='bicubic'`` or ``mode='lanczos'``, it's possible to cause overshoot. For some dtypes, it can produce
            negative values or values greater than 255 for images. Explicitly call ``result.clamp(min=0,max=255)``
            if you want to reduce the overshoot when displaying the image.
            For ``uint8`` inputs, it already performs saturating cast operation. So, no manual `clamp` operation is needed.

        .. note::
            Mode ``mode='lanczos'`` uses a Lanczos-3 windowed sinc filter (6 taps) and requires
            ``antialias=True``. It only supports 4-D input (i.e. 2D spatial) and CPU. With ``antialias=True``
            and ``align_corners=False``, the result matches PIL's ``Image.LANCZOS`` resampling filter.

        .. note::
            Mode ``mode='nearest-exact'`` matches Scikit-Image and PIL nearest neighbours interpolation
            algorithms and fixes known issues with ``mode='nearest'``. This mode is introduced to keep
            backward compatibility.
            Mode ``mode='nearest'`` matches buggy OpenCV's ``INTER_NEAREST`` interpolation algorithm.
    """
    
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, size=None, scale_factor=None,
                mode: str = 'nearest', align_corners=None,
                recompute_scale_factor=None, antialias: bool = False) -> torch.Tensor:
        """
        Interpolates (resizes) the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, ...) where ... represents spatial dimensions.
            size (optional): Output spatial size.
            scale_factor (optional): Multiplier for spatial size.
            mode (str, optional): Algorithm used for interpolation: 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area'.
            align_corners (optional): How to align corners when resizing.
            recompute_scale_factor (optional): Recompute scale_factor for backward compatibility.
            antialias (bool, optional): Apply antialiasing.

        Returns:
            torch.Tensor: Interpolated tensor.
        """
        return torch.nn.functional.interpolate(
            x, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias
        )


def get_input_groups():
    json_path = os.path.join(os.path.dirname(__file__), "28_Interpolate.json")
    with open(json_path, "r") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    
    input_groups = []
    for idx, case in enumerate(cases):
        inputs = case["inputs"]
        x_info = inputs[0]
        
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
        
        size = None
        scale_factor = None
        mode = "nearest"
        align_corners = None
        
        for inp in inputs[1:]:
            if inp["name"] == "size":
                size = inp["value"]
            elif inp["name"] == "scale_factor":
                scale_factor = inp["value"]
            elif inp["name"] == "mode":
                mode = inp["value"]
            elif inp["name"] == "align_corners":
                align_corners = inp["value"]
        
        input_groups.append([x, size, scale_factor, mode, align_corners])
    return input_groups


def get_init_inputs():
    return []
