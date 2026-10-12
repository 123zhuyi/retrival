#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""RaggedBinCount multi-path golden in the TestSpec format.

Only one registration key is needed: kernel and GEIR both resolve the snake-case
operator name and share :class:`RaggedBinCountKernelSpec`.  The legacy
``__golden__`` entry stays available and calls the same Torch computation core.

The general reference math is a Torch competitor composition
(``torch.bincount`` per ragged row).  Fixed numerical-corner cases use an
independent arbitrary-precision CPU oracle because even FP64 cannot retain
every FP32 cancellation gap.  ``torch.bincount`` is semantically the per-row
form of the TensorFlow ``RaggedBincount`` operator this IR is compatible with.

Frozen contract points mirrored from ``README.md``:
  * within the public rank-0-to-2 range, ``weights`` is judged by element count
    rather than matching ``values`` shape: zero elements (``[0]``, ``[0, 3]``,
    ``[2, 0]`` alike) accumulates 1.0 per hit, and a non-empty ``weights`` pairs
    with ``values`` in flattened order;
  * ``binary_output=True`` writes exactly 0.0/1.0 and ignores ``weights``
    entirely, including NaN/Inf/negative weights;
  * ``values >= M`` is legal and ignored;
  * if any split is illegal or any value is negative, the **whole** output is
    bitwise ``+0.0`` and no weight is ever read.
"""

import math

import numpy as np
import torch


__spec__ = {
    # kernel + GEIR share this snake-case key and this one Spec class.
    "ragged_bin_count": "RaggedBinCountKernelSpec",
}

__golden__ = {
    "kernel": {"ragged_bin_count": "ragged_bin_count_golden"},
}


_TOL = {
    # The only native output dtype is FP32 (ragged_bin_count_def.cpp).
    #
    # ``cross_check`` compares the NPU and competitor against an FP64
    # high-precision approximation (see :func:`_compute`); its ratio answers
    # "is the NPU within the same order of accuracy as a mature GPU
    # implementation" -- which is the question worth asking for
    # the legacy long-row and unbounded VALUE fallback paths, whose FP32
    # summation order is not reproducible (``asc_atomic_add`` across multiple
    # AIV cores).  The bounded short ROW and bounded VALUE paths instead use
    # fixed-order accumulation/reduction.
    # Fixed numerical-corner cases use the exact integer oracle below because
    # FP64 is not exact across every possible FP32 cancellation gap.
    #
    # This requires a reachable remote XPU endpoint to execute ``third_party``.
    # Runs without one must pass ``--compare close`` (or ``stat_rel_err``)
    # explicitly on the command line, which overrides this entry.
    "float32": {"standard": "cross_check", "level": "L1"},
}


_FLOAT32_MAX = np.finfo(np.float32).max
_FLOAT32_MIN_SUBNORMAL = np.float32(2.0**-149)
_FLOAT32_MIN_NORMAL = np.float32(2.0**-126)
_FLOAT32_NEXT_MIN_NORMAL = np.asarray([0x00800001], dtype=np.uint32).view(np.float32)[0]
_FLOAT32_EXP24_MIN = np.asarray([0x0C000000], dtype=np.uint32).view(np.float32)[0]
_FLOAT32_NEG_EXP23_MAX = np.asarray([0x8BFFFFFF], dtype=np.uint32).view(np.float32)[0]


_PRECISION_SPECIAL_WEIGHTS = {
    "rbc_k_precision_special_pos_inf_i64": [float("inf"), 1.0, 2.0, 3.0],
    "rbc_k_precision_special_inf_cancel_i64": [
        float("inf"),
        -float("inf"),
        1.0,
        2.0,
    ],
    "rbc_k_precision_special_nan_i64": [float("nan"), 1.0, 2.0, 3.0],
    "rbc_k_precision_special_finite_overflow_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
    ],
    "rbc_k_precision_special_finite_residual_pos_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        1.0,
    ],
    "rbc_k_precision_special_finite_residual_neg_i64": [
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -1.0,
    ],
    "rbc_k_precision_special_subnormal_residual_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        _FLOAT32_MIN_SUBNORMAL,
    ],
    # Both inputs are normal, but their exact cancellation residual is the
    # minimum positive subnormal. This catches arithmetic FTZ before TwoSum
    # can observe and preserve the discarded bit.
    "rbc_k_precision_p0_normal_cancel_to_subnormal_i64": [
        _FLOAT32_NEXT_MIN_NORMAL,
        -_FLOAT32_MIN_NORMAL,
    ],
    # Keep the same minimum-subnormal result on a sparse, non-private ROW
    # output so the final direct-GM write is covered as well as accumulation.
    "rbc_k_precision_p0_direct_gm_subnormal_i64": [
        _FLOAT32_NEXT_MIN_NORMAL,
        -_FLOAT32_MIN_NORMAL,
    ],
    # Same-sign low-band normals can also create a subnormal TwoSum residual.
    # The exact result is 0x01400001; an FTZ expansion loses its final bit and
    # rounds to 0x01400000 unless the inputs select the exact accumulator.
    "rbc_k_precision_p0_same_sign_low_band_residual_i64": [
        _FLOAT32_MIN_NORMAL,
        _FLOAT32_NEXT_MIN_NORMAL,
        _FLOAT32_NEXT_MIN_NORMAL,
    ],
    # Lock the detector's inclusive raw-exponent upper boundary. The exp-24
    # operand itself does not require exact handling, while the exp-23 operand
    # must select it before their cancellation produces a subnormal result.
    "rbc_k_precision_p0_exp23_24_cancel_to_subnormal_i64": [
        _FLOAT32_EXP24_MIN,
        _FLOAT32_NEG_EXP23_MAX,
    ],
    "rbc_k_precision_special_max_boundary_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
    ],
    "rbc_k_precision_special_result_overflow_i64": [_FLOAT32_MAX, _FLOAT32_MAX],
    "rbc_k_precision_special_round_tie_even_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        1.0,
        2.0**-24,
    ],
    "rbc_k_precision_special_round_above_tie_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        1.0,
        2.0**-24,
        2.0**-25,
    ],
    # The first two terms land exactly halfway between adjacent FP32 values.
    # The final 1.0 is retained by FP64 golden accumulation but does not fit in
    # a two-component FP32 expansion without a third-order residual.
    "rbc_k_precision_special_third_order_residual_i64": [2.0**50, 2.0**26, 1.0],
    # One launch straddles the precise-work boundary: 64 * 64 == 4096 uses
    # the fixed-order path, while 65 * 64 falls back to the legacy scatter.
    "rbc_k_precision_p0_mixed_work_boundary_i32": (
        [2.0**50, 1.0, -(2.0**50)] + [0.0] * 61 + [1.0, -1.0 + 2.0**-23] + [0.0] * 63
    ),
    # Drive all 256 hits into one bin.  The leading overflow forces an exact
    # rescan and the balanced payload exercises the full scan: omitting the
    # final term would leave +FLT_MAX, while all 256 terms sum to bitwise +0.
    "rbc_k_precision_p0_exact_256_single_bin_i32": (
        [_FLOAT32_MAX] * 128 + [-_FLOAT32_MAX] * 128
    ),
    # Enter the exact rescan because of the first two finite terms, then make
    # that rescan account for both infinity signs.
    "rbc_k_precision_p0_exact_inf_cancel_after_overflow_i64": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        float("inf"),
        -float("inf"),
    ],
    "rbc_k_precision_p0_invalid_before_precise_weights_i32": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        float("inf"),
        -float("inf"),
    ],
    "rbc_k_precision_p0_binary_nonfinite_ignored_i32": [
        float("nan"),
        float("inf"),
        -float("inf"),
        _FLOAT32_MAX,
    ],
    "rbc_k_precision_p0_negative_inf_i32": [-float("inf"), 1.0, 2.0, 3.0],
    # FP64 cannot retain the unit term across this cancellation.  This case is
    # therefore also a regression for the integer-superaccumulator oracle
    # below, rather than relying on an FP64 approximation of the exact sum.
    "rbc_k_precision_p0_twosum_fp64_gap_i64": [
        2.0**100,
        1.0,
        -(2.0**100),
    ],
    # The leading overflow selects the exact rescan.  The final -4.0 has
    # exponent field 129, so its subtraction starts at accumulator word 2 and
    # propagates a borrow through the upper limbs.
    "rbc_k_precision_p0_exact_word2_borrow_i32": [
        _FLOAT32_MAX,
        _FLOAT32_MAX,
        -_FLOAT32_MAX,
        -_FLOAT32_MAX,
        -4.0,
    ],
    # VALUE mapping with one hot bin uses the bounded output-owner path. The
    # leading overflow requires an exact rescan, and the final term proves that
    # the complete 1025-element row was consumed.
    "rbc_k_precision_p0_value_hot_bin_exact_i32": (
        [_FLOAT32_MAX, _FLOAT32_MAX]
        + [0.0] * 1020
        + [-_FLOAT32_MAX, -_FLOAT32_MAX, 1.0]
    ),
    # Exercise the inclusive VALUE-owner work boundary: 131072 values and two
    # bins require 262144 comparisons.  All hits target bin zero; the balanced
    # maximum-finite payload forces the exact rescan through all 131072 terms,
    # while bin one independently proves that a no-hit owner leaves +0.0.
    "rbc_k_precision_p0_value_max_work_exact_i64": (
        [_FLOAT32_MAX] * 65536 + [-_FLOAT32_MAX] * 65536
    ),
    # Exercise the maximum term count admitted by the same work bound.  With
    # one bin, all 262144 values are visited and the exact rescan must consume
    # the complete balanced maximum-finite payload to recover bitwise +0.0.
    "rbc_k_precision_p0_value_max_values_exact_i32": (
        [_FLOAT32_MAX] * 131072 + [-_FLOAT32_MAX] * 131072
    ),
    # N=4096 selects four VALUE cores and therefore 64 global fixed
    # partitions of 64 elements each.  Place each non-zero term at a distinct
    # partition head so the local partials are individually exact and only
    # the fixed-order core-zero merge discovers the third-order residual.
    "rbc_k_precision_p0_value_merge_residual_i64": (
        [2.0**50] + [0.0] * 63 + [2.0**26] + [0.0] * 63 + [1.0] + [0.0] * (4096 - 129)
    ),
    # The finite overflow is likewise created only while merging distinct
    # partition partials; the exact full-row rescan must recover +1.0.
    "rbc_k_precision_p0_value_merge_overflow_i64": (
        [_FLOAT32_MAX]
        + [0.0] * 63
        + [_FLOAT32_MAX]
        + [0.0] * 63
        + [-_FLOAT32_MAX]
        + [0.0] * 63
        + [-_FLOAT32_MAX]
        + [0.0] * 63
        + [1.0]
        + [0.0] * (4096 - 257)
    ),
    # Core one owns indices [1024, 2048).  Its first partition overflows
    # locally, proving a REQUIRES_EXACT flag produced away from core zero is
    # visible to the final owner and causes a complete exact rescan.
    "rbc_k_precision_p0_value_remote_local_exact_i64": (
        [0.0] * 1024
        + [_FLOAT32_MAX, _FLOAT32_MAX]
        + [0.0] * (2048 - 1026)
        + [-_FLOAT32_MAX, -_FLOAT32_MAX]
        + [0.0] * (4095 - 2050)
        + [1.0]
    ),
    # Valid non-finite inputs in different partitions must produce the
    # canonical quiet NaN emitted by the integer superaccumulator.
    "rbc_k_precision_p0_value_nonfinite_cross_partition_i64": (
        [float("inf")] + [0.0] * 63 + [-float("inf")] + [0.0] * (4096 - 65)
    ),
    # A non-zero subnormal is routed to the exact bit-field accumulator so the
    # result is independent of the SIMT denormal arithmetic mode.
    "rbc_k_precision_p0_value_subnormal_i64": ([_FLOAT32_MIN_SUBNORMAL] + [0.0] * 4095),
    # Put two normal operands in distinct fixed partitions. Their merge must
    # select the exact rescan and retain the minimum subnormal residual.
    "rbc_k_precision_p0_value_normal_cancel_to_subnormal_i32": (
        [_FLOAT32_NEXT_MIN_NORMAL]
        + [0.0] * 63
        + [-_FLOAT32_MIN_NORMAL]
        + [0.0] * (4096 - 65)
    ),
    # Four VALUE-owned rows include an empty row and uneven segment lengths.
    # Unit weights make every expected count exactly representable in FP32.
    "rbc_k_precision_p0_value_multirow_owner_i64": [1.0] * 8192,
    "rbc_g_contract_rank2_cancel_precision_32": [
        2.0**50,
        1.0,
        -(2.0**50),
        1.0,
        -1.0 + 2.0**-23,
        0.0,
    ],
}


_PRECISION_SPECIAL_VALUES = {
    "rbc_k_precision_p0_invalid_before_precise_weights_i32": [0, -1, 0, 0],
    "rbc_k_precision_p0_binary_nonfinite_ignored_i32": [0, 0, 1, 7],
    "rbc_g_contract_rank2_cancel_precision_32": [0, 0, 0, 1, 1, 1],
    "rbc_k_precision_p0_value_multirow_owner_i64": [index & 1 for index in range(8192)],
}


def _customize_precision_inputs(splits, values, size, weights, **kwargs):
    """Inject fixed numerical corner cases used by the precision ST."""
    testcase_name = str(kwargs.get("testcase_name", ""))
    weight_payload = _PRECISION_SPECIAL_WEIGHTS.get(testcase_name)
    value_payload = _PRECISION_SPECIAL_VALUES.get(testcase_name)
    if weight_payload is None and value_payload is None:
        return splits, values, size, weights
    fixed_values = (
        np.zeros_like(values)
        if value_payload is None
        else np.asarray(value_payload, dtype=values.dtype).reshape(values.shape)
    )
    fixed_weights = (
        weights
        if weight_payload is None
        else np.asarray(weight_payload, dtype=np.float32).reshape(weights.shape)
    )
    return splits, fixed_values, size, fixed_weights


def _attr(kwargs, name, default):
    """CSV attributes arrive as strings; coerce them to the default's type."""
    value = kwargs.get(name, default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false", "yes", "no", "1", "0"):
            return text in ("true", "yes", "1")
        try:
            return type(default)(value)
        except Exception:
            return default
    return value


def _as_tensor(array):
    if isinstance(array, torch.Tensor):
        return array
    return torch.from_numpy(np.ascontiguousarray(array))


def _contract_inputs(splits, values, size, weights):
    """Flatten to the shapes the frozen contract works in."""
    splits_flat = _as_tensor(splits).reshape(-1).to(torch.int64)
    values_flat = _as_tensor(values).reshape(-1).to(torch.int64)
    size_flat = _as_tensor(size).reshape(-1).to(torch.int64)
    weights_flat = _as_tensor(weights).reshape(-1)
    return splits_flat, values_flat, size_flat, weights_flat


def _runtime_invalid(splits_flat, values_flat):
    """Reproduce the whole-output +0.0 guard: illegal split or negative value."""
    value_count = values_flat.numel()
    if splits_flat.numel() < 2:
        return True
    if bool((splits_flat < 0).any()) or bool((splits_flat > value_count).any()):
        return True
    if int(splits_flat[0].item()) != 0:
        return True
    if int(splits_flat[-1].item()) != value_count:
        return True
    if bool((splits_flat[1:] < splits_flat[:-1]).any()):
        return True
    if bool((values_flat < 0).any()):
        return True
    return False


def _float32_from_exact_units(total):
    """Round an integer multiple of 2^-149 to FP32, ties to even.

    Every finite FP32 input is exactly representable in these units.  Keeping
    the sum as a Python integer avoids the cancellation gap of an FP64 oracle
    (for example 2^100 + 1 - 2^100) and makes the special precision cases an
    independent correctly-rounded reference for the device superaccumulator.
    """
    negative = total < 0
    magnitude = -total if negative else total
    sign_bit = 0x80000000 if negative else 0
    if magnitude == 0:
        bits = 0
    else:
        highest_bit = magnitude.bit_length() - 1
        if highest_bit <= 22:
            bits = sign_bit | magnitude
        else:
            right_shift = highest_bit - 23
            significand = magnitude >> right_shift
            if right_shift != 0:
                remainder = magnitude & ((1 << right_shift) - 1)
                halfway = 1 << (right_shift - 1)
                if remainder > halfway or (
                    remainder == halfway and (significand & 1) != 0
                ):
                    significand += 1
                    if significand == 1 << 24:
                        significand >>= 1
                        highest_bit += 1
            exponent = highest_bit - 22
            if exponent >= 255:
                bits = sign_bit | 0x7F800000
            else:
                bits = sign_bit | (exponent << 23) | (significand & 0x007FFFFF)
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def _float64_from_exact_units(total):
    """Round an integer multiple of 2^-149 to FP64, ties to even."""
    negative = total < 0
    magnitude = -total if negative else total
    sign_bit = 0x8000000000000000 if negative else 0
    if magnitude == 0:
        bits = 0
    else:
        highest_bit = magnitude.bit_length() - 1
        if highest_bit <= 52:
            significand = magnitude << (52 - highest_bit)
        else:
            right_shift = highest_bit - 52
            significand = magnitude >> right_shift
            remainder = magnitude & ((1 << right_shift) - 1)
            halfway = 1 << (right_shift - 1)
            if remainder > halfway or (remainder == halfway and (significand & 1) != 0):
                significand += 1
                if significand == 1 << 53:
                    significand >>= 1
                    highest_bit += 1
        unbiased_exponent = highest_bit - 149
        if unbiased_exponent > 1023:
            bits = sign_bit | 0x7FF0000000000000
        else:
            exponent = unbiased_exponent + 1023
            bits = sign_bit | (exponent << 52) | (significand & 0x000FFFFFFFFFFFFF)
    return np.asarray([bits], dtype=np.uint64).view(np.float64)[0]


def _canonical_nan(result_dtype):
    if np.dtype(result_dtype) == np.dtype(np.float64):
        return np.asarray([0x7FF8000000000000], dtype=np.uint64).view(np.float64)[0]
    return np.asarray([0x7FC00000], dtype=np.uint32).view(np.float32)[0]


def _exact_sum(values, result_dtype):
    """Return the exact FP32-input sum rounded once to ``result_dtype``."""
    result_dtype = np.dtype(result_dtype)
    if result_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"Unsupported exact-reference dtype: {result_dtype}")

    total = 0
    has_nan = False
    has_positive_inf = False
    has_negative_inf = False
    for raw_value in np.asarray(values).reshape(-1):
        value = float(raw_value)
        if math.isnan(value):
            has_nan = True
            continue
        if math.isinf(value):
            if value > 0:
                has_positive_inf = True
            else:
                has_negative_inf = True
            continue

        numerator, denominator = value.as_integer_ratio()
        denominator_shift = denominator.bit_length() - 1
        unit_shift = 149 - denominator_shift
        if unit_shift >= 0:
            total += numerator << unit_shift
        else:
            divisor = 1 << (-unit_shift)
            if numerator % divisor != 0:
                raise ValueError("Exact-reference input is not on the FP32 value grid")
            total += numerator // divisor

    if has_nan or (has_positive_inf and has_negative_inf):
        return _canonical_nan(result_dtype)
    if has_positive_inf:
        return result_dtype.type(float("inf"))
    if has_negative_inf:
        return result_dtype.type(-float("inf"))
    if result_dtype == np.dtype(np.float64):
        return _float64_from_exact_units(total)
    return _float32_from_exact_units(total)


def _compute_exact_weighted_reference(
    splits_flat, values_flat, bins, weights_flat, result_dtype
):
    """Independent exact oracle for the fixed numerical-corner payloads."""
    rows = max(splits_flat.numel() - 1, 0)
    result = np.zeros((rows, bins), dtype=result_dtype)
    values_np = values_flat.cpu().numpy()
    weights_np = weights_flat.cpu().numpy()
    for row in range(rows):
        begin = int(splits_flat[row].item())
        end = int(splits_flat[row + 1].item())
        for bin_index in range(bins):
            selected = weights_np[begin:end][values_np[begin:end] == bin_index]
            result[row, bin_index] = _exact_sum(selected, result_dtype)
    return torch.from_numpy(result)


def _compute(splits, values, size, weights, **kwargs):
    """Return ``[output]`` as a Torch tensor in def.cpp output order."""
    binary_output = _attr(kwargs, "binary_output", False)
    splits_flat, values_flat, size_flat, weights_flat = _contract_inputs(
        splits, values, size, weights
    )

    rows = max(splits_flat.numel() - 1, 0)
    bins = int(size_flat[0].item()) if size_flat.numel() > 0 else 0
    output = torch.zeros((rows, bins), dtype=torch.float32)

    # Invalid input returns bitwise +0.0 before any weight is read.
    if _runtime_invalid(splits_flat, values_flat) or rows == 0 or bins == 0:
        return [output]

    has_weights = weights_flat.numel() > 0
    testcase_name = str(kwargs.get("testcase_name", ""))
    if (
        has_weights
        and not binary_output
        and testcase_name in _PRECISION_SPECIAL_WEIGHTS
    ):
        output_dtypes = _output_dtype_names(kwargs)
        result_dtype = (
            np.dtype(output_dtypes[0]) if output_dtypes else np.dtype(np.float32)
        )
        return [
            _compute_exact_weighted_reference(
                splits_flat, values_flat, bins, weights_flat, result_dtype
            )
        ]

    # Accumulate in FP64 and leave the final dtype decision to
    # ``_kernel_golden``.  Under ``cross_check`` TTK promotes
    # ``output_dtypes`` from FP32 to FP64, so the three-way gate must retain
    # this accumulator without an intermediate FP32 rounding.  Under
    # ``--compare close`` the adapter casts it exactly once to the declared
    # FP32 interface dtype.
    #
    # The legacy long-row and
    # VALUE fallback NPU paths use ``asc_atomic_add`` across multiple cores,
    # while the competitor has its own FP32 order; both can therefore be
    # reorderings of the same exact sum.  The bounded precise ROW/VALUE paths
    # are fixed-order, but use this same independent high-precision reference.
    # Comparing fallback outputs against another FP32 reordering would only
    # measure "how differently did these two reorder", which is not a precision
    # statement about either.  The unrounded FP64 accumulation is the
    # high-precision approximation used by the three-way gate.
    # It is not an exact oracle for arbitrarily wide cancellation gaps; fixed
    # numerical-corner payloads take the integer-superaccumulator path above.
    accumulator = torch.zeros((rows, bins), dtype=torch.float64)
    weights_fp64 = weights_flat.to(torch.float64) if has_weights else None

    for row in range(rows):
        begin = int(splits_flat[row].item())
        end = int(splits_flat[row + 1].item())
        segment = values_flat[begin:end]
        # values >= M are legal and ignored.
        keep = segment < bins
        kept = segment[keep]
        if kept.numel() == 0:
            continue
        if binary_output:
            # Bitwise exact 0/1; weights are never read on this path.
            accumulator[row] = accumulator[row].index_fill(0, kept, 1.0)
        elif has_weights:
            segment_weights = weights_fp64[begin:end][keep]
            accumulator[row] = torch.bincount(
                kept, weights=segment_weights, minlength=bins
            )[:bins]
        else:
            accumulator[row] = torch.bincount(kept, minlength=bins)[:bins].to(
                torch.float64
            )

    return [accumulator]


def _output_dtype_names(kwargs):
    output_dtypes = kwargs.get("output_dtypes") or []
    names = []
    for dtype in output_dtypes:
        if isinstance(dtype, (list, tuple)):
            dtype = dtype[0]
        names.append(str(dtype))
    return names


def _kernel_golden(splits, values, size, weights, **kwargs):
    """Kernel/GEIR container adapter: NumPy in, NumPy list out."""
    outputs = _compute(splits, values, size, weights, **kwargs)
    output_dtypes = _output_dtype_names(kwargs)
    result = []
    for index, output in enumerate(outputs):
        array = output.detach().cpu().contiguous().numpy()
        if index < len(output_dtypes):
            array = array.astype(output_dtypes[index], copy=False)
        result.append(np.ascontiguousarray(array))
    return result


def _competitor_forward(splits_flat, values_flat, bins, weights_flat, binary_output):
    """Optimal single-pass competitor form (see :class:`_RaggedBinCountCompose`)."""
    device = values_flat.device
    rows = max(splits_flat.numel() - 1, 0)
    output = torch.zeros(rows * bins, dtype=torch.float32, device=device)

    # Row id per value, from the ragged offsets: one repeat_interleave instead
    # of a Python loop over rows.
    counts = splits_flat[1:] - splits_flat[:-1]
    row_id = torch.repeat_interleave(
        torch.arange(rows, device=device, dtype=torch.int64), counts
    )

    keep = values_flat < bins  # values >= M are legal and ignored
    flat_index = row_id * bins + values_flat
    flat_index = flat_index[keep]

    if binary_output:
        # Weights are never read on this path; writes are idempotent 1.0.
        output[flat_index] = 1.0
    elif weights_flat.numel() > 0:
        output.index_add_(0, flat_index, weights_flat.to(torch.float32)[keep])
    else:
        output.index_add_(
            0,
            flat_index,
            torch.ones(flat_index.numel(), dtype=torch.float32, device=device),
        )

    return output.view(rows, bins)


class _RaggedBinCountCompose:
    """Independent competitor composition, executed on the remote GPU.

    Two requirements pull in different directions and both are met here:

    * It must not share an implementation with :func:`_compute`, or the two
      reference paths cannot cross-validate each other.  ``_compute`` is a
      per-row ``torch.bincount`` accumulating in FP64; this is a single flat
      ``index_add_`` in FP32 over a ``repeat_interleave``-derived row index.
    * It must be the competitor's **optimal** form, or the G/N ratio is
      inflated and the comparison is not an honest one.  The whole ragged
      problem is expressed as one scatter over the flattened output, so torch
      launches a constant number of kernels regardless of the row count -- the
      earlier per-row Python loop cost 66 ms on an A100 for a case the fused
      form finishes in tens of microseconds.

    CUDA ``index_add_`` can use a nondeterministic atomic schedule.  Do not
    enable PyTorch's global deterministic mode in this performance provider:
    the same object supplies ``--xpu-perf``, and the deterministic scatter is a
    materially different, slower competitor.  A reproducibility experiment
    should use a separate precision-only provider or frozen competitor output.
    """

    def __init__(self, **kwargs):
        self.binary_output = _attr(kwargs, "binary_output", False)

    def __call__(self, splits, values, size, weights, **kwargs):
        del kwargs
        splits_flat, values_flat, size_flat, weights_flat = _contract_inputs(
            splits, values, size, weights
        )
        rows = max(splits_flat.numel() - 1, 0)
        bins = int(size_flat[0].item()) if size_flat.numel() > 0 else 0
        if _runtime_invalid(splits_flat, values_flat) or rows == 0 or bins == 0:
            return [
                torch.zeros(
                    (rows, bins), dtype=torch.float32, device=values_flat.device
                )
            ]

        # NPU output is always FP32; align the competitor dtype.
        return [
            _competitor_forward(
                splits_flat, values_flat, bins, weights_flat, self.binary_output
            )
        ]


class RaggedBinCountKernelSpec:
    """Shared kernel/GEIR TestSpec; parameter names follow def.cpp."""

    @staticmethod
    def golden(splits, values, size, weights, **kwargs):
        return _kernel_golden(splits, values, size, weights, **kwargs)

    @staticmethod
    def customize_inputs(splits, values, size, weights, **kwargs):
        return _customize_precision_inputs(splits, values, size, weights, **kwargs)

    third_party = {"torch": _RaggedBinCountCompose}
    tolerance = _TOL


def ragged_bin_count_golden(splits, values, size, weights, *args, **kwargs):
    """Compatibility entry for the historical ``__golden__`` kernel loader."""
    del args
    legacy_kwargs = dict(kwargs)
    if not legacy_kwargs.get("output_dtypes"):
        # The old entry predates TestSpec and promises the fixed FP32 output.
        legacy_kwargs["output_dtypes"] = [["float32"]]
    return _kernel_golden(splits, values, size, weights, **legacy_kwargs)[0]


# 【不存在】ACLNN 通路：CMakeLists.txt 显式配置 ``ACLNNTYPE aclnn_exclude``，
# 本算子既无 op_host/op_api 目录，也不交付 aclnnRaggedBinCount 符号。
# 【不存在】e2e 通路：获批支持面为 Graph IR only（README.md 调用说明），
# torch_npu 二进制中无本算子的 aclnn 绑定入口。
