#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Generate the deterministic 1026-case SquareSumAll TTK kernel suite."""

import argparse
import ast
import csv
import io
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


BASE_CASE_COUNT = 1000
PUBLIC_FORMATS = ("NCHW", "NHWC")
PUBLIC_FORMAT_CASE_LABELS = (
    "minimum",
    "vector_aligned",
    "vector_tail",
    "tile_minus_one",
    "tile_aligned",
    "tile_plus_one",
    "core_split_minus_one",
    "core_split_aligned",
    "core_split_plus_one",
    "sixteen_cores_tail",
    "fifty_six_cores",
    "three_tile_loops",
)
PUBLIC_FORMAT_ELEMENTS = (
    1,
    64,
    65,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    65537,
    229376,
    458753,
)
PUBLIC_FORMAT_CASES_PER_FORMAT = len(PUBLIC_FORMAT_ELEMENTS)
TARGETED_CASE_COUNT = 2 + len(PUBLIC_FORMATS) * PUBLIC_FORMAT_CASES_PER_FORMAT
CASE_COUNT = BASE_CASE_COUNT + TARGETED_CASE_COUNT
SEED_COUNT = BASE_CASE_COUNT // 2
BASE_SEED_COUNT = 250
EXTRA_SEED_COUNT = SEED_COUNT - BASE_SEED_COUNT
MAX_CORE_COUNT = 56
TILE_ELEMENTS = 4096
MAX_DIMENSION = 1 << 20
# The operator only requires every dimension to be positive and the element count to fit in
# int64; MAX_ELEMENTS is the largest total this suite drives (56 cores x 4096 x 8 tiles + 1).
MAX_TILE_LOOPS = 8
MAX_ELEMENTS = MAX_CORE_COUNT * TILE_ELEMENTS * MAX_TILE_LOOPS + 1
GPU_ALIGNED_MIN_ELEMENTS = 8192
# partialCount is capped at 1728 and each partial owns 128 elements per
# grid-stride chunk.  chunkCount is stored in uint16_t, so key 1 accepts at
# most 1728 * 128 * 65535 elements; larger legal shapes fall back to key 0.
GPU_ALIGNED_MAX_ELEMENTS = 14_495_293_440
FP32_RTOL = 2**-10
FP32_ATOL = 2**-16
REQUIRED_MISMATCH_RATIO = 0.01
SPECIAL_SEEDS_PER_PROFILE = 2

CSV_HEADERS = (
    "testcase_name",
    "network_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "input_ori_shapes",
    "input_ori_formats",
    "output_ori_shapes",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes",
    "output_shape_unknown_indexes",
    "is_enabled",
    "remark",
    "soc_series",
    "priority",
    "dump_file_prefix",
    "manual_input_binaries",
    "manual_golden_binaries",
)

FINITE_PROFILE_NAMES = (
    "symmetric",
    "positive_negative",
    "negative_positive",
    "near_zero",
    "x1_wide",
    "x2_wide",
    "x1_zero",
    "x2_zero",
    "signed_constants",
    "irregular_bounds",
    "small_values",
    "signed_zero",
)

SPECIAL_PROFILE_NAMES = (
    "x1_nan",
    "x2_nan",
    "x1_posinf",
    "x2_posinf",
    "x1_neginf",
    "x2_neginf",
    "x1_mixed_nan",
    "x2_mixed_nan",
    "x1_mixed_inf",
    "x2_mixed_inf",
    "x1_inf_x2_nan",
    "x1_nan_x2_neginf",
    "both_zero",
    "subnormal_both",
    "x1_finite_overflow",
    "x2_finite_overflow",
)

NORMAL_PARAMETERS = (
    ((0.0, 1.0), (1.0, 0.5)),
    ((2.0, 0.5), (-2.0, 0.5)),
    ((-4.0, 0.25), (4.0, 0.25)),
    ((0.0, 2.0), (0.0, 0.1)),
    ((1.5, 1.25), (-0.5, 0.75)),
    ((-1.5, 0.75), (0.5, 1.25)),
    ((0.0, 0.1), (0.0, 1.5)),
    ((0.0, 1.5), (0.0, 0.1)),
    ((3.0, 0.2), (-3.0, 0.2)),
    ((-2.5, 0.4), (1.25, 0.8)),
    ((0.25, 0.1), (-0.25, 0.1)),
    ((-1.0, 1.0), (1.0, 1.0)),
)

KERNEL_PUBLIC_FORMAT_SHAPES: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    "NCHW": (
        (1, 1, 1, 1),
        (1, 2, 4, 8),
        (1, 1, 5, 13),
        (1, 3, 5, 273),
        (1, 8, 16, 32),
        (1, 17, 1, 241),
        (1, 1, 1, 8191),
        (1, 8, 32, 32),
        (1, 3, 1, 2731),
        (1, 1, 1, 65537),
        (1, 7, 256, 128),
        (1, 79, 1, 5807),
    ),
    "NHWC": (
        (1, 1, 1, 1),
        (1, 4, 8, 2),
        (1, 5, 13, 1),
        (1, 5, 273, 3),
        (1, 16, 32, 8),
        (1, 1, 241, 17),
        (1, 1, 8191, 1),
        (1, 32, 32, 8),
        (1, 1, 2731, 3),
        (1, 1, 65537, 1),
        (1, 256, 128, 7),
        (1, 1, 5807, 79),
    ),
}


@dataclass(frozen=True)
class CaseSeed:
    """One shape/profile seed; it expands to uniform and normal testcases."""

    category: str
    shape: Tuple[int, ...]
    profile_index: int
    special_profile: Optional[str] = None


def _elements(shape: Sequence[int]) -> int:
    return math.prod(shape)


def _used_cores(element_count: int) -> int:
    return min(MAX_CORE_COUNT, max(1, element_count // TILE_ELEMENTS))


def _tile_loops(element_count: int) -> int:
    core_count = _used_cores(element_count)
    max_core_elements = (element_count + core_count - 1) // core_count
    return (max_core_elements + TILE_ELEMENTS - 1) // TILE_ELEMENTS


def _tiling_key(element_count: int) -> int:
    """Expected Ascend950 route for the current product tiling policy."""
    return int(GPU_ALIGNED_MIN_ELEMENTS <= element_count <= GPU_ALIGNED_MAX_ELEMENTS)


def _distribute_power_shape(rank: int, exponent: int, phase: int) -> Tuple[int, ...]:
    base, extra = divmod(exponent, rank)
    exponents = [base + (1 if index < extra else 0) for index in range(rank)]
    rotation = phase % rank
    exponents = exponents[rotation:] + exponents[:rotation]
    return tuple(1 << value for value in exponents)


def _balanced_shapes(rank: int) -> List[Tuple[int, ...]]:
    exponent_candidates = (rank, 6, 8, 10, 12, 14, 16, 18, 20, 19, 17, 15)
    exponents = []
    for exponent in exponent_candidates:
        if exponent not in exponents:
            exponents.append(exponent)
        if len(exponents) == 9:
            break
    shapes = [
        _distribute_power_shape(rank, exponent, index)
        for index, exponent in enumerate(exponents)
    ]
    shapes.append(tuple([1] * (rank - 2) + [7, 3]))
    if rank == 2:
        shapes.append((15, 31))
    else:
        shapes.append(tuple([1] * (rank - 3) + [3, 7, 15]))
    assert len(shapes) == 11 and len(set(shapes)) == 11
    return shapes


def _factor_shape(rank: int, element_count: int, phase: int) -> Tuple[int, ...]:
    """Spread element_count over rank dimensions, rotating so `1` visits every position."""
    factors = [1] * rank
    remaining = element_count
    index = 0
    divisor = 2
    while index < rank - 1 and remaining > 1 and divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors[index] = divisor
            remaining //= divisor
            index += 1
        else:
            divisor += 1
    factors[rank - 1] = remaining
    rotation = phase % rank
    rotated = factors[rotation:] + factors[:rotation]
    assert math.prod(rotated) == element_count
    return tuple(rotated)


# Residues that make `baseCoreElements` land on either side of the 64-element vector width.
VECTOR_TAIL_RESIDUES = (
    0,
    1,
    2,
    3,
    7,
    8,
    9,
    15,
    16,
    17,
    23,
    24,
    25,
    31,
    32,
    33,
    39,
    40,
    41,
    47,
    48,
    49,
    55,
    56,
    57,
    60,
    61,
    62,
    63,
)

ODD_FACTOR_SHAPES = (
    (997,),
    (1009, 3),
    (13, 17),
    (3, 5, 7, 11),
    (7, 7, 7, 7),
    (11, 13, 17),
    (23, 29, 31),
    (101, 103),
    (5, 5, 5, 5, 5),
    (2, 3, 5, 7, 11, 13),
    (3, 3, 3, 3, 3, 3, 3, 3),
    (255, 255),
    (127, 129),
    (63, 65),
    (31, 33, 35),
    (15, 17, 19, 21),
    (9, 11, 13, 15, 17),
    (1, 3, 1, 5, 1, 7, 1, 9),
    (3, 1, 5, 1, 7, 1, 9, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, 1, 3),
    (1, 2, 1, 2, 1, 2, 1, 2),
    (2, 1, 2, 1, 2, 1, 2, 1),
    (6, 1, 1, 1, 1, 1, 1, 683),
    (683, 1, 1, 1, 1, 1, 1, 6),
    (1, 4099),
    (4099, 1),
    (1, 1, 8191),
    (8191, 1, 1),
    (1, 12289, 1),
    (5, 1, 13, 1, 3, 1, 7, 1),
    (43, 47, 53),
    (59, 61),
    (71, 73, 79),
    (83, 89),
    (1, 1, 1, 1, 1, 1, 3, 5461),
    (5461, 3, 1, 1, 1, 1, 1, 1),
    (17, 241),
    (241, 17),
)


def _build_extra_seeds(used_shapes: set) -> List[CaseSeed]:
    """Seeds that widen the base suite along the dimensions the Host tiling actually varies."""
    candidates: List[CaseSeed] = []

    # extraCoreCount == 0: totalElements divides evenly across every core count 1..56.
    for core_count in range(1, MAX_CORE_COUNT + 1):
        element_count = core_count * TILE_ELEMENTS
        rank = 1 + (core_count - 1) % 8
        candidates.append(
            CaseSeed(
                category=f"extra0_c{core_count:02d}",
                shape=_factor_shape(rank, element_count, core_count),
                profile_index=(core_count + 2) % len(FINITE_PROFILE_NAMES),
            )
        )

    # extraCoreCount == usedCoreNum - 1: the widest possible spread of the +1-element cores.
    for core_count in range(2, MAX_CORE_COUNT + 1):
        element_count = core_count * TILE_ELEMENTS + core_count - 1
        rank = 1 + core_count % 8
        candidates.append(
            CaseSeed(
                category=f"extramax_c{core_count:02d}",
                shape=_factor_shape(rank, element_count, core_count + 3),
                profile_index=(core_count + 7) % len(FINITE_PROFILE_NAMES),
            )
        )

    # Per-core tile loops 5..8, immediately below/at/above each boundary.
    for tile_count in range(5, MAX_TILE_LOOPS + 1):
        boundary = MAX_CORE_COUNT * TILE_ELEMENTS * tile_count
        for offset in (-1, 0, 1):
            element_count = boundary + offset
            candidates.append(
                CaseSeed(
                    category=f"tile_{tile_count}_{offset + 1}",
                    shape=(element_count,),
                    profile_index=(tile_count * 5 + offset) % len(FINITE_PROFILE_NAMES),
                )
            )

    # Single core, second tile carries `residue` elements -> sweeps the within-tile 64 tail.
    for index, residue in enumerate(VECTOR_TAIL_RESIDUES):
        element_count = TILE_ELEMENTS + residue
        candidates.append(
            CaseSeed(
                category=f"vtail_r{residue:02d}",
                shape=(element_count,),
                profile_index=(index + 3) % len(FINITE_PROFILE_NAMES),
            )
        )

    # Rank forms whose element counts factor badly, so `1` lands in every position.
    awkward_elements = (65, 129, 4097, 8193, 12289, 65537, 262145, 1048577)
    for rank in range(2, 9):
        for index, element_count in enumerate(awkward_elements):
            candidates.append(
                CaseSeed(
                    category=f"rankform_n{element_count}",
                    shape=_factor_shape(rank, element_count, rank + index),
                    profile_index=(rank * 5 + index) % len(FINITE_PROFILE_NAMES),
                )
            )

    # Shapes built purely from odd/prime dimensions.
    for index, shape in enumerate(ODD_FACTOR_SHAPES):
        candidates.append(
            CaseSeed(
                category=f"odd_{index:02d}",
                shape=shape,
                profile_index=(index + 6) % len(FINITE_PROFILE_NAMES),
            )
        )

    seeds: List[CaseSeed] = []
    for candidate in candidates:
        if candidate.shape in used_shapes:
            continue
        used_shapes.add(candidate.shape)
        seeds.append(candidate)
        if len(seeds) == EXTRA_SEED_COUNT:
            return seeds

    # Deterministic top-up so the suite always lands on exactly EXTRA_SEED_COUNT seeds.
    probe = 5003
    while len(seeds) < EXTRA_SEED_COUNT:
        rank = 1 + probe % 8
        shape = _factor_shape(rank, probe, probe)
        if shape not in used_shapes and probe <= MAX_ELEMENTS:
            used_shapes.add(shape)
            seeds.append(
                CaseSeed(f"topup_n{probe}", shape, probe % len(FINITE_PROFILE_NAMES))
            )
        probe += 7
    return seeds


def _build_seeds() -> List[CaseSeed]:
    seeds = []

    # One representative for every Host tiling core count from 1 through 56.
    for core_count in range(1, MAX_CORE_COUNT + 1):
        if core_count == 1:
            element_count = 127
        else:
            element_count = (
                TILE_ELEMENTS * core_count + (core_count * 17) % TILE_ELEMENTS
            )
        seeds.append(
            CaseSeed(
                category=f"core_{core_count:02d}",
                shape=(element_count,),
                profile_index=(core_count - 1) % len(FINITE_PROFILE_NAMES),
            )
        )

    # Vector, DMA alignment and low-element-count boundaries.
    critical_elements = (
        1,
        2,
        3,
        7,
        8,
        15,
        16,
        31,
        32,
        33,
        63,
        64,
        65,
        66,
        95,
        96,
        97,
        128,
        129,
        191,
        192,
        193,
        255,
        256,
        257,
        4097,
    )
    for index, element_count in enumerate(critical_elements):
        seeds.append(
            CaseSeed(
                category=f"vector_n{element_count}",
                shape=(element_count,),
                profile_index=(index + 5) % len(FINITE_PROFILE_NAMES),
            )
        )

    # With 56 cores, these totals put the longest core immediately below/at/above 1..4 tiles.
    for tile_count in range(1, 5):
        boundary = MAX_CORE_COUNT * TILE_ELEMENTS * tile_count
        for offset in (-1, 0, 1):
            element_count = boundary + offset
            seeds.append(
                CaseSeed(
                    category=f"tile_{tile_count}_{offset + 1}",
                    shape=(element_count,),
                    profile_index=(tile_count * 3 + offset) % len(FINITE_PROFILE_NAMES),
                )
            )
    seeds.extend(
        (
            CaseSeed("max_dim_minus_one", (MAX_DIMENSION - 1,), 9),
            CaseSeed("max_dim", (MAX_DIMENSION,), 10),
        )
    )
    assert len(seeds) == 96

    # Each higher rank contributes 11 skinny boundary shapes and 11 balanced/mixed shapes.
    skinny_elements = (1, 63, 64, 65, 127, 128, 129, 3583, 3584, 3585, MAX_DIMENSION)
    for rank in range(2, 9):
        for index, element_count in enumerate(skinny_elements):
            shape = tuple([1] * (rank - 1) + [element_count])
            seeds.append(
                CaseSeed(f"skinny_n{element_count}", shape, (rank + index) % 12)
            )
        for index, shape in enumerate(_balanced_shapes(rank)):
            seeds.append(
                CaseSeed(f"balanced_{index:02d}", shape, (rank * 3 + index) % 12)
            )

    assert len(seeds) == BASE_SEED_COUNT
    used_shapes = {seed.shape for seed in seeds}
    assert len(used_shapes) == BASE_SEED_COUNT

    seeds.extend(_build_extra_seeds(used_shapes))
    assert len(seeds) == SEED_COUNT
    assert len({seed.shape for seed in seeds}) == SEED_COUNT

    # Spread all exceptional-value profiles across ranks and element-count regimes.
    special_total = len(SPECIAL_PROFILE_NAMES) * SPECIAL_SEEDS_PER_PROFILE
    eligible_indexes = [
        index for index, seed in enumerate(seeds) if _elements(seed.shape) >= 65
    ]
    special_indexes = [
        eligible_indexes[index * (len(eligible_indexes) - 1) // (special_total - 1)]
        for index in range(special_total)
    ]
    assert len(set(special_indexes)) == special_total
    for offset, seed_index in enumerate(special_indexes):
        special_profile = SPECIAL_PROFILE_NAMES[offset % len(SPECIAL_PROFILE_NAMES)]
        seeds[seed_index] = replace(seeds[seed_index], special_profile=special_profile)
    return seeds


def _number(value: float) -> str:
    if value == 0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"
    return format(value, ".12g")


def _range_expression(values: Sequence[str]) -> str:
    suffix = "," if len(values) == 1 else ""
    return "(" + ", ".join(values) + suffix + ")"


def _scale_for_uniform(base_amplitude: float, element_count: int) -> float:
    # For a symmetric uniform input, E[sum(x^2)] ~= N*a^2/3. Keep large reductions moderate
    # so the FP32 standard's 1e-2 absolute-error cap remains meaningful.
    precision_amplitude = math.sqrt(3.0 * 8192.0 / element_count)
    return min(base_amplitude, precision_amplitude)


def _uniform_ranges(
    profile_index: int, element_count: int
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    amp5 = _scale_for_uniform(5.0, element_count)
    amp4 = _scale_for_uniform(4.0, element_count)
    amp1 = _scale_for_uniform(1.0, element_count)
    amp05 = _scale_for_uniform(0.5, element_count)
    constant = min(1.0, math.sqrt(8192.0 / element_count))

    profiles = (
        (
            (
                -_scale_for_uniform(5.0, element_count),
                _scale_for_uniform(5.0, element_count),
                0.0,
            ),
            (-amp4, amp4, 0.0),
        ),
        ((0.0, amp5, min(1e-6, amp5)), (-amp4, 0.0, max(-1e-6, -amp4))),
        ((-amp5, 0.0, max(-1e-6, -amp5)), (0.0, amp4, min(1e-6, amp4))),
        ((-1e-6, 1e-6, 0.0), (-1e-7, 1e-7, 0.0)),
        ((-amp5, amp5, 0.0), (-amp05, amp05, 0.0)),
        ((-amp05, amp05, 0.0), (-amp5, amp5, 0.0)),
        ((0.0, 0.0), (-amp4, amp4, 0.0)),
        ((-amp4, amp4, 0.0), (0.0, 0.0)),
        ((constant, constant), (-constant, -constant)),
        ((-0.684210526316 * amp5, amp5, 0.0), (-amp4, 0.5 * amp4, 0.0)),
        ((-0.01 * amp1, 0.01 * amp1, 0.0), (-0.001 * amp1, 0.001 * amp1, 0.0)),
        ((-amp1, amp1, -0.0, 0.0), (-amp05, amp05, -0.0, 0.0)),
    )
    selected = profiles[profile_index]
    return tuple(_number(value) for value in selected[0]), tuple(
        _number(value) for value in selected[1]
    )


def _normal_range(mean: float, sigma: float, element_count: int) -> Tuple[str, ...]:
    scale_cap = math.sqrt(8192.0 / element_count)
    adjusted_sigma = max(0.1, min(sigma, max(0.1, scale_cap)))
    mean_cap = max(0.0, scale_cap * 0.5)
    adjusted_mean = max(-mean_cap, min(mean, mean_cap))
    low = adjusted_mean - 3.0 * adjusted_sigma
    high = adjusted_mean + 3.0 * adjusted_sigma
    return (_number(low), _number(high), _number(adjusted_mean), "0.0")


def _normal_ranges(
    profile_index: int, element_count: int
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    x1_parameters, x2_parameters = NORMAL_PARAMETERS[profile_index]
    return (
        _normal_range(x1_parameters[0], x1_parameters[1], element_count),
        _normal_range(x2_parameters[0], x2_parameters[1], element_count),
    )


def _input_ranges(seed: CaseSeed, distribution: str) -> Tuple[str, str, str]:
    element_count = _elements(seed.shape)
    if distribution == "uniform":
        x1_range, x2_range = _uniform_ranges(seed.profile_index, element_count)
    else:
        x1_range, x2_range = _normal_ranges(seed.profile_index, element_count)

    profile_name = FINITE_PROFILE_NAMES[seed.profile_index]
    if seed.special_profile is not None:
        profile_name = seed.special_profile
        if seed.special_profile == "x1_nan":
            x1_range = ("nan", "nan")
        elif seed.special_profile == "x2_nan":
            x2_range = ("nan", "nan")
        elif seed.special_profile == "x1_posinf":
            x1_range = ("inf", "inf")
        elif seed.special_profile == "x2_posinf":
            x2_range = ("inf", "inf")
        elif seed.special_profile == "x1_neginf":
            x1_range = ("-inf", "-inf")
        elif seed.special_profile == "x2_neginf":
            x2_range = ("-inf", "-inf")
        elif seed.special_profile == "x1_mixed_nan":
            x1_range = x1_range + ("nan",)
        elif seed.special_profile == "x2_mixed_nan":
            x2_range = x2_range + ("nan",)
        elif seed.special_profile == "x1_mixed_inf":
            x1_range = x1_range + ("inf", "-inf")
        elif seed.special_profile == "x2_mixed_inf":
            x2_range = x2_range + ("inf", "-inf")
        elif seed.special_profile == "x1_inf_x2_nan":
            x1_range = ("inf", "inf")
            x2_range = ("nan", "nan")
        elif seed.special_profile == "x1_nan_x2_neginf":
            x1_range = ("nan", "nan")
            x2_range = ("-inf", "-inf")
        elif seed.special_profile == "both_zero":
            x1_range = ("0.0", "0.0")
            x2_range = ("0.0", "0.0")
        elif seed.special_profile == "subnormal_both":
            x1_range = ("1.40129846432e-45", "1.40129846432e-45")
            x2_range = ("-1.40129846432e-45", "-1.40129846432e-45")
        elif seed.special_profile == "x1_finite_overflow":
            x1_range = ("1.0e20", "1.0e20")
        elif seed.special_profile == "x2_finite_overflow":
            x2_range = ("-1.0e20", "-1.0e20")
        else:
            raise AssertionError(f"unknown special profile: {seed.special_profile}")

    expression = f"({_range_expression(x1_range)}, {_range_expression(x2_range)})"
    return expression, profile_name, f"{distribution}:{profile_name}"


def _case_name(
    distribution: str, case_index: int, seed: CaseSeed, profile_name: str
) -> str:
    distribution_code = "u" if distribution == "uniform" else "n"
    raw_name = (
        f"ssa_{distribution_code}_{case_index:03d}_r{len(seed.shape)}_"
        f"{seed.category}_{profile_name}"
    )
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw_name)


def _make_targeted_row(
    testcase_name: str,
    network_name: str,
    input_shapes: Tuple[Tuple[int, ...], Tuple[int, ...]],
    input_formats: Tuple[str, str],
    output_formats: Tuple[str, str],
    remark: str,
    input_ori_shapes: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
    input_ori_formats: Optional[Tuple[str, str]] = None,
    output_ori_shapes: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None,
    output_ori_formats: Optional[Tuple[str, str]] = None,
    input_data_ranges: Optional[str] = None,
) -> Dict[str, str]:
    row = {header: "" for header in CSV_HEADERS}
    row.update(
        {
            "testcase_name": testcase_name,
            "network_name": network_name,
            "op_name": "square_sum_all",
            "input_shapes": repr(input_shapes),
            "input_dtypes": repr(("float32", "float32")),
            "input_formats": repr(input_formats),
            "output_shapes": repr(((1,), (1,))),
            "output_dtypes": repr(("float32", "float32")),
            "output_formats": repr(output_formats),
            "attributes": "{}",
            "input_data_ranges": input_data_ranges
            or repr(((-4, 4, 0.0), (-4, 4, 0.0))),
            "precision_tolerances": repr(
                (
                    (FP32_RTOL, REQUIRED_MISMATCH_RATIO),
                    (FP32_RTOL, REQUIRED_MISMATCH_RATIO),
                )
            ),
            "absolute_precision": repr(FP32_ATOL),
            "output_inplace_indexes": "()",
            "output_shape_unknown_indexes": "()",
            "is_enabled": "True",
            "remark": remark,
            "priority": "1",
            "manual_input_binaries": "()",
            "manual_golden_binaries": "()",
        }
    )
    optional_fields = {
        "input_ori_shapes": input_ori_shapes,
        "input_ori_formats": input_ori_formats,
        "output_ori_shapes": output_ori_shapes,
        "output_ori_formats": output_ori_formats,
    }
    for field, value in optional_fields.items():
        if value is not None:
            row[field] = repr(value)
    return row


def _build_public_format_rows(
    network_name: str,
    testcase_prefix: str,
    shapes_by_format: Dict[str, Tuple[Tuple[int, ...], ...]],
    mode: str,
    profile_offset: int,
) -> List[Dict[str, str]]:
    rows = []
    for format_index, format_name in enumerate(PUBLIC_FORMATS):
        shapes = shapes_by_format[format_name]
        for case_index, (label, shape) in enumerate(
            zip(PUBLIC_FORMAT_CASE_LABELS, shapes), start=1
        ):
            profile_index = (case_index - 1 + format_index * 5 + profile_offset) % len(
                FINITE_PROFILE_NAMES
            )
            distribution = "uniform" if (case_index + profile_offset) % 2 else "normal"
            seed = CaseSeed("format", shape, profile_index)
            ranges, _, profile_key = _input_ranges(seed, distribution)
            element_count = _elements(shape)
            shape_pair = (shape, shape)
            rows.append(
                _make_targeted_row(
                    f"{testcase_prefix}_{format_name.lower()}_{case_index:02d}_{label}",
                    network_name,
                    shape_pair,
                    (format_name, format_name),
                    ("ND", "ND"),
                    f"mode={mode}; category=format; format={format_name}-to-ND; "
                    f"boundary={label}; profile={profile_key}; rank=4; elements={element_count}; "
                    f"expected_cores={_used_cores(element_count)}; "
                    f"max_core_tile_loops={_tile_loops(element_count)}; "
                    f"tiling_key={_tiling_key(element_count)}",
                    input_ori_shapes=shape_pair,
                    input_ori_formats=(format_name, format_name),
                    output_ori_shapes=((1,), (1,)),
                    output_ori_formats=("ND", "ND"),
                    input_data_ranges=ranges,
                )
            )
    return rows


def _build_targeted_rows() -> List[Dict[str, str]]:
    network_name = "square_sum_all_regression"
    rows = [
        _make_targeted_row(
            "ssa_x_001_rank0_scalar",
            network_name,
            ((), ()),
            ("ND", "ND"),
            ("ND", "ND"),
            "category=rank0; profile=scalar; rank=0; elements=1; expected_cores=1; "
            "对齐 canndev tiling 的 (GetDimNum()==0)?1 分支",
        ),
        _make_targeted_row(
            "ssa_x_005_fmt_nd_baseline",
            network_name,
            ((960,), (960,)),
            ("ND", "ND"),
            ("ND", "ND"),
            "category=format; profile=ND-baseline; elements=960; 公有格式路径的ND对照",
        ),
    ]
    rows.extend(
        _build_public_format_rows(
            network_name,
            "ssa_x_fmt",
            KERNEL_PUBLIC_FORMAT_SHAPES,
            "kernel",
            profile_offset=0,
        )
    )
    return rows


def _build_rows() -> List[Dict[str, str]]:
    seeds = _build_seeds()
    rows = []
    for distribution, priority in (("uniform", 1), ("normal", 2)):
        for case_index, seed in enumerate(seeds, start=1):
            element_count = _elements(seed.shape)
            ranges, profile_name, profile_key = _input_ranges(seed, distribution)
            shape_pair = (seed.shape, seed.shape)
            row = {header: "" for header in CSV_HEADERS}
            row.update(
                {
                    "testcase_name": _case_name(
                        distribution, case_index, seed, profile_name
                    ),
                    "network_name": "square_sum_all_regression",
                    "op_name": "square_sum_all",
                    "input_shapes": repr(shape_pair),
                    "input_dtypes": repr(("float32", "float32")),
                    "input_formats": repr(("ND", "ND")),
                    "output_shapes": repr(((1,), (1,))),
                    "output_dtypes": repr(("float32", "float32")),
                    "output_formats": repr(("ND", "ND")),
                    "attributes": "{}",
                    "input_data_ranges": ranges,
                    # close 回归档消费每对 (rtol, permitted mismatch ratio) 与下方 atol；
                    # 三方档由 TestSpec/CLI 改用 cross_check L1。
                    "precision_tolerances": repr(
                        (
                            (FP32_RTOL, REQUIRED_MISMATCH_RATIO),
                            (FP32_RTOL, REQUIRED_MISMATCH_RATIO),
                        )
                    ),
                    "absolute_precision": repr(FP32_ATOL),
                    "output_inplace_indexes": "()",
                    "output_shape_unknown_indexes": "()",
                    "is_enabled": "True",
                    "remark": (
                        f"distribution={distribution}; category={seed.category}; profile={profile_key}; "
                        f"rank={len(seed.shape)}; elements={element_count}; "
                        f"expected_cores={_used_cores(element_count)}; max_core_tile_loops={_tile_loops(element_count)}"
                    ),
                    "priority": str(priority),
                    "manual_input_binaries": "()",
                    "manual_golden_binaries": "()",
                }
            )
            rows.append(row)
    rows.extend(_build_targeted_rows())
    _audit(rows)
    return rows


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _audit_public_format_rows(
    rows: Sequence[Dict[str, str]],
    expected_shapes: Dict[str, Tuple[Tuple[int, ...], ...]],
) -> None:
    assert len(rows) == len(PUBLIC_FORMATS) * PUBLIC_FORMAT_CASES_PER_FORMAT
    for format_name in PUBLIC_FORMATS:
        format_rows = [
            row
            for row in rows
            if ast.literal_eval(row["input_formats"]) == (format_name, format_name)
        ]
        assert len(format_rows) == PUBLIC_FORMAT_CASES_PER_FORMAT
        shapes = [ast.literal_eval(row["input_shapes"])[0] for row in format_rows]
        assert tuple(shapes) == expected_shapes[format_name]
        assert tuple(_elements(shape) for shape in shapes) == PUBLIC_FORMAT_ELEMENTS
        assert all(len(shape) == 4 for shape in shapes)
        assert all(
            ast.literal_eval(row["input_shapes"])[0]
            == ast.literal_eval(row["input_shapes"])[1]
            for row in format_rows
        )
        assert all(
            ast.literal_eval(row["input_ori_shapes"])
            == ast.literal_eval(row["input_shapes"])
            for row in format_rows
        )
        assert all(
            ast.literal_eval(row["input_ori_formats"]) == (format_name, format_name)
            for row in format_rows
        )
        assert all(
            ast.literal_eval(row["output_formats"]) == ("ND", "ND")
            and ast.literal_eval(row["output_ori_formats"]) == ("ND", "ND")
            for row in format_rows
        )
        assert len({row["input_data_ranges"] for row in format_rows}) == len(
            format_rows
        )
        assert all(
            f"tiling_key={_tiling_key(_elements(shape))}" in row["remark"]
            for row, shape in zip(format_rows, shapes)
        )
        assert {_used_cores(count) for count in PUBLIC_FORMAT_ELEMENTS} == {
            1,
            2,
            16,
            56,
        }
        assert {_tile_loops(count) for count in PUBLIC_FORMAT_ELEMENTS} == {1, 2, 3}
        assert any(count % 64 == 0 for count in PUBLIC_FORMAT_ELEMENTS)
        assert any(count % 64 != 0 for count in PUBLIC_FORMAT_ELEMENTS)


def _audit(rows: Sequence[Dict[str, str]]) -> None:
    assert len(rows) == CASE_COUNT
    assert all(row["is_enabled"] == "True" for row in rows)
    names = [row["testcase_name"] for row in rows]
    assert len(set(names)) == CASE_COUNT

    base_rows = rows[:BASE_CASE_COUNT]
    targeted_rows = rows[BASE_CASE_COUNT:]
    assert len(targeted_rows) == TARGETED_CASE_COUNT
    parsed_shapes = [ast.literal_eval(row["input_shapes"])[0] for row in base_rows]
    parsed_ranges = [row["input_data_ranges"] for row in base_rows]
    semantic_keys = [
        (row["priority"], shape, ranges)
        for row, shape, ranges in zip(base_rows, parsed_shapes, parsed_ranges)
    ]
    assert len(set(semantic_keys)) == BASE_CASE_COUNT
    assert all(1 <= len(shape) <= 8 for shape in parsed_shapes)
    assert all(dimension >= 1 for shape in parsed_shapes for dimension in shape)
    assert all(1 <= _elements(shape) <= MAX_ELEMENTS for shape in parsed_shapes)

    priority_counts = Counter(row["priority"] for row in base_rows)
    assert priority_counts == {"1": BASE_CASE_COUNT // 2, "2": BASE_CASE_COUNT // 2}
    rank_counts = Counter(len(shape) for shape in parsed_shapes)
    assert set(rank_counts) == set(range(1, 9))
    assert all(count >= 40 for count in rank_counts.values())

    element_counts = [_elements(shape) for shape in parsed_shapes]
    assert set(_used_cores(count) for count in element_counts) == set(
        range(1, MAX_CORE_COUNT + 1)
    )
    assert set(range(1, MAX_TILE_LOOPS + 1)).issubset(
        set(_tile_loops(count) for count in element_counts)
    )

    # extraCoreCount = totalElements % usedCoreNum must reach both ends of its range.
    remainders = {
        (count % _used_cores(count), _used_cores(count)) for count in element_counts
    }
    assert any(remainder == 0 and cores > 1 for remainder, cores in remainders)
    assert any(remainder == cores - 1 and cores > 1 for remainder, cores in remainders)

    # baseCoreElements % 64 sweeps the within-tile vector tail.
    vector_tails = {(count // _used_cores(count)) % 64 for count in element_counts}
    assert len(vector_tails) >= 40 and 0 in vector_tails and 63 in vector_tails

    required_boundaries = {
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        TILE_ELEMENTS,
        TILE_ELEMENTS + 1,
        MAX_CORE_COUNT * TILE_ELEMENTS - 1,
        MAX_CORE_COUNT * TILE_ELEMENTS,
        MAX_CORE_COUNT * TILE_ELEMENTS + 1,
        MAX_DIMENSION - 1,
        MAX_DIMENSION,
        MAX_CORE_COUNT * TILE_ELEMENTS * MAX_TILE_LOOPS,
        MAX_ELEMENTS,
    }
    assert required_boundaries.issubset(set(element_counts))
    assert any(
        _is_power_of_two(dimension) and dimension > 1
        for shape in parsed_shapes
        for dimension in shape
    )
    assert any(
        _is_power_of_two(dimension + 1) and dimension > 1
        for shape in parsed_shapes
        for dimension in shape
    )

    # A leading, a middle and a trailing `1` must all occur in some rank>=3 shape.
    assert any(shape[0] == 1 and len(shape) >= 3 for shape in parsed_shapes)
    assert any(
        any(dimension == 1 for dimension in shape[1:-1])
        for shape in parsed_shapes
        if len(shape) >= 3
    )
    assert any(shape[-1] == 1 and len(shape) >= 3 for shape in parsed_shapes)

    for special_profile in SPECIAL_PROFILE_NAMES:
        assert (
            sum(f":{special_profile};" in row["remark"] for row in base_rows)
            == 2 * SPECIAL_SEEDS_PER_PROFILE
        )

    private_formats = {"FRACTAL_Z", "C1HWNCoC0", "NC1HWC0"}
    for row in rows:
        formats = set(ast.literal_eval(row["input_formats"]))
        formats.update(ast.literal_eval(row["output_formats"]))
        assert formats.isdisjoint(private_formats)
    _audit_public_format_rows(targeted_rows[2:], KERNEL_PUBLIC_FORMAT_SHAPES)


def _render(rows: Sequence[Dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _print_summary(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    base_rows = rows[:BASE_CASE_COUNT]
    shapes = [ast.literal_eval(row["input_shapes"])[0] for row in base_rows]
    element_counts = [_elements(shape) for shape in shapes]
    print(f"output={str(output_path)!r}")
    print(
        f"cases={len(rows)} uniform={BASE_CASE_COUNT // 2} normal={BASE_CASE_COUNT // 2} "
        f"targeted={TARGETED_CASE_COUNT} enabled={len(rows)} base_unique_shapes={len(set(shapes))}"
    )
    print(
        "rank_counts="
        + repr(dict(sorted(Counter(len(shape) for shape in shapes).items())))
    )
    print(
        f"core_counts=1-{MAX_CORE_COUNT} tile_loops={sorted(set(_tile_loops(n) for n in element_counts))}"
    )
    remainders = sorted({n % _used_cores(n) for n in element_counts})
    vector_tails = sorted({(n // _used_cores(n)) % 64 for n in element_counts})
    print(
        f"extra_core_counts={len(remainders)} values (min={remainders[0]} max={remainders[-1]})"
    )
    print(f"vector_tail_residues={len(vector_tails)} of 64 covered")
    print(
        f"element_range={min(element_counts)}-{max(element_counts)} "
        f"special_cases={2 * SPECIAL_SEEDS_PER_PROFILE * len(SPECIAL_PROFILE_NAMES)}"
    )


def _validated_output_path(raw_output: Path) -> Path:
    allowed_directory = Path(__file__).resolve().parent
    raw_text = str(raw_output)
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_text):
        raise SystemExit("output path contains control characters")
    candidate = raw_output.expanduser()
    if not candidate.is_absolute():
        candidate = allowed_directory / candidate
    if candidate.is_symlink():
        raise SystemExit("output path must not be a symbolic link")
    normalized = candidate.resolve(strict=False)
    if normalized.parent != allowed_directory or normalized.suffix.lower() != ".csv":
        raise SystemExit("output must be a .csv file in the generator directory")
    if len(normalized.name) > 255:
        raise SystemExit("output filename is too long")
    return normalized


def _write_output(output_path: Path, rendered: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(output_path, flags, 0o644)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output_file:
            descriptor = -1
            output_file.write(rendered)
    except OSError as error:
        if descriptor >= 0:
            os.close(descriptor)
        raise SystemExit(f"failed to write output CSV: {error.strerror}") from None


def main() -> None:
    default_output = Path(__file__).with_name("ttk_kernel_square_sum_all.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the checked-in CSV is up to date",
    )
    args = parser.parse_args()
    output_path = _validated_output_path(args.output)

    rows = _build_rows()
    rendered = _render(rows)
    if args.check:
        if (
            not output_path.is_file()
            or output_path.read_text(encoding="utf-8") != rendered
        ):
            raise SystemExit(f"generated CSV is stale: {str(output_path)!r}")
        print("generated_csv_is_current=yes")
    else:
        _write_output(output_path, rendered)
    _print_summary(rows, output_path)


if __name__ == "__main__":
    main()
