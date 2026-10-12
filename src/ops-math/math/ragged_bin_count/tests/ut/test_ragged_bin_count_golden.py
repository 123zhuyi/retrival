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

"""Regression tests for the RaggedBinCount TestSpec golden."""

import importlib.util
from pathlib import Path

import numpy as np


_GOLDEN_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location("ragged_bin_count_golden", _GOLDEN_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _golden(output_dtype):
    return _MODULE.RaggedBinCountKernelSpec.golden(
        np.asarray([0, 2], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0, 2.0**-24], dtype=np.float64),
        binary_output=False,
        output_dtypes=(output_dtype,),
        testcase_name="golden_dtype_regression",
    )[0]


def _special_golden(weights, output_dtype, input_dtype=np.float32):
    weights = np.asarray(weights, dtype=input_dtype)
    return _MODULE.RaggedBinCountKernelSpec.golden(
        np.asarray([0, weights.size], dtype=np.int64),
        np.zeros(weights.size, dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        weights,
        binary_output=False,
        output_dtypes=(output_dtype,),
        testcase_name="rbc_k_precision_p0_twosum_fp64_gap_i64",
    )[0]


def test_cross_check_promote_preserves_fp64_reference():
    result = _golden("float64")
    expected = np.float64(1.0) + np.float64(2.0**-24)

    assert result.dtype == np.float64
    assert result[0, 0] == expected


def test_close_rounds_once_to_declared_fp32_output():
    result = _golden("float32")
    expected = np.float32(np.float64(1.0) + np.float64(2.0**-24))

    assert result.dtype == np.float32
    assert result[0, 0] == expected


def test_special_cross_check_rounds_exact_sum_directly_to_fp64():
    result = _special_golden([1.0, 2.0**-24], "float64", np.float64)
    expected = np.float64(1.0) + np.float64(2.0**-24)

    assert result.dtype == np.float64
    assert result[0, 0] == expected


def test_special_close_rounds_exact_sum_once_to_fp32():
    result = _special_golden([1.0, 2.0**-24], "float32")

    assert result.dtype == np.float32
    assert result[0, 0] == np.float32(1.0)


def test_special_exact_sum_retains_wide_cancellation_gap():
    result = _special_golden([2.0**100, 1.0, -(2.0**100)], "float64", np.float64)

    assert result.dtype == np.float64
    assert result[0, 0] == np.float64(1.0)


def test_special_promote_keeps_finite_truth_beyond_fp32_range():
    maximum = np.finfo(np.float32).max
    result = _special_golden([maximum, maximum], "float64", np.float64)

    assert result.dtype == np.float64
    assert result[0, 0] == np.float64(maximum) * np.float64(2.0)


def test_special_nonfinite_semantics_are_preserved():
    result = _special_golden([float("inf"), -float("inf")], "float64", np.float64)

    assert result.dtype == np.float64
    assert np.isnan(result[0, 0])


def test_special_fp64_rounding_uses_ties_to_even():
    tie = _special_golden([2.0**100, 2.0**47], "float64", np.float64)
    above = _special_golden([2.0**100, 2.0**47, 2.0**46], "float64", np.float64)

    assert tie[0, 0] == np.float64(2.0**100)
    assert above[0, 0] == np.nextafter(np.float64(2.0**100), np.float64(np.inf))
