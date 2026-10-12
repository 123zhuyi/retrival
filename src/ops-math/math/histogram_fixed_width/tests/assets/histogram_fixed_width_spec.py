#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np

__spec__ = {"histogram_fixed_width": "HistogramFixedWidthTestSpec"}


class HistogramFixedWidthTestSpec:
    def golden(x, value_range, nbins, **kwargs):
        output_dtypes = kwargs.get("output_dtypes", ["int32"])
        out_dtype = output_dtypes[0] if output_dtypes else "int32"

        nbins_val = nbins[0]
        min_data = value_range[0]
        max_data = value_range[1]
        x_dtype = x.dtype.name

        if x_dtype in ("float16", "float32"):
            min_data = np.float32(min_data)
            max_data = np.float32(max_data)
            x = x.astype(np.float32)
        elif x_dtype == "int32":
            min_data = np.int32(min_data)
            max_data = np.int32(max_data)
            x = x.astype(np.int32)
        else:
            min_data = np.int64(min_data)
            max_data = np.int64(max_data)
            x = x.astype(np.int64)

        res = np.zeros(nbins_val, dtype=out_dtype)

        if x_dtype in ("float16", "float32"):
            is_min_neg_inf = min_data == np.float32(-np.inf)
            is_max_pos_inf = max_data == np.float32(np.inf)

            if is_min_neg_inf or is_max_pos_inf:
                valid = x[~((x == np.float32(np.inf)) | np.isnan(x))]
                if is_min_neg_inf:
                    res[nbins_val - 1] = valid.size
                else:
                    res[0] = valid.size
                return [res]

            mask = ~(
                (x == np.float32(np.inf)) | np.isnan(x) | (x == np.float32(-np.inf))
            )
            x_clean = x[mask]
            if x_clean.size == 0:
                return [res]

            range_val = np.float32(max_data - min_data)
            inv_range = np.float32(np.float32(1.0) / range_val)
            bins_f32 = np.float32(nbins_val)

            x_clamped = np.clip(x_clean, min_data, max_data)
            diff = np.float32(x_clamped - min_data)
            scaled = np.float32(diff * inv_range * bins_f32)
            idx = scaled.astype(np.int32)
            idx = np.clip(idx, 0, nbins_val - 1)

            counts = np.bincount(idx.ravel(), minlength=nbins_val)
            res[: len(counts)] = counts[:nbins_val].astype(out_dtype)
        else:
            compute_dtype = np.int64
            min_c = compute_dtype(min_data)
            max_c = compute_dtype(max_data)
            range_c = compute_dtype(max_data - min_data)
            if range_c != 0 and x.size > 0:
                x_c = x.astype(compute_dtype)
                x_clamped = np.clip(x_c, min_c, max_c)
                idx = ((x_clamped - min_c) * compute_dtype(nbins_val) / range_c).astype(
                    np.int32
                )
                idx = np.clip(idx, 0, nbins_val - 1)
                counts = np.bincount(idx.ravel(), minlength=nbins_val)
                res[: len(counts)] = counts[:nbins_val].astype(out_dtype)

        return [res]

    def customize_inputs(x, value_range, nbins, **kwargs):
        range_dtype = value_range.dtype
        x_f32 = x.astype(np.float32) if x.dtype == np.float16 else x
        lo = (
            float(value_range[0].astype(np.float32))
            if range_dtype == np.float16
            else float(value_range[0])
        )
        hi = (
            float(value_range[1].astype(np.float32))
            if range_dtype == np.float16
            else float(value_range[1])
        )

        if lo >= hi:
            finite_mask = np.isfinite(x_f32)
            if np.any(finite_mask):
                lo = float(np.min(x_f32[finite_mask]))
                hi = float(np.max(x_f32[finite_mask]))
            else:
                lo = -1.0
                hi = 1.0

        if lo >= hi:
            lo = lo - 1
            hi = hi + 1

        input_range = np.zeros(2, range_dtype)
        input_range[0] = lo
        input_range[1] = hi

        if x.dtype in (np.float16, np.float32):
            nbins_val = int(nbins[0])
            range_f64 = np.float64(hi) - np.float64(lo)
            if range_f64 > 0 and np.isfinite(range_f64):
                x_f64 = x.astype(np.float64)
                scaled = (x_f64 - np.float64(lo)) * np.float64(nbins_val) / range_f64
                scaled = np.clip(scaled, 0.0, np.float64(nbins_val))
                x = scaled.astype(x.dtype)
                input_range[0] = 0
                input_range[1] = nbins_val
                if range_dtype == np.float16:
                    input_range = input_range.astype(np.float16)
                    x = np.clip(x, np.float16(0), input_range[1])

        return (x, input_range, nbins)
