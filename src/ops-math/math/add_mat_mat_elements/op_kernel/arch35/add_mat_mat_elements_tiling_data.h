/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_mat_mat_elements_tiling_data.h
 * \brief AddMatMatElements TilingData structure (arch35, Ascend950)
 */

#ifndef ADD_MAT_MAT_ELEMENTS_TILING_DATA_H_
#define ADD_MAT_MAT_ELEMENTS_TILING_DATA_H_

#include <cstdint>

constexpr int64_t ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS = 3;  // c, a, b
constexpr int64_t ADD_MAT_MAT_ELEMENTS_MAX_OUTPUT_SLOTS = 1; // c_out
constexpr int64_t ADD_MAT_MAT_ELEMENTS_PHYS_NODES = 4;
constexpr int64_t ADD_MAT_MAT_ELEMENTS_RANK_MAX = 8; // max supported rank

struct AddMatMatElementsSplitResult {
    int64_t axis;     // UB split axis (inner-to-outer search)
    int64_t a_i;      // inner-axis tile size (elements)
    int64_t a_o;      // outer-axis tile count
    int64_t a_i_tail; // last-tile size (elements)
};

struct AddMatMatElementsMultiCoreResult {
    int64_t num_cores;   // participating cores (from GetCoreNumAiv)
    int64_t total_tiles; // total tiles
    int64_t tiles_main;  // tiles per core (main)
    int64_t cores_tail;  // cores with one extra tile
};

// beta/alpha are (1,) scalar tensors read from GM directly (not in TilingData).
struct AddMatMatElementsTilingData {
    AddMatMatElementsSplitResult split;                   // UB split
    AddMatMatElementsMultiCoreResult multicore;           // multi-core split
    int64_t rank;                                         // effective rank (1~8)
    int64_t per_buf_bytes;                                // per-buffer bytes = (ubAvailable / P) & ~31
    int64_t max_bro_shape[ADD_MAT_MAT_ELEMENTS_RANK_MAX]; // broadcast shape (= c.shape)
    int64_t num_inputs;                                   // broadcast tensor inputs (=3: c,a,b)
    int64_t num_outputs;                                  // outputs (=1)
    int64_t input_shapes[ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS][ADD_MAT_MAT_ELEMENTS_RANK_MAX]; // [0]=c,[1]=a,[2]=b
    int64_t input_strides[ADD_MAT_MAT_ELEMENTS_MAX_INPUT_SLOTS]
                         [ADD_MAT_MAT_ELEMENTS_RANK_MAX]; // GM strides (broadcast axis=0)
    int64_t output_shapes[ADD_MAT_MAT_ELEMENTS_MAX_OUTPUT_SLOTS]
                         [ADD_MAT_MAT_ELEMENTS_RANK_MAX]; // c_out.shape = c.shape
    int64_t output_strides[ADD_MAT_MAT_ELEMENTS_MAX_OUTPUT_SLOTS][ADD_MAT_MAT_ELEMENTS_RANK_MAX]; // c_out GM strides
};

#endif // ADD_MAT_MAT_ELEMENTS_TILING_DATA_H_
