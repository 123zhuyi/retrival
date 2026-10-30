/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_v2.cpp
 * \brief
 */

#define K_MAX_SHAPE_DIM 0
#include "../concat/arch35/one_axis_concat_all_align.h"
#include "../concat/arch35/one_axis_concat_no_align_same_shape_copy.h"
#include "../concat/arch35/one_axis_concat_no_align_same_shape_gather.h"
#include "../concat/arch35/one_axis_concat_no_align_diff_shape.h"
#include "../concat/arch35/one_axis_concat_pure_copy.h"
#include "../concat/arch35/one_axis_concat_simt.h"

#define NOTFIRST_ALIGN_SAME_BITWIDTH_1 2111
#define NOTFIRST_ALIGN_SAME_BITWIDTH_2 2112
#define NOTFIRST_ALIGN_SAME_BITWIDTH_4 2114
#define NOTFIRST_ALIGN_SAME_BITWIDTH_8 2118

#define NOTFIRST_ALIGN_DIFF_BITWIDTH_1 2121
#define NOTFIRST_ALIGN_DIFF_BITWIDTH_2 2122
#define NOTFIRST_ALIGN_DIFF_BITWIDTH_4 2124
#define NOTFIRST_ALIGN_DIFF_BITWIDTH_8 2128

#define NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1 2211
#define NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2 2212
#define NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4 2214

#define NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1 2311
#define NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2 2312
#define NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4 2314

#define NOTFIRST_NOALIGN_DIFF_BITWIDTH_1 2221
#define NOTFIRST_NOALIGN_DIFF_BITWIDTH_2 2222
#define NOTFIRST_NOALIGN_DIFF_BITWIDTH_4 2224
#define NOTFIRST_NOALIGN_DIFF_BITWIDTH_8 2228

#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_1 12111
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_2 12112
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_4 12114
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_8 12118

#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_1 12121
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_2 12122
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_4 12124
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_8 12128

#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1 12211
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2 12212
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4 12214

#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1 12311
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2 12312
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4 12314

#define SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_1 12221
#define SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_2 12222
#define SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_4 12224
#define SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_8 12228
#define PURE_COPY_NO_SPLIT_DIM1_TILINGKEY 20001
#define PURE_COPY_SPLIT_DIM1_TILINGKEY 20002
#define SIMT_TILINGKEY_1 30001
#define SIMT_TILINGKEY_2 30002
#define SIMT_TILINGKEY_4 30004
#define SIMT_TILINGKEY_8 30008

using namespace Concat;

extern "C" __global__ __aicore__ void concat_v2(GM_ADDR x, GM_ADDR dim, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    #if ORIG_DTYPE_X == DT_UINT8 || ORIG_DTYPE_X == DT_INT8 || ORIG_DTYPE_X == DT_BOOL || ORIG_DTYPE_X == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X == DT_FLOAT8_E5M2 || ORIG_DTYPE_X == DT_HIFLOAT8 || ORIG_DTYPE_X == DT_FLOAT8_E8M0 || ORIG_DTYPE_X == DT_BF16 || ORIG_DTYPE_X == DT_FLOAT16 || ORIG_DTYPE_X == DT_INT16 || ORIG_DTYPE_X == DT_UINT16
        TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_1);
        TILING_KEY_IS(NOTFIRST_ALIGN_DIFF_BITWIDTH_1);
        TILING_KEY_IS(NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1);
        TILING_KEY_IS(NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1);
        TILING_KEY_IS(NOTFIRST_NOALIGN_DIFF_BITWIDTH_1);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_1);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_1);
        TILING_KEY_IS(SIMT_TILINGKEY_1);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_1);
        TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_2);
        TILING_KEY_IS(NOTFIRST_ALIGN_DIFF_BITWIDTH_2);
        TILING_KEY_IS(NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2);
        TILING_KEY_IS(NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2);
        TILING_KEY_IS(NOTFIRST_NOALIGN_DIFF_BITWIDTH_2);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_2);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_2);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_2);
        TILING_KEY_IS(SIMT_TILINGKEY_2);
        TILING_KEY_IS(PURE_COPY_NO_SPLIT_DIM1_TILINGKEY);
        TILING_KEY_IS(PURE_COPY_SPLIT_DIM1_TILINGKEY);
        #if TILING_KEY_VAR == SIMT_TILINGKEY_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
            Concat::OneAxisConcatSimt<uint8_t> op(tilingData);
            op.ProcessForSimt(x, y);
            return;
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_SAME_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int8_t, true> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_DIFF_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int8_t, false> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignCopy<uint8_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignGather<uint8_t, uint16_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NOALIGN_DIFF_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint8_t, uint16_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int8_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int8_t, false, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignCopy<uint8_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignGather<uint8_t, uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_1
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint8_t, uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SIMT_TILINGKEY_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
            Concat::OneAxisConcatSimt<uint16_t> op(tilingData);
            op.ProcessForSimt(x, y);
            return;
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_SAME_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int16_t, true> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_DIFF_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int16_t, false> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignCopy<uint16_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NOALIGN_DIFF_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint16_t, uint16_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignGather<uint16_t, uint16_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int16_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int16_t, false, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignCopy<uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignGather<uint16_t, uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_2
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint16_t, uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == PURE_COPY_NO_SPLIT_DIM1_TILINGKEY
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatPureCopy<ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == PURE_COPY_SPLIT_DIM1_TILINGKEY
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatPureCopy<ConcatTilingData> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_UINT32 || ORIG_DTYPE_X == DT_INT32 || ORIG_DTYPE_X == DT_FLOAT || ORIG_DTYPE_X == DT_UINT64 || ORIG_DTYPE_X == DT_INT64 || ORIG_DTYPE_X == DT_DOUBLE || ORIG_DTYPE_X == DT_COMPLEX64
        TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_4);
        TILING_KEY_IS(NOTFIRST_ALIGN_DIFF_BITWIDTH_4);
        TILING_KEY_IS(NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4);
        TILING_KEY_IS(NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4);
        TILING_KEY_IS(NOTFIRST_NOALIGN_DIFF_BITWIDTH_4);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_4);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_4);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_4);
        TILING_KEY_IS(SIMT_TILINGKEY_4);
        TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_8);
        TILING_KEY_IS(NOTFIRST_ALIGN_DIFF_BITWIDTH_8);
        TILING_KEY_IS(NOTFIRST_NOALIGN_DIFF_BITWIDTH_8);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_8);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_8);
        TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_8);
        TILING_KEY_IS(SIMT_TILINGKEY_8);
        TILING_KEY_IS(PURE_COPY_NO_SPLIT_DIM1_TILINGKEY);
        TILING_KEY_IS(PURE_COPY_SPLIT_DIM1_TILINGKEY);
        #if TILING_KEY_VAR == SIMT_TILINGKEY_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
            Concat::OneAxisConcatSimt<uint32_t> op(tilingData);
            op.ProcessForSimt(x, y);
            return;
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_SAME_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int32_t, true> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_DIFF_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int32_t, false> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignCopy<uint32_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignGather<uint32_t, uint32_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NOALIGN_DIFF_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint32_t, uint32_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int32_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int32_t, false, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignCopy<uint32_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignGather<uint32_t, uint32_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_4
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint32_t, uint32_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SIMT_TILINGKEY_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
            Concat::OneAxisConcatSimt<uint64_t> op(tilingData);
            op.ProcessForSimt(x, y);
            return;
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_SAME_BITWIDTH_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int64_t, true> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_ALIGN_DIFF_BITWIDTH_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int64_t, false> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == NOTFIRST_NOALIGN_DIFF_BITWIDTH_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint32_t, uint32_t> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int64_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_ALIGN_DIFF_BITWIDTH_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatAllAlign<int64_t, false, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == SPLIT_CORE_DIM0_NOTFIRST_NOALIGN_DIFF_BITWIDTH_8
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatNoAlignDiffShape<uint32_t, uint32_t, ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == PURE_COPY_NO_SPLIT_DIM1_TILINGKEY
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
            Concat::OneAxisConcatPureCopy<ConcatTilingDataNoArray> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #elif TILING_KEY_VAR == PURE_COPY_SPLIT_DIM1_TILINGKEY
            GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
            Concat::OneAxisConcatPureCopy<ConcatTilingData> op(tilingData, pipe);
            op.Init(x, y);
            op.Process();
        #endif
    #endif
}