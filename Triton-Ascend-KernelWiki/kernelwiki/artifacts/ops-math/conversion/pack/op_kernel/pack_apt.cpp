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
 * \file pack.cpp
 * \brief pack ascendc impl
 */
#include "../concat/arch35/one_axis_concat_all_align.h"
#include "../concat/arch35/one_axis_concat_no_align_same_shape_copy.h"
#include "../concat/arch35/one_axis_concat_no_align_same_shape_gather.h"
#include "../concat/arch35/one_axis_concat_pure_copy.h"
#include "../concat/arch35/one_axis_concat_simt.h"

#define NOTFIRST_ALIGN_SAME_BITWIDTH_1 2111
#define NOTFIRST_ALIGN_SAME_BITWIDTH_2 2112
#define NOTFIRST_ALIGN_SAME_BITWIDTH_4 2114
#define NOTFIRST_ALIGN_SAME_BITWIDTH_8 2118

#define NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1 2211
#define NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2 2212
#define NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4 2214

#define NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1 2311
#define NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2 2312
#define NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4 2314

#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_1 12111
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_2 12112
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_4 12114
#define SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_8 12118

#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1 12211
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2 12212
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4 12214

#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1 12311
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2 12312
#define SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4 12314
#define PURE_COPY_NO_SPLIT_DIM1_TILINGKEY 20001
#define PURE_COPY_SPLIT_DIM1_TILINGKEY 20002
#define SIMT_TILINGKEY_1 30001
#define SIMT_TILINGKEY_2 30002
#define SIMT_TILINGKEY_4 30004
#define SIMT_TILINGKEY_8 30008

using namespace Concat;

extern "C" __global__ __aicore__ void pack(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    // simt 模板入口
    if (TILING_KEY_IS(SIMT_TILINGKEY_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
        Concat::OneAxisConcatSimt<uint8_t> op(tilingData);
        op.ProcessForSimt(x, y);
        return;
    } else if (TILING_KEY_IS(SIMT_TILINGKEY_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
        Concat::OneAxisConcatSimt<uint16_t> op(tilingData);
        op.ProcessForSimt(x, y);
        return;
    } else if (TILING_KEY_IS(SIMT_TILINGKEY_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
        Concat::OneAxisConcatSimt<uint32_t> op(tilingData);
        op.ProcessForSimt(x, y);
        return;
    } else if (TILING_KEY_IS(SIMT_TILINGKEY_8)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataForSimt, tilingData, tiling);
        Concat::OneAxisConcatSimt<uint64_t> op(tilingData);
        op.ProcessForSimt(x, y);
        return;
    }

    // 千位 首轴concat/非首轴 1/2
    // 百位 输入全对齐/非全对齐 1/2
    // 十位 input shape相同/不相同 1/2
    // 个位 位宽 1/2/4/8
    TPipe pipe;
    if (TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int8_t, true> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int16_t, true> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int32_t, true> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_ALIGN_SAME_BITWIDTH_8)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int64_t, true> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatNoAlignCopy<uint8_t> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatNoAlignCopy<uint16_t> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatNoAlignCopy<uint32_t> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatNoAlignGather<uint8_t, uint16_t> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatNoAlignGather<uint16_t, uint16_t> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatNoAlignGather<uint32_t, uint32_t> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int8_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int16_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int32_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_ALIGN_SAME_BITWIDTH_8)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatAllAlign<int64_t, true, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatNoAlignCopy<uint8_t, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatNoAlignCopy<uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_COPY_BITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatNoAlignCopy<uint32_t, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatNoAlignGather<uint8_t, uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatNoAlignGather<uint16_t, uint16_t, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(SPLIT_CORE_DIM0_NOTFIRST_NALIGN_SAME_GATHER_BITWIDTH_4)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatNoAlignGather<uint32_t, uint32_t, ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(PURE_COPY_NO_SPLIT_DIM1_TILINGKEY)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingDataNoArray, tilingData, tiling);
        Concat::OneAxisConcatPureCopy<ConcatTilingDataNoArray> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(PURE_COPY_SPLIT_DIM1_TILINGKEY)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        Concat::OneAxisConcatPureCopy<ConcatTilingData> op(tilingData, pipe);
        op.Init(x, y);
        op.Process();
    }
}