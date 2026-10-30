/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please
 * refer to the License for details. You may not use this file except in compliance with the License. THIS SOFTWARE IS
 * PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
 * repository for the full text of the License.
 */

/*!
 * \file split_v.cpp
 * \brief
 */

#include "split_v.h"

template <uint32_t schMode>
__global__ __aicore__ void split_v(GM_ADDR x, GM_ADDR sizeSplits, GM_ADDR splitDim, GM_ADDR y, GM_ADDR workSpace,
                                   GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SplitVTilingData);
    TPipe pipe;
    if constexpr (schMode == TILING_KEY_SPLIT_V_PURE_COPY) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataPureCopy, tiling_data, tiling);
        SplitVPureCopy<DTYPE_X> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_ONE_ROW_PURE_COPY) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataOneRowPureCopy, tiling_data, tiling);
        SplitVOneRowPureCopy<DTYPE_X> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_ONE_OUTER) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tiling_data, tiling);
        SplitV<DTYPE_X> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_GENERAL) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingData, tiling_data, tiling);
        SplitVGeneral<DTYPE_X> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_COMPACT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenCompact, tiling_data, tiling);
        SplitVSameLenCompact<half> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_LARGE_OUTER) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenCompact, tiling_data, tiling);
        SplitVSameLenCompactLargeOuter<half> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_DOUBLE_BUFFER) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenCompact, tiling_data, tiling);
        SplitVSameLenCompactDoubleBuffer<half> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_UNEVEN_COMPACT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataUnevenCompact, tiling_data, tiling);
        SplitVUnevenCompact<half> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_8BIT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenCompact, tiling_data, tiling);
        SplitVSameLenCompact8Bit<int8_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_PURE_COPY_8BIT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenPureCopy8Bit, tiling_data, tiling);
        SplitVSameLenPureCopy8Bit<int8_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_UNEVEN_COMPACT_8BIT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataUnevenCompact, tiling_data, tiling);
        SplitVUnevenCompact8Bit<int8_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_UNEVEN_PURE_COPY_16BIT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataUnevenPureCopy16Bit, tiling_data, tiling);
        SplitVUnevenPureCopy16Bit<half> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_INNER_COPY) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenInnerCopy, tiling_data, tiling);
        SplitVSameLenInnerCopy<DTYPE_X> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_UNEVEN_INNER_ALIGNED_MID) {
        GET_TILING_DATA_PTR_WITH_STRUCT(SplitVTilingDataUnevenInnerAlignedMid, tilingDataPtr, tiling);
        SplitVUnevenLenInnerCopy<DTYPE_X> op;
        op.Init(x, y, tilingDataPtr, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_COMPACT_32BIT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLenCompact, tiling_data, tiling);
        SplitVSameLenCompact<uint16_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_UNEVEN_COMPACT_32BIT) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataUnevenCompact, tiling_data, tiling);
        SplitVUnevenCompact<uint16_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN_PURE_COPY_WIDE) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLen, tiling_data, tiling);
        SplitVSameLen<uint16_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_UNEVEN_PURE_COPY_WIDE) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataUnevenPureCopy16Bit, tiling_data, tiling);
        SplitVUnevenPureCopy16Bit<uint16_t> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    } else if constexpr (schMode == TILING_KEY_SPLIT_V_SAME_LEN) {
        GET_TILING_DATA_WITH_STRUCT(SplitVTilingDataSameLen, tiling_data, tiling);
        SplitVSameLen<DTYPE_X> op;
        op.Init(x, y, &tiling_data, &pipe);
        op.Process();
    }
}
