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
 * \file sort_with_index.cpp
 * \brief sort_with_index cc impl
 */

#include "arch35/merge_sort_with_index.h"
#include "arch35/radix_sort_with_index_multi_block.h"
#include "arch35/radix_sort_with_index_single_block.h"

using namespace AscendC;
using namespace SortWithIndex;

const uint32_t SMALL_SIZE_MODE = 1;

template <typename XType, typename ConvertType, typename UnsignedType, typename IndexType>
__aicore__ inline void generateOpObject(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex, GM_ADDR globalWorkGm,
                                        GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    bool isDescend = tilingData.isDescend > 0 ? true : false;
    bool isInt32Range = tilingData.isInInt32Range == 1 ? true : false;
    bool isSingleBlock = tilingData.lastDimNeedCore == 1 ? true : false;
    uint32_t modeType = tilingData.modeType;
    TPipe pipe;
    if (isSingleBlock && SMALL_SIZE_MODE == modeType) {
        if (isDescend) {
            RadixSortWithIndexSingleBlock<XType, true, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, &tilingData, &pipe);
            radixSort.Process();
        } else {
            RadixSortWithIndexSingleBlock<XType, false, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, &tilingData, &pipe);
            radixSort.Process();
        }
        return;
    }

    if (isInt32Range) {
        if (isDescend) {
            RadixSortWithIndexMultiBlock<XType, UnsignedType, true, uint32_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, &tilingData, &pipe);
            radixSort.Process();
        } else {
            RadixSortWithIndexMultiBlock<XType, UnsignedType, false, uint32_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, &tilingData, &pipe);
            radixSort.Process();
        }
    } else {
        if (isDescend) {
            RadixSortWithIndexMultiBlock<XType, UnsignedType, true, int64_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, &tilingData, &pipe);
            radixSort.Process();
        } else {
            RadixSortWithIndexMultiBlock<XType, UnsignedType, false, int64_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, &tilingData, &pipe);
            radixSort.Process();
        }
    }
    
}

// todo MergeSort适配Int64
template <typename XType, typename ConvertType, typename IndexType>
__aicore__ inline void generateMergeSortObject(GM_ADDR x, GM_ADDR index, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm,
                                               GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isDescend = (tilingData.isDescend > 0) ? true : false;
    TPipe pipe;
    if (isDescend) {
        MergeSortWithIndex<XType, ConvertType, SortWithIndexTilingDataSimt, true, IndexType> mergeSort;
        mergeSort.Init(x, index, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    } else {
        MergeSortWithIndex<XType, ConvertType, SortWithIndexTilingDataSimt, false, IndexType> mergeSort;
        mergeSort.Init(x, index, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    }
}

extern "C" __global__ __aicore__ void sort_with_index(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR globalWorkGm = GetUserWorkspace(workspace);
    if (globalWorkGm == nullptr) {
        return;
    }
    if (TILING_KEY_IS(1001)) {
        generateOpObject<int8_t, uint8_t, uint8_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(1002)) {
        generateOpObject<int16_t, uint16_t, uint16_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(1003)) {
        generateOpObject<int32_t, uint32_t, uint32_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(1004)) {
        generateOpObject<int64_t, uint64_t, uint64_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(2001)) {
        generateOpObject<uint8_t, uint8_t, uint8_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(2004)) {
        generateOpObject<uint64_t, uint64_t, uint64_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(2003)) {
        generateOpObject<uint32_t, uint32_t, uint32_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(2002)) {
        generateOpObject<uint16_t, uint16_t, uint16_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(3003)) {
        generateOpObject<float, float, uint32_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(3002)) {
        generateOpObject<half, half, uint16_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(4002)) {
        generateOpObject<bfloat16_t, float, uint16_t, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(13003)) {
        generateMergeSortObject<float, float, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(13002)) {
        generateMergeSortObject<half, half, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    } else if (TILING_KEY_IS(14002)) {
        generateMergeSortObject<bfloat16_t, float, DTYPE_INDEX>(x, index, y, sortedIndex, globalWorkGm, tiling);
    }
}