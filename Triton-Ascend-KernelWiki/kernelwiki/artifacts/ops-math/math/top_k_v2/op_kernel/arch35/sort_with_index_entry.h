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
* \file sort_with_index_entry.h
* \brief sort_with_index kernel entry impl
*/

#ifndef SORT_WITH_INDEX_ENTRY_H
#define SORT_WITH_INDEX_ENTRY_H

#include "sort_with_index_merge_sort.h"
#include "sort_with_index_multi_block.h"
#include "sort_with_index_single_block.h"

using namespace AscendC;

const uint32_t SORT_WITH_INDEX_SMALL_SIZE_MODE = 1;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_INT64 = 1004;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_INT32 = 1003;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_INT16 = 1002;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_INT8 = 1001;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_UINT64 = 2004;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_UINT32 = 2003;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_UINT16 = 2002;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_UINT8 = 2001;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_FLOAT = 3003;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_FLOAT16 = 3002;
const uint32_t SORT_WITH_INDEX_COMMON_TILING_KEY_BF16 = 4002;
const uint32_t SORT_WITH_INDEX_MERGE_SORT_TILING_KEY_FLOAT = 13003;
const uint32_t SORT_WITH_INDEX_MERGE_SORT_TILING_KEY_FLOAT16 = 13002;
const uint32_t SORT_WITH_INDEX_MERGE_SORT_TILING_KEY_BF16 = 14002;

template <typename XType, typename ConvertType, typename UnsignedType, typename IndexType>
__aicore__ inline void generateOpForTopK(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex,
    GM_ADDR globalWorkGm, const TopKV2TilingDataSimd* tilingData, TPipe* tPipe)
{
    bool isDescend = tilingData->isLargest > 0 ? true : false;
    bool isInt32Range = tilingData->isInInt32RangeForSort == 1 ? true : false;
    bool isSingleBlock = tilingData->lastDimNeedCoreForSort == 1 ? true : false;
    uint32_t modeType = tilingData->modeTypeForSort;
    if (isSingleBlock && SORT_WITH_INDEX_SMALL_SIZE_MODE == modeType) {
        if (isDescend) {
            SortWithIndexSingleBlock<XType, true, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, tilingData, tPipe);
            radixSort.Process();
        } else {
            SortWithIndexSingleBlock<XType, false, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, tilingData, tPipe);
            radixSort.Process();
        }
        return;
    }

    if (isInt32Range) {
        if (isDescend) {
            SortWithIndexMultiBlock<XType, UnsignedType, true, uint32_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, tilingData, tPipe);
            radixSort.Process();
        } else {
            SortWithIndexMultiBlock<XType, UnsignedType, false, uint32_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, tilingData, tPipe);
            radixSort.Process();
        }
    } else {
        if (isDescend) {
            SortWithIndexMultiBlock<XType, UnsignedType, true, int64_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, tilingData, tPipe);
            radixSort.Process();
        } else {
            SortWithIndexMultiBlock<XType, UnsignedType, false, int64_t, IndexType> radixSort;
            radixSort.Init(x, index, y, sortedIndex, globalWorkGm, tilingData, tPipe);
            radixSort.Process();
        }
    }
}

// todo MergeSort适配Int64
template <typename XType, typename ConvertType, typename IndexType>
__aicore__ inline void generateMergeSortOpForTopK(GM_ADDR x, GM_ADDR index, GM_ADDR values, GM_ADDR indices,
    GM_ADDR globalWorkGm, const TopKV2TilingDataSimd* tilingData, TPipe* tPipe)
{
    bool isDescend = (tilingData->isLargest > 0) ? true : false;
    if (isDescend) {
        SortWithIndexMergeSort<XType, ConvertType, TopKV2TilingDataSimd, true, IndexType> mergeSort;
        mergeSort.Init(x, index, values, indices, globalWorkGm, tilingData, tPipe);
        mergeSort.ProcessSort();
    } else {
        SortWithIndexMergeSort<XType, ConvertType, TopKV2TilingDataSimd, false, IndexType> mergeSort;
        mergeSort.Init(x, index, values, indices, globalWorkGm, tilingData, tPipe);
        mergeSort.ProcessSort();
    }
}

template <typename T_INDEX_TO>
__aicore__ void sortwithindexForTopK(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR sortedIndex, GM_ADDR workspace,
                     const TopKV2TilingDataSimd* tilingData, TPipe* tPipe)
{
    if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_INT8) {
        generateOpForTopK<int8_t, uint8_t, uint8_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_INT16) {
        generateOpForTopK<int16_t, uint16_t, uint16_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_INT32) {
        generateOpForTopK<int32_t, uint32_t, uint32_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_INT64) {
        generateOpForTopK<int64_t, uint64_t, uint64_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_UINT8) {
        generateOpForTopK<uint8_t, uint8_t, uint8_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_UINT64) {
        generateOpForTopK<uint64_t, uint64_t, uint64_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_UINT32) {
        generateOpForTopK<uint32_t, uint32_t, uint32_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_UINT16) {
        generateOpForTopK<uint16_t, uint16_t, uint16_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_FLOAT) {
        generateOpForTopK<float, float, uint32_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_FLOAT16) {
        generateOpForTopK<half, half, uint16_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_COMMON_TILING_KEY_BF16) {
        generateOpForTopK<bfloat16_t, float, uint16_t, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_MERGE_SORT_TILING_KEY_FLOAT) {
        generateMergeSortOpForTopK<float, float, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_MERGE_SORT_TILING_KEY_FLOAT16) {
        generateMergeSortOpForTopK<half, half, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    } else if (tilingData->tilingKeyForSort == SORT_WITH_INDEX_MERGE_SORT_TILING_KEY_BF16) {
        generateMergeSortOpForTopK<bfloat16_t, float, T_INDEX_TO>(x, index, y, sortedIndex, workspace, tilingData, tPipe);
    }
}
#endif