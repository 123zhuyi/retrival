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
 * \file top_k_v2_apt.cpp
 * \brief top k v2 impl
 */

#ifndef TOP_K_V2_APT_H
#define TOP_K_V2_APT_H

#include <type_traits>

#include "arch35/radix_sort_top_k.h"
#include "arch35/radix_topk_constant.h"
#include "arch35/top_k_merge_sort.h"
#include "arch35/top_k_merge_sort_more_core.h"
#include "arch35/top_k_merge_sort_intra_core.h"
#include "arch35/radix_sort_top_k_single_block.h"
#include "arch35/radix_sort_top_k_single_core.h"
#include "arch35/radix_sort_top_k_inter_core_template_optimization.h"
#include "arch35/sort_and_top_k_more_core.h"
#include "arch35/top_k_non_last_small_non_transpose.h"

using namespace AscendC;
using namespace SortAndTopK;

#define TOPK_COMMON_TILING_KEY_INT64 1004
#define TOPK_COMMON_TILING_KEY_INT32 1003
#define TOPK_COMMON_TILING_KEY_INT16 1002
#define TOPK_COMMON_TILING_KEY_INT8 1001
#define TOPK_COMMON_TILING_KEY_UINT64 2004
#define TOPK_COMMON_TILING_KEY_UINT32 2003
#define TOPK_COMMON_TILING_KEY_UINT16 2002
#define TOPK_COMMON_TILING_KEY_UINT8 2001
#define TOPK_COMMON_TILING_KEY_FLOAT 3003
#define TOPK_COMMON_TILING_KEY_FLOAT16 3002
#define TOPK_COMMON_TILING_KEY_BF16 4002
#define TOPK_MERGE_SORT_TILING_KEY_FLOAT 13003
#define TOPK_MERGE_SORT_TILING_KEY_FLOAT16 13002
#define TOPK_MERGE_SORT_TILING_KEY_BF16 14002
#define TOPK_MERGE_SORT_MORE_CORE_TILING_KEY_FLOAT 23003
#define TOPK_MERGE_SORT_INTRA_CORE_TILING_KEY_FLOAT 33003

const uint32_t SINGLE_CORE_MODE = 1;
const uint32_t MULT_CORE_OPTIM_MODE = 4;
const uint32_t SORT_AND_TOP_K_MODE = 5;
const uint32_t NON_LAST_SMALL_AXIS_MODE = 8;
const uint32_t NON_LAST_SMALL_AXIS_MERGE_SORT = 1;

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, true, true, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        } else {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, true, false, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        }
    } else {
        if (isSort) {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, false, true, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        } else {
            RadixSortTopK<T, UNSINGED_TYPE, NUM_PASS, false, false, T_INDEX, T_INDEX_TO> radixSortTopK;
            radixSortTopK.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopK.ProcessTopK();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void SortAndTopKOpObject(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, 
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        SortAndTopK::SortAndTopKMoreCore<T, T_INDEX_TO, UNSINGED_TYPE, T_INDEX, 1> sortAndTopKMoreCore;
        sortAndTopKMoreCore.InitParam(x, values, indices, globalWorkGm, &tilingData, &tPipe);
        sortAndTopKMoreCore.ProcessTopK();
    } else {
        SortAndTopK::SortAndTopKMoreCore<T, T_INDEX_TO, UNSINGED_TYPE, T_INDEX, 0> sortAndTopKMoreCore;
        sortAndTopKMoreCore.InitParam(x, values, indices, globalWorkGm, &tilingData, &tPipe);
        sortAndTopKMoreCore.ProcessTopK();
    }
}

template <typename T, typename T_INDEX_TO, bool UseMergeSort>
__aicore__ inline void TopKNonLastSmallAxisOpObject(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe pipe;
    if (isLargest) {
        if (isSort) {
            topkV2::TopKNonLastSmallAxisNonTranspose<T, T_INDEX_TO, true, true, UseMergeSort> op;
            op.Init(x, values, indices, &tilingData, &pipe);
            op.Process();
        } else {
            topkV2::TopKNonLastSmallAxisNonTranspose<T, T_INDEX_TO, true, false, UseMergeSort> op;
            op.Init(x, values, indices, &tilingData, &pipe);
            op.Process();
        }
    } else {
        if (isSort) {
            topkV2::TopKNonLastSmallAxisNonTranspose<T, T_INDEX_TO, false, true, UseMergeSort> op;
            op.Init(x, values, indices, &tilingData, &pipe);
            op.Process();
        } else {
            topkV2::TopKNonLastSmallAxisNonTranspose<T, T_INDEX_TO, false, false, UseMergeSort> op;
            op.Init(x, values, indices, &tilingData, &pipe);
            op.Process();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleCoreOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, true, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        } else {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, true, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        }
    } else {
        if (isSort) {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, false, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        } else {
            RadixSortTopKSingleCore<T, UNSINGED_TYPE, NUM_PASS, false, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleCore;
            radixSortTopKSingleCore.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleCore.Process();
        }
    }
}

template <typename T, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKMultiCoreOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopKMultiCoreOptimization<T, true, true, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        } else {
            RadixSortTopKMultiCoreOptimization<T, true, false, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        }
    } else {
        if (isSort) {
            RadixSortTopKMultiCoreOptimization<T, false, true, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        } else {
            RadixSortTopKMultiCoreOptimization<T, false, false, T_INDEX_TO> radixSortTopKMultiCoreOptimization;
            radixSortTopKMultiCoreOptimization.Init(x, k, values, indices, globalWorkGm, &tilingData);
            radixSortTopKMultiCoreOptimization.Process();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX, typename T_INDEX_TO>
__aicore__ inline void RadixSortTopKSingleBlockOpObject(
    GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    TPipe tPipe;
    if (isLargest) {
        if (isSort) {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, true, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        } else {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, true, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        }
    } else {
        if (isSort) {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, false, true, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        } else {
            RadixSortTopKSingleBlock<T, UNSINGED_TYPE, false, false, T_INDEX, T_INDEX_TO> radixSortTopKSingleBlock;
            radixSortTopKSingleBlock.Init(x, k, values, indices, globalWorkGm, &tilingData, &tPipe);
            radixSortTopKSingleBlock.Process();
        }
    }
}

template <typename T, typename UNSINGED_TYPE, int32_t NUM_PASS, typename T_INDEX_TO>
__aicore__ inline void generateOpObject(
  GM_ADDR x,
  GM_ADDR k,
  GM_ADDR values,
  GM_ADDR indices,
  GM_ADDR globalWorkGm,
  GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    bool isSort = (tilingData.isSort > 0) ? true : false;
    bool isSingleBlock = (tilingData.lastDimNeedCore == 1) ? true : false;
    bool isSingleCore = (tilingData.modeType == SINGLE_CORE_MODE) ? true : false;
    bool isSortAndTopK = (tilingData.modeType == SORT_AND_TOP_K_MODE) ? true : false;
    bool isInInt32Range = (tilingData.isInInt32Range > 0) ? true : false;
    bool isMultiCoreOptimMode = (tilingData.modeType == MULT_CORE_OPTIM_MODE) ? true : false;
    bool isNonLastSmallAxis = (tilingData.modeType == NON_LAST_SMALL_AXIS_MODE) ? true : false;

    if (isNonLastSmallAxis) {
        if (tilingData.keyParams0 == NON_LAST_SMALL_AXIS_MERGE_SORT) {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
                TopKNonLastSmallAxisOpObject<T, T_INDEX_TO, true>(x, values, indices, tiling);
            } else {
                TopKNonLastSmallAxisOpObject<T, T_INDEX_TO, false>(x, values, indices, tiling);
            }
        } else {
            TopKNonLastSmallAxisOpObject<T, T_INDEX_TO, false>(x, values, indices, tiling);
        }
        return;
    }

    // 核内模板
    if (isSingleBlock) {
        if (isInInt32Range) {
            RadixSortTopKSingleBlockOpObject<T, UNSINGED_TYPE, NUM_PASS, int32_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        } else {
            RadixSortTopKSingleBlockOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    // SortAndTopK模板
    if (isSortAndTopK) {
        if (isInInt32Range) {
            SortAndTopKOpObject<T, UNSINGED_TYPE, NUM_PASS, uint32_t, T_INDEX_TO>(
                x, values, indices, globalWorkGm, tiling);
        } else {
            SortAndTopKOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
                x, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    // 单核多次处理模板（930新增模板用于性能优化）
    if (isSingleCore) {
        if (isInInt32Range) {
            RadixSortTopKSingleCoreOpObject<T, UNSINGED_TYPE, NUM_PASS, int32_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        } else {
            RadixSortTopKSingleCoreOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    if (isMultiCoreOptimMode) {
        if (isInInt32Range) {
            RadixSortTopKMultiCoreOpObject<T, T_INDEX_TO>(
                x, k, values, indices, globalWorkGm, tiling);
        }
        return;
    }

    // 多核处理排序轴模板（老模板）
    if (isInInt32Range) {
        RadixSortTopKOpObject<T, UNSINGED_TYPE, NUM_PASS, int32_t, T_INDEX_TO>(
            x, k, values, indices, globalWorkGm, tiling);
    } else {
        RadixSortTopKOpObject<T, UNSINGED_TYPE, NUM_PASS, int64_t, T_INDEX_TO>(
            x, k, values, indices, globalWorkGm, tiling);
    }
}

template <typename T, typename CONVERT_TYPE, typename INDEX_DTYPE>
__aicore__ inline void generateMergeTopKObject(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    TPipe pipe;
    if (isLargest) {
        topkV2::MergeSort<T, CONVERT_TYPE, TopKV2TilingDataSimd, true, INDEX_DTYPE> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    } else {
        topkV2::MergeSort<T, CONVERT_TYPE, TopKV2TilingDataSimd, false, INDEX_DTYPE> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.ProcessSort();
    }
}

template <typename T, typename CONVERT_TYPE, typename INDEX_DTYPE>
__aicore__ inline void generateMergeTopKMoreCoreObject(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    TPipe pipe;
    if (isLargest) {
        topkV2::TopKMergeSortMoreCore<T, CONVERT_TYPE, true, INDEX_DTYPE> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.Process();
    } else {
        topkV2::TopKMergeSortMoreCore<T, CONVERT_TYPE, false, INDEX_DTYPE> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.Process();
    }
}

template <typename T, typename CONVERT_TYPE, typename INDEX_DTYPE>
__aicore__ inline void generateMergeTopKIntraCoreObject(
    GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR globalWorkGm, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    bool isLargest = (tilingData.isLargest > 0) ? true : false;
    TPipe pipe;
    if (isLargest) {
        topkV2::TopKMergeSortIntraCore<T, INDEX_DTYPE, true> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.Process();
    } else {
        topkV2::TopKMergeSortIntraCore<T, INDEX_DTYPE, false> mergeSort;
        mergeSort.Init(x, values, indices, globalWorkGm, &tilingData, &pipe);
        mergeSort.Process();
    }
}

extern "C" __global__ __aicore__ void top_k_v2(GM_ADDR x, GM_ADDR k, GM_ADDR values, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR globalWorkGm = GetUserWorkspace(workspace);
    if (globalWorkGm == nullptr) {
        return;
    }
  
    #if ORIG_DTYPE_X == DT_INT64
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_INT64);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_INT64
            generateOpObject<int64_t, uint64_t, topkV2::B64_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_INT32
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_INT32);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_INT32
            generateOpObject<int32_t, uint32_t, topkV2::B32_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_INT16
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_INT16);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_INT16
            generateOpObject<int16_t, uint16_t, topkV2::B16_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_INT8
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_INT8);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_INT8
            generateOpObject<int8_t, uint8_t, topkV2::B8_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_UINT64
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_UINT64);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_UINT64
            generateOpObject<uint64_t, uint64_t, topkV2::B64_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_UINT32
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_UINT32);   
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_UINT32
            generateOpObject<uint32_t, uint32_t, topkV2::B32_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_UINT16
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_UINT16);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_UINT16
            generateOpObject<uint16_t, uint16_t, topkV2::B16_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif
    
    #if ORIG_DTYPE_X == DT_UINT8
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_UINT8);
        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_UINT8
            generateOpObject<uint8_t, uint8_t, topkV2::B8_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_FLOAT
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_FLOAT);
        TILING_KEY_IS(TOPK_MERGE_SORT_TILING_KEY_FLOAT);
        TILING_KEY_IS(TOPK_MERGE_SORT_MORE_CORE_TILING_KEY_FLOAT);
        TILING_KEY_IS(TOPK_MERGE_SORT_INTRA_CORE_TILING_KEY_FLOAT);

        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_FLOAT
            generateOpObject<float, uint32_t, topkV2::B32_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #elif TILING_KEY_VAR == TOPK_MERGE_SORT_TILING_KEY_FLOAT
            generateMergeTopKObject<float, float, DTYPE_INDICES>(x, values, indices, globalWorkGm, tiling);
        #elif TILING_KEY_VAR == TOPK_MERGE_SORT_MORE_CORE_TILING_KEY_FLOAT
            generateMergeTopKMoreCoreObject<float, float, DTYPE_INDICES>(x, values, indices, globalWorkGm, tiling);
        #elif TILING_KEY_VAR == TOPK_MERGE_SORT_INTRA_CORE_TILING_KEY_FLOAT
            generateMergeTopKIntraCoreObject<float, float, DTYPE_INDICES>(x, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_FLOAT16
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_FLOAT16);
        TILING_KEY_IS(TOPK_MERGE_SORT_TILING_KEY_FLOAT16);

        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_FLOAT16
            generateOpObject<half, uint16_t, topkV2::B16_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #elif TILING_KEY_VAR == TOPK_MERGE_SORT_TILING_KEY_FLOAT16
            generateMergeTopKObject<half, half, DTYPE_INDICES>(x, values, indices, globalWorkGm, tiling);
        #endif
    #endif

    #if ORIG_DTYPE_X == DT_BF16
        TILING_KEY_IS(TOPK_COMMON_TILING_KEY_BF16);
        TILING_KEY_IS(TOPK_MERGE_SORT_TILING_KEY_BF16);

        #if TILING_KEY_VAR == TOPK_COMMON_TILING_KEY_BF16
            generateOpObject<bfloat16_t, uint16_t, topkV2::B16_BITE_SIZE, DTYPE_INDICES>(x, k, values, indices, globalWorkGm, tiling);
        #elif TILING_KEY_VAR == TOPK_MERGE_SORT_TILING_KEY_BF16
            generateMergeTopKObject<bfloat16_t, float, DTYPE_INDICES>(x, values, indices, globalWorkGm, tiling);
        #endif
    #endif
}
#endif // TOP_K_V2_APT_H
