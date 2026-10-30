/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sort_apt.cpp
 * \brief
 */
#include <type_traits>

#include "kernel_tiling/kernel_tiling.h"
#include "basic_api/kernel_vec_intf.h"
#include "arch35/sort_tiling_key.h"
#include "arch35/sort_tiling_data.h"
#include "arch35/sort_radix_sort_more_core.h"
#include "arch35/sort_radix_sort_one_core.h"
#include "arch35/sort_merge_sort.h"
#include "arch35/merge_sort_big_size.h"
#include "arch35/sort_merge_intra_core.h"
#include "arch35/sort_small_axis_insertion.h"
#include "arch35/sort_small_axis_two_stage.h"
#include "arch35/sort_axis_one_copy.h"
#include "arch35/sort_non_last_small_axis.h"

using namespace AscendC;
using namespace Sort;

template <typename Op>
__aicore__ inline void LaunchSortKernel(GM_ADDR input, GM_ADDR values, GM_ADDR indices, GM_ADDR userWorkspace,
                                        const SortRegBaseTilingData* sortTiling, TPipe* pipeline)
{
    Op op;
    op.Init(input, values, indices, userWorkspace, sortTiling, pipeline);
    op.Process();
}

template <uint64_t isInt32, uint64_t isDescend>
__aicore__ inline void LaunchRadixMoreCore(GM_ADDR radixInput, GM_ADDR radixValues, GM_ADDR radixIndices,
                                           GM_ADDR radixWorkspace, const SortRegBaseTilingData* radixTiling,
                                           TPipe* radixPipeline)
{
    using IndexType = std::conditional_t<isInt32 == 1, uint32_t, int64_t>;
    using RadixType = std::conditional_t<
        sizeof(DTYPE_X) == sizeof(uint8_t), uint8_t,
        std::conditional_t<sizeof(DTYPE_X) == sizeof(uint16_t), uint16_t,
                           std::conditional_t<sizeof(DTYPE_X) == sizeof(uint32_t), uint32_t, uint64_t>>>;
    LaunchSortKernel<SortRadixMoreCore<DTYPE_X, DTYPE_Y2, RadixType, IndexType, isDescend>>(
        radixInput, radixValues, radixIndices, radixWorkspace, radixTiling, radixPipeline);
}

template <uint64_t schId, uint64_t isDescend>
__aicore__ inline void LaunchMergeSortRoute(GM_ADDR mergeInput, GM_ADDR mergeValues, GM_ADDR mergeIndices,
                                            GM_ADDR mergeWorkspace, const SortRegBaseTilingData* mergeTiling,
                                            TPipe* mergePipeline)
{
    constexpr bool isSort32SmallAxis = (schId == SORT_SCHID_8);
    if constexpr (std::is_same_v<bfloat16_t, DTYPE_X>) {
        LaunchSortKernel<MergeSort<DTYPE_X, DTYPE_Y2, float, isDescend, isSort32SmallAxis>>(
            mergeInput, mergeValues, mergeIndices, mergeWorkspace, mergeTiling, mergePipeline);
    } else if constexpr (std::is_same_v<half, DTYPE_X> || std::is_same_v<float, DTYPE_X>) {
        LaunchSortKernel<MergeSort<DTYPE_X, DTYPE_Y2, DTYPE_X, isDescend, isSort32SmallAxis>>(
            mergeInput, mergeValues, mergeIndices, mergeWorkspace, mergeTiling, mergePipeline);
    }
}

template <uint64_t schId, uint64_t isInt32, uint64_t isDescend>
__global__ __aicore__ void sort(GM_ADDR input, GM_ADDR sortedValues, GM_ADDR sortedIndices, GM_ADDR workBuffer,
                                GM_ADDR tilingAddress)
{
    REGISTER_TILING_DEFAULT(SortRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(SortRegBaseTilingData, sortTilingData, tilingAddress);

    GM_ADDR sortWorkspace = AscendC::GetUserWorkspace(workBuffer);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    TPipe sortPipeline;
    constexpr bool isDescending = (isDescend != 0);
    if constexpr (schId == SORT_SCHID_7) {
        LaunchSortKernel<Sort::SortAxisOneCopy<DTYPE_X, DTYPE_Y2>>(input, sortedValues, sortedIndices, sortWorkspace,
                                                                   &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_2) {
        LaunchRadixMoreCore<isInt32, isDescend>(input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData,
                                                &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_1) {
        LaunchSortKernel<SortRadixOneCore<DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_0 || schId == SORT_SCHID_8) {
        LaunchMergeSortRoute<schId, isDescend>(input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData,
                                               &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_3 && std::is_same_v<float, DTYPE_X>) {
        LaunchSortKernel<MergeSortBigSize<DTYPE_X, DTYPE_X, isDescending, DTYPE_Y2>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_4 && std::is_same_v<float, DTYPE_X>) {
        LaunchSortKernel<Sort::SortMergeIntraCore<DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_5 && std::is_same_v<bfloat16_t, DTYPE_X>) {
        LaunchSortKernel<Sort::SortSmallAxisInsertion<DTYPE_X, float, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_5) {
        LaunchSortKernel<Sort::SortSmallAxisInsertion<DTYPE_X, DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_6) {
        LaunchSortKernel<Sort::SortSmallAxisTwoStage<DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_9 || schId == SORT_SCHID_10) {
        constexpr bool useMergeSort = (schId == SORT_SCHID_9);
        constexpr bool supportMergeSort = std::is_same_v<DTYPE_X, half> || std::is_same_v<DTYPE_X, float> ||
                                          std::is_same_v<DTYPE_X, bfloat16_t>;
        if constexpr (!useMergeSort || supportMergeSort) {
            LaunchSortKernel<Sort::SortNonLastSmallAxis<DTYPE_X, DTYPE_Y2, isDescending, useMergeSort>>(
                input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
        }
    }
}
