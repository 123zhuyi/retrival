/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/kth_value_axis_one_copy.h"
#include "arch35/kth_value_merge_intra_core.h"
#include "arch35/kth_value_merge_sort_one_core.h"
#include "arch35/kth_value_merge_sort_more_core.h"
#include "arch35/kth_value_non_last_small_axis.h"
#include "arch35/kth_value_radix_more_core.h"
#include "arch35/kth_value_radix_one_core.h"
#include "arch35/kth_value_radix_select.h"
#include "arch35/kth_value_small_axis_insertion.h"
#include "arch35/kth_value_small_axis_short_rank_select.h"
#include "arch35/kth_value_small_axis_two_stage.h"
#include "arch35/kth_value_tiling_data.h"
#include "arch35/kth_value_tiling_key.h"

using namespace AscendC;

template <uint64_t schId>
__aicore__ inline void RunMergeSortRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData, TPipe* pipe)
{
    constexpr uint64_t isSort32SmallAxis = (schId == KTH_VALUE_SCHID_SORT32_SMALL_AXIS);
    if constexpr (IsSameType<bfloat16_t, DTYPE_X>::value) {
        KthValue::KthValueMergeSortOneCore<DTYPE_X, float, isSort32SmallAxis> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    } else if constexpr (IsSameType<float, DTYPE_X>::value || IsSameType<half, DTYPE_X>::value) {
        KthValue::KthValueMergeSortOneCore<DTYPE_X, DTYPE_X, isSort32SmallAxis> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    }
}

template <uint64_t isInt32>
__aicore__ inline void RunRadixMoreCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                             KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == 1) {
        if constexpr (isInt32 == 1) {
            KthValue::KthValueRadixMoreCore<DTYPE_X, uint32_t, uint8_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValue::KthValueRadixMoreCore<DTYPE_X, int64_t, uint8_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    }
    if constexpr (sizeof(DTYPE_X) == 2) {
        if constexpr (isInt32 == 1) {
            KthValue::KthValueRadixMoreCore<DTYPE_X, uint32_t, uint16_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValue::KthValueRadixMoreCore<DTYPE_X, int64_t, uint16_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    }
    if constexpr (sizeof(DTYPE_X) == 4) {
        if constexpr (isInt32 == 1) {
            KthValue::KthValueRadixMoreCore<DTYPE_X, uint32_t, uint32_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValue::KthValueRadixMoreCore<DTYPE_X, int64_t, uint32_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    }
    if constexpr (sizeof(DTYPE_X) == 8) {
        if constexpr (isInt32 == 1) {
            KthValue::KthValueRadixMoreCore<DTYPE_X, uint32_t, uint64_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValue::KthValueRadixMoreCore<DTYPE_X, int64_t, uint64_t> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    }
}

__aicore__ inline void RunSmallAxisInsertionRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                                  TPipe* pipe)
{
    if constexpr (IsSameType<bfloat16_t, DTYPE_X>::value) {
        KthValue::KthValueSmallAxisInsertion<DTYPE_X, float> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    } else {
        KthValue::KthValueSmallAxisInsertion<DTYPE_X, DTYPE_X> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    }
}

__aicore__ inline void RunMergeMoreCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                             KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (IsSameType<float, DTYPE_X>::value) {
        KthValue::KthValueMergeSortMoreCore<DTYPE_X, DTYPE_X, false, int64_t> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

__aicore__ inline void RunMergeIntraCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                              KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (IsSameType<float, DTYPE_X>::value) {
        KthValue::KthValueMergeIntraCore<DTYPE_X, int64_t, false> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

template <bool useMergeSort>
__aicore__ inline void RunNonLastSmallAxisRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                                KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (useMergeSort) {
        if constexpr (IsSameType<DTYPE_X, float>::value || IsSameType<DTYPE_X, half>::value ||
                      IsSameType<DTYPE_X, bfloat16_t>::value) {
            KthValue::KthValueNonLastSmallAxis<DTYPE_X, false, true> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    } else {
        KthValue::KthValueNonLastSmallAxis<DTYPE_X, false, false> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

__aicore__ inline void RunAxisOneCopyRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                           KthValueTilingData* tilingData, TPipe* pipe)
{
    KthValue::KthValueAxisOneCopy<DTYPE_X> op;
    op.Init(x, y1, y2, workspace, tilingData, pipe);
    op.Process();
}

__aicore__ inline void RunRadixOneCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                            TPipe* pipe)
{
    KthValue::KthValueRadixOneCore<DTYPE_X> op;
    op.Init(x, y1, y2, tilingData, pipe);
    op.Process();
}

__aicore__ inline void RunRadixSelectRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                           KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == 1) {
        KthValue::KthValueRadixSelect<DTYPE_X, uint8_t> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == 2) {
        KthValue::KthValueRadixSelect<DTYPE_X, uint16_t> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == 4) {
        KthValue::KthValueRadixSelect<DTYPE_X, uint32_t> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == 8) {
        KthValue::KthValueRadixSelect<DTYPE_X, uint64_t> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

__aicore__ inline void RunSmallAxisTwoStageRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                                 TPipe* pipe)
{
    KthValue::KthValueSmallAxisTwoStage<DTYPE_X> op;
    op.Init(x, y1, y2, tilingData, pipe);
    op.Process();
}

__aicore__ inline void RunSmallAxisShortRankSelectRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2,
                                                        KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == 8) {
        KthValue::KthValueSmallAxisShortRankSelect<DTYPE_X> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    }
}

template <uint64_t schId>
__aicore__ inline bool TryRunSmallAxisRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                            TPipe* pipe)
{
    if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_INSERTION) {
        RunSmallAxisInsertionRoute(x, y1, y2, tilingData, pipe);
        return true;
    }
    if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_TWO_STAGE) {
        RunSmallAxisTwoStageRoute(x, y1, y2, tilingData, pipe);
        return true;
    }
    if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_SHORT_RANK_SELECT) {
        RunSmallAxisShortRankSelectRoute(x, y1, y2, tilingData, pipe);
        return true;
    }
    return false;
}

template <uint64_t schId, uint64_t isInt32>
__global__ __aicore__ void kth_value(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(KthValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(KthValueTilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;
    if constexpr (schId == KTH_VALUE_SCHID_AXIS_ONE_COPY) {
        RunAxisOneCopyRoute(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_MERGE_SORT || schId == KTH_VALUE_SCHID_SORT32_SMALL_AXIS) {
        RunMergeSortRoute<schId>(x, y1, y2, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_RADIX_ONE_CORE) {
        RunRadixOneCoreRoute(x, y1, y2, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_RADIX_MORE_CORE) {
        RunRadixMoreCoreRoute<isInt32>(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_RADIX_SELECT) {
        RunRadixSelectRoute(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
    if (TryRunSmallAxisRoute<schId>(x, y1, y2, &tilingData, &pipe)) {
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_MERGE_MORE_CORE) {
        RunMergeMoreCoreRoute(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_MERGE_INTRA_CORE) {
        RunMergeIntraCoreRoute(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS) {
        RunNonLastSmallAxisRoute<true>(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
    if constexpr (schId == KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS_RADIX) {
        RunNonLastSmallAxisRoute<false>(x, y1, y2, usrWorkspace, &tilingData, &pipe);
        return;
    }
}
