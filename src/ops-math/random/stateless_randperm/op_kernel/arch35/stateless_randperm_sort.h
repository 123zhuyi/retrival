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
 * \file stateless_randperm_sort.h
 * \brief
 */

#ifndef STATELESS_RANDPERM_SORT_H
#define STATELESS_RANDPERM_SORT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../stateless_randperm_struct.h"
#include "../../sort/arch35/sort_tiling_data.h"
#include "../../sort/arch35/sort_radix_sort_more_core.h"
#include "../../sort/arch35/sort_radix_sort_one_core.h"
#include "../../sort/arch35/sort_merge_sort.h"
#include "../../sort/arch35/merge_sort_big_size.h"

using namespace AscendC;
using namespace Sort;

namespace StatelessRandperm {
template <typename Tr, typename Tx, uint64_t schId, uint64_t isInt32, uint64_t isDescend>
__aicore__ void Sort(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR usrWorkspace, const SortRegBaseTilingData* tilingData, TPipe* pipe)
{
    if constexpr (schId == 2) {
        if constexpr(sizeof(Tr) == 1) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<Tr, Tx, uint8_t, uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                SortRadixMoreCore<Tr, Tx, uint8_t, int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
        if constexpr(sizeof(Tr) == 2) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<Tr, Tx, uint16_t,uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                SortRadixMoreCore<Tr, Tx, uint16_t,int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
        if constexpr(sizeof(Tr) == 4) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<Tr, Tx, uint32_t,uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                SortRadixMoreCore<Tr, Tx, uint32_t,int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
        if constexpr(sizeof(Tr) == 8) {
            if constexpr(isInt32 == 1) {
                SortRadixMoreCore<Tr, Tx, uint64_t, uint32_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                SortRadixMoreCore<Tr, Tx, uint64_t, int64_t,isDescend> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
        return;
    }
    if constexpr (schId == 1) {
        if constexpr (isDescend == 1) {
            SortRadixOneCore<Tr, Tx, true> op;
            op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
            op.Process();
        } else {
            SortRadixOneCore<Tr, Tx, false> op;
            op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
            op.Process();
        }
        return;
    }
    if constexpr (schId == 0) {
        if constexpr (IsSameType<bfloat16_t, Tr>::value) {
            MergeSort<Tr, Tx, float, isDescend> op;
            op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
            op.Process();
        }
        if constexpr (IsSameType<float, Tr>::value) {
            MergeSort<Tr, Tx, Tr, isDescend> op;
            op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
            op.Process();
        }
        if constexpr (IsSameType<half, Tr>::value) {
            MergeSort<Tr, Tx, Tr, isDescend> op;
            op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
            op.Process();
        }
        return;
    }
    if constexpr (schId == 3) {
        if constexpr (IsSameType<Tr, bfloat16_t>::value) {
            if constexpr (isDescend == 1) {
                MergeSortBigSize<Tr, float, true, Tx> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                MergeSortBigSize<Tr, float, false, Tx> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
		if constexpr (IsSameType<float, Tr>::value) {
            if constexpr (isDescend == 1) {
                MergeSortBigSize<Tr, Tr, true, Tx> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                MergeSortBigSize<Tr, Tr, false, Tx> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
		if constexpr (IsSameType<half, Tr>::value) {
            if constexpr (isDescend == 1) {
                MergeSortBigSize<Tr, Tr, true, Tx> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            } else {
                MergeSortBigSize<Tr, Tr, false, Tx> op;
                op.Init(x, y1, y2, usrWorkspace, tilingData, pipe);
                op.Process();
            }
        }
        return;
    }
}

}

#endif // STATELESS_RANDPERM_SORT_H