/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file select_apt.cpp
 * \brief Select kernel
 */

#include "kernel_operator.h"
#include "arch35/select_dag.h"
#include "arch35/select_struct.h"
#include "arch35/select_simt.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

template <uint64_t schMode, uint64_t userDef>
__global__ __aicore__ void select(GM_ADDR condition, GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                  GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr(userDef == 1) {
        GET_TILING_DATA_WITH_STRUCT(SelectSimtTilingData, tilingDataIn, tiling);
        SelectOp::SelectSimt<int64_t> selectOp;
        selectOp.Init(condition, x1, x2, y, &tilingDataIn);
        selectOp.Process();
    }
    else{
        if constexpr (std::is_same<DTYPE_X1, bool>::value) {    
            using OpDag = SelectOp::Select<uint8_t>::OpDag;
            Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(condition, x1, x2, y);
        } else {
            using OpDag = SelectOp::Select<DTYPE_X1>::OpDag;
            Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
            sch.Process(condition, x1, x2, y);
        }
    }
}