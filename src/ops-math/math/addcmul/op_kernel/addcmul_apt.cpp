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
 * \file addcmul.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/addcmul_dag.h"
#include "arch35/addcmul_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

template <uint64_t schMode>
__global__ __aicore__ void addcmul(
    GM_ADDR input_data, GM_ADDR x1, GM_ADDR x2, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_INPUT_DATA, int32_t>::value) {
        using OpDag = AddcmulOp::AddcmulInt32Op<int32_t>::OpDag;
        Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_data, x1, x2, value, y);
    } else {
        using OpDag = AddcmulOp::AddcmulFloatOp<DTYPE_INPUT_DATA, DTYPE_VALUE>::OpDag;
        Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_data, x1, x2, value, y);
    }
}
