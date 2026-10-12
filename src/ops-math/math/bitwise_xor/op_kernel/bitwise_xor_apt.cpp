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
 * \file bitwise_xor_apt.cpp
 * \brief bitwise xor kernel
 */

#include "kernel_operator.h"
#include "arch35/bitwise_xor_dag.h"
#include "arch35/bitwise_xor_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

template <uint64_t schMode>
__global__ __aicore__ void bitwise_xor(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using OpDag = BitwiseXorOp::BitwiseXorCompute<DTYPE_X1>::OpDag;
    Ops::Base::BroadcastSch<schMode, OpDag> sch(tiling);
    sch.Process(x1, x2, y);
}