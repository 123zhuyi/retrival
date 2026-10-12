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
 * \file clip_by_value_v2.cpp
 * \brief clip_by_value_v2 kernel
 */

#include "kernel_operator.h"
#include "clip_by_value_v2_dag.h"
#include "clip_by_value_v2_struct.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void clip_by_value_v2(
    GM_ADDR x, GM_ADDR clipValueMin, GM_ADDR clipValueMax, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using OpDag = ClipByValueV2Op::ClipByValueV2Compute<DTYPE_X>::OpDag;
    BroadcastSch<schMode, OpDag> sch(tiling);
    sch.Process(x, clipValueMin, clipValueMax, y);
}
