/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "is_neg_inf.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace NsIsNegInf;

namespace {
enum class IsNegInfDtypeId : int64_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
};
}

extern "C" __global__ __aicore__ void is_neg_inf(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;
    auto formerNum = tilingData.formerNum;
    auto formerLength = tilingData.formerLength;
    auto tailLength = tilingData.tailLength;
    auto tileLength = tilingData.tileLength;
    auto dtypeId = tilingData.dtypeId;

    if (dtypeId == static_cast<int64_t>(IsNegInfDtypeId::FP32)) {
        KernelIsNegInfFp32Unified op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength, &pipe);
        op.Process();
    } else if (dtypeId == static_cast<int64_t>(IsNegInfDtypeId::FP16)) {
        KernelIsNegInf<half> op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength, &pipe);
        op.Process();
    } else if (dtypeId == static_cast<int64_t>(IsNegInfDtypeId::BF16)) {
        KernelIsNegInfBf16Unified op;
        op.Init(x, y, formerNum, formerLength, tailLength, tileLength, &pipe);
        op.Process();
    } else {
        return;
    }
}
