/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file pad_common.h
 * \brief pad_common
 */

#ifndef ASCENDC_PAD_COMMON_H
#define ASCENDC_PAD_COMMON_H
#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

namespace PadV3 {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t VL_SIZE = Ops::Base::GetVRegSize();
constexpr uint32_t UB_BLOCK = Ops::Base::GetUbBlockSize();
constexpr int32_t MAX_H_DIMS = 3;
constexpr int32_t B64_BYTES = 8;
constexpr int32_t KEY_BASE = 10;

} // namespace PadV3

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
};

#endif
