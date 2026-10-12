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
 * \file moe_finalize_routing_common.h
 * \brief
 */
#ifndef MOE_FINALIZE_ROUTING_COMMON
#define MOE_FINALIZE_ROUTING_COMMON

#include "kernel_operator.h"

namespace MoeFinalizeRouting {
using namespace AscendC;

constexpr int64_t ONE_BLK_SIZE = 32;
constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t INT32_BYTES = 4;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t PARALLEL_NUM = 2;

__aicore__ inline int64_t PadProcessInt32(int64_t param)
{
    return ONCE_ALGN_NUM_INT32 - param % ONCE_ALGN_NUM_INT32;
}

__aicore__ inline int64_t Int32AlignmentProcess(int64_t param)
{
    return (param + ONCE_ALGN_NUM_INT32 - 1) / ONCE_ALGN_NUM_INT32 * ONCE_ALGN_NUM_INT32;
}
} // namespace MoeFinalizeRouting
#endif // MOE_FINALIZE_ROUTING_COMMON
