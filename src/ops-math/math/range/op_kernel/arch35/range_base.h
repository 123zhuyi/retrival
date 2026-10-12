/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file range_base.h
 * \brief
 */

#ifndef RANGE_BASE_H
#define RANGE_BASE_H

#include "kernel_operator.h"

namespace Range {
using namespace AscendC;

constexpr int64_t ONCE_ALGN_NUM_INT32 = 8;
constexpr int64_t ONCE_ALGN_NUM_BYTE_256 = 256;
constexpr int64_t ONCE_ALGN_MASK = 64;
constexpr int64_t IS_SIZE_B8 = 8;

template <typename T>
class RangeBase {
public:
    __aicore__ inline RangeBase(){};

public:
    int64_t usedCoreNum_;
    int64_t totalCoreNum_;
    int64_t hardwareUbSize_;
    int64_t totalElementNum_;
    int64_t numOfPerCore_;
    int64_t numOfTailCore_;
    int64_t ubOneLoopNum_;
    int64_t loopOfPerCore_;
    int64_t perOfPerCore_;
    int64_t tailOfPerCore_;
    int64_t loopOfTailCore_;
    int64_t perOfTailCore_;
    int64_t tailOfTailCore_;
};

} // namespace Range
#endif // RANGE_BASE_H