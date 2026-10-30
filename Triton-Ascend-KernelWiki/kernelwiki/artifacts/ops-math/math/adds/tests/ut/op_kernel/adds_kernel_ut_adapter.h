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
 * \file adds_kernel_ut_adapter.h
 * \brief Adds 算子 Kernel UT 适配头文件（提供 TypeFromId 模板）
 * 
 * atvoss 框架的 TypeFromId 在 UT 编译模式下需要适配
 */

#ifndef ADDS_KERNEL_UT_ADAPTER_H_
#define ADDS_KERNEL_UT_ADAPTER_H_

#include "kernel_operator.h"

namespace Ops::Base {

template <uint64_t dtype>
struct TypeFromId;

template <>
struct TypeFromId<1> { using type = half; };       // TPL_FP16

template <>
struct TypeFromId<2> { using type = bfloat16_t; }; // TPL_BF16

template <>
struct TypeFromId<3> { using type = float; };      // TPL_FP32

template <>
struct TypeFromId<4> { using type = int16_t; };    // TPL_INT16

template <>
struct TypeFromId<5> { using type = int32_t; };    // TPL_INT32

template <>
struct TypeFromId<6> { using type = int64_t; };    // TPL_INT64

}

#endif // ADDS_KERNEL_UT_ADAPTER_H_