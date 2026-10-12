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
 * \file gcd.h
 * \brief Gcd AiCore kernel
 */

#ifndef GCD_H_
#define GCD_H_

#include "gcd_tiling_data.h"
#include "gcd_tiling_key.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace {

#include "detail/gcd_scalar_utils.h"
#include "detail/gcd_kernel_common.h"
#include "detail/gcd_kernel_range.h"
#include "detail/gcd_kernel_scalar.h"
#include "detail/gcd_kernel_float.h"
#include "detail/gcd_kernel_int32.h"
#include "detail/gcd_kernel_int8.h"
#include "detail/gcd_kernel_uint8.h"
#include "detail/gcd_kernel_int16.h"
#include "detail/gcd_kernel_half.h"
#include "detail/gcd_kernel_bf16.h"
#include "detail/gcd_kernel_mixed.h"

} // namespace
} // namespace AscendC

#endif // GCD_H_
