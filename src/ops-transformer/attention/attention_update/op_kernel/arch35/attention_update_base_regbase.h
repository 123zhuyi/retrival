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
 * \file attention_update_base_regbase.h
 * \brief
 */
#ifndef ATTENTION_UPDATE_BASE_REGBASE_H_
#define ATTENTION_UPDATE_BASE_REGBASE_H_

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/load_store_utils.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"

namespace AttentionUpdateOpt {
using namespace AscendC;

static constexpr uint32_t UNROLL_NUM = 2;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize();
static constexpr uint32_t VREG_SIZE = Ops::Base::GetVRegSize();

} // namespace AttentionUpdateOpt
#endif // ATTENTION_UPDATE_BASE_REGBASE_H_
