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
 * \file clip_by_norm_no_div_sum_infershape.cpp
 * \brief ClipByNormNoDivSum operator graph infershape
 */
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/infershape_broadcast_util.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4ClipByNormNoDivSum(gert::InferShapeContext* context)
{
    const size_t INPUT_NUM_FOUR = 4;
    return Ops::Base::InferShape4Broadcast(context, INPUT_NUM_FOUR);
}
IMPL_OP_INFERSHAPE(ClipByNormNoDivSum).InferShape(InferShape4ClipByNormNoDivSum);
} // namespace ops
