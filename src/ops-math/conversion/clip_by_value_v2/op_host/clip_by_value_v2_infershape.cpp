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
 * \file clip_by_value_v2_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "op_common/op_host/util/shape_util.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/infershape_broadcast_util.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4ClipByValueV2(gert::InferShapeContext* context)
{
    const size_t INPUT_NUM_THREE = 3;
    return Ops::Base::InferShape4Broadcast(context, INPUT_NUM_THREE);
}
IMPL_OP_INFERSHAPE(ClipByValueV2).InferShape(InferShape4ClipByValueV2);
} // namespace ops
