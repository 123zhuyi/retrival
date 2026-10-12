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
 * \file drop_out_v3_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "infershape_elewise_util.h"
#include "log/log.h"
#include "random/random_common/op_host/random_infershape_base.h"
using namespace ge;
namespace ops {
static constexpr size_t DROP_OUT_V3_GRAD_GRAD_Y = 0;
static constexpr size_t DROP_OUT_V3_GRAD_MASK = 1;
static constexpr size_t DROP_OUT_V3_GRAD_SCALE = 2;
static constexpr size_t DROP_OUT_V3_GRAD_GRAD_X = 0;

static graphStatus InferShapeDropOutV3Grad(gert::InferShapeContext* context)
{
    const std::unordered_map<std::string, size_t>& inputMap = {
        {"grad_y", DROP_OUT_V3_GRAD_GRAD_Y}, {"mask", DROP_OUT_V3_GRAD_MASK}, {"scale", DROP_OUT_V3_GRAD_SCALE}};
    const std::unordered_map<std::string, size_t>& outputMap = {{"grad_x", DROP_OUT_V3_GRAD_GRAD_X}};
    int32_t mode = ops::randomCommon::MODE_NO_DEPENDENCY;
    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}
IMPL_OP_INFERSHAPE(DropOutV3Grad).InferShape(InferShapeDropOutV3Grad);

} // namespace ops
