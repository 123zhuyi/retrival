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
 * \file stateless_uniform_infershape.cpp
 * \brief InferShape for StatelessUniform operator
 *        Reads input shape tensor (input0) via ValueDepend to set output shape.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "random/random_common/op_host/random_infershape_base.h"

using namespace ge;
namespace ops {

static constexpr size_t INPUT_SHAPE_IDX = 0;
static constexpr size_t INPUT_SEED_IDX = 1;
static constexpr size_t INPUT_OFFSET_IDX = 2;
static constexpr size_t INPUT_FROM_IDX = 3;
static constexpr size_t INPUT_TO_IDX = 4;
static constexpr size_t OUTPUT_Y_IDX = 0;

static ge::graphStatus InferShapeStatelessUniform(gert::InferShapeContext* context)
{
    const std::unordered_map<std::string, size_t>& inputMap = {
        {"shape", INPUT_SHAPE_IDX}};
    const std::unordered_map<std::string, size_t>& outputMap = {{"y", OUTPUT_Y_IDX}};
    int32_t mode = ops::randomCommon::MODE_DEPENDENCY;

    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}

IMPL_OP_INFERSHAPE(StatelessUniform).InputsDataDependency({0}).InferShape(InferShapeStatelessUniform);

} // namespace ops
