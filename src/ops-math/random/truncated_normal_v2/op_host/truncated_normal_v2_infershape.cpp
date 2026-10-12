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
 * \file truncated_normal_v2_infershape.cpp
 * \brief
 */
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "random/random_common/op_host/random_infershape_base.h"

using namespace ge;
namespace ops {
static constexpr size_t TRUNCATED_NORMAL_SHAPE = 0;
static constexpr size_t TRUNCATED_NORMAL_OFFSET = 1;
static constexpr size_t TRUNCATED_NORMAL_Y = 0;

static graphStatus InferShapeTruncatedNormalV2(gert::InferShapeContext* context)
{
    const std::unordered_map<std::string, size_t>& inputMap = {
        {"shape", TRUNCATED_NORMAL_SHAPE},
        {"offset", TRUNCATED_NORMAL_OFFSET}};
    const std::unordered_map<std::string, size_t>& outputMap = {
        {"y", TRUNCATED_NORMAL_Y},
        {"offset", TRUNCATED_NORMAL_OFFSET}};
    int32_t mode = ops::randomCommon::MODE_DEPENDENCY;
    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}
IMPL_OP_INFERSHAPE(TruncatedNormalV2).InputsDataDependency({0}).InferShape(InferShapeTruncatedNormalV2);

} // namespace ops