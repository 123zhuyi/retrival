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
 * \file stateless_bernoulli_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/util/math_util.h"
#include "op_host/util/const_util.h"
#include "random/random_common/op_host/random_infershape_base.h"

using namespace ge;
namespace ops {
static constexpr size_t STATELESS_BERNOULLI_SHAPE = 0;
static constexpr size_t STATELESS_BERNOULLI_PROB = 1;
static constexpr size_t STATELESS_BERNOULLI_SEED = 2;
static constexpr size_t STATELESS_BERNOULLI_OFFSET = 3;
static constexpr size_t STATELESS_BERNOULLI_Y = 0;

static graphStatus InferShapeStatelessBernoulli(gert::InferShapeContext* context)
{
    const std::unordered_map<std::string, size_t>& inputMap = {
        {"shape", STATELESS_BERNOULLI_SHAPE},
        {"prob", STATELESS_BERNOULLI_PROB},
        {"seed", STATELESS_BERNOULLI_SEED},
        {"offset", STATELESS_BERNOULLI_OFFSET}};
    const std::unordered_map<std::string, size_t>& outputMap = {{"y", STATELESS_BERNOULLI_Y}};
    int32_t mode = ops::randomCommon::MODE_DEPENDENCY;
    return ops::randomCommon::CommonInferShape(context, inputMap, outputMap, mode);
}
IMPL_OP_INFERSHAPE(StatelessBernoulli).InputsDataDependency({0}).InferShape(InferShapeStatelessBernoulli);

} // namespace ops