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
 * \file random_infershape_base.h
 * \brief
 */
#ifndef RANDOM_INFERSHAPE_BASE_H
#define RANDOM_INFERSHAPE_BASE_H

#include <unordered_map>
#include <string>
#include <cmath>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_util.h"
#include "util/shape_util.h"
#include "util/const_util.h"
#include "op_api/op_util.h"

namespace ops {
namespace randomCommon {
static constexpr int32_t MODE_DEPENDENCY = 0;
static constexpr int32_t MODE_NO_DEPENDENCY = 1;

// 使用时默认输入输出的位置为0。mode = 0:只支持int32和int64，输入输出的dim相等,使用时必须有值依赖。
// mode = 1:只支持float,float16和bf16，输出shape直接使用输入shape，可以无需值依赖。
ge::graphStatus CommonInferShape(
    gert::InferShapeContext* context, const std::unordered_map<std::string, size_t>& requiredInputMap,
    const std::unordered_map<std::string, size_t>& outputMap, int32_t mode,
    const std::unordered_map<std::string, size_t>& optionalInputMap = {});

template <typename T>
ge::graphStatus HandleShapeTensor(gert::Shape& outputShape, size_t xShapeSize, const T* xShapeData);

bool InferShapeForUnknow(
    gert::InferShapeContext* context, const gert::Shape& inShape, gert::Shape& outShape, int64_t& maskIndex,
    int64_t& offsetIndex);

bool DependencyMode(const gert::Tensor* inTensor, gert::Shape& outShape, size_t xShapeSize);

bool InputAndOutputCheck(
    gert::InferShapeContext* context, const std::unordered_map<std::string, size_t>& requiredInputMap,
    const std::unordered_map<std::string, size_t>& outputMap, int64_t& maskIndex, int64_t& offsetIndex,
    const std::unordered_map<std::string, size_t>& optionalInputMap = {});

} // namespace randomCommon
} // namespace ops

#endif
