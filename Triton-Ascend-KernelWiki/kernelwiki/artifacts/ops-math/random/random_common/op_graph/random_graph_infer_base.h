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
 * \file random_graph_infer_base.h
 * \brief
 */
#ifndef RANDOM_GRAPH_INFER_BASE_H
#define RANDOM_GRAPH_INFER_BASE_H

#include <cmath>
#include "register/op_impl_registry.h"
#include <unordered_map>
#include <string>
#include "log/log.h"
#include "op_util.h"
namespace ops {
namespace GraphCommon {
static constexpr size_t MODE_ATTR = 0;               // 模式：attr作为输出类型。
static constexpr size_t MODE_INPUT_EQUAL_OUTPUT = 1; // 模式：输入输出数据类型相同。
static constexpr size_t MODE_ONE_TYPE = 2;           // 模式：固定一种输出类型。

using OutputSpec = std::tuple<std::string, size_t, ge::DataType>;

// 默认输入输出的位置为0。mode = 0：attr作为输出类型的情况,需要dtypeIndex。mode = 1：无attr且输入输出类型保持一致。
// mode = 2：固定一种输出类型。
ge::graphStatus CommonInferType(
    gert::InferDataTypeContext* context, int32_t mode, int32_t dtypeIndex = 0,
    const std::vector<OutputSpec>& extraOutputMap = {}, const std::set<ge::DataType>& supportDtype = {},
    bool isCheck = false);

// dtypeIndex：在attr里输出所在的位置。
ge::graphStatus InferDataTypeByAttr(
    gert::InferDataTypeContext* context, const int32_t dtypeIndex, ge::DataType& OutDtype);

} // namespace graphCommon
} // namespace ops
#endif
