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
 * \file stateless_truncated_normal_v2_graph_infer.cpp
 * \brief stateless_truncated_normal_v2 operator graph infer resource
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "random/random_common/op_graph/random_graph_infer_base.h"

using namespace ge;
namespace ops {
static constexpr size_t OUT_ATTR_IDX = 0;

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    int32_t mode = ops::GraphCommon::MODE_ATTR;
    int32_t dtypeIndex = OUT_ATTR_IDX;
    const std::vector<ops::GraphCommon::OutputSpec>& extraOutputMap = {};
    const std::set<ge::DataType>& supportDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_DOUBLE};
    bool isCheck = true;
    return ops::GraphCommon::CommonInferType(context, mode, dtypeIndex, extraOutputMap, supportDtype, isCheck);
}

IMPL_OP(StatelessTruncatedNormalV2).InferDataType(InferDataType);
} // namespace ops
