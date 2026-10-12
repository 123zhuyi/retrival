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
 * \file dropout_gen_mask_infershape.cpp
 * \brief
 */
#include "util/shape_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "random/random_common/op_graph/random_graph_infer_base.h"

using namespace ge;
namespace ops {
static constexpr size_t STATELESS_DROP_OUT_GEN_OUT_ATTR_IDX = 0;

static ge::graphStatus InferDataTypeStatelessDropOutGenMask(gert::InferDataTypeContext* context)
{
    int32_t mode = ops::GraphCommon::MODE_INPUT_EQUAL_OUTPUT;
    int32_t dtypeIndex = STATELESS_DROP_OUT_GEN_OUT_ATTR_IDX;
    const std::vector<ops::GraphCommon::OutputSpec>& extraOutputMap = {};
    const std::set<ge::DataType>& supportDtype = {ge::DT_UINT8};
    bool isCheck = true;
    return ops::GraphCommon::CommonInferType(context, mode, dtypeIndex, extraOutputMap, supportDtype, isCheck);
}

IMPL_OP(StatelessDropOutGenMask).InferDataType(InferDataTypeStatelessDropOutGenMask);
} // namespace ops