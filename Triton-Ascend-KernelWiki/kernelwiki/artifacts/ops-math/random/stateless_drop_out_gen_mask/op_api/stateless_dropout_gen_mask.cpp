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
 * \file stateless_dropout_gen_mask.cpp
 * \brief
 */

#include "stateless_dropout_gen_mask.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(StatelessDropOutGenMask);

static const int64_t BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;
static const std::initializer_list<DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_UINT8};

// 根据芯片型号，dtype 判断AICore 是否支持
static inline bool IsAiCoreSupport(DataType dType)
{
    if (IsRegBase()) {
        return CheckType(dType, AICORE_DTYPE_SUPPORT_LIST);
    }
    return false;
}

static inline int64_t inferShape(const aclIntArray* shape)
{
    int64_t size = 1;
    for (size_t index = 0; index < shape->Size(); index++) {
        size *= (*shape)[index];
    }
    return (size + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
}

// AICPU算子kernel
static const aclTensor* StatelessDropoutGenMaskAiCpu(
    const aclTensor* shape, const aclTensor* prob, const aclTensor* seed, const aclTensor* seed1,
    const aclTensor* offset, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StatelessDropoutGenMaskAiCpu, shape, prob, seed, seed1, offset, out);
    static internal::AicpuTaskSpace space("StatelessDropOutGenMask");
    ADD_TO_LAUNCHER_LIST_AICPU(
        StatelessDropOutGenMask, OP_ATTR_NAMES(), OP_INPUT(shape, prob, seed, seed1, offset), OP_OUTPUT(out));
    return out;
}

// AICORE算子kernel
static const aclTensor* StatelessDropoutGenMaskAiCore(
    const aclTensor* shape, const aclTensor* prob, const aclTensor* seed, const aclTensor* seed1,
    const aclTensor* offset, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(StatelessDropoutGenMaskAiCore, shape, prob, seed, seed1, offset, out);

    // 使用框架宏 ADD_TO_LAUNCHER_LIST_AICORE, 将AiCore StatelessDropoutGenMask 算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(
        StatelessDropOutGenMask, OP_ATTR_NAMES(), OP_INPUT(shape, prob, seed, seed1, offset), OP_OUTPUT(out));
    return out;
}

const aclTensor* StatelessDropoutGenMask(
    const aclIntArray* shape, const aclTensor* prob, const aclTensor* seed, const aclTensor* seed1,
    const aclTensor* offset, aclOpExecutor* executor)
{
    auto outShape = op::Shape{inferShape(shape)};
    auto out = executor->AllocTensor(outShape, DataType::DT_UINT8);

    auto shapeTensor = executor->ConvertToTensor(shape, seed->GetDataType());
    if (IsAiCoreSupport(out->GetDataType())) {
        return StatelessDropoutGenMaskAiCore(shapeTensor, prob, seed, seed1, offset, out, executor);
    } else {
        return StatelessDropoutGenMaskAiCpu(shapeTensor, prob, seed, seed1, offset, out, executor);
    }
}

} // namespace l0op