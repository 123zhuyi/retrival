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
 * \file histogram.cpp
 * \brief
 */
#include "histogram.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Histogram);
OP_TYPE_REGISTER(HistogramV2);
// AiCore支持的类型
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_SELF = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_INT16, op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static bool IsAiCoreSupport(const aclTensor* self, const aclTensor* out)
{
    if (self == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "self is nullptr, check AiCoreSupport failed.");
        return false;
    }
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out is nullptr, check AiCoreSupport failed.");
        return false;
    }
    auto checkSelfType = CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_SELF);
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch != NpuArch::DAV_2201 && !IsRegBase(npuArch) && npuArch != NpuArch::DAV_2002) {
        return false;
    }
    return checkSelfType;
}

// AiCore的执行逻辑
inline const aclTensor* HistogramAiCore(
    const aclTensor* self, const aclTensor* min, const aclTensor* max, const aclTensor* out, int64_t bins,
    aclOpExecutor* executor)
{
    L0_DFX(HistogramAiCore, self, min, max, out, bins);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        HistogramV2, OP_INPUT(self, min, max), OP_OUTPUT(out), OP_ATTR(bins));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HistogramAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
        return nullptr);
    return out;
}

// AiCPU的执行逻辑
inline const aclTensor* HistogramAiCPU(
    const aclTensor* self, const aclTensor* out, int64_t bins, float min, float max, aclOpExecutor* executor)
{
    L0_DFX(HistogramAiCPU, self, bins, min, max, out);
    auto desDtype = (self->GetDataType() == op::DataType::DT_FLOAT16) ? op::DataType::DT_FLOAT : self->GetDataType();
    auto histogramOut = executor->AllocTensor(out->GetViewShape(), desDtype);

    static internal::AicpuTaskSpace space("Histogram");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        Histogram, OP_ATTR_NAMES({"bins", "min", "max"}), OP_INPUT(self), OP_OUTPUT(histogramOut),
        OP_ATTR(bins, min, max));
    OP_CHECK(
        ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "HistogramAiCPU ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return histogramOut;
}

const aclTensor* Histogram(
    const aclTensor* self, const aclTensor* min, const aclTensor* max, const aclTensor* out, int64_t bins,
    float minValue, float maxValue, aclOpExecutor* executor)
{
    if (IsAiCoreSupport(self, out)) {
        auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
        // 当前950支持输出int32和f32，其他芯片只支持int32
        auto desDtype = ((out->GetDataType() == op::DataType::DT_FLOAT16 || out->GetDataType() == op::DataType::DT_FLOAT) && IsRegBase(npuArch)) 
            ? op::DataType::DT_FLOAT : op::DataType::DT_INT32;
        auto outAiCore = executor->AllocTensor(out->GetViewShape(), desDtype);
        return HistogramAiCore(self, min, max, outAiCore, bins, executor);
    } else {
        return HistogramAiCPU(self, out, bins, minValue, maxValue, executor);
    }
}
} // namespace l0op
