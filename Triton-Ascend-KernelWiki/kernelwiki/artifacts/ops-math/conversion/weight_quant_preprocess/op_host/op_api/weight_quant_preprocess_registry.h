/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WEIGHT_QUANT_PREPROCESS_REGISTRY_H_
#define WEIGHT_QUANT_PREPROCESS_REGISTRY_H_

#include "opdev/common_types.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include <functional>
#include <unordered_map>
#include <vector>

#define QUANT_DATA_FLOW_LIST \
    X(UNKNOWN, 0)            \
    X(MM_MX_A8W4, 1)         \
    X(GMM_MX_A8W4, 2)

enum class QuantDataFlow : uint32_t {
#define X(name, value) name = value,
    QUANT_DATA_FLOW_LIST
#undef X
};

inline const char* QuantDataFlowToString(QuantDataFlow flow)
{
    static const char* QUANT_DATA_FLOW_NAMES[] = {
#define X(name, value) #name,
        QUANT_DATA_FLOW_LIST
#undef X
    };
    uint32_t idx = static_cast<uint32_t>(flow);
    if (idx < sizeof(QUANT_DATA_FLOW_NAMES) / sizeof(QUANT_DATA_FLOW_NAMES[0])) {
        return QUANT_DATA_FLOW_NAMES[idx];
    }
    return "INVALID";
}

struct QuantContext {
    const aclTensor* weight = nullptr;
    const aclTensor* weightScale = nullptr;
    const aclTensor* weightOffsetOptional = nullptr;
    const aclTensor* biasOptional = nullptr;
    const op::DataType xDtype = op::DataType::DT_UNDEFINED;
    const op::DataType xScaleDtype = op::DataType::DT_UNDEFINED;
    const int64_t kGroupSize = -1;
    NpuArch npuArch = NpuArch::DAV_RESV;
    QuantDataFlow dataFlow = QuantDataFlow::UNKNOWN;
    aclTensor* outWeight = nullptr;
    aclTensor* outWeightScale = nullptr;
    aclTensor* outWeightOffsetOptional = nullptr;
    aclTensor* outBiasOptional = nullptr;
    aclOpExecutor* executor = nullptr;
};

using JudgeFunc = std::function<bool(QuantContext&)>;
using CheckFunc = std::function<aclnnStatus(const QuantContext&)>;
using ProcessFunc = std::function<aclnnStatus(QuantContext&)>;

struct DataFlowEntry {
    JudgeFunc judge;
    std::vector<CheckFunc> checks;
    std::vector<ProcessFunc> processes;
};

extern const std::unordered_map<NpuArch, std::vector<DataFlowEntry>> NPU_DATA_FLOW_REGISTRY_MAP;

#endif // WEIGHT_QUANT_PREPROCESS_REGISTRY_H_
