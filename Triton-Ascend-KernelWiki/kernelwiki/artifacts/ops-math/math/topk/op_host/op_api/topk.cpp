/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topk.h"
#include "math/sort_with_index/op_api/sort_with_index.h"
#include "aclnn_kernels/cast.h"
#include "op_api/aclnn_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(TopKV2);
OP_TYPE_REGISTER(TopKV3);
OP_TYPE_REGISTER(RadixTopK);
OP_TYPE_REGISTER(TopK);

const int64_t MAX_AICORE_CALC_INPUTSIZE = 32768;
const int64_t MAX_AICORE_CALC_DIM = 8;
// 实测下来int64尾轴在22以内，且在MIN_AICORE_CALC_REG_BASE_INT64_INPUTSIZE规定的范围内，aicpu相比aicore更快
const int64_t MAX_AICORE_CALC_REG_BASE_INT64_DIM = 22;
const int64_t MAX_AICORE_CALC_REG_BASE_INT64_INPUTSIZE = 25000000;
const int64_t MIN_AICORE_CALC_REG_BASE_INT64_INPUTSIZE = 65000;
const int64_t MAX_K = 16;

constexpr int64_t TWO_THOUSAND = 2000;
// 排序轴大于该阈值时，走SortAndTopK模板，SortWithIndex场景不涉及
constexpr int64_t SORT_AND_TOP_K_THRESHOLD = 10000000;
// RadixTopK 算子阈值
constexpr int64_t RADIX_TOP_K_S_THRESHOLD_1 = 12000000;
constexpr int64_t RADIX_TOP_K_S_K_RATIO_1 = 100;
constexpr int64_t RADIX_TOP_K_S_THRESHOLD_2 = 100000000;
constexpr int64_t RADIX_TOP_K_S_K_RATIO_2 = 50;
constexpr int64_t RADIX_TOP_K_MIN_K = 1000;
constexpr int64_t MAX_INT32_INPUTSIZE = 2147483647;

static const std::initializer_list<op::DataType> ANCIENT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> FUTURE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64,  op::DataType::DT_INT32,   op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT64, op::DataType::DT_UINT32,  op::DataType::DT_UINT16, op::DataType::DT_UINT8,
    op::DataType::DT_BF16,   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static bool IsFloatTypeSoc(SocVersion version)
{
    return (version >= SocVersion::ASCEND910B && version <= SocVersion::ASCEND910E) ||
           (version >= SocVersion::ASCEND310P && version <= SocVersion::ASCEND310C) ||
           (version == SocVersion::ASCEND610LITE);
}

// 根据芯片类型、dtype判断算子是否支持走AiCore
static bool IsAiCoreSupport(const aclTensor* self, int64_t k)
{
    SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
    // 在1980上，需要判断inputsize和k的大小，因为在k较小时，aicpu性能比aicore更好；但1971无此场景；
    if (version == SocVersion::ASCEND910) {
        auto inputShape = self->GetViewShape();
        int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
        int64_t inputSize = 1;
        for (int64_t i = 0; i < tmpDim; i++) {
            inputSize *= inputShape.GetDim(i);
        }
        if (inputSize > MAX_AICORE_CALC_INPUTSIZE && k < MAX_AICORE_CALC_DIM) {
            return false;
        }
    }
    if (version == SocVersion::ASCEND310P) {
        OP_LOGW("l0op::TopK use ANCIENT_DTYPE_SUPPORT_LIST for socVerison[%d]", static_cast<int32_t>(version));
        return CheckType(self->GetDataType(), ANCIENT_DTYPE_SUPPORT_LIST);
    } else if (version == SocVersion::ASCEND910B || version == SocVersion::ASCEND910_93) {
        OP_LOGW("l0op::TopK use CURRENT_DTYPE_SUPPORT_LIST for socVerison[%d]", static_cast<int32_t>(version));
        return CheckType(self->GetDataType(), CURRENT_DTYPE_SUPPORT_LIST);
    } else if (IsRegBase()) {
        // 在950的int64场景上，需要判断inputsize和排序轴的大小，因为在排序轴较小时，aicpu性能比aicore更好；
        if (self->GetDataType() == op::DataType::DT_INT64) {
            auto inputShape = self->GetViewShape();
            int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
            int64_t inputSize = 1;
            for (int64_t i = 0; i < tmpDim; i++) {
                inputSize *= inputShape.GetDim(i);
            }

            if (inputSize < MAX_AICORE_CALC_REG_BASE_INT64_INPUTSIZE &&
                inputSize > MIN_AICORE_CALC_REG_BASE_INT64_INPUTSIZE &&
                inputShape.GetDim(tmpDim - 1) < MAX_AICORE_CALC_REG_BASE_INT64_DIM) {
                return false;
            }
        }
        OP_LOGW("l0op::TopK use FUTURE_DTYPE_SUPPORT_LIST for socVerison[%d]", static_cast<int32_t>(version));
        return CheckType(self->GetDataType(), FUTURE_DTYPE_SUPPORT_LIST);
    } else {
        // 非以上平台，使用旧有逻辑处理。
    }

    if (IsFloatTypeSoc(version)) {
        return CheckType(
            self->GetDataType(), {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16});
    }
    // 910、310芯片
    return self->GetDataType() == op::DataType::DT_FLOAT16;
}

static bool IsRadixTopKSupport(const aclTensor* self, int64_t k)
{
    SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
    auto inputShape = self->GetViewShape();
    int64_t dimNum = static_cast<int64_t>(inputShape.GetDimNum());
    int64_t sortLen = inputShape.GetDim(dimNum - 1);
    bool socCheck = version == SocVersion::ASCEND910B || version == SocVersion::ASCEND910_93;
    bool dtypeCheck = CheckType(self->GetDataType(), {op::DataType::DT_FLOAT16, op::DataType::DT_BF16});
    bool shapeCheck = false;
    if (sortLen > MAX_INT32_INPUTSIZE) {
        shapeCheck = false;
    } else if (sortLen >= RADIX_TOP_K_S_THRESHOLD_2) {
        shapeCheck = sortLen > RADIX_TOP_K_S_K_RATIO_2 * k;
    } else if (sortLen >= RADIX_TOP_K_S_THRESHOLD_1) {
        shapeCheck = sortLen > RADIX_TOP_K_S_K_RATIO_1 * k;
    }
    return socCheck && dtypeCheck && shapeCheck && k > RADIX_TOP_K_MIN_K;
}

// 根据芯片类型、dtype判断算子是否支持走TopKV3
static bool IsAscendCSupport(const aclTensor* self, int64_t k)
{
    SocVersion version = GetCurrentPlatformInfo().GetSocVersion();
    if (version == SocVersion::ASCEND310P && CheckType(self->GetDataType(), {op::DataType::DT_FLOAT16}) && k < MAX_K) {
        return true;
    }
    return false;
}

// AICORE算子kernel
std::tuple<aclTensor*, aclTensor*> TopkV2AiCore(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopkV2AiCore, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV2算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(TopKV2, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

std::tuple<aclTensor*, aclTensor*> TopkV2AiCoreForDavid(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, op::DataType indicesDType, aclOpExecutor* executor)
{
    L0_DFX(TopkV2AiCoreForDavid, self, k, dim, largest, sorted, values, indices, indicesDType);
    ADD_TO_LAUNCHER_LIST_AICORE(
        TopKV2, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest, indicesDType));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

std::tuple<aclTensor*, aclTensor*> RadixTopK(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, op::DataType indicesDType, aclOpExecutor* executor)
{
    L0_DFX(RadixTopK, self, k, dim, largest, sorted, values, indices, indicesDType);
    ADD_TO_LAUNCHER_LIST_AICORE(
        RadixTopK, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest, indicesDType));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

std::tuple<aclTensor*, aclTensor*> TopkV3(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopkV3, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore TopKV3算子加入任务队列
    ADD_TO_LAUNCHER_LIST_AICORE(TopKV3, OP_INPUT(self, k), OP_OUTPUT(values, indices), OP_ATTR(sorted, dim, largest));
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}

// SortWithIndex排序时，将输出索引类型int64转为int32，计算结束后再将结果转为int64，能获取更好的性能
static bool IsSortWithIndex(const aclTensor* self, int64_t k, bool sorted) {
    auto inputShape = self->GetViewShape();
    int64_t dimNum = static_cast<int64_t>(inputShape.GetDimNum());

    return (k > TWO_THOUSAND) && (inputShape.GetDim(dimNum - 1) <= SORT_AND_TOP_K_THRESHOLD) && sorted;
}

// AICPU算子kernel
std::tuple<aclTensor*, aclTensor*> TopkAiCpu(
    const aclTensor* self, const aclTensor* k, int64_t dim, bool largest, bool sorted, aclTensor* values,
    aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopkAiCpu, self, k, dim, largest, sorted, values, indices);
    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu TopK算子加入任务队列
    static internal::AicpuTaskSpace space("TopK");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        TopK, OP_ATTR_NAMES({"sorted", "largest", "dim"}), OP_INPUT(self, k), OP_OUTPUT(values, indices),
        OP_ATTR(sorted, largest, dim));
    if (ret != ACLNN_SUCCESS) {
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    return std::tuple<aclTensor*, aclTensor*>(values, indices);
}
std::tuple<aclTensor*, aclTensor*> Topk(
    const aclTensor* self, int64_t k, int64_t dim, bool largest, bool sorted, op::DataType indicesDType,
    aclOpExecutor* executor)
{
    op::Shape outShape = self->GetStorageShape();
    outShape.SetDim(dim, k);

    const aclScalar* kScalar = executor->AllocScalar(k);
    if (kScalar == nullptr) {
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    const aclTensor* kTensor = executor->ConvertToTensor(kScalar, op::ToOpDataType(ACL_INT32));
    auto valuesOut = executor->AllocTensor(outShape, self->GetDataType(), self->GetStorageFormat());
    aclTensor* indicesOut = nullptr;
    if (IsRegBase() && !IsSortWithIndex(self, k, sorted)) {
        indicesOut = executor->AllocTensor(outShape, indicesDType, self->GetStorageFormat());
    } else {
        indicesOut = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetStorageFormat());
    }

    if (IsAiCoreSupport(self, k)) {
        if (IsAscendCSupport(self, k)) {
            return TopkV3(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
        } else {
            if (IsRegBase()) {
                return TopkV2AiCoreForDavid(
                    self, kTensor, dim, largest, sorted, valuesOut, indicesOut, indicesDType, executor);
            } else if (IsRadixTopKSupport(self, k)) {
                return RadixTopK(
                    self, kTensor, dim, largest, sorted, valuesOut, indicesOut, indicesDType, executor);
            } else {
                return TopkV2AiCore(self, kTensor, dim, largest, sorted, valuesOut, indicesOut, executor);
            }
        }
    } else {
        // AiCpu 路径：indices 固定返回 INT32，若需其他类型则后续 Cast
        aclTensor* indicesInt32 = executor->AllocTensor(outShape, op::DataType::DT_INT32, self->GetStorageFormat());
        auto result = TopkAiCpu(self, kTensor, dim, largest, sorted, valuesOut, indicesInt32, executor);
        if (std::get<0>(result) == nullptr || std::get<1>(result) == nullptr) {
            return result;
        }
        if (IsRegBase() && indicesDType != op::DataType::DT_INT32) {
            auto indicesCast = Cast(indicesInt32, indicesDType, executor);
            return std::tuple<aclTensor*, aclTensor*>(valuesOut, const_cast<aclTensor*>(indicesCast));
        }
        return std::tuple<aclTensor*, aclTensor*>(valuesOut, indicesInt32);
    }
}
} // namespace l0op
