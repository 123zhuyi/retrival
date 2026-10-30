/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <bitset>

#include "reduce_logsumexp.h"
#include "math/add/op_api/add.h"
#include "math/sub/op_api/sub.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "math/reduce_max/op_api/reduce_max.h"
#include "conversion/masked_fill/op_api/masked_fill.h"
#include "math/abs/op_api/abs.h"
#include "math/equal/op_api/equal.h"
#include "aclnn_kernels/cast.h"
#include "conversion/fill/op_api/fill.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"
#include "op_api/aclnn_check.h"
#include "aclnn_logsumexp.h"
#include "op_api/level2_base_caculation.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_LEN = 64;
constexpr size_t MAX_DIM_LEN = 8;

static bool CheckNotNull(const aclTensor* self, const aclIntArray* dim, aclTensor* out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dim, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

// 算子支持的所有dtype
static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BOOL};
static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BOOL,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self, aclTensor* out) {
    // 检查self的数据类型是否支持
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B ||
        op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910_93 ||
        IsRegBase()) {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, INPUT_DTYPE_SUPPORT_LIST_910B, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST_910B, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(self, INPUT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckPromoteType(const aclTensor* self, aclTensor* out) {
    // 检查self能否转换为out的dtype
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

    return true;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dim) {
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    // self为标量时，dim range [-1, 0]
    if (selfDimNum <= 0) {
        selfDimNum = 1;
    }
    // 获取dim元素
    for (size_t i = 0; i < dim->Size(); i++) {
        if (dim->operator[](i) >= selfDimNum || dim->operator[](i) < (-selfDimNum)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "provided dim %ld not in the range of input tensor size %ld.",
                    dim->operator[](i), selfDimNum);
            return false;
        }
    }
    // dim值不能重复
    uint64_t dimMask[64] = {0};
    for (size_t i = 0; i < dim->Size(); i++) {
        auto dimValue = dim->operator[](i);
        // 负数维度要转成正值后在mask中判断重复
        if (dim->operator[](i) < 0) {
            dimValue = dim->operator[](i) + selfDimNum;
        }
        if (dimMask[dimValue] == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims",
                    dim->operator[](i));
            return false;
        } else {
            dimMask[dimValue] = 1;
        }
    }
    return true;
}

static void ExpectShapeInferWithDimMask(
    const op::Shape& selfShape, const aclIntArray* dim, bool keepDim, op::Shape& expectShape)
{
    bitset<MAX_MASK_LEN64> dimMask = bitset<MAX_MASK_LEN64>();

    // dim为空时，所有轴都视为mask，与竞品一致
    if (dim->Size() == 0) {
        dimMask.flip();
    }
    for (size_t i = 0; i < dim->Size(); i++) {
        int64_t index = GetPosDimWithStd(dim->operator[](i), selfShape.GetDimNum());
        // 前序已校验，此处dim不会重复
        dimMask.set(index);
    }

    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (!dimMask[i]) {
            expectShape.AppendDim(selfShape.GetDim(i));
        } else if (keepDim) {
            // dim为空时 所有轴reduce
            expectShape.AppendDim(1);
        }
    }
}

static bool CheckShape(const aclTensor* self, aclTensor* out, const aclIntArray* dim, bool keepDim) {
    // 是否小于8维
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

    // 是否满足由self，dim，keepDim推导得到的shape
    op::Shape expectShape;
    ExpectShapeInferWithDimMask(self->GetViewShape(), dim, keepDim, expectShape);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, aclTensor* out, bool keepDim) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dim, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查reduce的轴是否超出self维度范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查out的shape是否满足reduce推导  是否考虑兼容：shape不满足也可计算
    CHECK_RET(CheckShape(self, out, dim, keepDim), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<int64_t> shape = FillScalarGetShapeValue(out);
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSumExpGetWorkspaceSize(const aclTensor* self, const aclIntArray* dim, bool keepDim,
                                           aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnLogSumExp, DFX_IN(self, dim, keepDim), DFX_OUT(out));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    // 参数检查
    auto ret = CheckParams(self, dim, out, keepDim);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor处理
    if (self->IsEmpty()) {
        // 空tensor填充-INFINITY
        ret = FillScalar(out, -INFINITY, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ret;
    }

    // 检查Format
    if(self->GetStorageFormat() != Format::FORMAT_ND){
        OP_LOGW("Format only support ND");
    }
    
    // 将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 1.logSumExp需要支持整型，整型转换为fp32,和竞品保持一致。
    // 2.logSumExp输入为bfloat16转换为fp32
    // 3.reduceMax输入为float16转换为fp32
    auto promoteType = self->GetDataType();
    if (self->GetDataType() == op::DataType::DT_FLOAT16 || self->GetDataType() == op::DataType::DT_BF16 ||
        IsIntegralType(self->GetDataType(), true)) {
        promoteType = op::DataType::DT_FLOAT;
    }
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输入dim为空aclIntArray时，添加输入self的dim
    if (dim->Size() == 0) {
        op::Shape shape = self->GetViewShape();
        size_t dimDum = shape.GetDimNum();
        int64_t appendDim[dimDum];
        for (uint64_t i = 0; i < dimDum; i++) {
            appendDim[i] = static_cast<int64_t>(i);
        }
        dim = uniqueExecutor.get()->AllocIntArray(appendDim, dimDum);
    }

    auto selfMax = l0op::ReduceMax(selfCasted, dim, true, true, uniqueExecutor.get());
    CHECK_RET(selfMax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto absSelfMax = l0op::Abs(selfMax, uniqueExecutor.get());
    CHECK_RET(absSelfMax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<float> infVector = {INFINITY};
    auto infTensor = uniqueExecutor.get()->ConvertToTensor(infVector.data(), infVector.size(),
                                                           absSelfMax->GetDataType());
    auto infMask = l0op::Equal(absSelfMax, infTensor, uniqueExecutor.get());
    CHECK_RET(infMask != nullptr, ACLNN_ERR_INNER_NULLPTR);

    FVector<float> zeroVector = {0};
    auto zeroTensor = uniqueExecutor.get()->ConvertToTensor(zeroVector.data(), zeroVector.size(),
                                                            selfMax->GetDataType());
    auto infZeroSelfMax = l0op::MaskedFill(selfMax, infMask, zeroTensor, uniqueExecutor.get());
    CHECK_RET(infZeroSelfMax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfSubMax = l0op::Sub(selfCasted, infZeroSelfMax, uniqueExecutor.get());
    CHECK_RET(selfSubMax != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用logSumExp算子kernel
    auto logSumExpOpOut = l0op::ReduceLogSumExp(selfSubMax, dim, keepDim, uniqueExecutor.get());
    CHECK_RET(logSumExpOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (!keepDim && (infZeroSelfMax->GetViewShape().GetDimNum() > 0)) {
        infZeroSelfMax = l0op::SqueezeNd(infZeroSelfMax, dim, uniqueExecutor.get());
        CHECK_RET(infZeroSelfMax != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto logSumExpAddOut = l0op::Add(logSumExpOpOut, infZeroSelfMax, uniqueExecutor.get());
    CHECK_RET(logSumExpAddOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto logSumExpAddOutCasted = l0op::Cast(logSumExpAddOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(logSumExpAddOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(logSumExpAddOutCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSumExp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnLogSumExp);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif