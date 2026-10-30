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
#include "aclnn_reduce_nansum.h"
#include "reduce_nansum.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "conversion/fill/op_api/fill.h"
#include "math/reduce_any/op_api/reduce_any.h"
#include "math/reduce_sum/op_api/reduce_sum_op.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
using std::bitset;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_LEN = 64;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT8,
    op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT8,   op::DataType::DT_BOOL,  op::DataType::DT_BF16};

// 950 (DAV_3510/IsRegBase) 上 nansum 仅支持浮点类型（fp32/fp16/bf16），整型与 bool 在 950 走 ReduceSum/ReduceAny 路径
static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> GetDtypeSupportList()
{
    // 950 上 nansum 仅支持浮点（fp32/fp16/bf16），910B 系列（910B/910_93/RegBase）支持完整类型
    auto socVer = GetCurrentPlatformInfo().GetSocVersion();
    if (socVer == SocVersion::ASCEND950) {
        return ASCEND950_DTYPE_SUPPORT_LIST;
    }
    return ASCEND910B_DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor* self, const aclIntArray* dim, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(dim, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, aclDataType dtype, const aclTensor* out)
{
    // nansum只在ascend910B和ascend950上支持
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93 &&
        GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND950) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ReduceNansum is not supported on this device.");
        return false;
    }
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    // 检查dtype指定的数据类型是否支持
    if (!CheckType(op::ToOpDataType(dtype), supportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "type %s should be in dtype support list [%s].",
            op::ToString(op::ToOpDataType(dtype)).GetString(), op::ToString(supportList).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, op::ToOpDataType(dtype), return false);

    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* out)
{
    // 检查self能否转换为out的dtype
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

    return true;
}

static inline uint64_t GetPosDim(int64_t dim, int64_t dimNum)
{
    if (dimNum <= 0) {
        dimNum = 1;
    }
    return dim >= 0 ? dim : dim + dimNum;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dim)
{
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    // self为标量时，dim range [-1, 0]
    if (selfDimNum <= 0) {
        selfDimNum = 1;
    }
    // dim为负时需要转正校验
    bitset<MAX_MASK_LEN> dimMask = bitset<MAX_MASK_LEN>();

    for (size_t i = 0; i < dim->Size(); i++) {
        if ((*dim)[i] >= selfDimNum || (*dim)[i] < (-selfDimNum)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Provided dim %ld must be in the range of [%ld, %ld].", (*dim)[i], -selfDimNum,
                selfDimNum - 1);
            return false;
        }
        uint64_t index = GetPosDim((*dim)[i], selfDimNum);
        // dim重复
        if (dimMask[index]) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %lu appears multiple times in the list of dims.", index);
            return false;
        }

        dimMask.set(index);
    }

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* dim, aclDataType dtype, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, dim, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查self、out的数据类型是否合法
    CHECK_RET(CheckDtypeValid(self, dtype, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查reduce的轴是否超出self维度范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查最大维度是否超过8
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
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

aclnnStatus aclnnReduceNansumGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* dim, bool keepDim, aclDataType dtype, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnReduceNansum, DFX_IN(self, dim, keepDim, dtype), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, dtype, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 输入self为空tensor时，直接返回dtype类型的空tensor
    if (self->IsEmpty()) {
        // 空tensor填充0
        ret = FillScalar(out, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 空dims处理
    if (dim->Size() == 0) {
        op::Shape shape = self->GetViewShape();
        size_t dimDum = shape.GetDimNum();
        int64_t appendDim[dimDum];
        for (uint64_t i = 0; i < dimDum; i++) {
            appendDim[i] = i;
        }
        dim = uniqueExecutor.get()->AllocIntArray(appendDim, dimDum);
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将输入self的数据类型进行转换
    auto selfContiguousCasted = l0op::Cast(selfContiguous, op::ToOpDataType(dtype), uniqueExecutor.get());

    const aclTensor* reduceNansumOut = nullptr;
    // 整型不存在nan,调用ReduceSum支持整型
    if (op::ToOpDataType(dtype) == op::DataType::DT_BOOL) {
        reduceNansumOut = l0op::ReduceAny(selfContiguousCasted, dim, keepDim, uniqueExecutor.get());
    } else if (IsIntegralType(self->GetDataType(), true)) {
        reduceNansumOut = l0op::ReduceSumOp(selfContiguousCasted, dim, keepDim, uniqueExecutor.get());
    } else {
        // 调用ReduceNansum算子kernel,将输入self的数据类型转换成指定的数据类型
        reduceNansumOut = l0op::ReduceNansum(selfContiguousCasted, dim, keepDim, uniqueExecutor.get());
    }
    CHECK_RET(reduceNansumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckShapeAndScalarSame(reduceNansumOut, out), ACLNN_ERR_PARAM_INVALID);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(reduceNansumOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnReduceNansum(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReduceNansum);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
