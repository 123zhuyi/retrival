/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_expand.h"
#include "aclnn_kernels/contiguous.h"
#include "expand.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_SUPPORT_DIM = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_UINT8, DataType::DT_INT8,
    DataType::DT_INT32,   DataType::DT_INT64, DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_UINT8, DataType::DT_INT8,
    DataType::DT_INT32,   DataType::DT_INT64, DataType::DT_BOOL,  DataType::DT_BF16};

static bool CheckNotNull(const aclTensor* self, const aclIntArray* size, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(size, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch arch)
{
    static const std::map<NpuArch, std::initializer_list<DataType>> dataTypeSupportedMap = {
        {NpuArch::DAV_2201, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST},
        {NpuArch::DAV_3510, ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST}};
    auto found = dataTypeSupportedMap.find(arch);
    if (found == dataTypeSupportedMap.end()) {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
    return found->second;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    // 检查self的数据类型是否在expand算子的支持列表内
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& dTypeSupportList = GetSupportDtypeList(curArch);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dTypeSupportList, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* size, const aclTensor* out)
{
    op::Shape sizeShape;
    for (int64_t i = 0; i < static_cast<int64_t>(size->Size()); i++) {
        sizeShape.AppendDim((*size)[i]);
    }
    const auto& outShape = out->GetViewShape();
    const auto& selfShape = self->GetViewShape();
    if (selfShape == sizeShape && outShape == sizeShape) {
        return true;
    }
    auto sizeDimNum = sizeShape.GetDimNum();
    auto selfDimNum = selfShape.GetDimNum();
    if (sizeDimNum < selfDimNum) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "the number of size %zu must be greater or equal to the number of dimensions in the self %zu.",
            sizeDimNum, selfDimNum);
        return false;
    }
    auto offset = sizeDimNum - selfDimNum;
    op::Shape expectShape;
    expectShape.SetDimNum(sizeDimNum);
    for (size_t i = 0; i < selfDimNum; i++) {
        if (sizeShape.GetDim(offset + i) == -1) {
            expectShape.SetDim(offset + i, selfShape.GetDim(i));
        }
        if ((selfShape.GetDim(i) != sizeShape.GetDim(offset + i)) && (selfShape.GetDim(i) != 1)) {
            OP_LOGE(
                ACL_ERROR_INVALID_PARAM, "the self shape [%s] is not match size [%s].",
                op::ToString(selfShape).GetString(), op::ToString(sizeShape).GetString());
            return false;
        }
        expectShape.SetDim(offset + i, sizeShape.GetDim(offset + i));
    }
    for (size_t i = 0; i < offset; i++) {
        expectShape.SetDim(i, sizeShape.GetDim(i));
    }
    if (outShape != expectShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "expect out shape to be same as expectShape, but got out shape [%s], expect shape [%s]",
            op::ToString(outShape).GetString(), op::ToString(expectShape).GetString());
        return false;
    }
    return true;
}

static bool CheckMaxDimension(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIM, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIM, return false);
    return true;
}

static void CheckFormat(const aclTensor* self)
{
    // 检查format，若是NZ格式，则添加警告
    if (self->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("Format of self gets [%s], this format may lead to precision failure.",
        op::ToString(self->GetStorageFormat()).GetString());
    }
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* size, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, size, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入、输出的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输出的shape和size是否匹配
    CHECK_RET(CheckShape(self, size, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查最大维度是否超过8

    CHECK_RET(CheckMaxDimension(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查format，若是NZ格式，则添加警告
    CheckFormat(self);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpandGetWorkspaceSize(
    const aclTensor* self, const aclIntArray* size, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnExpand, DFX_IN(self, size), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, size, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (self->IsEmpty() || out->IsEmpty()) {
        OP_LOGI("aclnnExpandGetWorkspaceSize", "empty tensor!");
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfShape = self->GetViewShape();
    auto selfDimNum = selfShape.GetDimNum();
    const aclTensor* viewCopyResult;
    if (selfDimNum == 0 && size->Size() == 0) { // special case: tensor contains the scalar value of 0-dim
        viewCopyResult = l0op::ViewCopy(self, out, uniqueExecutor.get());
    } else { // norm case
        // 固定写法，将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 进行计算
        auto expandOut = l0op::Expand(selfContiguous, size, uniqueExecutor.get());
        CHECK_RET(expandOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        viewCopyResult = l0op::ViewCopy(expandOut, out, uniqueExecutor.get());
    }
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnExpand(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnExpand);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
