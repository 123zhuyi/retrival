/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "op_api/op_api_def.h"
#include "aclnn_one_hot.h"
#include "aclnn_kernels/contiguous.h"
#include "one_hot.h"
#include "aclnn_kernels/transdata.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MIN_NUM_CLASSES = 0;
static const int64_t MIN_AXIS = -1;
static const size_t SIMT_SELF_MAX_DIM_NUM = 7;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32, op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> VALUE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> VALUE_DTYPE_SUPPORT_LIST_950 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_INT64,   op::DataType::DT_INT8,  op::DataType::DT_UINT8};

static inline bool CheckNotNull(
    const aclTensor* self, const aclTensor* onValue, const aclTensor* offValue, const aclTensor* out)
{
    // self, out, onValue, offValue不能为空指针
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(onValue, return false);
    OP_CHECK_NULL(offValue, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* onValue, const aclTensor* offValue, const aclTensor* out)
{
    static bool isSimtVersion = IsRegBase();
    if (isSimtVersion) {
        // 检查self的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_950, return false);
        // 检查out的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(out, VALUE_DTYPE_SUPPORT_LIST_950, return false);
        // 检查onValue的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(onValue, VALUE_DTYPE_SUPPORT_LIST_950, return false);
        // 检查offValue的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(offValue, VALUE_DTYPE_SUPPORT_LIST_950, return false);
    } else {
        // 检查self的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
        // 检查out的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(out, VALUE_DTYPE_SUPPORT_LIST, return false);
        // 检查onValue的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(onValue, VALUE_DTYPE_SUPPORT_LIST, return false);
        // 检查offValue的数据类型是否在支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(offValue, VALUE_DTYPE_SUPPORT_LIST, return false);
    }

    // 检查onValue和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(onValue, out->GetDataType(), return false);

    // 检查offValue和out的数据类型是否一致
    OP_CHECK_DTYPE_NOT_MATCH(offValue, out->GetDataType(), return false);
    return true;
}

static bool CheckMaxDimension(
    const aclTensor* self, const aclTensor* onValue, const aclTensor* offValue, const aclTensor* out)
{
    static bool isSimtVersion = IsRegBase();
    if (isSimtVersion) {
        OP_CHECK_MAX_DIM(self, SIMT_SELF_MAX_DIM_NUM, return false);
    } else {
        OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    }
    OP_CHECK_MAX_DIM(onValue, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(offValue, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static bool CheckMaxNumClasses(const aclTensor* self, int64_t numClasses)
{
    // l2层，numClasses应大于等于0
    if (self->IsEmpty()) {
        // 空tensor时，numClasses应大于0
        if (numClasses <= MIN_NUM_CLASSES) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "NumClasses should be greater than %ld, but got %ld.", MIN_NUM_CLASSES,
                numClasses);
            return false;
        }
    } else if (numClasses < MIN_NUM_CLASSES) {
        // 其他情况下，numClasses应大于等于0
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "NumClasses should be greater and equal than %ld, but got %ld.", MIN_NUM_CLASSES,
            numClasses);
        return false;
    }
    return true;
}

static bool CheckAxisRange(const aclTensor* self, int64_t axis)
{
    if (axis < MIN_AXIS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The axis should be greater and equal than %ld, but got %ld.", MIN_AXIS, axis);
        return false;
    }

    op::Shape selfShape = self->GetViewShape();
    int64_t selfDimNum = selfShape.GetDimNum();
    if (axis > selfDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The axis should be lower and equal than %ld, but got %ld.", selfDimNum, axis);
        return false;
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* self, int64_t numClasses, int64_t axis, const aclTensor* out)
{
    int64_t selfDimNum = self->GetViewShape().GetDimNum();
    OP_CHECK_WRONG_DIMENSION(out, static_cast<size_t>(selfDimNum + 1), return false);

    int64_t dim = axis < 0 ? axis + selfDimNum + 1 : axis;
    op::Shape outShape;
    outShape.SetDimNum(selfDimNum + 1);
    for (int64_t i = 0; i < dim; i++) {
        outShape.SetDim(i, self->GetViewShape().GetDim(i));
    }
    outShape.SetDim(dim, numClasses);
    for (int64_t i = dim + 1; i < selfDimNum + 1; i++) {
        outShape.SetDim(i, self->GetViewShape().GetDim(i - 1));
    }

    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);

    return true;
}

static inline aclnnStatus CheckParams(
    const aclTensor* self, int64_t numClasses, const aclTensor* onValue, const aclTensor* offValue, int64_t axis,
    const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(self, onValue, offValue, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查self, out的数据类型是否合法
    CHECK_COND(CheckDtypeValid(self, onValue, offValue, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查最大维度是否超过8
    CHECK_COND(CheckMaxDimension(self, onValue, offValue, out), ACLNN_ERR_PARAM_INVALID, "CheckMaxDimension failed!");

    // 4. 检查numClasses的取值范围
    CHECK_COND(CheckMaxNumClasses(self, numClasses), ACLNN_ERR_PARAM_INVALID, "CheckMaxNumClasses failed!");

    // 5. 检查axis的取值范围
    CHECK_COND(CheckAxisRange(self, axis), ACLNN_ERR_PARAM_INVALID, "CheckAxisRange failed!");

    // 6. 检查out的shape是否与self的shape在axis轴插入numClasses后的shape不一致
    CHECK_COND(CheckShapeValid(self, numClasses, axis, out), ACLNN_ERR_PARAM_INVALID, "CheckShapeValid failed!");

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnOneHotGetWorkspaceSize(
    const aclTensor* self, int numClasses, const aclTensor* onValue, const aclTensor* offValue, int64_t axis,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnOneHot, DFX_IN(self, numClasses, onValue, offValue, axis), DFX_OUT(out));

    // 参数检查
    aclnnStatus ret = CheckParams(self, numClasses, onValue, offValue, axis, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    UniqueExecutor uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor* uniqueExecutorInst = uniqueExecutor.get();
    CHECK_RET(uniqueExecutorInst != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 空tensor，或numClasses为0，直接返回空tensor即可
    if (self->IsEmpty() || numClasses == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入self转换成连续的tensor
    const aclTensor* selfContiguous = l0op::Contiguous(self, uniqueExecutorInst);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto reformat = l0op::ReFormat(selfContiguous, Format::FORMAT_ND);
    CHECK_RET(reformat != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 初始化参数
    const aclTensor* numClassesTensor = uniqueExecutorInst->ConvertToTensor(
        uniqueExecutorInst->AllocScalar(numClasses),
        self->GetDataType() == op::DataType::DT_UINT8 ? op::DataType::DT_INT32 : self->GetDataType());

    // 调用OneHot算子kernel
    auto oneHotOut = l0op::OneHot(reformat, numClassesTensor, onValue, offValue, axis, uniqueExecutorInst);
    CHECK_RET(oneHotOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(oneHotOut, out, uniqueExecutorInst);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnOneHot(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnOneHot);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
