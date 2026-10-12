/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_equal.h"
#include "aclnn_kernels/cast.h"
#include "conversion/fill/op_api/fill.h"
#include "tensor_equal.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* TensorEqual 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_1)
 *               \               /
 *             TensorEqual(workspace_2)
 *                        |
 *                     ViewCopy
 *                        |
 *                      result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,  op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,   op::DataType::DT_UINT8,  op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64,
    op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_INT32,  op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,  op::DataType::DT_INT8,   op::DataType::DT_UINT8,  op::DataType::DT_BOOL,
    op::DataType::DT_DOUBLE, op::DataType::DT_UINT16, op::DataType::DT_UINT32, op::DataType::DT_UINT64};

// 列出out所能支持的所有dtype
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL};

// 算子支持的最大维度
static const size_t DIM_SUPPORT_MAX = 8;

static bool CheckNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckSocVersionGe910B(void) {
  return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
         GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 如果soc是1980芯片，则不支持DT_BF16，需要校验拦截，否则Cast报错
  bool is910BSocVersion = CheckSocVersionGe910B();
  const std::initializer_list<DataType> DTYPE_SUPPORT_LIST =
    is910BSocVersion ? DTYPE_SUPPORT_910B_LIST : DTYPE_SUPPORT_910_LIST;

  // 检查self和other能否做数据类型推导
  op::DataType promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
    return false;
  }

  // 检查promoteType的数据类型是否在equal算子的支持列表内
  if (!CheckType(promoteType, DTYPE_SUPPORT_LIST)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s get promoteType dtype %s should be in " \
            "dtype support list [%s].", op::ToString(self->GetDataType()).GetString(),
            op::ToString(other->GetDataType()).GetString(), op::ToString(promoteType).GetString(),
            op::ToString(DTYPE_SUPPORT_LIST).GetString());
    return false;
  }

  // 检查out的数据类型是否是BOOL
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckMaxShape(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, DIM_SUPPORT_MAX, return false);
  OP_CHECK_MAX_DIM(other, DIM_SUPPORT_MAX, return false);
  OP_CHECK_MAX_DIM(out, DIM_SUPPORT_MAX, return false);
  return true;
}

static bool CheckOutShape(const aclTensor *out) {
  op::Shape outShape;
  outShape.SetDimNum(1);
  outShape.SetDim(0, 1);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 1. 检查两个入参参数是否为空指针；out为空指针时不报错，结果输出None(python中的空指针)
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 入参tensor最大维度检查
  CHECK_RET(CheckMaxShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 输出tensor形状检查
  CHECK_RET(CheckOutShape(out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnEqual, DFX_IN(self, other), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if ((self->GetViewShape() != other->GetViewShape()) || (self->IsEmpty() && other->IsEmpty())) {
    int64_t dim = 1;
    const aclTensor *dims = (uniqueExecutor.get())->ConvertToTensor(&dim, 1, op::DataType::DT_INT64);
    aclIntArray *outShape = (uniqueExecutor.get())->AllocIntArray(&dim, 1);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // [False]
    int64_t val = 0;
    // [True]
    if ((self->IsEmpty() && other->IsEmpty()) && (self->GetViewShape() == other->GetViewShape())) {
      val = 1;
    }
    const aclTensor *value = (uniqueExecutor.get())->ConvertToTensor(&val, 1, out->GetDataType());
    CHECK_RET(value != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Fill算子kernel，对一维一元张量赋予bool值
    auto equalOpOut = l0op::Fill(dims, value, outShape, uniqueExecutor.get());
    CHECK_RET(equalOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(equalOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
  if (promoteType == op::DataType::DT_BF16) {
      promoteType = op::DataType::DT_FLOAT;
  }
  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
  auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用TensorEqual算子kernel
  auto equalOpOut = l0op::TensorEqual(selfCasted, otherCasted, uniqueExecutor.get());
  CHECK_RET(equalOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto viewCopyResult = l0op::ViewCopy(equalOpOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEqual(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnEqual);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
