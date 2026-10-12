/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tanh_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_tanh_backward.h"

#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  switch (curArch) {
    case NpuArch::DAV_2201:
    case NpuArch::DAV_3510: {
      return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    case NpuArch::DAV_1001: {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
    default: {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
  }
}

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput) {
  OP_CHECK_NULL(gradOutput, return false);
  OP_CHECK_NULL(output, return false);
  OP_CHECK_NULL(gradInput, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput) {
  const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
  // 检查gradOutput的数据类型是否在TanhGrad算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);

  // 检查output的数据类型是否在TanhGrad算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(output, dtypeSupportList, return false);

  // 检查gradInput的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, dtypeSupportList, return false);

  // 检查gradOutput和output/gradInput是否dtype一致
  OP_CHECK_DTYPE_NOT_MATCH(gradOutput, output->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(gradOutput, gradInput->GetDataType(), return false);

  return true;
}

static bool CheckShapeValid(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput) {
  OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(output, MAX_SUPPORT_DIMS_NUMS, return false);

  // 检查gradOutput和output broadcast之后与gradInput是否shape一致
  Shape dstShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, output, dstShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, dstShape, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(gradOutput, output, gradInput), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(gradOutput, output, gradInput), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查是否shape一致
  CHECK_RET(CheckShapeValid(gradOutput, output, gradInput), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTanhBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output,
                                              aclTensor *gradInput, uint64_t *workspaceSize,
                                              aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTanhBackward, DFX_IN(gradOutput, output), DFX_OUT(gradInput));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(gradOutput, output, gradInput);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // TanhGrad算子的空tensor在kernel中支持
  if (gradOutput->IsEmpty() || output->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入gradOutputContiguous转换成连续的tensor
  auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
  CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入output转换成连续的tensor
  auto outputContiguous = l0op::Contiguous(output, uniqueExecutor.get());
  CHECK_RET(outputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* tanhBackwardOpOut = nullptr;
  if (IsRegBase()) {
    // 调用TanhGrad算子kernel
    auto tanhBackwardOpOutBeforeCast = l0op::TanhGrad(gradOutputContiguous, outputContiguous, uniqueExecutor.get());
    CHECK_RET(tanhBackwardOpOutBeforeCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用Cast算子kernel
    tanhBackwardOpOut = l0op::Cast(tanhBackwardOpOutBeforeCast, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(tanhBackwardOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // 调用TanhGrad算子kernel
    tanhBackwardOpOut = l0op::TanhGrad(gradOutputContiguous, outputContiguous, uniqueExecutor.get());
    CHECK_RET(tanhBackwardOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(tanhBackwardOpOut, gradInput, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTanhBackward(void *workspace, uint64_t workspaceSize,
                              aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTanhBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
