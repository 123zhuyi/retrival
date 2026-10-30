/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_right_shift.h"
#include "right_shift.h"
#include "conversion/view_copy/op_api/view_copy.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t MAX_INPUT_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_INT8, op::DataType::DT_INT16,
  op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_UINT8, op::DataType::DT_UINT16,
  op::DataType::DT_UINT32, op::DataType::DT_UINT64};

static bool CheckNotNull(const aclTensor *input, const aclTensor *shiftBits, const aclTensor *out)
{
  OP_CHECK_NULL(input, return false);
  OP_CHECK_NULL(shiftBits, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *input, const aclTensor *shiftBits, const aclTensor *out)
{
  OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(shiftBits, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckShape(const aclTensor *input, const aclTensor *shiftBits, const aclTensor *out)
{
  const int64_t inputDim = input->GetViewShape().GetDimNum();
  const int64_t shiftBitsDim = shiftBits->GetViewShape().GetDimNum();
  OP_CHECK(
    inputDim <= MAX_INPUT_DIM,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dim num should be less than or equal to 8."), return false);
  OP_CHECK(
      shiftBitsDim <= MAX_INPUT_DIM,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ShiftBits dim num should be less than or equal to 8."), return false);
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(input, shiftBits, broadcastShape, return false);
  if (broadcastShape != out->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *input, const aclTensor *shiftBits, const aclTensor *out)
{
  CHECK_RET(CheckNotNull(input, shiftBits, out), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(input, shiftBits, out), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(input, shiftBits, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static bool CheckPromoteType(const aclTensor *input, const aclTensor *shiftBits, const aclTensor *out, op::DataType promoteType) {
  // 检查input和shiftBits能否做数据类型推导
  if (promoteType == DataType::DT_UNDEFINED) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype [%s] and shiftBits dtype [%s] can not promote dtype.",
              op::ToString(input->GetDataType()).GetString(), op::ToString(shiftBits->GetDataType()).GetString());
      return false;
  }
  // 检查推导后的数据类型是否能转换为输出的数据类型
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  return true;
}

aclnnStatus aclnnRightShiftGetWorkspaceSize(const aclTensor *input, const aclTensor *shiftBits,
                                            aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnRightShift, DFX_IN(input, shiftBits), DFX_OUT(out));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数dtype检查
  auto ret = CheckParams(input, shiftBits, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (input->IsEmpty() || shiftBits->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("The input or shiftBits is empty, skip rightshift.");
    return ACLNN_SUCCESS;
  }
  //
  auto promoteType = op::PromoteType(input->GetDataType(), shiftBits->GetDataType());
  CHECK_RET(CheckPromoteType(input, shiftBits, out, promoteType), ACLNN_ERR_PARAM_INVALID);

  auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
  CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto inputCasted = l0op::Cast(inputContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(inputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto shiftBitsContiguous = l0op::Contiguous(shiftBits, uniqueExecutor.get());
  CHECK_RET(shiftBitsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto shiftBitsdCasted = l0op::Cast(shiftBitsContiguous, promoteType, uniqueExecutor.get());
  CHECK_RET(shiftBitsdCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto outShifted = l0op::RightShift(inputCasted, shiftBitsdCasted, uniqueExecutor.get());
  CHECK_RET(outShifted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果转换成输出out的数据类型
  auto outCasted = l0op::Cast(outShifted, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(outCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果outCasted拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(outCasted, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnRightShift(void *workspace, uint64_t workspaceSize,
                            aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnRightShift);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif
