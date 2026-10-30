/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_logical_xor.h"
#include "not_equal.h"
#include "aclnn_kernels/cast.h"
#include "../../reduce_any/op_api/reduce_any.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* LogicalXor 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             LogicalXor(workspace_4)
 *                    |
 *               Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t ADD_VIEW_SHAPE_NUM = 2;

// 根据API定义，列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_INT8,
  op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
  op::DataType::DT_BOOL, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

inline static bool CheckSocVersionIsSupportBf16(void) {
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

// 检查输入的数据类型是否在算子的支持列表内
inline static bool CheckDtypeValid(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 如果soc是1980芯片，则不支持DT_BF16，需要校验拦截
  if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16 ||
                                          other->GetDataType() == op::DataType::DT_BF16 ||
                                          out->GetDataType() == op::DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of aclnnLogicalXor is not support bfloat16 in current socversion.");
    return false;
  }

  // 检查self的数据类型是否在logical_xor算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查other的数据类型是否在logical_xor算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, DTYPE_SUPPORT_LIST, return false);
  
  // 检查out的数据类型是否在logical_xor算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
  return true;
}

inline static bool CheckShape(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(other, MAX_DIM_LEN, return false);
 
  op::Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
 
  if (broadcastShape != out->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }
  return true;
}

inline static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查双输入是否能broadcast
  CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor* InputProcessForComplex(const aclTensor* input, aclOpExecutor* executor) {
  op::Shape expectViewShape;
  auto sizeDimNum = input->GetViewShape().GetDimNum();
  auto viewSizeDimNum = sizeDimNum + 1;
  expectViewShape.SetDimNum(viewSizeDimNum);
  for (size_t i = 0; i < sizeDimNum; i++) {
    expectViewShape.SetDim(i, input->GetViewShape().GetDim(i));
  }
  expectViewShape.SetDim(sizeDimNum, ADD_VIEW_SHAPE_NUM);

  auto inputViews = executor->CreateView(input, expectViewShape, input->GetViewOffset());
  if (input->GetDataType() == op::DataType::DT_COMPLEX64) {
    inputViews->SetDataType(op::DataType::DT_FLOAT);
  } else {
    inputViews->SetDataType(op::DataType::DT_DOUBLE);
  }
  auto inputViewsCasted = l0op::Cast(inputViews, op::DataType::DT_BOOL, executor);
  CHECK_RET(inputViewsCasted != nullptr, nullptr);
  const int64_t vecReduceDim[] = {-1};
  auto inputCastedAll = l0op::ReduceAny(inputViewsCasted, executor->AllocIntArray(vecReduceDim, 1), false, executor);
  CHECK_RET(inputCastedAll != nullptr, nullptr);

  return inputCastedAll;
}

static void CheckFormat(const aclTensor* self, const aclTensor* target){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  ge::Format targetStorageFormat = target->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND || targetStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnLogicalXor only support format ND.");
  }
}

aclnnStatus aclnnLogicalXorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                            uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnLogicalXor, DFX_IN(self, other), DFX_OUT(out));

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  CheckFormat(self, other);

  // 空tensor处理
  if (self->IsEmpty() || other->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 检查self的类型是否为复数
  auto selfContigsDtype = selfContiguous->GetDataType();
  const aclTensor* selfCastedAll = nullptr;
  if (selfContigsDtype == op::DataType::DT_COMPLEX64 || selfContigsDtype == op::DataType::DT_COMPLEX128) {
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN - 1, return ACLNN_ERR_PARAM_INVALID);
    selfCastedAll = InputProcessForComplex(selfContiguous, uniqueExecutor.get());
  } else {
    // 将输入self的数据类型转换成DT_BOOL类型
    selfCastedAll = l0op::Cast(selfContiguous, op::DataType::DT_BOOL, uniqueExecutor.get());
  }
  CHECK_RET(selfCastedAll != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 检查other的类型是否为复数
  auto otherContigsDtype = otherContiguous->GetDataType();
  const aclTensor* otherCastedAll = nullptr;
  if (otherContigsDtype == op::DataType::DT_COMPLEX64 || otherContigsDtype == op::DataType::DT_COMPLEX128) {
    OP_CHECK_MAX_DIM(other, MAX_DIM_LEN - 1, return ACLNN_ERR_PARAM_INVALID);
    otherCastedAll = InputProcessForComplex(otherContiguous, uniqueExecutor.get());
  } else {
    // 将输入self的数据类型转换成DT_BOOL类型
    otherCastedAll = l0op::Cast(otherContiguous, op::DataType::DT_BOOL, uniqueExecutor.get());
  }
  CHECK_RET(otherCastedAll != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行NotEqual计算
  auto logical_xorOpOut = l0op::NotEqual(selfCastedAll, otherCastedAll, uniqueExecutor.get());
  CHECK_RET(logical_xorOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(logical_xorOpOut, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogicalXor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
   L2_DFX_PHASE_2(aclnnLogicalXor);
  // 调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
