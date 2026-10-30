/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_logical_or.h"
#include "logical_or.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include <set>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* LogicalOr 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *     Cast(workspace_1)     Cast(workspace_3)
 *               \            /
 *             LogicalOr(workspace_4)
 *                    |
 *               Cast(workspace_5)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

constexpr size_t MAX_DIM_LEN = 8;

// 所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_FLOAT16,  op::DataType::DT_INT32,      op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16,      op::DataType::DT_INT8,     op::DataType::DT_UINT8,      op::DataType::DT_INT16,
    op::DataType::DT_INT64,     op::DataType::DT_BOOL,     op::DataType::DT_COMPLEX64,  op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64,  op::DataType::DT_UINT64,     op::DataType::DT_INT32,    op::DataType::DT_UINT32,
    op::DataType::DT_INT16,  op::DataType::DT_UINT16,     op::DataType::DT_INT8,     op::DataType::DT_UINT8,
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT,      op::DataType::DT_FLOAT16,  op::DataType::DT_BF16,
    op::DataType::DT_BOOL,   op::DataType::DT_COMPLEX128, op::DataType::DT_COMPLEX64};

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

inline static bool CheckDtypeValid(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 如果soc是1980芯片，则不支持DT_BF16，需要校验拦截
  if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16 ||
                                          other->GetDataType() == op::DataType::DT_BF16 ||
                                          out->GetDataType() == op::DataType::DT_BF16)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of aclnnLogicalOr is not support bfloat16 in current socversion.");
    return false;
  }

  auto outSuportList = IsRegBase() ? OUT_DTYPE_SUPPORT_LIST : DTYPE_SUPPORT_LIST;
  // 检查self的数据类型是否在logical_or算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

  // 检查other的数据类型是否在logical_or算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(other, DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在logical_or算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, outSuportList, return false);
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

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查双输入是否能broadcast
  CHECK_RET(CheckShape(self, other, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus GetWorkspaceSizeCommon(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (self->IsEmpty() || other->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  if (self->GetStorageFormat() != Format::FORMAT_ND) {
    OP_LOGW("Format only support ND");
  }

  // 将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // kernel支持的数据类型：BOOL/INT8/UINT8/INT16/INT32/INT64/FLOAT16/BF16/FLOAT
  // 策略：
  //   1. 两个输入类型相同且kernel支持 → 直接调用kernel，无需Cast
  //   2. 两个输入类型不同或kernel不支持 → 都Cast到BOOL，调用BOOL kernel
  static const std::set<op::DataType> KERNEL_SUPPORTED_DTYPE_SET = {
      op::DataType::DT_BOOL,    op::DataType::DT_INT8,   op::DataType::DT_UINT8,
      op::DataType::DT_INT16,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
      op::DataType::DT_FLOAT16, op::DataType::DT_BF16,   op::DataType::DT_FLOAT};

  op::DataType selfDtype = selfContiguous->GetDataType();
  op::DataType otherDtype = otherContiguous->GetDataType();
  if (IsRegBase() && selfDtype == otherDtype && KERNEL_SUPPORTED_DTYPE_SET.count(selfDtype)) {
    // 两个输入类型相同且kernel支持 → 直接调用kernel
    auto logical_orOpOut = l0op::LogicalOr(selfContiguous, otherContiguous, uniqueExecutor.get());
    CHECK_RET(logical_orOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = (out->GetDataType() == op::DataType::DT_BOOL) ?
      logical_orOpOut : l0op::Cast(logical_orOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // 两个输入类型不同或kernel不支持 → 都Cast到BOOL，调用BOOL kernel
    auto selfBool = (selfDtype == op::DataType::DT_BOOL) ?
      selfContiguous : l0op::Cast(selfContiguous, op::DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(selfBool != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto otherBool = (otherDtype == op::DataType::DT_BOOL) ?
      otherContiguous : l0op::Cast(otherContiguous, op::DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(otherBool != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto logical_orOpOut = l0op::LogicalOr(selfBool, otherBool, uniqueExecutor.get());
    CHECK_RET(logical_orOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = (out->GetDataType() == op::DataType::DT_BOOL) ?
      logical_orOpOut : l0op::Cast(logical_orOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogicalOrGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                           uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnLogicalOr, DFX_IN(self, other), DFX_OUT(out));
  return GetWorkspaceSizeCommon(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnLogicalOr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
   L2_DFX_PHASE_2(aclnnLogicalOr);
  // 调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceLogicalOrGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize,
                                                  aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceLogicalOr, DFX_IN(selfRef, other), DFX_OUT(selfRef));
  return GetWorkspaceSizeCommon(selfRef, other, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceLogicalOr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                  aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceLogicalOr);
  // 调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif