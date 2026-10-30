/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_s_where.h"
#include "select.h"
#include "aclnn_kernels/cast.h"
#include "math/expand/op_api/expand.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t MAX_DIM = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,     op::DataType::DT_UINT64, op::DataType::DT_INT64,
    op::DataType::DT_UINT32,    op::DataType::DT_FLOAT16,   op::DataType::DT_UINT16, op::DataType::DT_INT16,
    op::DataType::DT_INT8,      op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910B_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_UINT64, op::DataType::DT_INT64,
    op::DataType::DT_UINT32,    op::DataType::DT_FLOAT16,    op::DataType::DT_UINT16, op::DataType::DT_INT16,
    op::DataType::DT_INT8,      op::DataType::DT_UINT8,      op::DataType::DT_DOUBLE, op::DataType::DT_BOOL,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> CONDITION_DTYPE_SUPPORT = {
  op::DataType::DT_BOOL, op::DataType::DT_UINT8};

static bool CheckDtypeValid(
    const aclTensor* self, const aclTensor* other, const aclTensor* condition, const aclTensor* out)
{
    const std::initializer_list<op::DataType> dtypeSupportList =
        GetDtypeSupportListV3(DTYPE_SUPPORT_910B_LIST, DTYPE_SUPPORT_910_LIST);

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);

  OP_CHECK_DTYPE_NOT_SUPPORT(other, dtypeSupportList, return false);

  OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);

  OP_CHECK_DTYPE_NOT_SUPPORT(condition, CONDITION_DTYPE_SUPPORT, return false);
  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* condition, const aclTensor* out)
{
  OP_CHECK_MAX_DIM(self, MAX_DIM, return false);
  OP_CHECK_MAX_DIM(other, MAX_DIM, return false);
  OP_CHECK_MAX_DIM(condition, MAX_DIM, return false);

  op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);

  if (!BroadcastInferShape(condition->GetViewShape(), broadcastShape, broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of self other and condition can't broadcast.");
    return false;
  }

  if (broadcastShape != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }

  return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclTensor* condition, const aclTensor* other, const aclTensor* out)
{
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull4Tensor(self, other, condition, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, other, condition, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入shape
  CHECK_RET(CheckShape(self, other, condition, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSWhereGetWorkspaceSize(
    const aclTensor* condition, const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
  L2_DFX_PHASE_1(aclnnSWhere, DFX_IN(condition, self, other), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, condition, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // add算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (self->IsEmpty() || condition->IsEmpty() || other->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入condition转换成连续的tensor
  auto conditionContiguous = l0op::Contiguous(condition, uniqueExecutor.get());
  CHECK_RET(conditionContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 调用cast转换类型
  auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
  auto conditionCast = l0op::Cast(conditionContiguous, op::DataType::DT_BOOL, uniqueExecutor.get());
  CHECK_RET(conditionCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 根据self和other的数据决定是否进行Cast
  const aclTensor* selRes = nullptr;
  if (selfContiguous->GetDataType() == DataType::DT_BOOL && otherContiguous->GetDataType() == DataType::DT_BOOL) {
    // 创建输入输出的View，并将数据类型设置为int8
        auto selfView = uniqueExecutor.get()->CreateView(
            selfContiguous, selfContiguous->GetViewShape(), selfContiguous->GetViewOffset());
    CHECK_RET(selfView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfView->SetDataType(DataType::DT_INT8);
        auto otherView = uniqueExecutor.get()->CreateView(
            otherContiguous, otherContiguous->GetViewShape(), otherContiguous->GetViewOffset());
    CHECK_RET(otherView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    otherView->SetDataType(DataType::DT_INT8);
    // 调用SelectV2算子计算
    auto opRes = l0op::SelectV2(conditionCast, selfView, otherView, uniqueExecutor.get());
    CHECK_RET(opRes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 将结果设置为原始的Bool类型
    auto resView = uniqueExecutor.get()->CreateView(opRes, opRes->GetViewShape(), opRes->GetViewOffset());
    CHECK_RET(resView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    resView->SetDataType(DataType::DT_BOOL);
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    selRes = l0op::ViewCopy(resView, out, uniqueExecutor.get());
  } else {
    auto selfCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto otherCast = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto opOut = l0op::SelectV2(conditionCast, selfCast, otherCast, uniqueExecutor.get());
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selCast = l0op::Cast(opOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(selCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    selRes = l0op::ViewCopy(selCast, out, uniqueExecutor.get());
  }
  CHECK_RET(selRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnSWhere(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnSWhere);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
