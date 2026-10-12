/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_scale.h"
#include "scale.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType> X_DTYPE_SUPPORT_LIST_ASCEND310P = {op::DataType::DT_FLOAT,
                                                                                op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_93 || socVersion == SocVersion::ASCEND910B || IsRegBase()) {
      return X_DTYPE_SUPPORT_LIST_ASCEND910B;
  } else if (socVersion == SocVersion::ASCEND310P) {
      return X_DTYPE_SUPPORT_LIST_ASCEND310P;
  } else {
      OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented", op::ToString(socVersion).GetString());
      return EMPTY_LIST;
  }
}

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* scale, const aclTensor* out) {
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(scale, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* scale, const aclTensor* bias, const aclTensor* out) {
  OP_CHECK_DTYPE_NOT_SUPPORT(x, GetDtypeSupportList(), return false);
  if (bias != nullptr) {
    OP_CHECK_DTYPE_NOT_SAME(scale, bias, return false);
  }
  OP_CHECK_DTYPE_NOT_SAME(x, scale, return false);
  OP_CHECK_DTYPE_NOT_SAME(x, out, return false);

  return true;
}

static bool CheckShapeFromBlob(const aclTensor* x, const aclTensor* scale, int64_t axis, int64_t numAxes) {
  // check with scaleFromBlob = True
  if (numAxes == 0 && scale->GetViewShape().GetShapeSize() != 1) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when num_axes is 0, scale must have one num, but shape is %s.",
            op::ToString(scale->GetViewShape()).GetString());
    return false;
  }
  
  if (numAxes == 0) {
    return true;
  }

  op::Shape xShape = x->GetViewShape();
  int64_t xRank = static_cast<int64_t>(xShape.GetDimNum());
  int64_t newAxis = axis >= 0 ? axis : xRank + axis;
  int64_t scaleLen = numAxes == -1L ? xRank - newAxis : numAxes;

  if ((scaleLen + newAxis) > xRank || scaleLen <= 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the sum of scale axis(%ld) and numAxes(%ld) must be <= xRank(%ld).", newAxis,
            numAxes, xRank);
    return false;
  }

  op::Shape expectScaleShape;
  expectScaleShape.SetDimNum(scaleLen);
  for (int64_t i = 0; i < scaleLen; i++) {
    expectScaleShape.SetDim(i, xShape.GetDim(newAxis + i));
  }

  if (expectScaleShape != scale->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "the scale must be %s, when x is %s and attr[axis(%ld), numAxes(%ld)], but get %s.",
            op::ToString(expectScaleShape).GetString(), op::ToString(xShape).GetString(), axis, numAxes,
            op::ToString(scale->GetViewShape()).GetString());
    return false;
  }

  return true;
}

static bool CheckShapeNoFromBlob(const aclTensor* x, const aclTensor* scale, int64_t axis) {
  // check with scaleFromBlob = false
  op::Shape xShape = x->GetViewShape();
  op::Shape scaleShape = scale->GetViewShape();
  int64_t xRank = static_cast<int64_t>(xShape.GetDimNum());
  int64_t newAxis = axis >= 0 ? axis : xRank + axis;
  int64_t scaleLen = static_cast<int64_t>(scaleShape.GetDimNum());
  if ((scaleLen + newAxis) > xRank || scaleLen < 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the sum of scale axis(%ld) and scale_rank(%ld) must be <= xRank(%ld).", newAxis,
            scaleLen, xRank);
    return false;
  }

  // supported scalar broadcast when scale is scalar or [1] tensor
  if (scaleShape.GetShapeSize() == 1 && scaleLen <= 1) {
    return true;
  }

  for (int64_t i = 0; i < scaleLen; i++) {
    if (xShape.GetDim(newAxis + i) != scaleShape.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the scale shape(%s) must be the same with x shape(%s) from axis dim(%ld).",
              op::ToString(scale->GetViewShape()).GetString(), op::ToString(xShape).GetString(), axis);
      return false;
    }
  }

  return true;
}

static bool CheckDim(const aclTensor* x, int64_t axis, int64_t numAxes) {
  // check with scaleFromBlob = True
  op::Shape xShape = x->GetViewShape();
  int64_t xRank = static_cast<int64_t>(xShape.GetDimNum());
  if (axis < -xRank || axis >= xRank) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "axis must be within the range of [%ld, %ld), but it is %ld.", -xRank, xRank,
            axis);
    return false;
  }

  if (numAxes < -1L) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numAxes must be >= -1, but it is %ld.", numAxes);
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* scale, const aclTensor* bias, int64_t axis,
                       int64_t numAxes, bool scaleFromBlob, const aclTensor* out) {
  // x和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, x, return false);
  if (bias != nullptr) {
    OP_CHECK_SHAPE_NOT_EQUAL(bias, scale, return false);
  }

  // x的数据维度不能超过8
  OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(scale, MAX_DIM_LEN, return false);

  // check dim
  CHECK_RET(CheckDim(x, axis, numAxes), false);

  // check with scaleFromBlob = True
  if (scaleFromBlob) {
    CHECK_RET(CheckShapeFromBlob(x, scale, axis, numAxes), false);
  } else {
    CHECK_RET(CheckShapeNoFromBlob(x, scale, axis), false);
  }

  return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* scale, const aclTensor* bias, int64_t axis,
                               int64_t numAxes, bool scaleFromBlob, const aclTensor* out) {
  CHECK_RET(CheckNotNull(x, scale, out), ACLNN_ERR_PARAM_NULLPTR);

  CHECK_RET(CheckDtypeValid(x, scale, bias, out), ACLNN_ERR_PARAM_INVALID);

  CHECK_RET(CheckShape(x, scale, bias, axis, numAxes, scaleFromBlob, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor* x, const aclTensor* scale, const aclTensor* bias,
                                       int64_t axis, int64_t numAxes, bool scaleFromBlob,
                                       aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnScale, DFX_IN(x, scale, bias, axis, numAxes, scaleFromBlob), DFX_OUT(y));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(x, scale, bias, axis, numAxes, scaleFromBlob, y);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (x->IsEmpty()) {
    *workspaceSize = 0UL;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // x如果非连续，需要转连续
  auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // scale如果非连续，需要转连续
  auto scaleContiguous = l0op::Contiguous(scale, uniqueExecutor.get());
  CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* biasContiguous = nullptr;
  // bias如果不为空指针且非连续，需要转连续
  if (bias != nullptr) {
    biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  auto scaleResult = l0op::Scale(xContiguous, scaleContiguous, biasContiguous, axis, numAxes,
                                 scaleFromBlob, uniqueExecutor.get());
  CHECK_RET(scaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult = l0op::ViewCopy(scaleResult, y, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnScale(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnScale);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
