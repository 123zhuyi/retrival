/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_flatten.cpp
 * \brief
 */
#include "aclnn_flatten.h"
#include "flatten.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int64_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> Ascend910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> Ascend910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT64, op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_UINT32, op::DataType::DT_UINT64, op::DataType::DT_BOOL, op::DataType::DT_BF16};

struct ShapeResult {
  Shape outShape{};
  bool  isLegal{true};
};

static inline int64_t MakeWrapDim(int64_t dim, int64_t dimPostExpr) {
  // 支持0维tensor
  if (dimPostExpr <= 0) {
    dimPostExpr = 1;
  }
  if (dim < 0) {
    dim += dimPostExpr;
  }
  return dim;
}

static ShapeResult GetOutShape(const aclTensor *self, const int64_t axis) {
  // self的dimNum为0时，axis只能为0，out的shape为[1,1];
  // self的dimNum为1时，axis只能为0，out的shape为[1,self.GetDim(0)]。
  ShapeResult shapeResult;
  int64_t dim0 = 1;
  int64_t dim1 = 1;
  size_t selfDim = self->GetViewShape().GetDimNum();
  for (int64_t i = 0; i < axis; i++) {
    int64_t dim = self->GetViewShape().GetDim(i);
    if (dim > 0 && (dim0 > INT64_MAX / dim)) {
        shapeResult.isLegal = false;
        return shapeResult;
    }
      dim0 *= dim;
  }
  for (size_t i = axis; i < selfDim; i++) {
      int64_t dim = self->GetViewShape().GetDim(i);
    if (dim > 0 && (dim1 > INT64_MAX / dim)) {
        shapeResult.isLegal = false;
        return shapeResult;
    }
      dim1 *= dim;
  }

  shapeResult.outShape.AppendDim(dim0);
  shapeResult.outShape.AppendDim(dim1);
  return shapeResult;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  // self、index、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
  // self和out数据类型必须一样
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  // 获取芯片类型,判断是1971还是1980
  bool is910bSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
                           IsRegBase());
  const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_CURRENT =
    is910bSocVersion ? Ascend910B_DTYPE_SUPPORT_LIST : Ascend910_DTYPE_SUPPORT_LIST;

  // 检查self的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_CURRENT, return false);

  return true;
}

static bool CheckShape(const aclTensor *self, const int64_t axis, const aclTensor *out) {
  // self的dim小于8维
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

  // self的维度需要大于axis取正之后的值
  size_t selfDim = self->GetViewShape().GetDimNum();
  int64_t axisWrap = MakeWrapDim(axis, static_cast<int64_t>(selfDim));
  // 如果self的dim为0或者1，则axis只能取0。
  if (selfDim <= 1 && axisWrap > 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Dimension out of range (expected to be in range of [-1, 0], but got %ld)",
            axisWrap);
    return false;
  }
  // 其他情况则
  if ((axisWrap < 0) || (selfDim > 1 && static_cast<int64_t>(selfDim) <= axisWrap)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Dimension out of range (expected to be in range of [-%zu, %zu], but got %ld)",
            selfDim, selfDim - 1, axis);
    return false;
  }

  // 根据算子语义，推导算子输出shape
  ShapeResult outShapeResult = GetOutShape(self, axisWrap);

  if (!outShapeResult.isLegal) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The output shape exceeds the int64_t data range");    
    return false;
  }

  // 判断out的shape与推导出的输出shape是否相等
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShapeResult.outShape, return false);

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

static aclnnStatus CheckParams(const aclTensor *self, const int64_t axis, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查shape是否满足约束
  CHECK_RET(CheckShape(self, axis, out), ACLNN_ERR_PARAM_INVALID);

  // 检查format，若是NZ格式，则添加警告
  CheckFormat(self);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor *self, int64_t axis, aclTensor *out,
                                         uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnFlatten, DFX_IN(self, axis), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, axis, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 算子的空tensor的支持情况
  if (self->IsEmpty() || out->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // axis取正
  int64_t axisWrap = MakeWrapDim(axis, static_cast<int64_t>(self->GetViewShape().GetDimNum()));

  // self如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (IsRegBase()) {
    auto selfReshape = l0op::Reshape(selfContiguous, out->GetViewShape(), uniqueExecutor.get());
    CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(selfReshape, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // 调用l0算子Flatten进行计算
    auto flattenResult = l0op::Flatten(selfContiguous, uniqueExecutor.get(), axisWrap);
    CHECK_RET(flattenResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(flattenResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlatten(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnFlatten);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
