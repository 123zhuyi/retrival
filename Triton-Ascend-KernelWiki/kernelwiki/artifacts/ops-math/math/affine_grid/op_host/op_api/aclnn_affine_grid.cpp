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
 * \file aclnn_affine_grid.cpp
 * \brief
 */

#include "aclnn_affine_grid.h"
#include "aclnn_kernels/contiguous.h"
#include "affine_grid.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t DIM_LEN = 3; 
static const int64_t DIM_N = 0;
static const int64_t DIM_C = 1;
static const int64_t DIM_D = 2;
static const int64_t DIM_H = 3;
static const int64_t DIM_W = 4;
static const int64_t DIM_H_2D = 2;
static const int64_t DIM_W_2D = 3;
static const int64_t AXIS_2D = 2;
static const int64_t AXIS = 3;
static const int64_t SECOND_DIM = 2;

// 根据API定义，需要列出所能支持的所有dtype，算子只支持AICPU
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor *theta, const aclIntArray *size, const aclTensor *out) {
  OP_CHECK_NULL(theta, return false);
  OP_CHECK_NULL(size, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *theta, const aclTensor *out) {
  // 根据芯片类型获取数据类型支持列表
  bool isAscend910BSocVersion = (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
                                   GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E);
  const std::initializer_list<op::DataType> dtypeSupportList =
      isAscend910BSocVersion ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;
  // 检查theta的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(theta, dtypeSupportList, return false);
  // 检查out的数据类型是否与self一致
  OP_CHECK_DTYPE_NOT_MATCH(out, theta->GetDataType(), return false);
  return true;
}

static bool CheckShape(const aclTensor *theta, const aclIntArray *size, const aclTensor *out) {
  if (theta->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected theta to be not null tensor.");
    return false;
  }
  // size的大小是4或者5
  if (size->Size() == 4) { // size的大小是4时，theta的维度为(N, 2, 3)
    if (theta->GetViewShape().GetDimNum() != DIM_LEN || theta->GetViewShape().GetDim(1) != DIM_H_2D ||
        theta->GetViewShape().GetDim(SECOND_DIM) != DIM_W_2D || theta->GetViewShape().GetDim(DIM_N) != (*size)[DIM_N]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected a batch of 2D affine matrices of shape Nx2x3 for size "
                                       "[%ld, %ld, %ld, %ld]. Got %s.", (*size)[DIM_N], (*size)[DIM_C],
              (*size)[DIM_H_2D], (*size)[DIM_W_2D], op::ToString(theta->GetViewShape()).GetString());
      return false;
    }
    op::Shape outShape;
    outShape.AppendDim((*size)[DIM_N]);
    outShape.AppendDim((*size)[DIM_H_2D]);
    outShape.AppendDim((*size)[DIM_W_2D]);
    outShape.AppendDim(AXIS_2D);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);
  } else if (size->Size() == 5) { // size的大小是5时，theta的维度为(N, 3, 4)
    if (theta->GetViewShape().GetDimNum() != DIM_LEN || theta->GetViewShape().GetDim(1) != DIM_H ||
        theta->GetViewShape().GetDim(SECOND_DIM) != DIM_W || theta->GetViewShape().GetDim(DIM_N) != (*size)[DIM_N]) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected a batch of 3D affine matrices of shape Nx3x4 for size "
                                       "[%ld, %ld, %ld, %ld, %ld]. Got %s.", (*size)[DIM_N], (*size)[DIM_C],
              (*size)[DIM_D], (*size)[DIM_H], (*size)[DIM_W], op::ToString(theta->GetViewShape()).GetString());
      return false;
    }
    op::Shape outShape;
    outShape.AppendDim((*size)[DIM_N]);
    outShape.AppendDim((*size)[DIM_D]);
    outShape.AppendDim((*size)[DIM_H]);
    outShape.AppendDim((*size)[DIM_W]);
    outShape.AppendDim(AXIS);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, outShape, return false);
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AffineGridGenerator needs 4d or 5d size(input).");
    return false;
  }
  return true;
}

static void CheckFormat(const aclTensor* theta) {
    ge::Format thetaStorageFormat = theta->GetStorageFormat();
    if (thetaStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnAffineGrid doesn't support format NZ.");
    }
}

static inline aclnnStatus CheckParams(const aclTensor *theta, const aclIntArray *size, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(theta, size, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(theta, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查tensor的维度
  CHECK_RET(CheckShape(theta, size, out), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAffineGridGetWorkspaceSize(const aclTensor *theta, const aclIntArray *size, bool alignCorners,
                                            aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  
  L2_DFX_PHASE_1(aclnnAffineGrid, DFX_IN(theta, size, alignCorners), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(theta, size, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  // 检查格式
  CheckFormat(theta);

  // 固定写法，将输入theta转换成连续的tensor
  auto thetaContiguous = l0op::Contiguous(theta, uniqueExecutor.get());
  CHECK_RET(thetaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 进行AffineGrid计算
  const aclTensor *affineGridOut = l0op::AffineGrid(thetaContiguous, size, alignCorners, uniqueExecutor.get());
  CHECK_RET(affineGridOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // size的大小是4时，输出的维度为(N, H, W, 2);size的大小是5时，输出的维度为(N, D, H, W, 3)
  const int64_t dim5Shape[] = {(*size)[DIM_N], (*size)[DIM_D], (*size)[DIM_H], (*size)[DIM_W], AXIS};
  const int64_t dim4Shape[] = {(*size)[DIM_N], (*size)[DIM_H_2D], (*size)[DIM_W_2D], AXIS_2D};

  auto outReshape = l0op::Reshape(affineGridOut, uniqueExecutor.get()->AllocIntArray((size->Size() == 4) ? dim4Shape :
                                                                                     dim5Shape, size->Size()),
                                  uniqueExecutor.get());
  CHECK_RET(outReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyMinResult = l0op::ViewCopy(outReshape, out, uniqueExecutor.get());
  CHECK_RET(viewCopyMinResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnAffineGrid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnAffineGrid);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif