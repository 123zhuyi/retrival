/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_complex.h"
#include "aclnn_kernels/cast.h"
#include "complex.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor *real, const aclTensor *imag, const aclTensor *out) {
  OP_CHECK_NULL(real, return false);
  OP_CHECK_NULL(imag, return false);
  OP_CHECK_NULL(out, return false);

  return true;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                             DataType::DT_FLOAT,
                                                                             DataType::DT_DOUBLE};
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {DataType::DT_COMPLEX32,
                                                                              DataType::DT_COMPLEX64,
                                                                              DataType::DT_COMPLEX128};
// 根据API定义，列出所有可能的输入输出dtype pair
static const std::initializer_list<std::pair<op::DataType, op::DataType>> DTYPE_PAIR = {
  {DataType::DT_FLOAT16, DataType::DT_COMPLEX32},
  {DataType::DT_FLOAT, DataType::DT_COMPLEX64},
  {DataType::DT_DOUBLE, DataType::DT_COMPLEX128}
};

static bool CheckDtypeValid(const aclTensor *real, const aclTensor *imag, const aclTensor *out) {
  // 检查real的数据类型是否在Complex算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(real, INPUT_DTYPE_SUPPORT_LIST, return false);

  // 检查imag的数据类型是否在Complex算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(imag, INPUT_DTYPE_SUPPORT_LIST, return false);

  // 检查out的数据类型是否在Complex算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST, return false);

  // 检查real和imag的数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(real, imag, return false);

  // 检查real/imag/out的dtype推导关系
  auto inOutPair = std::pair<op::DataType, op::DataType>(real->GetDataType(), out->GetDataType());
  bool isSupport = std::find(DTYPE_PAIR.begin(), DTYPE_PAIR.end(), inOutPair) != DTYPE_PAIR.end();
  if (!isSupport) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype and out dtype must be paired. Current dtype is %s and %s.",
            op::ToString(real->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
  }
  return isSupport;
}

static bool CheckFormat(const aclTensor* real, const aclTensor* imag, const aclTensor* out)
{
    // 检查 real、imag、out 三者的 format 是否一致
    if (real->GetStorageFormat() != imag->GetStorageFormat() ||
        real->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of real, imag and out should be equal, "
            "real [%s], imag [%s], out [%s].",
            op::ToString(real->GetStorageFormat()).GetString(),
            op::ToString(imag->GetStorageFormat()).GetString(),
            op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // 拦截 real 的 format 非 ND 的场景
    if (op::IsPrivateFormat(real->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Format of real only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW, "
                "but got [%s].",
                op::ToString(real->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static bool CheckOutShape(const aclTensor *real, const aclTensor *imag, const aclTensor *out) {
  const size_t MAX_DIM = 8;
  OP_CHECK_MAX_DIM(real, MAX_DIM, return false);
  OP_CHECK_MAX_DIM(imag, MAX_DIM, return false);

  op::Shape outShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(real, imag, outShape, return false);

  if (outShape != out->GetViewShape()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BroadcastShape %s is not equal out's shape %s.",
    op::ToString(outShape).GetString(), op::ToString(out->GetViewShape()).GetString());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *real, const aclTensor *imag, const aclTensor *out) {
  // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(real, imag, out), ACLNN_ERR_PARAM_INVALID);

  // 2. 检查双输入是否能broadcast,检查boradcast后的输出与out是否一致
  CHECK_RET(CheckOutShape(real, imag, out), ACLNN_ERR_PARAM_INVALID);

  // 3. A5后的场景Format校验
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
    CHECK_RET(CheckFormat(real, imag, out), ACLNN_ERR_PARAM_INVALID);
  }

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnComplexGetWorkspaceSize(const aclTensor *real, const aclTensor *imag, aclTensor *out,
                                         uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnComplex, DFX_IN(real, imag), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 检查三个入参参数是否为空指针
  CHECK_RET(CheckNotNull(real, imag, out), ACLNN_ERR_PARAM_NULLPTR);

  // 空tensor处理
  if (real->IsEmpty() || imag->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，参数检查
  auto ret = CheckParams(real, imag, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，将输入real转换成连续的tensor
  auto realContiguous = l0op::Contiguous(real, uniqueExecutor.get());
  CHECK_RET(realContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入imag转换成连续的tensor
  auto imagContiguous = l0op::Contiguous(imag, uniqueExecutor.get());
  CHECK_RET(imagContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Complex算子kernel
  auto complexOpOut = l0op::Complex(realContiguous, imagContiguous, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(complexOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(complexOpOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnComplex(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnComplex);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
