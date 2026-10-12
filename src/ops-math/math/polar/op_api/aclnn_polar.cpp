/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aclnn_polar.h"
#include "aclnn_kernels/contiguous.h"
#include "polar.h"
#include "../../complex/op_api/complex.h"
#include "../../mul/op_api/mul.h"
#include "../../sin/op_api/sin.h"
#include "../../cos/op_api/cos.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/platform.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT 
};
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
  op::DataType::DT_COMPLEX64
};
static bool CheckNotNull(const aclTensor* input, const aclTensor* angle, const aclTensor *out)
{
  OP_CHECK_NULL(input, return false);
  OP_CHECK_NULL(angle, return false);
  OP_CHECK_NULL(out, return false); 

  return true;
}

static bool CheckDtypeValid(const aclTensor *input, const aclTensor *angle, const aclTensor *out)
{
    // 检查input的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);

    // 检查out的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUTPUT_DTYPE_SUPPORT_LIST, return false);

    // 检查angle的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(angle, DTYPE_SUPPORT_LIST, return false);

    // input和angle数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(input, angle->GetDataType(), return false);
    return true;
}

constexpr size_t MAX_DIM_LEN = 8;

static bool CheckShape(const aclTensor *input, const aclTensor *angle, const aclTensor *out) {
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(angle, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(input, angle, broadcastShape, return false);

    if (broadcastShape != out->GetViewShape()) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
              op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
      return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* input, const aclTensor* angle, const aclTensor* out)
{
    // 输入输出的格式需要一致
    if (input->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal. input [%s], out [%s].",
            ToString(input->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    if (angle->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of angle and output should be equal. angle [%s], out [%s].",
            ToString(angle->GetStorageFormat()).GetString(), ToString(out->GetStorageFormat()).GetString());
        return false;
    }

    // input格式不能是私有格式
    if (IsPrivateFormat(input->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* angle, const aclTensor* out){
    // 1. 检查是否为空指针
    CHECK_RET(CheckNotNull(input, angle, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在支持的数据范围之内且数据类型是否相同
    CHECK_RET(CheckDtypeValid(input, angle, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入输出维度是否小于8维且输入输出维度是否相同
    CHECK_RET(CheckShape(input, angle, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查数据格式是否支持
    if (IsRegBase()){
      CHECK_RET(CheckFormat(input, angle, out), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor* input, const aclTensor* angle, aclTensor* out,
                                       uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnPolar, DFX_IN(input, angle), DFX_OUT(out));
    // 参数检查，
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  auto ret = CheckParams(input, angle, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

   if (input->IsEmpty() || angle->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
    }

  //开始计算
  //input如果非连续，要转成连续
  auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
  CHECK_RET(inputContiguous != nullptr,ACLNN_ERR_INNER_NULLPTR);
  //angle如果非连续，要转成连续
  auto angleContiguous = l0op::Contiguous(angle, uniqueExecutor.get());
  CHECK_RET(angleContiguous != nullptr,ACLNN_ERR_INNER_NULLPTR);
  const aclTensor* output = nullptr;
  if (IsRegBase()) {
    output = l0op::Polar(inputContiguous, angleContiguous, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    //调用l0算子sin和cos，得到角度的sin，cos值
    auto angleSin = l0op::Sin(angleContiguous, uniqueExecutor.get());
    CHECK_RET(angleSin != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto angleCos = l0op::Cos(angleContiguous, uniqueExecutor.get()); 
    CHECK_RET(angleCos != nullptr, ACLNN_ERR_INNER_NULLPTR);
    //调用l0算子mul,得到复数的实部和虚部
    auto absSin = l0op::Mul(angleSin, inputContiguous, uniqueExecutor.get());
    auto absCos = l0op::Mul(angleCos, inputContiguous, uniqueExecutor.get());
    CHECK_RET(absSin != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(absCos != nullptr, ACLNN_ERR_INNER_NULLPTR);
    //调用l0算子complex，完成计算
    output = l0op::Complex(absCos, absSin, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto copyResult = l0op::ViewCopy(output, out, uniqueExecutor.get());
  CHECK_RET(copyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

// 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnPolar(void *workspace, uint64_t workspaceSize,aclOpExecutor *executor, aclrtStream stream){
    L2_DFX_PHASE_2(aclnnPolar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif