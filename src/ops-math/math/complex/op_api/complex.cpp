/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "complex.h"
#include "opdev/data_type_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/platform.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(Complex);

static const std::initializer_list<op::DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND950_AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self) {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
    return CheckType(self->GetDataType(), ASCEND950_AICORE_DTYPE_SUPPORT_LIST);
  }
  return CheckType(self->GetDataType(), ASCEND910_AICORE_DTYPE_SUPPORT_LIST);
}

// AICORE算子kernel
static const aclTensor *ComplexAiCore(const aclTensor *real, const aclTensor *imag,
    aclTensor *out, DataType Tout, aclOpExecutor *executor) {
  L0_DFX(ComplexAiCore, real, imag, out, Tout);
  // 使用框架宏ADD_TO_KERNEL_OBJ_LIST，将Complex算子加入任务队列
  // op::OP_TYPE_Complex是算子的OpType，real、imag是算子的输入，out是算子的输出
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Complex, OP_INPUT(real, imag), OP_OUTPUT(out), OP_ATTR(Tout));
  OP_CHECK(ret == ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ComplexAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}

// AICPU算子kernel, tf_kernel
static const aclTensor *ComplexAiCpu(const aclTensor *real, const aclTensor *imag,
    aclTensor *out, DataType Tout, aclOpExecutor *executor) {
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将Complex算子加入任务队列
  L0_DFX(ComplexAiCpu, real, imag, out, Tout);

  static internal::AicpuTaskSpace space("Complex", ge::DEPEND_IN_SHAPE, true);
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Complex,
                                        OP_ATTR_NAMES({"Tout"}),
                                        OP_INPUT(real, imag),
                                        OP_OUTPUT(out),
                                        OP_ATTR(Tout));
  OP_CHECK(ret ==  ACL_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ComplexAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  return out;
}

const aclTensor *Complex(const aclTensor *real, const aclTensor *imag, DataType Tout, aclOpExecutor *executor) {
  if (real->GetDataType() != imag->GetDataType()) {
    return nullptr;
  }
  // 根据Complex算子语义，通过输入shape推导算子输出shape
  // 不同算子语义不同，推导的逻辑也不同，需要根据算子具体功能实现推导逻辑
  op::Shape broadcastShape;
  if (!BroadcastInferShape(real->GetViewShape(), imag->GetViewShape(), broadcastShape)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(real->GetViewShape()).GetString(),
            op::ToString(imag->GetViewShape()).GetString());
    return nullptr;
  }

  // 根据推导出的输出shape申请输出tensor
  // 第一个参数是输出shape，第二个参数是输出的dtype
  DataType outType{DataType::DT_COMPLEX64};
  if (real->GetDataType() == DataType::DT_FLOAT16) {
    outType = DataType::DT_COMPLEX32;
  }
  auto out = executor->AllocTensor(broadcastShape, Tout);

  if (IsAiCoreSupport(real) && Tout == outType) {
    return ComplexAiCore(real, imag, out, Tout, executor);
  } else {
    if (Tout == DataType::DT_COMPLEX32) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dtpye not support complex32.");
      return nullptr;
    }
    return ComplexAiCpu(real, imag, out, Tout, executor);
  }
}
}  // namespace l0op

