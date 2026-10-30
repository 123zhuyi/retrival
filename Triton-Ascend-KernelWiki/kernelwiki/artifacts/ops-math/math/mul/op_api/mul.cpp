/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Mul);

static const std::initializer_list<DataType> ASCEND910_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT8, DataType::DT_UINT8,
    DataType::DT_INT64, DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT8, DataType::DT_UINT8,
    DataType::DT_INT64, DataType::DT_BF16, DataType::DT_COMPLEX64, DataType::DT_BOOL};

static const std::initializer_list<DataType> REGBASE_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,     DataType::DT_FLOAT16,   DataType::DT_BF16,  DataType::DT_INT32,
    DataType::DT_INT8,      DataType::DT_UINT8,     DataType::DT_INT64, ge::DT_INT16,
    DataType::DT_COMPLEX32, DataType::DT_COMPLEX64, DataType::DT_BOOL};

static const std::initializer_list<DataType> ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_INT8, DataType::DT_UINT8};

static constexpr int64_t DIM_FOUR = 4;

static inline const std::initializer_list<DataType>& GetAiCoreDtypeSupportListBySocVersion() {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  switch (curArch) {
    case NpuArch::DAV_2201: {
      return ASCEND910B_AICORE_DTYPE_SUPPORT_LIST;
    }
    case NpuArch::DAV_3510: {
      return REGBASE_AICORE_DTYPE_SUPPORT_LIST;
    }
    case NpuArch::DAV_1001: {
      return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
    }
    case NpuArch::DAV_3102: {
      return ASCEND610LITE_AICORE_DTYPE_SUPPORT_LIST;
    }
    default: {
      return ASCEND910_AICORE_DTYPE_SUPPORT_LIST;
    }
  }
}

bool IsDoubleSupport(const aclTensor *self, const aclTensor *other) {
  if (IsRegBase() && self->GetDataType() == DataType::DT_DOUBLE && 
      other->GetDataType() == DataType::DT_DOUBLE) {
    return true;
  }
  return false;
}

static bool isBroadcastTemplateNonContiguousSupport(const aclTensor* input)
{
    auto viewShape = input->GetViewShape();
    size_t shapeDim = viewShape.GetDimNum();

    // dim > 4 不支持
    if (shapeDim > DIM_FOUR) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. shapeDim: %d > 4", shapeDim);
        return false;
    }

    // 仅regbase类的芯片支持
    if (!IsRegBase()) {
        OP_LOGI("Broadcast Template NonContiguous UnSupported. not RegBase");
        return false;
    }

    return true;
}

// 根据芯片类型、dtype判断算子是否支持走AiCore
inline static bool IsAiCoreSupport(const aclTensor *self) {
  return CheckType(self->GetDataType(), GetAiCoreDtypeSupportListBySocVersion());
}

bool IsMulSupportNonContiguous(const aclTensor* self, const aclTensor *other) {
  bool isSupportNonContiguous = isBroadcastTemplateNonContiguousSupport(self) && isBroadcastTemplateNonContiguousSupport(other);
  return isSupportNonContiguous && ((IsAiCoreSupport(self) && IsAiCoreSupport(other)) || IsDoubleSupport(self, other));
}

// AICORE算子kernel
static const aclTensor *MulAiCore(const aclTensor *self, const aclTensor *other, const aclTensor *mulOut,
                                  aclOpExecutor *executor) {
  L0_DFX(MulAiCore, self, other, mulOut);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Mul算子加入任务队列
  // Mul是算子的OpType，self、other是算子的输入，mulOut是算子的输出
  ADD_TO_LAUNCHER_LIST_AICORE(Mul, OP_INPUT(self, other), OP_OUTPUT(mulOut));
  return mulOut;
}

// AICPU算子kernel
static const aclTensor *MulAiCpu(const aclTensor *self, const aclTensor *other, aclTensor *mulOut,
                                 aclOpExecutor *executor) {
  L0_DFX(MulAiCpu, self, other);
  // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Mul算子加入任务队列
  // Mul是算子的OpType，self、other是算子的输入，mulOut是算子的输出
  static internal::AicpuTaskSpace space("Mul");
  auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Mul, OP_ATTR_NAMES(), OP_INPUT(self, other), OP_OUTPUT(mulOut));
  CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
  return mulOut;
}

const aclTensor *Mul(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor) {
  Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return nullptr);

  bool isMixDataType = (self->GetDataType() == DataType::DT_FLOAT16 && other->GetDataType() == DataType::DT_FLOAT) ||
                       (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_FLOAT16) ||
                       (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_FLOAT) ||
                       (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_BF16);

  auto mulOut = isMixDataType ? executor->AllocTensor(broadcastShape, DataType::DT_FLOAT)
                              : executor->AllocTensor(broadcastShape, self->GetDataType());
  if (isMixDataType || (IsAiCoreSupport(self) && IsAiCoreSupport(other)) || IsDoubleSupport(self, other)) {
    return MulAiCore(self, other, mulOut, executor);
  }

  return MulAiCpu(self, other, mulOut, executor);
}

const aclTensor *MulInplace(const aclTensor *self, const aclTensor *rfRes, aclOpExecutor *executor) {
  Shape broadcastShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, rfRes, broadcastShape, return nullptr);

  // 校验输出tensor的shape和rfRes tensor一致
  if (broadcastShape != rfRes->GetViewShape()) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self and rfRes broadcastShape [%s] not equal to rfRes shape [%s], do no support inplace from the 'rfRes' tensor!", 
          op::ToString(broadcastShape).GetString(), op::ToString(rfRes->GetViewShape()).GetString());
      return nullptr;
  }

  // 校验输出tensor的dtype和rfRes tensor一致
  bool isMixDataType = (self->GetDataType() == DataType::DT_FLOAT16 && rfRes->GetDataType() == DataType::DT_FLOAT) ||
                        (self->GetDataType() == DataType::DT_FLOAT && rfRes->GetDataType() == DataType::DT_FLOAT16) ||
                        (self->GetDataType() == DataType::DT_BF16 && rfRes->GetDataType() == DataType::DT_FLOAT) ||
                        (self->GetDataType() == DataType::DT_FLOAT && rfRes->GetDataType() == DataType::DT_BF16);
  if (isMixDataType && (rfRes->GetDataType() == DataType::DT_FLOAT16 || rfRes->GetDataType() == DataType::DT_BF16)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out dtype DataType::DT_FLOAT not equal to rfRes dtype [%s], do no support inplace from the 'rfRes' tensor!", 
          op::ToString(rfRes->GetDataType()).GetString());
      return nullptr;
  }

  auto mulOut = const_cast<aclTensor*>(rfRes);

  if (isMixDataType || (IsAiCoreSupport(self) && IsAiCoreSupport(rfRes)) || IsDoubleSupport(self, rfRes)) {
    return MulAiCore(self, rfRes, mulOut, executor);
  }

  return MulAiCpu(self, rfRes, mulOut, executor);
}

} // namespace l0op
