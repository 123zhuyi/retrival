/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "precision_compare_util.h"

#include "op_api/aclnn_check.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
static const size_t DIM_SUPPORT_MAX = 8;
static const std::initializer_list<op::DataType> INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT32};

static inline bool CheckNotNull(const aclTensor *golden, const aclTensor *realdata, const aclTensor *out) {
  OP_CHECK_NULL(golden, return false);
  OP_CHECK_NULL(realdata, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static inline bool CheckDtypeValid(const aclTensor *golden, const aclTensor *realdata, const aclTensor *out) {
  OP_CHECK_DTYPE_NOT_SAME(golden, realdata, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(golden, INPUT_DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool CheckShape(const aclTensor *golden, const aclTensor *realdata, const aclTensor *out) {
  OP_CHECK_SHAPE_NOT_EQUAL(golden, realdata, return false);
  OP_CHECK_MAX_DIM(golden, DIM_SUPPORT_MAX, return false);
  size_t outDimNum = out->GetViewShape().GetDimNum();
  if (outDimNum != 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out must be scalar, but rank is %zu", outDimNum);
    return false;
  }
  return true;
}

aclnnStatus PrecisionCompareCheckParams(const aclTensor *golden, const aclTensor *realdata, const aclTensor *out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93 && !IsRegBase()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The function is unsupported by the current SOC version [%s] ",
            op::ToString(socVersion).GetString());
    return ACLNN_ERR_PARAM_INVALID;
  }
  CHECK_RET(CheckNotNull(golden, realdata, out), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(golden, realdata, out), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(golden, realdata, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif