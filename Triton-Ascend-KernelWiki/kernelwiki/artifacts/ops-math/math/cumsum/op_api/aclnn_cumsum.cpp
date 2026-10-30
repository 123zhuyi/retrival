/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_cumsum.h"
#include "cumsum.h"
#include "math/cumsum_cube/op_host/op_api/cumsum_cube.h"
#include "aclnn_kernels/cast.h"
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
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace{
static constexpr size_t MAX_DIM_LEN = 8;
static constexpr int64_t CUMSUM_CUBE_MIN_SUPPORT_BATCH = 12800;
static constexpr int64_t CUMSUM_CUBE_MIN_SUPPORT_DIM = 512;
static constexpr int64_t CUMSUM_CUBE_MAX_SUPPORT_SIZE = 50000000;

static inline bool CheckNotNull(const aclTensor *self, const aclTensor *out) {
  // self、out不能为空指针
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_DOUBLE, DataType::DT_UINT8,
    DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT64, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128
};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_DOUBLE, DataType::DT_UINT8,
    DataType::DT_INT8, DataType::DT_INT16, DataType::DT_INT64, DataType::DT_COMPLEX64, DataType::DT_COMPLEX128,
    DataType::DT_BF16
};

static const inline std::initializer_list<DataType>& GetSupportDtypeList(NpuArch npuArch)
{
    static const std::initializer_list<DataType> emptyDtypes = {};
    if (npuArch == NpuArch::DAV_2002 || npuArch == NpuArch::DAV_1001 ||
        npuArch == NpuArch::DAV_3002) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    } else if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return emptyDtypes;
    }
}

static inline bool CheckDtypeValidWithoutDtype(const aclTensor *self, const aclTensor *out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  const auto& DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(npuArch);
  if (DTYPE_SUPPORT_LIST_CURRENT.size() == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
    return false;
  }

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST_CURRENT, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_CURRENT, return false);

  // 检查self和out的数据类型是否一致
  OP_CHECK_DTYPE_NOT_SAME(self, out, return false);

  return true;
}


static inline bool CheckDtypeValid(aclDataType dtype, const aclTensor *out) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
  const auto& DTYPE_SUPPORT_LIST_CURRENT = GetSupportDtypeList(npuArch);
  if (DTYPE_SUPPORT_LIST_CURRENT.size() == 0) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
    return false;
  }

  // 检查out的数据类型是否在支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_CURRENT, return false);

  // 检查指定dtype和out的数据类型是否一致
  DataType dtypeNew = ToOpDataType(dtype);
  OP_CHECK_DTYPE_NOT_MATCH(out, dtypeNew, return false);

  return true;
}

static inline bool CheckDim(const aclTensor *self, int64_t dim) {
  auto selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  // 当输入张量是0维时，当作1维处理
  if (selfDimNum == 0) {
    selfDimNum = 1;
  }
  // 检查指定dim是否在self的维度范围内
  if (dim >= selfDimNum || dim < -selfDimNum) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected dim to be in range of [%ld, %ld], but got %ld.", -selfDimNum,
            selfDimNum - 1, dim);
    return false;
  }

  return true;
}

static inline bool CheckShape(const aclTensor *self, const aclTensor *out) {
  // self和out的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);

  // self的数据维度不能超过8
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);

  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t dim, aclDataType dtype, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在支持范围内，且数据类型是否一致
  CHECK_RET(CheckDtypeValid(dtype, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out的shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查指定dim是否在self的维度范围内
  CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsWithoutDtype(const aclTensor *self, int64_t dim, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在支持范围内，且数据类型是否一致
  CHECK_RET(CheckDtypeValidWithoutDtype(self, out), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self和out的shape是否一致
  CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查指定dim是否在self的维度范围内
  CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static inline bool CheckShapeIsSupport(const aclTensor *self, int64_t dim) {
  auto selfDimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
  // 当输入张量是0维时，不支持
  if (selfDimNum == 0) {
    return false;
  }
  if (dim < 0) {
    dim = selfDimNum + dim;
  }
  int64_t batchNum = 1;
  int64_t channelNum = 1;
  for (int i = 0; i <= dim; ++i) {
    if(self->GetViewShape().GetDim(i) > CUMSUM_CUBE_MAX_SUPPORT_SIZE){
      return false;
    }
  }
  if (dim != 0) {
    for (int i = 0; i < dim; ++i) {
        batchNum *= self->GetViewShape().GetDim(i);
    }
  }
  if (dim != selfDimNum - 1) {
    return false;
  }
  channelNum = self->GetViewShape().GetDim(dim);
  if (batchNum >= CUMSUM_CUBE_MIN_SUPPORT_BATCH && channelNum >= CUMSUM_CUBE_MIN_SUPPORT_DIM) {
    return true;
  }
  return false;
}

static inline bool CheckCubeSupport(const aclTensor* out, int64_t dim) {
  // 获取芯片类型
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool isSupport = true;
  auto selfDimNum = static_cast<int64_t>(out->GetViewShape().GetDimNum());
  // 当输入张量是0维时，当作1维处理
  if (selfDimNum == 0) {
    selfDimNum = 1;
  }
  if(socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93){
    isSupport = false;
  }
  auto dtype = out->GetDataType();
  if(dtype != DataType::DT_FLOAT && dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16){
    isSupport = false;
  }
  if(!CheckShapeIsSupport(out, dim)) {
    isSupport = false;
  }
  return isSupport;
}
}; // namespace

aclnnStatus aclnnCumsumV2GetWorkspaceSize(const aclTensor *self, int64_t dim,
                                          bool exclusive, bool reverse, aclTensor *out,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnCumsumV2, DFX_IN(self, dim, exclusive, reverse), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParamsWithoutDtype(self, dim, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果self是空tensor，则out也是空tensor，直接返回
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成和out一致
  auto castSelf = l0op::Cast(contiguousSelf, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 由dim初始化一个张量，作为LO的API入参
  const aclTensor *dimTensor = nullptr;
  if (dim == 0 || dim > INT32_MAX) {
    dimTensor = (uniqueExecutor.get())->ConvertToTensor(&dim, 1, DataType::DT_INT64);
  } else {
    dimTensor = (uniqueExecutor.get())->ConvertToTensor(&dim, 1, DataType::DT_INT32);
  }
  CHECK_RET(dimTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Cumsum算子kernel
  auto cumsumOut = l0op::Cumsum(castSelf, dimTensor, exclusive, reverse, uniqueExecutor.get());
  CHECK_RET(cumsumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(cumsumOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCumsumV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnCumsumV2);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype, aclTensor *out,
                                        uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnCumsum, DFX_IN(self, dim, dtype), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(self, dim, dtype, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 如果self是空tensor，则out也是空tensor，直接返回
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的Tensor
  auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将输入self的数据类型转换成和out一致
  auto castSelf = l0op::Cast(contiguousSelf, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if(CheckCubeSupport(out, dim)){
    auto cumsumOut = l0op::CumsumCube(castSelf, dim, uniqueExecutor.get());
    CHECK_RET(cumsumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(cumsumOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR); 
    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;   
  }

  // 由dim初始化一个张量，作为LO的API入参
  const aclTensor *dimTensor = nullptr;
  if (dim == 0 || dim > INT32_MAX) {
    dimTensor = (uniqueExecutor.get())->ConvertToTensor(&dim, 1, DataType::DT_INT64);
  } else {
    dimTensor = (uniqueExecutor.get())->ConvertToTensor(&dim, 1, DataType::DT_INT32);
  }
  CHECK_RET(dimTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用Cumsum算子kernel
  auto cumsumOut = l0op::Cumsum(castSelf, dimTensor, uniqueExecutor.get());
  CHECK_RET(cumsumOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(cumsumOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnCumsum);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif
