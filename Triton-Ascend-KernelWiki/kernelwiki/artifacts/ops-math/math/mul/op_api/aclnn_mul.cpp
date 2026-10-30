/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include "aclnn_mul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/logical_and/op_api/logical_and.h"
#include "mul.h"
#include "math/muls/op_api/muls.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"


using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Mul 算子的完整计算流程如下:
 *         self                       other
 *           |                          |
 *           \                          /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *           \                          /
 *      Cast(workspace_1)      Cast(workspace_3)
 *                    \          /
 *                  Mul(workspace_4)
 *                          |
 *                  Cast(workspace_5)
 *                          |
 *                       ViewCopy
 *                          |
 *                        result
 */

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_DOUBLE, DataType::DT_INT8,
  DataType::DT_UINT8, DataType::DT_INT16, DataType::DT_INT64, DataType::DT_BOOL, DataType::DT_COMPLEX128,
  DataType::DT_COMPLEX64};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
  DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_DOUBLE, DataType::DT_INT8,
  DataType::DT_UINT8, DataType::DT_INT16, DataType::DT_INT64, DataType::DT_BOOL, DataType::DT_COMPLEX128,
  DataType::DT_COMPLEX64, DataType::DT_BF16};

static op::DataType InnerTypeToComplexType(const op::DataType input) {
  switch (input) {
    case op::DataType::DT_BF16:
      // BFloat16 has range equivalent to Float,
      // so we map it to ComplexFloat.
      return op::DataType::DT_COMPLEX64;
    case op::DataType::DT_FLOAT16:
      return op::DataType::DT_COMPLEX32;
    case op::DataType::DT_FLOAT:
      return op::DataType::DT_COMPLEX64;
    case op::DataType::DT_DOUBLE:
      return op::DataType::DT_COMPLEX128;
    case op::DataType::DT_COMPLEX32:
      return op::DataType::DT_COMPLEX32;
    case op::DataType::DT_COMPLEX64:
      return op::DataType::DT_COMPLEX64;
    case op::DataType::DT_COMPLEX128:
      return op::DataType::DT_COMPLEX128;
    default:
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unknown Complex ScalarType for [%s]", ToString(input).GetString());
      return op::DataType::DT_UNDEFINED;
  }
}

static op::DataType CombineCategoriesWithComplex(const op::DataType higher, const op::DataType lower) {
  if(IsComplexType(higher)) {
    return higher;
  } else if (IsComplexType(lower)) {
    // preserve value type of higher if it is floating type.
    if (IsFloatingType(higher)) {
      return InnerTypeToComplexType(higher);
    }
    // in case of integral input
    // lower complex takes precedence.
    return lower;
  } else if (IsFloatingType(higher)) {
    return higher;
  }
  if (higher == op::DataType::DT_BOOL || IsFloatingType(lower)) {
    return op::PromoteType(higher, lower);
  }
  if (higher != op::DataType::DT_UNDEFINED) {
    return higher;
  }
  return lower;
}

static op::DataType GetScalarDefaultDtype(const op::DataType input) {
  if (IsComplexType(input)) {
    return op::DataType::DT_COMPLEX64;
  } else if (IsFloatingType(input)) {
    return op::DataType::DT_FLOAT;
  }
  return input;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

inline static bool isFloatType(const DataType type) {
  return type == DataType::DT_DOUBLE || type == DataType::DT_FLOAT ||
         type == DataType::DT_FLOAT16 || type == DataType::DT_BF16;
}

inline static bool CheckMulsNotNull(const aclTensor *self, const aclScalar *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

inline static bool CheckMulNotNull(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(other, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

inline static bool CheckInplaceMulsNotNull(const aclTensor *selfRef, const aclScalar *other) {
  OP_CHECK_NULL(selfRef, return false);
  OP_CHECK_NULL(other, return false);
  return true;
}

inline static bool CheckInplaceMulNotNull(const aclTensor *selfRef, const aclTensor *other) {
  OP_CHECK_NULL(selfRef, return false);
  OP_CHECK_NULL(other, return false);
  return true;
}

inline static bool CheckMulsDtype(const aclTensor *self, const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  return true;
}

inline static bool CheckMulDtype(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
  return true;
}

inline static bool CheckInplaceMulDtype(const aclTensor *selfRef, const aclTensor *other) {
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, supportList, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);
  return true;
}

static inline float GetCastedFloat(const op::DataType tensorDtype, const aclScalar* scalar) {
  float castedRes = 0;
  switch (tensorDtype) {
    case DataType::DT_FLOAT16:
        castedRes = static_cast<float>(scalar->ToFp16());
        break;
    case DataType::DT_BF16:
        castedRes = static_cast<float>(scalar->ToBf16());
        break;
    default:
        castedRes = scalar->ToFloat();
        break;
  }
  return castedRes;
}

static inline bool IsFloatEqual(float a, float b)
{
  return std::abs(a - b) <= std::numeric_limits<float>::epsilon();
}

static DataType InferTensorScalarDtype(const aclTensor* self, const aclScalar* other, const aclTensor* out) {
  if (IsRegBase()) {
    auto scalarDefaultDtype = GetScalarDefaultDtype(other->GetDataType());
    auto promoteType = CombineCategoriesWithComplex(self->GetDataType(), scalarDefaultDtype);
    if (promoteType == DataType::DT_FLOAT16 || promoteType == DataType::DT_BF16) {
      bool keepB16 = IsFloatEqual(GetCastedFloat(promoteType, other), other->ToFloat());
      promoteType = keepB16 ? promoteType : DataType::DT_FLOAT;
    }
    if (promoteType == DataType::DT_COMPLEX32) {
      promoteType = DataType::DT_COMPLEX64;
    }
    return promoteType;
  }

  if (IsComplexType(self->GetDataType()) || IsComplexType(other->GetDataType())) {
    return PromoteType(self->GetDataType(), other->GetDataType());
  }
  if (isFloatType(self->GetDataType())) {
    // BF16场景使用FP32需转换到FP32进行计算
    return self->GetDataType() != DataType::DT_BF16 ? self->GetDataType() : DataType::DT_FLOAT;
  }
  if ((self->GetDataType() == DataType::DT_BOOL && other->GetDataType() == DataType::DT_DOUBLE) ||
      (other->GetDataType() == DataType::DT_DOUBLE && out->GetDataType() == DataType::DT_FLOAT)) {
    // 针对该场景使用FLOAT计算，避免影响性能，与原有场景保持一致
    return DataType::DT_FLOAT;
  }
  if (isFloatType(other->GetDataType()) || self->GetDataType() == DataType::DT_BOOL) {
    return PromoteType(self->GetDataType(), other->GetDataType());
  }
  return self->GetDataType();
}

inline static bool CheckMulsPromoteDtype(const aclTensor* self, const aclScalar* other, const aclTensor* out) {
  if (!IsRegBase()) {
    return true;
  }
  auto inferDtype = InferTensorScalarDtype(self, other, out);
  if (inferDtype == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
    return false;
  }
  const auto& supportList = GetDtypeSupportList();
  if (!CheckType(inferDtype, supportList)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "input dtype %s and %s after promote is %s, which should be in dtype support list %s.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString(),
            op::ToString(inferDtype).GetString(), op::ToString(supportList).GetString());
    return false;
  }

  OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), inferDtype, return false);
  OP_CHECK_RESULT_DTYPE_CAST_FAILED(other->GetDataType(), inferDtype, return false);

  OP_CHECK_RESULT_DTYPE_CAST_FAILED(inferDtype, out->GetDataType(), return false);
  return true;
}

inline static bool CheckMulPromoteType(const aclTensor *self, const aclTensor *other, const aclTensor* out) {
  // 检查self和other能否做数据类型推导
  DataType promoteType = PromoteType(self->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnMul self dtype [%s] and other dtype [%s] to be promotable but check failed.",
            ToString(self->GetDataType()).GetString(), ToString(other->GetDataType()).GetString());
    return false;
  }
  if (!IsRegBase()) {
    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
  }
  return true;
}

inline static bool CheckInplaceMulPromoteType(const aclTensor *selfRef, const aclTensor *other) {
  // 检查selfRef和other能否做数据类型推导
  DataType promoteType = PromoteType(selfRef->GetDataType(), other->GetDataType());
  if (promoteType == DataType::DT_UNDEFINED) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expected aclnnInplaceMul selfRef dtype [%s] and other dtype [%s] to be promotable but check failed.",
            ToString(selfRef->GetDataType()).GetString(), ToString(other->GetDataType()).GetString());
    return false;
  }
  if (!IsRegBase()) {
    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, selfRef->GetDataType(), return false);
  }
  return true;
}

inline static bool CheckMulShape(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  Shape dstShape;
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, dstShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, dstShape, return false);
  return true;
}

inline static bool CheckInplaceMulShape(const aclTensor *selfRef, const aclTensor *other) {
  Shape dstShape;
  OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfRef, other, dstShape, return false);
  OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(selfRef, dstShape, return false);
  return true;
}

inline static aclnnStatus CheckMulsParams(const aclTensor *self, const aclScalar *other, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckMulsNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入与输出的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckMulsDtype(self, out), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查self和other的数据类型能否promote以及转换为输出的数据类型
  CHECK_RET(CheckMulsPromoteDtype(self, other, out), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查输入输出shape间关系
  OP_CHECK_SHAPE_NOT_EQUAL(self, out, return ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckMulParams(const aclTensor *self, const aclTensor *other, const aclTensor *out) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckMulNotNull(self, other, out), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入与输出的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckMulDtype(self, other, out), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckMulPromoteType(self, other, out), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查输入输出shape间关系
  CHECK_RET(CheckMulShape(self, other, out), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckInplaceMulsParams(const aclTensor *selfRef, const aclScalar *other) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckInplaceMulsNotNull(selfRef, other), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  const auto& supportList = GetDtypeSupportList();
  OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, supportList, return ACLNN_ERR_PARAM_INVALID);
  // 3. 检查self和other的数据类型能否promote以及转换为输出的数据类型
  CHECK_RET(CheckMulsPromoteDtype(selfRef, other, selfRef), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckInplaceMulParams(const aclTensor *selfRef, const aclTensor *other) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckInplaceMulNotNull(selfRef, other), ACLNN_ERR_PARAM_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckInplaceMulDtype(selfRef, other), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查输入dtype是否支持类型推导
  CHECK_RET(CheckInplaceMulPromoteType(selfRef, other), ACLNN_ERR_PARAM_INVALID);
  // 4. 检查输入间shape关系
  CHECK_RET(CheckInplaceMulShape(selfRef, other), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

inline static bool IsMulMixDtypeSupport(const aclTensor *self, const aclTensor *other) {
  return (self->GetDataType() == DataType::DT_FLOAT16 && other->GetDataType() == DataType::DT_FLOAT) ||
         (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_FLOAT16) ||
         (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_FLOAT) ||
         (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_BF16);
}

static void MulsCheckFormat(const aclTensor* self){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnMuls only support format ND.");
  }
}

aclnnStatus aclnnMulsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,
                                      uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnMuls, DFX_IN(self, other), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckMulsParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  MulsCheckFormat(self);

  // 空tensor处理
  if (self->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 获取tensor与scalar推导后的dtype
  auto inferDtype = InferTensorScalarDtype(self, other, out);

  auto selfWithStride = uniqueExecutor.get()->CreateView(
      self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
  CHECK_RET(selfWithStride != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* resTensor = nullptr;
  bool canUseMuls = IsRegBase() && 
                    (self->GetDataType() == DataType::DT_BF16 ||
                     self->GetDataType() == DataType::DT_FLOAT16) &&
                    GetScalarDefaultDtype(other->GetDataType()) == DataType::DT_FLOAT;
  canUseMuls = canUseMuls || (!IsRegBase() &&
                              self->GetDataType() == DataType::DT_BF16 &&
                              other->GetDataType() == DataType::DT_DOUBLE);
  if (canUseMuls) {
    // BF16的tensor与DOUBLE类型的scalar需调用Muls，确保不降低精度的同时输出BF16
    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    resTensor = l0op::Muls(selfContiguous, other->ToFloat(), uniqueExecutor.get());
  } else {
    // 将other转换为aclTensor
    auto otherTensor = uniqueExecutor.get()->ConvertToTensor(other, inferDtype);
    if (self->GetDataType() == inferDtype && l0op::IsMulSupportNonContiguous(self, otherTensor)) {
      resTensor = l0op::Mul(selfWithStride, otherTensor, uniqueExecutor.get());
    } else {
      // 固定写法，将输入self转换成连续的tensor
      auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
      CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // 将输入self的数据类型转换成推导后的dtype
      auto selfCast = l0op::Cast(selfContiguous, inferDtype, uniqueExecutor.get());
      CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      // 调用主体计算函数
      resTensor = l0op::Mul(selfCast, otherTensor, uniqueExecutor.get());
    }
  }
  CHECK_RET(resTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(resTensor, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

static void MulCheckFormat(const aclTensor* self, const aclTensor* other){
  ge::Format selfStorageFormat = self->GetStorageFormat();
  ge::Format otherStorageFormat = other->GetStorageFormat();
  if (selfStorageFormat != ge::Format::FORMAT_ND || otherStorageFormat != ge::Format::FORMAT_ND){
    OP_LOGW("aclnnMul only support format ND.");
  }
}

/* 非连续 Mul kernel 的退化场景判定。
 * 仅当某个输入同时满足以下四点时，非连续 Mul kernel 的访存代价才显著高于"转连续 + dense Mul"，
 * 此时转连续才稳定收益（阈值经 Ascend950 上板扫描标定：命中即不慢于非连续，各 dtype 无回退）：
 *   (1) 外层广播倍数大——某维 size==1 而对端 size>1，该输入被重复读 K 次（K >= 阈值）；
 *   (2) payload 不小——元素数 >= 阈值（过小则 launch-bound，转连续多一个 kernel 反而慢）；
 *   (3) 内层连续块极短——最内 stride==1 连续块 < 阈值（无可向量化的连续搬运段）；
 *   (4) gather 跨度大——连续块断裂处 stride >= 阈值（真稀疏散读；过小则近连续，非连续本就快）。
 * 缺任一条非连续 kernel 更优，保持原路径。注：对 self/other 各判一次取 OR（单边），
 * 不覆盖"两端各自不达标但联合广播"的冷门场景。
 */
constexpr int64_t MUL_NC_BCAST_FACTOR_MIN = 320;  // 广播重复次数下限（上板标定）
constexpr int64_t MUL_NC_PAYLOAD_MIN = 512;       // payload 元素数下限（<512 launch-bound；按元素覆盖各 dtype）
constexpr int64_t MUL_NC_CONTIG_RUN_MIN = 32;     // 最内连续块达到该长度即视为可向量化，不强转
constexpr int64_t MUL_NC_GATHER_STRIDE_MIN = 24;  // gather 断裂处 stride 下限（小于此近连续，非连续更快）
constexpr size_t MUL_NC_MAX_DIM = 8;              // 张量最大维数（Ascend tensor 上限），用于定容小数组

// 一次遍历得到「最内连续块元素数 run」与「连续性断裂处 stride（gather 跨度）」。
// 收集 size>1 维并按 stride 升序排列（与维序无关，可正确处理转置）；run 达阈值即饱和封顶，
// 避免对超大 shape 做无界乘法溢出。gather 为 0 表示全连续/连续块够长（视为非稀疏）。
struct MulRunGather {
  int64_t run;
  int64_t gather;
};
static MulRunGather MulInnerRunAndGather(const aclTensor* x) {
  const auto shape = x->GetViewShape();
  const size_t n = shape.GetDimNum();
  std::array<int64_t, MUL_NC_MAX_DIM> strides{};
  std::array<int64_t, MUL_NC_MAX_DIM> sizes{};
  size_t m = 0;
  for (size_t d = 0; d < n && m < MUL_NC_MAX_DIM; ++d) {
    const int64_t sz = shape.GetDim(d);
    const int64_t st = x->GetViewStrides()[d];
    if (sz <= 1 || st == 0) { continue; }  // size-1 与 stride-0(广播)维不参与连续块/gather 计算
    strides[m] = st;
    sizes[m] = sz;
    ++m;
  }
  for (size_t i = 1; i < m; ++i) {
    const int64_t ks = strides[i];
    const int64_t kz = sizes[i];
    size_t j = i;
    while (j > 0 && strides[j - 1] > ks) { strides[j] = strides[j - 1]; sizes[j] = sizes[j - 1]; --j; }
    strides[j] = ks;
    sizes[j] = kz;
  }
  int64_t run = 1;  // run is also the expected stride of the next contiguous dim
  for (size_t i = 0; i < m; ++i) {
    if (strides[i] != run) { return {run, strides[i]}; }  // contiguity breaks here; this stride is the gather span
    if (run >= MUL_NC_CONTIG_RUN_MIN || sizes[i] >= MUL_NC_CONTIG_RUN_MIN) { return {MUL_NC_CONTIG_RUN_MIN, 0}; }
    run *= sizes[i];  // guarded above: run<MIN && size<MIN here => product < MIN^2, no overflow
  }
  return {run, 0};
}

// x 相对 y 的广播(重复)倍数：x 某维 size==1 且 y 对齐维 size>1
static int64_t MulBroadcastFactor(const aclTensor* x, const aclTensor* y) {
  const auto xs = x->GetViewShape();
  const auto ys = y->GetViewShape();
  const size_t xn = xs.GetDimNum();
  const size_t yn = ys.GetDimNum();
  const size_t n = xn > yn ? xn : yn;
  int64_t factor = 1;
  for (size_t i = 0; i < n; ++i) {
    const int64_t xd = (i < xn) ? xs.GetDim(xn - 1 - i) : 1;
    const int64_t yd = (i < yn) ? ys.GetDim(yn - 1 - i) : 1;
    if (xd == 1 && yd > 1) {
      if (factor >= MUL_NC_BCAST_FACTOR_MIN || yd >= MUL_NC_BCAST_FACTOR_MIN) { return MUL_NC_BCAST_FACTOR_MIN; }
      factor *= yd;  // guarded above: factor<MIN && yd<MIN here => product < MIN^2, no overflow
    }
  }
  return factor;
}

static int64_t MulViewNumel(const aclTensor* x) {
  const auto shape = x->GetViewShape();
  int64_t numel = 1;
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    const int64_t d = shape.GetDim(i);
    if (numel >= MUL_NC_PAYLOAD_MIN || d >= MUL_NC_PAYLOAD_MIN) { return MUL_NC_PAYLOAD_MIN; }
    numel *= d;  // guarded above: numel<MIN && d<MIN here => product < MIN^2, no overflow
  }
  return numel;
}


// x 是否命中"稀疏广播"退化模式（相对对端 y）：广播倍数大 + payload 大 + 内层连续块短 + gather 跨度大
static bool MulIsSparseBroadcastOperand(const aclTensor* x, const aclTensor* y) {
  const MulRunGather rg = MulInnerRunAndGather(x);
  return MulBroadcastFactor(x, y) >= MUL_NC_BCAST_FACTOR_MIN &&
         MulViewNumel(x) >= MUL_NC_PAYLOAD_MIN &&
         rg.run < MUL_NC_CONTIG_RUN_MIN &&
         rg.gather >= MUL_NC_GATHER_STRIDE_MIN;
}

// 任一输入命中稀疏广播退化模式时，强制走"转连续 + l0op::Mul"而非非连续 Mul kernel
static bool MulPreferContiguous(const aclTensor* self, const aclTensor* other) {
  const bool prefer = MulIsSparseBroadcastOperand(self, other) ||
                      MulIsSparseBroadcastOperand(other, self);
  if (prefer) {
    OP_LOGI("aclnnMul: sparse-broadcast operand detected, route to Contiguous + Mul instead of non-contiguous kernel.");
  }
  return prefer;
}

aclnnStatus aclnnMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                     uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnMul, DFX_IN(self, other), DFX_OUT(out));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckMulParams(self, other, out);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  MulCheckFormat(self, other);

  // 空tensor处理
  if (self->IsEmpty() || other->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 判断输入是否符合kernel支持的混合输入类型
  bool isMixDataType = IsMulMixDtypeSupport(self, other);

  // 命中稀疏广播退化模式时，非连续 Mul kernel 反而更慢，回退到转连续路径。
  // preferContiguous 同时作用于下方两个非连续门：mix-dtype 分支经 isSupportNonContiguous，
  // normal 分支则在其条件里直接 && !preferContiguous（两处需保持一致）。
  bool preferContiguous = MulPreferContiguous(self, other);
  bool isSupportNonContiguous = IsRegBase() && !preferContiguous;
  auto selfWithStride = uniqueExecutor.get()->CreateView(
      self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
  CHECK_RET(selfWithStride != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto otherWithStride = uniqueExecutor.get()->CreateView(
      other, other->GetViewShape(), other->GetStorageShape(), other->GetViewStrides(), other->GetViewOffset());
  CHECK_RET(otherWithStride != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* resTensor = nullptr;
  if (isMixDataType) {
    // 无需调用Cast对输入进行隐式数据类型转换
    if (isSupportNonContiguous) {
      resTensor = l0op::Mul(selfWithStride, otherWithStride, uniqueExecutor.get());
    } else {
      // 固定写法，将输入self转换成连续的tensor
      auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
      CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // 固定写法，将输入other转换成连续的tensor
      auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
      CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
      resTensor = l0op::Mul(selfContiguous, otherContiguous, uniqueExecutor.get());
    }
  } else {
    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());

    // 调用主体计算函数
    if (self->GetDataType() == promoteType && other->GetDataType() == promoteType &&
        l0op::IsMulSupportNonContiguous(self, other) && !preferContiguous) {
      resTensor = l0op::Mul(selfWithStride, otherWithStride, uniqueExecutor.get());
    } else {
      // 固定写法，将输入self转换成连续的tensor
      auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
      CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // 固定写法，将输入other转换成连续的tensor
      auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
      CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // 将输入self的数据类型转换成隐式数据类型
      auto selfCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
      CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

      // 将输入other的数据类型转换成隐式数据类型
      auto otherCast = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
      CHECK_RET(otherCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

      resTensor = l0op::Mul(selfCast, otherCast, uniqueExecutor.get());
    }
  }
  CHECK_RET(resTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(resTensor, out->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMulsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize,
                                             aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceMuls, DFX_IN(selfRef, other), DFX_OUT(selfRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckInplaceMulsParams(selfRef, other);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空tensor处理
  if (selfRef->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 获取tensor与scalar推导后的dtype
  auto inferDtype = InferTensorScalarDtype(selfRef, other, selfRef);

  // 固定写法，将输入selfRef转换成连续的tensor
  auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
  CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* resTensor = nullptr;
  bool canUseMuls = IsRegBase() && 
                    (selfRefContiguous->GetDataType() == DataType::DT_BF16 ||
                     selfRefContiguous->GetDataType() == DataType::DT_FLOAT16) &&
                    GetScalarDefaultDtype(other->GetDataType()) == DataType::DT_FLOAT;
  canUseMuls = canUseMuls || (!IsRegBase() &&
                              selfRefContiguous->GetDataType() == DataType::DT_BF16 &&
                              other->GetDataType() == DataType::DT_DOUBLE);
  if (canUseMuls) {
    // BF16的tensor与DOUBLE类型的scalar需调用Muls，确保不降低精度的同时输出BF16
    resTensor = l0op::Muls(selfRefContiguous, other->ToFloat(), uniqueExecutor.get());
  } else {
    // 将输入selfRef的数据类型转换成推导后的dtype
    auto selfRefCast = l0op::Cast(selfRefContiguous, inferDtype, uniqueExecutor.get());
    CHECK_RET(selfRefCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将other转换为aclTensor
    auto otherTensor = uniqueExecutor.get()->ConvertToTensor(other, inferDtype);
    // 调用主体计算函数
    resTensor = l0op::Mul(selfRefCast, otherTensor, uniqueExecutor.get());
  }
  CHECK_RET(resTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出selfRef的数据类型
  auto castOut = l0op::Cast(resTensor, selfRef->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出selfRef上，selfRef可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMulGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize,
                                            aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnInplaceMul, DFX_IN(selfRef, other), DFX_OUT(selfRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckInplaceMulParams(selfRef, other);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  
  // 检查格式
  MulCheckFormat(selfRef, other);

  // 空tensor处理
  if (selfRef->IsEmpty() || other->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入selfRef转换成连续的tensor
  auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
  CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入other转换成连续的tensor
  auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
  CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 判断输入是否符合kernel支持的混合输入类型
  bool isMixDataType = IsMulMixDtypeSupport(selfRefContiguous, otherContiguous);

  const aclTensor* resTensor = nullptr;
  if (IsRegBase() && isMixDataType) {
    resTensor = l0op::Mul(selfRefContiguous, otherContiguous, uniqueExecutor.get());
  } else {
    // 获取selfRef和other的隐式转换数据类型
    auto promoteType = PromoteType(selfRef->GetDataType(), other->GetDataType());

    // 将输入selfRef的数据类型转换成隐式数据类型
    auto selfRefCast = l0op::Cast(selfRefContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfRefCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入other的数据类型转换成隐式数据类型
    auto otherCast = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(otherCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用主体计算函数
    resTensor = l0op::Mul(selfRefCast, otherCast, uniqueExecutor.get());
  }
  CHECK_RET(resTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果转换成输出selfRef的数据类型
  auto castOut = l0op::Cast(resTensor, selfRef->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出selfRef上，selfRef可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnMuls(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMuls);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnMul);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceMuls(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceMuls);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplaceMul);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif