/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_div.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "math/floor_div/op_api/floordiv.h"
#include "math/real_div/op_api/realdiv.h"
#include "math/trunc/op_api/trunc.h"
#include "math/muls/op_api/muls.h"
#include "math/truncate_div/op_api/truncate_div.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"
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
#include "conversion/broadcast_to/op_api/broadcast_to.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

op::DataType PromoteIntegerInputsToFloat(const op::DataType input)
{
    if (IsIntegralType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

// fp16/bf16 提升到 float 做计算(对齐 CUDA acc_type),其余 dtype 原样返回。
static inline op::DataType PromoteLowPrecToFloat(const op::DataType input)
{
    if (input == op::DataType::DT_FLOAT16 || input == op::DataType::DT_BF16) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

static op::DataType InnerTypeToComplexType(const op::DataType input)
{
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

static op::DataType CombineCategoriesWithComplex(const op::DataType higher, const op::DataType lower)
{
    if (IsComplexType(higher)) {
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

static op::DataType GetScalarDefaultDtype(const op::DataType input)
{
    if (IsComplexType(input)) {
        return op::DataType::DT_COMPLEX64;
    } else if (IsFloatingType(input)) {
        return op::DataType::DT_FLOAT;
    }
    return input;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,     op::DataType::DT_INT64,     op::DataType::DT_INT32,
    op::DataType::DT_INT16,   op::DataType::DT_INT8,      op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT64,     op::DataType::DT_INT32,
    op::DataType::DT_INT16,   op::DataType::DT_INT8,  op::DataType::DT_UINT8,     op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,    op::DataType::DT_BF16,  op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const int MODE_REAL_DIV = 0;
static const int MODE_TRUNC_DIV = 1;
static const int MODE_FLOOR_DIV = 2;

// TruncateDiv 支持的类型组合映射表（不需要类型提升）
static const std::initializer_list<std::pair<op::DataType, op::DataType>> TRUNC_DTYPE_MAPPING = {
    {op::DataType::DT_BF16, op::DataType::DT_BF16},     {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT}, {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT},   {op::DataType::DT_FLOAT, op::DataType::DT_INT32},
    {op::DataType::DT_INT32, op::DataType::DT_INT32},   {op::DataType::DT_INT32, op::DataType::DT_FLOAT},
    {op::DataType::DT_UINT8, op::DataType::DT_UINT8},   {op::DataType::DT_INT8, op::DataType::DT_INT8},
    {op::DataType::DT_INT64, op::DataType::DT_INT64},   {op::DataType::DT_INT16, op::DataType::DT_INT16}};

static bool isInTruncDtypeMapping(const op::DataType selfDtype, const op::DataType otherDtype)
{
    for (const auto& pair : TRUNC_DTYPE_MAPPING) {
        if (pair.first == selfDtype && pair.second == otherDtype) {
            return true;
        }
    }
    return false;
}

static const std::initializer_list<std::pair<op::DataType, op::DataType>> AllowedMixDtypePairs = {
    {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT16}, {op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT},
    {op::DataType::DT_FLOAT16, op::DataType::DT_BF16},    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16},
    {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT},     {op::DataType::DT_FLOAT, op::DataType::DT_BF16},
    {op::DataType::DT_BF16, op::DataType::DT_FLOAT16},    {op::DataType::DT_BF16, op::DataType::DT_FLOAT},
    {op::DataType::DT_BF16, op::DataType::DT_BF16}};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (npuArch == NpuArch::DAV_2201 || IsRegBase(npuArch)) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(self, return false);
    return true;
}

static inline op::DataType CompatibleInferDivDtype(const op::DataType selfDtype, const op::DataType otherDtype)
{
    // RealDiv算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(selfDtype, otherDtype);
    // 下沉PTA入口操作将入参类型转化成FLOAT进行后续处理，int32除外（仅910B/910C的L0算子已支持int32精度）
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    bool isInt32PrecisionSupported = socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
    promoteType = (IsFloatingType(promoteType) || IsComplexType(promoteType) || promoteType == op::DataType::DT_BOOL ||
                   (selfDtype == op::DataType::DT_INT32 && otherDtype == op::DataType::DT_INT32 &&
                    isInt32PrecisionSupported)) ?
                      promoteType :
                      op::DataType::DT_FLOAT;
    return promoteType;
}

static inline aclnnStatus CheckDivModComplexDtype(const op::DataType promoteType, const int mode)
{
    if ((mode == MODE_TRUNC_DIV || mode == MODE_FLOOR_DIV) &&
        (promoteType == op::DataType::DT_COMPLEX128 || promoteType == op::DataType::DT_COMPLEX64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promoteType do not support DT_COMPLEX128 or DT_COMPLEX64.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static inline op::DataType InferDivModeDtype(const op::DataType selfDtype, const op::DataType otherDtype,
                                             const int mode)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto promoteType = op::PromoteType(selfDtype, otherDtype);
    // 下沉PTA入口操作将入参类型转化成FLOAT进行后续处理
    if (mode == MODE_REAL_DIV && promoteType != op::DataType::DT_INT32 && promoteType != op::DataType::DT_BOOL) {
        // IterateBase 配置特殊处理
        promoteType = PromoteIntegerInputsToFloat(promoteType);
    }
    if (mode == MODE_TRUNC_DIV && promoteType == DataType::DT_DOUBLE && !IsRegBase(npuArch)) {
        promoteType = DataType::DT_FLOAT;
    }
    return promoteType;
}

static inline op::DataType CompatibleInferDivsDtype(const op::DataType selfDtype, const op::DataType otherDtype)
{
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    bool isInt32PrecisionSupported = socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
    auto promoteType = (IsFloatingType(selfDtype) || IsComplexType(selfDtype) ||
                        (selfDtype == op::DataType::DT_INT32 && otherDtype == op::DataType::DT_INT32 &&
                         isInt32PrecisionSupported)) ?
                           selfDtype :
                           op::DataType::DT_FLOAT;
    promoteType = (selfDtype == op::DataType::DT_BOOL && otherDtype == op::DataType::DT_BOOL) ? selfDtype : promoteType;
    promoteType = (IsComplexType(otherDtype)) ? op::PromoteType(promoteType, otherDtype) : promoteType;
    return promoteType;
}

static aclnnStatus CompatibleInferDivModeDtype(const op::DataType selfDtype, const op::DataType otherDtype,
                                               const int mode, op::DataType& promoteType)
{
    promoteType = op::PromoteType(selfDtype, otherDtype);
    if ((mode == MODE_TRUNC_DIV || mode == MODE_FLOOR_DIV) &&
        (promoteType == op::DataType::DT_COMPLEX128 || promoteType == op::DataType::DT_COMPLEX64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promoteType do not support DT_COMPLEX128 or DT_COMPLEX64.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 根据mode分三种场景调用算子计算
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    bool isInt32PrecisionSupported = socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
    if (mode == MODE_FLOOR_DIV) {
        promoteType = (promoteType == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : promoteType;
    } else {
        promoteType = ((promoteType != op::DataType::DT_FLOAT) && (promoteType != op::DataType::DT_FLOAT16) &&
                       (promoteType != op::DataType::DT_COMPLEX64) && (promoteType != op::DataType::DT_COMPLEX128) &&
                       (promoteType != op::DataType::DT_BF16) && (promoteType != op::DataType::DT_BOOL) &&
                       !(selfDtype == op::DataType::DT_INT32 && otherDtype == op::DataType::DT_INT32 &&
                         isInt32PrecisionSupported)) ?
                          op::DataType::DT_FLOAT :
                          promoteType;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CompatibleInferDivsModeDtype(const op::DataType selfDtype, const op::DataType otherDtype,
                                                const int mode, op::DataType& promoteType)
{
    promoteType = op::PromoteType(selfDtype, otherDtype);
    if ((mode == MODE_TRUNC_DIV || mode == MODE_FLOOR_DIV) &&
        (promoteType == op::DataType::DT_COMPLEX128 || promoteType == op::DataType::DT_COMPLEX64)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promoteType do not support DT_COMPLEX128 or DT_COMPLEX64.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 根据mode分三种场景调用算子计算
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    bool isInt32PrecisionSupported = socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;
    if (mode == MODE_FLOOR_DIV) {
        promoteType = (promoteType == op::DataType::DT_BOOL) ? op::DataType::DT_FLOAT : promoteType;
    } else {
        promoteType = ((selfDtype != op::DataType::DT_FLOAT) && (selfDtype != op::DataType::DT_FLOAT16) &&
                       (selfDtype != op::DataType::DT_BF16) && (promoteType != op::DataType::DT_BOOL) &&
                       !(selfDtype == op::DataType::DT_INT32 && otherDtype == op::DataType::DT_INT32 &&
                         isInt32PrecisionSupported)) ?
                          op::DataType::DT_FLOAT :
                          selfDtype;
        promoteType = (IsComplexType(selfDtype) || IsComplexType(otherDtype)) ? op::PromoteType(selfDtype, otherDtype) :
                                                                                promoteType;
    }
    return ACLNN_SUCCESS;
}

static inline op::DataType InferDivsModeDtype(const op::DataType selfDtype, const op::DataType otherDtype,
                                              const int mode)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto scalarDefaultDtype = GetScalarDefaultDtype(otherDtype);
    auto promoteType = CombineCategoriesWithComplex(selfDtype, scalarDefaultDtype);
    if (mode == MODE_REAL_DIV && promoteType != op::DataType::DT_INT32 && promoteType != op::DataType::DT_BOOL) {
        // IterateBase 配置特殊处理
        promoteType = PromoteIntegerInputsToFloat(promoteType);
    }

    if (mode == MODE_TRUNC_DIV && promoteType == DataType::DT_DOUBLE && !IsRegBase(npuArch)) {
        promoteType = DataType::DT_FLOAT;
    }

    if (mode == MODE_TRUNC_DIV && (promoteType == DataType::DT_BF16 || promoteType == DataType::DT_FLOAT16) &&
        IsRegBase(npuArch)) {
        promoteType = DataType::DT_FLOAT;
    }

    if (promoteType == DataType::DT_COMPLEX32) {
        promoteType = DataType::DT_COMPLEX64;
    }

    return promoteType;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* other)
{
    auto supportList = GetDtypeSupportList();
    // 检查other的数据类型是否在div算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

    // 检查self的数据类型是否在div算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    return true;
}

static bool CheckDtypeValidScalar(const aclTensor* self, const aclScalar* other)
{
    auto supportList = GetDtypeSupportList();
    // 检查other的数据类型是否在div算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(other, supportList, return false);

    // 检查self的数据类型是否在div算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    return true;
}

static bool CheckPromoteType(const aclTensor* self, const aclTensor* other, const aclTensor* y, const int mode)
{
    // 检查self和other能否做数据类型推导
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto promoteType = (IsRegBase(npuArch)) ? InferDivModeDtype(self->GetDataType(), other->GetDataType(), mode) :
                                              op::PromoteType(self->GetDataType(), other->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
                op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
        return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    bool outDtypeToFloat = IsRegBase() && mode == MODE_REAL_DIV &&
                           (promoteType == op::DataType::DT_INT32 || promoteType == op::DataType::DT_BOOL);
    auto computeDtype = outDtypeToFloat ? op::DataType::DT_FLOAT : promoteType;
    if (GetCurrentPlatformInfo().GetCurNpuArch() >= NpuArch::DAV_3510 && mode == MODE_TRUNC_DIV) {
        return true;
    }
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(computeDtype, y->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* other, const aclTensor* y)
{
    // 输入维度不超过8维
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(other, MAX_SUPPORT_DIMS_NUMS, return false);

    // self和other需满足broadcast关系
    op::Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(y, broadcastShape, return false);
    return true;
}

static bool CheckMode(int mode)
{
    if (mode > MODE_FLOOR_DIV || mode < MODE_REAL_DIV) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mode should be between 0 and 2, but current is %d", mode);
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 格式不能是私有格式
    // 校验self格式
    if (IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    // 校验other格式
    if (IsPrivateFormat(other->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    // 校验out格式
    if (IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }

    return true;
}

static bool CheckFormatScalar(const aclTensor* self, const aclTensor* out)
{
    // 格式不能是私有格式
    // 校验out格式
    if (IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    // 校验self格式
    if (IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* other, const aclTensor* y, const int mode)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, other, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查双输入是否能broadcast
    CHECK_RET(CheckShape(self, other, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, other), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, other, y, mode), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据格式是否支持
    CHECK_RET(CheckFormat(self, other, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

inline static bool isDivsMixDtypeSupport(const aclTensor* self, const aclScalar* other)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        return false;
    }
    return (self->GetDataType() == DataType::DT_FLOAT16 && other->GetDataType() == DataType::DT_FLOAT) ||
           (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_FLOAT16) ||
           (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_FLOAT) ||
           (self->GetDataType() == DataType::DT_FLOAT && other->GetDataType() == DataType::DT_BF16) ||
           (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_DOUBLE) ||
           (self->GetDataType() == DataType::DT_FLOAT16 && other->GetDataType() == DataType::DT_BF16) ||
           (self->GetDataType() == DataType::DT_BF16 && other->GetDataType() == DataType::DT_FLOAT16);
}

inline static bool checkMixDtypeConditions(DataType selfDtype, DataType otherDtype)
{
    return std::find(AllowedMixDtypePairs.begin(), AllowedMixDtypePairs.end(),
                     std::pair<op::DataType, op::DataType>(selfDtype, otherDtype)) != AllowedMixDtypePairs.end();
}

inline static bool isMixDtypeScalarSupport(const aclTensor* self, const aclScalar* other)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        return false;
    }
    return checkMixDtypeConditions(self->GetDataType(), other->GetDataType());
}

inline static bool isMixDtypeTensorSupport(const aclTensor* self, const aclTensor* other)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        return false;
    }
    return checkMixDtypeConditions(self->GetDataType(), other->GetDataType());
}

static aclnnStatus HandleMixDataTypeDiv(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor,
                                        const aclTensor** divOpOut)
{
    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入other转换成连续的tensor
    auto otherContiguous = l0op::Contiguous(other, executor);
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *divOpOut = l0op::RealDiv(selfContiguous, otherContiguous, false, executor);
    CHECK_RET(*divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus HandleNotMixDataTypeDiv(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor,
                                           const aclTensor** divOpOut)
{
    // RealDiv算子需要对self和other两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    auto promoteType = (!IsRegBase(npuArch)) ?
                           CompatibleInferDivDtype(self->GetDataType(), other->GetDataType()) :
                           InferDivModeDtype(self->GetDataType(), other->GetDataType(), MODE_REAL_DIV);

    // 处理self输入
    const aclTensor* selfProcessed = nullptr;
    if (self->GetDataType() == promoteType && l0op::IsRealDivSupportNonContiguous(self)) {
        selfProcessed = executor->CreateView(self, self->GetViewShape(), self->GetStorageShape(),
                                             self->GetViewStrides(), self->GetViewOffset());
    } else {
        // 固定写法，将输入self转换成连续的tensor
        auto selfContiguous = l0op::Contiguous(self, executor);
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        selfProcessed = l0op::Cast(selfContiguous, promoteType, executor);
    }
    CHECK_RET(selfProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理other输入
    const aclTensor* otherProcessed = nullptr;
    if (other->GetDataType() == promoteType && l0op::IsRealDivSupportNonContiguous(other)) {
        otherProcessed = executor->CreateView(other, other->GetViewShape(), other->GetStorageShape(),
                                              other->GetViewStrides(), other->GetViewOffset());
    } else {
        // 固定写法，将输入other转换成连续的tensor
        auto otherContiguous = l0op::Contiguous(other, executor);
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 将输入other的数据类型转换成隐式数据类型，根据具体算子语义按需调用
        otherProcessed = l0op::Cast(otherContiguous, promoteType, executor);
    }
    CHECK_RET(otherProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子RealDiv进行计算
    *divOpOut = l0op::RealDiv(selfProcessed, otherProcessed, MODE_REAL_DIV, executor);
    CHECK_RET(*divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDivGetWorkspaceSize(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDiv, DFX_IN(self, other), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, other, out, MODE_REAL_DIV);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // div算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty() || other->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool isMixDataType = isMixDtypeTensorSupport(self, other);
    const aclTensor* divOpOut = nullptr;
    if (isMixDataType) {
        auto mixResult = HandleMixDataTypeDiv(self, other, uniqueExecutor.get(), &divOpOut);
        CHECK_RET(mixResult == ACLNN_SUCCESS, mixResult);
    } else {
        auto notMixResult = HandleNotMixDataTypeDiv(self, other, uniqueExecutor.get(), &divOpOut);
        CHECK_RET(notMixResult == ACLNN_SUCCESS, notMixResult);
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(divOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDiv);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static bool CheckNotNullScalar(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(other, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckPromoteTypeScalar(const aclTensor* self, const aclScalar* other, const aclTensor* y, const int mode)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (IsRegBase(npuArch)) {
        // 检查self和other能否做数据类型推导
        auto promoteType = InferDivsModeDtype(self->GetDataType(), other->GetDataType(), mode);
        if (promoteType == DataType::DT_UNDEFINED) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and other dtype %s can not promote dtype.",
                    op::ToString(self->GetDataType()).GetString(), op::ToString(other->GetDataType()).GetString());
            return false;
        }
        if (GetCurrentPlatformInfo().GetCurNpuArch() >= NpuArch::DAV_3510 && mode == MODE_TRUNC_DIV) {
            return true;
        }
        // 检查推导后的数据类型能否转换为输出的数据类型
        if (mode == MODE_REAL_DIV && (promoteType == op::DataType::DT_INT32 || promoteType == op::DataType::DT_BOOL)) {
            OP_CHECK_RESULT_DTYPE_CAST_FAILED(op::DataType::DT_FLOAT, y->GetDataType(), return false);
        } else {
            OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, y->GetDataType(), return false);
        }
        return true;
    }
    // 检查self的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), y->GetDataType(), return false);
    return true;
}

static bool CheckShapeScalar(const aclTensor* self, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    if (self->GetViewShape() != y->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
                op::ToString(self->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParamsScalar(const aclTensor* self, const aclScalar* other, const aclTensor* y, const int mode)
{
    CHECK_RET(CheckNotNullScalar(self, other, y), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckFormatScalar(self, y), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckDtypeValidScalar(self, other), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeScalar(self, y), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckPromoteTypeScalar(self, other, y, mode), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool CanUseMuls(const aclTensor* self, const aclScalar* other)
{
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (!IsRegBase(npuArch)) {
        return false;
    }
    if (self->GetDataType() != op::DataType::DT_FLOAT16 && self->GetDataType() != op::DataType::DT_BF16 &&
        self->GetDataType() != op::DataType::DT_FLOAT) {
        return false;
    }
    if (other->GetDataType() != op::DataType::DT_FLOAT16 && other->GetDataType() != op::DataType::DT_BF16 &&
        other->GetDataType() != op::DataType::DT_FLOAT && other->GetDataType() != op::DataType::DT_DOUBLE) {
        return false;
    }
    if (!op::IsContiguous(self) && other->GetDataType() == op::DataType::DT_DOUBLE) {
        return false;
    }

    return true;
}

aclnnStatus aclnnDivsGetWorkspaceSize(const aclTensor* self, const aclScalar* other, aclTensor* out,
                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDivs, DFX_IN(self, other), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 调用适配aclScalar参数检查
    auto ret = CheckParamsScalar(self, other, out, MODE_REAL_DIV);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    bool isSupportNonContiguous = IsRegBase(npuArch);
    auto socVersion = op::GetCurrentPlatformInfo().GetSocVersion();
    bool isInt32PrecisionSupported = socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93;

    // 判断输入是否符合kernel支持的混合输入类型
    bool isMixDataType = isDivsMixDtypeSupport(self, other);
    const aclTensor* divOpOut = nullptr;
    if (isMixDataType) {
        // aclScalar转aclTensor
        auto promoteType = other->GetDataType() == op::DataType::DT_DOUBLE ? op::DataType::DT_FLOAT :
                                                                             other->GetDataType();
        auto otherConvert = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
        CHECK_RET(otherConvert != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfProcessed = isSupportNonContiguous ?
                                 uniqueExecutor.get()->CreateView(self, self->GetViewShape(), self->GetStorageShape(),
                                                                  self->GetViewStrides(), self->GetViewOffset()) :
                                 l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);
        divOpOut = l0op::RealDiv(selfProcessed, otherConvert, true, uniqueExecutor.get());
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        auto promoteType = (!IsRegBase(npuArch)) ?
                               CompatibleInferDivsDtype(self->GetDataType(), other->GetDataType()) :
                               InferDivsModeDtype(self->GetDataType(), other->GetDataType(), MODE_REAL_DIV);
        promoteType = (IsFloatingType(self->GetDataType()) || IsComplexType(self->GetDataType()) ||
                       (self->GetDataType() == op::DataType::DT_INT32 &&
                        other->GetDataType() == op::DataType::DT_INT32 && isInt32PrecisionSupported)) ?
                          self->GetDataType() :
                          op::DataType::DT_FLOAT;
        promoteType = (self->GetDataType() == op::DataType::DT_BOOL && other->GetDataType() == op::DataType::DT_BOOL) ?
                          self->GetDataType() :
                          promoteType;
        promoteType = (IsComplexType(other->GetDataType())) ? op::PromoteType(promoteType, other->GetDataType()) :
                                                              promoteType;
        if (IsRegBase(npuArch)) {
            promoteType = op::PromoteType(self->GetDataType(), other->GetDataType()) == op::DataType::DT_INT32 ?
                              op::DataType::DT_INT32 :
                              promoteType;
            // fp16/bf16 在 float 下计算,避免标量被提前降精
            promoteType = PromoteLowPrecToFloat(promoteType);
        }

        bool canUseMuls = CanUseMuls(self, other);
        if (self->GetDataType() == promoteType && l0op::IsRealDivSupportNonContiguous(self) && !canUseMuls) {
            // aclScalar转aclTensor
            auto otherConvert = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
            CHECK_RET(otherConvert != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto selfWithStride = uniqueExecutor.get()->CreateView(self, self->GetViewShape(), self->GetStorageShape(),
                                                                   self->GetViewStrides(), self->GetViewOffset());
            divOpOut = l0op::RealDiv(selfWithStride, otherConvert, MODE_REAL_DIV, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
            CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
            CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            if (canUseMuls) {
                // 倒数在 double 下计算后单次舍入到 float,对齐 CUDA 标量除法精度
                float invB = static_cast<float>(1.0 / other->ToDouble());
                aclScalar* invBPtr = uniqueExecutor.get()->AllocScalar(invB);
                divOpOut = l0op::Muls(selfCasted, invBPtr->ToFloat(), uniqueExecutor.get());
            } else {
                // aclScalar转aclTensor
                auto otherConvert = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
                CHECK_RET(otherConvert != nullptr, ACLNN_ERR_INNER_NULLPTR);
                divOpOut = l0op::RealDiv(selfCasted, otherConvert, MODE_REAL_DIV, uniqueExecutor.get());
            }
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    auto castOut = l0op::Cast(divOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDivs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDivs);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// 返回为[1]的intarray shape
static inline aclIntArray* GetBaseShape(aclOpExecutor* executor)
{
    int64_t tensorShape[1] = {};
    tensorShape[0] = 1;
    auto res = executor->AllocIntArray(tensorShape, 1);
    return res;
}

aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor* self, const aclTensor* other, int mode, aclTensor* out,
                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDivMod, DFX_IN(self, other, mode), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, other, out, mode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckMode(mode), ACLNN_ERR_PARAM_INVALID);

    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
    CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = selfContiguous;
    auto otherCasted = otherContiguous;

    bool isMixDataType = isMixDtypeTensorSupport(self, other);
    const aclTensor* divOpOut = nullptr;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    // TruncateDiv 特殊处理：IsRegBase && mode=MODE_TRUNC_DIV && 类型组合在映射表中，不做类型提升
    if (IsRegBase(npuArch) && mode == MODE_TRUNC_DIV) {
        // 如果tensor为0维，则转换为1维tensor
        if (otherContiguous->GetViewShape().GetDimNum() == 0) {
            auto baseShape = GetBaseShape(uniqueExecutor.get());
            auto broadcastOut = l0op::BroadcastTo(otherContiguous, baseShape, uniqueExecutor.get());
            CHECK_RET(broadcastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            otherCasted = broadcastOut;
        }
        if (isInTruncDtypeMapping(self->GetDataType(), other->GetDataType())) {
            divOpOut = l0op::TruncateDiv(selfCasted, otherCasted, uniqueExecutor.get());
        } else {
            op::DataType promoteType;
            promoteType = InferDivModeDtype(self->GetDataType(), other->GetDataType(), mode);
            bool needToFloat = (promoteType == op::DataType::DT_BOOL);
            promoteType = needToFloat ? op::DataType::DT_FLOAT : promoteType;
            auto complexRet = CheckDivModComplexDtype(promoteType, mode);
            CHECK_RET(complexRet == ACLNN_SUCCESS, complexRet);
            selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
            CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            otherCasted = l0op::Cast(otherCasted, promoteType, uniqueExecutor.get());
            CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            divOpOut = l0op::TruncateDiv(selfCasted, otherCasted, uniqueExecutor.get());
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (isMixDataType) {
        if (mode == MODE_FLOOR_DIV) {
            divOpOut = l0op::FloorDiv(selfCasted, otherCasted, false, uniqueExecutor.get());
        } else {
            divOpOut = l0op::RealDiv(selfCasted, otherCasted, false, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            if (mode == MODE_TRUNC_DIV && divOpOut->GetDataType() != op::DataType::DT_INT64 &&
                divOpOut->GetDataType() != op::DataType::DT_INT16) {
                divOpOut = l0op::InplaceTrunc(divOpOut, uniqueExecutor.get());
            }
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        op::DataType promoteType;
        bool needToInt32 = false;
        op::DataType oriType = out->GetDataType();
        if (!IsRegBase(npuArch)) {
            auto promoteRet = CompatibleInferDivModeDtype(self->GetDataType(), other->GetDataType(), mode, promoteType);
            CHECK_RET(promoteRet == ACLNN_SUCCESS, promoteRet);
        } else {
            promoteType = InferDivModeDtype(self->GetDataType(), other->GetDataType(), mode);
            auto complexRet = CheckDivModComplexDtype(promoteType, mode);
            CHECK_RET(complexRet == ACLNN_SUCCESS, complexRet);
            // customization
            bool needToFloat = (promoteType == op::DataType::DT_BOOL && mode == MODE_FLOOR_DIV);
            promoteType = needToFloat ? op::DataType::DT_FLOAT : promoteType;
            // aicore is not supported, aicpu has problems when div 0
            needToInt32 = (promoteType == op::DataType::DT_INT16 && mode == MODE_FLOOR_DIV) ||
                          ((promoteType == op::DataType::DT_INT8 || promoteType == op::DataType::DT_UINT8 ||
                            promoteType == op::DataType::DT_INT16) &&
                           mode == MODE_TRUNC_DIV);
            oriType = promoteType;
            promoteType = needToInt32 ? op::DataType::DT_INT32 : promoteType;
        }
        selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 根据mode分三种场景调用算子计算
        if (mode == MODE_FLOOR_DIV) {
            divOpOut = l0op::FloorDiv(selfCasted, otherCasted, uniqueExecutor.get());
        } else {
            divOpOut = l0op::RealDiv(selfCasted, otherCasted, mode, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            if (mode == MODE_TRUNC_DIV && divOpOut->GetDataType() != op::DataType::DT_INT64 &&
                divOpOut->GetDataType() != op::DataType::DT_INT16) {
                divOpOut = l0op::Trunc(divOpOut, uniqueExecutor.get());
            }
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (needToInt32) {
            divOpOut = l0op::Cast(divOpOut, oriType, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    auto castOut = l0op::Cast(divOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDivMod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDivMod);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor* self, const aclScalar* other, int mode, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDivMods, DFX_IN(self, other, mode), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 调用适配aclScalar参数检查
    auto ret = CheckParamsScalar(self, other, out, mode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    CHECK_RET(CheckMode(mode), ACLNN_ERR_PARAM_INVALID);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = selfContiguous;
    bool isMixDataType = isMixDtypeScalarSupport(self, other);
    const aclTensor* divOpOut = nullptr;
    auto npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    // TruncateDiv 特殊处理：IsRegBase && mode=MODE_TRUNC_DIV && 类型组合在映射表中，不做类型提升
    if (IsRegBase(npuArch) && mode == MODE_TRUNC_DIV) {
        if (isInTruncDtypeMapping(self->GetDataType(), other->GetDataType())) {
            auto otherConvert = uniqueExecutor.get()->ConvertToTensor(other, other->GetDataType());
            CHECK_RET(otherConvert != nullptr, ACLNN_ERR_INNER_NULLPTR);
            divOpOut = l0op::TruncateDiv(selfContiguous, otherConvert, uniqueExecutor.get());
        } else {
            op::DataType promoteType;
            promoteType = InferDivsModeDtype(self->GetDataType(), other->GetDataType(), mode);
            bool needToFloat = (promoteType == op::DataType::DT_BOOL);
            promoteType = needToFloat ? op::DataType::DT_FLOAT : promoteType;
            auto complexRet = CheckDivModComplexDtype(promoteType, mode);
            CHECK_RET(complexRet == ACLNN_SUCCESS, complexRet);
            selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
            CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            auto otherCasted = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
            CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
            divOpOut = l0op::TruncateDiv(selfCasted, otherCasted, uniqueExecutor.get());
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (isMixDataType) {
        auto otherConvert = uniqueExecutor.get()->ConvertToTensor(other, other->GetDataType());
        CHECK_RET(otherConvert != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (mode == MODE_FLOOR_DIV) {
            divOpOut = l0op::FloorDiv(selfCasted, otherConvert, true, uniqueExecutor.get());
        } else if (mode == MODE_REAL_DIV) {
            divOpOut = l0op::RealDiv(selfCasted, otherConvert, true, uniqueExecutor.get());
        } else if (mode == MODE_TRUNC_DIV) {
            divOpOut = l0op::RealDiv(selfCasted, otherConvert, false, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            divOpOut = l0op::InplaceTrunc(divOpOut, uniqueExecutor.get());
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        op::DataType promoteType;
        bool needToInt32 = false;
        op::DataType oriType = out->GetDataType();
        if (!IsRegBase(npuArch)) {
            auto promoteRet = CompatibleInferDivsModeDtype(self->GetDataType(), other->GetDataType(), mode,
                                                           promoteType);
            CHECK_RET(promoteRet == ACLNN_SUCCESS, promoteRet);
        } else {
            promoteType = InferDivsModeDtype(self->GetDataType(), other->GetDataType(), mode);
            // trueDiv/floorDiv 在 float 下计算,避免标量被压到 fp16/bf16 丢精度;trunc 保持原推导
            if (mode == MODE_REAL_DIV || mode == MODE_FLOOR_DIV) {
                promoteType = PromoteLowPrecToFloat(promoteType);
            }
            auto complexRet = CheckDivModComplexDtype(promoteType, mode);
            CHECK_RET(complexRet == ACLNN_SUCCESS, complexRet);
            // customization
            bool needToFloat = (promoteType == op::DataType::DT_BOOL && mode == MODE_FLOOR_DIV);
            promoteType = needToFloat ? op::DataType::DT_FLOAT : promoteType;
            // aicore is not supported, aicpu has problems when div 0
            needToInt32 = (promoteType == op::DataType::DT_INT16 && mode == MODE_FLOOR_DIV) ||
                          ((promoteType == op::DataType::DT_INT8 || promoteType == op::DataType::DT_UINT8 ||
                            promoteType == op::DataType::DT_INT16) &&
                           mode == MODE_TRUNC_DIV);
            oriType = promoteType;
            promoteType = needToInt32 ? op::DataType::DT_INT32 : promoteType;
        }
        selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherCasted = uniqueExecutor.get()->ConvertToTensor(other, promoteType);
        CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 根据mode分三种场景调用算子计算
        if (mode == MODE_FLOOR_DIV) {
            divOpOut = l0op::FloorDiv(selfCasted, otherCasted, uniqueExecutor.get());
        } else {
            divOpOut = l0op::RealDiv(selfCasted, otherCasted, mode, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            if (mode == MODE_TRUNC_DIV && divOpOut->GetDataType() != op::DataType::DT_INT64 &&
                divOpOut->GetDataType() != op::DataType::DT_INT16) {
                divOpOut = l0op::Trunc(divOpOut, uniqueExecutor.get());
            }
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (needToInt32) {
            divOpOut = l0op::Cast(divOpOut, oriType, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    auto castOut = l0op::Cast(divOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDivMods(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDivMods);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static inline aclnnStatus CheckInplace(const aclTensor* selfRef, const aclTensor* other)
{
    OP_CHECK(selfRef != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected selfRef not to be null."),
             return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(other != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected other not to be null."),
             return ACLNN_ERR_PARAM_NULLPTR);
    op::Shape broadcastShape;
    OP_CHECK(
        BroadcastInferShape(selfRef->GetViewShape(), other->GetViewShape(), broadcastShape),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape of selfRef and other can't broadcast, got %s, %s.",
                op::ToString(selfRef->GetViewShape()).GetString(), op::ToString(other->GetViewShape()).GetString()),
        return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(selfRef->GetViewShape() == broadcastShape,
             OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Expected shape of selfRef should be %s, but got %s.",
                     op::ToString(broadcastShape).GetString(), op::ToString(selfRef->GetViewShape()).GetString()),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceDivGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    auto ret = CheckInplace(selfRef, other);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto out = const_cast<aclTensor*>(selfRef);
    if (isMixDtypeTensorSupport(selfRef, other)) {
        L2_DFX_PHASE_1(aclnnInplaceDiv, DFX_IN(selfRef, other), DFX_OUT(out));
        // 固定写法，创建OpExecutor
        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

        // 固定写法，参数检查
        auto retMix = CheckParams(selfRef, other, out, MODE_REAL_DIV);
        CHECK_RET(retMix == ACLNN_SUCCESS, retMix);

        // div算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
        if (selfRef->IsEmpty() || other->IsEmpty()) {
            // 根据实际支持情况补充
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }

        // 固定写法，将输入self转换成连续的tensor
        auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
        CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将输入other转换成连续的tensor
        auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto divOpOut = l0op::RealDiv(selfRefContiguous, otherContiguous, true, uniqueExecutor.get());
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果转换成输出out的数据类型
        auto castOut = l0op::Cast(divOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，获取计算过程中需要使用的workspace大小
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    } else {
        return aclnnDivGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
    }
}

aclnnStatus aclnnInplaceDiv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceDiv);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceDivsGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnDivsGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceDivs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceDivs);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceDivModGetWorkspaceSize(aclTensor* selfRef, const aclTensor* other, int mode,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckInplace(selfRef, other);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto out = const_cast<aclTensor*>(selfRef);
    if (isMixDtypeTensorSupport(selfRef, other)) {
        L2_DFX_PHASE_1(aclnnInplaceDivMod, DFX_IN(selfRef, other, mode), DFX_OUT(out));

        auto uniqueExecutor = CREATE_EXECUTOR();
        CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

        auto retMix = CheckParams(selfRef, other, out, mode);
        CHECK_RET(retMix == ACLNN_SUCCESS, retMix);
        CHECK_RET(CheckMode(mode), ACLNN_ERR_PARAM_INVALID);

        if (selfRef->IsEmpty() || other->IsEmpty()) {
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }

        auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
        CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        const aclTensor* divOpOut = nullptr;
        if (mode == MODE_FLOOR_DIV) {
            divOpOut = l0op::FloorDiv(selfRefContiguous, otherContiguous, true, uniqueExecutor.get());
        } else if (mode == MODE_REAL_DIV) {
            divOpOut = l0op::RealDiv(selfRefContiguous, otherContiguous, true, uniqueExecutor.get());
        } else if (mode == MODE_TRUNC_DIV) {
            divOpOut = l0op::RealDiv(selfRefContiguous, otherContiguous, false, uniqueExecutor.get());
            CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            divOpOut = l0op::InplaceTrunc(divOpOut, uniqueExecutor.get());
        }
        CHECK_RET(divOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(divOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    } else {
        return aclnnDivModGetWorkspaceSize(selfRef, other, mode, out, workspaceSize, executor);
    }
}

aclnnStatus aclnnInplaceDivMod(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceDivMod);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(aclTensor* selfRef, const aclScalar* other, int mode,
                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(selfRef);
    return aclnnDivModsGetWorkspaceSize(selfRef, other, mode, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceDivMods(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceDivMods);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
