/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_fmod_tensor.h"

#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "mod.h"
#include "conversion/squeeze/op_host/op_api/squeeze.h"
#include "conversion/unsqueeze/op_host/op_api/unsqueeze.h"
#include "aclnn_kernels/transdata.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_UINT8};
static const std::initializer_list<DataType> emptyDtypes = {};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_COMPLEX = {
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<DataType>& GetDtypeSupportList(SocVersion socVersion)
{
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
    IsRegBase()) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else if (socVersion == SocVersion::ASCEND910) {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    } else if (socVersion == SocVersion::ASCEND310P) {
        return ASCEND310P_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented", op::ToString(socVersion).GetString());
        return emptyDtypes;
    }
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

static inline DataType PromoteTypeScalarV35(const op::DataType tensorDtype, const op::DataType scalarDtype)
{
    auto scalarDefaultDtype = GetScalarDefaultDtype(scalarDtype);
    auto promoteType = CombineCategoriesWithComplex(tensorDtype, scalarDefaultDtype);
    return promoteType;
}

// tensor + scalar混合场景下，推导出应该cast的dtype (并不是promoteType)
static inline DataType PromoteTypeScalar(const op::DataType selfDtype, const op::DataType otherDtype)
{
    if (IsRegBase()) {
        return PromoteTypeScalarV35(selfDtype, otherDtype);
    }

    if (IsFloatingType(selfDtype)) {
        return selfDtype;
    } else {
        if (IsFloatingType(otherDtype) || selfDtype == op::DataType::DT_BOOL) {
            return op::PromoteType(selfDtype, otherDtype);
        }
        return selfDtype;
    }
}

static inline bool IsAiCoreComputeDtype(const op::DataType dtype)
{
    return dtype == op::DataType::DT_BF16 || dtype == op::DataType::DT_FLOAT16 ||
           dtype == op::DataType::DT_FLOAT || dtype == op::DataType::DT_INT32;
}

static inline bool IsNarrowIntegerDtype(const op::DataType dtype)
{
    return dtype == op::DataType::DT_INT8 || dtype == op::DataType::DT_UINT8;
}

static inline bool IsIntegerScalarDtype(const op::DataType dtype)
{
    return dtype == op::DataType::DT_INT8 || dtype == op::DataType::DT_UINT8 ||
           dtype == op::DataType::DT_INT32 || dtype == op::DataType::DT_INT64;
}

static inline DataType SelectAiCoreComputeDtype(const op::DataType promoteType, const op::DataType outDtype)
{
    if (IsAiCoreComputeDtype(outDtype)) {
        return outDtype;
    }
    return promoteType;
}

static inline DataType SelectAiCoreScalarComputeDtype(
    const op::DataType castDtype, const op::DataType selfDtype, const op::DataType scalarDtype,
    const op::DataType outDtype)
{
    if (IsAiCoreComputeDtype(outDtype)) {
        return outDtype;
    }
    if (IsAiCoreComputeDtype(castDtype)) {
        return castDtype;
    }
    if (IsNarrowIntegerDtype(selfDtype) && IsIntegerScalarDtype(scalarDtype)) {
        return op::DataType::DT_INT32;
    }
    return castDtype;
}

// 得到tensor的维度数
static inline int64_t GetTensorDimNum(const aclTensor* self)
{
    return (int64_t)(self->GetViewShape().GetDimNum());
}

// Tensor self, Tensor other 检查参数是否为空指针
static aclnnStatus CheckNotNullTensorTensor(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(other, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// Tensor self, Scalar other 检查参数是否为空指针
static aclnnStatus CheckNotNullTensorScalar(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    OP_CHECK_NULL(self, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(other, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// 1. self和other能推导出合理的数据类型 promoteType  2. promoteType能cast成out  3. promoteType属于支持的dtype
static bool CheckPromoteType(const op::DataType selfDtype, const op::DataType otherDtype, const op::DataType outDtype)
{
    // 检查self和other能否做数据类型推导
    auto promoteType = op::PromoteType(selfDtype, otherDtype);
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self dtype %s and other dtype %s can not promote dtype.",
            op::ToString(selfDtype).GetString(), op::ToString(otherDtype).GetString());
        return false;
    }

    // 判断self和other推导出的数据类型能cast成out
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, outDtype, return false);

    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList(socVersion);
    if (DTYPE_SUPPORT_LIST.size() == 0) {
        return false;
    }
    if (!CheckType(promoteType, DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Promote type %s should be in dtype support list [%s].",
            op::ToString(promoteType).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
        return false;
    }

    return true;
}

// 1. self和other没有complex  2. self能cast成castDtype  3. castDtype为算子支持的数据类型  4. castDtype能cast成out
static bool CheckPromoteTypeTensorScalar(
    const op::DataType selfDtype, const op::DataType otherDtype, const op::DataType outDtype)
{
    // 检查self和other没有为complex
    if (CheckType(selfDtype, DTYPE_SUPPORT_LIST_COMPLEX) || CheckType(otherDtype, DTYPE_SUPPORT_LIST_COMPLEX)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fmod_npu not implemented for dtype complex.");
        return false;
    }

    auto castDtype = PromoteTypeScalar(selfDtype, otherDtype);
    // 检查self能cast成 castDtype
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDtype, castDtype, return false);

    // castDtype的数据类型属于支持的数据类型
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList(socVersion);
    if (DTYPE_SUPPORT_LIST.size() == 0) {
        return false;
    }
    if (!CheckType(castDtype, DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "expected dtype %s should be in dtype support list [%s].",
            op::ToString(castDtype).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
        return false;
    }

    // castDtype的数据类型能cast成out
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(castDtype, outDtype, return false);

    return true;
}


// tensor维度数不能超过8维
static inline bool CheckTensorDimSize(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}


// other must be broadcastable to self, and out must keep self shape.
static bool CheckBroadcastShape(const aclTensor* self, const aclTensor* other, const aclTensor* out, bool isInplace)
{
    (void)isInplace;
    op::Shape broadcastShape;
    if (IsRegBase()) {
        OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
    } else {
        OP_CHECK_BROADCAST(self, other, return false);
        BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);
    }

    if (broadcastShape != self->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Broadcast shape %s should be equal to self shape %s.",
            op::ToString(broadcastShape).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }
    if (out->GetViewShape() != self->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "out shape %s should be equal to self shape %s.",
            op::ToString(out->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }
    return true;
}

// 如果为0维tensor，那么转换为1维tensor。其余情况转成连续tensor, 然后cast成对应的promote type
static const aclTensor* InitializeTensor(const aclTensor* x, op::DataType dtype, aclOpExecutor* executor)
{
    auto xContiguous = l0op::Contiguous(x, executor);
    if (xContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor xContiguous should not be null.");
        return nullptr;
    }

    // 如果tensor为0维，则转换为1维tensor
    if (GetTensorDimNum(xContiguous) == 0) {
        int64_t tensorShape[1] = {};
        tensorShape[0] = 1;
        auto baseShape = executor->AllocIntArray(tensorShape, 1);
        xContiguous = l0op::BroadcastTo(xContiguous, baseShape, executor);
        if (xContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "After broadcast, tensor xContiguous should not be null.");
            return nullptr;
        }
    }
    if (x->GetDataType() != dtype) {
        xContiguous = l0op::Cast(xContiguous, dtype, executor);
        if (xContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "After cast, tensor xContiguous should not be null.");
            return nullptr;
        }
    }
    return xContiguous;
}

// 取得tensor的shape
static aclIntArray* GetTensorShape(const aclTensor* self, aclOpExecutor* executor)
{
    int64_t tensorSize = GetTensorDimNum(self);
    // 0维场景，返回1维1元素的shape
    if (tensorSize == 0) {
        int64_t tensorShape[1] = {};
        tensorShape[0] = 1;
        auto baseShape = executor->AllocIntArray(tensorShape, 1);
        return baseShape;
    }

    std::vector<int64_t> tensorShape(tensorSize);
    for (int64_t i = 0; i < tensorSize; i++) {
        tensorShape[i] = (self->GetViewShape())[i];
    }
    auto res = executor->AllocIntArray(tensorShape.data(), tensorSize);
    return res;
}

// broadcast成对应shape
static const aclTensor* BroadcastTensor(
    const aclTensor* x, const aclTensor* out, const aclIntArray* broadcastShape, aclOpExecutor* executor)
{
    // 涉及3维->4维，4维->5维，因此都reformat成ND
    x = l0op::ReFormat(x, op::Format::FORMAT_ND);
    if (x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Before broadcast, Reformat result is nullptr.");
    }
    if (x->GetViewShape() != out->GetViewShape()) {
        x = l0op::BroadcastTo(x, broadcastShape, executor);
        if (x == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast result is nullptr.");
        }
        x = l0op::Contiguous(x, executor);
        if (x == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast contiguous result is nullptr.");
        }
    }

    // 涉及3维->4维，4维->5维，因此都reformat成ND
    x = l0op::ReFormat(x, op::Format::FORMAT_ND);
    if (x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reformat result is nullptr.");
    }

    return x;
}

static aclnnStatus FmodMainProcess(
    const aclTensor* selfContiguous, const aclTensor* otherContiguous, const aclTensor* out, bool needUnsqueeze,
    aclOpExecutor* executor)
{
    auto fmodOut = l0op::Mod(selfContiguous, otherContiguous, executor);
    CHECK_RET(fmodOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (fmodOut->GetDataType() != out->GetDataType()) {
        fmodOut = l0op::Cast(fmodOut, out->GetDataType(), executor);
        CHECK_RET(fmodOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (needUnsqueeze) {
        int64_t squeezeDim = 0;
        fmodOut = l0op::SqueezeNd(fmodOut, squeezeDim, executor);
        CHECK_RET(fmodOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewcopyResult = l0op::ViewCopy(fmodOut, out, executor);
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

// 提取公共的CheckParamsTensorScalar逻辑
static aclnnStatus CheckParamsTensorScalarCommon(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullTensorScalar(self, other, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    // 2. self和out的shape一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return ACLNN_ERR_PARAM_INVALID);
    // 3. self和other没有complex + self能cast成castDtype + castDtype为算子支持的数据类型 + castDtype能cast成out
    CHECK_RET(
        CheckPromoteTypeTensorScalar(self->GetDataType(), other->GetDataType(), out->GetDataType()),
        ACLNN_ERR_PARAM_INVALID);
    // 4. 维度数不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// Tensor self, Tensor other
static aclnnStatus CheckParamsTensorTensor(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullTensorTensor(self, other, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    // 2. self和other能推导出合理的数据类型 + self和other推导除的数据类型能cast成out + promoteType为算子支持的数据类型
    CHECK_RET(CheckPromoteType(self->GetDataType(), other->GetDataType(), out->GetDataType()), ACLNN_ERR_PARAM_INVALID);
    // 3. self、other、out之间的shape可以互相Broadcast
    CHECK_RET(CheckBroadcastShape(self, other, out, false), ACLNN_ERR_PARAM_INVALID);
    // 4. 维度数不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(other), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// Tensor self, Tensor other Inplace
static aclnnStatus CheckParamsInplaceTensorTensor(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullTensorTensor(self, other, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    // 2. self和other能推导出合理的数据类型 + self和out的dtype一致 + promoteType为算子支持的数据类型
    CHECK_RET(CheckPromoteType(self->GetDataType(), other->GetDataType(), out->GetDataType()), ACLNN_ERR_PARAM_INVALID);
    // 3. self、other、out之间的shape可以互相Broadcast
    CHECK_RET(CheckBroadcastShape(self, other, out, true), ACLNN_ERR_PARAM_INVALID);
    // 4. 维度数不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(other), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// Tensor self, Scalar other
static aclnnStatus CheckParamsTensorScalar(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    return CheckParamsTensorScalarCommon(self, other, out);
}

// Tensor self, Scalar other Inplace
static aclnnStatus CheckParamsInplaceTensorScalar(const aclTensor* self, const aclScalar* other, const aclTensor* out)
{
    return CheckParamsTensorScalarCommon(self, other, out);
}

// 提取SetWorkspaceAndRelease逻辑
static aclnnStatus SetWorkspaceAndRelease(
    UniqueExecutor& uniqueExecutor, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 提取RunFmodProcessAndRelease逻辑，消除else分支重复
static aclnnStatus RunFmodProcessAndRelease(
    const aclTensor* self, const aclTensor* other, const aclTensor* out,
    UniqueExecutor& uniqueExecutor, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    bool needUnsqueeze = (GetTensorDimNum(out) == 0);
    auto fmodRes =
        FmodMainProcess(self, other, out, needUnsqueeze, uniqueExecutor.get());
    CHECK_RET(fmodRes == ACLNN_SUCCESS, fmodRes);

    return SetWorkspaceAndRelease(uniqueExecutor, workspaceSize, executor);
}

// Tensor self, Tensor other
aclnnStatus ExecFmodTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // self / other为空tensor，返回空tensor
    if (self->IsEmpty() || other->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto promoteType = op::PromoteType(self->GetDataType(), other->GetDataType());
    auto computeType = SelectAiCoreComputeDtype(promoteType, out->GetDataType());
    if (IsAiCoreComputeDtype(computeType)) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfCasted = l0op::Cast(selfContiguous, computeType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherCasted = l0op::Cast(otherContiguous, computeType, uniqueExecutor.get());
        CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto modOpOut = l0op::Mod(selfCasted, otherCasted, uniqueExecutor.get());
        CHECK_RET(modOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(modOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return SetWorkspaceAndRelease(uniqueExecutor, workspaceSize, executor);
    } else {
        auto selfContiguous = InitializeTensor(self, promoteType, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherContiguous = InitializeTensor(other, promoteType, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 需要做broadcast
        auto broadcastShape = GetTensorShape(out, uniqueExecutor.get());
        selfContiguous = BroadcastTensor(selfContiguous, out, broadcastShape, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        otherContiguous = BroadcastTensor(otherContiguous, out, broadcastShape, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return RunFmodProcessAndRelease(
            selfContiguous, otherContiguous, out, uniqueExecutor, workspaceSize, executor);
    }
}

// Tensor self, Scalar other
static aclnnStatus FmodScalarGetWorkspaceSizeCommon(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckParamsTensorScalar(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // self为空tensor，直接返回空tensor
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto castDtype = PromoteTypeScalar(self->GetDataType(), other->GetDataType());
    auto computeType = SelectAiCoreScalarComputeDtype(
        castDtype, self->GetDataType(), other->GetDataType(), out->GetDataType());
    if (IsAiCoreComputeDtype(computeType)) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfCasted = l0op::Cast(selfContiguous, computeType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto otherTensor = uniqueExecutor.get()->ConvertToTensor(other, computeType);
        CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto modOpOut = l0op::Mod(selfCasted, otherTensor, uniqueExecutor.get());
        CHECK_RET(modOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(modOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return SetWorkspaceAndRelease(uniqueExecutor, workspaceSize, executor);
    } else {
        auto selfContiguous = InitializeTensor(self, castDtype, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherContiguous = uniqueExecutor.get()->ConvertToTensor(other, castDtype);
        auto selfShape = GetTensorShape(selfContiguous, uniqueExecutor.get());
        CHECK_RET(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        otherContiguous = l0op::BroadcastTo(otherContiguous, selfShape, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return RunFmodProcessAndRelease(
            selfContiguous, otherContiguous, out, uniqueExecutor, workspaceSize, executor);
    }
}

static aclnnStatus FmodInplaceScalarGetWorkspaceSizeCommon(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckParamsInplaceTensorScalar(selfRef, other, selfRef);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return FmodScalarGetWorkspaceSizeCommon(selfRef, other, selfRef, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnFmodScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFmodScalar, DFX_IN(self, other), DFX_OUT(out));
    return FmodScalarGetWorkspaceSizeCommon(self, other, out, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnFmodScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFmodScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

ACLNN_API aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFmodScalar, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    return FmodInplaceScalarGetWorkspaceSizeCommon(selfRef, other, workspaceSize, executor);
}

ACLNN_API aclnnStatus aclnnInplaceFmodScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFmodScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// 非inplace
ACLNN_API aclnnStatus aclnnFmodTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFmodTensor, DFX_IN(self, other), DFX_OUT(out));
    auto ret = CheckParamsTensorTensor(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecFmodTensorGetWorkspaceSize(self, other, out, workspaceSize, executor);
}

// inplace
ACLNN_API aclnnStatus aclnnInplaceFmodTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceFmodTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckParamsInplaceTensorTensor(selfRef, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecFmodTensorGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

// Tensor self, Tensor other
ACLNN_API aclnnStatus aclnnFmodTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFmodTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// Tensor self, Tensor other
ACLNN_API aclnnStatus aclnnInplaceFmodTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceFmodTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
