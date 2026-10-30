/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_remainder.h"

#include "op_api/op_api_def.h"
#include "op_api/aclnn_check.h"

#include "conversion/broadcast_to/op_api/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "floor_mod.h"
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
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64,  op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310P_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_DOUBLE};
static const std::initializer_list<DataType> emptyDtypes = {};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_COMPLEX = {
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static const std::initializer_list<DataType>& GetDtypeSupportList(SocVersion socVersion)
{
    if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 || IsRegBase()) {
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

// 得到tensor的维度数
static inline int64_t GetTensorDimNum(const aclTensor* self) { return (int64_t)(self->GetViewShape().GetDimNum()); }

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
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "remainder_npu not implemented for dtype complex.");
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

// 1. self和other没有complex  2. other能cast成outDtype  3. outDtype为算子支持的数据类型
static bool CheckPromoteTypeScalarTensor(
    const op::DataType selfDtype, const op::DataType otherDtype, const op::DataType outDtype)
{
    // 检查self和other没有为complex
    if (CheckType(selfDtype, DTYPE_SUPPORT_LIST_COMPLEX) || CheckType(otherDtype, DTYPE_SUPPORT_LIST_COMPLEX)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "remainder_npu not implemented for dtype complex.");
        return false;
    }
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    auto castDtype = IsRegBase() ? PromoteTypeScalarV35(otherDtype, selfDtype) : otherDtype;
    // 检查other能cast成 outDtype
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(castDtype, outDtype, return false);

    // outDtype的数据类型属于支持的数据类型
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList(socVersion);
    if (DTYPE_SUPPORT_LIST.size() == 0) {
        return false;
    }
    if (!CheckType(outDtype, DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "out dtype %s should be in dtype support list [%s].",
            op::ToString(outDtype).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString());
        return false;
    }

    return true;
}

// input和out的shape应该一致
static inline bool CheckInputOutShape(const aclTensor* input, const aclTensor* out, const std::string& printStr)
{
    if (input->GetViewShape() != out->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s and out should have same shape.", printStr.c_str());
        return false;
    }
    return true;
}

// tensor维度数不能超过8维
static inline bool CheckTensorDimSize(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

// 确认是不是 0维 / 1维1元素的特殊场景
static inline bool CheckShapeIsSpecial(const op::Shape shape)
{
    if (shape.GetDimNum() == 0) {
        return true;
    }
    return (shape.GetDimNum() == 1 && shape.GetDim(0) == 1);
}

// self和other符合broadcast关系 + broadcast结果shape与out一致
static bool CheckBroadcastShape(const aclTensor* self, const aclTensor* other, const aclTensor* out, bool isInplace)
{
    op::Shape broadcastShape;
    if (IsRegBase()) {
        OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, other, broadcastShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, broadcastShape, return false);
        return true;
    }
    // self和other必须符合broadcast关系
    OP_CHECK_BROADCAST(self, other, return false);
    BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape);

    // self和other的broadcast shape必须与out的shape一致
    if (broadcastShape != out->GetViewShape()) {
        // 如果非inplace场景 + broadcast shape是1维1元素，那么out shape可以为 0维/1维1元素
        if ((!isInplace) && CheckShapeIsSpecial(broadcastShape) && CheckShapeIsSpecial(out->GetViewShape())) {
            return true;
        }

        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "expected consistent tensor shape for the broadcast shape and out, but got %s and %s respectively.",
            op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
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
    }

    // 涉及3维->4维，4维->5维，因此都reformat成ND
    x = l0op::ReFormat(x, op::Format::FORMAT_ND);
    if (x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reformat result is nullptr.");
    }

    return x;
}

static aclnnStatus RemainderMainProcess(
    const aclTensor* selfContiguous, const aclTensor* otherContiguous, const aclTensor* out, bool needUnsqueeze,
    aclOpExecutor* executor)
{
    auto remainderOut = l0op::FloorMod(selfContiguous, otherContiguous, executor);
    CHECK_RET(remainderOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (remainderOut->GetDataType() != out->GetDataType()) {
        remainderOut = l0op::Cast(remainderOut, out->GetDataType(), executor);
        CHECK_RET(remainderOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (needUnsqueeze) {
        int64_t squeezeDim = 0;
        remainderOut = l0op::SqueezeNd(remainderOut, squeezeDim, executor);
        CHECK_RET(remainderOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto viewcopyResult = l0op::ViewCopy(remainderOut, out, executor);
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

// Scalar self, Tensor other
static aclnnStatus CheckParamsScalarTensor(const aclScalar* self, const aclTensor* other, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(self, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(other, return ACLNN_ERR_INNER_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_INNER_NULLPTR);
    // 2. other和out的shape一致
    OP_CHECK_SHAPE_NOT_EQUAL(other, out, return ACLNN_ERR_PARAM_INVALID);
    // 3. self和other没有complex + other能cast成outDtype + outDtype为算子支持的数据类型
    CHECK_RET(
        CheckPromoteTypeScalarTensor(self->GetDataType(), other->GetDataType(), out->GetDataType()),
        ACLNN_ERR_PARAM_INVALID);
    // 4. 维度数不能超过8维
    CHECK_RET(CheckTensorDimSize(other), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDimSize(out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 提取SetWorkspaceAndRelease逻辑
static aclnnStatus SetWorkspaceAndRelease(
    UniqueExecutor& uniqueExecutor, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 提取RunRemainderProcessAndRelease逻辑，消除else分支重复
static aclnnStatus RunRemainderProcessAndRelease(
    const aclTensor* self, const aclTensor* other, const aclTensor* out, UniqueExecutor& uniqueExecutor,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    bool needUnsqueeze = (GetTensorDimNum(out) == 0);
    auto remainderRes = RemainderMainProcess(self, other, out, needUnsqueeze, uniqueExecutor.get());
    CHECK_RET(remainderRes == ACLNN_SUCCESS, remainderRes);

    return SetWorkspaceAndRelease(uniqueExecutor, workspaceSize, executor);
}

// Tensor self, Tensor other
aclnnStatus ExecRemainderTensorTensorGetWorkspaceSize(
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
    if (IsRegBase() && promoteType != op::DataType::DT_DOUBLE) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherCasted = l0op::Cast(otherContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto floorModOpOut = l0op::FloorMod(selfCasted, otherCasted, uniqueExecutor.get());
        CHECK_RET(floorModOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(floorModOpOut, out->GetDataType(), uniqueExecutor.get());
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

        return RunRemainderProcessAndRelease(
            selfContiguous, otherContiguous, out, uniqueExecutor, workspaceSize, executor);
    }
}

// Tensor self, Scalar other
aclnnStatus ExecRemainderTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // self为空tensor，直接返回空tensor
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto castDtype = PromoteTypeScalar(self->GetDataType(), other->GetDataType());
    if (IsRegBase() && castDtype != op::DataType::DT_DOUBLE) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selfCasted = l0op::Cast(selfContiguous, castDtype, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto floorModOpOut =
            l0op::FloorMod(selfCasted, uniqueExecutor.get()->ConvertToTensor(other, castDtype), uniqueExecutor.get());
        CHECK_RET(floorModOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(floorModOpOut, out->GetDataType(), uniqueExecutor.get());
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

        return RunRemainderProcessAndRelease(
            selfContiguous, otherContiguous, out, uniqueExecutor, workspaceSize, executor);
    }
}

// 非inplace
aclnnStatus aclnnRemainderTensorTensorGetWorkspaceSize(
    const aclTensor* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRemainderTensorTensor, DFX_IN(self, other), DFX_OUT(out));
    auto ret = CheckParamsTensorTensor(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecRemainderTensorTensorGetWorkspaceSize(self, other, out, workspaceSize, executor);
}

aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(
    const aclTensor* self, const aclScalar* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRemainderTensorScalar, DFX_IN(self, other), DFX_OUT(out));
    auto ret = CheckParamsTensorScalar(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecRemainderTensorScalarGetWorkspaceSize(self, other, out, workspaceSize, executor);
}

// Scalar self, Tensor other
aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(
    const aclScalar* self, const aclTensor* other, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnRemainderScalarTensor, DFX_IN(self, other), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsScalarTensor(self, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // other为空tensor，直接返回空tensor
    if (other->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (IsRegBase() && PromoteTypeScalarV35(other->GetDataType(), self->GetDataType()) != op::DataType::DT_DOUBLE) {
        auto castDtype = PromoteTypeScalarV35(other->GetDataType(), self->GetDataType());
        auto otherContiguous = l0op::Contiguous(other, uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherCasted = l0op::Cast(otherContiguous, castDtype, uniqueExecutor.get());
        CHECK_RET(otherCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto floorModOpOut =
            l0op::FloorMod(uniqueExecutor.get()->ConvertToTensor(self, castDtype), otherCasted, uniqueExecutor.get());
        CHECK_RET(floorModOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto castOut = l0op::Cast(floorModOpOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return SetWorkspaceAndRelease(uniqueExecutor, workspaceSize, executor);
    } else {
        auto selfContiguous = uniqueExecutor.get()->ConvertToTensor(self, out->GetDataType());
        auto otherShape = GetTensorShape(other, uniqueExecutor.get());
        CHECK_RET(otherShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        selfContiguous = l0op::BroadcastTo(selfContiguous, otherShape, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto otherContiguous = InitializeTensor(other, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(otherContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        return RunRemainderProcessAndRelease(
            selfContiguous, otherContiguous, out, uniqueExecutor, workspaceSize, executor);
    }
}

// inplace
aclnnStatus aclnnInplaceRemainderTensorTensorGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRemainderTensorTensor, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckParamsInplaceTensorTensor(selfRef, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecRemainderTensorTensorGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceRemainderTensorScalarGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* other, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceRemainderTensorScalar, DFX_IN(selfRef, other), DFX_OUT(selfRef));
    auto out = const_cast<aclTensor*>(selfRef);
    auto ret = CheckParamsInplaceTensorScalar(selfRef, other, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ExecRemainderTensorScalarGetWorkspaceSize(selfRef, other, out, workspaceSize, executor);
}

// Tensor self, Tensor other
aclnnStatus aclnnRemainderTensorTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRemainderTensorTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// Tensor self, Scalar other
aclnnStatus aclnnRemainderTensorScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRemainderTensorScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// Scalar self, Tensor other
aclnnStatus aclnnRemainderScalarTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRemainderScalarTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// Tensor self, Tensor other
aclnnStatus aclnnInplaceRemainderTensorTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRemainderTensorTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// Tensor self, Scalar other
aclnnStatus aclnnInplaceRemainderTensorScalar(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRemainderTensorScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif