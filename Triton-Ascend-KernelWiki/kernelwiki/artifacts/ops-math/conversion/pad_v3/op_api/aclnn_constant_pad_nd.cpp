/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_constant_pad_nd.h"
#include "padv3.h"
#include "conversion/fill/op_api/fill.h"
#include "conversion/strided_slice/op_api/strided_slice.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16,   DataType::DT_INT8,      DataType::DT_DOUBLE,
    DataType::DT_INT16, DataType::DT_INT64, DataType::DT_UINT64,    DataType::DT_UINT32,    DataType::DT_UINT16,
    DataType::DT_UINT8, DataType::DT_BOOL,  DataType::DT_COMPLEX64, DataType::DT_COMPLEX128};

static const std::initializer_list<DataType> DTYPE_SUPPORT_910B_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16,   DataType::DT_INT8,       DataType::DT_DOUBLE,
    DataType::DT_INT16, DataType::DT_INT64, DataType::DT_UINT64,    DataType::DT_UINT32,     DataType::DT_UINT16,
    DataType::DT_UINT8, DataType::DT_BOOL,  DataType::DT_COMPLEX64, DataType::DT_COMPLEX128, DataType::DT_BF16};

static const std::initializer_list<DataType> DTYPE_SUPPORT_REGBASE_LIST = {
    DataType::DT_FLOAT,       DataType::DT_INT32,         DataType::DT_FLOAT16,     DataType::DT_INT8,
    DataType::DT_DOUBLE,      DataType::DT_INT16,         DataType::DT_INT64,       DataType::DT_UINT64,
    DataType::DT_UINT32,      DataType::DT_UINT16,        DataType::DT_UINT8,       DataType::DT_BOOL,
    DataType::DT_COMPLEX64,   DataType::DT_COMPLEX128,    DataType::DT_BF16,        DataType::DT_HIFLOAT8,
    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT8_E8M0, DataType::DT_FLOAT4_E2M1,
    DataType::DT_FLOAT4_E1M2};

static const std::initializer_list<DataType> DTYPE_SUPPORT_FP8_FP4_LIST = {
    DataType::DT_HIFLOAT8,    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN,
    DataType::DT_FLOAT8_E8M0, DataType::DT_FLOAT4_E2M1, DataType::DT_FLOAT4_E1M2};

static const size_t DIM_BOUND = 8;
static const size_t SIZE_T_TWICE = 2;
static const uint32_t POSITIVE = 1;
static const uint32_t NEGETIVE = 2;
static const std::string MODE = "constant";

static bool CheckNotNull(const aclTensor* self, const aclIntArray* pad, const aclScalar* value, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(pad, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    bool is910bSocVersion = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
                             GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST = is910bSocVersion ? DTYPE_SUPPORT_910B_LIST :
                                                                                        DTYPE_SUPPORT_LIST;
    if (IsRegBase()) {
        CURRENT_DTYPE_SUPPORT_LIST = DTYPE_SUPPORT_REGBASE_LIST;
    }
    // 检查self与out数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclIntArray* pad, const aclTensor* out, uint32_t& signSymbol)
{
    size_t selfDim = self->GetViewShape().GetDimNum();
    size_t outDim = out->GetViewShape().GetDimNum();
    size_t padLen = pad->Size();
    size_t padCover = padLen / SIZE_T_TWICE;
    OP_CHECK_MAX_DIM(self, DIM_BOUND, return false);
    OP_CHECK_MAX_DIM(out, DIM_BOUND, return false);

    if (selfDim != outDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of self[%zu] and out[%zu] can't be different.", selfDim, outDim);
        return false;
    }

    // pad元素个数必须为偶数
    if (padLen % SIZE_T_TWICE != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Pad len must be divisible by 2");
        return false;
    }
    // 判断pad数组是否符合要求
    if (padCover > selfDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Expected aclnnConstantPadNd pad len [%zu] to not be greater than twice of [%zu] but check failed.",
                padLen, selfDim);
        return false;
    }

    bool hasZero = false;
    // pad中每个值都不能让out的shape小于0，如果pad中存在正数，则out的shape中不能有0
    for (size_t i = 0; i < selfDim; ++i) {
        int64_t curShape = self->GetViewShape().GetDim(selfDim - i - 1);
        int64_t begin = (SIZE_T_TWICE * i) >= padLen ? 0 : (*pad)[SIZE_T_TWICE * i];
        int64_t end = (SIZE_T_TWICE * i + 1) >= padLen ? 0 : (*pad)[SIZE_T_TWICE * i + 1];
        int64_t newShape = curShape + begin + end;
        int64_t min = std::min(begin, end);
        min = std::min(min, begin + end);
        if (curShape + min < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The input size %ld plus padding %ld and %ld resulted in a negative output size,"
                    " which is invalid. Check dimension %zu of your input.",
                    curShape, begin, end, selfDim - i - 1);
            return false;
        }
        if (begin > 0 || end > 0) {
            signSymbol |= POSITIVE;
        }
        if (begin < 0 || end < 0) {
            signSymbol |= NEGETIVE;
        }
        if (newShape != out->GetViewShape().GetDim(selfDim - i - 1)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "output shape at dim %zu check failed,"
                    "The output shape you provided is %ld at dim %zu,"
                    " and the expected one is %ld at dim %zu based on the infershape.",
                    selfDim - i - 1, out->GetViewShape().GetDim(selfDim - i - 1), selfDim - i - 1, newShape,
                    selfDim - i - 1);
            return false;
        }
        if (i < padCover && newShape == 0) {
            hasZero = true;
        }
    }

    if (hasZero && ((signSymbol & POSITIVE) == POSITIVE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The output size %s with zero element is invalid, please check your input.",
                op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static bool Checkformat(const aclTensor* self, const aclTensor* out)
{
    if (self->GetStorageFormat() != out->GetStorageFormat() &&
        self->GetStorageFormat() != static_cast<op::Format>(ACL_FORMAT_ND)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "If Input tensor's format not ND, Input tensor's format[%s] should be same with output's format[%s].",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGW("Format of input gets [%s] and output gets [%s], this format may lead to precision failure",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
    }
    return true;
}

static bool CheckPadForFp8(const aclTensor* self, const uint32_t& signSymbol)
{
    // self的数据类型为fp8时，pad数组中不能有负数，StridedSlice不支持FLOAT8_E8M0类型
    if (CheckType(self->GetDataType(), DTYPE_SUPPORT_FP8_FP4_LIST) && (signSymbol & NEGETIVE) == NEGETIVE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For fp8 data type, pad array cannot contain negative values.");
        return false;
    }
    return true; // 非fp8
}

static aclnnStatus CheckParams(const aclTensor* self, const aclIntArray* pad, const aclScalar* value,
                               const aclTensor* out, uint32_t& signSymbol)
{
    // 1. 检查参数是否为空指针
    CHECK_COND(CheckNotNull(self, pad, value, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");

    // 3. 检查format是否满足约束
    CHECK_COND(Checkformat(self, out), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");

    // 4. 检查shape是否满足约束
    CHECK_COND(CheckShape(self, pad, out, signSymbol), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");

    // 5. 检查pad是否满足约束，StridedSlice不支持FLOAT8_E8M0类型，所以限制pad数组中不能有负数
    CHECK_COND(CheckPadForFp8(self, signSymbol), ACLNN_ERR_PARAM_INVALID, "CheckPadForFp8 failed!");

    return ACLNN_SUCCESS;
}

static aclIntArray* GetRealPad(const aclTensor* self, const aclIntArray* pad, aclOpExecutor* executor)
{
    size_t selfDim = self->GetViewShape().GetDimNum();
    size_t padCover = pad->Size() / SIZE_T_TWICE;
    FVector<int64_t> padVec;
    for (size_t i = 0; i < (selfDim - padCover); ++i) {
        padVec.push_back(0);
        padVec.push_back(0);
    }
    for (size_t i = padCover; i > 0; --i) {
        padVec.push_back((*pad)[i * SIZE_T_TWICE - SIZE_T_TWICE]);
        padVec.push_back((*pad)[i * SIZE_T_TWICE - 1]);
    }
    return executor->AllocIntArray(padVec.data(), padVec.size());
}

static aclnnStatus GetIntArr(const aclTensor* self, const aclIntArray* realPad, aclIntArray** noneNegPad,
                             aclOpExecutor* executor)
{
    // 构造非负pad以及begin、end数组
    size_t selfDim = self->GetViewShape().GetDimNum();
    FVector<int64_t> noneNeg;
    for (size_t i = 0; i < selfDim; ++i) {
        noneNeg.push_back(0);
        noneNeg.push_back(0);
    }
    for (size_t i = 0; i < realPad->Size(); ++i) {
        if ((*realPad)[i] > 0) {
            noneNeg[i] = (*realPad)[i];
        }
    }

    (*noneNegPad) = executor->AllocIntArray(noneNeg.data(), noneNeg.size());
    CHECK_RET(*noneNegPad != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus HandleSelfEmpty(const aclScalar* value, aclTensor* out, aclOpExecutor* executor)
{
    size_t outDim = out->GetViewShape().GetDimNum();
    FVector<int64_t> index;
    for (size_t i = 0; i < outDim; ++i) {
        index.push_back(out->GetViewShape().GetDim(i));
    }

    const aclTensor* indexTensor = executor->ConvertToTensor(index.data(), index.size(), DataType::DT_INT64);
    CHECK_RET(indexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将value转换为tensor，并且数据类型转换为self的数据类型
    const aclTensor* valueTensor = nullptr;
    if (out->GetDataType() == DataType::DT_BOOL) {
        auto valueTensorBool = executor->ConvertToTensor(value, DataType::DT_BOOL);
        CHECK_RET(valueTensorBool != nullptr, ACLNN_ERR_INNER_NULLPTR);
        valueTensor = l0op::Cast(valueTensorBool, DataType::DT_INT8, executor);
    } else {
        valueTensor = executor->ConvertToTensor(value, out->GetDataType());
    }
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indexArr = executor->AllocIntArray(index.data(), index.size());
    CHECK_RET(indexArr != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调fill算子
    auto fillOut = l0op::Fill(indexTensor, valueTensor, indexArr, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* fillOutCasted = fillOut;
    if (out->GetDataType() == DataType::DT_BOOL) {
        fillOutCasted = l0op::Cast(fillOut, DataType::DT_BOOL, executor);
        CHECK_RET(fillOutCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(fillOutCasted, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus DoPadV3(const aclTensor* self, const aclIntArray* noneNegPad, const aclScalar* value,
                           const aclTensor** padV3Result, aclOpExecutor* executor)
{
    auto padTensor = executor->ConvertToTensor(noneNegPad, DataType::DT_INT32);
    CHECK_RET(padTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    DataType dataType = self->GetDataType();
    if (dataType == DataType::DT_BOOL) {
        dataType = DataType::DT_INT8;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = selfContiguous;
    if (self->GetDataType() == DataType::DT_BOOL) {
        // 将bool类型承输入self的转换成int8的数据类型
        selfCasted = l0op::Cast(selfContiguous, dataType, executor);
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (CheckType(self->GetDataType(), DTYPE_SUPPORT_FP8_FP4_LIST)) {
        const uint8_t* valueData = reinterpret_cast<const uint8_t*>(value->GetData());
        size_t valueDataSize = op::TypeSize(value->GetDataType());
        auto valueTensor = executor->ConvertToTensor(value, self->GetDataType());
        for (size_t i = 0; i < valueDataSize; i++) {
            uint8_t valueDataIdx = valueData[i];
            CHECK_COND(valueDataIdx == 0 || valueTensor != nullptr, ACLNN_ERR_PARAM_INVALID,
                       "Fp8/Fp4 only support pad constant value 0.");
        }
        CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 调用l0算子PadV3进行计算
        (*padV3Result) = l0op::PadV3(selfCasted, padTensor, valueTensor, MODE, true, executor);
        CHECK_RET((*padV3Result) != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // 将value转换为tensor，并且数据类型转换为self的数据类型
        return ACLNN_SUCCESS;
    }
    auto valueTensor = executor->ConvertToTensor(value, self->GetDataType());
    CHECK_RET(valueTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valueCasted = valueTensor;
    if (self->GetDataType() == DataType::DT_BOOL) {
        // 将bool类型的value转换成int8的数据类型
        valueCasted = l0op::Cast(valueTensor, dataType, executor);
        CHECK_RET(valueCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 调用l0算子PadV3进行计算
    (*padV3Result) = l0op::PadV3(selfCasted, padTensor, valueCasted, MODE, true, executor);
    CHECK_RET(padV3Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET((*padV3Result) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus DoStridedSlice(const aclTensor* padV3Result, const aclIntArray* realPad, aclTensor* out,
                                  aclOpExecutor* executor)
{
    op::Shape curShape = padV3Result->GetViewShape();
    size_t outDim = out->GetViewShape().GetDimNum();
    DataType dataType = out->GetDataType();
    if (dataType == DataType::DT_BOOL) {
        dataType = DataType::DT_INT8;
    }
    FVector<int64_t> beginFV;
    FVector<int64_t> endFV;
    for (size_t i = 0; i < outDim; ++i) {
        beginFV.push_back(0);
        endFV.push_back(padV3Result->GetViewShape().GetDim(i));
    }
    FVector<int64_t> stride;
    for (size_t i = 0; i < outDim; ++i) {
        stride.push_back(1);
    }
    aclIntArray* begin = executor->AllocIntArray(beginFV.data(), beginFV.size());
    aclIntArray* end = executor->AllocIntArray(endFV.data(), endFV.size());
    aclIntArray* strideArr = executor->AllocIntArray(stride.data(), stride.size());
    auto stridesTensor = executor->ConvertToTensor(strideArr, op::ToOpDataType(ACL_INT64));
    CHECK_RET(strideArr != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto stridedResult = padV3Result;
    for (size_t i = 0; i < outDim; ++i) {
        // 遍历每个维度，更新y的shape以及begin、end数组
        if ((*realPad)[SIZE_T_TWICE * i] == 0 && (*realPad)[SIZE_T_TWICE * i + 1] == 0) {
            continue;
        }
        (*begin)[i] = ((*realPad)[SIZE_T_TWICE * i] < 0) ? -(*realPad)[SIZE_T_TWICE * i] : 0;
        (*end)[i] += ((*realPad)[SIZE_T_TWICE * i + 1] < 0) ? (*realPad)[SIZE_T_TWICE * i + 1] : 0;
        curShape[i] = (*end)[i] - (*begin)[i];
        auto y = executor->AllocTensor(curShape, dataType);
        auto beginTensor = executor->ConvertToTensor(begin, op::ToOpDataType(ACL_INT64));
        auto endTensor = executor->ConvertToTensor(end, op::ToOpDataType(ACL_INT64));
        stridedResult = l0op::StridedSlice(stridedResult, y, beginTensor, endTensor, stridesTensor, executor);
        CHECK_RET(stridedResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        (*begin)[i] = 0;
        (*end)[i] = stridedResult->GetViewShape().GetDim(i);
    }
    auto stridedResultCasted = stridedResult;
    if (out->GetDataType() == DataType::DT_BOOL) {
        stridedResultCasted = l0op::Cast(stridedResult, DataType::DT_BOOL, executor);
        CHECK_RET(stridedResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(stridedResultCasted, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor* self, const aclIntArray* pad, const aclScalar* value,
                                               aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnConstantPadNd, DFX_IN(self, pad, value), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    uint32_t signSymbol = 0; // 0代表pad全0或没有元素，1代表有正数，2代表有负数，3代表同时有正数和负数
    // 固定写法，参数检查
    auto ret = CheckParams(self, pad, value, out, signSymbol);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (signSymbol == 0) {
        // 先转连续，再拷贝
        auto selfContig = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(selfContig, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 固定写法，获取计算过程中需要使用的workspace大小
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 空Tensor处理
    if (self->IsEmpty() && !CheckType(self->GetDataType(), DTYPE_SUPPORT_FP8_FP4_LIST)) {
        ret = HandleSelfEmpty(value, out, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        // 固定写法，获取计算过程中需要使用的workspace大小
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto realPad = GetRealPad(self, pad, uniqueExecutor.get());
    CHECK_RET(realPad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    aclIntArray* noneNegPad = realPad;
    const aclTensor* padV3Result = self;
    if ((signSymbol & NEGETIVE) == NEGETIVE) {
        ret = GetIntArr(self, realPad, &noneNegPad, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    if (signSymbol != NEGETIVE) {
        // 调用l0算子PadV3进行计算
        ret = DoPadV3(self, noneNegPad, value, &padV3Result, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        padV3Result = selfContiguous;
        if (self->GetDataType() == DataType::DT_BOOL) {
            // 将bool类型承输入self的转换成int8的数据类型
            padV3Result = l0op::Cast(selfContiguous, DataType::DT_INT8, uniqueExecutor.get());
            CHECK_RET(padV3Result != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    if ((signSymbol & NEGETIVE) == 0) {
        auto padV3ResultCasted = padV3Result;
        if (out->GetDataType() == DataType::DT_BOOL) {
            padV3ResultCasted = l0op::Cast(padV3Result, DataType::DT_BOOL, uniqueExecutor.get());
            CHECK_RET(padV3ResultCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy(padV3ResultCasted, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // pad存在负数的路径
        ret = DoStridedSlice(padV3Result, realPad, out, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConstantPadNd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConstantPadNd);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
