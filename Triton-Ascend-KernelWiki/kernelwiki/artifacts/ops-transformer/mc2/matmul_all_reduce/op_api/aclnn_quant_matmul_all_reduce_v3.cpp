/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_quant_matmul_all_reduce_v3.cpp
 * \brief
 */
#include "aclnn_quant_matmul_all_reduce_v3.h"

#include "matmul_all_reduce_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

extern aclnnStatus aclnnInnerMatmulAllReduceGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x3,
    const aclTensor* antiquantScale, const aclTensor* antiquantOffset, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* commQuantScale1, const aclTensor* commQuantScale2,
    const char* group, const char* reduceOp, bool transposeX1, bool transposeX2, int64_t commTurn,
    int64_t antiquantGroupSize, int64_t groupSize, int64_t yDtype, int64_t commQuantMode, const aclTensor* output,
    uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);
extern "C" uint64_t NnopbaseMsprofSysTime();
extern "C" void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId& dfxId);
extern "C" aclnnStatus __attribute__((weak)) NnopbaseDisableOptionalInput(void* executor, const size_t irIndex);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_QUANT = {op::DataType::DT_INT8};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_DEQUANT_310P = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_310P = {op::DataType::DT_FLOAT16};

// 检查入参是否为nullptr
static bool CheckNotNull(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* dequantScale, const aclTensor* output)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(dequantScale, return false);
    OP_CHECK_NULL(output, return false);
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* x3, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const aclTensor* output)
{
    const auto& dequantDtypeSupport = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 ?
                                          DTYPE_SUPPORT_LIST_DEQUANT_310P :
                                          DTYPE_SUPPORT_LIST_DEQUANT;

    const auto& outDtypeSupport = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 ?
                                      DTYPE_SUPPORT_LIST_310P :
                                      DTYPE_SUPPORT_LIST;

    // 检查x1、x2、bias、scale、offset、x3、output的数据类型是否在算子的支持列表内
    // 对于量化来说，x1,x2只为INT8
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_QUANT, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(dequantScale, dequantDtypeSupport, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(output, outDtypeSupport, return false);
    // 检查bias、offset、x3的数据类型是否在算子的支持列表内
    if (bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
    }
    if (x3 != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x3, outDtypeSupport, return false);
        // 检查x3和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(x3, output, return false);
    }
    // 检查x1和x2的数据类型是否相同
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    // BF16场景，dequantScale的dtype与output一致，为BF16
    if (output->GetDataType() == op::DataType::DT_BF16 || dequantScale->GetDataType() == op::DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_SAME(dequantScale, output, return false);
    }
    // pertokenScale必须为float32
    if (pertokenScale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(pertokenScale, DTYPE_SUPPORT_LIST_PERTOKEN, return false);
        // FP16场景，dequantScale的dtype与pertokenScale一致，为float32
        if (output->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(dequantScale, pertokenScale, return false);
        }
    }
    if (commQuantScale1Optional != nullptr && commQuantScale2Optional != nullptr) {
        // 检查commQuantScale1 commQuantScale2的数据类型是否在算子的支持列表内
        OP_CHECK_DTYPE_NOT_SUPPORT(commQuantScale1Optional, DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(commQuantScale2Optional, DTYPE_SUPPORT_LIST, return false);
        // 检查commQuantScale1Optional commQuantScale2Optional和output的数据类型是否相同
        OP_CHECK_DTYPE_NOT_SAME(commQuantScale1Optional, output, return false);
        OP_CHECK_DTYPE_NOT_SAME(commQuantScale2Optional, output, return false);
    }
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckAttr(const char* reduceOp, int64_t streamMode)
{
    if (strcmp(reduceOp, REDUCE_OP_SUM) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduceOp to be sum, but got %s.", reduceOp);
        return false;
    }
    if (streamMode != NUM_ACL_STOP_ON_FAILURE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected streamMode to be 1, but got %ld.", streamMode);
        return false;
    }
    return true;
}

static void ProcessTransposedX2(const aclTensor* x2, uint64_t& x2Dim0, uint64_t& x2Dim1, ge::AscendString& x2ShapeStr)
{
    op::Shape x2ViewShape = x2->GetViewShape();
    x2ViewShape.SetDim(0, x2Dim0);
    x2ViewShape.SetDim(1, x2Dim1);
    if (QuantMatmulAllReduceIsAclnnPreTransposed(x2)) {
        x2ShapeStr = op::ToString(x2ViewShape);
    }
    OP_LOGD("MatmulAllReduce, x2 view shape is %s", x2ShapeStr.GetString());
}

static bool CheckShape(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* commQuantScale1Optional, const aclTensor* commQuantScale2Optional,
    const aclTensor* x3, const aclTensor* output)
{
    bool isWeightNZ = MatmulAllReduceIsWeightNZFormat(x2);

    OP_CHECK_MIN_DIM(x1, TWO_DIMS, return false);
    OP_CHECK_MAX_DIM(x1, THREE_DIMS, return false);
    if (isWeightNZ) {
        return true;
    }
    // x2的维度为2维,x1的维度为2D或者3D，output的维数与x1一致,weightNZ场景下，x2可能为4维
    OP_CHECK_WRONG_DIMENSION(x2, TWO_DIMS, return false);

    uint64_t x2Dim0 = QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    uint64_t x2Dim1 = QuantMatmulAllReduceIsAclnnPreTransposed(x2) ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x2ShapeStr = op::ToString(x2->GetViewShape());
    ProcessTransposedX2(x2, x2Dim0, x2Dim1, x2ShapeStr);
    // 仅支持x2矩阵转置，x1不支持转置, x1.GetDimNum(1) == x2.GetDimNum(0)
    const size_t x1Len = x1->GetViewShape().GetDimNum();
    if (static_cast<uint64_t>(x1->GetViewShape().GetDim(x1Len - 1)) != x2Dim0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected last dim of x1 to be equal to first dim of x2, but got x1 shape: %s, x2 shape: %s.",
            op::ToString(x1->GetViewShape()).GetString(), x2ShapeStr.GetString());
        return false;
    }

    // output的最后一维与x2的最后一维相同
    op::Shape outShape = x1->GetViewShape();
    outShape.SetDim(x1Len - 1, x2Dim1);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(output, outShape, return false);

    // x1 shape [s,m,k], x2 shape [k,n], output shape [s,m,n], bias shape [n]
    if (bias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(bias, ONE_DIM, return false);
        op::Shape biasShape;
        biasShape.AppendDim(output->GetViewShape().GetDim(x1Len - 1));
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(bias, biasShape, return false);
    }

    // x3 shape [s,m,n]
    if (x3 != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(x3, output, return false);
    }

    // scale和offset为per-tensor则shape为[1]，为per-channel则shape为[n]或者[1,n]
    OP_CHECK_MIN_DIM(dequantScale, ONE_DIM, return false);
    OP_CHECK_MAX_DIM(dequantScale, TWO_DIMS, return false);
    const size_t scaleLen = dequantScale->GetViewShape().GetDimNum();
    if (!(scaleLen == DIM_LEN_ONE && (dequantScale->GetViewShape().GetDim(0) == NUM_ONE ||
                                      dequantScale->GetViewShape().GetDim(0) == outShape.GetDim(x1Len - 1))) &&
        !(scaleLen == DIM_LEN_TWO && dequantScale->GetViewShape().GetDim(0) == NUM_ONE &&
          dequantScale->GetViewShape().GetDim(1) == outShape.GetDim(x1Len - 1))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Expected dequantScale be [1] or [n] or [1,n], last dim of dequantScale should be %ld or 1, "
            "but got dequantScale shape: %s.",
            output->GetViewShape().GetDim(x1Len - 1), op::ToString(dequantScale->GetViewShape()).GetString());
        return false;
    }

    // pertokenScale的shape为[m]
    if (pertokenScale != nullptr) {
        OP_CHECK_MIN_DIM(pertokenScale, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(pertokenScale, ONE_DIM, return false);
        const size_t pertokenScaleLen = pertokenScale->GetViewShape().GetDimNum();
        int64_t x1M = 1;
        for (size_t dim = 0; dim < x1Len - 1; dim++) {
            x1M *= x1->GetViewShape().GetDim(dim);
        }
        if (!(pertokenScaleLen == DIM_LEN_ONE && pertokenScale->GetViewShape().GetDim(0) == x1M)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Expected pertokenScale be [%ld], but got pertokenScale shape: %s.", x1M,
                op::ToString(pertokenScale->GetViewShape()).GetString());
            return false;
        }
    }

    // commQuantScale1、commQuantScale2必须同时存在，且shape为[1，n]或[n]
    if ((commQuantScale1Optional != nullptr && commQuantScale2Optional == nullptr) ||
        (commQuantScale1Optional == nullptr && commQuantScale2Optional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "comm_quantScale_1 and comm_quant_scale_2 should both not be nullptr.");
        return false;
    }
    if (commQuantScale1Optional != nullptr && commQuantScale2Optional != nullptr) {
        OP_CHECK_MIN_DIM(commQuantScale1Optional, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(commQuantScale1Optional, TWO_DIMS, return false);
        OP_CHECK_MIN_DIM(commQuantScale2Optional, ONE_DIM, return false);
        OP_CHECK_MAX_DIM(commQuantScale2Optional, TWO_DIMS, return false);
        const size_t commQuantScale1OptionalLen = commQuantScale1Optional->GetViewShape().GetDimNum();
        const size_t commQuantScale2OptionalLen = commQuantScale2Optional->GetViewShape().GetDimNum();
        if (commQuantScale1OptionalLen != commQuantScale2OptionalLen) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected commQuantScale1 and commQuantScale2 have the same shape, "
                "but got commQuantScale1 shape: %s, commQuantScale2 shape: %s.",
                op::ToString(commQuantScale1Optional->GetViewShape()).GetString(),
                op::ToString(commQuantScale2Optional->GetViewShape()).GetString());
            return false;
        }
        uint64_t commQuantScale1Dim0 = commQuantScale1Optional->GetViewShape().GetDim(0);
        uint64_t commQuantScale2Dim0 = commQuantScale2Optional->GetViewShape().GetDim(0);
        uint64_t commQuantScale1Dim1 = 0;
        uint64_t commQuantScale2Dim1 = 0;
        if (commQuantScale1OptionalLen == TWO_DIMS && commQuantScale2OptionalLen == TWO_DIMS) {
            commQuantScale1Dim1 = commQuantScale1Optional->GetViewShape().GetDim(1);
            commQuantScale2Dim1 = commQuantScale2Optional->GetViewShape().GetDim(1);
        }

        if ((!(commQuantScale1OptionalLen == ONE_DIM && commQuantScale1Dim0 == x2Dim1 &&
               commQuantScale2Dim0 == x2Dim1)) &&
            (!(commQuantScale1OptionalLen == TWO_DIMS && commQuantScale1Dim0 == 1 && commQuantScale2Dim0 == 1 &&
               commQuantScale1Dim1 == x2Dim1 && commQuantScale2Dim1 == x2Dim1))) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Expected commQuantScale1 and commQuantScale2 be [n] or [1,n], last dim should be %ld, "
                "but got commQuantScale1 shape: %s, commQuantScale2 shape: %s.",
                x2Dim1, op::ToString(commQuantScale1Optional->GetViewShape()).GetString(),
                op::ToString(commQuantScale2Optional->GetViewShape()).GetString());
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* dequantScale,
    const aclTensor* pertokenScale, const aclTensor* x3, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const char* reduceOp, int64_t streamMode, const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, dequantScale, output), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(
        CheckDtypeValid(
            x1, x2, bias, dequantScale, pertokenScale, x3, commQuantScale1Optional, commQuantScale2Optional, output),
        ACLNN_ERR_PARAM_INVALID);

    // 3. 检查attr是否符合规则
    CHECK_RET(CheckAttr(reduceOp, streamMode), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(
        CheckShape(
            x1, x2, bias, dequantScale, pertokenScale, commQuantScale1Optional, commQuantScale2Optional, x3, output),
        ACLNN_ERR_PARAM_INVALID);

    // 5.【A2】检查x2矩阵非连续合法性
    if (op::GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B &&
        !MatmulAllReduceCheckValidEmptyTensor(x1, x2)) {
        CHECK_RET(MatmulAllReduceCheckValidContiguous(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static const aclTensor* CopyTensor(const aclTensor* x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (size_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }
    OP_LOGD("MatmulAllReduce, CopyTensor storageDimsNum is %lu.", storageDimsNum);
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    auto stride = x2->GetViewStrides();
    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_UNDEFINED;
    auto stgFormat = ge::GetPrimaryFormat(x2->GetStorageFormat());
    if (stgFormat == Format::FORMAT_ND) {
        OP_LOGD("MatmulAllReduce, CopyTensor format is ACL_FORMAT_ND");
        format = aclFormat::ACL_FORMAT_ND;
    } else if (stgFormat == Format::FORMAT_FRACTAL_NZ) {
        format = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    }
    return aclCreateTensor(
        storageDims.data(), storageDimsNum, dataType, stride.data(), offset, format, storageDims.data(), storageDimsNum,
        x2->GetTensor()->GetAddr());
}

aclnnStatus aclnnQuantMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional, const aclTensor* x3Optional,
    const aclTensor* dequantScale, const aclTensor* pertokenScaleOptional, const aclTensor* commQuantScale1Optional,
    const aclTensor* commQuantScale2Optional, const char* group, const char* reduceOp, int64_t commTurn,
    int64_t streamMode, const aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // 固定写法，参数检查
    auto retParam = CheckParams(
        x1, x2, biasOptional, dequantScale, pertokenScaleOptional, x3Optional, commQuantScale1Optional,
        commQuantScale2Optional, reduceOp, streamMode, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // dequantScale转为uint64
    auto dequant = const_cast<aclTensor*>(dequantScale);
    if (dequant == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmulAllReduce, dequant is nullptr.");
        return ACLNN_ERR_INNER;
    }
    if (dequant->GetDataType() == op::DataType::DT_INT64) {
        dequant->SetDataType(op::DataType::DT_UINT64);
    }

    // 目前不支持x1进行transpose
    bool transposeX1 = false;
    bool transposeX2 = IsTransposeLastTwoDims(x2) || QuantMatmulAllReduceIsAclnnPreTransposed(x2);
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    int64_t antiquantGroupSize = 0;
    auto tempX2 = x2;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2002 && MatmulAllReduceIsWeightNZFormat(x2)) {
        if(x2->GetTensor() == nullptr){
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Tensor of x2 is null.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        tempX2 = CopyTensor(x2);
    }
    uint64_t yDtype = static_cast<uint64_t>(output->GetDataType());
    aclnnStatus ret = aclnnInnerMatmulAllReduceGetWorkspaceSize(
        x1, tempX2, biasOptional, x3Optional, scale, offset, dequant, pertokenScaleOptional, commQuantScale1Optional,
        commQuantScale2Optional, group, reduceOp, transposeX1, transposeX2, commTurn, antiquantGroupSize, 0, yDtype, 0,
        output, workspaceSize, executor);

    OP_LOGI("Group %s, reduce op %s, trans flag %u %u, ret %d.", group, reduceOp, transposeX1, transposeX2, ret);
#ifdef MC2_UT
    ret = 0;
#endif
    if (ret == 0) {
        if (NnopbaseDisableOptionalInput != nullptr) {
            NnopbaseDisableOptionalInput(*executor, 4U); // 4 is input irIndex
            NnopbaseDisableOptionalInput(*executor, 5U); // 5 is input irIndex
        }
    }
    OP_LOGD("QuantMatmulAllReduce, end ret %d", ret);
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnQuantMatmulAllReduceV3(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAllReduce(workspace, workspaceSize, executor, stream);
    OP_LOGD("QuantMatmulAllReduce, aclnnQuantMatmulAllReduceV3 ret %d", ret);

    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "QuantMatmulAllReduce, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }

    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
