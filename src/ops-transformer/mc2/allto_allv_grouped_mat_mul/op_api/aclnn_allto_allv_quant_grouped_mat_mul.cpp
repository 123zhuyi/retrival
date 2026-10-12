/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include <algorithm>

#include "securec.h"
#include "op_mc2.h"
#include "acl/acl.h"
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "hccl_util.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "aclnn_allto_allv_quant_grouped_mat_mul.h"
#include "allto_allv_grouped_mat_mul_checker.h"

namespace {
using namespace op;

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

enum class QuantModeType : int64_t {
    NO_QUANT = 0,
    PERTENSOR_QUANT = 1,
    PERCHANNEL_QUANT = 2,
    PERTOKEN_QUANT = 3,
    PERGROUP_QUANT = 4,
    PERBLOCK_QUANT = 5,
    MX_QUANT = 6,
};

// 需要使用的常量定义
static constexpr int64_t ZERO = 0;
static constexpr size_t MAX_BSK_LEN = 52428800U;
static constexpr size_t MAX_H1_LEN = 65536U;
static constexpr size_t MAX_EXPERT_SIZE = 256U;
static constexpr size_t MAX_E_SIZE = 32U;
static constexpr size_t MAX_H2_LEN = 12288U;
static constexpr size_t MAX_N_LEN = 65536U;
static constexpr size_t MIN_K_LEN = 2U;
static constexpr size_t MAX_K_LEN = 8U;
static constexpr size_t TWO_DIMS = 2U;
static constexpr size_t THREE_DIMS = 3U;

static bool CheckNullStatus(const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
                            const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
                            bool permuteOutFlag, const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // // 检查必选入参出参为非空
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr) && (mmXScaleOptional != nullptr) && (mmWeightScaleOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr) && (mmXScaleOptional == nullptr) && (mmWeightScaleOptional == nullptr)))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, left: %u, right: %u, "
            "mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is nullptr: %u, mmXScaleOptional is nullptr: %u,"
            "mmWeightScaleOptional is nullptr: %u,",
            (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr) &&
               (mmXScaleOptional != nullptr) && (mmWeightScaleOptional != nullptr))),
            (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr) &&
               (mmXScaleOptional == nullptr) && (mmWeightScaleOptional == nullptr))),
            mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr, mmXScaleOptional == nullptr,
            mmWeightScaleOptional == nullptr);
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Optional output flag does not match optional output ptr.");
        return false;
    }
    return true;
}

// 检查必要输入是否为空/quantMode=1，必须非空/1
static bool CheckNotNull(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY,
                         const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, int64_t gmmXQuantMode,
                         int64_t gmmWeightQuantMode)
{
    if (gmmX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmX should not be null.");
        return false;
    }
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmWeight should not be null.");
        return false;
    }
    if (gmmY == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmY should not be null.");
        return false;
    }
    if (gmmXScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXScale should not be null.");
        return false;
    }
    if (gmmWeightScale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightScale should not be null.");
        return false;
    }
    if (gmmXQuantMode != static_cast<int64_t>(QuantModeType::PERTENSOR_QUANT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmXQuantMode should be 1.");
        return false;
    }
    if (gmmWeightQuantMode != static_cast<int64_t>(QuantModeType::PERTENSOR_QUANT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeightQuantMode should be 1.");
        return false;
    }
    return true;
}

// 根据API定义，列出输入的所能支持的所有dtype
static const std::initializer_list<op::DataType> IN_DTYPE_SUPPORT_LIST = {op::DataType::DT_HIFLOAT8};
// 根据API定义，列出输入Scale所能支持的所有dtype
static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
// 根据API定义，列出输出output所能支持的所有dtype
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                           op::DataType::DT_BF16};
// 校验所有输入的参数类型是否正确
static bool CheckDtypesValid(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                             const aclTensor *gmmWeightScale, const aclTensor *mmXOptional,
                             const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                             const aclTensor *mmWeightScaleOptional, const aclTensor *gmmY,
                             const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmX, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeight, IN_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmY, OUT_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(gmmXScale, SCALE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gmmWeightScale, SCALE_DTYPE_SUPPORT_LIST, return false);

    if ((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr) &&
        (mmXScaleOptional != nullptr) && (mmWeightScaleOptional != nullptr)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(mmXOptional, IN_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightOptional, IN_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmYOptional, OUT_DTYPE_SUPPORT_LIST, return false);

        OP_CHECK_DTYPE_NOT_SUPPORT(mmXScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(mmWeightScaleOptional, SCALE_DTYPE_SUPPORT_LIST, return false);
    }

    if (permuteOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteOutOptional, IN_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

// 检查暂不支持的输入参数是否为空，必须为空
static bool CheckNotSupportNull(const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
                                const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional)
{
    if (gmmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmXOffsetOptional should be null.");
        return false;
    }
    if (gmmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input gmmWeightOffsetOptional should be null.");
        return false;
    }
    if (mmXOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input mmXOffsetOptional should be null.");
        return false;
    }
    if (mmWeightOffsetOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input mmWeightOffsetOptional should be null.");
        return false;
    }
    return true;
}


// 检查是否有空tensor
static bool CheckEmptyTensor(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmY)
{
    if(gmmX->GetViewShape().GetDim(0) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with zero dimM, which is unsupported.");
        return false;
    }
    if(gmmX->GetViewShape().GetDim(1) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmX is empty tensor with one dimK, which is unsupported.");
        return false;
    }
    if(gmmWeight->GetViewShape().GetDim(0) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with zero dimE, which is unsupported.");
        return false;
    }
    if(gmmWeight->GetViewShape().GetDim(1) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with one dimK, which is unsupported.");
        return false;
    }
    if(gmmWeight->GetViewShape().GetDim(2) == ZERO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight is empty tensor with three dimN, which is unsupported.");
        return false;
    }
    return true;
}

bool is_power_of_two(int64_t n)
{
    return n > 0 && (n & (n - 1)) == 0 && n >= 2 && n <= 128;
}

// 检查所有要用到的输入format是否为ND，如果内部不为ND格式，会打印warning日志
static bool CheckFormat(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                        const aclTensor *gmmWeightScale, const aclTensor *mmXOptional,
                        const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                        const aclTensor *mmWeightScaleOptional, const aclTensor *gmmY, const aclTensor *mmYOptional)
{
    if ((IsPrivateFormat(gmmX->GetStorageFormat())) || IsPrivateFormat(gmmWeight->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "gmmX format %s or gmmWeight format %s do not support private format.",
                op::ToString(gmmX->GetStorageFormat()).GetString(), op::ToString(gmmWeight->GetStorageFormat()).GetString());
        return false;
    }
    if ((IsPrivateFormat(gmmXScale->GetStorageFormat())) || (IsPrivateFormat(gmmWeightScale->GetStorageFormat()))) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "gmmXScale format %s or gmmWeightScale format %s do not support private format.",
            op::ToString(gmmXScale->GetStorageFormat()).GetString(), op::ToString(gmmWeightScale->GetStorageFormat()).GetString());
        return false;
    }
    if (mmXOptional != nullptr) {
        if ((IsPrivateFormat(mmXOptional->GetStorageFormat())) || (IsPrivateFormat(mmXScaleOptional->GetStorageFormat()))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmXOptional format %s or mmXScaleOptional format %s do not support private format.",
                    op::ToString(mmXOptional->GetStorageFormat()).GetString(),
                    op::ToString(mmXScaleOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (mmWeightOptional != nullptr) {
        if ((IsPrivateFormat(mmWeightOptional->GetStorageFormat())) ||
            (IsPrivateFormat(mmWeightScaleOptional->GetStorageFormat()))) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmWeightOptional format %s or mmWeightScaleOptional format %s do not support private format.",
                    op::ToString(mmWeightOptional->GetStorageFormat()).GetString(),
                    op::ToString(mmWeightScaleOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(gmmY->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "gmmY format %s does not support private format.",
                op::ToString(gmmY->GetStorageFormat()).GetString());
        return false;
    }
    if (mmYOptional != nullptr) {
        if (IsPrivateFormat(mmYOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmYOptional format %s does not support private format.",
                    op::ToString(mmYOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckGmmWeightValid(const aclTensor *gmmWeight) {
    if (gmmWeight == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "In AlltoAllvQuantGroupedMatmul, input gmmWeight should not be null.");
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(gmmWeight, THREE_DIMS, return false);
  	if (gmmWeight->IsEmpty()) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In AlltoAllvQuantGroupedMatmul, input gmmWeight do not support empty tensor.");
    	return false;
  	}
    return true;
}

static bool CheckMmWeightValid(const aclTensor *mmWeightOptional) {
    if (mmWeightOptional == nullptr) {
        return false;
    }
    OP_CHECK_WRONG_DIMENSION(mmWeightOptional, TWO_DIMS, return false);
  	if (mmWeightOptional->IsEmpty()) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In AlltoAllvQuantGroupedMatmul, input mmWeightOptional is empty tensor.");
    	return false;
  	}
    return true;
}


// 处理支持转置的tensor物理排布不连续问题（gmmWeight）
static const aclTensor *TransGmmWeightTensor(const aclTensor *gmmWeight)
{
    uint64_t storageShapeDimNum = gmmWeight->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = gmmWeight->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = gmmWeight->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = gmmWeight->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[1] = gmmWeight->GetViewShape().GetDim(2);
    viewDim[2] = gmmWeight->GetViewShape().GetDim(1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(gmmWeight, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = gmmWeight->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[1] = transStride[2];
    stride[2] = transStride[1];

    auto offset = gmmWeight->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, gmmWeight->GetTensor()->GetAddr());
}

// 处理支持转置的tensor物理排布不连续问题（mmWeightOptional）
static const aclTensor *TransMmWeightOptionalTensor(const aclTensor *mmWeightOptional)
{
    uint64_t storageShapeDimNum = mmWeightOptional->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDim(storageShapeDimNum);
    for (uint64_t i = 0; i < storageShapeDimNum; i++) {
        storageDim[i] = mmWeightOptional->GetStorageShape().GetDim(i);
    }

    uint64_t viewShapeDimNum = mmWeightOptional->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDim;
    viewDim.resize(viewShapeDimNum);
    for (uint64_t i = 0; i < viewShapeDimNum; i++) {
        viewDim[i] = mmWeightOptional->GetViewShape().GetDim(i);
    }
    // transpose the viewshape last two dimensions
    viewDim[0] = mmWeightOptional->GetViewShape().GetDim(1);
    viewDim[1] = mmWeightOptional->GetViewShape().GetDim(0);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(mmWeightOptional, &dataType);
    std::vector<int64_t> stride(viewShapeDimNum);
    auto transStride = mmWeightOptional->GetViewStrides();
    stride = std::vector<int64_t>(transStride.begin(), transStride.end());
    // transpose the two dimensions
    stride[0] = transStride[1];
    stride[1] = transStride[0];

    auto offset = mmWeightOptional->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDim.data(), viewShapeDimNum, dataType, stride.data(), offset, format, storageDim.data(),
                           storageShapeDimNum, mmWeightOptional->GetTensor()->GetAddr());
}

// 检查tensor是否连续
bool IsTransposeLastTwoDims(const aclTensor *tensor) {
    // 当输入tensor的shape小于2或者大于6的时候，返回错误
    if (tensor->GetViewShape().GetDimNum() < 2 || tensor->GetViewShape().GetDimNum() > 6) {
        return false;
    }
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - 2;
    // 根据stride步长判断tensor是否连续取值的
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        if (tensor->GetViewShape().GetDim(dim1) == 1 && tensor->GetViewShape().GetDim(dim2) == 1) {		// 表示tensor为1x1的大小，不存在非连续问题
            return false;
          }
        return true;
      }
    return false;
}


static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale,
                               const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional,
                               const aclTensor *gmmWeightOffsetOptional, const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
                               const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional,
                               const aclTensor *mmWeightScaleOptional, const aclTensor *mmXOffsetOptional,
                               const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
                               int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode,
                               const char *group, int64_t epWorldSize, bool permuteOutFlag, const aclTensor *gmmY,
                               const aclTensor *mmYOptional, const aclTensor *permuteOutOptional)
{
    // 检查空状态
    CHECK_RET(CheckNullStatus(sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                              mmXScaleOptional, mmWeightScaleOptional, permuteOutFlag, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查group长度是否小于等于128
    CHECK_RET(Mc2AlltoAllvGMMChecker::CheckGroup(group), ACLNN_ERR_PARAM_INVALID);
    // 检查参数是否为空
    CHECK_RET(CheckNotNull(gmmX, gmmWeight, gmmY, gmmXScale, gmmWeightScale, gmmXQuantMode, gmmWeightQuantMode),
              ACLNN_ERR_PARAM_INVALID);
    // 检查暂不支持的参数是否为空，不影响场景
    CHECK_RET(
        CheckNotSupportNull(gmmXOffsetOptional, gmmWeightOffsetOptional, mmXOffsetOptional, mmWeightOffsetOptional),
        ACLNN_ERR_PARAM_INVALID);
    // 检查空tensor
    CHECK_RET(CheckEmptyTensor(gmmX, gmmWeight, gmmY), ACLNN_ERR_PARAM_INVALID);
    // 检查所有输入/量化数据类型
    CHECK_RET(CheckDtypesValid(gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmXOptional, mmWeightOptional,
                               mmXScaleOptional, mmWeightScaleOptional, gmmY, mmYOptional, permuteOutOptional),
              ACLNN_ERR_PARAM_INVALID);
    // 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(gmmX, gmmWeight, gmmXScale, gmmWeightScale, mmXOptional, mmWeightOptional, mmXScaleOptional,
                          mmWeightScaleOptional, gmmY, mmYOptional),
              ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("AlltoAllvQuantGroupedMatmul checkParams success");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t groupSize, int64_t yDtype, int64_t mmDtype, const aclTensor *gmmY,
    const aclTensor *mmYOptional, const aclTensor *permuteOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern "C" aclnnStatus aclnnInnerAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" aclnnStatus InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const aclTensor *gmmXScale, const aclTensor *gmmWeightScale, const aclTensor *gmmXOffsetOptional,
    const aclTensor *gmmWeightOffsetOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    bool permuteOutFlag, int64_t gmmXQuantMode, int64_t gmmWeightQuantMode, int64_t mmXQuantMode,
    int64_t mmWeightQuantMode, int64_t groupSize, const aclTensor *gmmY, const aclTensor *mmYOptional,
    const aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{

    int64_t yDtype = gmmY->GetDataType();
    int64_t mmDtype = mmYOptional == nullptr ? 0 : mmYOptional->GetDataType();

    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScale,
        gmmWeightScale, gmmXOffsetOptional, gmmWeightOffsetOptional, mmXScaleOptional, mmWeightScaleOptional,
        mmXOffsetOptional, mmWeightOffsetOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, groupSize,
        yDtype, mmDtype, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvQuantGroupedMatmul, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *gmmXScale, const aclTensor *gmmWeightScale,
    const aclTensor *gmmXOffsetOptional, const aclTensor *gmmWeightOffsetOptional,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const aclTensor *mmXScaleOptional, const aclTensor *mmWeightScaleOptional,
    const aclTensor *mmXOffsetOptional, const aclTensor *mmWeightOffsetOptional, int64_t gmmXQuantMode,
    int64_t gmmWeightQuantMode, int64_t mmXQuantMode, int64_t mmWeightQuantMode, const char *group, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight,
    int64_t groupSize, bool permuteOutFlag, aclTensor *gmmY, aclTensor *mmYOptional, aclTensor *permuteOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前支持转置的gmmWeight涉及该处理
    CHECK_RET(CheckGmmWeightValid(gmmWeight), ACLNN_ERR_PARAM_NULLPTR);	// 先检查gmmWeight是否合法，避免非法操作
    bool notContiguous = IsTransposeLastTwoDims(gmmWeight);    // notContiguous标识gmmWeight是否是非连续的，通常在pytorch经过.t()会导致gmmWeight非连续
    auto transposeGmmWeight = gmmWeight;    // 复制一个gmmWeight
    if (notContiguous && transGmmWeight) {    // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gmmWeight not contiguous, and set gmmWeight transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {    // 只有当非连续时，才会涉及到转连续等情况
        transGmmWeight = !transGmmWeight;
        // 把非连续gmmWeight转成连续
        transposeGmmWeight = TransGmmWeightTensor(gmmWeight);
        CHECK_RET(transposeGmmWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("gmmWeight is a non-contiguous tensor. The original dim1 is %ld, and dim2 is %ld. After processing, transposeGmmWeight dim1 is %ld, and dim2 is %ld.",
            gmmWeight->GetViewShape().GetDim(1), gmmWeight->GetViewShape().GetDim(2), transposeGmmWeight->GetViewShape().GetDim(1), transposeGmmWeight->GetViewShape().GetDim(2));
    }

    // 处理非连续Tensor，目前支持转置的mmWeightOptional涉及该处理
    if (CheckMmWeightValid(mmWeightOptional)) {
        bool notContiguous = IsTransposeLastTwoDims(mmWeightOptional); // notContiguous标识mmWeightOptional是否是非连续的
        auto transMmWeightOptional = mmWeightOptional;
        if (notContiguous && transMmWeight) { // 当非连续和转置同时生效时，判断为错误用法，直接报错
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "mmWeightOptional not contiguous, and set mmWeightOptional transpose, it is error!");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            transMmWeight = !transMmWeight;
            // 把非连续mmWeightOptional转成连续
            transMmWeightOptional = TransMmWeightOptionalTensor(mmWeightOptional);
            CHECK_RET(transMmWeightOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
            OP_LOGD("mmWeightOptional is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After "
                    "processing, transMmWeightOptional dim0 is %ld, and dim1 is %ld.",
                    mmWeightOptional->GetViewShape().GetDim(0), mmWeightOptional->GetViewShape().GetDim(1),
                    transMmWeightOptional->GetViewShape().GetDim(0), transMmWeightOptional->GetViewShape().GetDim(1));
            mmWeightOptional = transMmWeightOptional;
        }
        
    }
    aclnnStatus ret_param = CheckParams(
        gmmX, transposeGmmWeight, gmmXScale, gmmWeightScale, gmmXOffsetOptional, gmmWeightOffsetOptional,
        sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, mmXScaleOptional,
        mmWeightScaleOptional, mmXOffsetOptional, mmWeightOffsetOptional, gmmXQuantMode, gmmWeightQuantMode,
        mmXQuantMode, mmWeightQuantMode, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = Mc2AlltoAllvGMMChecker::CheckSendAndRecv(sendCounts, recvCounts, gmmX, gmmY);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    aclnnStatus ret = InnerAlltoAllvQuantGroupedMatMulGetWorkspaceSize(
        gmmX, transposeGmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, gmmXScale,
        gmmWeightScale, gmmXOffsetOptional, gmmWeightOffsetOptional, mmXScaleOptional, mmWeightScaleOptional,
        mmXOffsetOptional, mmWeightOffsetOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight,
        transMmWeight, permuteOutFlag, gmmXQuantMode, gmmWeightQuantMode, mmXQuantMode, mmWeightQuantMode, groupSize,
        gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    return ret;
}

extern "C" aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void *workspace, uint64_t workspaceSize,
                                                        aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
    return ret;
}
} // namespace
