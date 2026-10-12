/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <dlfcn.h>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "securec.h"

#include "graph/types.h"
#include "aclnn_kernels/transdata.h"
#include "log/log.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"

#include "aclnn_npu_format_cast.h"
#include "op_api/aclnn_check.h"

#define OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(aclnnName, tensor, supportList, retExpr)                               \
    do {                                                                                                              \
        if (!CheckType(tensor->GetDataType(), supportList)) {                                                         \
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(                                                                    \
                aclnnName, #tensor, op::ToString(tensor->GetDataType()).GetString(),                                  \
                "The dtype of " + std::string(#tensor) + " must be one of " + op::ToString(supportList).GetString()); \
            retExpr;                                                                                                  \
        }                                                                                                             \
    } while (0)

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
namespace {
static constexpr size_t DIMS_TWO = 2;
static constexpr size_t DIMS_THREE = 3;
static constexpr size_t DIMS_FOUR = 4;
static constexpr size_t DIMS_FIVE = 5;
static constexpr size_t DIMS_SIX = 6;
static constexpr size_t DIMS_EIGHT = 8;
static constexpr int64_t BLOCK_SIZE = 32;
static constexpr size_t FRACTAL_NZ_C0_4B = 64;
static constexpr int64_t C0_SIZE = 16;
static constexpr int64_t N0_SIZE = 16;
static constexpr int64_t NUM_ONE = 1;
static constexpr int64_t NUM_TWO = 2;
static constexpr int64_t NUM_SIXTEEN = 16;
static constexpr int NUM_FOUR = 4;
static constexpr const char* ACLNN_NAME = "aclnnNpuFormatCast";

const std::set<std::pair<op::Format, op::Format>> kTransdataForwardFormatPairsRegBase = {
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {op::Format::FORMAT_NCL, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {op::Format::FORMAT_NCL, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {op::Format::FORMAT_NCL, op::Format::FORMAT_FRACTAL_NZ},
};

const std::set<std::pair<op::Format, op::Format>> kTransdataForwardFormatPairs910B = {
    {op::Format::FORMAT_ND, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_NCL, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_NCHW, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_NCDHW, op::Format::FORMAT_FRACTAL_NZ},
    {op::Format::FORMAT_NCDHW, op::Format::FORMAT_NDC1HWC0},
    {op::Format::FORMAT_NCDHW, op::Format::FORMAT_FRACTAL_Z_3D},
    {op::Format::FORMAT_DHWCN, op::Format::FORMAT_FRACTAL_Z_3D},
    {op::Format::FORMAT_NCHW, op::Format::FORMAT_NC1HWC0},
    {op::Format::FORMAT_NHWC, op::Format::FORMAT_NC1HWC0},
    {op::Format::FORMAT_NCHW, op::Format::FORMAT_FRACTAL_Z},
    {op::Format::FORMAT_HWCN, op::Format::FORMAT_FRACTAL_Z},
    {op::Format::FORMAT_NDHWC, op::Format::FORMAT_NDC1HWC0},
};

static const std::initializer_list<DataType> ASCEND950_WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8,  DataType::DT_FLOAT,         DataType::DT_FLOAT16,     DataType::DT_BF16,
    DataType::DT_INT32, DataType::DT_FLOAT8_E4M3FN, DataType::DT_FLOAT4_E2M1, DataType::DT_HIFLOAT8,
    DataType::DT_UINT8, DataType::DT_FLOAT4_E1M2,   DataType::DT_FLOAT8_E8M0};

static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8,        DataType::DT_UINT8,    DataType::DT_FLOAT,       DataType::DT_FLOAT16,
    DataType::DT_BF16,        DataType::DT_INT32,    DataType::DT_UINT32,      DataType::DT_FLOAT8_E4M3FN,
    DataType::DT_FLOAT4_E2M1, DataType::DT_HIFLOAT8, DataType::DT_FLOAT4_E1M2, DataType::DT_FLOAT8_E8M0};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8, DataType::DT_UINT8, DataType::DT_FLOAT, DataType::DT_FLOAT16,
    DataType::DT_BF16, DataType::DT_INT32, DataType::DT_UINT32};

static const std::initializer_list<op::Format> INPUT_FORMAT_TO_NZ_SUPPORT_LIST = {
    op::Format::FORMAT_ND, op::Format::FORMAT_NCL, op::Format::FORMAT_NCHW, op::Format::FORMAT_NCDHW};

static const std::initializer_list<std::pair<DataType, op::Format>> INPUT_DTYPE_FORMAT_NZ_TO_ND_SUPPORT_LIST = {
    {DataType::DT_FLOAT4_E2M1, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_FLOAT4_E2M1, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {DataType::DT_INT8, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_INT8, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {DataType::DT_INT8, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {DataType::DT_UINT8, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_UINT8, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {DataType::DT_UINT8, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {DataType::DT_FLOAT8_E4M3FN, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_FLOAT16, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_BF16, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_INT32, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_INT32, op::Format::FORMAT_FRACTAL_NZ_C0_2},
    {DataType::DT_INT32, op::Format::FORMAT_FRACTAL_NZ_C0_4},
    {DataType::DT_INT32, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {DataType::DT_INT32, op::Format::FORMAT_FRACTAL_NZ_C0_32},
    {DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ},
    {DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ_C0_2},
    {DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ_C0_4},
    {DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ_C0_16},
    {DataType::DT_FLOAT, op::Format::FORMAT_FRACTAL_NZ_C0_32}};
inline static bool CheckDtypeValid(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    // 检查src的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(srcTensor, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    // 检查dst的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(dstTensor, ASCEND910B_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool IsNonQuantMatmulDtype(int dtype, op::Format dstFormat = op::Format::FORMAT_ND)
{
    // weight类型为FLOAT且dstFormat为FORMAT_FRACTAL_NZ_C0_16或FORMAT_FRACTAL_NZ_C0_32时，伪量化fp4场景
    return (dtype == ge::DT_FLOAT && dstFormat != op::Format::FORMAT_FRACTAL_NZ_C0_16 &&
            dstFormat != op::Format::FORMAT_FRACTAL_NZ_C0_32) ||
           dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static bool IsQuantMatmulDtype(const DataType srcDtype, const DataType dstDtype)
{
    return srcDtype == dstDtype &&
           (srcDtype == ge::DT_INT8 || srcDtype == ge::DT_UINT8 || srcDtype == ge::DT_FLOAT8_E4M3FN ||
            srcDtype == ge::DT_HIFLOAT8 || srcDtype == ge::DT_UINT8 || srcDtype == ge::DT_FLOAT4_E2M1 ||
            srcDtype == ge::DT_FLOAT4_E1M2);
}

static bool CheckInputFormatSupportedToNz(const op::Format inputFormat)
{
    return std::find(INPUT_FORMAT_TO_NZ_SUPPORT_LIST.begin(), INPUT_FORMAT_TO_NZ_SUPPORT_LIST.end(), inputFormat) !=
           INPUT_FORMAT_TO_NZ_SUPPORT_LIST.end();
}

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
        return INT64_MIN;
    }
    return (x + y - 1) / y;
}

static bool IsFloat8E8m0NnSrcFormat(op::Format srcFormat)
{
    return srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL ||
           srcFormat == op::Format::FORMAT_NCHW;
}

static bool IsFloat8E8m0NdToNn(DataType srcDtype, int additionalDtype, op::Format srcFormat, op::Format dstFormat)
{
    return srcDtype == ge::DT_FLOAT8_E8M0 && additionalDtype == ge::DT_FLOAT8_E8M0 &&
           IsFloat8E8m0NnSrcFormat(srcFormat) && dstFormat == op::Format::FORMAT_FRACTAL_NZ;
}

static aclnnStatus CheckFloat8E8m0NnShape(const gert::Shape& viewShape)
{
    auto viewShapeDim = viewShape.GetDimNum();
    OP_CHECK(viewShapeDim == DIMS_THREE || viewShapeDim == DIMS_FOUR,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 ACLNN_NAME, "srcTensor", std::to_string(viewShapeDim),
                 "FLOAT8_E8M0 ND to Ascend affinity format only supports 3D or 4D scale shape"),
             return ACLNN_ERR_PARAM_INVALID);
    for (size_t i = 0; i < viewShapeDim; ++i) {
        OP_CHECK(viewShape.GetDim(i) > 0,
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "srcTensor", op::ToString(viewShape).GetString(),
                                                       "Dims of FLOAT8_E8M0 scale must be greater than 0"),
                 return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK(viewShape.GetDim(viewShapeDim - NUM_ONE) == NUM_TWO,
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "srcTensor", op::ToString(viewShape).GetString(),
                                                   "The last dim of FLOAT8_E8M0 scale must be 2"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool ValidFloat8E8m0NnDstShape(const gert::Shape& srcViewShape, const gert::Shape& dstStorageShape)
{
    auto srcDimNum = srcViewShape.GetDimNum();
    auto dstDimNum = dstStorageShape.GetDimNum();
    OP_CHECK(dstDimNum == srcDimNum + NUM_ONE,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                 ACLNN_NAME, "srcTensor, dstTensor", std::to_string(srcDimNum) + ", " + std::to_string(dstDimNum),
                 "The StorageShape dim of dstTensor must be srcTensor dim plus 1 for FLOAT8_E8M0 Ascend affinity "
                 "format"),
             return false);

    for (uint64_t i = 0; i < srcDimNum - DIMS_THREE; ++i) {
        OP_CHECK(
            srcViewShape.GetDim(i) == dstStorageShape.GetDim(i),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                ACLNN_NAME, "srcTensor, dstTensor",
                std::string(op::ToString(srcViewShape).GetString()) + ", " + op::ToString(dstStorageShape).GetString(),
                "Prefix dims of FLOAT8_E8M0 Ascend affinity format must keep unchanged"),
            return false);
    }

    int64_t kScale = srcViewShape.GetDim(srcDimNum - DIMS_THREE);
    int64_t n = srcViewShape.GetDim(srcDimNum - DIMS_TWO);
    int64_t n1 = CeilDiv(n, NUM_SIXTEEN);
    uint64_t n1Axis = srcDimNum - DIMS_THREE;
    OP_CHECK(dstStorageShape.GetDim(n1Axis) == n1 && dstStorageShape.GetDim(n1Axis + NUM_ONE) == kScale &&
                 dstStorageShape.GetDim(n1Axis + DIMS_TWO) == NUM_SIXTEEN &&
                 dstStorageShape.GetDim(n1Axis + DIMS_THREE) == NUM_TWO,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 ACLNN_NAME, "srcTensor, dstTensor",
                 std::string(op::ToString(srcViewShape).GetString()) + ", " + op::ToString(dstStorageShape).GetString(),
                 "FLOAT8_E8M0 Ascend affinity format storage shape must be (..., ceil(N/16), KScale, 16, 2)"),
             return false);
    return true;
}

inline int64_t Ceil(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
        return INT64_MIN;
    }
    return ((x + y - 1) / y) * y;
}

static aclnnStatus ValidateNonQuantMatmulParams([[maybe_unused]] int32_t additionalDtype, const gert::Shape& viewShape,
                                                size_t viewShapeDim, [[maybe_unused]] int32_t dstFormat)
{
    // 拦截mm非量化K=1场景, 防止后续matmul2Mul报错
    int64_t kDim = viewShape.GetDim(viewShapeDim - 2);
    OP_CHECK(kDim != 1,
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                 ACLNN_NAME, "srcTensor", op::ToString(viewShape).GetString(),
                 "The axis k of srcTensor cannot be equal to 1 when additionalDtype equals the dtype of srcTensor"),
             return ACLNN_ERR_PARAM_INVALID);
    // mm非量化当前仅只支持2~6维
    OP_CHECK(
        viewShapeDim >= 2 && viewShapeDim <= 6,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            ACLNN_NAME, "srcTensor", std::to_string(viewShapeDim),
            "The shape dim of srcTensor must be between 2 and 6 when additionalDtype equals the dtype of srcTensor"),
        return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateQuantMatmulParams(int32_t additionalDtype, [[maybe_unused]] const gert::Shape& viewShape,
                                             size_t viewShapeDim)
{
    OP_CHECK(additionalDtype == ge::DT_INT8 || additionalDtype == ge::DT_UINT8 ||
                 additionalDtype == ge::DT_FLOAT8_E4M3FN || additionalDtype == ge::DT_HIFLOAT8 ||
                 additionalDtype == ge::DT_FLOAT4_E2M1 || additionalDtype == ge::DT_FLOAT4_E1M2,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 ACLNN_NAME, "additionalDtype", op::ToString(static_cast<op::DataType>(additionalDtype)).GetString(),
                 "additionalDtype must be int8/uint8/float8_e4m3fn/hifloat8/float4_e2m1/float4_e1m2 "
                 "when additionalDtype equals the dtype of srcTensor and additionalDtype is not float16 or bfloat16"),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(viewShapeDim >= 2 && viewShapeDim <= 6,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 ACLNN_NAME, "srcTensor", std::to_string(viewShapeDim),
                 "The ViewShape dim of srcTensor must be between 2 and 6 "
                 "when additionalDtype equals the dtype of srcTensor and additionalDtype is int8"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateWeightQuantMatmulParams(DataType srcDtype, int32_t additionalDtype, size_t viewShapeDim,
                                                   const int dstFormat)
{
    if (srcDtype == ge::DT_INT32) {
        OP_CHECK(
            additionalDtype == ge::DT_FLOAT16 || additionalDtype == ge::DT_BF16 || additionalDtype == ge::DT_INT8,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ACLNN_NAME, "additionalDtype", op::ToString(static_cast<op::DataType>(additionalDtype)).GetString(),
                "The value of additionalDtype must be float16/bfloat16/int8 when the dtype of srcTensor is int32"),
            return ACLNN_ERR_PARAM_INVALID);
    } else if (srcDtype == ge::DT_FLOAT) {
        OP_CHECK(
            additionalDtype == ge::DT_FLOAT16 || additionalDtype == ge::DT_BF16 ||
                additionalDtype == ge::DT_FLOAT8_E4M3FN || additionalDtype == ge::DT_HIFLOAT8 ||
                additionalDtype == ge::DT_UINT8,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ACLNN_NAME, "additionalDtype", op::ToString(static_cast<op::DataType>(additionalDtype)).GetString(),
                "additionalDtype must be float16, bfloat16, float8_e4m3fn, hifloat8 or uint8 "
                "when the dtype of srcTensor is float32"),
            return ACLNN_ERR_PARAM_INVALID);
    }

    // WeightQuanBatchMatmul 场景拦截，仅支持srctensor的viewshape维度为2或3
    OP_CHECK(viewShapeDim == 2 || viewShapeDim == 3,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 ACLNN_NAME, "srcTensor", std::to_string(viewShapeDim),
                 "The srcTensor must be 2 or 3 when additionalDtype is not equal to srcTensor's dtype"),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(dstFormat == op::Format::FORMAT_FRACTAL_NZ,
             OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "dstFormat", std::to_string(dstFormat), "29(FORMAT_FRACTAL_NZ)"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckCalculateSizeAndFormatInputs(const aclTensor* srcTensor, [[maybe_unused]] const int dstFormat,
                                                     [[maybe_unused]] const int additionalDtype)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(IsRegBase() || socVersion == SocVersion::ASCEND910_93 || socVersion == SocVersion::ASCEND910B,
             OP_LOGW("Only support Ascend950/ASCEND910_93/ASCEND910B"), return ACLNN_ERR_RUNTIME_ERROR);
    OP_CHECK(srcTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "srcTensor is nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);

    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, srcTensor, WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static bool CheckEmptyTensor(const aclTensor* srcTensor)
{
    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    for (size_t i = 0; i < viewShapeDim; i++) {
        if (viewShape.GetDim(i) == 0) {
            return false;
        }
    }
    return true;
}

static aclnnStatus FallbackAclop(const aclTensor* tensor)
{
    if (!IsRegBase()) {
        bool emptyTensor = CheckEmptyTensor(tensor);
        if (!emptyTensor) {
            OP_LOGW("aclnn empty tensor is not supported, fallback aclop.");
            return ACLNN_ERR_RUNTIME_ERROR;
        }
        return ACLNN_SUCCESS;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus Check95NdToNzCalculateSizeAndFormatInputs(const aclTensor* srcTensor, const int dstFormat,
                                                             const int additionalDtype)
{
    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, srcTensor, ASCEND950_WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);
    [[maybe_unused]] op::Format srcFormat = srcTensor->GetStorageFormat();
    auto srcDtype = srcTensor->GetDataType();
    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    if (IsFloat8E8m0NdToNn(srcDtype, additionalDtype, srcFormat, static_cast<op::Format>(dstFormat))) {
        return CheckFloat8E8m0NnShape(viewShape);
    }
    // check input shape
    if (additionalDtype != static_cast<int>(srcDtype)) {
        aclnnStatus ret = ValidateWeightQuantMatmulParams(srcDtype, additionalDtype, viewShapeDim, dstFormat);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
    } else {
        if (IsNonQuantMatmulDtype(additionalDtype) && dstFormat == op::Format::FORMAT_FRACTAL_NZ) {
            aclnnStatus ret = ValidateNonQuantMatmulParams(additionalDtype, viewShape, viewShapeDim, dstFormat);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        } else {
            aclnnStatus ret = ValidateQuantMatmulParams(additionalDtype, viewShape, viewShapeDim);
            if (ret != ACLNN_SUCCESS) {
                return ret;
            }
        }
    }

    return ACLNN_SUCCESS;
}

static bool CheckFormatValid(DataType srcDtype, DataType dstDtype, op::Format srcFormat, op::Format dstFormat)
{
    if (IsQuantMatmulDtype(srcDtype, dstDtype)) {
        // QuantBatchMatmul 拦截场景
        OP_CHECK(CheckInputFormatSupportedToNz(srcFormat) && dstFormat == op::Format::FORMAT_FRACTAL_NZ,
                 OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                     ACLNN_NAME, "srcTensor, dstTensor",
                     std::string(op::ToString(srcFormat).GetString()) + ", " + op::ToString(dstFormat).GetString(),
                     "The format of srcTensor must be ND/NCL/NCHW/NCDHW and the format of dstTensor must be FRACTAL_NZ "
                     "when the dtype of srcTensor equals int8/uint8/float8_e4m3fn/hifloat8"),
                 return false);
    } else if (IsNonQuantMatmulDtype(srcDtype, dstFormat)) {
        // 非量化Matmul 拦截场景
        OP_CHECK((srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL) &&
                     dstFormat == op::Format::FORMAT_FRACTAL_NZ,
                 OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                     ACLNN_NAME, "srcTensor, dstTensor",
                     std::string(op::ToString(srcFormat).GetString()) + ", " + op::ToString(dstFormat).GetString(),
                     "The format of srcTensor must be ND/NCL and the format of dstTensor must be FRACTAL_NZ "
                     "when the dtype of srcTensor equals float16 or bfloat16"),
                 return false);
    } else {
        // WeightQuantBatchMatmul 拦截场景
        OP_CHECK((srcDtype == ge::DT_INT32 || srcDtype == ge::DT_FLOAT || srcDtype == ge::DT_FLOAT8_E4M3FN ||
                  srcDtype == ge::DT_HIFLOAT8 || srcDtype == ge::DT_UINT8) &&
                     srcFormat == op::Format::FORMAT_ND &&
                     (dstFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16 ||
                      dstFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32 || dstFormat == op::Format::FORMAT_FRACTAL_NZ),
                 OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                     ACLNN_NAME, "srcTensor, dstTensor",
                     std::string(op::ToString(srcFormat).GetString()) + ", " + op::ToString(dstFormat).GetString(),
                     "The format of srcTensor must be ND and "
                     "the format of dstTensor must be FRACTAL_NZ_C0_16/FRACTAL_NZ_C0_32/FRACTAL_NZ "
                     "when the dtype of srcTensor equals int32/float/float8_e4m3fn/hifloat8/uint8"),
                 return false);
    }
    return true;
}

static aclnnStatus CheckGetWorkSpaceSizeInputs(const aclTensor* srcTensor, aclTensor* dstTensor)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    OP_CHECK(IsRegBase() || socVersion == SocVersion::ASCEND910_93 || socVersion == SocVersion::ASCEND910B,
             OP_LOGW("Only support Ascend950/ASCEND910_93/ASCEND910B"), return ACLNN_ERR_RUNTIME_ERROR);
    OP_CHECK(IsContiguous(srcTensor),
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "srcTensor", "non-contiguous",
                                                   "srcTensor must be contiguous."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(IsContiguous(dstTensor),
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "dstTensor", "non-contiguous",
                                                   "dstTensor must be contiguous."),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, srcTensor, WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, dstTensor, WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus Check95NdToNzGetWorkSpaceSizeInputs(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, srcTensor, ASCEND950_WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, dstTensor, ASCEND950_WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);

    op::Format srcFormat = srcTensor->GetStorageFormat();
    auto srcViewShape = srcTensor->GetViewShape();
    auto srcviewShapeDim = srcViewShape.GetDimNum();
    DataType srcDtype = srcTensor->GetDataType();
    DataType dstDtype = dstTensor->GetDataType();
    op::Format dstFormat = dstTensor->GetStorageFormat();
    auto storageShape = dstTensor->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    if (IsFloat8E8m0NdToNn(srcDtype, static_cast<int>(dstDtype), srcFormat, dstFormat)) {
        auto ret = CheckFloat8E8m0NnShape(srcViewShape);
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        OP_CHECK(ValidFloat8E8m0NnDstShape(srcViewShape, storageShape),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of output Ascend affinity format tensor is invalid."),
                 return ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }
    if (!CheckFormatValid(srcDtype, dstDtype, srcFormat, dstFormat)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (IsQuantMatmulDtype(srcDtype, dstDtype) || IsNonQuantMatmulDtype(srcDtype, dstFormat)) {
        // QuantBatchMatmul &&
        // 非量化matmul仅支持srcTensor的viewshape维度为2到6，转换后的dstTensor的storageshape维度为4到8
        OP_CHECK(srcviewShapeDim >= DIMS_TWO && srcviewShapeDim <= DIMS_SIX && storageShapeDim >= DIMS_FOUR &&
                     storageShapeDim <= DIMS_EIGHT,
                 OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                     ACLNN_NAME, "srcTensor, dstTensor",
                     std::to_string(srcviewShapeDim) + ", " + std::to_string(storageShapeDim),
                     "The ViewShape dim of srcTensor must be between 2 and 6 and the StorageShape dim of dstTensor "
                     "must be between 4 and 8 "
                     "when the dtype of srcTensor equals the dtype of dstTensor"),
                 return ACLNN_ERR_PARAM_INVALID);
    } else {
        // WeightQuantBatchMatmul仅支持srcTensor的shape维度为2/3，转换后的dstTensor的shape的维度为4/5
        OP_CHECK(
            (srcviewShapeDim == DIMS_TWO || srcviewShapeDim == DIMS_THREE) &&
                (storageShapeDim == DIMS_FOUR || storageShapeDim == DIMS_FIVE),
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                ACLNN_NAME, "srcTensor, dstTensor",
                std::to_string(srcviewShapeDim) + ", " + std::to_string(storageShapeDim),
                "The ViewShape dim of srcTensor must be 2 or 3 and the StorageShape dim of dstTensor must be 4 or 5 "
                "when srcDtype is not equal to dstDtype"),
            return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static bool IsNzFormat(op::Format srcFormat)
{
    return srcFormat == op::Format::FORMAT_FRACTAL_NZ || srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_2 ||
           srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_4 || srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16 ||
           srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32;
}

static bool IsNz2Nd(op::Format srcFormat, op::Format dstFormat)
{
    // dstFormat是基础格式就认为是转ND.
    bool isDstNd = dstFormat == op::Format::FORMAT_ND || dstFormat == op::Format::FORMAT_NCL ||
                   dstFormat == op::Format::FORMAT_NCHW || dstFormat == op::Format::FORMAT_NCDHW;

    return IsNzFormat(srcFormat) && isDstNd;
}

static bool ValidNzShape(const aclTensor* srcTensor)
{
    auto srcStorageShape = srcTensor->GetStorageShape();
    auto srcStorageShapeDim = srcStorageShape.GetDimNum();
    // 仅支持输入Nz格式tensor的storageShape维度在4到8之间
    OP_CHECK(srcStorageShapeDim >= DIMS_FOUR && srcStorageShapeDim <= DIMS_EIGHT,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                 ACLNN_NAME, "srcTensor", std::to_string(srcStorageShapeDim).c_str(),
                 "The StorageShape dim of srcTensor must be between 4 and 8 for input tensor in Nz format"),
             return false);
    // 非空tensor的NZ storageShape不允许含0；空tensor保留由逻辑shape推导出的0维
    if (!srcTensor->IsEmpty()) {
        for (uint64_t i = 0; i < srcStorageShapeDim; ++i) {
            OP_CHECK(srcStorageShape[i] != 0,
                     OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                         ACLNN_NAME, "srcTensor", op::ToString(srcStorageShape).GetString(),
                         "Each axis of storageShape must not be 0 when srcTensor is not empty"),
                     return false);
        }
    }
    // c0需要和shape匹配
    int64_t c0 = static_cast<int64_t>(srcStorageShape.GetDim(srcStorageShapeDim - 1)); // 倒数第1维为Nz shape的C0轴
    int64_t expectedC0 = -1;
    op::Format srcFormat = srcTensor->GetStorageFormat();
    if (srcFormat == op::Format::FORMAT_FRACTAL_NZ) {
        auto srcDataType = srcTensor->GetDataType();
        expectedC0 = srcDataType == ge::DT_FLOAT4_E2M1 ? FRACTAL_NZ_C0_4B :
                                                         BLOCK_SIZE / ge::GetSizeByDataType(srcDataType);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_2) {
        expectedC0 = 2; // c0 should be 2
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_4) {
        expectedC0 = 4; // c0 should be 4
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_16) {
        expectedC0 = 16; // c0 should be 16
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_NZ_C0_32) {
        expectedC0 = 32; // c0 should be 32
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(
            ACLNN_NAME, "srcTensor", op::ToString(srcFormat).GetString(),
            "[FORMAT_FRACTAL_NZ, FORMAT_FRACTAL_NZ_C0_2, FORMAT_FRACTAL_NZ_C0_4, FORMAT_FRACTAL_NZ_C0_16, "
            "FORMAT_FRACTAL_NZ_C0_32]");
        return false;
    }
    OP_CHECK(c0 == expectedC0,
             OP_LOGE_FOR_INVALID_VALUE(ACLNN_NAME, "c0", std::to_string(c0), std::to_string(expectedC0)), return false);

    return true;
}

static bool ValidNz2NdShape(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    auto srcStorageShape = srcTensor->GetStorageShape();
    auto dstViewShape = dstTensor->GetViewShape();
    auto srcStorageShapeDim = srcStorageShape.GetDimNum();
    auto dstViewShapeDim = dstViewShape.GetDimNum();
    // Nz维度比ND维度多2
    OP_CHECK(dstViewShapeDim + DIMS_TWO == srcStorageShapeDim,
             OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                 ACLNN_NAME, "srcTensor, dstTensor",
                 std::to_string(srcStorageShapeDim) + ", " + std::to_string(dstViewShapeDim),
                 "Nz-dimensions must be equal to ND-dimensions plus 2"),
             return false);
    OP_CHECK(dstViewShapeDim >= 2,
             OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "dstTensor", std::to_string(dstViewShapeDim).c_str(),
                                                      "The ViewShape dim of ND must be larger than or equal to 2"),
             return false);
    // 需要Nz shape[0:-4] == ND shape[0:-2]
    for (uint64_t i = 0; i < dstViewShapeDim - DIMS_TWO; ++i) {
        if (srcStorageShape.GetDim(i) != dstViewShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                ACLNN_NAME, "srcTensor, dstTensor",
                std::string(op::ToString(srcStorageShape).GetString()) + ", " + op::ToString(dstViewShape).GetString(),
                "ND dimensions except last 2 must match Nz dimensions except last 4");
            return false;
        }
    }
    // 需要满足 n1 = ceil(n/n0), c1 = ceil(c/c0)
    int64_t n = static_cast<int64_t>(dstViewShape.GetDim(dstViewShapeDim - 2));        // 倒数第2维为ND的n轴
    int64_t c = static_cast<int64_t>(dstViewShape.GetDim(dstViewShapeDim - 1));        // 倒数第1维为ND的c轴
    int64_t c1 = static_cast<int64_t>(srcStorageShape.GetDim(srcStorageShapeDim - 4)); // 倒数第4维为Nz的c1轴
    int64_t n1 = static_cast<int64_t>(srcStorageShape.GetDim(srcStorageShapeDim - 3)); // 倒数第3维为Nz的n1轴
    int64_t n0 = static_cast<int64_t>(srcStorageShape.GetDim(srcStorageShapeDim - 2)); // 倒数第2维为Nz的n0轴
    int64_t c0 = static_cast<int64_t>(srcStorageShape.GetDim(srcStorageShapeDim - 1)); // 倒数第1维为Nz的c0轴
    OP_CHECK(n1 == CeilDiv(n, n0) && c1 == CeilDiv(c, c0),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 ACLNN_NAME, "dstTensor, srcTensor",
                 "(n, c) = (" + std::to_string(n) + ", " + std::to_string(c) +
                     "), "
                     "(c1, n1, n0, c0) = (" +
                     std::to_string(c1) + ", " + std::to_string(n1) + ", " + std::to_string(n0) + ", " +
                     std::to_string(c0) + ")",
                 "n1 must be equal to ceil(n/n0) and c1 must be equal to ceil(c/c0) when converting Nz to ND"),
             return false);

    return true;
}

static bool ValidDtypeFormatForNz2Nd(const aclTensor* srcTensor)
{
    DataType srcDtype = srcTensor->GetDataType();
    op::Format srcFormat = srcTensor->GetStorageFormat();
    std::pair<DataType, op::Format> targetPair = {srcDtype, srcFormat};
    auto it = std::find(INPUT_DTYPE_FORMAT_NZ_TO_ND_SUPPORT_LIST.begin(),
                        INPUT_DTYPE_FORMAT_NZ_TO_ND_SUPPORT_LIST.end(), targetPair);
    OP_CHECK(it != INPUT_DTYPE_FORMAT_NZ_TO_ND_SUPPORT_LIST.end(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support converting [%s] to ND when srcDtype is [%s].",
                     op::ToString(srcFormat).GetString(), op::ToString(srcDtype).GetString()),
             return false);
    return true;
}

static aclnnStatus Check95NzToNdGetWorkSpaceSizeInputs(const aclTensor* srcTensor, const aclTensor* dstTensor)
{
    // check dtype
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, srcTensor, ASCEND950_WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, dstTensor, ASCEND950_WEIGHT_DTYPE_SUPPORT_LIST,
                                           return ACLNN_ERR_PARAM_INVALID);
    // check the shape of input tensor matches Nz format
    OP_CHECK(ValidNzShape(srcTensor), OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of input Nz tensor is invalid."),
             return ACLNN_ERR_PARAM_INVALID);
    // check the shape of output ND tensor matches the input Nz tensor
    OP_CHECK(ValidNz2NdShape(srcTensor, dstTensor),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of ND output does not match the shape of Nz input."),
             return ACLNN_ERR_PARAM_INVALID);
    // check the dtype and format of input
    OP_CHECK(ValidDtypeFormatForNz2Nd(srcTensor),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid format and data type for NZ-to-ND conversion"),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

bool IsSupportedTransdataForwardPairRegBase(op::Format src, op::Format dst)
{
    return kTransdataForwardFormatPairsRegBase.count({src, dst});
}

bool IsSupportedTransdataForwardPair910B(op::Format src, op::Format dst)
{
    return kTransdataForwardFormatPairs910B.count({src, dst});
}

aclnnStatus CalcNdToNz(const aclTensor* srcTensor, int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
                       int* actualFormat)
{
    DataType srcDtype = srcTensor->GetDataType();
    int64_t c0 = 16; // 默认NZ分型的c0为16
    if (!IsRegBase()) {
        aclnnStatus ret = FallbackAclop(srcTensor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        c0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
        *actualFormat = op::Format::FORMAT_FRACTAL_NZ;
    }
    // 根据A矩阵和B矩阵数据类型计算实际转换成NZ格式后的C0大小
    if (IsRegBase()) {
        if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
            // 当前要求C0 * ge::GetSizeByDataType(dtype) = 32B
            c0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
            if (additionalDtype == ge::DT_FLOAT4_E2M1 || additionalDtype == ge::DT_FLOAT4_E1M2) {
                c0 = FRACTAL_NZ_C0_4B;
            }
            *actualFormat = op::Format::FORMAT_FRACTAL_NZ;
        } else {
            // 当A矩阵数据类型大小为2B，C0 = 16
            if (ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype)) == 2) {
                *actualFormat = op::Format::FORMAT_FRACTAL_NZ_C0_16;
            }

            if (static_cast<op::DataType>(additionalDtype) == DataType::DT_FLOAT8_E4M3FN ||
                static_cast<op::DataType>(additionalDtype) == DataType::DT_INT8) {
                // A8场景，B32/1B=BLOCK_SIZE
                c0 = BLOCK_SIZE;
                *actualFormat = op::Format::FORMAT_FRACTAL_NZ_C0_32;
            }
        }
    }

    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    int64_t srcTensorDimFirst = viewShape.GetDim(viewShapeDim - 2); // 倒数第2维为ND的shape第一维
    int64_t srcTensorDimLast = viewShape.GetDim(viewShapeDim - 1);  // 倒数第1维为ND的shape第二维
    *dstShapeSize = static_cast<uint64_t>(srcTensor->GetViewShape().GetDimNum()) +
                    2; // NZ维度大小固定为srcTensor的维度+2

    // 非报错情况下申请dstShape数组内存，由上层调用者释放
    try {
        *dstShape = new int64_t[*dstShapeSize]();
    } catch (std::bad_alloc& e) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for the NZ");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    for (uint64_t i = 0; i < *dstShapeSize; i++) {
        if (i == *dstShapeSize - 4) {        // 修改最后4维的NZ数据
            (*dstShape)[*dstShapeSize - 4] = // 当前Nz分型倒数第4维大小为CeilDiv(srcTensorDimLast, c0)
                CeilDiv(srcTensorDimLast, c0);
            (*dstShape)[*dstShapeSize - 3] = // 当前Nz分型倒数第3维大小固定为CeilDiv(srcTensorDimFirst, 16)
                CeilDiv(srcTensorDimFirst, 16);
            (*dstShape)[*dstShapeSize - 2] = 16; // 当前Nz分型倒数第2维大小固定为16
            (*dstShape)[*dstShapeSize - 1] = c0; // 当前Nz分型倒数第1维大小为C0
            break;
        }
        (*dstShape)[i] = viewShape.GetDim(i);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus CalcFloat8E8m0NdToNn(const aclTensor* srcTensor, int64_t** dstShape, uint64_t* dstShapeSize,
                                 int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    auto ret = CheckFloat8E8m0NnShape(viewShape);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGW("Failed to check FLOAT8_E8M0 Ascend affinity format shape"), return ret);

    int64_t kScale = viewShape.GetDim(viewShapeDim - DIMS_THREE);
    int64_t n = viewShape.GetDim(viewShapeDim - DIMS_TWO);
    int64_t n1 = CeilDiv(n, NUM_SIXTEEN);
    *dstShapeSize = static_cast<uint64_t>(viewShapeDim + NUM_ONE);
    try {
        *dstShape = new int64_t[*dstShapeSize]();
    } catch (std::bad_alloc& e) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for the Ascend affinity format");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    for (uint64_t i = 0; i < *dstShapeSize; ++i) {
        if (i < viewShapeDim - DIMS_THREE) {
            (*dstShape)[i] = viewShape.GetDim(i);
        } else if (i == viewShapeDim - DIMS_THREE) {
            (*dstShape)[i] = n1;
        } else if (i == viewShapeDim - DIMS_TWO) {
            (*dstShape)[i] = kScale;
        } else if (i == viewShapeDim - NUM_ONE) {
            (*dstShape)[i] = NUM_SIXTEEN;
        } else {
            (*dstShape)[i] = NUM_TWO;
        }
    }
    *actualFormat = op::Format::FORMAT_FRACTAL_NZ;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNd(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                     uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto viewShape = srcTensor->GetViewShape();
    auto shapeDim = viewShape.GetDimNum();
    *dstShapeSize = shapeDim;
    std::vector<int64_t> shape(shapeDim);
    for (size_t i = 0; i < shapeDim; ++i) {
        shape[i] = viewShape.GetDim(i);
    }

    try {
        *dstShape = new int64_t[shapeDim];
        for (size_t i = 0; i < shapeDim; ++i) {
            (*dstShape)[i] = shape[i];
        }
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for ND");
        return ACLNN_ERR_RUNTIME_ERROR;
    }

    *actualFormat = op::Format::FORMAT_ND;
    return ACLNN_SUCCESS;
}

// 根据A矩阵和B矩阵数据类型计算实际转换成私有格式后的C0大小
static int64_t GetC0ByAdditionalDtype(const aclTensor* srcTensor, int additionalDtype)
{
    int64_t C0 = C0_SIZE;
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    return C0;
}

// 申请dstShape数组内存，由上层调用者释放
static aclnnStatus AllocDstShape(int64_t** dstShape, uint64_t* dstShapeSize, const std::initializer_list<int64_t>& dims,
                                 const char* formatName)
{
    *dstShapeSize = dims.size();
    try {
        *dstShape = new int64_t[dims.size()];
        uint64_t idx = 0;
        for (auto dim : dims) {
            (*dstShape)[idx] = dim;
            ++idx;
        }
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for %s", formatName);
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCDHWToNDC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                                uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);
    int64_t C1 = CeilDiv(C, C0);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, D, C1, H, W, C0}, "NDC1HWC0"); // 6HD
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NDC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNDHWCToNDC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                                uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t D = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C = viewShape.GetDim(4);
    int64_t C1 = CeilDiv(C, C0);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, D, C1, H, W, C0}, "NDC1HWC0"); // 6HD
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NDC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCHWToNC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                              uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, C1, H, W, C0}, "NC1HWC0"); // 5HD
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNHWCToNC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                              uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, C1, H, W, C0}, "NC1HWC0"); // 5HD
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCHWToFRACTALZ(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                               uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    int64_t N0 = N0_SIZE; // 私有格式的分形要求
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    ret = AllocDstShape(dstShape, dstShapeSize, {C1 * H * W, N1, N0, C0}, "FRACTAL_Z"); // FRACTAL_Z
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_FRACTAL_Z;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcHWCNToFRACTALZ(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                               uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    int64_t N0 = N0_SIZE; // 私有格式的分形要求
    auto viewShape = srcTensor->GetViewShape();
    int64_t H = viewShape.GetDim(0);
    int64_t W = viewShape.GetDim(1);
    int64_t C = viewShape.GetDim(2);
    int64_t N = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    ret = AllocDstShape(dstShape, dstShapeSize, {C1 * H * W, N1, N0, C0}, "FRACTAL_Z"); // FRACTAL_Z
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_FRACTAL_Z;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCDHWToFZ3D(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                            uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);
    int64_t N0 = 16; // 私有格式的分形要求
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    ret = AllocDstShape(dstShape, dstShapeSize, {D * C1 * H * W, N1, N0, C0}, "FZ3D"); // FZ3D
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_FRACTAL_Z_3D;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcDHWCNToFZ3D(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                            uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t C0 = GetC0ByAdditionalDtype(srcTensor, additionalDtype);
    int64_t N0 = N0_SIZE; // 私有格式的分形要求
    auto viewShape = srcTensor->GetViewShape();
    int64_t D = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);
    int64_t N = viewShape.GetDim(4);
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    ret = AllocDstShape(dstShape, dstShapeSize, {D * C1 * H * W, N1, N0, C0}, "FZ3D"); // FZ3D
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_FRACTAL_Z_3D;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNCDHW(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                        uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (srcTensor->GetViewShape().GetDimNum() != DIMS_FIVE) {
        OP_LOGW("The ViewShape dim of srcTensor must be 5 when converting to NCDHW, fallback to aclop.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, C, D, H, W}, "NCDHW"); // 5HD
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NCDHW;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNCHW(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                       uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (srcTensor->GetViewShape().GetDimNum() != DIMS_FOUR) {
        OP_LOGW("The ViewShape dim of srcTensor must be 4 when converting to NCHW, fallback to aclop.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, C, H, W}, "NCHW"); // NCHW
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NCHW;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNHWC(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                       uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (srcTensor->GetViewShape().GetDimNum() != DIMS_FOUR) {
        OP_LOGW("The ViewShape dim of srcTensor must be 4 when converting to NHWC, fallback to aclop.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, C, H, W}, "NHWC"); // NHWC
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NHWC;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToHWCN(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                       uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (srcTensor->GetViewShape().GetDimNum() != DIMS_FOUR) {
        OP_LOGW("The ViewShape dim of srcTensor must be 4 when converting to HWCN, fallback to aclop.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t H = viewShape.GetDim(0);
    int64_t W = viewShape.GetDim(1);
    int64_t C = viewShape.GetDim(2);
    int64_t N = viewShape.GetDim(3);

    ret = AllocDstShape(dstShape, dstShapeSize, {H, W, C, N}, "HWCN"); // HWCN
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_HWCN;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToDHWCN(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                        uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (srcTensor->GetViewShape().GetDimNum() != DIMS_FIVE) {
        OP_LOGW("The ViewShape dim of srcTensor must be 5 when converting to DHWCN, fallback to aclop.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t D = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);
    int64_t N = viewShape.GetDim(4);

    ret = AllocDstShape(dstShape, dstShapeSize, {D, H, W, C, N}, "DHWCN"); // DHWCN
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_DHWCN;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNDHWC(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                        uint64_t* dstShapeSize, int* actualFormat)
{
    aclnnStatus ret = FallbackAclop(srcTensor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (srcTensor->GetViewShape().GetDimNum() != DIMS_FIVE) {
        OP_LOGW("The ViewShape dim of srcTensor must be 5 when converting to NDHWC, fallback to aclop.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t D = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C = viewShape.GetDim(4);

    ret = AllocDstShape(dstShape, dstShapeSize, {N, D, H, W, C}, "NDHWC"); // NDHWC
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *actualFormat = op::Format::FORMAT_NDHWC;
    return ACLNN_SUCCESS;
}

const aclTensor* TransFloat8E8m0NdToNn(const aclTensor* srcTensor, aclOpExecutor* executor)
{
    auto viewShape = srcTensor->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    int64_t kScale = viewShape.GetDim(viewShapeDim - DIMS_THREE);
    int64_t n = viewShape.GetDim(viewShapeDim - DIMS_TWO);
    int64_t n1 = CeilDiv(n, NUM_SIXTEEN);
    int64_t alignedN = n1 * NUM_SIXTEEN;
    auto srcByteView = executor->CreateView(const_cast<aclTensor*>(srcTensor), viewShape, srcTensor->GetViewOffset());
    CHECK_RET(srcByteView != nullptr, nullptr);
    srcByteView->SetDataType(op::DataType::DT_INT8);
    const aclTensor* tensorToReshape = srcByteView;
    if (alignedN != n) {
        std::vector<int64_t> paddings(viewShapeDim * DIMS_TWO, 0);
        size_t nAxis = viewShapeDim - DIMS_TWO;
        paddings[nAxis * DIMS_TWO + NUM_ONE] = alignedN - n;
        auto paddingsArray = executor->AllocIntArray(paddings.data(), paddings.size());
        auto paddingsTensor = executor->ConvertToTensor(paddingsArray, op::DataType::DT_INT64);
        CHECK_RET(paddingsTensor != nullptr, nullptr);
        tensorToReshape = l0op::Pad(srcByteView, paddingsTensor, executor);
        CHECK_RET(tensorToReshape != nullptr, nullptr);
    }

    std::vector<int64_t> reshapeShape;
    reshapeShape.reserve(viewShapeDim + NUM_ONE);
    for (size_t i = 0; i < viewShapeDim - DIMS_THREE; ++i) {
        reshapeShape.emplace_back(viewShape.GetDim(i));
    }
    reshapeShape.emplace_back(kScale);
    reshapeShape.emplace_back(n1);
    reshapeShape.emplace_back(NUM_SIXTEEN);
    reshapeShape.emplace_back(NUM_TWO);

    auto reshapeArray = executor->AllocIntArray(reshapeShape.data(), reshapeShape.size());
    auto reshapeTensor = l0op::Reshape(tensorToReshape, reshapeArray, executor);
    CHECK_RET(reshapeTensor != nullptr, nullptr);

    std::vector<int64_t> perm(reshapeShape.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        perm[i] = static_cast<int64_t>(i);
    }
    size_t kScaleAxis = viewShapeDim - DIMS_THREE;
    size_t n1Axis = viewShapeDim - DIMS_TWO;
    std::swap(perm[kScaleAxis], perm[n1Axis]);
    auto permArray = executor->AllocIntArray(perm.data(), perm.size());
    auto transposeTensor = const_cast<aclTensor*>(l0op::Transpose(reshapeTensor, permArray, executor));
    CHECK_RET(transposeTensor != nullptr, nullptr);
    return transposeTensor;
}

aclnnStatus aclnnNpuFormatCastCalculateSizeAndFormat(const aclTensor* srcTensor, const int dstFormat,
                                                     int additionalDtype, int64_t** dstShape, uint64_t* dstShapeSize,
                                                     int* actualFormat)
{
    auto ret = CheckCalculateSizeAndFormatInputs(srcTensor, dstFormat, additionalDtype);
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGW("Failed to check inputs"), return ACLNN_ERR_PARAM_INVALID);
    op::Format srcFormat = srcTensor->GetStorageFormat();
    if (additionalDtype == -1) {
        additionalDtype = static_cast<int>(srcTensor->GetDataType());
    }
    if (IsFloat8E8m0NdToNn(srcTensor->GetDataType(), additionalDtype, srcFormat, static_cast<op::Format>(dstFormat))) {
        OP_CHECK(IsRegBase(),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FLOAT8_E8M0 ND to Ascend affinity format only supports Ascend950."),
                 return ACLNN_ERR_PARAM_INVALID);
        auto retNdToNn = Check95NdToNzCalculateSizeAndFormatInputs(srcTensor, dstFormat, additionalDtype);
        OP_CHECK(retNdToNn == ACLNN_SUCCESS, OP_LOGW("Failed to check inputs"), return ACLNN_ERR_PARAM_INVALID);
        return CalcFloat8E8m0NdToNn(srcTensor, dstShape, dstShapeSize, actualFormat);
    }
    if (dstFormat == op::Format::FORMAT_FRACTAL_NZ &&
        ((srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL) ||
         (additionalDtype == static_cast<int>(srcTensor->GetDataType()) && CheckInputFormatSupportedToNz(srcFormat)))) {
        // ASCEND950校验特殊场景
        if (IsRegBase()) {
            auto retNdToNz = Check95NdToNzCalculateSizeAndFormatInputs(srcTensor, dstFormat, additionalDtype);
            OP_CHECK(retNdToNz == ACLNN_SUCCESS, OP_LOGW("Failed to check inputs"), return ACLNN_ERR_PARAM_INVALID);
        }
        return CalcNdToNz(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (IsNzFormat(srcFormat) && dstFormat == op::Format::FORMAT_ND) {
        if (!IsRegBase()) {
            OP_CHECK(srcFormat == op::Format::FORMAT_FRACTAL_NZ,
                     OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "srcTensor", op::ToString(srcFormat).GetString(),
                                                "FORMAT_FRACTAL_NZ"),
                     return ACLNN_ERR_PARAM_INVALID);
        } else {
            // ASCEND950校验特殊场景
            OP_CHECK(ValidNzShape(srcTensor),
                     OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of input Nz tensor is invalid."),
                     return ACLNN_ERR_PARAM_INVALID);
        }
        return CalcToNd(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NCDHW && dstFormat == op::Format::FORMAT_NDC1HWC0) {
        return CalcNCDHWToNDC1HWC0(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NDC1HWC0 && dstFormat == op::Format::FORMAT_NCDHW) {
        return CalcToNCDHW(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NCDHW && dstFormat == op::Format::FORMAT_FRACTAL_Z_3D) {
        return CalcNCDHWToFZ3D(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_Z_3D && dstFormat == op::Format::FORMAT_NCDHW) {
        return CalcToNCDHW(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NCHW && dstFormat == op::Format::FORMAT_NC1HWC0) {
        return CalcNCHWToNC1HWC0(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NC1HWC0 && dstFormat == op::Format::FORMAT_NCHW) {
        return CalcToNCHW(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NHWC && dstFormat == op::Format::FORMAT_NC1HWC0) {
        return CalcNHWCToNC1HWC0(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NC1HWC0 && dstFormat == op::Format::FORMAT_NHWC) {
        return CalcToNHWC(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NCHW && dstFormat == op::Format::FORMAT_FRACTAL_Z) {
        return CalcNCHWToFRACTALZ(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_Z && dstFormat == op::Format::FORMAT_NCHW) {
        return CalcToNCHW(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_HWCN && dstFormat == op::Format::FORMAT_FRACTAL_Z) {
        return CalcHWCNToFRACTALZ(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_Z && dstFormat == op::Format::FORMAT_HWCN) {
        return CalcToHWCN(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_DHWCN && dstFormat == op::Format::FORMAT_FRACTAL_Z_3D) {
        return CalcDHWCNToFZ3D(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_FRACTAL_Z_3D && dstFormat == op::Format::FORMAT_DHWCN) {
        return CalcToDHWCN(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NDHWC && dstFormat == op::Format::FORMAT_NDC1HWC0) {
        return CalcNDHWCToNDC1HWC0(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    } else if (srcFormat == op::Format::FORMAT_NDC1HWC0 && dstFormat == op::Format::FORMAT_NDHWC) {
        return CalcToNDHWC(srcTensor, additionalDtype, dstShape, dstShapeSize, actualFormat);
    }
    OP_LOGW("aclnnNpuFormatCastCalculateSizeAndFormat unsupported format transformation");
    return ACLNN_ERR_RUNTIME_ERROR;
}

aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor* srcTensor, aclTensor* dstTensor,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNpuFormatCast, DFX_IN(srcTensor), DFX_OUT(dstTensor));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(srcTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dstTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ret = CheckGetWorkSpaceSizeInputs(srcTensor, dstTensor);
    op::Format srcFormat = srcTensor->GetStorageFormat();
    op::Format dstFormat = dstTensor->GetStorageFormat();
    bool isFloat8E8m0Nn = IsFloat8E8m0NdToNn(srcTensor->GetDataType(), static_cast<int>(dstTensor->GetDataType()),
                                             srcFormat, dstFormat);
    // ASCEND950校验特殊场景
    if (IsRegBase()) {
        if (dstFormat == op::Format::FORMAT_FRACTAL_NZ &&
            ((srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL) ||
             (srcTensor->GetDataType() == dstTensor->GetDataType() && CheckInputFormatSupportedToNz(srcFormat)))) {
            ret = Check95NdToNzGetWorkSpaceSizeInputs(srcTensor, dstTensor);
        } else if (IsNz2Nd(srcFormat, dstFormat)) {
            ret = Check95NzToNdGetWorkSpaceSizeInputs(srcTensor, dstTensor);
        }
    } else {
        // check输入参数合法性 910B场景
        CHECK_RET(CheckDtypeValid(srcTensor, dstTensor) != false, ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to check aclnnNpuFormatCastGetWorkSpaceSizeInputs."),
             return ACLNN_ERR_PARAM_INVALID);

    const bool isNdToNz = IsNzFormat(dstFormat) &&
                          ((srcFormat == op::Format::FORMAT_ND || srcFormat == op::Format::FORMAT_NCL) ||
                           (srcTensor->GetDataType() == dstTensor->GetDataType() &&
                            CheckInputFormatSupportedToNz(srcFormat)));
    const bool isNzToNd = IsNz2Nd(srcFormat, dstFormat);
    if (IsRegBase() && srcTensor->IsEmpty() && (isNdToNz || isNzToNd)) {
        OP_CHECK(dstTensor->IsEmpty(),
                 OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "dstTensor",
                                                       op::ToString(dstTensor->GetViewShape()).GetString(),
                                                       "dstTensor must be empty when srcTensor is empty"),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK(
            srcTensor->GetViewShape() == dstTensor->GetViewShape(),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "srcTensor, dstTensor",
                                                   std::string(op::ToString(srcTensor->GetViewShape()).GetString()) +
                                                       ", " + op::ToString(dstTensor->GetViewShape()).GetString(),
                                                   "srcTensor and dstTensor must have the same view shape"),
            return ACLNN_ERR_PARAM_INVALID);
        const aclTensor* nzTensor = isNdToNz ? dstTensor : srcTensor;
        const aclTensor* ndTensor = isNdToNz ? srcTensor : dstTensor;
        OP_CHECK(ValidNzShape(nzTensor),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The storage shape of empty Nz tensor is invalid."),
                 return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK(ValidNz2NdShape(nzTensor, ndTensor),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "The storage shape of empty Nz tensor does not match the view shape of empty ND tensor."),
                 return ACLNN_ERR_PARAM_INVALID);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    aclTensor* formatTensor;
    if (IsRegBase()) {
        formatTensor = const_cast<aclTensor*>(srcTensor);
        // 适配srcFormat为NCL的场景
        if (isFloat8E8m0Nn) {
            formatTensor->SetViewFormat(op::Format::FORMAT_ND);
            formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
            formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
            formatTensor->SetOriginalShape(srcTensor->GetViewShape());
            formatTensor->SetStorageShape(srcTensor->GetViewShape());

            dstTensor->SetViewFormat(op::Format::FORMAT_ND);
            dstTensor->SetViewShape(srcTensor->GetViewShape());
            dstTensor->SetOriginalFormat(op::Format::FORMAT_ND);
            dstTensor->SetOriginalShape(srcTensor->GetOriginalShape());
        } else if ((IsQuantMatmulDtype(srcTensor->GetDataType(), dstTensor->GetDataType()) &&
                    dstFormat == op::Format::FORMAT_FRACTAL_NZ) ||
                   srcFormat == op::Format::FORMAT_NCL) {
            formatTensor->SetViewFormat(op::Format::FORMAT_ND);
            formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
            formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
        }
        if (IsQuantMatmulDtype(srcTensor->GetDataType(), dstTensor->GetDataType()) &&
            dstFormat == op::Format::FORMAT_FRACTAL_NZ) {
            formatTensor->SetOriginalShape(srcTensor->GetViewShape());
            formatTensor->SetStorageShape(srcTensor->GetViewShape());

            dstTensor->SetViewFormat(op::Format::FORMAT_ND);
            dstTensor->SetViewShape(srcTensor->GetViewShape());
            dstTensor->SetOriginalFormat(op::Format::FORMAT_ND);
            dstTensor->SetOriginalShape(srcTensor->GetOriginalShape());
        } else if (IsSupportedTransdataForwardPairRegBase(srcFormat, dstFormat)) {
            // 适配srcFormat为NCL的场景
            // 公有转私有Format
            formatTensor->SetOriginalShape(srcTensor->GetViewShape());
            formatTensor->SetStorageShape(srcTensor->GetViewShape());

            dstTensor->SetViewFormat(srcTensor->GetViewFormat());
            dstTensor->SetViewShape(srcTensor->GetViewShape());
            dstTensor->SetOriginalFormat(srcTensor->GetOriginalFormat());
            dstTensor->SetOriginalShape(srcTensor->GetOriginalShape());
        } else {
            // 私有转公有Format
            if (IsRegBase() && IsNz2Nd(srcFormat, dstFormat)) {
                dstTensor->SetStorageFormat(op::Format::FORMAT_ND);
            }
            dstTensor->SetOriginalShape(dstTensor->GetViewShape());
            dstTensor->SetStorageShape(dstTensor->GetViewShape());

            formatTensor->SetViewFormat(dstTensor->GetViewFormat());
            formatTensor->SetViewShape(dstTensor->GetViewShape());
            formatTensor->SetOriginalFormat(dstTensor->GetOriginalFormat());
            formatTensor->SetOriginalShape(dstTensor->GetOriginalShape());
        }
    } else {
        formatTensor = uniqueExecutor.get()->CreateView(srcTensor, srcTensor->GetViewShape(),
                                                        srcTensor->GetViewOffset());
        if (IsSupportedTransdataForwardPair910B(srcFormat, dstFormat)) {
            // 适配srcFormat为NCL的场景
            formatTensor->SetOriginalShape(srcTensor->GetViewShape());
            formatTensor->SetStorageShape(srcTensor->GetViewShape());
            dstTensor->SetViewFormat(srcTensor->GetViewFormat());
            dstTensor->SetViewShape(srcTensor->GetViewShape());
            dstTensor->SetOriginalFormat(srcTensor->GetOriginalFormat());
            dstTensor->SetOriginalShape(srcTensor->GetOriginalShape());
            // 公有转私有Format
            if (dstFormat == op::Format::FORMAT_FRACTAL_NZ) {
                formatTensor->SetViewFormat(op::Format::FORMAT_ND);
                formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
                formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
            }
        } else if (IsSupportedTransdataForwardPair910B(dstFormat, srcFormat)) {
            // 适配srcFormat为NCL的场景
            // 私有转公有Format
            formatTensor->SetStorageFormat(srcTensor->GetStorageFormat());
            formatTensor->SetStorageShape(srcTensor->GetStorageShape());
            formatTensor->SetViewFormat(dstTensor->GetViewFormat());
            formatTensor->SetViewShape(dstTensor->GetViewShape());
            formatTensor->SetOriginalFormat(dstTensor->GetViewFormat());
            formatTensor->SetOriginalShape(dstTensor->GetViewShape());
            if (srcFormat == op::Format::FORMAT_FRACTAL_NZ) {
                dstTensor->SetStorageFormat(op::Format::FORMAT_ND);
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported format transformation from [%s] to [%s]",
                    op::ToString(srcFormat).GetString(), op::ToString(dstFormat).GetString());
            return false;
        }
    }
    aclTensor* outTensor;
    int64_t dstDimNum = dstTensor->GetStorageShape().GetDimNum();
    if (isFloat8E8m0Nn) {
        outTensor = const_cast<aclTensor*>(TransFloat8E8m0NdToNn(formatTensor, uniqueExecutor.get()));
    } else if (dstTensor->GetStorageShape().GetDim(dstDimNum - NUM_ONE) == NUM_SIXTEEN &&
               ge::GetSizeByDataType(srcTensor->GetDataType()) >= NUM_FOUR && !IsRegBase()) {
        outTensor = const_cast<aclTensor*>(
            l0op::TransDataSpecial(formatTensor, dstTensor->GetStorageFormat(), 1, uniqueExecutor.get()));
    } else {
        outTensor = const_cast<aclTensor*>(
            l0op::TransData(formatTensor, dstTensor->GetStorageFormat(), 1, uniqueExecutor.get()));
    }
    CHECK_RET(outTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outTensor->SetViewFormat(dstTensor->GetViewFormat());
    outTensor->SetViewShape(dstTensor->GetViewShape());
    outTensor->SetOriginalFormat(dstTensor->GetOriginalFormat());
    outTensor->SetOriginalShape(dstTensor->GetOriginalShape());
    outTensor->SetStorageFormat(dstTensor->GetStorageFormat());
    outTensor->SetStorageShape(dstTensor->GetStorageShape());
    aclTensor* dstCopyTensor = dstTensor;
    if (isFloat8E8m0Nn) {
        dstCopyTensor = uniqueExecutor.get()->CreateView(dstTensor, dstTensor->GetViewShape(),
                                                         dstTensor->GetViewOffset());
        CHECK_RET(dstCopyTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        dstCopyTensor->SetDataType(op::DataType::DT_INT8);
        dstCopyTensor->SetViewFormat(dstTensor->GetViewFormat());
        dstCopyTensor->SetViewShape(dstTensor->GetViewShape());
        dstCopyTensor->SetOriginalFormat(dstTensor->GetOriginalFormat());
        dstCopyTensor->SetOriginalShape(dstTensor->GetOriginalShape());
        dstCopyTensor->SetStorageFormat(dstTensor->GetStorageFormat());
        dstCopyTensor->SetStorageShape(dstTensor->GetStorageShape());
    }
    auto viewCopyResult = l0op::ViewCopy(outTensor, dstCopyTensor, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuFormatCast(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNpuFormatCast);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

} // namespace
#ifdef __cplusplus
}
#endif
