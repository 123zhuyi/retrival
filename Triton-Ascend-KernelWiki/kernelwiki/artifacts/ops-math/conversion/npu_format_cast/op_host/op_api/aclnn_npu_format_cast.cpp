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
    DataType::DT_UINT8, DataType::DT_FLOAT4_E1M2};

static const std::initializer_list<DataType> WEIGHT_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT8,        DataType::DT_UINT8,    DataType::DT_FLOAT,      DataType::DT_FLOAT16,
    DataType::DT_BF16,        DataType::DT_INT32,    DataType::DT_UINT32,     DataType::DT_FLOAT8_E4M3FN,
    DataType::DT_FLOAT4_E2M1, DataType::DT_HIFLOAT8, DataType::DT_FLOAT4_E1M2};

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

inline int64_t Ceil(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
        return INT64_MIN;
    }
    return ((x + y - 1) / y) * y;
}

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The y is zero");
        return INT64_MIN;
    }
    return (x + y - 1) / y;
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
    for (size_t i = 0; i < viewShapeDim; i++) {
        OP_CHECK(
            viewShape.GetDim(i) != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "srcTensor", op::ToString(viewShape).GetString(),
                                                  "srcTensor must not be empty tensor, each dimension must not be 0"),
            return ACLNN_ERR_PARAM_INVALID);
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
    CHECK_RET(srcTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(dstTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
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
    // 校验输入Nz格式tensor不含有值为0的轴
    for (uint64_t i = 0; i < srcStorageShapeDim; ++i) {
        OP_CHECK(
            srcStorageShape[i] != 0,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "srcTensor", op::ToString(srcStorageShape).GetString(),
                                                  "srcTensor cannot be empty, each axis of storageShape cannot be 0"),
            return false);
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

aclnnStatus CalcToNd(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                     uint64_t* dstShapeSize, int* actualFormat)
{
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

aclnnStatus CalcNCDHWToNDC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                                uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);
    int64_t C1 = CeilDiv(C, C0);

    *dstShapeSize = DIMS_SIX; // 6HD
    try {
        *dstShape = new int64_t[6]{N, D, C1, H, W, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NDC1HWC0");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NDC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNDHWCToNDC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                                uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t D = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C = viewShape.GetDim(4);
    int64_t C1 = CeilDiv(C, C0);

    *dstShapeSize = DIMS_SIX; // 6HD
    try {
        *dstShape = new int64_t[6]{N, D, C1, H, W, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NDC1HWC0");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NDC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCHWToNC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                              uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);

    *dstShapeSize = DIMS_FIVE; // 5HD
    try {
        *dstShape = new int64_t[5]{N, C1, H, W, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NC1HWC0");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNHWCToNC1HWC0(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                              uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);

    *dstShapeSize = DIMS_FIVE; // 5HD
    try {
        *dstShape = new int64_t[5]{N, C1, H, W, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NC1HWC0");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NC1HWC0;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCHWToFRACTALZ(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                               uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    int64_t N0 = N0_SIZE; // 私有格式的分形要求
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    *dstShapeSize = DIMS_FOUR; // FRACTAL_Z
    try {
        *dstShape = new int64_t[4]{C1 * H * W, N1, N0, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for FRACTAL_Z");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_FRACTAL_Z;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcHWCNToFRACTALZ(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                               uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    int64_t N0 = N0_SIZE; // 私有格式的分形要求
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t H = viewShape.GetDim(0);
    int64_t W = viewShape.GetDim(1);
    int64_t C = viewShape.GetDim(2);
    int64_t N = viewShape.GetDim(3);
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);
    *dstShapeSize = DIMS_FOUR; // FRACTAL_Z
    try {
        *dstShape = new int64_t[4]{C1 * H * W, N1, N0, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for FRACTAL_Z");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_FRACTAL_Z;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcNCDHWToFZ3D(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                            uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);
    int64_t N0 = 16; // 私有格式的分形要求
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    *dstShapeSize = DIMS_FOUR; // FZ3D
    try {
        *dstShape = new int64_t[4]{D * C1 * H * W, N1, N0, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for FZ3D");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_FRACTAL_Z_3D;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcDHWCNToFZ3D(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                            uint64_t* dstShapeSize, int* actualFormat)
{
    int64_t C0 = C0_SIZE;
    int64_t N0 = N0_SIZE; // 私有格式的分形要求
    DataType srcDtype = srcTensor->GetDataType();
    if (static_cast<op::DataType>(additionalDtype) == srcDtype) {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(srcDtype);
    } else {
        C0 = BLOCK_SIZE / ge::GetSizeByDataType(static_cast<op::DataType>(additionalDtype));
    }
    auto viewShape = srcTensor->GetViewShape();
    int64_t D = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);
    int64_t N = viewShape.GetDim(4);
    int64_t C1 = CeilDiv(C, C0);
    int64_t N1 = CeilDiv(N, N0);

    *dstShapeSize = DIMS_FOUR; // FZ3D
    try {
        *dstShape = new int64_t[4]{D * C1 * H * W, N1, N0, C0};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for FZ3D");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_FRACTAL_Z_3D;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNCDHW(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                        uint64_t* dstShapeSize, int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t D = viewShape.GetDim(2);
    int64_t H = viewShape.GetDim(3);
    int64_t W = viewShape.GetDim(4);

    *dstShapeSize = DIMS_FIVE; // 5HD
    try {
        *dstShape = new int64_t[5]{N, C, D, H, W};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NCDHW");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NCDHW;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNCHW(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                       uint64_t* dstShapeSize, int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t C = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);

    *dstShapeSize = DIMS_FOUR; // NCHW
    try {
        *dstShape = new int64_t[4]{N, C, H, W};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NCHW");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NCHW;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNHWC(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                       uint64_t* dstShapeSize, int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);

    *dstShapeSize = DIMS_FOUR; // NHWC
    try {
        *dstShape = new int64_t[4]{N, C, H, W};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NHWC");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NHWC;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToHWCN(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                       uint64_t* dstShapeSize, int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t H = viewShape.GetDim(0);
    int64_t W = viewShape.GetDim(1);
    int64_t C = viewShape.GetDim(2);
    int64_t N = viewShape.GetDim(3);

    *dstShapeSize = DIMS_FOUR; // HWCN
    try {
        *dstShape = new int64_t[4]{H, W, C, N};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for HWCN");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_HWCN;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToDHWCN(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                        uint64_t* dstShapeSize, int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t D = viewShape.GetDim(0);
    int64_t H = viewShape.GetDim(1);
    int64_t W = viewShape.GetDim(2);
    int64_t C = viewShape.GetDim(3);
    int64_t N = viewShape.GetDim(4);

    *dstShapeSize = DIMS_FIVE; // DHWCN
    try {
        *dstShape = new int64_t[5]{D, H, W, C, N};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for DHWCN");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_DHWCN;
    return ACLNN_SUCCESS;
}

aclnnStatus CalcToNDHWC(const aclTensor* srcTensor, [[maybe_unused]] int additionalDtype, int64_t** dstShape,
                        uint64_t* dstShapeSize, int* actualFormat)
{
    auto viewShape = srcTensor->GetViewShape();
    int64_t N = viewShape.GetDim(0);
    int64_t D = viewShape.GetDim(1);
    int64_t H = viewShape.GetDim(2);
    int64_t W = viewShape.GetDim(3);
    int64_t C = viewShape.GetDim(4);

    *dstShapeSize = DIMS_FIVE; // NDHWC
    try {
        *dstShape = new int64_t[5]{N, D, H, W, C};
    } catch (...) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "Failed to allocate memory for NDHWC");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    *actualFormat = op::Format::FORMAT_NDHWC;
    return ACLNN_SUCCESS;
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

    auto ret = CheckGetWorkSpaceSizeInputs(srcTensor, dstTensor);
    op::Format srcFormat = srcTensor->GetStorageFormat();
    op::Format dstFormat = dstTensor->GetStorageFormat();
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

    aclTensor* formatTensor;
    if (IsRegBase()) {
        formatTensor = const_cast<aclTensor*>(srcTensor);
        // 适配srcFormat为NCL的场景
        if ((IsQuantMatmulDtype(srcTensor->GetDataType(), dstTensor->GetDataType()) &&
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
    if (dstTensor->GetStorageShape().GetDim(dstDimNum - NUM_ONE) == NUM_SIXTEEN &&
        ge::GetSizeByDataType(srcTensor->GetDataType()) >= NUM_FOUR && !IsRegBase()) {
        outTensor = const_cast<aclTensor*>(
            l0op::TransDataSpecial(formatTensor, dstTensor->GetStorageFormat(), 1, uniqueExecutor.get()));
    } else {
        outTensor = const_cast<aclTensor*>(
            l0op::TransData(formatTensor, dstTensor->GetStorageFormat(), 1, uniqueExecutor.get()));
    }
    CHECK_RET(outTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outTensor->SetViewFormat(dstTensor->GetViewFormat());
    outTensor->SetOriginalFormat(dstTensor->GetOriginalFormat());
    outTensor->SetStorageFormat(dstTensor->GetStorageFormat());
    auto viewCopyResult = l0op::ViewCopy(outTensor, dstTensor, uniqueExecutor.get());
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
