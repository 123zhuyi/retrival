/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_allto_all.h"
#include "securec.h"
#include "matmul_allto_all_util.h"
#include "op_mc2.h"
#include "acl/acl.h"
#include "op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/format_utils.h"
#include "aclnn_kernels/transdata.h"
#include "hccl_util.h"

namespace {

using namespace op;
using namespace matmul_allto_all_check;

constexpr size_t THREE_DIMS = 3U;

// 检查必要输入是否为空，必须非空
static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                         const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* output) {
    if (x1 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1 should not be null.");
        return false;
    }
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2 should not be null.");
        return false;
    }
    if(op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
        if (bias == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input bias should not be null.");
            return false;
        }
    }
    if (x1Scale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x1Scale should not be null.");
        return false;
    }
    if (x2Scale == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input x2Scale should not be null.");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Output should not be null.");
        return false;
    }
    return true;
}

//检查是否有空tensor
static bool CheckNotEmptyTensor(const aclTensor* x1, const aclTensor* x2, bool transposeX2) {
    auto mVal = x1->GetViewShape().GetDim(0);
    auto kVal1 = x1->GetViewShape().GetDim(1);
    auto kVal2 = transposeX2 ? x2->GetViewShape().GetDim(1) : x2->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    OP_API_CHECK((mVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "QuantMatmulAlltoAll, x1 is empty tensor with zero dimM, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal1 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "QuantMatmulAlltoAll, x1 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((kVal2 == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "QuantMatmulAlltoAll, x2 is empty tensor with zero dimK, which is unsupported.");
        return false;
    });
    OP_API_CHECK((nVal == ZERO), {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "QuantMatmulAlltoAll, x2 is empty tensor with zero dimN, which is unsupported.");
        return false;
    });
    return true;
}

// 校验Scale为1D时的shape（KC量化）
static bool Check1DScaleShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                              const aclTensor* x2Scale, bool transposeX2) {
    OP_CHECK_WRONG_DIMENSION(x1Scale, ONE_DIM, return false);
    OP_CHECK_WRONG_DIMENSION(x2Scale, ONE_DIM, return false);
    auto mVal = x1->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x1ScaleDim = x1Scale->GetViewShape().GetDim(0);
    if (x1ScaleDim != mVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The m-axis of x1 and x1scale should be same, but x1's m-axis is: %ld and x1Scale's is: %ld.", mVal, x1ScaleDim);
        return false;
    }
    auto x2ScaleDim = x2Scale->GetViewShape().GetDim(0);
    if (x2ScaleDim != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "The n-axis of x2 and x2ScaleDim should be same, but x2's n-axis is: %ld and x2ScaleDim is: %ld.", nVal, x2ScaleDim);
        return false;
    }
    return true;
}

// 校验Scale为3D时的shape（MX量化）
static bool Check3DScaleShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                              const aclTensor* x2Scale, bool transposeX2) {
    OP_CHECK_WRONG_DIMENSION(x1Scale, THREE_DIMS, return false);
    OP_CHECK_WRONG_DIMENSION(x2Scale, THREE_DIMS, return false);
    auto mVal = x1->GetViewShape().GetDim(0);
    auto nVal = transposeX2 ? x2->GetViewShape().GetDim(0) : x2->GetViewShape().GetDim(1);
    auto x1ScaleMVal = x1Scale->GetViewShape().GetDim(0);
    auto x2ScaleNVal = transposeX2 ? x2Scale->GetViewShape().GetDim(0) : x2Scale->GetViewShape().GetDim(1);
    auto x1ScaleKVal = x1Scale->GetViewShape().GetDim(1);
    auto x2ScaleKVal = transposeX2 ? x2Scale->GetViewShape().GetDim(1) : x2Scale->GetViewShape().GetDim(0);
    auto x1ScaleLastDim = x1Scale->GetViewShape().GetDim(2);
    auto x2ScaleLastDim = x2Scale->GetViewShape().GetDim(2);
    if (x1ScaleMVal != mVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        	"The m-axis of x1 and x1scale should be same, but x1's m-axis is: %ld and x1Scale's is: %ld.", mVal, x1ScaleMVal);
        return false;
    }
    if (x2ScaleNVal != nVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        	"The n-axis of x2 and x2ScaleDim should be same, but x2's n-axis is: %ld and x2ScaleDim is: %ld.", nVal, x2ScaleNVal);
        return false;
    }
    if (x1ScaleKVal != x2ScaleKVal) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        	"The k-axis of x1scale and x2scale should be same, but x1scale's k-axis is: %ld and x2Scale's k-axis is: %ld.", x1ScaleKVal, x2ScaleKVal);
        return false;
    }
    if (x1ScaleLastDim != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        	"The last dim of x1scale should be 2, but now it is: %ld.", x1ScaleLastDim);
        return false;
    }
    if (x2ScaleLastDim != TWO) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        	"The last dim of x2scale should be 2, but now it is: %ld.", x2ScaleLastDim);
        return false;
    }
    return true;
}

// 校验输入Scaleshape
static bool CheckScaleShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
                            int64_t x1QuantMode, int64_t x2QuantMode, bool transposeX2) {
    bool scaleShapeValid = true;
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
        static_cast<QuantModeType>(x2QuantMode) == QuantModeType::MX_QUANT) {
        OP_API_CHECK(!transposeX2, {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In the mx quantization scenario, x2 must be transposed.");
            return false;
        });
        scaleShapeValid = Check3DScaleShape(x1, x2, x1Scale, x2Scale, transposeX2);
    } else if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::PERTOKEN_QUANT &&
               static_cast<QuantModeType>(x2QuantMode) == QuantModeType::PERCHANNEL_QUANT) {
        scaleShapeValid = Check1DScaleShape(x1, x2, x1Scale, x2Scale, transposeX2);
    }
    return scaleShapeValid;
}

// 910B数据类型校验
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X = {
    op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_BIAS = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_FLOAT,
    op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_Y = {
    op::DataType::DT_FLOAT16,
    op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_SCALE = {
    op::DataType::DT_FLOAT};

static bool CheckKCBiasDtypesValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                   const aclTensor* x2Scale, const aclTensor* bias, const aclTensor* y) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST_X, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_SUPPORT_LIST_Y, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, DTYPE_SUPPORT_LIST_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, DTYPE_SUPPORT_LIST_SCALE, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_SUPPORT_LIST_BIAS, return false);
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return false);
    auto biasDtype = bias->GetDataType();
    auto yDtype = y->GetDataType();
    if (biasDtype != ge::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SAME(y, bias, return false);
    } else if (yDtype != ge::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "When bias' dtype is [DT_FLOAT], y's dtype must be [DT_BF16], but get [%s].", op::ToString(yDtype).GetString());
        return false;
    }
    return true;
}

// 950数据类型校验
// 量化模式下X支持的FP8数据类型（PerToken量化、PerChannel量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> X_DTYPE_FP8_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2};
// 量化模式下Bias支持的数据类型（KC量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT};
// 量化模式下Scale支持的FP8数据类型（MXQuant量化）(A5)
static const std::initializer_list<op::DataType> SCALE_DTYPE_FP8_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT8_E8M0};
// 量化模式下Scale支持的FP32数据类型（PerToken量化、PerChannel量化）(A5)
static const std::initializer_list<op::DataType> SCALE_DTYPE_FP32_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT};
// 量化模式下Output支持的数据类型（KC量化、MXQuant量化）(A5)
static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST_A5 = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};

// KC量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckKCQuantDtypesValidA5(const aclTensor* x1, const aclTensor* x2,
                               		  const aclTensor* x1Scale, const aclTensor* x2Scale,
                               		  const aclTensor* biasOptional, const aclTensor* output) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_FP8_SUPPORT_LIST_A5, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_FP8_SUPPORT_LIST_A5, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, SCALE_DTYPE_FP32_SUPPORT_LIST_A5, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, SCALE_DTYPE_FP32_SUPPORT_LIST_A5, return false);
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, BIAS_DTYPE_SUPPORT_LIST_A5, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST_A5, return false);
    return true;
}

// MX量化场景下，校验所有输入的参数类型是否正确(A5)
static bool CheckMXQuantDtypesValidA5(const aclTensor* x1, const aclTensor* x2,
                               		  const aclTensor* x1Scale, const aclTensor* x2Scale,
                               		  const aclTensor* biasOptional, const aclTensor* output) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, X_DTYPE_FP8_SUPPORT_LIST_A5, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, X_DTYPE_FP8_SUPPORT_LIST_A5, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, SCALE_DTYPE_FP8_SUPPORT_LIST_A5, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, SCALE_DTYPE_FP8_SUPPORT_LIST_A5, return false);
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, BIAS_DTYPE_SUPPORT_LIST_A5, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(output, OUTPUT_DTYPE_SUPPORT_LIST_A5, return false);
    return true;
}

// 校验所有场景的数据类型是否在各自的支持列表中
static bool CheckDtypesValid(const aclTensor* x1, const aclTensor* x2,
                             const int64_t x1QuantMode, const int64_t x2QuantMode,
                             const aclTensor* x1Scale, const aclTensor* x2Scale,
                             const aclTensor* biasOptional, const aclTensor* output) {
    bool isAllDtypesValid = false;
    // 根据量化场景和芯片型号进入不同分支判断
    if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::PERTOKEN_QUANT &&
        static_cast<QuantModeType>(x2QuantMode) == QuantModeType::PERCHANNEL_QUANT) {
        if(op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
            isAllDtypesValid = CheckKCBiasDtypesValid(x1, x2, x1Scale, x2Scale, biasOptional, output);
        } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            isAllDtypesValid = CheckKCQuantDtypesValidA5(x1, x2, x1Scale, x2Scale, biasOptional, output);
        }
    } else if (static_cast<QuantModeType>(x1QuantMode) == QuantModeType::MX_QUANT &&
               static_cast<QuantModeType>(x2QuantMode) == QuantModeType::MX_QUANT) {
        isAllDtypesValid = CheckMXQuantDtypesValidA5(x1, x2, x1Scale, x2Scale, biasOptional, output);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "The input x1QuantMode %ld and x2QuantMode %ld do not match any currently supported quantization mode scenarios.",
            x1QuantMode, x2QuantMode);
    }
    return isAllDtypesValid;
}

// 检查所有要用到的输入format是否为ND，不支持私有格式，如果内部不为ND格式，会打印warning日志，并将format转换为ND格式
static bool CheckFormat(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                        const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* output) {
    // 输入格式不支持私有格式
    if (IsPrivateFormat(x1->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, x1 format %s does not support private format.",
                op::ToString(x1->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, x2 format %s does not support private format.",
                op::ToString(x2->GetStorageFormat()).GetString());
        return false;
    }
    if (biasOptional != nullptr) {
        if (IsPrivateFormat(biasOptional->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, biasOptional format %s does not support private format.",
                op::ToString(biasOptional->GetStorageFormat()).GetString());
            return false;
        }
    }
    if (IsPrivateFormat(x1Scale->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, x1Scale format %s does not support private format.",
                op::ToString(x1Scale->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(x2Scale->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, x2Scale format %s does not support private format.",
                op::ToString(x2Scale->GetStorageFormat()).GetString());
        return false;
    }
    if (IsPrivateFormat(output->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "aclnnQuantMatmulAlltoAll, output format %s does not support private format.",
                op::ToString(output->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

// 兼容性处理，非ND格式转换为ND格式
static bool ReFormatNotND(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                          const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* output) {
    // 内部只处理ND格式，这里做reformat操作
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1 origin format is %s.", op::ToString(x1->GetStorageFormat()).GetString());
        x1 = l0op::ReFormat(x1, op::Format::FORMAT_ND);
        CHECK_RET(x1 != nullptr, false);
    }
    if (x2->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2 origin format is %s.", op::ToString(x2->GetStorageFormat()).GetString());
        x2 = l0op::ReFormat(x2, op::Format::FORMAT_ND);
        CHECK_RET(x2 != nullptr, false);
    }
    if (biasOptional != nullptr) {
        if (biasOptional->GetStorageFormat() != op::Format::FORMAT_ND) {
            OP_LOGW("bias origin format is %s.", op::ToString(biasOptional->GetStorageFormat()).GetString());
            biasOptional = l0op::ReFormat(biasOptional, op::Format::FORMAT_ND);
            CHECK_RET(biasOptional != nullptr, false);
        }
    }
    if (x1Scale->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x1Scale origin format is %s.", op::ToString(x1Scale->GetStorageFormat()).GetString());
        x1Scale = l0op::ReFormat(x1Scale, op::Format::FORMAT_ND);
        CHECK_RET(x1Scale != nullptr, false);
    }
    if (x2Scale->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("x2Scale origin format is %s.", op::ToString(x2Scale->GetStorageFormat()).GetString());
        x2Scale = l0op::ReFormat(x2Scale, op::Format::FORMAT_ND);
        CHECK_RET(x2Scale != nullptr, false);
    }
    if (output->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGW("output origin format is %s.", op::ToString(output->GetStorageFormat()).GetString());
        output = l0op::ReFormat(output, op::Format::FORMAT_ND);
        CHECK_RET(output != nullptr, false);
    }
    return true;
}

static aclnnStatus CheckAndHandleParams(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                        const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* commScaleOptional,
                                        const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional, const char* group,
                                        const aclIntArray* alltoAllAxesOptional, int64_t x1QuantMode, int64_t x2QuantMode,
                                        int64_t commQuantMode, int64_t commQuantDtype, int64_t groupSize,
                                        bool transposeX1, bool transposeX2, const aclTensor* output)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, biasOptional, x1Scale, x2Scale, output), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查空tensor
    CHECK_RET(CheckNotEmptyTensor(x1, x2, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape
    CHECK_RET(CheckShapeMMAA(x1, x2, biasOptional, transposeX2, output), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckScaleShape(x1, x2, x1Scale, x2Scale, x1QuantMode, x2QuantMode, transposeX2), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypesValid(x1, x2, x1QuantMode, x2QuantMode, x1Scale, x2Scale, biasOptional, output), ACLNN_ERR_PARAM_INVALID);
    // 5. 检查输入的数据格式是否为ND
    CHECK_RET(CheckFormat(x1, x2, biasOptional, x1Scale, x2Scale, output), ACLNN_ERR_PARAM_INVALID);
    // 6.兼容性处理非ND格式
    CHECK_RET(ReFormatNotND(x1, x2, biasOptional, x1Scale, x2Scale, output), ACLNN_ERR_PARAM_INVALID);
    // 7. 检查groupSize是否和当前场景匹配
    CHECK_RET(CheckGroupSizeValid(groupSize, x1QuantMode, x2QuantMode), ACLNN_ERR_PARAM_INVALID);
    // 8. 检查alltoAllAxes是否为空或者[-1,-2]
    CHECK_RET(CheckAlltoAllAxes(alltoAllAxesOptional, true), ACLNN_ERR_PARAM_INVALID);
    // 9. 检查transposeX1是否合法, 目前不能为true
    CHECK_RET(CheckTransposeX1(transposeX1), ACLNN_ERR_PARAM_INVALID);
    // 10. 检查group长度是否小于等于128
    CHECK_RET(CheckGroupLength(group), ACLNN_ERR_PARAM_INVALID);
    // 11. 检查预留参数，不影响场景
    CheckReservedParams(commScaleOptional, x1OffsetOptional, x2OffsetOptional, commQuantMode, commQuantDtype);
    // 如果所有检查都通过，且reformat也通过，输出参数检查成功
    OP_LOGD("aclnnQuantMatmulAlltoAll checkParams success");
    return ACLNN_SUCCESS;
}
} // namespace

// L0层两段式接口Inner，根据算子原型op_graph/matmul_allto_all_proto.h，由模板自动生成。非量化L2层接口和量化L2层接口共用一套L0层接口。
// worldSize为硬件方参数，在aclnn侧不感知。yDtype在aclnn侧不感知。这两个参数需要在Inner接口处声明，在aclnn侧通过默认值传参。
extern "C" aclnnStatus aclnnInnerMatmulAlltoAllGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                                                const aclTensor* x1ScaleOptional, const aclTensor* x2ScaleOptional,
                                                                const aclTensor* commScaleOptional,
                                                                const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional,
                                                                char* group, int64_t worldSize, const aclIntArray* all2allAxesOptional,
                                                                int64_t yDtype, int64_t x1QuantMode, int64_t x2QuantMode,
                                                                int64_t commQuantMode, int64_t commQuantDtype,
                                                                bool transposeX1, bool transposeX2, int64_t groupSize,
                                                                const aclTensor* out, uint64_t *workspaceSize, aclOpExecutor **executor);
extern "C" aclnnStatus aclnnInnerMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// 两段式接口
extern "C" aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* biasOptional,
                                                                const aclTensor* x1Scale, const aclTensor* x2Scale,
                                                                const aclTensor* commScaleOptional,
                                                                const aclTensor* x1OffsetOptional, const aclTensor* x2OffsetOptional,
                                                                const aclIntArray* alltoAllAxesOptional, const char* group,
                                                                int64_t x1QuantMode, int64_t x2QuantMode,
                                                                int64_t commQuantMode, int64_t commQuantDtype, int64_t groupSize,
                                                                bool transposeX1, bool transposeX2, const aclTensor* output,
                                                                uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 处理非连续Tensor，目前只有支持转置的x2涉及该处理
    CHECK_RET(CheckX2Valid(x2), ACLNN_ERR_PARAM_INVALID);	// 先检查x2是否合法，避免访问空指针等等非法操作
    bool notContiguous = IsTransposeLastTwoDims(x2);    // notContiguous标识x2是否是非连续的，通常在pytorch经过.t()会导致x2非连续
    auto transX2 = x2;    // 复制一个x2
    if (notContiguous && transposeX2) {    // 当非连续和转置同时生效时，判断为错误用法，直接报错
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 not contiguous, and set x2 transpose, it is error!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (notContiguous && GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {    // 只有当非连续时，才会涉及到转连续等情况
        transposeX2 = true;
        // 把非连续x2转成连续
        transX2 = TransX2Tensor(x2);
        CHECK_RET(transX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        OP_LOGD("X2 is a non-contiguous tensor. The original dim0 is %ld, and dim1 is %ld. After processing, transX2 dim0 is %ld, and dim1 is %ld.",
            x2->GetViewShape().GetDim(0), x2->GetViewShape().GetDim(1), transX2->GetViewShape().GetDim(0), transX2->GetViewShape().GetDim(1));
    }
    aclnnStatus retParam = CheckAndHandleParams(
        x1, transX2, biasOptional, x1Scale, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional, group, alltoAllAxesOptional,
        x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype, groupSize, transposeX1, transposeX2, output);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    // Inner接口部分入参类型和aclnn接口不一致，需要重新包装，同时Inner接口额外需要部分参数，按算子原型模板和实际业务逻辑生成
    const aclTensor* out = output;
    int64_t worldSize = -1;
    char* str_group = const_cast<char*>(group);
    // ACL和GE的datatype枚举值对undefined定义不同，inner接口进入到算子内部，需要使用GE枚举值
    commQuantDtype = op::DataType::DT_UNDEFINED;
    // 生成yDtype，图模式需要
    int64_t enumYDtype = op::DataType::DT_UNDEFINED;
    auto yDtype = output->GetDataType();
    if (yDtype == op::DataType::DT_FLOAT) {
        enumYDtype = 0;
    } else if (yDtype == op::DataType::DT_FLOAT16) {
        enumYDtype = 1;
    } else if (yDtype == op::DataType::DT_BF16) {
        enumYDtype = 27;
    }
    aclnnStatus ret = aclnnInnerMatmulAlltoAllGetWorkspaceSize(
        x1, transX2, biasOptional, x1Scale, x2Scale, commScaleOptional, x1OffsetOptional, x2OffsetOptional,
        str_group, worldSize, alltoAllAxesOptional, enumYDtype, x1QuantMode, x2QuantMode, commQuantMode, commQuantDtype,
        transposeX1, transposeX2, groupSize, out, workspaceSize, executor);
    OP_LOGD("QuantMatmulAlltoAll, aclnnnInnerGetWorkspaceSize ret %d.", ret);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnQuantMatmulAlltoAllGetWorkspaceSize interface call failed.");
    }
    return ret;
}

extern "C" aclnnStatus aclnnQuantMatmulAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
    aclnnStatus ret = aclnnInnerMatmulAlltoAll(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER,
                "This is an error in launch aicore, aclnnQuantMatmulAlltoAll interface call failed.");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}
