/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "quant_matmul_activation_quant.h"
#include "util/math_util.h"
#include "quant_matmul_activation_quant_util.h"

using namespace op;
using namespace QBMMActivationQuant;

namespace l0op {

OP_TYPE_REGISTER(QuantMatmulActivationQuant);
constexpr int64_t DIM_TWO = 2L;
constexpr int64_t BLOCKSIZE = 32L;

const std::array<aclTensor*, QUANT_MATMUL_ACTIVATION_QUANT_OUT_NUM> QuantMatmulActivationQuant(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    bool transposeX1, bool transposeX2, int64_t groupSize, const char* activationType, int64_t y_dtype,
    const char* quantMode, const char* roundMode, int64_t scaleAlg, double dstTypeMax, aclOpExecutor* executor)
{
    L0_DFX(QuantMatmulActivationQuant, x1, x2, bias, x1Scale, x2Scale, transposeX1, transposeX2, groupSize,
           activationType, y_dtype, quantMode, roundMode, scaleAlg, dstTypeMax);

    Format format = Format::FORMAT_ND;
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    op::Shape yOutShape = x1Shape;
    auto x1DimNum = x1Shape.GetDimNum();
    auto x2DimNum = x2Shape.GetDimNum();

    auto yOutDim0 = transposeX1 ? x1->GetViewShape().GetDim(x1DimNum - LAST_FIRST_DIM_INDEX) :
                                  x1->GetViewShape().GetDim(x1DimNum - LAST_SECOND_DIM_INDEX);
    auto yOutDim1 = transposeX2 ? x2->GetViewShape().GetDim(x2DimNum - LAST_SECOND_DIM_INDEX) :
                                  x2->GetViewShape().GetDim(x2DimNum - LAST_FIRST_DIM_INDEX);
    yOutShape.SetDim(x1DimNum - LAST_SECOND_DIM_INDEX, yOutDim0);
    yOutShape.SetDim(x1DimNum - LAST_FIRST_DIM_INDEX, yOutDim1);

    op::Shape yScaleOutShape = yOutShape;
    auto yScaleOutDim1 = (Ops::Base::CeilDiv(yOutDim1, BLOCKSIZE) + MXFP_MULTI_BASE_SIZE - 1) / MXFP_MULTI_BASE_SIZE;
    yScaleOutShape.SetDim(x1DimNum - LAST_FIRST_DIM_INDEX, yScaleOutDim1);
    yScaleOutShape.AppendDim(DIM_TWO);

    auto yOut = executor->AllocTensor(yOutShape, x1->GetDataType(), format);
    auto yScaleOut = executor->AllocTensor(yScaleOutShape, x1Scale->GetDataType(), format);

    if (yOut == nullptr || yScaleOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alloc tensor failed.");
        return {yOut, yScaleOut};
    }

    // 运行QuantMatmulActivationQuant算子的InferShape函数，推导输出shape
    auto ret = INFER_SHAPE(QuantMatmulActivationQuant, OP_INPUT(x1, x2, bias, x1Scale, x2Scale),
                           OP_OUTPUT(yOut, yScaleOut),
                           OP_ATTR(transposeX1, transposeX2, groupSize, activationType, y_dtype, quantMode, roundMode,
                                   scaleAlg, dstTypeMax));

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return {nullptr, nullptr};
    }

    OP_LOGD("l0 transposeX1 = %s, transposeX2 = %s", transposeX1 ? "true" : "false", transposeX2 ? "true" : "false");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantMatmulActivationQuant, OP_INPUT(x1, x2, bias, x1Scale, x2Scale),
                                      OP_OUTPUT(yOut, yScaleOut),
                                      OP_ATTR(transposeX1, transposeX2, groupSize, activationType, y_dtype, quantMode,
                                              roundMode, scaleAlg, dstTypeMax));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {yOut, yScaleOut};
}

} // namespace l0op
