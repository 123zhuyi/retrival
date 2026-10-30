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
 * \file stft.cpp
 * \brief
 */

#include "stft.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(STFT);

static const int X1_NFFT = 400;
static const int X1_HOP = 160;
static const int TYPE_SIZE = 4; // float32
static const int BLOCK_SIZE = 32;
static const int PACKAGE_SIZE = 128;
static const int MAX_CACHE_SIZE = 500 * 1024 * 1024;                   // 500MB
static const uint64_t MAX_GM_SIZE = (uint64_t)35 * 1024 * 1024 * 1024; // 35GB

bool IsStftAiCoreSupported(
    const aclTensor* self, const aclTensor* window, int64_t nFft, int64_t hopLength, int64_t winLength, bool normalized,
    bool onesided, bool returnComplex)
{
    (void)winLength;
    bool res = false;
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        return res;
    }
    op::Shape shape = self->GetViewShape();
    int64_t batch = shape.GetDimNum() == 2 ? shape.GetDim(0) : 1;
    int64_t len = shape.GetDimNum() == 2 ? shape.GetDim(1) : shape.GetDim(0);
    if (hopLength <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect hopLength > 0");
        return false;
    }
    int64_t matmulN = (len - nFft) / hopLength + 1;
    int64_t matmulM = onesided ? nFft / 2 + 1 : nFft;
    int alignNum =
        (nFft == X1_NFFT && hopLength == X1_HOP && normalized == false && onesided == true && returnComplex == false) ?
            BLOCK_SIZE / TYPE_SIZE :
            PACKAGE_SIZE / TYPE_SIZE;
    int64_t nFftAlign = (nFft + alignNum - 1) / alignNum * alignNum;
    int64_t cacheSize = (matmulM * nFftAlign) * TYPE_SIZE * 2;
    // gm size = input + output + splitWindow workspace + matmul workspace
    uint64_t gmSize = batch * len * TYPE_SIZE + batch * matmulM * matmulN * 2 * TYPE_SIZE +
                      batch * matmulN * nFftAlign * TYPE_SIZE + batch * matmulM * matmulN * 2 * TYPE_SIZE;
    if (cacheSize <= MAX_CACHE_SIZE && gmSize <= MAX_GM_SIZE && self->GetDataType() == op::DataType::DT_FLOAT) {
        if (window == nullptr || window->GetDataType() == op::DataType::DT_FLOAT) {
            res = true;
        }
    }
    return res;
}

// AICORE算子kernel
static const aclTensor* StftAiCore(
    const aclTensor* self, const aclTensor* plan, const aclTensor* window, aclTensor* out, int64_t nFft,
    int64_t hopLength, int64_t winLength, bool normalized, bool onesided, bool returnComplex, aclOpExecutor* executor)
{
    L0_DFX(StftAiCore, self, plan, window, nFft, hopLength, winLength, normalized, onesided, returnComplex, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICORE，将AiCore Stft算子加入任务队列
    auto retAiCore = ADD_TO_LAUNCHER_LIST_AICORE(
        STFT, OP_ATTR_NAMES({"hop_length", "win_length", "normalized", "onesided", "return_complex", "n_fft"}),
        OP_INPUT(self, plan, window), OP_OUTPUT(out),
        OP_ATTR(hopLength, winLength, normalized, onesided, returnComplex, nFft));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        retAiCore != ACLNN_SUCCESS, return nullptr, "STFT ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}

// AICPU算子kernel
static const aclTensor* StftAiCpu(
    const aclTensor* self, const aclTensor* window, aclTensor* out, int64_t nFft, int64_t hopLength, int64_t winLength,
    bool normalized, bool onesided, bool returnComplex, aclOpExecutor* executor)
{
    L0_DFX(StftAiCpu, self, window, nFft, hopLength, winLength, normalized, onesided, returnComplex, out);

    // 使用框架宏ADD_TO_LAUNCHER_LIST_AICPU，将AiCpu Stft算子加入任务队列
    static internal::AicpuTaskSpace space("STFT");
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        STFT, OP_ATTR_NAMES({"hop_length", "win_length", "normalized", "onesided", "return_complex", "n_fft"}),
        OP_INPUT(self, window), OP_OUTPUT(out),
        OP_ATTR(hopLength, winLength, normalized, onesided, returnComplex, nFft));
    CHECK_RET(ret == ACLNN_SUCCESS, nullptr);
    return out;
}

static op::DataType GetOutputTypeByInput(const aclTensor* self, bool returnComplex)
{
    op::DataType outputType = op::DataType::DT_UNDEFINED;
    op::DataType inputType = self->GetDataType();
    if (returnComplex) {
        if (inputType == op::DataType::DT_COMPLEX64 || inputType == op::DataType::DT_COMPLEX128) {
            outputType = inputType;
        } else if (inputType == op::DataType::DT_FLOAT) {
            outputType = op::DataType::DT_COMPLEX64;
        } else if (inputType == op::DataType::DT_DOUBLE) {
            outputType = op::DataType::DT_COMPLEX128;
        }
    } else {
        if (inputType == op::DataType::DT_FLOAT || inputType == op::DataType::DT_DOUBLE) {
            outputType = inputType;
        } else if (inputType == op::DataType::DT_COMPLEX64) {
            outputType = op::DataType::DT_FLOAT;
        } else if (inputType == op::DataType::DT_COMPLEX128) {
            outputType = op::DataType::DT_DOUBLE;
        }
    }
    return outputType;
}

static op::Shape GetOutputShape(
    const aclTensor* self, int64_t nFft, int64_t hopLength, bool onesided, bool returnComplex)
{
    op::Shape shape = self->GetViewShape();
    int64_t batch = shape.GetDimNum() == 2 ? shape.GetDim(0) : 0;
    int64_t len = shape.GetDimNum() == 2 ? shape.GetDim(1) : shape.GetDim(0);
    if (hopLength <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expect hopLength > 0, change hopLength = 1");
        hopLength = 1;
    }
    int64_t frames = (len - nFft) / hopLength + 1;
    int64_t n = onesided ? nFft / 2 + 1 : nFft;

    op::Shape outputShape;
    op::Shape complexWithBatch{batch, n, frames};
    op::Shape complexWithoutBatch{n, frames};
    op::Shape realWithBatch{batch, n, frames, 2};
    op::Shape realWithoutBatch{n, frames, 2};

    if (returnComplex) {
        outputShape = batch > 0 ? complexWithBatch : complexWithoutBatch;
    } else {
        outputShape = batch > 0 ? realWithBatch : realWithoutBatch;
    }
    return outputShape;
}

const aclTensor* Stft(
    const aclTensor* self, const aclTensor* plan, const aclTensor* window, int64_t nFft, int64_t hopLength,
    int64_t winLength, bool normalized, bool onesided, bool returnComplex, aclOpExecutor* executor)
{
    op::Shape outShape = GetOutputShape(self, nFft, hopLength, onesided, returnComplex);
    op::DataType outType = GetOutputTypeByInput(self, returnComplex);

    auto out = executor->AllocTensor(outShape, outType, self->GetStorageFormat());
    CHECK_RET(out != nullptr, nullptr);

    if (IsStftAiCoreSupported(self, window, nFft, hopLength, winLength, normalized, onesided, returnComplex)) {
        return StftAiCore(
            self, plan, window, out, nFft, hopLength, winLength, normalized, onesided, returnComplex, executor);
    } else {
        return StftAiCpu(self, window, out, nFft, hopLength, winLength, normalized, onesided, returnComplex, executor);
    }
}

} // namespace l0op
