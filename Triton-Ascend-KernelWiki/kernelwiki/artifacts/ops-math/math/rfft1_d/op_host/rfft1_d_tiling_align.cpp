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
 * \file rfft1d_tiling_align.cpp
 * \brief
 */
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "exe_graph/runtime/shape.h"
#include "rfft1_d_tiling_base.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>

static const uint8_t MAX_FACTORS_LEN = 3;
static const uint32_t USER_WORKSPACE_SIZE = 2147483648;
static const uint32_t LAST_FACTOR = 64;
static const uint32_t COMPLEX_PART = 2;
static const uint32_t MATMUL_SIZE_MULTIPLIER = 24;
static const uint32_t SIZE_PER_BATCH_MULTIPLIER = 4;
static const uint32_t FIRST_FACTOR = 2;
static const uint32_t BYTES_ALIGN = 8;
static const uint32_t ROW_PAD = 16;
static const uint32_t COL_PAD = 8;
static const uint32_t TWIDDLE_MATRICES_AMOUNT = 2;
static const gert::Shape g_vec_1_shape = {1};

static const uint32_t DFT_BORDER_VALUE = 4096;

using namespace AscendC;
using namespace matmul_tiling;

namespace optiling {

class Rfft1DTiling : public Rfft1DBaseTiling {
public:
    explicit Rfft1DTiling(gert::TilingContext* context) : Rfft1DBaseTiling(context)
    {}

protected:
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    bool IsCapable() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);

private:
    uint32_t batches = 1;
    uint32_t dftRealOffsets[3] = {0, 1, 1};
    uint32_t dftImagOffsets[3] = {0, 1, 1};
    uint32_t twiddleOffsets[3] = {0, 0, 1};
    Rfft1DTilingData tiling;
    void CalcDftSizes(const uint32_t factors[], const bool isBluestein, const uint32_t len);
};

const gert::Shape& Rfft1DTiling::EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

bool Rfft1DTiling::IsCapable()
{
    return true;
}

ge::graphStatus Rfft1DTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t Rfft1DTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus Rfft1DTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

static void CalcColleyTukeyFactors(uint32_t factors[], std::vector<uint32_t> availableFactors, const uint32_t len)
{
    std::vector<uint32_t> factorsTmp;
    int curFactorIndex = availableFactors.size() - 1;
    uint32_t tmpN = len;

    if (tmpN == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else if (tmpN > DFT_BORDER_VALUE) {
        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];

            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmp.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }

        while (factorsTmp.size() < MAX_FACTORS_LEN) {
            factorsTmp.emplace_back(1);
        }
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = factorsTmp[i];
        }
    }
}

static void CalcBluesteinFactors(uint32_t factors[], std::vector<uint32_t> availableFactors, const uint32_t pow2)
{
    std::vector<uint32_t> factorsTmpBluestein;
    if (pow2 == LAST_FACTOR * LAST_FACTOR * LAST_FACTOR * COMPLEX_PART) {
        for (size_t i = 0; i < MAX_FACTORS_LEN; i++) {
            factors[i] = LAST_FACTOR;
        }
        factors[0] = LAST_FACTOR * COMPLEX_PART;
    } else {
        uint32_t tmpN = pow2;
        int curFactorIndex = availableFactors.size() - 1;

        while (curFactorIndex >= 0) {
            uint32_t curFactor = availableFactors[curFactorIndex];
            while (tmpN % curFactor == 0) {
                tmpN /= curFactor;
                factorsTmpBluestein.emplace_back(curFactor);
            }
            curFactorIndex -= 1;
        }

        while (factorsTmpBluestein.size() < MAX_FACTORS_LEN) {
            factorsTmpBluestein.emplace_back(1);
        }

        for (size_t i = 0; i < MAX_FACTORS_LEN; ++i) {
            factors[i] = factorsTmpBluestein[i];
        }
    }
}

void Rfft1DTiling::CalcDftSizes(const uint32_t factors[], const bool isBluestein, const uint32_t len)
{
    uint32_t dftRealOverallSize = 0;
    uint32_t dftImagOverallSize = 0;
    uint32_t twiddleOverallSize = 0;

    size_t prevFactors = 1;

    for (size_t curIndex = 0; curIndex < MAX_FACTORS_LEN; ++curIndex) {
        size_t curFactor = factors[curIndex];
        size_t rowsNum = curFactor * (1 + static_cast<size_t>(curIndex == 0 && isBluestein));
        size_t colsNum = curFactor * (2 - static_cast<size_t>(curIndex != 0));
        size_t dftCurSize = rowsNum * colsNum;
        size_t twiddleCurSize = rowsNum * prevFactors * COMPLEX_PART;

        if (curIndex != 0) {
            dftRealOffsets[curIndex] = dftRealOverallSize;
            dftImagOffsets[curIndex] = dftImagOverallSize;
            if (curIndex != 1) {
                twiddleOffsets[curIndex] = twiddleOverallSize;
            }
        }

        dftRealOverallSize += dftCurSize;
        if (curIndex != 0) {
            dftImagOverallSize += dftCurSize;
            twiddleOverallSize += twiddleCurSize;
        }

        prevFactors *= curFactor;
    }
    uint32_t fftPadRow = len + (COL_PAD - len % COL_PAD);
    uint32_t fftPadCol = ((len % ROW_PAD) != 0) ? len + (ROW_PAD - len % ROW_PAD) : len;
    dftRealOverallSize = len <= DFT_BORDER_VALUE ? fftPadRow * fftPadCol : dftRealOverallSize;

    tiling.set_dftRealOverallSize(dftRealOverallSize);
    tiling.set_dftImagOverallSize(dftImagOverallSize);
    tiling.set_twiddleOverallSize(twiddleOverallSize);
    tiling.set_fftMatrOverallSize(
        dftRealOverallSize + dftImagOverallSize + TWIDDLE_MATRICES_AMOUNT * twiddleOverallSize);

    tiling.set_dftRealOffsets(dftRealOffsets);
    tiling.set_dftImagOffsets(dftImagOffsets);
    tiling.set_twiddleOffsets(twiddleOffsets);
}

ge::graphStatus Rfft1DTiling::DoOpTiling()
{
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = EnsureNotScalar(inputX->GetStorageShape());

    // Calculate number of batches
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        batches *= xShape.GetDim(i);
    }

    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const uint32_t len = *runtimeAttrs->GetAttrPointer<uint32_t>(0);
    const uint32_t norm = *runtimeAttrs->GetAttrPointer<uint32_t>(1);

    uint32_t factors[MAX_FACTORS_LEN] = {1, 1, 1};
    uint32_t prevRadices[MAX_FACTORS_LEN] = {1, 1, 1};
    uint32_t nextRadices[MAX_FACTORS_LEN] = {len / factors[0], 1, 1};
    uint8_t prevRadicesAlign[MAX_FACTORS_LEN] = {0, 1, 1};

    // Calculate factors
    std::vector<uint32_t> availableFactors(LAST_FACTOR - 1);
    std::iota(availableFactors.begin(), availableFactors.end(), COMPLEX_PART);

    CalcColleyTukeyFactors(factors, availableFactors, len);

    const bool isBluestein = (len % LAST_FACTOR != 0) || (factors[0] * factors[1] * factors[2] != len);
    const uint32_t pow2 = COMPLEX_PART * uint32_t(std::pow(2, std::ceil(std::log2(double(len)))));
    const uint32_t lengthPad = isBluestein ? pow2 : len;

    if (isBluestein) {
        CalcBluesteinFactors(factors, availableFactors, pow2);
    }

    for (uint8_t i = 1; i < MAX_FACTORS_LEN; ++i) {
        prevRadices[i] = prevRadices[i - 1] * factors[i - 1];
        nextRadices[i] = nextRadices[i - 1] / factors[i];
        prevRadicesAlign[i] = (COMPLEX_PART * prevRadices[i] % BYTES_ALIGN) == 0;
    }

    auto roundUpBlock = [](const uint32_t& src, const uint32_t blockLen) {
        return src != 0 ? src + (blockLen - src % blockLen) % blockLen : blockLen;
    };

    const uint32_t tailSize =
        COMPLEX_PART * (((len / COMPLEX_PART) + 1) - (factors[2] / COMPLEX_PART) * (len / factors[2]));
    const uint32_t tmpLenPerBatch = 3 * roundUpBlock(
                                            COMPLEX_PART * (isBluestein ? lengthPad : len) + factors[2] * tailSize + 1,
                                            BYTES_ALIGN * SIZE_PER_BATCH_MULTIPLIER);

    tiling.set_length(len);
    tiling.set_isBluestein(isBluestein);
    tiling.set_lengthPad(lengthPad);
    tiling.set_outLength((len / COMPLEX_PART) + 1);
    tiling.set_batchesPerCore(batches / coreNum);
    tiling.set_leftOverBatches(batches % coreNum);
    tiling.set_normal(norm);
    tiling.set_factors(factors);
    tiling.set_prevRadices(prevRadices);
    tiling.set_nextRadices(nextRadices);
    tiling.set_prevRadicesAlign(prevRadicesAlign);
    tiling.set_tailSize(tailSize);
    tiling.set_tmpLenPerBatch(tmpLenPerBatch);
    tiling.set_tmpSizePerBatch(tmpLenPerBatch * SIZE_PER_BATCH_MULTIPLIER);
    tiling.set_matmulTmpsLen(MATMUL_SIZE_MULTIPLIER * tmpLenPerBatch);
    tiling.set_matmulTmpsSize(MATMUL_SIZE_MULTIPLIER * tmpLenPerBatch * sizeof(float));

    CalcDftSizes(factors, isBluestein, len);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Rfft1DTiling::PostTiling()
{
    context_->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    auto userWorkspace = USER_WORKSPACE_SIZE;
    auto sysWorkspace = 16 * 1024 * 1024;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = userWorkspace + sysWorkspace;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Rfft1D, Rfft1DTiling, 10000);

} // namespace optiling
