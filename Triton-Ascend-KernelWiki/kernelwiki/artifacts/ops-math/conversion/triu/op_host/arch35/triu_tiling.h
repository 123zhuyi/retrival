/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file trilu.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_TRILU_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_TRILU_H

#include <algorithm>
#include "triangulator_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "assert.h"
#include "op_host/tiling_base_util.h"

namespace optiling {

constexpr size_t INDEX_K = 1;        // k,optional input
constexpr size_t INDEX_X = 0;        // x,input
constexpr size_t INDEX_DIAGONAL = 0; // diagonal,attribute
constexpr uint8_t MIN_DIM_NUM = 2;   // the min dim number of input

constexpr uint8_t BLOCK_BYTES = 32; // 32 bytes for one block
constexpr uint8_t MAX_ELT_NUM = 64; // the max element number of one matrix for tiling mode 2
// tiling mode
constexpr int64_t TILING_MODE_OUTPUT_ZERO = 0;
constexpr int64_t TILING_MODE_OUTPUT_INPUT = 1;
constexpr int64_t TILING_MODE_SMALL_MATRIX = 2;
constexpr int64_t TILING_MODE_SMALL_ROW = 3;
constexpr int64_t TILING_MODE_NORMAL = 4;
constexpr int64_t TILING_MODE_BIG_ROW = 5;
const gert::Shape g_vec_1_shape = {1};

inline static const gert::Shape& EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.IsScalar()) {
        return g_vec_1_shape;
    }
    return in_shape;
}

int64_t GetEltNumPerCore(int64_t availableUBSize, int32_t availableCoreNum, int64_t unitNum, uint8_t dtypeBytes,
                         int64_t totalEltNum);
int64_t GetAlignedSize(int64_t unitSize);
void PrintInfo(gert::TilingContext* context);
ge::graphStatus TilingPrepare4Trilu(gert::TilingParseContext* context);

template <bool upper>
int64_t GetMask(int64_t row, int64_t col, int64_t diagonal)
{
    int64_t mask = 0;
    uint8_t idx = 0;
    for (auto r = 0; r < row; ++r) {
        for (auto c = 0; c < col; ++c) {
            if ((!upper && c > r + diagonal) || (upper && c < r + diagonal)) {
                mask |= 1UL << idx;
            }
            ++idx;
        }
    }
    return mask;
}

template <bool upper>
ge::graphStatus SetTilingMode(const optiling::TriluCompileInfo* compileInfo, optiling::TriluTilingParams* params,
                              uint8_t dtypeBytes)
{
    auto diag = params->diagonal;
    auto row = params->row;
    auto col = params->col;
    if ((!upper && diag <= -row) || (upper && diag >= col)) { // output all zeros
        params->tilingMode = TILING_MODE_OUTPUT_ZERO;
        return ge::GRAPH_SUCCESS;
    }

    if ((!upper && diag >= col - 1) || (upper && diag <= -row + 1)) { // output just copy input
        params->tilingMode = TILING_MODE_OUTPUT_INPUT;
        return ge::GRAPH_SUCCESS;
    }

    if (row * col < MAX_ELT_NUM) { // the matrix is too small
        params->tilingMode = TILING_MODE_SMALL_MATRIX;
        return ge::GRAPH_SUCCESS;
    }

    if (col * dtypeBytes <= BLOCK_BYTES) { // the row is too small
        params->tilingMode = TILING_MODE_SMALL_ROW;
        return ge::GRAPH_SUCCESS;
    }

    if ((col * dtypeBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES <=
        compileInfo->availableUBSize) { // most scenario
        params->tilingMode = TILING_MODE_NORMAL;
        return ge::GRAPH_SUCCESS;
    }

    params->tilingMode = TILING_MODE_BIG_ROW; // the row is too large to put into ub
    return ge::GRAPH_SUCCESS;
}

template <bool upper>
ge::graphStatus DoTiling(const optiling::TriluCompileInfo* compileInfo, optiling::TriluTilingParams* params,
                         uint8_t dtypeBytes)
{
    if (dtypeBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    auto eltNumPerBlock = BLOCK_BYTES / dtypeBytes;
    int64_t totalEltNum = (params->matrixNum * params->row * params->col + eltNumPerBlock - 1) / eltNumPerBlock *
                          eltNumPerBlock;

    switch (params->tilingMode) {
        case TILING_MODE_OUTPUT_ZERO:
        case TILING_MODE_OUTPUT_INPUT: {
            params->eltNumPerCore = GetEltNumPerCore(compileInfo->availableUBSize, compileInfo->availableAICoreNum,
                                                     BLOCK_BYTES / dtypeBytes, dtypeBytes, totalEltNum);
            params->taskNum = (totalEltNum + params->eltNumPerCore - 1) / params->eltNumPerCore;
            return ge::GRAPH_SUCCESS;
        }
        case TILING_MODE_SMALL_MATRIX: {
            params->mask = GetMask<upper>(params->row, params->col, params->diagonal);
            int64_t alignedMatrix = GetAlignedSize(params->row * params->col * dtypeBytes);
            params->eltNumPerCore = GetEltNumPerCore(compileInfo->availableUBSize, compileInfo->availableAICoreNum,
                                                     alignedMatrix * params->row * params->col, dtypeBytes,
                                                     totalEltNum);
            params->taskNum = (totalEltNum + params->eltNumPerCore - 1) / params->eltNumPerCore;
            return ge::GRAPH_SUCCESS;
        }
        case TILING_MODE_SMALL_ROW:
        case TILING_MODE_NORMAL:
        case TILING_MODE_BIG_ROW: {
            int64_t alignedRow = GetAlignedSize(params->col * dtypeBytes);
            params->eltNumPerCore = GetEltNumPerCore(compileInfo->availableUBSize, compileInfo->availableAICoreNum,
                                                     alignedRow * params->col, dtypeBytes, totalEltNum);
            if (params->eltNumPerCore == 0) { // the aligned row is too large to put into ub, tiling_mode fallback to 5
                params->tilingMode = TILING_MODE_BIG_ROW;
                params->eltNumPerCore = params->col;
                params->taskNum = params->matrixNum * params->row;
            } else {
                params->taskNum = (totalEltNum + params->eltNumPerCore - 1) / params->eltNumPerCore;
            }
            return ge::GRAPH_SUCCESS;
        }
        default:
            return ge::GRAPH_FAILED;
    }
}

template <bool diagonal_as_input>
void GetDiagonalValue(gert::TilingContext* context, int64_t& diag)
{
    const int64_t* diagPtr = nullptr;
    if (!diagonal_as_input) {
        auto attrs = context->GetAttrs();
        if (attrs != nullptr) {
            diagPtr = attrs->GetInt(INDEX_DIAGONAL);
        }
    } else {
        auto diagTensorPtr = context->GetOptionalInputTensor(INDEX_K);
        if (diagTensorPtr) {
            if (diagTensorPtr->GetDataType() == ge::DT_INT32) {
                const int32_t* kPtr = diagTensorPtr->GetData<int32_t>();
                diag = kPtr ? static_cast<int64_t>(*kPtr) : 0;
                return;
            }
            diagPtr = diagTensorPtr->GetData<int64_t>();
        }
    }
    diag = diagPtr ? *diagPtr : 0;
}

template <bool upper, bool diagonal_as_input>
ge::graphStatus Tiling4Trilu(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4Tril in");
    auto compileInfo = reinterpret_cast<const TriluCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto params = context->GetTilingData<TriluTilingParams>();
    OP_CHECK_NULL_WITH_CONTEXT(context, params);
    auto xDesc = context->GetInputDesc(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    auto xDtypeBytes = GetSizeByDataType(xDtype);
    OP_CHECK_IF(xDtypeBytes == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(),
                                                      "The dtype size of x must be greater than 0."),
                return ge::GRAPH_FAILED);
    auto xShape = context->GetInputShape(INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const auto& xOriginShape = EnsureNotScalar(xShape->GetOriginShape());
    auto dimNum = xOriginShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(dimNum).c_str(),
                                                         "The shape dim of x must be greater than or equal to 2."),
                return ge::GRAPH_FAILED);
    auto row = xOriginShape.GetDim(dimNum - 2);
    auto col = xOriginShape.GetDim(dimNum - 1);
    int64_t matrixNum = 1;
    for (size_t i = 0; i + MIN_DIM_NUM < dimNum; ++i) {
        matrixNum *= xOriginShape.GetDim(i);
    }
    int64_t diag = 0;
    GetDiagonalValue<diagonal_as_input>(context, diag);
    params->row = row;
    params->col = col;
    params->matrixNum = matrixNum;
    params->diagonal = diag;
    OP_CHECK_IF(context == nullptr, OP_LOGE("Triangulator", "context should not be nullptr."), return ge::GRAPH_FAILED);
    TriangulatorTiling tiling(context, upper);
    auto ret = tiling.RunTriangulatorTilingAscendC(compileInfo, params, xDtypeBytes);
    return ret;
}

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_TRILU_H
