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
 * \file addr_v2_tiling.cpp
 * \brief addr_v2 host tiling (arch22 / Ascend910B)
 *
 * Design ref: DESIGN.md §8 (Host Tiling Design)
 *   - 平台信息动态获取：GetCoreNumAiv() / GetCoreMemSize(UB)（禁止硬编码）
 *   - 3 分支 tilingKey 生成（§6.2）: alpha==0 → 0, beta==0 → 1, else → 2
 *   - 多核切分 N 维 former/tail（§6.4）
 *   - UB 切分 M 维 tileM（§6.5 buffer cost）
 *   - self 广播模式判定（§8.2）: 0=[N,M] 1=[1,M] 2=[N,1] 3=[1,1]
 *
 * Validation logic ported from A5 addr_v2_tiling.cpp (CheckDtype/CheckShapes/CheckBroadcast).
 * Tiling calculation ported from verified .asc reference (addr_v2_tiling_calc.h).
 */

#include <algorithm>
#include <cstdint>
#include <string>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "op_host/util/fp16.h"
#include "util/bfloat16.h"
#include "../../op_kernel/arch22/addr_v2_struct.h"
#include "addr_v2_tiling.h"

namespace optiling {

using namespace ge;

constexpr static uint32_t INPUT_IDX_X1 = 0;
constexpr static uint32_t INPUT_IDX_X2 = 1;
constexpr static uint32_t INPUT_IDX_X3 = 2;
constexpr static uint32_t INPUT_IDX_BETA = 3;
constexpr static uint32_t INPUT_IDX_ALPHA = 4;
constexpr static uint32_t OUTPUT_IDX_Y = 0;
constexpr static size_t WORKSPACE_NUM = 1;
constexpr static uint32_t WS_SYS_SIZE = 0U;
constexpr static int32_t DIM_NUM_TWO = 2;

// ============================================================================
// 平台信息获取（动态获取，禁止硬编码）
// ============================================================================
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubCapacity, int64_t* aivCoreCount)
{
    auto* platInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platInfo);
    platform_ascendc::PlatformAscendC aivPlat(platInfo);
    *aivCoreCount = aivPlat.GetCoreNumAiv();
    if (*aivCoreCount <= 0) {
        OP_LOGE(context, "AddrV2: invalid aivCoreCount %ld", *aivCoreCount);
        return ge::GRAPH_FAILED;
    }
    aivPlat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubCapacity);
    if (*ubCapacity == 0) {
        OP_LOGE(context, "AddrV2: ubCapacity is 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// dtype 校验（复用 A5 CheckDtype 逻辑）
// ============================================================================
static bool CheckDtype(gert::TilingContext* context, ge::DataType& x1Dtype)
{
    auto x1Desc = context->GetInputDesc(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    x1Dtype = x1Desc->GetDataType();
    auto x2Desc = context->GetInputDesc(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);
    auto x3Desc = context->GetInputDesc(INPUT_IDX_X3);
    OP_CHECK_NULL_WITH_CONTEXT(context, x3Desc);
    auto betaDesc = context->GetInputDesc(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    auto alphaDesc = context->GetInputDesc(INPUT_IDX_ALPHA);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaDesc);
    auto yDesc = context->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    ge::DataType x2Dtype = x2Desc->GetDataType();
    ge::DataType x3Dtype = x3Desc->GetDataType();
    ge::DataType betaDtype = betaDesc->GetDataType();
    ge::DataType alphaDtype = alphaDesc->GetDataType();
    ge::DataType yDtype = yDesc->GetDataType();

    if (x1Dtype != x2Dtype || x2Dtype != x3Dtype || x3Dtype != betaDtype || betaDtype != alphaDtype ||
        alphaDtype != yDtype) {
        OP_LOGE(context, "AddrV2: dtypes of x1, x2, x3, beta, alpha and y must be the same");
        return false;
    }

    // 校验 dtype 在支持列表内
    const std::set<ge::DataType> supported = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                              ge::DT_INT8,  ge::DT_UINT8,   ge::DT_BOOL};
    if (supported.count(x1Dtype) == 0) {
        OP_LOGE(context, "AddrV2: unsupported dtype %d", static_cast<int>(x1Dtype));
        return false;
    }
    return true;
}

// ============================================================================
// shape 校验（复用 A5 CheckShapes/CheckBroadcast 逻辑）
// ============================================================================
static bool CheckShapes(gert::TilingContext* context, int64_t& N, int64_t& M, uint32_t& selfBcastMode)
{
    auto x1Shape = context->GetInputShape(INPUT_IDX_X1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    auto x2Shape = context->GetInputShape(INPUT_IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    auto x3Shape = context->GetInputShape(INPUT_IDX_X3);
    OP_CHECK_NULL_WITH_CONTEXT(context, x3Shape);
    auto yShape = context->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    N = x2Shape->GetStorageShape().GetDim(0);
    M = x3Shape->GetStorageShape().GetDim(0);

    // 校验空 tensor
    if (x1Shape->GetStorageShape().GetShapeSize() <= 0 || x2Shape->GetStorageShape().GetShapeSize() <= 0 ||
        x3Shape->GetStorageShape().GetShapeSize() <= 0 || yShape->GetStorageShape().GetShapeSize() <= 0) {
        OP_LOGE(context, "AddrV2: x1, x2, x3 and y do not support empty tensors");
        return false;
    }

    // 校验 x2, x3 为 1 维
    if (x2Shape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context, "AddrV2: x2 must be 1D");
        return false;
    }
    if (x3Shape->GetStorageShape().GetDimNum() != 1) {
        OP_LOGE(context, "AddrV2: x3 must be 1D");
        return false;
    }

    // 校验 x1 维度和 broadcast
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    if (x1DimNum == 1) {
        int64_t inM = x1Shape->GetStorageShape().GetDim(0);
        if (inM != 1 && inM != M) {
            OP_LOGE(context, "AddrV2: x1 shape [%ld] not broadcastable to [%ld, %ld]", inM, N, M);
            return false;
        }
        selfBcastMode = (inM == 1) ? ADDR_V2_BCAST_SCALAR : ADDR_V2_BCAST_ROW;
    } else if (x1DimNum == DIM_NUM_TWO) {
        int64_t inN = x1Shape->GetStorageShape().GetDim(0);
        int64_t inM = x1Shape->GetStorageShape().GetDim(1);
        if ((inN != 1 && inN != N) || (inM != 1 && inM != M)) {
            OP_LOGE(context, "AddrV2: x1 shape [%ld, %ld] not broadcastable to [%ld, %ld]", inN, inM, N, M);
            return false;
        }
        if (inN == 1 && inM == 1) {
            selfBcastMode = ADDR_V2_BCAST_SCALAR;
        } else if (inN == 1) {
            selfBcastMode = ADDR_V2_BCAST_ROW;
        } else if (inM == 1) {
            selfBcastMode = ADDR_V2_BCAST_COL;
        } else {
            selfBcastMode = ADDR_V2_BCAST_NONE;
        }
    } else {
        OP_LOGE(context, "AddrV2: x1 dim should be 1 or 2");
        return false;
    }

    // 校验 y shape
    if (yShape->GetStorageShape().GetDimNum() != DIM_NUM_TWO || yShape->GetStorageShape().GetDim(0) != N ||
        yShape->GetStorageShape().GetDim(1) != M) {
        OP_LOGE(context, "AddrV2: y shape mismatch, expected [%ld, %ld]", N, M);
        return false;
    }
    return true;
}

// ============================================================================
// 获取 beta/alpha 标量值（复用 A5 GetConstData 逻辑）
// ============================================================================
template <typename T>
static bool GetConstValue(gert::TilingContext* context, uint32_t inputIdx, bool isEmpty, T emptyValue, T& data)
{
    if (isEmpty) {
        data = emptyValue;
    } else {
        auto tensor = context->GetInputTensor(inputIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
        const T* value = tensor->GetData<T>();
        OP_CHECK_NULL_WITH_CONTEXT(context, value);
        data = value[0];
    }
    return true;
}

static bool GetBetaAlpha(gert::TilingContext* context, ge::DataType dtype, bool betaEmpty, bool alphaEmpty, float& beta,
                         float& alpha)
{
    if (dtype == ge::DT_FLOAT) {
        float tmpBeta = 0, tmpAlpha = 0;
        if (!GetConstValue<float>(context, INPUT_IDX_BETA, betaEmpty, 1.0f, tmpBeta) ||
            !GetConstValue<float>(context, INPUT_IDX_ALPHA, alphaEmpty, 1.0f, tmpAlpha)) {
            return false;
        }
        beta = tmpBeta;
        alpha = tmpAlpha;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        uint16_t tmpBeta = 0, tmpAlpha = 0;
        if (!GetConstValue<uint16_t>(context, INPUT_IDX_BETA, betaEmpty, (dtype == ge::DT_FLOAT16) ? 0x3C00 : 0x3F80,
                                     tmpBeta) ||
            !GetConstValue<uint16_t>(context, INPUT_IDX_ALPHA, alphaEmpty, (dtype == ge::DT_FLOAT16) ? 0x3C00 : 0x3F80,
                                     tmpAlpha)) {
            return false;
        }
        if (dtype == ge::DT_FLOAT16) {
            beta = float(*(reinterpret_cast<Ops::Base::fp16_t*>(&tmpBeta)));
            alpha = float(*(reinterpret_cast<Ops::Base::fp16_t*>(&tmpAlpha)));
        } else {
            beta = float(*(reinterpret_cast<Ops::Base::bfloat16*>(&tmpBeta)));
            alpha = float(*(reinterpret_cast<Ops::Base::bfloat16*>(&tmpAlpha)));
        }
    } else if (dtype == ge::DT_INT8 || dtype == ge::DT_BOOL) {
        int8_t tmpBeta = 0, tmpAlpha = 0;
        if (!GetConstValue<int8_t>(context, INPUT_IDX_BETA, betaEmpty, int8_t(1), tmpBeta) ||
            !GetConstValue<int8_t>(context, INPUT_IDX_ALPHA, alphaEmpty, int8_t(1), tmpAlpha)) {
            return false;
        }
        beta = static_cast<float>(tmpBeta);
        alpha = static_cast<float>(tmpAlpha);
    } else if (dtype == ge::DT_UINT8) {
        uint8_t tmpBeta = 0, tmpAlpha = 0;
        if (!GetConstValue<uint8_t>(context, INPUT_IDX_BETA, betaEmpty, uint8_t(1), tmpBeta) ||
            !GetConstValue<uint8_t>(context, INPUT_IDX_ALPHA, alphaEmpty, uint8_t(1), tmpAlpha)) {
            return false;
        }
        beta = static_cast<float>(tmpBeta);
        alpha = static_cast<float>(tmpAlpha);
    } else {
        OP_LOGE(context, "AddrV2: unsupported dtype for const data");
        return false;
    }
    return true;
}

// ============================================================================
// 按 (dtype 类别, tilingKey) 计算 UB 内每元素字节数（DESIGN.md §6.5）
// 移植自 .asc reference BufferCostPerElem
// ============================================================================
static uint32_t BufferCostPerElem(ge::DataType dtype, uint32_t tilingKey)
{
    bool isFloat = (dtype == ge::DT_FLOAT);
    bool isFp16Bf16 = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16);
    bool isIntU8 = (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8 || dtype == ge::DT_BOOL);
    uint32_t dtypeSize = isFloat ? 4 : (isFp16Bf16 ? 2 : 1);

    const uint32_t vec2Depth = 1;
    const uint32_t selfDepth = 2;
    const uint32_t outDepth = 2;

    bool needVec2 = (tilingKey != ADDR_V2_TILING_KEY_WITHOUT_ALPHA);
    bool needSelf = (tilingKey != ADDR_V2_TILING_KEY_WITHOUT_BETA);

    uint32_t queBytes = 0;
    if (needVec2) {
        queBytes += vec2Depth * dtypeSize;
    }
    if (needSelf) {
        queBytes += selfDepth * dtypeSize;
    }
    queBytes += outDepth * dtypeSize;

    uint32_t fp32BufCount = 0;
    if (isFloat) {
        fp32BufCount = 0;
    } else if (isFp16Bf16) {
        if (needVec2) {
            fp32BufCount += 1;
        }
        if (needSelf) {
            fp32BufCount += 1;
        }
        if (needVec2) {
            fp32BufCount += 1;
        }
        fp32BufCount += 1;
    } else if (isIntU8) {
        if (needVec2) {
            fp32BufCount += 1;
        }
        if (needSelf) {
            fp32BufCount += 1;
        }
        if (needVec2) {
            fp32BufCount += 1;
        }
        fp32BufCount += 1;
        fp32BufCount += 2;
    }
    uint32_t fp32Bytes = fp32BufCount * 4;
    uint32_t halfBytes = isIntU8 ? 2 : 0;

    return queBytes + fp32Bytes + halfBytes;
}

// ============================================================================
// 主 Tiling 函数
// ============================================================================
static ge::graphStatus AddrV2TilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter AddrV2TilingFunc");

    // 1. 平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "AddrV2: GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. dtype 校验
    ge::DataType x1Dtype = ge::DT_UNDEFINED;
    OP_CHECK_IF(!CheckDtype(context, x1Dtype), OP_LOGE(context, "AddrV2: CheckDtype error"), return ge::GRAPH_FAILED);

    // 3. shape 校验 + 广播模式判定
    int64_t N = 0, M = 0;
    uint32_t selfBcastMode = ADDR_V2_BCAST_NONE;
    OP_CHECK_IF(!CheckShapes(context, N, M, selfBcastMode), OP_LOGE(context, "AddrV2: CheckShapes error"),
                return ge::GRAPH_FAILED);

    // 3.1 GM 偏移溢出校验：kernel 已修复为 64 位地址，但仍进行合理上限校验
    //     防止异常输入导致 N*M 超过 UINT32_MAX（约 4.3e9）
    constexpr int64_t UINT32_MAX_VAL = static_cast<int64_t>(0xFFFFFFFF);
    if (N > 0 && M > 0 && N * M > UINT32_MAX_VAL) {
        OP_LOGE(context, "AddrV2: total elements N*M=%ld exceeds UINT32_MAX, not supported", N * M);
        return ge::GRAPH_FAILED;
    }

    // 4. workspace
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = WS_SYS_SIZE;

    // 5. beta/alpha 标量值
    auto betaShape = context->GetInputShape(INPUT_IDX_BETA);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaShape);
    auto alphaShape = context->GetInputShape(INPUT_IDX_ALPHA);
    OP_CHECK_NULL_WITH_CONTEXT(context, alphaShape);
    bool betaEmpty = (betaShape->GetStorageShape().GetShapeSize() == 0);
    bool alphaEmpty = (alphaShape->GetStorageShape().GetShapeSize() == 0);

    float beta = 0, alpha = 0;
    OP_CHECK_IF(!GetBetaAlpha(context, x1Dtype, betaEmpty, alphaEmpty, beta, alpha),
                OP_LOGE(context, "AddrV2: GetBetaAlpha error"), return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "AddrV2: beta=%f, alpha=%f, N=%ld, M=%ld", beta, alpha, N, M);

    // 6. tilingKey 生成（alpha==0 优先，依据 §6.2）
    uint32_t tilingKey = 0;
    if (alpha == 0.0f) {
        tilingKey = ADDR_V2_TILING_KEY_WITHOUT_ALPHA;
    } else if (beta == 0.0f) {
        tilingKey = ADDR_V2_TILING_KEY_WITHOUT_BETA;
    } else {
        tilingKey = ADDR_V2_TILING_KEY_WITH_BETA_ALPHA;
    }

    // 7. TilingData
    AddrV2TilingData* tiling = context->GetTilingData<AddrV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AddrV2TilingData), 0, sizeof(AddrV2TilingData)) != EOK,
                OP_LOGE(context, "AddrV2: memset tiling failed"), return ge::GRAPH_FAILED);

    tiling->totalRows = static_cast<uint32_t>(N);
    tiling->totalCols = static_cast<uint32_t>(M);
    tiling->tilingKey = tilingKey;
    tiling->selfBroadcastMode = selfBcastMode;
    tiling->betaValue = beta;
    tiling->alphaValue = alpha;

    // 8. 多核切分 N 维 former/tail（依据 §6.4）
    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, N));
    if (blockDim == 0) {
        blockDim = 1;
    }
    tiling->blockDim = blockDim;

    uint32_t tailRows = static_cast<uint32_t>(N) / blockDim;
    uint32_t formerRows = tailRows + ((static_cast<uint32_t>(N) % blockDim != 0) ? 1 : 0);
    uint32_t formerNum = static_cast<uint32_t>(N) - tailRows * blockDim;
    tiling->formerNum = formerNum;
    tiling->formerRows = formerRows;
    tiling->tailRows = tailRows;

    // 9. UB 切分 M 维 tileM（依据 §6.5）
    uint32_t dtypeSize = (x1Dtype == ge::DT_FLOAT) ? 4 : (x1Dtype == ge::DT_FLOAT16 || x1Dtype == ge::DT_BF16) ? 2 : 1;
    uint32_t alignElem = ADDR_V2_ALIGN_BYTES / dtypeSize;
    if (alignElem == 0) {
        alignElem = 1;
    }

    uint32_t cost = BufferCostPerElem(x1Dtype, tilingKey);
    if (cost == 0) {
        cost = 1;
    }

    int64_t availUb = static_cast<int64_t>(ubSize) - 1024;
    if (availUb < 1024) {
        availUb = 1024;
    }
    uint32_t tileMMax = static_cast<uint32_t>(availUb / static_cast<int64_t>(cost));

    uint32_t tileM;
    if (static_cast<uint32_t>(M) <= tileMMax) {
        tileM = static_cast<uint32_t>(M);
    } else {
        tileM = (tileMMax / alignElem) * alignElem;
        if (tileM == 0) {
            tileM = alignElem;
        }
    }
    tiling->tileM = tileM;
    tiling->tileMLoop = (static_cast<uint32_t>(M) + tileM - 1) / tileM;
    tiling->tileMTail = static_cast<uint32_t>(M) - (tiling->tileMLoop - 1) * tileM;

    // 10. BlockDim
    context->SetBlockDim(blockDim);

    // 11. TilingKey: 按 dtype 维度（编译期实例化选择）
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(x1Dtype));

    OP_LOGI(context->GetNodeName(), "AddrV2: blockDim=%u, tileM=%u, tileMLoop=%u, tilingKey=%u, bcastMode=%u", blockDim,
            tileM, tiling->tileMLoop, tilingKey, selfBcastMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddrV2(gert::TilingParseContext* context)
{
    auto* compileInfo = context->GetCompiledInfo<AddrV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto* platInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platInfo);
    platform_ascendc::PlatformAscendC aivPlat(platInfo);
    compileInfo->coreNum = aivPlat.GetCoreNumAiv();
    aivPlat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddrV2)
    .Tiling(AddrV2TilingFunc)
    .TilingParse<AddrV2CompileInfo>(TilingParseForAddrV2)
    .TilingInputsDataDependency({INPUT_IDX_BETA, INPUT_IDX_ALPHA});

} // namespace optiling
