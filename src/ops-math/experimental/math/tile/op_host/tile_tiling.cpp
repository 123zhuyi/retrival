/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tile_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "../op_kernel/tile_tiling_data.h"
#include "../op_kernel/tile_tiling_key.h"
#include <vector>
#include <cstring>

namespace optiling {

// ========== Hardware / platform constants ==========
// UB is split into two buffers (input + output) for ping-pong operation
constexpr int32_t TILE_BUFFER_NUM = 2;
// Ascend DMA minimum transfer granularity: 32 bytes (one data block)
constexpr int32_t DATA_BLOCK_BYTES = 32;
// Gather offset table is uint32-aligned, grouped in units of 8 elements
constexpr int32_t BYTE_ALIGN_SIZE = 8;
// Vector register width is 256B: holds 128 fp16 elements or 64 fp32 elements
constexpr int32_t VREG_WIDTH_FP16 = 128;
constexpr int32_t VREG_WIDTH_FP32 = 64;
// Each Gather offset table entry occupies 4 bytes (uint32)
constexpr int32_t OFFSET_UINT32_SIZE = 4;

// ========== Data type byte sizes ==========
constexpr int32_t ELEM_BYTES_INT8 = 1;
constexpr int32_t ELEM_BYTES_FP16 = 2;
constexpr int32_t ELEM_BYTES_FP32 = 4;

// ========== Scheduling mode selection thresholds ==========
// The following thresholds are derived from performance profiling on Atlas A2 (Ascend 910B)
// and are used to choose the most efficient processing path at runtime.

// innerDim element-count thresholds that control whether the ScalarBuild path is enabled:
//   <=24: enabled for 1/2-byte types (SetValue loop overhead is acceptable)
//   <=32: relaxed for 1/2-byte types with even innerDim (uint32 packing optimization applies)
//   <=64: relaxed for 4-byte types (per-element SetValue cost is relatively lower)
constexpr int32_t SMALL_INNER_DIM_THRESHOLD = 24;
constexpr int32_t MODERATE_INNER_DIM_THRESHOLD = 32;
constexpr int32_t LARGE_INNER_DIM_THRESHOLD = 64;

// Minimum workload for the ScalarBuild path: total input elements or per-core rows must reach 256;
// prevents using the scalar path on very small tensors where extra logic overhead dominates
constexpr int32_t MIN_INPUT_ELEMS_FOR_SCALAR = 256;
constexpr int32_t MIN_ROWS_PER_CORE_FOR_SCALAR = 256;

// Minimum amplification ratio (repeatPeriod / repeatInputPeriod) for seed-amplify to pay off:
//   >=4: general paths (one seed write reused >=4 times justifies the extra read-back)
//   >=3: VecGather path has a lower bar (Gather instruction overhead is high, needs less reuse)
constexpr int32_t MIN_PERIOD_RATIO_FOR_BENEFIT = 4;
constexpr int32_t MIN_PERIOD_RATIO_FOR_VECGATHER = 3;

// VecGather path constraints:
//   seed rows <=40 (offset table must stay in UB; too many rows squeeze data space)
//   output row width >=125 elements (short rows make Gather startup cost disproportionate)
//   only 2/4-byte types supported (Gather instruction requires element size >= 2 bytes)
constexpr int32_t MAX_PERIOD_FOR_VECGATHER = 40;
constexpr int32_t MIN_OUTPUT_INNER_DIM_FOR_VECGATHER = 125;
constexpr int32_t MIN_ELEM_BYTES_FOR_VECGATHER = 2;
constexpr int32_t MAX_ELEM_BYTES_FOR_VECGATHER = 4;

// Readback path requires innerDim >= 12 elements so that each DMA transfer covers
// several data blocks; below this threshold DMA startup cost exceeds scalar build
constexpr int32_t MIN_INNER_DIM_FOR_READBACK = 12;

// DmaBuild path requires innerDim >= 11 and seed rows >= 50:
//   too-small innerDim makes per-row DMA writes inefficient
//   too few seed rows means read-back amplification cannot recoup the extra GM I/O
constexpr int32_t MIN_INNER_DIM_FOR_DMABUILD = 11;
constexpr int32_t MIN_REP_INPUT_PERIOD_FOR_DMABUILD = 50;

// ========== Multi-core grouping / period reuse thresholds ==========

// Amplification >= 3 needed for source-grouping (periodsPerSource) to outweigh management overhead
constexpr int32_t MIN_AMPLIFICATION_RATIO = 3;
// When amplification <= 2, try to find a dimension with higher amplification
constexpr int32_t MIN_AMP_THRESHOLD = 2;

// Unique-source count thresholds for enabling source-grouping optimization:
//   >= 8: standard bar — enough sources to distribute evenly across cores
//   >= 4: relaxed for int8 / small-element types (small per-transfer volume needs more parallelism)
constexpr int32_t MIN_NSRC_THRESHOLD = 8;
constexpr int32_t MIN_NSRC_SMALL = 4;

// Periods-per-source (pps) range: too large causes excessive serial work per core
constexpr int32_t MAX_PPS_THRESHOLD = 20;
// int8 types require pps >= 8 before small-source grouping is worthwhile
constexpr int32_t MIN_PPS_THRESHOLD = 8;

// Source grouping only shows clear benefit when seed rows (repeatInputPeriod) >= 100
constexpr int32_t MIN_RIP_FOR_BENEFIT = 100;

// ========== Output size thresholds ==========

// Total output <= 2 KB: force single core (multi-core scheduling overhead exceeds computation)
constexpr int32_t SMALL_OUTPUT_BYTES_THRESHOLD = 2048;
// Total output <= 4 KB: cap core count in splitByMult mode to avoid over-partitioning
constexpr int32_t MEDIUM_OUTPUT_BYTES_THRESHOLD = 4096;

// Seed template <= 256 B is considered too small; cap blockDim to AIC core count
// to avoid repeated scheduling overhead for tiny templates
constexpr int32_t SMALL_TEMPLATE_BYTES = 256;
// Total output >= 64 KB: Readback path DMA read-back amplification has enough payoff
constexpr int32_t LARGE_OUTPUT_BYTES_THRESHOLD = 65536;

// In BuildOnce mode, merge cores only when period count >= 10 to ensure >= 2 periods per core
constexpr int32_t MIN_PERIODS_FOR_MERGE = 10;

struct TileCompileInfo {};

static ge::graphStatus TilingParseForTile([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum, int64_t& aicCoreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    aicCoreNum = ascendcPlatform.GetCoreNumAic();
    if (aicCoreNum <= 0) {
        aicCoreNum = coreNum;
    }
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static void MergeConsecutiveDims(std::vector<int32_t>& shape, std::vector<int32_t>& multiples)
{
    bool changed = true;
    while (changed) {
        changed = false;
        for (int32_t idx = static_cast<int32_t>(shape.size()) - 1; idx >= 1; idx--) {
            if (multiples[idx] == 1) {
                shape[idx - 1] *= shape[idx];
                shape.erase(shape.begin() + idx);
                multiples.erase(multiples.begin() + idx);
                changed = true;
                break;
            }
        }
    }
}

static void AlignLastDimForDMA(std::vector<int32_t>& shape, std::vector<int32_t>& multiples, int32_t elemBytes)
{
    while (shape.size() >= 2) {
        int32_t lastIdx = static_cast<int32_t>(shape.size()) - 1;
        if (shape[lastIdx] * elemBytes % DATA_BLOCK_BYTES == 0) {
            break;
        }
        int32_t secLast = lastIdx - 1;
        if (multiples[secLast] != 1) {
            break;
        }
        shape[lastIdx] = shape[secLast] * shape[lastIdx];
        shape.erase(shape.begin() + secLast);
        multiples.erase(multiples.begin() + secLast);
    }
}

static void RemoveRedundantDims(std::vector<int32_t>& shape, std::vector<int32_t>& multiples)
{
    std::vector<int32_t> newShape;
    std::vector<int32_t> newMult;
    for (size_t idx = 0; idx < shape.size(); idx++) {
        if (shape[idx] == 1 && multiples[idx] == 1) {
            continue;
        }
        if (!newShape.empty() && newShape.back() == 1 && shape[idx] == 1) {
            newMult.back() *= multiples[idx];
        } else {
            newShape.push_back(shape[idx]);
            newMult.push_back(multiples[idx]);
        }
    }
    if (newShape.empty()) {
        newShape.push_back(1);
        newMult.push_back(1);
    }
    shape = newShape;
    multiples = newMult;
}

static void MergeDims(std::vector<int32_t>& shape, std::vector<int32_t>& multiples, int32_t elemBytes)
{
    MergeConsecutiveDims(shape, multiples);
    AlignLastDimForDMA(shape, multiples, elemBytes);
    RemoveRedundantDims(shape, multiples);
}

static void ComputeTilingParams(
    TileTilingData& tilingData, const std::vector<int32_t>& origShape, const std::vector<int32_t>& origMult,
    int32_t elemBytes, int32_t blockDim, int32_t ubSize, int32_t aicBlockDim)
{
    (void)aicBlockDim;
    std::vector<int32_t> shape = origShape;
    std::vector<int32_t> multiples = origMult;
    MergeDims(shape, multiples, elemBytes);

    int32_t ndim = static_cast<int32_t>(shape.size());
    tilingData.numDims = ndim;
    tilingData.elemBytes = elemBytes;
    tilingData.blockDim = blockDim;
    tilingData.ubSize = ubSize;
    tilingData.totalInputElems = 1;
    tilingData.totalOutputElems = 1;
    for (int32_t idx = 0; idx < ndim; idx++) {
        tilingData.inputShape[idx] = shape[idx];
        tilingData.multiples[idx] = multiples[idx];
        tilingData.outputShape[idx] = shape[idx] * multiples[idx];
        tilingData.totalInputElems *= shape[idx];
        tilingData.totalOutputElems *= tilingData.outputShape[idx];
    }
    tilingData.inputStrides[ndim - 1] = 1;
    tilingData.outputStrides[ndim - 1] = 1;
    for (int32_t idx = ndim - 2; idx >= 0; idx--) {
        tilingData.inputStrides[idx] = tilingData.inputStrides[idx + 1] * tilingData.inputShape[idx + 1];
        tilingData.outputStrides[idx] = tilingData.outputStrides[idx + 1] * tilingData.outputShape[idx + 1];
    }

    int32_t innerDim = tilingData.inputShape[ndim - 1];
    int32_t innerMult = tilingData.multiples[ndim - 1];
    int32_t outputInnerDim = innerDim * innerMult;
    int32_t outerCount = (outputInnerDim > 0) ? (tilingData.totalOutputElems / outputInnerDim) : 0;
    int32_t alignElems = (elemBytes > 0) ? (DATA_BLOCK_BYTES / elemBytes) : 1;
    if (alignElems < 1) {
        alignElems = 1;
    }
    int32_t innerDimAligned = ((innerDim + alignElems - 1) / alignElems) * alignElems;
    int32_t bufElems = (elemBytes > 0) ? (ubSize / elemBytes) : 0;
    bool splitByMult = (outerCount < blockDim && innerMult > 1 && innerDimAligned <= bufElems);
    int32_t totalWork = splitByMult ? (outerCount * innerMult) : outerCount;
    if (!splitByMult && innerDimAligned > bufElems && outerCount < blockDim) {
        int32_t chunkSize = (bufElems / alignElems) * alignElems;
        if (chunkSize > 0) {
            int32_t numChunks = (innerDim + chunkSize - 1) / chunkSize;
            totalWork = outerCount * numChunks;
        }
    }
    if (totalWork < blockDim) {
        tilingData.blockDim = (totalWork > 0) ? totalWork : 1;
    }
    if (!splitByMult) {
        int64_t outputBytes = static_cast<int64_t>(tilingData.totalOutputElems) * elemBytes;
        if (outputBytes <= SMALL_OUTPUT_BYTES_THRESHOLD && tilingData.blockDim > 1) {
            tilingData.blockDim = 1;
        }
    }
    if (splitByMult && tilingData.blockDim > outerCount && outerCount >= 2) {
        int64_t outputBytes = static_cast<int64_t>(tilingData.totalOutputElems) * elemBytes;
        if (outputBytes <= MEDIUM_OUTPUT_BYTES_THRESHOLD ||
            (innerDim * elemBytes >= DATA_BLOCK_BYTES &&
             static_cast<int64_t>(outerCount) * outputInnerDim * elemBytes <= ubSize)) {
            tilingData.blockDim = outerCount;
        }
    }

    tilingData.repeatPeriod = 0;
    tilingData.repeatInputPeriod = 0;
    tilingData.periodsPerSource = 0;
    tilingData.nUniqueSources = 0;
    if (!splitByMult && ndim >= 2) {
        int32_t rowsPerCore = (outerCount + tilingData.blockDim - 1) / tilingData.blockDim;
        int32_t bestIPeriod = rowsPerCore + 1;
        int32_t bestRepeatDim = -1;
        for (int32_t d = ndim - 2; d >= 0; d--) {
            if (multiples[d] <= 1) {
                continue;
            }
            int32_t stride_d = 1;
            for (int32_t dd = d + 1; dd < ndim - 1; dd++) {
                stride_d *= tilingData.outputShape[dd];
            }
            int32_t iPeriod = shape[d] * stride_d;
            int32_t oPeriod = tilingData.outputShape[d] * stride_d;
            if (iPeriod <= 0 || iPeriod > rowsPerCore) {
                continue;
            }
            if (oPeriod <= iPeriod) {
                continue;
            }
            if (iPeriod < bestIPeriod) {
                bestIPeriod = iPeriod;
                tilingData.repeatPeriod = oPeriod;
                tilingData.repeatInputPeriod = iPeriod;
                bestRepeatDim = d;
            }
        }
        if (bestRepeatDim >= 0) {
            int32_t bestAmp = tilingData.repeatPeriod / tilingData.repeatInputPeriod;
            if (bestAmp <= MIN_AMP_THRESHOLD) {
                for (int32_t d = ndim - 2; d >= 0; d--) {
                    if (multiples[d] <= 1) {
                        continue;
                    }
                    int32_t stride_d = 1;
                    for (int32_t dd = d + 1; dd < ndim - 1; dd++) {
                        stride_d *= tilingData.outputShape[dd];
                    }
                    int32_t iPeriod = shape[d] * stride_d;
                    int32_t oPeriod = tilingData.outputShape[d] * stride_d;
                    if (iPeriod <= 0 || iPeriod > rowsPerCore || oPeriod <= iPeriod) {
                        continue;
                    }
                    int32_t amp = oPeriod / iPeriod;
                    if (amp <= bestAmp) {
                        continue;
                    }
                    int32_t totalP = outerCount / oPeriod;
                    if (totalP < tilingData.blockDim / MIN_NSRC_THRESHOLD) {
                        continue;
                    }
                    tilingData.repeatPeriod = oPeriod;
                    tilingData.repeatInputPeriod = iPeriod;
                    bestRepeatDim = d;
                    break;
                }
            }
            int32_t pps = 1;
            bool ppsValid = true;
            for (int32_t dd = 0; dd < bestRepeatDim; dd++) {
                if (shape[dd] == 1) {
                    pps *= multiples[dd];
                } else {
                    ppsValid = false;
                    break;
                }
            }
            if (!ppsValid) {
                pps = 1;
                int32_t candidatePPS = 1;
                for (int32_t dd = 0; dd < bestRepeatDim; dd++) {
                    candidatePPS *= multiples[dd];
                }
                int32_t nUnique = 1;
                for (int32_t dd = 0; dd < bestRepeatDim; dd++) {
                    nUnique *= shape[dd];
                }
                int32_t rP_local = tilingData.repeatPeriod;
                int32_t totalP_local = (rP_local > 0) ? (outerCount / rP_local) : 0;
                int32_t candidateNSrc = (candidatePPS > 0) ? (totalP_local / candidatePPS) : 0;
                if (candidateNSrc > 1 && candidateNSrc == nUnique && totalP_local == candidateNSrc * candidatePPS) {
                    int32_t checkIdx = candidateNSrc * rP_local;
                    int32_t srcOff1 = 0;
                    int32_t tmpVal = checkIdx;
                    for (int32_t d2 = ndim - 2; d2 >= 0; d2--) {
                        int32_t oc = tmpVal % tilingData.outputShape[d2];
                        tmpVal /= tilingData.outputShape[d2];
                        srcOff1 += (oc % tilingData.inputShape[d2]) * tilingData.inputStrides[d2];
                    }
                    if (srcOff1 == 0) {
                        pps = candidatePPS;
                    }
                }
            }
            int32_t riP = tilingData.repeatInputPeriod;
            int32_t rP = tilingData.repeatPeriod;
            int32_t totalP = (rP > 0) ? (outerCount / rP) : 0;
            int32_t nSrc = (pps > 0 && totalP > 0 && totalP % pps == 0) ? totalP / pps : 0;
            if ((nSrc >= MIN_NSRC_THRESHOLD && nSrc <= tilingData.blockDim && riP >= MIN_RIP_FOR_BENEFIT &&
                 pps <= MAX_PPS_THRESHOLD) ||
                (nSrc >= MIN_NSRC_THRESHOLD && nSrc <= tilingData.blockDim && elemBytes == ELEM_BYTES_INT8) ||
                (nSrc >= MIN_NSRC_SMALL && nSrc <= tilingData.blockDim && elemBytes == ELEM_BYTES_INT8 &&
                 pps >= MIN_PPS_THRESHOLD) ||
                (nSrc >= MIN_NSRC_SMALL && nSrc <= tilingData.blockDim && pps > MIN_NSRC_SMALL &&
                 (rP / riP) >= MIN_AMPLIFICATION_RATIO && static_cast<int64_t>(rP) * outputInnerDim <= bufElems) ||
                (nSrc >= MIN_NSRC_THRESHOLD && nSrc <= tilingData.blockDim && pps <= MAX_PPS_THRESHOLD &&
                 totalP >= tilingData.blockDim * 2) ||
                (nSrc > tilingData.blockDim && elemBytes <= ELEM_BYTES_FP16 && pps <= MAX_PPS_THRESHOLD &&
                 (nSrc + tilingData.blockDim - 1) / tilingData.blockDim <= pps)) {
                tilingData.periodsPerSource = pps;
            }
            if (bestRepeatDim >= 0) {
                int32_t nUniqueCalc = 1;
                for (int32_t dd = 0; dd < bestRepeatDim; dd++) {
                    nUniqueCalc *= shape[dd];
                }
                if (nUniqueCalc > 1 && nUniqueCalc < totalP) {
                    tilingData.nUniqueSources = nUniqueCalc;
                }
            }
        }
        if (bestRepeatDim < 0 && innerDimAligned <= bufElems) {
            bool smallID =
                (innerDim <= SMALL_INNER_DIM_THRESHOLD ||
                 (elemBytes >= ELEM_BYTES_FP32 && innerDim <= LARGE_INNER_DIM_THRESHOLD));
            int32_t candidateIP = outerCount + 1;
            for (int32_t d = ndim - 2; d >= 0; d--) {
                if (multiples[d] <= 1) {
                    continue;
                }
                int32_t stride_d = 1;
                for (int32_t dd = d + 1; dd < ndim - 1; dd++) {
                    stride_d *= tilingData.outputShape[dd];
                }
                int32_t iPeriod = shape[d] * stride_d;
                int32_t oPeriod = tilingData.outputShape[d] * stride_d;
                if (iPeriod <= 0 || oPeriod <= iPeriod) {
                    continue;
                }
                int32_t amp = oPeriod / iPeriod;
                if (amp < MIN_AMPLIFICATION_RATIO) {
                    continue;
                }
                int32_t neededBD = outerCount / iPeriod;
                if (neededBD < 1)
                    neededBD = 1;
                if (neededBD >= tilingData.blockDim) {
                    continue;
                }
                if (!smallID) {
                    continue;
                }
                if (iPeriod < candidateIP) {
                    candidateIP = iPeriod;
                    tilingData.repeatPeriod = oPeriod;
                    tilingData.repeatInputPeriod = iPeriod;
                    bestRepeatDim = d;
                    tilingData.blockDim = neededBD;
                }
            }
        }
    }
    if (!splitByMult && tilingData.blockDim > 1 && outerCount > 0 && innerDimAligned <= bufElems) {
        int32_t rPC = (outerCount + tilingData.blockDim - 1) / tilingData.blockDim;
        int32_t rowBytes = outputInnerDim * elemBytes;
        if (rowBytes > 0 && rowBytes % DATA_BLOCK_BYTES != 0 && rPC > 1) {
            int32_t g = rowBytes, bv = DATA_BLOCK_BYTES;
            while (bv) {
                int32_t t = bv;
                bv = g % bv;
                g = t;
            }
            int32_t aStep = DATA_BLOCK_BYTES / g;
            if (aStep > 1 && rPC > aStep) {
                rPC = ((rPC + aStep - 1) / aStep) * aStep;
            }
        }
        int32_t effBD = (rPC > 0) ? ((outerCount + rPC - 1) / rPC) : 1;
        if (effBD > 0 && effBD < tilingData.blockDim) {
            tilingData.blockDim = effBD;
        }
    }
}

static ge::graphStatus TileTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    int64_t aicCoreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum, aicCoreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dtype, typeLength);
    int32_t elemBytes = static_cast<int32_t>(typeLength);
    OP_CHECK_IF(elemBytes <= 0, OP_LOGE(context, "unsupported data type, elemBytes <= 0"), return ge::GRAPH_FAILED);

    int32_t ndim = static_cast<int32_t>(inputShape->GetStorageShape().GetDimNum());
    std::vector<int32_t> shape(ndim);
    for (int32_t idx = 0; idx < ndim; idx++) {
        shape[idx] = static_cast<int32_t>(inputShape->GetStorageShape().GetDim(idx));
    }

    const gert::Tensor* multiplesTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, multiplesTensor);
    const void* multiplesAddr = multiplesTensor->GetAddr();
    OP_CHECK_NULL_WITH_CONTEXT(context, multiplesAddr);

    auto multiplesDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, multiplesDesc);
    ge::DataType multiplesDtype = multiplesDesc->GetDataType();

    const gert::StorageShape* multiplesShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, multiplesShape);
    int32_t multiplesLen = static_cast<int32_t>(multiplesShape->GetStorageShape().GetDim(0));
    std::vector<int32_t> mult(multiplesLen);
    if (multiplesDtype == ge::DT_INT64) {
        const int64_t* multiplesData64 = reinterpret_cast<const int64_t*>(multiplesAddr);
        for (int32_t idx = 0; idx < multiplesLen; idx++) {
            mult[idx] = static_cast<int32_t>(multiplesData64[idx]);
        }
    } else {
        const int32_t* multiplesData32 = reinterpret_cast<const int32_t*>(multiplesAddr);
        for (int32_t idx = 0; idx < multiplesLen; idx++) {
            mult[idx] = multiplesData32[idx];
        }
    }

    while (static_cast<int32_t>(shape.size()) < multiplesLen) {
        shape.insert(shape.begin(), 1);
    }
    while (static_cast<int32_t>(mult.size()) < static_cast<int32_t>(shape.size())) {
        mult.insert(mult.begin(), 1);
    }

    int32_t blockDim = static_cast<int32_t>(coreNum);
    int32_t aicBlockDim = static_cast<int32_t>(aicCoreNum);
    int32_t bufferSize = static_cast<int32_t>(ubSize) / TILE_BUFFER_NUM;

    TileTilingData* tiling = context->GetTilingData<TileTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(TileTilingData), 0, sizeof(TileTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    ComputeTilingParams(*tiling, shape, mult, elemBytes, blockDim, bufferSize, aicBlockDim);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    int32_t ae = (elemBytes > 0) ? (DATA_BLOCK_BYTES / elemBytes) : 1;
    if (ae < 1) {
        ae = 1;
    }
    int32_t fID = tiling->inputShape[tiling->numDims - 1];
    int32_t fIM = tiling->multiples[tiling->numDims - 1];
    int32_t fIDA = ((fID + ae - 1) / ae) * ae;
    int32_t fOD = fID * fIM;
    bool fAligned = (fID == fIDA);
    int32_t fOC = (fOD > 0) ? (tiling->totalOutputElems / fOD) : 0;
    bool fSplit = (fOC < tiling->blockDim && fIM > 1 && fIDA <= bufferSize / elemBytes);
    int32_t fRowsPerCore = (tiling->blockDim > 0) ? ((fOC + tiling->blockDim - 1) / tiling->blockDim) : 0;
    int32_t rPre = tiling->repeatPeriod;
    int32_t riPre = tiling->repeatInputPeriod;
    bool hasRepeatBenefit =
        (rPre > riPre && riPre > 0 && (rPre / riPre) >= MIN_PERIOD_RATIO_FOR_BENEFIT && tiling->totalInputElems > 1);
    bool useScalarMode =
        !fSplit && !fAligned && fIM > 1 && fOD <= (bufferSize / elemBytes) &&
        (tiling->totalInputElems > MIN_INPUT_ELEMS_FOR_SCALAR || fRowsPerCore >= MIN_ROWS_PER_CORE_FOR_SCALAR ||
         hasRepeatBenefit) &&
        (fID <= SMALL_INNER_DIM_THRESHOLD || (elemBytes >= ELEM_BYTES_FP32 && fID <= LARGE_INNER_DIM_THRESHOLD));

    int32_t rP = tiling->repeatPeriod;
    int32_t riP = tiling->repeatInputPeriod;
    int32_t fBufElems = (elemBytes > 0) ? (bufferSize / elemBytes) : 0;
    bool canBuildOnce =
        rP > riP && riP > 1 && (rP / riP) > 1 && riP * fOD <= fBufElems && (riP - 1) * fID + fIDA <= fBufElems;
    if (!useScalarMode && canBuildOnce && !fSplit && !fAligned && fIM > 1 && fOD <= (bufferSize / elemBytes) &&
        (tiling->totalInputElems > MIN_INPUT_ELEMS_FOR_SCALAR || fRowsPerCore >= MIN_ROWS_PER_CORE_FOR_SCALAR ||
         hasRepeatBenefit) &&
        elemBytes <= ELEM_BYTES_FP16 && fID <= MODERATE_INNER_DIM_THRESHOLD && fID % 2 == 0) {
        useScalarMode = true;
    }
    if (!useScalarMode && canBuildOnce && !fSplit && !fAligned && fIM > 1 && fOD <= (bufferSize / elemBytes) &&
        hasRepeatBenefit && elemBytes == ELEM_BYTES_INT8 && fID <= LARGE_INNER_DIM_THRESHOLD) {
        useScalarMode = true;
    }
    bool useBuildOnce = useScalarMode && canBuildOnce;

    bool useVecGather = false;
    if (useBuildOnce && elemBytes >= MIN_ELEM_BYTES_FOR_VECGATHER && elemBytes <= MAX_ELEM_BYTES_FOR_VECGATHER &&
        fIM > 1 && !fAligned && (rP / riP) >= MIN_PERIOD_RATIO_FOR_VECGATHER && riP <= MAX_PERIOD_FOR_VECGATHER &&
        fID * fIM >= MIN_OUTPUT_INNER_DIM_FOR_VECGATHER) {
        int32_t vreg = (elemBytes == ELEM_BYTES_FP16) ? VREG_WIDTH_FP16 : VREG_WIDTH_FP32;
        int32_t gRow = (fOD > vreg) ? (((fOD + ae - 1) / ae) * ae) : vreg;
        int32_t offU = ((fOD + BYTE_ALIGN_SIZE - 1) / BYTE_ALIGN_SIZE) * BYTE_ALIGN_SIZE;
        int32_t offT = (offU * OFFSET_UINT32_SIZE + elemBytes - 1) / elemBytes;
        int32_t offA = ((offT + ae - 1) / ae) * ae;
        int32_t total = offA + riP * gRow;
        if (total <= fBufElems && riP * fIDA <= fBufElems) {
            useVecGather = true;
        }
    }

    bool useDmaBuild = useBuildOnce && elemBytes >= ELEM_BYTES_FP32 && (fID % 2) == 1 &&
                       fID >= MIN_INNER_DIM_FOR_DMABUILD && riP >= MIN_REP_INPUT_PERIOD_FOR_DMABUILD &&
                       riP * fIDA <= fBufElems;

    bool readbackOverride = false;
    if (useDmaBuild && rP > riP && riP > 0 && (rP / riP) >= MIN_PERIOD_RATIO_FOR_BENEFIT && riP * fOD <= fBufElems &&
        fOD <= fBufElems) {
        readbackOverride = true;
    }

    int32_t gRP = rPre, gRiP = riPre;
    if (gRP <= gRiP && tiling->numDims >= 2) {
        int32_t bestIP = INT32_MAX;
        for (int32_t d = tiling->numDims - 2; d >= 0; d--) {
            if (tiling->multiples[d] <= 1)
                continue;
            int32_t sd = 1;
            for (int32_t dd = d + 1; dd < tiling->numDims - 1; dd++)
                sd *= tiling->outputShape[dd];
            int32_t ip = tiling->inputShape[d] * sd;
            int32_t op = tiling->outputShape[d] * sd;
            if (ip <= 0 || op <= ip)
                continue;
            if (ip < bestIP) {
                bestIP = ip;
                gRP = op;
                gRiP = ip;
            }
        }
    }

    bool useReadbackAmplify = !useScalarMode && !fSplit && !fAligned && fIM > 1 && gRP > gRiP && gRiP > 0 &&
                              (gRP / gRiP) >= MIN_PERIOD_RATIO_FOR_BENEFIT && fOD <= (bufferSize / elemBytes) &&
                              fID > MIN_INNER_DIM_FOR_READBACK && static_cast<int64_t>(fIDA) * fIM <= fBufElems &&
                              (static_cast<int64_t>(fID) * elemBytes <= SMALL_TEMPLATE_BYTES ||
                               static_cast<int64_t>(fOC) * fOD * elemBytes >= LARGE_OUTPUT_BYTES_THRESHOLD);

    bool useInnerOneFill = fSplit && fID == 1 && fIM > ae;

    bool buildOnceToReadback = false;
    if (useBuildOnce && gRP > gRiP && gRiP > 0) {
        int32_t totalPeriods_bo = (gRP > 0) ? (fOC / gRP) : 0;
        int32_t totalSeeds_bo = totalPeriods_bo * gRiP;
        int32_t seedsPerCore_bo =
            (tiling->blockDim > 0) ? (totalSeeds_bo + tiling->blockDim - 1) / tiling->blockDim : 0;
        if (totalPeriods_bo > 0 && totalPeriods_bo < tiling->blockDim / 2 && seedsPerCore_bo >= MIN_NSRC_THRESHOLD &&
            fOD <= fBufElems && static_cast<int64_t>(fIDA) * fIM <= fBufElems) {
            useBuildOnce = false;
            buildOnceToReadback = true;
        }
    }

    if (useInnerOneFill) {
        tiling->blockDim = 1;
    }

    uint64_t tilingKey;
    if (readbackOverride) {
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_READBACK);
    } else if (useDmaBuild) {
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_DMABUILD);
    } else if (useVecGather) {
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_VECGATHER);
    } else if (useBuildOnce) {
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_BUILDONCE);
    } else if (buildOnceToReadback) {
        if (gRP != rPre || gRiP != riPre) {
            tiling->repeatPeriod = gRP;
            tiling->repeatInputPeriod = gRiP;
        }
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_READBACK);
        if (tiling->blockDim * MIN_AMPLIFICATION_RATIO < static_cast<int32_t>(coreNum)) {
            tiling->blockDim = static_cast<int32_t>(coreNum);
        }
    } else if (useInnerOneFill) {
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_DMABUILD);
    } else if (useReadbackAmplify) {
        if (gRP != rPre || gRiP != riPre) {
            tiling->repeatPeriod = gRP;
            tiling->repeatInputPeriod = gRiP;
        }
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_READBACK);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_DEFAULT);
        if (!fSplit && !fAligned && fIM > 1 && fOD <= fBufElems && tiling->numDims >= 2 &&
            (fID <= MIN_INNER_DIM_FOR_READBACK || (elemBytes >= ELEM_BYTES_FP32 && fID <= LARGE_INNER_DIM_THRESHOLD))) {
            int32_t bestIP2 = INT32_MAX;
            int32_t foundRP = 0, foundRiP = 0;
            for (int32_t d = tiling->numDims - 2; d >= 0; d--) {
                if (tiling->multiples[d] <= 1)
                    continue;
                int32_t sd = 1;
                for (int32_t dd = d + 1; dd < tiling->numDims - 1; dd++)
                    sd *= tiling->outputShape[dd];
                int32_t ip = tiling->inputShape[d] * sd;
                int32_t op = tiling->outputShape[d] * sd;
                if (ip <= 0 || op <= ip)
                    continue;
                if ((op / ip) < MIN_PERIOD_RATIO_FOR_BENEFIT)
                    continue;
                bool bo = ip > 1 && ip * fOD <= fBufElems && (ip - 1) * fID + fIDA <= fBufElems;
                if (!bo)
                    continue;
                int32_t tp = fOC / op;
                if (tp < 1)
                    continue;
                if (ip < bestIP2) {
                    bestIP2 = ip;
                    foundRP = op;
                    foundRiP = ip;
                }
            }
            if (foundRP > 0 && foundRiP > 0) {
                tiling->repeatPeriod = foundRP;
                tiling->repeatInputPeriod = foundRiP;
                tilingKey = GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_BUILDONCE);
            }
        }
    }
    context->SetTilingKey(tilingKey);

    bool isPeriodBased = tilingKey == GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_BUILDONCE) ||
                         tilingKey == GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_DMABUILD) ||
                         tilingKey == GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_VECGATHER);
    if (isPeriodBased && tiling->repeatPeriod > 0) {
        int32_t totalPeriods = fOC / tiling->repeatPeriod;
        int32_t remainder = fOC - totalPeriods * tiling->repeatPeriod;
        int32_t pps = tiling->periodsPerSource;
        int32_t neededCores = 0;
        if (pps > 0 && totalPeriods > 0 && totalPeriods % pps == 0) {
            int32_t nSrc = totalPeriods / pps;
            if (nSrc > 0 && nSrc < tiling->blockDim) {
                neededCores = nSrc + (remainder > 0 ? 1 : 0);
            }
        } else if (pps == 0) {
            neededCores = totalPeriods + (remainder > 0 ? 1 : 0);
        }
        if (neededCores > 0 && neededCores < tiling->blockDim) {
            int32_t periodsPerCore = (totalPeriods + neededCores - 1) / neededCores;
            if (periodsPerCore < 2 && tilingKey == GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_BUILDONCE) &&
                totalPeriods > MIN_PERIODS_FOR_MERGE) {
                neededCores = (totalPeriods + 1) / 2;
            }
            if (neededCores > 0 && neededCores < tiling->blockDim) {
                tiling->blockDim = neededCores;
            }
        }
    }

    if (isPeriodBased && tilingKey == GET_TPL_TILING_KEY(TILE_TPL_SCH_MODE_BUILDONCE) && tiling->repeatPeriod > 0 &&
        tiling->repeatInputPeriod > 0) {
        int32_t boRiP = tiling->repeatInputPeriod;
        int32_t boOID = tiling->inputShape[tiling->numDims - 1] * tiling->multiples[tiling->numDims - 1];
        int64_t templateBytes = static_cast<int64_t>(boRiP) * boOID * elemBytes;
        if (templateBytes < SMALL_TEMPLATE_BYTES && tiling->blockDim > aicBlockDim && aicBlockDim > 0) {
            tiling->blockDim = aicBlockDim;
        }
    }

    context->SetBlockDim(static_cast<uint32_t>(tiling->blockDim));

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Tile)
    .Tiling(TileTilingFunc)
    .TilingParse<TileCompileInfo>(TilingParseForTile)
    .InputsDataDependency({1});
} // namespace optiling
