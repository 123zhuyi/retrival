/* Copyright (c) 2025 Tianjin University, Ltd.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* This file was copied from project Tianjin University flash-linear-attention-npu */

/*!
 * \file reduce_std_with_mean_tiling.cpp
 * \brief Host-side tiling for ReduceStdWithMean simple version
 *
 * Tiling strategy:
 *   - Multi-core split along non-reduce dimension
 *   - UB tile along reduce dimension (single-dimension tiling)
 *   - TilingKey: schMode-based (REDUCE_STD_SCH_FP16=0, FP32=1, BF16=2)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/reduce_std_with_mean_tiling_data.h"
#include "../op_kernel/reduce_std_with_mean_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

static constexpr int64_t UB_RESERVED = 8 * 1024;
static constexpr int64_t BUFFER_NUM = 2;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum==0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize==0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, int64_t& reduceLen,
                                         ge::DataType& dataType, int64_t& correction, float& eps, bool& invert)
{
    auto inputSelf = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputSelf);
    auto selfShape = EnsureNotScalar(inputSelf->GetStorageShape());
    totalNum = selfShape.GetShapeSize();
    int64_t rank = selfShape.GetDimNum();

    // Read dim attr to compute the correct reduce length
    const gert::RuntimeAttrs* dimAttrs = context->GetAttrs();
    reduceLen = 1;
    if (dimAttrs) {
        auto dimList = dimAttrs->GetListInt(0);
        if (dimList != nullptr) {
            const int64_t* dimData = dimList->GetData();
            int64_t dimSize = dimList->GetSize();
            for (int64_t d = 0; d < dimSize; d++) {
                int64_t axis = dimData[d];
                if (axis < 0)
                    axis += rank;
                if (axis >= 0 && axis < rank) {
                    reduceLen *= selfShape.GetDim(axis);
                }
            }
            if (dimSize == 0) {
                reduceLen = selfShape.GetDim(rank - 1);
            }
        } else {
            reduceLen = selfShape.GetDim(rank - 1);
        }
    } else {
        reduceLen = selfShape.GetDim(rank - 1);
    }
    OP_CHECK_IF(reduceLen <= 0, OP_LOGE(context, "reduceLen <= 0"), return ge::GRAPH_FAILED);

    const std::set<ge::DataType> supported = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supported.count(dataType) == 0) {
        OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    correction = 0;
    eps = 0.0f;
    invert = false;
    if (attrs) {
        if (attrs->GetInt(5)) {
            correction = *(attrs->GetInt(5));
        }
        if (attrs->GetBool(3)) {
            invert = *(attrs->GetBool(3));
        }
        if (attrs->GetFloat(4)) {
            eps = *(attrs->GetFloat(4));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    constexpr size_t WORKSPACE_NUM = 1;
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReduceStdWithMeanTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo"),
                return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    int64_t reduceLen = 0;
    ge::DataType dataType = ge::DT_FLOAT16;
    int64_t correction = 0;
    float eps = 0.0f;
    bool invert = false;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalNum, reduceLen, dataType, correction, eps, invert) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo"), return ge::GRAPH_FAILED);

    if (totalNum == 0 || reduceLen == 0) {
        context->SetBlockDim(1);
        uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_SCH_FP16);
        if (dataType == ge::DT_FLOAT)
            tk = GET_TPL_TILING_KEY(REDUCE_STD_SCH_FP32);
        else if (dataType == ge::DT_BF16)
            tk = GET_TPL_TILING_KEY(REDUCE_STD_SCH_BF16);
        context->SetTilingKey(tk);
        return ge::GRAPH_SUCCESS;
    }

    int64_t nonReduceNum = totalNum / reduceLen;
    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "SetWorkspace"), return ge::GRAPH_FAILED);

    ReduceStdWithMeanTilingData* tiling = context->GetTilingData<ReduceStdWithMeanTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ReduceStdWithMeanTilingData), 0, sizeof(ReduceStdWithMeanTilingData)) != EOK,
                OP_LOGE(context, "memset tiling"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = GetUbBlockSize(context);

    // Multi-core split along non-reduce dimension
    int64_t blockFactor = CeilAlign(CeilDiv(nonReduceNum, coreNum), ubBlockSize);
    if (blockFactor < ubBlockSize) {
        blockFactor = ubBlockSize;
    }
    int64_t usedCoreNum = CeilDiv(nonReduceNum, blockFactor);

    // UB tile length:
    //   self(buffered) + mean(buffered) + scratch + work + opt castBuf
    // Both VECIN queues are allocated in Init regardless of invert (for simplicity);
    // the mean queue is only used in the Two-Pass (invert=true) path.
    int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    int64_t perElemUB = 2 * BUFFER_NUM * typeSize;        // self + mean VECIN queues
    perElemUB += 2 * static_cast<int64_t>(sizeof(float)); // scratch + work (both fp32)
    if (dataType != ge::DT_FLOAT) {
        perElemUB += static_cast<int64_t>(sizeof(float)); // fp16/bf16 cast buffer
    }
    int64_t ubAvail = static_cast<int64_t>(ubSize) - UB_RESERVED;
    int64_t ubForData = ubAvail - static_cast<int64_t>(sizeof(float)); // output buffer
    if (ubForData <= 0) {
        ubForData = ubAvail;
    }

    int64_t ubLength = FloorAlign(FloorDiv(ubForData, perElemUB), ubBlockSize);
    if (ubLength < 1) {
        ubLength = 1;
    }
    if (ubLength > reduceLen) {
        ubLength = reduceLen;
    }

    tiling->totalNonReduce = nonReduceNum;
    tiling->reduceLength = reduceLen;
    tiling->blockFactor = blockFactor;
    tiling->ubLength = ubLength;
    tiling->correction = correction;
    tiling->eps = eps;
    tiling->invert = invert;

    context->SetBlockDim(usedCoreNum);

    uint64_t tk = GET_TPL_TILING_KEY(REDUCE_STD_SCH_FP16);
    if (dataType == ge::DT_FLOAT)
        tk = GET_TPL_TILING_KEY(REDUCE_STD_SCH_FP32);
    else if (dataType == ge::DT_BF16)
        tk = GET_TPL_TILING_KEY(REDUCE_STD_SCH_BF16);
    context->SetTilingKey(tk);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForReduceStdWithMean([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReduceStdWithMeanCompileInfo {};
IMPL_OP_OPTILING(ReduceStdWithMean)
    .Tiling(ReduceStdWithMeanTilingFunc)
    .TilingParse<ReduceStdWithMeanCompileInfo>(TilingParseForReduceStdWithMean);

} // namespace optiling
