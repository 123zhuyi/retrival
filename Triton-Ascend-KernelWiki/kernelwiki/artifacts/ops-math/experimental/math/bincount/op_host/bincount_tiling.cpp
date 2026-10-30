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
 * \file bincount_tiling.cpp
 * \brief bincount tiling
 */

#include "log/log.h"
#include "util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/bincount_tiling_data.h"
#include "../op_kernel/bincount_tiling_key.h"

namespace optiling {
// 与 kernel 侧常量保持一致（op_kernel/bincount.h）
constexpr uint32_t BINCOUNT_BUFFER_NUM = 2;
constexpr uint32_t BINCOUNT_TILE_DATA_NUM = 4096;
constexpr uint32_t BINCOUNT_MIN_SLOT = 4;
constexpr uint32_t BINCOUNT_UB_BLOCK = 32;
// 小输入阈值：多核同步 + workspace 往返 + 归并的固定开销远超实际计算量，
// 输入元素数 <= 该阈值时改用单核直算（kernel 侧 coreNum==1 跳过 SyncAll/归并）。
constexpr uint64_t BINCOUNT_SINGLE_CORE_MAX = 4096;
// 大输出阈值：多核归并代价是 coreNum*outLength（0 号核标量逐桶累加），outLength 大时
// 归并远超并行散射的收益，反而比单核慢几百倍。outLength > 该阈值时也走单核。
constexpr uint64_t BINCOUNT_MULTICORE_MAX_OUT = 2048;

static uint32_t BincountDtypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1;
        case ge::DT_INT16:
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_INT32:
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_INT64:
        case ge::DT_DOUBLE:
            return 8;
        default:
            return 4;
    }
}

// 判定是否走 GM 大 L 回退路径：私有直方图(2 份)放不下 UB 则 largeL=true；
// out=double 且超 UB 无法位拼接原子读改写,不支持,返回 GRAPH_FAILED。
static ge::graphStatus DecideLargeL(gert::TilingContext* context, platform_ascendc::PlatformAscendC& platform,
                                    uint64_t outLength, bool hasWeights, int64_t coreNum, bool& largeL)
{
    largeL = false;
    uint32_t selfSize = 4U;
    auto arrayDesc = context->GetInputDesc(0);
    if (arrayDesc != nullptr) {
        selfSize = BincountDtypeSize(arrayDesc->GetDataType());
    }
    // 累加类型字节宽：out=double 时 kernel 内部以 float 累加（位拼接写回），acc=4；其余按 bins dtype。
    bool binsIsDouble = false;
    uint32_t accSize = 8U;
    auto binsDesc = context->GetOutputDesc(0);
    if (binsDesc != nullptr) {
        ge::DataType bdt = binsDesc->GetDataType();
        binsIsDouble = (bdt == ge::DT_DOUBLE);
        accSize = binsIsDouble ? 4U : BincountDtypeSize(bdt);
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t alignedHistBytes = ((outLength * accSize + BINCOUNT_UB_BLOCK - 1U) / BINCOUNT_UB_BLOCK) *
                                BINCOUNT_UB_BLOCK;
    uint64_t ubNeed = static_cast<uint64_t>(BINCOUNT_BUFFER_NUM) * BINCOUNT_TILE_DATA_NUM * selfSize +
                      (hasWeights ? static_cast<uint64_t>(BINCOUNT_BUFFER_NUM) * BINCOUNT_TILE_DATA_NUM * accSize :
                                    0U) +
                      2U * alignedHistBytes + static_cast<uint64_t>(BINCOUNT_MIN_SLOT) * sizeof(int64_t) +
                      static_cast<uint64_t>(coreNum) * BINCOUNT_MIN_SLOT * sizeof(int64_t);
    if (ubSize > 0 && ubNeed > ubSize) {
        OP_CHECK_IF(binsIsDouble,
                    OP_LOGE(context,
                            "Bincount outLength=%lu with out=double exceeds UB (need ~%lu B > %lu B). "
                            "Huge output range with DOUBLE output is not supported (bit-spliced double "
                            "cannot be atomically read-modified on GM).",
                            outLength, ubNeed, ubSize),
                    return ge::GRAPH_FAILED);
        largeL = true; // 非 double：走 GM 直接散射路径
    }
    return ge::GRAPH_SUCCESS;
}

// 填 TilingData + 设 BlockDim/TilingKey + 计算并写 workspace 大小。
static ge::graphStatus WriteTilingData(gert::TilingContext* context, uint64_t totalNum, uint64_t outLength,
                                       int64_t coreNum, bool hasWeights, bool largeL, uint32_t blockSize,
                                       uint32_t sysWorkspaceSize)
{
    BincountTilingData* tiling = context->GetTilingData<BincountTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(BincountTilingData), 0, sizeof(BincountTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalNum = totalNum;
    tiling->outLength = outLength;
    tiling->coreNum = static_cast<uint64_t>(coreNum);
    tiling->bigCoreNum = (coreNum > 0) ? (totalNum % static_cast<uint64_t>(coreNum)) : 0;
    tiling->tileDataNum = 4096;
    tiling->hasWeights = hasWeights ? 1U : 0U;
    tiling->largeL = largeL ? 1U : 0U;
    context->SetBlockDim(coreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(BINCOUNT_TPL_SCH_MODE_0));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    size_t minWsBytes = static_cast<size_t>(coreNum) * blockSize;
    // GM 散射路径不使用跨核直方图 workspace（否则 coreNum*L*8 会爆显存）
    size_t histWsBytes = largeL ? 0U : static_cast<size_t>(coreNum) * (static_cast<size_t>(outLength) + blockSize) * 8U;
    currentWorkspace[0] = minWsBytes + histWsBytes + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BincountTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    int64_t coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid (<= 0)"), return ge::GRAPH_FAILED);
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is invalid (0)"), return ge::GRAPH_FAILED);

    auto arrayShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, arrayShape);
    uint64_t totalNum = arrayShape->GetStorageShape().GetShapeSize();

    auto binsShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, binsShape);
    uint64_t outLength = binsShape->GetStorageShape().GetShapeSize();

    auto weightsShape = context->GetInputShape(2);
    bool hasWeights = (weightsShape != nullptr);

    if (totalNum > 0 && static_cast<uint64_t>(coreNum) > totalNum) {
        coreNum = static_cast<int64_t>(totalNum);
    }

    // 直方图常驻 UB（2 份）；放得下走快路径，放不下走 GM 回退（largeL）。详见 DecideLargeL。
    bool largeL = false;
    ge::graphStatus largeLStatus = DecideLargeL(context, ascendcPlatform, outLength, hasWeights, coreNum, largeL);
    if (largeLStatus != ge::GRAPH_SUCCESS) {
        return largeLStatus;
    }

    // 单核直算的两种情形（large-L 仍走单核 GM 路径,不在此处理）：
    //   1) 小输入：多核同步/归并固定开销 > 实际计算；
    //   2) 大输出：多核归并代价 coreNum*outLength 过高,单核反而快。
    if (!largeL && totalNum > 0 && (totalNum <= BINCOUNT_SINGLE_CORE_MAX || outLength > BINCOUNT_MULTICORE_MAX_OUT)) {
        coreNum = 1;
    }

    return WriteTilingData(context, totalNum, outLength, coreNum, hasWeights, largeL, blockSize, sysWorkspaceSize);
}

IMPL_OP_OPTILING(Bincount).Tiling(BincountTilingFunc);
} // namespace optiling
