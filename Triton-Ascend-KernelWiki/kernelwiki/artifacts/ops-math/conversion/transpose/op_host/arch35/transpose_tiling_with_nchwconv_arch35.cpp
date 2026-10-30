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
 * \file transpose_tiling_with_nchwconv_arch35.cpp
 * \brief transpose_tiling_nchwconv_arch35
 */
#include "transpose_tiling_with_nchwconv_arch35.h"

namespace optiling {
namespace TransposeWithVCONV {
static constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;
static constexpr int64_t TRANSELEM = 16;
static constexpr int8_t BUFFER_TENSOR_NUM = 2; // 输入输出各1块

void TransposeVCONVTiling::CalcBasicInfo()
{
    AvailableUbSize = platInfo_.ubSize / BUFFER_TENSOR_NUM / BUFFER_NUM;
    basicInfo_.RLen = shapeInfo_.reducedInShape[0];
    basicInfo_.CLen = shapeInfo_.reducedInShape[1];
    basicInfo_.RAlignBlock = Ops::Base::CeilDiv(basicInfo_.RLen, TRANSELEM);
    basicInfo_.CAlignBlock = Ops::Base::CeilDiv(basicInfo_.CLen, TRANSELEM);
    basicInfo_.RAlignBlockElem = basicInfo_.RAlignBlock * TRANSELEM;
    basicInfo_.CAlignBlockElem = basicInfo_.CAlignBlock * TRANSELEM;
}

void TransposeVCONVTiling::CalcRSplitInfo()
{
    rCoreSplitInfo_.AlignBlockFactor = Ops::Base::CeilDiv(basicInfo_.RAlignBlock, platInfo_.coreNum);
    rCoreSplitInfo_.BlockFactor = rCoreSplitInfo_.AlignBlockFactor * TRANSELEM;
    rCoreSplitInfo_.BlockCount = Ops::Base::CeilDiv(basicInfo_.RAlignBlock, rCoreSplitInfo_.AlignBlockFactor);
    rCoreSplitInfo_.AlignBlockTailFactor = basicInfo_.RAlignBlock -
                                           (rCoreSplitInfo_.BlockCount - 1) * rCoreSplitInfo_.AlignBlockFactor;
    rCoreSplitInfo_.BlockTailFactor = rCoreSplitInfo_.AlignBlockTailFactor * TRANSELEM;
    basicInfo_.UsedCoreNum = rCoreSplitInfo_.BlockCount;
    basicInfo_.IsRSplit = true;
}

void TransposeVCONVTiling::CalcCSplitInfo()
{
    cCoreSplitInfo_.AlignBlockFactor = Ops::Base::CeilDiv(basicInfo_.CAlignBlock, platInfo_.coreNum);
    cCoreSplitInfo_.BlockFactor = cCoreSplitInfo_.AlignBlockFactor * TRANSELEM;
    cCoreSplitInfo_.BlockCount = Ops::Base::CeilDiv(basicInfo_.CAlignBlock, cCoreSplitInfo_.AlignBlockFactor);
    cCoreSplitInfo_.AlignBlockTailFactor = basicInfo_.CAlignBlock -
                                           (cCoreSplitInfo_.BlockCount - 1) * cCoreSplitInfo_.AlignBlockFactor;
    cCoreSplitInfo_.BlockTailFactor = cCoreSplitInfo_.AlignBlockTailFactor * TRANSELEM;
    basicInfo_.UsedCoreNum = cCoreSplitInfo_.BlockCount;
}

void TransposeVCONVTiling::FillBlockPara(CoreSplitPara& coreSplitPara, CoreSplitInfo& corePara)
{
    coreSplitPara.set_AlignBlockFactor(corePara.AlignBlockFactor);
    coreSplitPara.set_BlockFactor(corePara.BlockFactor);
    coreSplitPara.set_BlockCount(corePara.BlockCount);
    coreSplitPara.set_AlignBlockTailFactor(corePara.AlignBlockTailFactor);
    coreSplitPara.set_BlockTailFactor(corePara.BlockTailFactor);
}

void TransposeVCONVTiling::FillUbPara(UbSplitPara& ubSplitPara, UbParamInfo& ubPara)
{
    ubSplitPara.set_MainCoreUbAlignFactor(ubPara.MainCoreUbAlignFactor);
    ubSplitPara.set_MainCoreUbFactor(ubPara.MainCoreUbFactor);
    ubSplitPara.set_MainCoreUbCount(ubPara.MainCoreUbCount);
    ubSplitPara.set_MainCoreTailUbAlignFactor(ubPara.MainCoreTailUbAlignFactor);
    ubSplitPara.set_MainCoreTailUbFactor(ubPara.MainCoreTailUbFactor);
    ubSplitPara.set_TailCoreUbAlignFactor(ubPara.TailCoreUbAlignFactor);
    ubSplitPara.set_TailCoreUbFactor(ubPara.TailCoreUbFactor);
    ubSplitPara.set_TailCoreUbCount(ubPara.TailCoreUbCount);
    ubSplitPara.set_TailCoreTailUbAlignFactor(ubPara.TailCoreTailUbAlignFactor);
    ubSplitPara.set_TailCoreTailUbFactor(ubPara.TailCoreTailUbFactor);
}

void TransposeVCONVTiling::WriteTilingData()
{
    tiling_.set_AvailableUbSize(AvailableUbSize);
    tiling_.set_UsedCoreNum(basicInfo_.UsedCoreNum);
    tiling_.set_MainCoreLoopCount(basicInfo_.MainCoreLoopCount);
    tiling_.set_TailCoreLoopCount(basicInfo_.TailCoreLoopCount);
    tiling_.set_RLen(basicInfo_.RLen);
    tiling_.set_CLen(basicInfo_.CLen);
    tiling_.set_RAlignBlock(basicInfo_.RAlignBlock);
    tiling_.set_CAlignBlock(basicInfo_.CAlignBlock);
    tiling_.set_RAlignBlockElem(basicInfo_.RAlignBlockElem);
    tiling_.set_CAlignBlockElem(basicInfo_.CAlignBlockElem);
    tiling_.set_IsRSplit(basicInfo_.IsRSplit);
    tiling_.set_IsRCSplit(basicInfo_.IsRCSplit);
    FillBlockPara(tiling_.rSplitPara, rCoreSplitInfo_);
    FillBlockPara(tiling_.cSplitPara, cCoreSplitInfo_);
    FillUbPara(tiling_.rUbSplitPara, rUbParamInfo_);
    FillUbPara(tiling_.cUbSplitPara, cUbParamInfo_);

    tiling_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling_.GetDataSize());
}

void TransposeVCONVTiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "Entering PrintTilingData.");
    OP_LOGD(context_->GetNodeName(),
            "AvailableUbSize: %d, UsedCoreNum: %d, MainCoreLoopCount: %d, TailCoreLoopCount: %d, RLen: %d, \
        CLen: %d, RAlignBlock: %d, CAlignBlock: %d, RAlignBlockElem: %d, \
        CAlignBlockElem: %d, IsRSplit: %d, IsRCSplit: %d",
            AvailableUbSize, basicInfo_.UsedCoreNum, basicInfo_.MainCoreLoopCount, basicInfo_.TailCoreLoopCount,
            basicInfo_.RLen, basicInfo_.CLen, basicInfo_.RAlignBlock, basicInfo_.CAlignBlock,
            basicInfo_.RAlignBlockElem, basicInfo_.CAlignBlockElem, basicInfo_.IsRSplit, basicInfo_.IsRCSplit);
    OP_LOGD(context_->GetNodeName(),
            "core split info is: rAlignBlockFactor: %d, rBlockFactor: %d, rBlockCount: %d, rAlignBlockTailFactor: %d, \
        rBlockTailFactor: %d, cAlignBlockFactor: %d, cBlockFactor: %d, cBlockCount: %d, cAlignBlockTailFactor: %d, \
        cBlockTailFactor: %d",
            rCoreSplitInfo_.AlignBlockFactor, rCoreSplitInfo_.BlockFactor, rCoreSplitInfo_.BlockCount,
            rCoreSplitInfo_.AlignBlockTailFactor, rCoreSplitInfo_.BlockTailFactor, cCoreSplitInfo_.AlignBlockFactor,
            cCoreSplitInfo_.BlockFactor, cCoreSplitInfo_.BlockCount, cCoreSplitInfo_.AlignBlockTailFactor,
            cCoreSplitInfo_.BlockTailFactor);

    OP_LOGD(context_->GetNodeName(),
            "r ub params info is: rMainCoreUbAlignFactor: %d, rMainCoreUbFactor: %d, rMainCoreUbCount: %d, \
        rMainCoreTailUbAlignFactor: %d, rMainCoreTailUbFactor: %d, rTailCoreUbAlignFactor: %d, rTailCoreUbFactor: %d, \
        rTailCoreUbCount: %d, rTailCoreTailUbAlignFactor: %d, rTailCoreTailUbFactor: %d",
            rUbParamInfo_.MainCoreUbAlignFactor, rUbParamInfo_.MainCoreUbFactor, rUbParamInfo_.MainCoreUbCount,
            rUbParamInfo_.MainCoreTailUbAlignFactor, rUbParamInfo_.MainCoreTailUbFactor,
            rUbParamInfo_.TailCoreUbAlignFactor, rUbParamInfo_.TailCoreUbFactor, rUbParamInfo_.TailCoreUbCount,
            rUbParamInfo_.TailCoreTailUbAlignFactor, rUbParamInfo_.TailCoreTailUbFactor);
    OP_LOGD(context_->GetNodeName(),
            "c ub params info is :cMainCoreUbAlignFactor: %d, cMainCoreUbFactor: %d, cMainCoreUbCount: %d, \
        cMainCoreTailUbAlignFactor: %d, cMainCoreTailUbFactor: %d, cTailCoreUbAlignFactor: %d, cTailCoreUbFactor: %d, \
        cTailCoreUbCount: %d, cTailCoreTailUbAlignFactor: %d, cTailCoreTailUbFactor: %d",
            cUbParamInfo_.MainCoreUbAlignFactor, cUbParamInfo_.MainCoreUbFactor, cUbParamInfo_.MainCoreUbCount,
            cUbParamInfo_.MainCoreTailUbAlignFactor, cUbParamInfo_.MainCoreTailUbFactor,
            cUbParamInfo_.TailCoreUbAlignFactor, cUbParamInfo_.TailCoreUbFactor, cUbParamInfo_.TailCoreUbCount,
            cUbParamInfo_.TailCoreTailUbAlignFactor, cUbParamInfo_.TailCoreTailUbFactor);
}

ge::graphStatus TransposeVCONVTiling::SetTilingKeyAndCore()
{
    tilingKey_ = static_cast<uint64_t>(SplitMode::VCONV_TRANSPOSE);
    context_->SetTilingKey(tilingKey_);
    OP_CHECK_IF(context_->SetBlockDim(basicInfo_.UsedCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set used core num is failed!"), return ge::GRAPH_FAILED);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeVCONVTiling::CalcCFullLoadRUbSplitInfo()
{
    SetCFullLoadCUbSplitInfo();
    if (rCoreSplitInfo_.BlockFactor * basicInfo_.CAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "Entering CalcCFullLoadRUbSplitInfo if branch");
        SetRBlockFactorUbFullLoadSplitInfo();
        basicInfo_.MainCoreLoopCount = 1;
        basicInfo_.TailCoreLoopCount = 1;
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering CalcCFullLoadRUbSplitInfo else branch");
        rUbParamInfo_.MainCoreUbAlignFactor = AvailableUbSize / shapeInfo_.eleLenInBytes / basicInfo_.CAlignBlockElem /
                                              TRANSELEM;
        rUbParamInfo_.MainCoreUbFactor = rUbParamInfo_.MainCoreUbAlignFactor * TRANSELEM;
        rUbParamInfo_.MainCoreUbCount = Ops::Base::CeilDiv(rCoreSplitInfo_.BlockFactor, rUbParamInfo_.MainCoreUbFactor);
        rUbParamInfo_.MainCoreTailUbAlignFactor = rCoreSplitInfo_.AlignBlockFactor -
                                                  (rUbParamInfo_.MainCoreUbCount - 1) *
                                                      rUbParamInfo_.MainCoreUbAlignFactor;
        rUbParamInfo_.MainCoreTailUbFactor = rUbParamInfo_.MainCoreTailUbAlignFactor * TRANSELEM;

        rUbParamInfo_.TailCoreUbAlignFactor = (rUbParamInfo_.MainCoreUbAlignFactor >=
                                               rCoreSplitInfo_.AlignBlockTailFactor) ?
                                                  rCoreSplitInfo_.AlignBlockTailFactor :
                                                  rUbParamInfo_.MainCoreUbAlignFactor;
        rUbParamInfo_.TailCoreUbFactor = rUbParamInfo_.TailCoreUbAlignFactor * TRANSELEM;
        rUbParamInfo_.TailCoreUbCount = Ops::Base::CeilDiv(rCoreSplitInfo_.BlockTailFactor,
                                                           rUbParamInfo_.TailCoreUbFactor);
        rUbParamInfo_.TailCoreTailUbAlignFactor = rCoreSplitInfo_.AlignBlockTailFactor -
                                                  (rUbParamInfo_.TailCoreUbCount - 1) *
                                                      rUbParamInfo_.TailCoreUbAlignFactor;
        rUbParamInfo_.TailCoreTailUbFactor = rUbParamInfo_.TailCoreTailUbAlignFactor * TRANSELEM;

        basicInfo_.MainCoreLoopCount = rUbParamInfo_.MainCoreUbCount;
        basicInfo_.TailCoreLoopCount = rUbParamInfo_.TailCoreUbCount;
    }
    return ge::GRAPH_SUCCESS;
}

void TransposeVCONVTiling::SetCFullLoadCUbSplitInfo()
{
    cUbParamInfo_.MainCoreUbAlignFactor = basicInfo_.CAlignBlock;
    cUbParamInfo_.MainCoreUbFactor = basicInfo_.CAlignBlockElem;
    cUbParamInfo_.MainCoreUbCount = 1;
    cUbParamInfo_.MainCoreTailUbAlignFactor = cUbParamInfo_.MainCoreUbAlignFactor;
    cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreUbFactor;

    cUbParamInfo_.TailCoreUbAlignFactor = basicInfo_.CAlignBlock;
    cUbParamInfo_.TailCoreUbFactor = basicInfo_.CAlignBlockElem;
    cUbParamInfo_.TailCoreUbCount = 1;
    cUbParamInfo_.TailCoreTailUbAlignFactor = cUbParamInfo_.TailCoreUbAlignFactor;
    cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.TailCoreUbFactor;
    cCoreSplitInfo_.AlignBlockFactor = basicInfo_.CAlignBlock;
    cCoreSplitInfo_.BlockFactor = basicInfo_.CAlignBlockElem;
    cCoreSplitInfo_.BlockCount = 1;
    cCoreSplitInfo_.AlignBlockTailFactor = cCoreSplitInfo_.AlignBlockFactor;
    cCoreSplitInfo_.BlockTailFactor = cCoreSplitInfo_.BlockFactor;
}

ge::graphStatus TransposeVCONVTiling::CalcRFullLoadCUbSplitInfo()
{
    SetRFullLoadCUbSplitInfo();
    if (cCoreSplitInfo_.BlockFactor * basicInfo_.RAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "Entering CalcRFullLoadCUbSplitInfo if branch");
        cUbParamInfo_.MainCoreUbAlignFactor = cCoreSplitInfo_.AlignBlockFactor;
        cUbParamInfo_.MainCoreUbFactor = cCoreSplitInfo_.BlockFactor;
        cUbParamInfo_.MainCoreUbCount = 1;
        cUbParamInfo_.MainCoreTailUbAlignFactor = cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreUbFactor;

        cUbParamInfo_.TailCoreUbAlignFactor = cCoreSplitInfo_.AlignBlockTailFactor;
        cUbParamInfo_.TailCoreUbFactor = cCoreSplitInfo_.BlockTailFactor;
        cUbParamInfo_.TailCoreUbCount = 1;
        cUbParamInfo_.TailCoreTailUbAlignFactor = cUbParamInfo_.TailCoreUbAlignFactor;
        cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.TailCoreUbFactor;
        basicInfo_.MainCoreLoopCount = 1;
        basicInfo_.TailCoreLoopCount = 1;
    } else {
        OP_LOGD(context_->GetNodeName(), "Entering CalcRFullLoadCUbSplitInfo else branch");
        cUbParamInfo_.MainCoreUbAlignFactor = AvailableUbSize / shapeInfo_.eleLenInBytes / basicInfo_.RAlignBlockElem /
                                              TRANSELEM;
        cUbParamInfo_.MainCoreUbFactor = cUbParamInfo_.MainCoreUbAlignFactor * TRANSELEM;
        cUbParamInfo_.MainCoreUbCount = Ops::Base::CeilDiv(cCoreSplitInfo_.BlockFactor, cUbParamInfo_.MainCoreUbFactor);
        cUbParamInfo_.MainCoreTailUbAlignFactor = cCoreSplitInfo_.AlignBlockFactor -
                                                  (cUbParamInfo_.MainCoreUbCount - 1) *
                                                      cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreTailUbAlignFactor * TRANSELEM;

        cUbParamInfo_.TailCoreUbAlignFactor = (cUbParamInfo_.MainCoreUbAlignFactor >=
                                               cCoreSplitInfo_.AlignBlockTailFactor) ?
                                                  cCoreSplitInfo_.AlignBlockTailFactor :
                                                  cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.TailCoreUbFactor = cUbParamInfo_.TailCoreUbAlignFactor * TRANSELEM;
        cUbParamInfo_.TailCoreUbCount = Ops::Base::CeilDiv(cCoreSplitInfo_.BlockTailFactor,
                                                           cUbParamInfo_.TailCoreUbFactor);
        cUbParamInfo_.TailCoreTailUbAlignFactor = cCoreSplitInfo_.AlignBlockTailFactor -
                                                  (cUbParamInfo_.TailCoreUbCount - 1) *
                                                      cUbParamInfo_.TailCoreUbAlignFactor;
        cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.TailCoreTailUbAlignFactor * TRANSELEM;

        basicInfo_.MainCoreLoopCount = cUbParamInfo_.MainCoreUbCount;
        basicInfo_.TailCoreLoopCount = cUbParamInfo_.TailCoreUbCount;
    }
    return ge::GRAPH_SUCCESS;
}

void TransposeVCONVTiling::SetRFullLoadCUbSplitInfo()
{
    rUbParamInfo_.MainCoreUbAlignFactor = basicInfo_.RAlignBlock;
    rUbParamInfo_.MainCoreUbFactor = basicInfo_.RAlignBlockElem;
    rUbParamInfo_.MainCoreUbCount = 1;
    rUbParamInfo_.MainCoreTailUbAlignFactor = rUbParamInfo_.MainCoreUbAlignFactor;
    rUbParamInfo_.MainCoreTailUbFactor = rUbParamInfo_.MainCoreUbFactor;

    rUbParamInfo_.TailCoreUbAlignFactor = basicInfo_.RAlignBlock;
    rUbParamInfo_.TailCoreUbFactor = basicInfo_.RAlignBlockElem;
    rUbParamInfo_.TailCoreUbCount = 1;
    rUbParamInfo_.TailCoreTailUbAlignFactor = rUbParamInfo_.TailCoreUbAlignFactor;
    rUbParamInfo_.TailCoreTailUbFactor = rUbParamInfo_.TailCoreUbFactor;

    rCoreSplitInfo_.AlignBlockFactor = basicInfo_.RAlignBlock;
    rCoreSplitInfo_.BlockFactor = basicInfo_.RAlignBlockElem;
    rCoreSplitInfo_.BlockCount = 1;
    rCoreSplitInfo_.AlignBlockTailFactor = rCoreSplitInfo_.AlignBlockFactor;
    rCoreSplitInfo_.BlockTailFactor = rCoreSplitInfo_.BlockFactor;
}

ge::graphStatus TransposeVCONVTiling::CalcRCNotFullLoadUbSplitInfo()
{
    if (rCoreSplitInfo_.BlockFactor * TRANSELEM * shapeInfo_.eleLenInBytes > AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "BlockFactor * 16 > AvailableUbSize!");
        return ge::GRAPH_FAILED;
    } else {
        basicInfo_.IsRCSplit = true;
        OP_LOGD(context_->GetNodeName(), "Entering CalcRCNotFullLoadUbSplitInfo branch");
        SetRBlockFactorUbFullLoadSplitInfo();

        cUbParamInfo_.MainCoreUbAlignFactor = AvailableUbSize / shapeInfo_.eleLenInBytes / rCoreSplitInfo_.BlockFactor /
                                              TRANSELEM;
        cUbParamInfo_.MainCoreUbFactor = cUbParamInfo_.MainCoreUbAlignFactor * TRANSELEM;
        cUbParamInfo_.MainCoreUbCount = Ops::Base::CeilDiv(basicInfo_.CAlignBlockElem, cUbParamInfo_.MainCoreUbFactor);
        cUbParamInfo_.MainCoreTailUbAlignFactor = basicInfo_.CAlignBlock - (cUbParamInfo_.MainCoreUbCount - 1) *
                                                                               cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.MainCoreTailUbFactor = cUbParamInfo_.MainCoreTailUbAlignFactor * TRANSELEM;

        cUbParamInfo_.TailCoreUbAlignFactor = cUbParamInfo_.MainCoreUbAlignFactor;
        cUbParamInfo_.TailCoreUbFactor = cUbParamInfo_.MainCoreUbFactor;
        cUbParamInfo_.TailCoreUbCount = cUbParamInfo_.MainCoreUbCount;
        cUbParamInfo_.TailCoreTailUbAlignFactor = cUbParamInfo_.MainCoreTailUbAlignFactor;
        cUbParamInfo_.TailCoreTailUbFactor = cUbParamInfo_.MainCoreTailUbFactor;

        basicInfo_.MainCoreLoopCount = cUbParamInfo_.MainCoreUbCount;
        basicInfo_.TailCoreLoopCount = cUbParamInfo_.TailCoreUbCount;

        cCoreSplitInfo_.AlignBlockFactor = basicInfo_.CAlignBlock;
        cCoreSplitInfo_.BlockFactor = basicInfo_.CAlignBlockElem;
        cCoreSplitInfo_.BlockCount = 1;
        cCoreSplitInfo_.AlignBlockTailFactor = cCoreSplitInfo_.AlignBlockFactor;
        cCoreSplitInfo_.BlockTailFactor = cCoreSplitInfo_.BlockFactor;
        return ge::GRAPH_SUCCESS;
    }
}

void TransposeVCONVTiling::SetRBlockFactorUbFullLoadSplitInfo()
{
    rUbParamInfo_.MainCoreUbAlignFactor = rCoreSplitInfo_.AlignBlockFactor;
    rUbParamInfo_.MainCoreUbFactor = rCoreSplitInfo_.BlockFactor;
    rUbParamInfo_.MainCoreUbCount = 1;
    rUbParamInfo_.MainCoreTailUbAlignFactor = rUbParamInfo_.MainCoreUbAlignFactor;
    rUbParamInfo_.MainCoreTailUbFactor = rUbParamInfo_.MainCoreUbFactor;

    rUbParamInfo_.TailCoreUbAlignFactor = rCoreSplitInfo_.AlignBlockTailFactor;
    rUbParamInfo_.TailCoreUbFactor = rCoreSplitInfo_.BlockTailFactor;
    rUbParamInfo_.TailCoreUbCount = 1;
    rUbParamInfo_.TailCoreTailUbAlignFactor = rUbParamInfo_.TailCoreUbAlignFactor;
    rUbParamInfo_.TailCoreTailUbFactor = rUbParamInfo_.TailCoreUbFactor;
}

ge::graphStatus TransposeVCONVTiling::CalcBlockAndUbSplitInfo()
{
    if (TRANSELEM * basicInfo_.CAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
        CalcRSplitInfo();
        return CalcCFullLoadRUbSplitInfo();
    } else {
        if (TRANSELEM * basicInfo_.RAlignBlockElem * shapeInfo_.eleLenInBytes <= AvailableUbSize) {
            CalcCSplitInfo();
            return CalcRFullLoadCUbSplitInfo();
        } else {
            CalcRSplitInfo();
            return CalcRCNotFullLoadUbSplitInfo();
        }
    }
}

ge::graphStatus TransposeVCONVTiling::DoTiling()
{
    CalcBasicInfo();
    OP_CHECK_IF(CalcBlockAndUbSplitInfo() != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "Stop to run vconv tiling, block_factor need split!"),
                return ge::GRAPH_FAILED);

    WriteTilingData();
    PrintTilingData();
    return SetTilingKeyAndCore();
}

} // namespace TransposeWithVCONV
} // namespace optiling
