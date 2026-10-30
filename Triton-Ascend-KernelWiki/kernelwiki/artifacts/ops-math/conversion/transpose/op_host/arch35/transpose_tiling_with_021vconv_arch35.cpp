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
 * \file transpose_tiling_with_021vconv_arch35.cpp
 * \brief tiling implementation for 3D 021 transpose using TransDataTo5HD
 */

#include "transpose_tiling_with_021vconv_arch35.h"

namespace optiling {
namespace Transpose021WithVCONV {
static constexpr size_t SYS_WORKSPACE_SIZE = static_cast<size_t>(16) * 1024 * 1024;

void Transpose021VCONVTiling::CalcBasicInfo()
{
    basicInfo_.AvailableUbSize = platInfo_.ubSize / NUM_TWO / BUFFER_NUM;
    basicInfo_.NLen = shapeInfo_.reducedInShape[0];
    basicInfo_.HLen = shapeInfo_.reducedInShape[1];
    basicInfo_.WLen = shapeInfo_.reducedInShape[2];
    basicInfo_.BlockElem = (shapeInfo_.eleLenInBytes == B8_BYTES)  ? BLOCKELEM_8BIT :
                           (shapeInfo_.eleLenInBytes == B32_BYTES) ? BLOCKELEM_32BIT :
                                                                     BLOCKELEM_16BIT;
    basicInfo_.UseRConv = basicInfo_.HLen >= basicInfo_.WLen;
    basicInfo_.UseHSplit = basicInfo_.NLen < DIM_FIVE && !basicInfo_.UseRConv &&
                           basicInfo_.HLen >= basicInfo_.NLen * TRANSELEM;
    if (basicInfo_.UseRConv) {
        int64_t hAlignUnit = (shapeInfo_.eleLenInBytes == 1) ? basicInfo_.BlockElem : TRANSELEM;
        int64_t hAlignBlock = Ops::Base::CeilDiv(basicInfo_.HLen, hAlignUnit);
        int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, basicInfo_.BlockElem);
        basicInfo_.HAlignBlockElem = hAlignBlock * hAlignUnit;
        basicInfo_.WAlignBlockElem = wAlignBlock * basicInfo_.BlockElem;
    } else {
        int64_t hAlignUnit = (shapeInfo_.eleLenInBytes == 1) ? basicInfo_.BlockElem : TRANSELEM;
        int64_t wAlignUnit = (shapeInfo_.eleLenInBytes == 1) ? basicInfo_.BlockElem : TRANSELEM;
        int64_t hAlignBlock = Ops::Base::CeilDiv(basicInfo_.HLen, hAlignUnit);
        int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, wAlignUnit);
        basicInfo_.HAlignBlockElem = hAlignBlock * hAlignUnit;
        basicInfo_.WAlignBlockElem = wAlignBlock * wAlignUnit;
    }
}

void Transpose021VCONVTiling::CalcNSplitInfo()
{
    int64_t coreNum = platInfo_.coreNum;
    basicInfo_.NPerCore = Ops::Base::CeilDiv(basicInfo_.NLen, coreNum);
    basicInfo_.NTailCore = basicInfo_.NLen - (coreNum - 1) * basicInfo_.NPerCore;
    if (basicInfo_.NTailCore <= 0) {
        basicInfo_.UsedCoreNum = Ops::Base::CeilDiv(basicInfo_.NLen, basicInfo_.NPerCore);
        basicInfo_.NTailCore = basicInfo_.NLen - (basicInfo_.UsedCoreNum - 1) * basicInfo_.NPerCore;
    } else {
        basicInfo_.UsedCoreNum = coreNum;
    }
    if (basicInfo_.NTailCore <= 0) {
        basicInfo_.NTailCore = basicInfo_.NPerCore;
    }
}

void Transpose021VCONVTiling::CalcHSplitInfo()
{
    int64_t coreNum = platInfo_.coreNum;
    int64_t hAlignTotal = basicInfo_.HAlignBlockElem;
    int64_t minHPerCore = (shapeInfo_.eleLenInBytes == 4) ? TRANSELEM : (NUM_TWO * TRANSELEM);
    int64_t maxUsedCores = Ops::Base::CeilDiv(hAlignTotal, minHPerCore);
    int64_t effectiveCoreNum = std::min(coreNum, maxUsedCores);
    int64_t hPerCore = Ops::Base::CeilDiv(hAlignTotal, effectiveCoreNum);
    hPerCore = Ops::Base::CeilDiv(hPerCore, TRANSELEM) * TRANSELEM;
    if (shapeInfo_.eleLenInBytes == 1 && hPerCore % (NUM_TWO * TRANSELEM) != 0) {
        hPerCore = Ops::Base::CeilDiv(hPerCore, NUM_TWO * TRANSELEM) * NUM_TWO * TRANSELEM;
    }
    basicInfo_.HPerCore = hPerCore;
    int64_t usedCoreNum = Ops::Base::CeilDiv(hAlignTotal, hPerCore);
    basicInfo_.HTailCore = hAlignTotal - (usedCoreNum - 1) * hPerCore;
    if (basicInfo_.HTailCore <= 0) {
        basicInfo_.HTailCore = hPerCore;
    }
    basicInfo_.UsedCoreNum = usedCoreNum;
    basicInfo_.NPerCore = basicInfo_.NLen;
    basicInfo_.NTailCore = basicInfo_.NLen;
}

ge::graphStatus Transpose021VCONVTiling::CalcUbSplitInfo()
{
    if (basicInfo_.UseHSplit) {
        return CalcUbSplitHSplit();
    }
    if (basicInfo_.UseRConv) {
        return CalcUbSplitRConv();
    }
    return CalcUbSplitCConv();
}

ge::graphStatus Transpose021VCONVTiling::CalcUbSplitHSplit()
{
    if (shapeInfo_.eleLenInBytes == 1) {
        OP_LOGD(context_->GetNodeName(), "8-bit H-split not supported for 021 VCONV branch");
        return ge::GRAPH_FAILED;
    }
    int64_t blockElem = basicInfo_.BlockElem;
    int64_t hPerCore = basicInfo_.HPerCore;
    int64_t hTailCore = basicInfo_.HTailCore;
    int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, blockElem);
    int64_t wAlignBlockElem = wAlignBlock * blockElem;
    if (hPerCore * wAlignBlockElem * shapeInfo_.eleLenInBytes <= basicInfo_.AvailableUbSize) {
        rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(hPerCore, TRANSELEM);
        rUbParamInfo_.UbFactor = hPerCore;
        rUbParamInfo_.UbCount = 1;
        rUbParamInfo_.UbTailAlignFactor = Ops::Base::CeilDiv(hTailCore, TRANSELEM);
        rUbParamInfo_.UbTailFactor = hTailCore;
        cUbParamInfo_.UbAlignFactor = wAlignBlock;
        cUbParamInfo_.UbFactor = wAlignBlockElem;
        cUbParamInfo_.UbCount = 1;
        cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
        cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
        basicInfo_.UbLoopCount = 1;
        return ge::GRAPH_SUCCESS;
    }
    int64_t wPerUbAlignBlock = basicInfo_.AvailableUbSize / shapeInfo_.eleLenInBytes / hPerCore / blockElem;
    if (wPerUbAlignBlock <= 0) {
        OP_LOGD(context_->GetNodeName(), "H too large for 021 VCONV H-split branch");
        return ge::GRAPH_FAILED;
    }
    cUbParamInfo_.UbAlignFactor = wPerUbAlignBlock;
    cUbParamInfo_.UbFactor = wPerUbAlignBlock * blockElem;
    cUbParamInfo_.UbCount = Ops::Base::CeilDiv(wAlignBlock, cUbParamInfo_.UbAlignFactor);
    cUbParamInfo_.UbTailAlignFactor = wAlignBlock - (cUbParamInfo_.UbCount - 1) * cUbParamInfo_.UbAlignFactor;
    cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbTailAlignFactor * blockElem;
    rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(hPerCore, TRANSELEM);
    rUbParamInfo_.UbFactor = hPerCore;
    rUbParamInfo_.UbCount = 1;
    rUbParamInfo_.UbTailAlignFactor = Ops::Base::CeilDiv(hTailCore, TRANSELEM);
    rUbParamInfo_.UbTailFactor = hTailCore;
    basicInfo_.UbLoopCount = cUbParamInfo_.UbCount;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Transpose021VCONVTiling::CalcUbSplitRConv()
{
    int64_t blockElem = basicInfo_.BlockElem;
    int64_t eleSize = shapeInfo_.eleLenInBytes;
    if (TRANSELEM * basicInfo_.WAlignBlockElem * eleSize > basicInfo_.AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "W dimension too large for 021 VCONV branch");
        return ge::GRAPH_FAILED;
    }
    if (basicInfo_.HAlignBlockElem * basicInfo_.WAlignBlockElem * eleSize <= basicInfo_.AvailableUbSize) {
        rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSELEM);
        rUbParamInfo_.UbFactor = basicInfo_.HAlignBlockElem;
        rUbParamInfo_.UbCount = 1;
        rUbParamInfo_.UbTailAlignFactor = rUbParamInfo_.UbAlignFactor;
        rUbParamInfo_.UbTailFactor = rUbParamInfo_.UbFactor;
        cUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.WLen, blockElem);
        cUbParamInfo_.UbFactor = basicInfo_.WAlignBlockElem;
        cUbParamInfo_.UbCount = 1;
        cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
        cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
        basicInfo_.UbLoopCount = 1;
        return ge::GRAPH_SUCCESS;
    }
    rUbParamInfo_.UbAlignFactor = basicInfo_.AvailableUbSize / eleSize / basicInfo_.WAlignBlockElem / TRANSELEM;
    if (eleSize == 1 && rUbParamInfo_.UbAlignFactor % NUM_TWO != 0) {
        rUbParamInfo_.UbAlignFactor -= 1;
    }
    rUbParamInfo_.UbFactor = rUbParamInfo_.UbAlignFactor * TRANSELEM;
    int64_t hAlignBlock = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSELEM);
    rUbParamInfo_.UbCount = Ops::Base::CeilDiv(hAlignBlock, rUbParamInfo_.UbAlignFactor);
    int64_t tailAlignBlock = hAlignBlock - (rUbParamInfo_.UbCount - 1) * rUbParamInfo_.UbAlignFactor;
    if (eleSize == 1 && tailAlignBlock % NUM_TWO != 0) {
        tailAlignBlock += 1;
    }
    rUbParamInfo_.UbTailAlignFactor = tailAlignBlock;
    rUbParamInfo_.UbTailFactor = tailAlignBlock * TRANSELEM;
    cUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.WLen, blockElem);
    cUbParamInfo_.UbFactor = basicInfo_.WAlignBlockElem;
    cUbParamInfo_.UbCount = 1;
    cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
    cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
    basicInfo_.UbLoopCount = rUbParamInfo_.UbCount;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Transpose021VCONVTiling::CalcUbSplitCConv()
{
    int64_t eleSize = shapeInfo_.eleLenInBytes;
    if (TRANSELEM * basicInfo_.HAlignBlockElem * eleSize > basicInfo_.AvailableUbSize) {
        OP_LOGD(context_->GetNodeName(), "H dimension too large for 021 VCONV branch");
        return ge::GRAPH_FAILED;
    }
    if (basicInfo_.HAlignBlockElem * basicInfo_.WAlignBlockElem * eleSize <= basicInfo_.AvailableUbSize) {
        rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSELEM);
        rUbParamInfo_.UbFactor = basicInfo_.HAlignBlockElem;
        rUbParamInfo_.UbCount = 1;
        rUbParamInfo_.UbTailAlignFactor = rUbParamInfo_.UbAlignFactor;
        rUbParamInfo_.UbTailFactor = rUbParamInfo_.UbFactor;
        cUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.WLen, TRANSELEM);
        cUbParamInfo_.UbFactor = basicInfo_.WAlignBlockElem;
        cUbParamInfo_.UbCount = 1;
        cUbParamInfo_.UbTailAlignFactor = cUbParamInfo_.UbAlignFactor;
        cUbParamInfo_.UbTailFactor = cUbParamInfo_.UbFactor;
        basicInfo_.UbLoopCount = 1;
        return ge::GRAPH_SUCCESS;
    }
    cUbParamInfo_.UbAlignFactor = basicInfo_.AvailableUbSize / eleSize / basicInfo_.HAlignBlockElem / TRANSELEM;
    if (eleSize == 1 && cUbParamInfo_.UbAlignFactor % NUM_TWO != 0) {
        cUbParamInfo_.UbAlignFactor -= 1;
    }
    if (cUbParamInfo_.UbAlignFactor <= 0) {
        OP_LOGD(context_->GetNodeName(), "W dimension too large for 8-bit 021 VCONV CConv branch");
        return ge::GRAPH_FAILED;
    }
    cUbParamInfo_.UbFactor = cUbParamInfo_.UbAlignFactor * TRANSELEM;
    int64_t wAlignBlock = Ops::Base::CeilDiv(basicInfo_.WLen, TRANSELEM);
    cUbParamInfo_.UbCount = Ops::Base::CeilDiv(wAlignBlock, cUbParamInfo_.UbAlignFactor);
    int64_t tailAlignBlock = wAlignBlock - (cUbParamInfo_.UbCount - 1) * cUbParamInfo_.UbAlignFactor;
    if (eleSize == 1 && tailAlignBlock % NUM_TWO != 0) {
        tailAlignBlock += 1;
    }
    cUbParamInfo_.UbTailAlignFactor = tailAlignBlock;
    cUbParamInfo_.UbTailFactor = tailAlignBlock * TRANSELEM;
    rUbParamInfo_.UbAlignFactor = Ops::Base::CeilDiv(basicInfo_.HLen, TRANSELEM);
    rUbParamInfo_.UbFactor = basicInfo_.HAlignBlockElem;
    rUbParamInfo_.UbCount = 1;
    rUbParamInfo_.UbTailAlignFactor = rUbParamInfo_.UbAlignFactor;
    rUbParamInfo_.UbTailFactor = rUbParamInfo_.UbFactor;
    basicInfo_.UbLoopCount = cUbParamInfo_.UbCount;
    return ge::GRAPH_SUCCESS;
}

void Transpose021VCONVTiling::FillUbPara(Transpose021UbSplitPara& ubSplitPara, UbParamInfo& ubPara)
{
    ubSplitPara.set_UbAlignFactor(ubPara.UbAlignFactor);
    ubSplitPara.set_UbFactor(ubPara.UbFactor);
    ubSplitPara.set_UbCount(ubPara.UbCount);
    ubSplitPara.set_UbTailAlignFactor(ubPara.UbTailAlignFactor);
    ubSplitPara.set_UbTailFactor(ubPara.UbTailFactor);
}

void Transpose021VCONVTiling::WriteTilingData()
{
    tiling_.set_AvailableUbSize(basicInfo_.AvailableUbSize);
    tiling_.set_UsedCoreNum(basicInfo_.UsedCoreNum);
    tiling_.set_NLen(basicInfo_.NLen);
    tiling_.set_HLen(basicInfo_.HLen);
    tiling_.set_WLen(basicInfo_.WLen);
    tiling_.set_HAlignBlockElem(basicInfo_.HAlignBlockElem);
    tiling_.set_WAlignBlockElem(basicInfo_.WAlignBlockElem);
    tiling_.set_NPerCore(basicInfo_.NPerCore);
    tiling_.set_NTailCore(basicInfo_.NTailCore);
    tiling_.set_UbLoopCount(basicInfo_.UbLoopCount);
    tiling_.set_UseRConv(basicInfo_.UseRConv);
    tiling_.set_UseHSplit(basicInfo_.UseHSplit);
    tiling_.set_HPerCore(basicInfo_.HPerCore);
    tiling_.set_HTailCore(basicInfo_.HTailCore);
    FillUbPara(tiling_.rUbSplitPara, rUbParamInfo_);
    FillUbPara(tiling_.cUbSplitPara, cUbParamInfo_);

    tiling_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling_.GetDataSize());
}

ge::graphStatus Transpose021VCONVTiling::SetTilingKeyAndCore()
{
    context_->SetTilingKey(static_cast<uint64_t>(SplitMode::VCONV_021_TRANSPOSE));
    OP_CHECK_IF(context_->SetBlockDim(basicInfo_.UsedCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Set used core num is failed!"), return ge::GRAPH_FAILED);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Transpose021VCONVTiling::DoTiling()
{
    CalcBasicInfo();
    if (basicInfo_.UseHSplit) {
        CalcHSplitInfo();
        int64_t nSplitCoreNum = std::min(basicInfo_.NLen, platInfo_.coreNum);
        if (basicInfo_.UsedCoreNum < std::max(static_cast<int64_t>(NUM_TWO), nSplitCoreNum)) {
            basicInfo_.UseHSplit = false;
            CalcNSplitInfo();
        }
    } else {
        CalcNSplitInfo();
    }
    OP_CHECK_IF(CalcUbSplitInfo() != ge::GRAPH_SUCCESS,
                OP_LOGD(context_->GetNodeName(), "Stop to run 021 VCONV tiling, UB split failed!"),
                return ge::GRAPH_FAILED);

    WriteTilingData();
    return SetTilingKeyAndCore();
}

} // namespace Transpose021WithVCONV
} // namespace optiling
