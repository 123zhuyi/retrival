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
 * \file cumsum_tiling_ascendc_int.cc
 * \brief calc corenum and threadnum for AscendC kernel
 */

#include "cumsum_tiling_ascendc_int_arch35.h"
#include "cumsum_tiling.h"
#include "log/log.h"
#include "util/const_util.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"

namespace optiling {
namespace cumsum {

ge::graphStatus Cumsum4IntTiling::GetHardwareInfo()
{
    auto compileInfo = reinterpret_cast<const CumsumCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    ubSize_ = static_cast<int64_t>(compileInfo->ub_size);
    coreNum_ = static_cast<int64_t>(compileInfo->core_num);
    blockSize_ = static_cast<int64_t>(compileInfo->blockSize);
    cacheLine_ = static_cast<int64_t>(compileInfo->clSize);
    vlSize_ = static_cast<int64_t>(compileInfo->vRegSize);
    OP_CHECK_IF(
        (coreNum_ <= 0 || ubSize_ <= 0 || blockSize_ <= 0 || cacheLine_ <= 0 || vlSize_ <= 0),
        OP_LOGE(
            context_->GetNodeName(),
            "Cumsum4Int GetHardwareInfo Failed, Core count:%ld, UB size:%ld, "
            "Block size:%ld, Cache line:%ld, VL size:%ld.",
            coreNum_, ubSize_, blockSize_, cacheLine_, vlSize_),
        return ge::GRAPH_FAILED);

    auto dtype = context_->GetInputDesc(0)->GetDataType();
    dtypeSize = GetSizeByDataType(dtype);
    if (dtypeSize == int64_t(1)) {
        vlSize_ /= NUM_TWO;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Cumsum4IntTiling::GetInputDims()
{
    constexpr int64_t inputIdxAxis = 1;
    int64_t axisAttr = static_cast<int64_t>(-1);
    OP_CHECK_IF(
        !Ops::Base::GetConstInt(context_, inputIdxAxis, axisAttr),
        OP_LOGE(context_->GetNodeName(), "Axis GetConstInt error!"), return ge::GRAPH_FAILED);

    auto xStorage = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    gert::Shape xShape_ = Ops::Base::EnsureNotScalar(xStorage->GetStorageShape());
    size_t xDimNum = xShape_.GetDimNum();
    OP_CHECK_IF(
        axisAttr >= static_cast<int64_t>(xDimNum) || axisAttr < static_cast<int64_t>(-xDimNum),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "axis",
            std::to_string(axisAttr).c_str(),
            ("The value of axis must be within the range [-" + std::to_string(xDimNum) + 
             ", " + std::to_string(xDimNum - 1) + "].").c_str()),
        return ge::GRAPH_FAILED);

    if (axisAttr < 0) {
        axisAttr += static_cast<int64_t>(xDimNum);
    }
    uint32_t axis = static_cast<uint32_t>(axisAttr);

    midAxisLen = xShape_[axis];
    if (axis == 0u) {
        for (size_t i = 1; i < xDimNum; i++) {
            rightAxisLen = rightAxisLen * xShape_[i];
        }
    } else if (axis == static_cast<uint32_t>(xDimNum) - 1U) {
        for (size_t i = 0; i < xDimNum - 1UL; i++) {
            leftAxisLen = leftAxisLen * xShape_[i];
        }
    } else {
        for (size_t i = 0; i < axis; i++) {
            leftAxisLen = leftAxisLen * xShape_[i];
        }
        for (size_t i = static_cast<size_t>(axis) + 1UL; i < xDimNum; i++) {
            rightAxisLen = rightAxisLen * xShape_[i];
        }
    }

    laOffset_ = midAxisLen * rightAxisLen;
    rOffset_ = rightAxisLen;

    return ge::GRAPH_SUCCESS;
}

void Cumsum4IntTiling::AdjustTensor4TDRA()
{
    tensorSize_ = maxTensorSize;
    int64_t tmpRALpUnit = Ops::Base::CeilAlign(std::min(rightAxisLen * dtypeSize, cacheLine_), blockSize_) / dtypeSize;
    int64_t tmpRLpUnit = std::min(tensorSize_ / tmpRALpUnit, midAxisLen);
    if (rightAxisLen * dtypeSize > cacheLine_) {
        OP_CHECK_IF(
            dtypeSize == 0,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "input",
                Ops::Base::ToString(context_->GetInputDesc(0)->GetDataType()).c_str(),
                "The dtype size of input must be greater than 0."),
            return);
        OP_CHECK_IF(
            cacheLine_ == 0, OP_LOGE("AdjustTensor4TDRA", "cacheLine_ is zero"),
            return);
        tmpRALpUnit =
            std::min(tensorSize_ / tmpRLpUnit * dtypeSize / cacheLine_ * cacheLine_ / dtypeSize, rightAxisLen);
    }
    // split cores on R
    uint32_t rWeight = CalcAxisWeight(Ops::Base::CeilDiv(midAxisLen, tmpRLpUnit));
    if (rWeight > uint32_t(NUM_TWO) * CalcAxisWeight(Ops::Base::CeilDiv(rightAxisLen, tmpRALpUnit)) &&
        rWeight > uint32_t(NUM_TWO) * CalcAxisWeight(leftAxisLen) && tmpRALpUnit <= minTensorSize) {
        tensorSize_ = std::min(
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(midAxisLen, coreNum_) * tmpRALpUnit, minTensorSize), maxTensorSize);
        return;
    }
    // split cores on RA
    if (CalcAxisWeight(Ops::Base::CeilDiv(rightAxisLen, tmpRALpUnit)) > CalcAxisWeight(leftAxisLen) &&
        tmpRLpUnit == midAxisLen) {
        tensorSize_ = std::min(
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(rightAxisLen, coreNum_), minTensorSize) * midAxisLen,
            maxTensorSize);
    }
}

void Cumsum4IntTiling::AdjustTensor4TDLA(int64_t comLeftA)
{
    tensorSize_ = maxTensorSize;
    OP_CHECK_IF(
        (comLeftA == 0 || raLpUnit_ == 0), OP_LOGE("AdjustTensor4TDLA", "comLeftA or raLpUnit_ is zero"), return);
    int64_t tmpRLpUnit = std::min(tensorSize_ / comLeftA / raLpUnit_, midAxisLen);
    if (tmpRLpUnit == midAxisLen) {
        int64_t tmpTensorSize_ = Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(leftAxisLen, comLeftA), coreNum_) * comLeftA * midAxisLen *
                rightAxisLen,
            minTensorSize);
        tensorSize_ = std::min(tmpTensorSize_, tensorSize_);
    }
}

void Cumsum4IntTiling::AdjustTensor4TDR(int64_t comLeftA)
{
    // to avoid ub overflow for solving bank group conflict
    tensorSize_ = maxTensorSize - rsvUB / dtypeSize;
    int64_t tmpRLpUnit = std::min(tensorSize_ / rightAxisLen, midAxisLen);
    OP_CHECK_IF((comLeftA == 0), OP_LOGE("AdjustTensor4TDR", "comLeftA is zero"), return);
    int64_t rLpUnit = (tmpRLpUnit > comLeftA) ? tmpRLpUnit / comLeftA * comLeftA : tmpRLpUnit;
    int64_t rLpCnt = Ops::Base::CeilDiv(midAxisLen, rLpUnit);
    // split cores on R
    if (CalcAxisWeight(leftAxisLen) * uint32_t(NUM_TWO) < CalcAxisWeight(rLpCnt)) {
        int64_t tmpTensorSize_ =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(midAxisLen * rightAxisLen, minTensorSize), coreNum_) * minTensorSize;
        tensorSize_ = std::min(tmpTensorSize_, tensorSize_);
    }
}

bool Cumsum4IntTiling::CheckBGC(int64_t comLeftA, int64_t arSize)
{
    constexpr int64_t halfBGCnt = 8;
    constexpr int64_t parallBytes = 512;
    int64_t comCnt = std::min(halfBGCnt, comLeftA);
    int64_t sizeInByte =
        (midAxisLen > rLpUnit_) ? Ops::Base::CeilAlign(arSize * dtypeSize, blockSize_) : arSize * dtypeSize;
    if (sizeInByte % blockSize_ == 0 && sizeInByte * comCnt % parallBytes == 0) {
        return true;
    }
    return false;
}

void Cumsum4IntTiling::AdjustLARLpUnit(int64_t comLeftA)
{
    int64_t arSize = Ops::Base::CeilAlign(rLpUnit_ * raLpUnit_ * dtypeSize + blockSize_, blockSize_) / dtypeSize;
    if (arSize * laLpUnit_ > maxTensorSize) {
        int64_t tmpLALpUnit = tensorSize_ / arSize;
        if (tmpLALpUnit < NUM_TWO) {
            rLpUnit_ = (rLpUnit_ * raLpUnit_ - blockSize_ / dtypeSize) / raLpUnit_;
            arSize = Ops::Base::CeilAlign(rLpUnit_ * raLpUnit_ * dtypeSize, blockSize_) / dtypeSize;
            tmpLALpUnit = tensorSize_ / arSize;
        }
        OP_CHECK_IF((comLeftA == 0), OP_LOGE("AdjustLARLpUnit", "comLeftA is zero"), return);
        tmpLALpUnit = (tmpLALpUnit > comLeftA) ? tmpLALpUnit / comLeftA * comLeftA : tmpLALpUnit;
        laLpUnit_ = std::min(tmpLALpUnit, leftAxisLen);
    }
}

void Cumsum4IntTiling::GetAxisLpUnit()
{
    // enable db
    maxTensorSize = std::min(ubSize_ / NUM_TWO / NUM_TWO / blockSize_ * blockSize_ / dtypeSize, MAX_TENSOR_SIZE);
    minTensorSize = cacheLine_ / dtypeSize;
    raLpUnit_ = (rightAxisLen * dtypeSize > cacheLine_) ? minTensorSize : rightAxisLen;

    if (rightAxisLen * dtypeSize > vlSize_ / NUM_TWO) {
        // TD rightA
        AdjustTensor4TDRA();
        laLpUnit_ = 1;
        int64_t raLpUnitBA = Ops::Base::CeilAlign(raLpUnit_ * dtypeSize, blockSize_) / dtypeSize;
        rLpUnit_ = std::min(tensorSize_ / raLpUnitBA, midAxisLen);
        if (rightAxisLen * dtypeSize > cacheLine_) {
            // block align
            raLpUnit_ = tensorSize_ / rLpUnit_ * dtypeSize / cacheLine_ * cacheLine_ / dtypeSize;
            raLpUnit_ = (raLpUnit_ > rightAxisLen) ? rightAxisLen : raLpUnit_;
        }
        tensorSize_ = maxTensorSize;
        return;
    }

    int64_t comLeftA = vlSize_ / (raLpUnit_ * dtypeSize);
    int64_t tmpRLpUnit = 0;
    // TD leftA
    if (leftAxisLen > coreNum_ / CORE_GATE) {
        int64_t comLeftA_ = std::min(comLeftA, leftAxisLen);
        AdjustTensor4TDLA(comLeftA_);
        tmpRLpUnit = tensorSize_ / comLeftA_ / raLpUnit_;
        rLpUnit_ = std::min(tmpRLpUnit, midAxisLen);
        int64_t arSize = Ops::Base::CeilAlign(rLpUnit_ * raLpUnit_ * dtypeSize, blockSize_) / dtypeSize;
        int64_t tmpLALpUnit = tensorSize_ / arSize;
        tmpLALpUnit = (tmpLALpUnit > comLeftA_) ? tmpLALpUnit / comLeftA_ * comLeftA_ : tmpLALpUnit;
        laLpUnit_ = std::min(tmpLALpUnit, leftAxisLen);
        tensorSize_ = maxTensorSize;
        // to avoid bank group conflict
        if (CheckBGC(comLeftA_, rLpUnit_ * raLpUnit_)) {
            AdjustLARLpUnit(comLeftA_);
        }
        // to avoid going into branch Group-R
        OP_CHECK_IF(
            (laLpUnit_ == 0 || rLpUnit_ == 0), OP_LOGE("GetAxisLpUnit", "laLpUnit_ or rLpUnit_ is zero"), return);
        if (leftAxisLen / laLpUnit_ >= midAxisLen / rLpUnit_ / NUM_TWO || rLpUnit_ == midAxisLen) {
            return;
        }
    }

    // TD R
    AdjustTensor4TDR(comLeftA);
    laLpUnit_ = 1;
    tmpRLpUnit = std::min(tensorSize_ / raLpUnit_, midAxisLen);
    rLpUnit_ = (tmpRLpUnit > comLeftA) ? tmpRLpUnit / comLeftA * comLeftA : tmpRLpUnit;
    tensorSize_ = maxTensorSize;
}

uint32_t Cumsum4IntTiling::CalcAxisWeight(int64_t lpCnt)
{
    uint32_t weight = 0;

    if (lpCnt >= coreNum_) {
        weight += static_cast<uint32_t>(coreNum_);
    }
    if (lpCnt % coreNum_ == 0) {
        weight += static_cast<uint32_t>(coreNum_);
    } else {
        weight += static_cast<uint32_t>(lpCnt % coreNum_);
    }

    return weight;
}

void Cumsum4IntTiling::GetMCTilingInfo()
{
    int64_t laLpCnt = Ops::Base::CeilDiv(leftAxisLen, laLpUnit_);
    int64_t rLpCnt = Ops::Base::CeilDiv(midAxisLen, rLpUnit_);
    int64_t raLpCnt = Ops::Base::CeilDiv(rightAxisLen, raLpUnit_);
    uint32_t laAxisWeight = CalcAxisWeight(laLpCnt);
    uint32_t rAxisWeight = CalcAxisWeight(rLpCnt);
    uint32_t raAxisWeight = CalcAxisWeight(raLpCnt);
    // R轴分核场景性能较差，除非核数明显占优
    uint32_t maxWeight = std::max(laAxisWeight, std::max(rAxisWeight / uint32_t(NUM_TWO), raAxisWeight));
    int64_t mLpCnt = 0;

    if (maxWeight == laAxisWeight) {
        isRBlockAxis_ = 0;
        usedCoreCnt_ = Ops::Base::CeilDiv(laLpCnt, Ops::Base::CeilDiv(laLpCnt, coreNum_));
        mLpCnt = Ops::Base::CeilDiv(laLpCnt, usedCoreCnt_);
        ntcLALen_ = mLpCnt * laLpUnit_;
        tcLALen_ = leftAxisLen - ntcLALen_ * (usedCoreCnt_ - 1);
        ntcRLen_ = midAxisLen;
        tcRLen_ = midAxisLen;
        ntcRALen_ = rightAxisLen;
        tcRALen_ = rightAxisLen;
        blockOffset_ = ntcLALen_ * laOffset_;
    } else if (maxWeight == raAxisWeight) {
        isRBlockAxis_ = 0;
        usedCoreCnt_ = Ops::Base::CeilDiv(raLpCnt, Ops::Base::CeilDiv(raLpCnt, coreNum_));
        mLpCnt = Ops::Base::CeilDiv(raLpCnt, usedCoreCnt_);
        ntcRALen_ = mLpCnt * raLpUnit_;
        tcRALen_ = rightAxisLen - ntcRALen_ * (usedCoreCnt_ - 1);
        ntcLALen_ = leftAxisLen;
        tcLALen_ = leftAxisLen;
        ntcRLen_ = midAxisLen;
        tcRLen_ = midAxisLen;
        blockOffset_ = ntcRALen_;
    } else {
        isRBlockAxis_ = 1;
        usedCoreCnt_ = Ops::Base::CeilDiv(rLpCnt, Ops::Base::CeilDiv(rLpCnt, coreNum_));
        mLpCnt = Ops::Base::CeilDiv(rLpCnt, usedCoreCnt_);
        ntcRLen_ = mLpCnt * rLpUnit_;
        tcRLen_ = midAxisLen - ntcRLen_ * (usedCoreCnt_ - 1);
        ntcLALen_ = leftAxisLen;
        tcLALen_ = leftAxisLen;
        ntcRALen_ = rightAxisLen;
        tcRALen_ = rightAxisLen;
        blockOffset_ = ntcRLen_ * rOffset_;
    }
}

void Cumsum4IntTiling::CalcTilingKey()
{
    if (isRBlockAxis_ == 1) {
        tilingKey_ = CUM_WITH_GROUP;
    } else if (rightAxisLen * dtypeSize >= vlSize_) {
        tilingKey_ = CUM_NO_SPLIT;
    } else {
        tilingKey_ = CUM_AR_SPLIT;
    }
}

ge::graphStatus Cumsum4IntTiling::CalcTilingData()
{
    OP_CHECK_IF(
        (GetInputDims() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "Cumsum4IntTiling failed to get input shape info."), return ge::GRAPH_FAILED);

    GetAxisLpUnit();
    GetMCTilingInfo();
    CalcTilingKey();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Cumsum4IntTiling::GetAttrInfo()
{
    auto attrs = context_->GetAttrs();
    const bool* exclusive = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, exclusive);
    const bool* reverse = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reverse);

    OP_LOGI("Cumsum4IntTiling", "The exclusive is: %d.", *exclusive);
    OP_LOGI("Cumsum4IntTiling", "The reverse is: %d.", *reverse);
    isExclusive_ = static_cast<int64_t>(*exclusive);
    isReverse_ = static_cast<int64_t>(*reverse);

    return ge::GRAPH_SUCCESS;
}

void Cumsum4IntTiling::WriteTilingData()
{
    context_->SetBlockDim(usedCoreCnt_);
    context_->SetTilingKey(tilingKey_);
    if (tilingKey_ == CUM_WITH_GROUP) {
        OP_CHECK_IF(context_->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "Failed to set ScheduleMode!"),
                    return);
    }
    tilingData_.set_tilingKey(tilingKey_);
    tilingData_.set_isExclusive(isExclusive_);
    tilingData_.set_isReverse(isReverse_);
    tilingData_.set_isRBlockAxis(isRBlockAxis_);
    tilingData_.set_tensorSize(tensorSize_);
    tilingData_.set_usedCoreCnt(usedCoreCnt_);
    tilingData_.set_blockOffset(blockOffset_);
    tilingData_.set_ntcLALen(ntcLALen_);
    tilingData_.set_tcLALen(tcLALen_);
    tilingData_.set_laLpUnit(laLpUnit_);
    tilingData_.set_laOffset(laOffset_);
    tilingData_.set_ntcRLen(ntcRLen_);
    tilingData_.set_tcRLen(tcRLen_);
    tilingData_.set_rLpUnit(rLpUnit_);
    tilingData_.set_rOffset(rOffset_);
    tilingData_.set_ntcRALen(ntcRALen_);
    tilingData_.set_tcRALen(tcRALen_);
    tilingData_.set_raLpUnit(raLpUnit_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    constexpr size_t sysWorkspaceSize = size_t(16 * 1024 * 1024);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
}

std::string Cumsum4IntTiling::PrintTilingData()
{
    std::string tilingStr;

    tilingStr += std::to_string(tilingKey_) + ",";
    tilingStr += std::to_string(isExclusive_) + ",";
    tilingStr += std::to_string(isReverse_) + ",";
    tilingStr += std::to_string(isRBlockAxis_) + ",";
    tilingStr += std::to_string(tensorSize_) + ",";
    tilingStr += std::to_string(usedCoreCnt_) + ",";
    tilingStr += std::to_string(blockOffset_) + ",";
    tilingStr += std::to_string(ntcLALen_) + ",";
    tilingStr += std::to_string(tcLALen_) + ",";
    tilingStr += std::to_string(laLpUnit_) + ",";
    tilingStr += std::to_string(laOffset_) + ",";
    tilingStr += std::to_string(ntcRLen_) + ",";
    tilingStr += std::to_string(tcRLen_) + ",";
    tilingStr += std::to_string(rLpUnit_) + ",";
    tilingStr += std::to_string(rOffset_) + ",";
    tilingStr += std::to_string(ntcRALen_) + ",";
    tilingStr += std::to_string(tcRALen_) + ",";
    tilingStr += std::to_string(raLpUnit_) + ".";

    return tilingStr;
}

ge::graphStatus Cumsum4IntTiling::DoTiling()
{
    OP_CHECK_IF(
        (GetHardwareInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "Cumsum4IntTiling failed to get hardware info."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (GetAttrInfo() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "Cumsum4IntTiling failed to get input attr."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (CalcTilingData() != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "Cumsum4IntTiling failed to caculate tiling data."), return ge::GRAPH_FAILED);
    WriteTilingData();
    OP_LOGI("Cumsum4IntTiling", "The tiling data is: %s", PrintTilingData().c_str());

    return ge::GRAPH_SUCCESS;
}
} // namespace cumsum

ge::graphStatus TilingCumsum4Int(gert::TilingContext* context)
{
    cumsum::Cumsum4IntTiling cum4IntTiling(context);
    return cum4IntTiling.DoTiling();
}

} // namespace optiling
