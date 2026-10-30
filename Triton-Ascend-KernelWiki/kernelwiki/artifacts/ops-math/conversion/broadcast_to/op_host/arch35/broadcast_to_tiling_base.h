/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file broadcast_to_tiling_base.h
 * \brief calc corenum and threadnum for AscendC kernel
 */
#ifndef BROADCASTTO_TILING_NDDMA_H_
#define BROADCASTTO_TILING_NDDMA_H_
#include <array>
#include <cstdint>

#include "broadcast_to_tiling_arch35.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
constexpr size_t brctoMaxDMADimNum = 0x5;
constexpr size_t brctoMaxADimNum = static_cast<size_t>(0x8) * 3; // axisSize, axisInASize, axisOutASize
constexpr size_t brctoMaxBDimNum = static_cast<size_t>(0x8) * 2; // axisSize, axisASize

BEGIN_TILING_DATA_DEF(BroadcastToTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(uint8_t, doubleMode);
TILING_DATA_FIELD_DEF(uint8_t, uAxisCnt);  // axis count in ub
TILING_DATA_FIELD_DEF(uint8_t, bufferCnt); // buffer count in ub
TILING_DATA_FIELD_DEF(uint8_t, blockAxis); // 0: A, 1: B, 2, U
TILING_DATA_FIELD_DEF(uint32_t, tensorSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreCnt);
TILING_DATA_FIELD_DEF(int64_t, ntcALen);
TILING_DATA_FIELD_DEF(int64_t, tcALen);
TILING_DATA_FIELD_DEF(int64_t, ntcBLen);
TILING_DATA_FIELD_DEF(int64_t, tcBLen);
TILING_DATA_FIELD_DEF(int64_t, ntcULen);
TILING_DATA_FIELD_DEF(int64_t, tcULen);
TILING_DATA_FIELD_DEF(int64_t, aLpUnit);
TILING_DATA_FIELD_DEF(int64_t, uLpUnit);
TILING_DATA_FIELD_DEF(int64_t, uInOffset);
TILING_DATA_FIELD_DEF(int64_t, uOutOffset);
TILING_DATA_FIELD_DEF(int32_t, isUNotB);
TILING_DATA_FIELD_DEF(int32_t, isLastDimB); // 0: A, 1:B
TILING_DATA_FIELD_DEF(int32_t, aAxesNum);
TILING_DATA_FIELD_DEF(int32_t, bAxesNum);
TILING_DATA_FIELD_DEF_ARR(uint64_t, brctoMaxDMADimNum, xSrcStride);
TILING_DATA_FIELD_DEF_ARR(uint32_t, brctoMaxDMADimNum, xDstStride);
TILING_DATA_FIELD_DEF_ARR(uint32_t, brctoMaxDMADimNum, xSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, brctoMaxADimNum, aAxesParams);
TILING_DATA_FIELD_DEF_ARR(int64_t, brctoMaxBDimNum, bAxesParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BroadcastTo, BroadcastToTilingData);

ge::graphStatus Tiling4BroadcastToAscendC(gert::TilingContext* context, const gert::Shape* inShapePtr,
                                          const gert::Shape* outShapePtr);

namespace brcto {
constexpr int64_t MAX_TENSOR_SIZE = 0xff00; // to avoid index overflow when B8
constexpr int64_t TILING_MODE_NDDMA = 11000;
constexpr int64_t TILING_MODE_UB_BRC = 11001;
constexpr int64_t TILING_MODE_LAST_DIM_LARGE_A = 11002;
constexpr int64_t TILING_MODE_LAST_DIM_LARGE_B = 11003;
constexpr int64_t TILING_MODE_FULL_NDDMA = 11004;
constexpr int64_t TILING_MODE_LAST_DIM_SMALL_A = 11005;
constexpr size_t MAX_DIM_NUM = 0x10;
constexpr size_t BRCTO_MAX_DIM_NUM = 0x8;
constexpr size_t aParamUnit = 3;
constexpr size_t bParamUnit = 2;
constexpr int64_t nTwo = 2;
constexpr size_t kSyncWorkSpaceSize = static_cast<size_t>(16) * 1024 * 1024;
constexpr int64_t maxDataSize = static_cast<int64_t>(128) * 1024;
constexpr float coreFactor = 0.75;
constexpr int64_t LAST_DIM_GATE = 8;

ge::graphStatus GetShapeInfo(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape);
ge::graphStatus CheckSameDimNum(const gert::TilingContext* context, const gert::Shape& inShape,
                                const gert::Shape& outShape, size_t& dimNum);
ge::graphStatus GetABFlag(const gert::TilingContext* context, const gert::Shape& inShape, const gert::Shape& outShape,
                          std::array<bool, MAX_DIM_NUM>& abInfo);
void AdjustShapesToSameDimNum(gert::Shape& inShape, size_t outDimNum);
ge::graphStatus MergeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape);
ge::graphStatus DeleteOneSizeAxis(const gert::TilingContext* context, gert::Shape& inShape, gert::Shape& outShape);

class BroadcastToTilingAscendC {
public:
    explicit BroadcastToTilingAscendC(gert::TilingContext* context, const gert::Shape* inShapePtr,
                                      const gert::Shape* outShapePtr)
        : context_(context), inShapePtr_(inShapePtr), outShapePtr_(outShapePtr) {};
    ge::graphStatus DoTiling();

    // define stay with implementation so that its implementation can be found
    // when GetHardwareInfo() is called
    template <typename T>
    ge::graphStatus GetHardwareInfo()
    {
        // use user specify compile info
        auto compileInfo = context_->GetCompileInfo<T>();
        OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
        ubSize_ = static_cast<int64_t>(compileInfo->ubSize);
        coreNum_ = static_cast<int64_t>(compileInfo->coreNum);
        blockSize_ = static_cast<int64_t>(compileInfo->blockSize);
        cacheLine_ = static_cast<int64_t>(compileInfo->clSize);
        vlSize_ = static_cast<int64_t>(compileInfo->vRegSize);
        if (coreNum_ <= 0 || ubSize_ <= 0 || blockSize_ <= 0 || cacheLine_ <= 0 || vlSize_ <= 0) {
            std::string valueMsg = "coreNum=" + std::to_string(coreNum_) + ", ubSize=" + std::to_string(ubSize_) +
                                   ", blockSize=" + std::to_string(blockSize_) +
                                   ", cacheLine=" + std::to_string(cacheLine_) + ", vlSize=" + std::to_string(vlSize_);
            std::string reasonMsg = "BroadcastTo GetHardwareInfo failed, all values must be positive.";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "hardwareInfo", valueMsg.c_str(),
                                                  reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }

        auto inputDesc = context_->GetInputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
        auto dtype = inputDesc->GetDataType();
        dtypeSize_ = GetSizeByDataType(dtype);
        OP_LOGI(context_->GetNodeName(), "The ub size is: %ld", ubSize_);

        return ge::GRAPH_SUCCESS;
    }

private:
    void CalcTilingData();
    int64_t CalcDimSize(const gert::Shape*& shapePtr, size_t begDim, size_t endDim);
    int64_t UpdateTensorSize(int64_t tensorSize);
    void GetUAxisInfo();
    void GetDMAAxesParams();
    void GetABAxesParams();
    void GetAxesInfo();
    void GetMCTilingInfo();
    void OptimizeMCTilingForDoubleMode();
    void UpdateDimSize(int64_t& aDims, int64_t& bDims, int64_t& brwAxis, int64_t& outLastDim);
    void CheckBrwd(int64_t& aDims, int64_t& bDims, int64_t& brwAxis, bool& isBrwd);
    int64_t CalcTensorSize4Brwd(int64_t aDims, int64_t bDims, int64_t brwAxis);
    int64_t CalcTensorSize4NBrwd(int64_t aDims, int64_t bDims, int64_t outLastDim);
    void AdjustBrwdSize(int64_t& brwSize, int64_t uAxis);
    void CalcTensorSize();
    void CalcTilingKey();
    void CalcDBMode();
    void UpdateTilingKey();
    ge::graphStatus WriteTilingData();
    ge::graphStatus SetBlockCnt();
    std::string PrintTilingData();
    uint32_t CalcAxisWeight(int64_t lpCnt);

private:
    gert::TilingContext* context_ = nullptr;
    const gert::Shape* inShapePtr_ = nullptr;
    const gert::Shape* outShapePtr_ = nullptr;
    BroadcastToTilingData tilingData_;

    int64_t coreNum_{0};
    int64_t ubSize_{0};
    int64_t vlSize_{0};
    int64_t cacheLine_{0};
    int64_t blockSize_{0};

    int64_t aAxisLen_{1};
    int64_t bAxisLen_{1};
    int64_t uAxisLen_{1};
    int64_t dtypeSize_{1};
    size_t uAxis_{0};
    bool isDMABrcA_{false};
    std::array<bool, MAX_DIM_NUM> abInfo_{0};

    int64_t maxTensorSize_{0};
    int64_t minTensorSize_{0};

    int64_t tilingKey_{0};
    int64_t dFactor_{1};
    int8_t doubleMode_{0};
    int8_t blockAxis_{0};
    int8_t uAxisCnt_{1};
    int8_t bufferCnt_{nTwo};
    int64_t tensorSize_{0};
    int64_t usedCoreCnt_{0};
    int64_t ntcALen_{1};
    int64_t tcALen_{0};
    int64_t ntcBLen_{1};
    int64_t tcBLen_{0};
    int64_t ntcULen_{1};
    int64_t tcULen_{0};
    int64_t aLpUnit_{1};
    int64_t uLpUnit_{1};
    int64_t uInOffset_{0};
    int64_t uOutOffset_{0};
    int32_t isUNotB_{1};
    int32_t isLastDimB_{0};
    int32_t aAxesNum_{0};
    int32_t bAxesNum_{0};
    uint64_t xSrcStride_[brctoMaxDMADimNum]{0};
    uint32_t xDstStride_[brctoMaxDMADimNum]{0};
    uint32_t xSize_[brctoMaxDMADimNum]{1, 1, 1, 1, 1};
    int64_t aAxesParams_[brctoMaxADimNum]{0};
    int64_t bAxesParams_[brctoMaxBDimNum]{0};
};

} // namespace brcto

} // namespace optiling
#endif // BROADCASTTO_TILING_NDDMA_H_
