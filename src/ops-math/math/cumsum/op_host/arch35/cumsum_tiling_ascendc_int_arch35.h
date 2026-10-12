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
 * \file cumsum_tiling_ascendc_int.h
 * \brief calc corenum and threadnum for AscendC kernel
 */
#ifndef CUMSUM_TILING_INT_H_
#define CUMSUM_TILING_INT_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Cum4IntTilingData)
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, isExclusive);
TILING_DATA_FIELD_DEF(int64_t, isReverse);
TILING_DATA_FIELD_DEF(int64_t, isRBlockAxis);
TILING_DATA_FIELD_DEF(int64_t, tensorSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreCnt);
TILING_DATA_FIELD_DEF(int64_t, blockOffset);
TILING_DATA_FIELD_DEF(int64_t, ntcLALen);
TILING_DATA_FIELD_DEF(int64_t, tcLALen);
TILING_DATA_FIELD_DEF(int64_t, laLpUnit);
TILING_DATA_FIELD_DEF(int64_t, laOffset);
TILING_DATA_FIELD_DEF(int64_t, ntcRLen);
TILING_DATA_FIELD_DEF(int64_t, tcRLen);
TILING_DATA_FIELD_DEF(int64_t, rLpUnit);
TILING_DATA_FIELD_DEF(int64_t, rOffset);
TILING_DATA_FIELD_DEF(int64_t, ntcRALen);
TILING_DATA_FIELD_DEF(int64_t, tcRALen);
TILING_DATA_FIELD_DEF(int64_t, raLpUnit);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cumsum_11000, Cum4IntTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_11001, Cum4IntTilingData);
REGISTER_TILING_DATA_CLASS(Cumsum_11002, Cum4IntTilingData);

ge::graphStatus TilingCumsum4Int(gert::TilingContext* context);

namespace cumsum {
constexpr int64_t NUM_TWO = 2;
constexpr int64_t CORE_GATE = 4;
constexpr int64_t CUM_NO_SPLIT = 11000;
constexpr int64_t CUM_AR_SPLIT = 11001;
constexpr int64_t CUM_WITH_GROUP = 11002;
constexpr int64_t MAX_TENSOR_SIZE = 0xff00; // to avoid index overflow when B8

class Cumsum4IntTiling {
public:
    explicit Cumsum4IntTiling(gert::TilingContext* context) : context_(context) {};
    ge::graphStatus DoTiling();

private:
    ge::graphStatus GetHardwareInfo();
    ge::graphStatus GetInputDims();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus CalcTilingData();
    void AdjustTensor4TDRA();
    void AdjustTensor4TDLA(int64_t comLeftA);
    void AdjustTensor4TDR(int64_t comLeftA);
    void AdjustLARLpUnit(int64_t comLeftA);
    bool CheckBGC(int64_t comLeftA, int64_t arSize);
    void GetAxisLpUnit();
    void GetMCTilingInfo();
    void CalcTilingKey();
    void WriteTilingData();
    std::string PrintTilingData();
    uint32_t CalcAxisWeight(int64_t lpCnt);

private:
    gert::TilingContext* context_ = nullptr;
    Cum4IntTilingData tilingData_;

    int64_t coreNum_{0};
    int64_t ubSize_{0};
    int64_t vlSize_{0};
    int64_t cacheLine_{0};
    int64_t blockSize_{0};

    int64_t leftAxisLen = 1;
    int64_t midAxisLen = 1; // cumsum axis
    int64_t rightAxisLen = 1;
    int64_t dtypeSize = 1;
    int64_t minTensorSize = 0;
    int64_t maxTensorSize = 0;
    int64_t rsvUB = 2048;

    int64_t tilingKey_;
    int64_t isExclusive_;
    int64_t isReverse_;
    int64_t isRBlockAxis_;
    int64_t tensorSize_;
    int64_t usedCoreCnt_;
    int64_t blockOffset_;
    int64_t ntcLALen_;
    int64_t tcLALen_;
    int64_t laLpUnit_;
    int64_t laOffset_;
    int64_t ntcRLen_;
    int64_t tcRLen_;
    int64_t rLpUnit_;
    int64_t rOffset_;
    int64_t ntcRALen_;
    int64_t tcRALen_;
    int64_t raLpUnit_;
};
} // namespace cumsum

} // namespace optiling
#endif // CUMSUM_TILING_INT_H_
