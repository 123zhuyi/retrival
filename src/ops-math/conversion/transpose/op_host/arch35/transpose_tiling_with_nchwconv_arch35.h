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
 * \file transpose_tiling_with_nchwconv_arch35.h
 * \brief transpose_tiling_with_nchwconv_arch35
 */
#ifndef _TRANSPOSE_TILING_WITH_NCHWCONV_ARCH35_H_
#define _TRANSPOSE_TILING_WITH_NCHWCONV_ARCH35_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

#include "util/math_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "transpose_tiling_base.h"
#include "transpose_tiling_arch35.h"

namespace optiling {
namespace TransposeWithVCONV {
static constexpr int8_t BUFFER_NUM = 2;

BEGIN_TILING_DATA_DEF(CoreSplitPara)
TILING_DATA_FIELD_DEF(int64_t, AlignBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, BlockFactor);
TILING_DATA_FIELD_DEF(int64_t, BlockCount);
TILING_DATA_FIELD_DEF(int64_t, AlignBlockTailFactor);
TILING_DATA_FIELD_DEF(int64_t, BlockTailFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CoreSplitParaOp, CoreSplitPara);

BEGIN_TILING_DATA_DEF(UbSplitPara)
TILING_DATA_FIELD_DEF(int64_t, MainCoreUbAlignFactor);
TILING_DATA_FIELD_DEF(int64_t, MainCoreUbFactor);
TILING_DATA_FIELD_DEF(int64_t, MainCoreUbCount);
TILING_DATA_FIELD_DEF(int64_t, MainCoreTailUbAlignFactor);
TILING_DATA_FIELD_DEF(int64_t, MainCoreTailUbFactor);
TILING_DATA_FIELD_DEF(int64_t, TailCoreUbAlignFactor);
TILING_DATA_FIELD_DEF(int64_t, TailCoreUbFactor);
TILING_DATA_FIELD_DEF(int64_t, TailCoreUbCount);
TILING_DATA_FIELD_DEF(int64_t, TailCoreTailUbAlignFactor);
TILING_DATA_FIELD_DEF(int64_t, TailCoreTailUbFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(UbSplitParaOp, UbSplitPara);

BEGIN_TILING_DATA_DEF(TransposeVCONVTilingData)
TILING_DATA_FIELD_DEF(int64_t, AvailableUbSize);
TILING_DATA_FIELD_DEF(int64_t, UsedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, MainCoreLoopCount);
TILING_DATA_FIELD_DEF(int64_t, TailCoreLoopCount);
TILING_DATA_FIELD_DEF(int64_t, RLen);
TILING_DATA_FIELD_DEF(int64_t, CLen);
TILING_DATA_FIELD_DEF(int64_t, RAlignBlock);
TILING_DATA_FIELD_DEF(int64_t, CAlignBlock);
TILING_DATA_FIELD_DEF(int64_t, RAlignBlockElem);
TILING_DATA_FIELD_DEF(int64_t, CAlignBlockElem);
TILING_DATA_FIELD_DEF(bool, IsRSplit);
TILING_DATA_FIELD_DEF(bool, IsRCSplit);
TILING_DATA_FIELD_DEF_STRUCT(CoreSplitPara, rSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(CoreSplitPara, cSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(UbSplitPara, rUbSplitPara);
TILING_DATA_FIELD_DEF_STRUCT(UbSplitPara, cUbSplitPara);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose_10007, TransposeVCONVTilingData);

struct PlatInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct BasicInfo {
    int64_t AvailableUbSize;
    int64_t UsedCoreNum;
    int64_t MainCoreLoopCount;
    int64_t TailCoreLoopCount;
    int64_t RLen;
    int64_t CLen;
    int64_t RAlignBlock;
    int64_t CAlignBlock;
    int64_t RAlignBlockElem;
    int64_t CAlignBlockElem;
    bool IsRSplit = false;
    bool IsRCSplit = false;
};

struct CoreSplitInfo {
    // r 方向
    int64_t AlignBlockFactor;
    int64_t BlockFactor;
    int64_t BlockCount;
    int64_t AlignBlockTailFactor;
    int64_t BlockTailFactor;
};

struct UbParamInfo {
    // 主核
    int64_t MainCoreUbAlignFactor;
    int64_t MainCoreUbFactor;
    int64_t MainCoreUbCount;
    int64_t MainCoreTailUbAlignFactor;
    int64_t MainCoreTailUbFactor;
    // 尾核
    int64_t TailCoreUbAlignFactor;
    int64_t TailCoreUbFactor;
    int64_t TailCoreUbCount;
    int64_t TailCoreTailUbAlignFactor;
    int64_t TailCoreTailUbFactor;
};

class TransposeVCONVTiling {
public:
    explicit TransposeVCONVTiling(gert::TilingContext* context, const PlatInfo& platInfo, const ShapeInfo& shapeInfo)
        : context_(context), platInfo_(platInfo), shapeInfo_(shapeInfo) {};
    ge::graphStatus DoTiling();

private:
    void CalcBasicInfo();
    void CalcRSplitInfo();
    void CalcCSplitInfo();
    ge::graphStatus CalcBlockAndUbSplitInfo();
    ge::graphStatus CalcCFullLoadRUbSplitInfo();
    ge::graphStatus CalcRFullLoadCUbSplitInfo();
    ge::graphStatus CalcRCNotFullLoadUbSplitInfo();
    void SetCFullLoadCUbSplitInfo();
    void SetRFullLoadCUbSplitInfo();
    void SetRBlockFactorUbFullLoadSplitInfo();
    void WriteTilingData();
    void FillBlockPara(CoreSplitPara& coreSplitPara, CoreSplitInfo& corePara);
    void FillUbPara(UbSplitPara& ubSplitPara, UbParamInfo& ubPara);
    void PrintTilingData();
    ge::graphStatus SetTilingKeyAndCore();

private:
    gert::TilingContext* context_ = nullptr;
    int64_t tilingKey_{0};
    int64_t AvailableUbSize{0};
    PlatInfo platInfo_;
    ShapeInfo shapeInfo_;
    BasicInfo basicInfo_{};
    UbParamInfo rUbParamInfo_{};
    UbParamInfo cUbParamInfo_{};
    CoreSplitInfo rCoreSplitInfo_{};
    CoreSplitInfo cCoreSplitInfo_{};
    TransposeVCONVTilingData tiling_;
};

} // namespace TransposeWithVCONV
} // namespace optiling
#endif
