/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file triangulator_tiling.h
 * \brief
 */

#ifndef TRIANGULATOR_TILING_H_
#define TRIANGULATOR_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TriangulatorTilingData)
TILING_DATA_FIELD_DEF(int64_t, bufferSize);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TriangulatorNormalTilingData)
TILING_DATA_FIELD_DEF(int64_t, bufferSize);
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, diagOffset);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, rowInner);
TILING_DATA_FIELD_DEF(int64_t, colInner);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TriangulatorTinyTilingData)
TILING_DATA_FIELD_DEF(int64_t, bufferSize);
TILING_DATA_FIELD_DEF(int64_t, planeArea);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, highInner);
TILING_DATA_FIELD_DEF(int64_t, highOuter);
TILING_DATA_FIELD_DEF(int64_t, highTail);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 4, highMask);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TriangulatorMediumTilingData)
TILING_DATA_FIELD_DEF(int64_t, bufferSize);
TILING_DATA_FIELD_DEF(int64_t, row);
TILING_DATA_FIELD_DEF(int64_t, col);
TILING_DATA_FIELD_DEF(int64_t, copyCols);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);
TILING_DATA_FIELD_DEF(int64_t, highInner);
TILING_DATA_FIELD_DEF(int64_t, highOuter);
TILING_DATA_FIELD_DEF(int64_t, highTail);
TILING_DATA_FIELD_DEF(int64_t, headRows);
TILING_DATA_FIELD_DEF(int64_t, midRows);
TILING_DATA_FIELD_DEF(int64_t, tailRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Tril_21, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Triu_21, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Tril_22, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Triu_22, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Tril_24, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Triu_24, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Tril_28, TriangulatorNormalTilingData)
REGISTER_TILING_DATA_CLASS(Triu_28, TriangulatorNormalTilingData)

REGISTER_TILING_DATA_CLASS(Tril_31, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Triu_31, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Tril_32, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Triu_32, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Tril_34, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Triu_34, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Tril_38, TriangulatorTinyTilingData)
REGISTER_TILING_DATA_CLASS(Triu_38, TriangulatorTinyTilingData)

REGISTER_TILING_DATA_CLASS(Tril_41, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Triu_41, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Tril_42, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Triu_42, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Tril_44, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Triu_44, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Tril_48, TriangulatorMediumTilingData)
REGISTER_TILING_DATA_CLASS(Triu_48, TriangulatorMediumTilingData)

REGISTER_TILING_DATA_CLASS(Tril, TriangulatorTilingData)
REGISTER_TILING_DATA_CLASS(Triu, TriangulatorTilingData)

struct TriangulatorBaseInfoParams {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    int64_t bufferSize{0};
    int64_t bufferEleNum{0};

    int64_t row{0};
    int64_t col{0};
    int64_t diagOffset{0};
    int64_t fusedFrontAxis{0};
    int64_t dtypeBytes{0};
    int64_t vRegSize{0};
};

struct TriangulatorSplitCoreParams {
    int64_t rowInner{0};
    int64_t rowOuter{0};
    int64_t rowTail{0};

    int64_t colInner{0};
    int64_t colOuter{0};
    int64_t colTail{0};

    int64_t highInner{0};
    int64_t highOuter{0};
    int64_t highTail{0};
    uint64_t highMask[4] = {0, 0, 0, 0};
    int64_t baseBlockNum{0};

    int64_t headRows{0};
    int64_t midRows{0};
    int64_t tailRows{0};
    int64_t copyCols{0};

    int64_t normalCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t tilingKey{0};
    int64_t tilingMode{0};
};

struct TriluCompileInfo {
    int32_t availableAICoreNum{0};
    int32_t availableUBSize{0};
};

struct TriluTilingParams {
    int64_t tilingMode{0};
    int64_t matrixNum{0};
    int64_t row{0};
    int64_t col{0};
    int64_t diagonal{0};
    int64_t taskNum{0};
    int64_t eltNumPerCore{0};
    int64_t mask{0};
    int64_t usedCoreNum{0};
};

class TriangulatorTiling {
public:
    explicit TriangulatorTiling(gert::TilingContext* context, bool operaterType)
        : context_(context), operatorType_(operaterType), nodeName_(context->GetNodeName())
    {}
    ~TriangulatorTiling()
    {}
    ge::graphStatus RunTriangulatorTilingAscendC(
        const TriluCompileInfo* compileInfo, const TriluTilingParams* params, uint8_t dtypeBytes);
    ge::graphStatus GetInputInfo(const TriluTilingParams* params, uint8_t dtypeBytes);
    ge::graphStatus GetPlatformInfo(const TriluCompileInfo* compileInfo);
    ge::graphStatus DoTiling();
    void ComputeTilingKey();
    void ComputeTilingMode();
    void ComputeTinyInfo();
    void ComputeMediumInfo();
    void ComputeNormalInfo();
    void ComputeMask();
    ge::graphStatus PostTiling();
    void DumpTilingInfo() const;
    void SaveToTilingData();

protected:
    gert::TilingContext* context_ = nullptr;
    bool operatorType_; // false : tril  true : triu
    const ge::char_t* nodeName_;

private:
    TriangulatorBaseInfoParams baseInfoOp;
    TriangulatorSplitCoreParams splitCoreOp;
    TriangulatorTilingData tilingData;
    TriangulatorNormalTilingData normalTilingData;
    TriangulatorTinyTilingData tinyTilingData;
    TriangulatorMediumTilingData mediumTilingData;
};
} // namespace optiling
#endif // TRIANGULATOR_TILING_H_