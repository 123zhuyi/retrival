/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATRIX_SET_DIAG_TILING_ARCH35_BASE_H_
#define MATRIX_SET_DIAG_TILING_ARCH35_BASE_H_

#include "exe_graph/runtime/tiling_context.h"
#include "conversion/matrix_set_diag_v2/op_kernel/arch35/matrix_set_diag_v2_tilingdata.h"

namespace optiling {
class MatrixSetDiagTilingBase {
private:
    // tiling context
    gert::TilingContext* context_;

    // 输入参数
    MatrixSetDiagInputInfo inputInfo_;
    int32_t dSizeExpand_{1};

    /* data */
    // soc info
    uint64_t ubSize_{0};
    uint64_t ubBlockSize_{0};
    uint64_t bufferSize_{0};
    uint64_t vectorSize_{0};
    uint32_t coreNum_{0};
    uint32_t realCoreNum_{0};
    int32_t ubBlockElements_{0};
    int32_t cacheLineElements_{0};

    uint64_t ubFactor_{0};
    uint64_t ubPerCount_{0};
    uint64_t ubTotalCount_{0};
    uint64_t ubPerTail_{0};
    uint64_t tailAxisDataSize_{0};
    uint64_t diagDataSize_{0};

    // tiling key param
    bool isVLFullLoad_{false};
    uint8_t way_{0};
    bool isCutTail_{false};
    bool isBigShape_{false};

public:
    explicit MatrixSetDiagTilingBase(gert::TilingContext* context, MatrixSetDiagInputInfo inputInfo)
        : context_(context), inputInfo_(inputInfo) {};
    ~MatrixSetDiagTilingBase() {};

    ge::graphStatus DoTiling();

private:
    // 数据获取
    ge::graphStatus GetSocInfo();
    void CalcInputInfo();

    // tiling 计算
    ge::graphStatus DoOpTiling();
    ge::graphStatus Tiling4CutTail();
    ge::graphStatus Tiling4NoCutTail();
    void CalculateCutTailTilingParams(uint64_t ubMaxInputxSize, MSDV2CutTailTilingData* tilingData);
    void GetOptimizeTilingNoCutTail();

    // Tiling4NoCutTail 辅助方法
    uint64_t DetermineWayAndGetAdditionTileSize();
    ge::graphStatus CalculateValidBufSize(uint64_t additionTileSize, uint64_t& validBufSize);
    void CalculateUbFactorAndCheck(uint64_t validBufSize);
    ge::graphStatus FillNoCutTailTilingData();

    // v1 版本 tiling 计算
    ge::graphStatus Tiling4CutW();
    void CalUbFactor();
    void GetOptimizeTiling();
    uint64_t CalSizeTaken(uint64_t factor);

    // 辅助函数
    template <typename T>
    inline T AlignBlock(T elementCount);

    // 打印
    void ShowCutTailTilingData();
    void ShowNoCutTailTilingData();
    void ShowTilingData();

    void FillsTilingData(MatrixSetDiagV2TilingData& tilingData);
    void FillsTilingDataV1(MatrixSetDiagTilingData& tilingData);
};
} // namespace optiling

#endif
