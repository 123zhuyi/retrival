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
 * \file transpose_tiling_with_gather_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_WITH_GATHER_ARCH35_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_WITH_GATHER_ARCH35_H_

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
namespace TransWithGather {
constexpr int8_t MAX_TRANS_AXIS_NUM = 8;
constexpr int8_t UB_MAX_DIM_NUM = 6;
constexpr int8_t UB_MAX_BRW_NUM = 3;
constexpr int32_t MTE_GATE = 0x8000;

BEGIN_TILING_DATA_DEF(GatherTransposeTilingData)
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, dataTensorSize);  // unit is B
TILING_DATA_FIELD_DEF(uint32_t, indexTensorSize); // unit is B
TILING_DATA_FIELD_DEF(uint32_t, usedCoreCnt);
TILING_DATA_FIELD_DEF(int8_t, blkAxesCnt);
TILING_DATA_FIELD_DEF(int8_t, blkInUbCutPos);
TILING_DATA_FIELD_DEF(int8_t, blkOutUbCutPos);
TILING_DATA_FIELD_DEF(int8_t, ubAxesCnt);
TILING_DATA_FIELD_DEF(int8_t, inUbInCutPos);
TILING_DATA_FIELD_DEF(int8_t, inUbOutCutPos);
TILING_DATA_FIELD_DEF(int8_t, outUbInCutPos);
TILING_DATA_FIELD_DEF(int8_t, outUbOutCutPos);

TILING_DATA_FIELD_DEF(int64_t, blkFactor);
TILING_DATA_FIELD_DEF(int64_t, blkTailFactor);

TILING_DATA_FIELD_DEF(int64_t, inUbCutAxisSize);
TILING_DATA_FIELD_DEF(int64_t, outUbCutAxisSize);
TILING_DATA_FIELD_DEF(int32_t, inUbCutAxisFactor);
TILING_DATA_FIELD_DEF(int32_t, outUbCutAxisFactor);
TILING_DATA_FIELD_DEF(int64_t, axis0InSrcStride);
TILING_DATA_FIELD_DEF(int64_t, axis1InSrcStride);
TILING_DATA_FIELD_DEF(int64_t, axis2InSrcStride);
TILING_DATA_FIELD_DEF(int64_t, axis0OutDstStride);
TILING_DATA_FIELD_DEF(int64_t, axis1OutDstStride);
TILING_DATA_FIELD_DEF(int64_t, axis2OutDstStride);

TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TRANS_AXIS_NUM, blkAxes);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TRANS_AXIS_NUM, blkAxesInAOffset);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TRANS_AXIS_NUM, blkAxesOutAOffset);
TILING_DATA_FIELD_DEF_ARR(int32_t, UB_MAX_DIM_NUM, inUbAxes);
TILING_DATA_FIELD_DEF_ARR(int32_t, UB_MAX_DIM_NUM, outUbAxes);
TILING_DATA_FIELD_DEF_ARR(int8_t, UB_MAX_DIM_NUM, ubPerm);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose_10006, GatherTransposeTilingData);

struct PlatInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t cacheLineSize = 0;
    int64_t ubBlockSize = 0;
};

struct UbSplitInfo {
    int64_t inUbCutAxisSize = 1;
    int64_t outUbCutAxisSize = 1;
    int64_t axis0InSrcStride = 0; // loop layer from outer to inner: 2, 1, 0
    int64_t axis1InSrcStride = 0;
    int64_t axis2InSrcStride = 0;
    int64_t axis0OutDstStride = 0;
    int64_t axis1OutDstStride = 0;
    int64_t axis2OutDstStride = 0;
    int32_t inUbCutAxisFactor = 1;
    int32_t outUbCutAxisFactor = 1;
    // for count gather index
    int32_t inUbAxes[UB_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1}; // maybe changed accorrding to MTE2
    int32_t outUbAxes[UB_MAX_DIM_NUM] = {1, 1, 1, 1, 1, 1};
    int8_t ubPerm[UB_MAX_DIM_NUM] = {0xf, 0xf, 0xf, 0xf, 0xf, 0xf}; // // maybe changed accorrding to MTE2
    int8_t ubAxesCnt = 0;
    // for update inUbAxes and outUbAxes
    int8_t inUbInCutPos = 0;  // include right offset to be move in burst length
    int8_t inUbOutCutPos = 0; // maybe changed accorrding to MTE2
    int8_t outUbInCutPos = 0;
    int8_t outUbOutCutPos = 0; // include right offset to be move out burst length
};

struct BlockSplitInfo {
    int64_t blkAxes[MAX_TRANS_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t blkAxesInAOffset[MAX_TRANS_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t blkAxesOutAOffset[MAX_TRANS_AXIS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t blkFactor = 1;
    int64_t blkTailFactor = 0;
    uint32_t usedCoreCnt = 0;
    int8_t blkAxesCnt = 0;
    // to count what time to update input and output ub cut axes
    int8_t blkInUbCutPos = -1;
    int8_t blkOutUbCutPos = -1;
};

struct UbPermInfo {
    int8_t cnt = 0;
    int8_t perm[UB_MAX_BRW_NUM] = {0xf, 0xf, 0xf};
};

class TransposeGatherTiling {
public:
    explicit TransposeGatherTiling(gert::TilingContext* context, const PlatInfo& platInfo, const ShapeInfo& shapeInfo)
        : context_(context), platInfo_(platInfo), shapeInfo_(shapeInfo) {};
    ge::graphStatus DoTiling();

private:
    int64_t CalcShapeSize(const std::vector<int64_t>& shape, int64_t beg, int64_t end);
    int64_t CalcSqrtedTensor(int64_t elemInTensor);
    void CalcTensorSize();
    void CalcInUbPerm(int64_t sqrtedTensor);
    void CalcOutUbPerm(int64_t sqrtedTensor);
    void CalcUbAxisCutFactor(int64_t elemInTensor, int64_t sqrtedTensor, bool isLastInPermLeft, bool isLastOutPermLeft,
                             const std::set<int8_t>& viceAllUbPerm);
    ge::graphStatus CalcUbAxesInfo(const int64_t (&tmpInAxes)[MAX_TRANS_AXIS_NUM],
                                   const int64_t (&tmpOutAxes)[MAX_TRANS_AXIS_NUM],
                                   const int8_t (&tmpOutPerm)[MAX_TRANS_AXIS_NUM]);
    ge::graphStatus CalcUbSplitInfo4Gather(int64_t elemInTensor, int64_t sqrtedTensor);
    void CalcUbSplitInfo4MTE();
    void AdjustInUbAxesPosition();
    void AdjustUbCutAxisFactor(int32_t& axisFactor, int8_t axisFlag, int64_t elemInTensor);
    ge::graphStatus CalcUbSplitInfo();
    ge::graphStatus CalcBlockSplitInfo();
    void WriteTilingData();
    std::string PrintTilingData();
    ge::graphStatus SetTilingKeyAndCore();
    bool CheckBC(int32_t steps);

private:
    gert::TilingContext* context_ = nullptr;

    uint64_t tilingKey_ = static_cast<uint64_t>(SplitMode::GATHER_TRANSPOSE);
    uint32_t dataTensorSize_ = 0;
    uint32_t indexTensorSize_ = 0;
    UbSplitInfo ubSplitInfo_;
    BlockSplitInfo blkSplitInfo_;
    GatherTransposeTilingData tilingData_;

    PlatInfo platInfo_;
    ShapeInfo shapeInfo_;
    UbPermInfo inUbPerm_;
    UbPermInfo outUbPerm_;
    std::set<int8_t> inUbPermSet_;
    std::set<int8_t> allUbPerm_;
};
} // namespace TransWithGather
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSPOSE_TILING_WITH_GATHER_ARCH35_H_
