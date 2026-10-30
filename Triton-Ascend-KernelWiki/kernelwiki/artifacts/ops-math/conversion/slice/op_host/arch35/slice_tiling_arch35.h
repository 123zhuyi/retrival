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
 * \file slice_tiling_base.h
 * \brief
 */
#ifndef CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_SLICE_TILING_H_
#define CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_SLICE_TILING_H_

#include "conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.h"
#include "conversion/slice/op_kernel/arch35/slice_struct.h"

namespace optiling {

constexpr int64_t SLICE_KEY_MOVE_UNALIGN_GATHER = 301;
constexpr int64_t SLICE_KEY_TWO_DIM_SMALL_SHAPE = 400;
constexpr int64_t MIN_OUTPUT_SIZE = int64_t(512 * 1024);
constexpr size_t INPUT_X_INDEX = 0;
constexpr int64_t SLICE_CONST2 = 2;

struct SliceCompileParam {
    int64_t block_dim{1};
    int64_t ub_size{0};
    uint32_t cacheLineSize{0};
    bool isAscendc{false};
};

struct SliceParasRuntime2 {
    gert::Shape input;
    gert::Shape output_shape;
    gert::Shape begin_list;
    gert::Shape end_list;
    gert::Shape stride_list;
    int64_t tiling_mode = 0;
    int64_t core_num = 0;
    bool is_begin_const = true;
};

class SliceTiling : public StrideSliceTiling {
public:
    explicit SliceTiling(gert::TilingContext* context) : StrideSliceTiling(context) {};

protected:
    void CalMaxSplitDim() override;
    void SetTilingMode() override;
    void FillTilingData() override;
    void PrintTilingData() override;
    void SetBlockDimAndTilingKey() override;
    void FillSliceBaseTilingData(SliceBaseTilingData& tilingData);
    void FillSliceTilingData150();
    void FillSliceTilingData101();
    void FillSliceTilingData102();
    void FillSliceTilingData103();
    void FillSliceTilingData100();
    void FillSliceTilingData300();
    void FillSliceTilingData400();
    void FillSliceTilingDataOther();
    void PrintSliceBaseTilingData(SliceBaseTilingData& tilingData);
    void PrintSliceTilingData150();
    void PrintSliceTilingData101();
    void PrintSliceTilingData102();
    void PrintSliceTilingData103();
    void PrintSliceTilingData100();
    void PrintSliceTilingData300();
    void PrintSliceTilingData400();
    void PrintSliceTilingDataOther();
    void SetRowsStepsParamsFor150(SliceMoveAlignLast2DimTilingData& tilingData);
    void SetShortMoveAlignParams(SliceMoveAlignParams& params, const MoveAlignV2Info& actInfo);
    void CalSliceRowsStepsParams();
    void SetSliceGatherTilingMode();
    void SetTwoDimSmallShapeTilingMode();
    void SliceGatherUbSplitLastTwoDim();
    void SliceGatherUbSplitLastThreeDim();
    void SliceGatherUbSplitLastFourDim();
    void SetMoveAlignParamsSlice(StridedSliceMoveAlignParams2& params, const MoveAlignV2Info& actInfo);
    void SetRowsStepsParamsSlice(StridedSliceTilingData2& tilingData);

private:
    SliceTilingData sliceTilingData_;
    SliceMoveAlignLast2DimTilingData sliceMoveAlignLast2DimTilingData_;
    SliceMoveAlignLastDimTilingData sliceMoveAlignLastDimTilingData_;
    SliceMoveAlignTilingData sliceMoveAlignTilingData_;
    SliceNDDMATilingData sliceNDDMATilingData_;
    SliceNDDMALastDimTilingData sliceNDDMALastDimTilingData_;
    SliceMoveAlignGatherTilingData sliceMoveAlignGatherTilingData_;
    SliceTwoDimSmallSapeTilingData sliceTwoDimSmallSapeTilingData_;

    int64_t inLoopSteps_ = 1;
    int64_t outLoopSteps_ = 1;
    bool isUnalignCopy_ = false;
};
ge::graphStatus SliceTilingForAscendC(
    gert::TilingContext* context, int64_t coreNum, int64_t ubSize, int64_t cacheLineSize, SliceParasRuntime2& param,
    const ge::DataType dtype);
} // namespace optiling
#endif // CANN_OPS_BUILT_IN_OP_TILING_RUNTIME_SLICE_TILING_H_