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
 * \file pad_v3_grad_tiling_arch35.h
 * \brief ac pad v3 grad tiling h
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_V3_GRAD_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_struct.h"
#include "conversion/pad_v3_grad/op_kernel/arch35/pad_v3_grad_tilingkey.h"
#include <string>

namespace optiling {

struct padV3GradPaddingInfo {
    gert::Shape padFront;
    gert::Shape padBack;
};

struct PadV3GradCompileInfo {
    int64_t max_output_size;
    int64_t core_num;
    // how much input data can the UB store, instead of UB's size in Bytes
    int64_t ub_size;
    int64_t size;
    int64_t dtype_rate;
    bool paddings_contiguous;
    std::string mode;
    // chip's UB size in Bytes
    int64_t total_ub_size;
    int64_t output_dim4_max_cnt;
    // x data_size in Bytes
    int64_t x_bytes_size;
    // is support data_move_pad
    bool is_support_data_move_pad;
    std::string soc_version;
};
struct PadV3GradUbTileInfo {
    uint8_t ubSplitAxis = 0;
    uint32_t ubSplitFactor = 0;
    int64_t ubTotalCnt = 0;
    int64_t ubPerCoreCnt = 0;
    uint32_t usedCoreNum = 0;
};

class PadV3GradACTiling {
public:
    explicit PadV3GradACTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus DoTiling();

private:
    template <typename T>
    std::string ToString(const T* value, size_t size);
    uint64_t GetSizeOfBlockAlign(uint64_t inputSize, uint64_t alignBlockSize);

    ge::graphStatus Init();              // 获取coreNum,ubSize,vecSize,cacheLineSize,blockSize
    ge::graphStatus GetShapeAttrsInfo(); // 获取校验入参和shape信息

    ge::graphStatus CheckModeInputParam(int64_t inShapeV, int64_t padFront, int64_t padBack);
    ge::graphStatus DimensionCollapse();              // constant合轴
    ge::graphStatus DimensionCollapseMode();          // 其他模式合轴
    void EmptyTensorCollapse();                       // 空tensor时合为一根轴
    void EmptyTensorCollapseMode();                   // 其他模式空tensor时操作
    ge::graphStatus ComputeAfterPaddingsAndStrides(); // 计算合轴后的paddings和strides

    ge::graphStatus DoTilingModeEdge();
    ge::graphStatus DoTilingModeMirror();
    ge::graphStatus DoTilingModeCircular();
    ge::graphStatus DoTilingModeConstant();

    void DoTilingWithSIMTEdge();   // edge模板DoTilingWithSIMT
    void DoTilingWithSIMTMirror(); // reflect模板DoTilingWithSIMT
    void DoTilingWithSIMTCircular();
    void DoTilingWithSIMTConstant();
    void FillsAndPrintTilingData();

    void DoFindSplitAxisByInput(bool isBigLastDim);
    bool CheckTilingInfoSatisfied(PadV3GradUbTileInfo& tilingInfo);
    void GetOptimizeTiling(const PadV3GradUbTileInfo& oldTilingInfo, PadV3GradUbTileInfo& newTilingInfo);
    void TilingInfoTune();
    void TilingInfoTuneForNormal(uint64_t lastShapeSizeAlign);
    void CalculateTilingKeyMirror();
    void CalculateTilingKeyCircular();
    bool ClampBufferSizeAndCheckBigShape(uint64_t lastShapeSizeAlign);
    void DoTilingWithSIMDMirror();
    void DoTilingWithSIMDCircular();

    ge::graphStatus GetShapesAndDtypes();
    ge::graphStatus GetPaddings();
    template <typename T>
    void GetPaddingsToShape(const gert::Tensor* paddingsTensor);

private:
    gert::TilingContext* context_ = nullptr;
    uint32_t coreNum_{0};
    uint64_t ubSize_{0};
    uint64_t blockSize_{32};
    uint64_t vectorSize_{256};
    uint32_t cacheLineSize_{256};
    PadV3GradACTilingData* tilingData_{nullptr};

    // tiling key
    uint64_t tilingKey_{0};
    uint8_t padMode_{TPL_MODE_REFLECT};
    bool isBigShape_{false};
    bool isSimt_{true};
    uint8_t cutMode_{TPL_SIMD_BIG};

    uint8_t dimNum_{1};
    uint8_t ubAxis_{0};            // 切分维度
    uint32_t ubFactor_{0};         // 每次处理的切分轴的个数
    uint32_t ubPerCount_{0};       // 每个核处理的ubFactor_个数
    uint32_t ubTotalCount_{0};     // 一共所需处理的ubFactor_个数
    uint32_t outTileSize_{0};      // 对齐后处理的数据量
    uint32_t additionTileSize_{0}; // 额外的计算空间

    bool paddingContiguous_{true};
    bool isEmptyTensor_{false};
    uint16_t inputRank_{0};
    padV3GradPaddingInfo paddings_;
    ge::DataType paramsDtype_{ge::DT_FLOAT};
    uint32_t dtypeBytes_{0};
    uint64_t bufferSize_{4};

    uint64_t inShapeSize_{1};
    uint64_t outShapeSize_{1};
    uint64_t outShapeSizeLastTwoDim_{1};

    bool isPadAllPositive_{true};
    bool isPadAllNegative_{true};
    bool isUseSlice_{false};

public:
    bool isCircularPadGrad_{false};
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_PAD_TILING_H_
