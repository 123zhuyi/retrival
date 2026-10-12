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
 * \file cdist_tiling.h
 * \brief
 */

#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "../../op_kernel/cdist_tiling_data.h"
#include "../../op_kernel/cdist_brc_tilingdata.h" // 方案2: POD CdistBrcTilingData
#include "../../op_kernel/cdist_tiling_key.h"

namespace optiling {
constexpr int64_t WORK_SPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024);
constexpr int64_t MIN_DIM_LEN = 2;
constexpr int64_t M_SIZE = 256;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
// 方案三 (round2): UB bytes reserved (subtracted from ubSize before the Normal UB-split solves) to hold
// the two fixed HL fp32 compute planes (2 * 8192 * 4 = 64KB) + partBuf scratch (1KB) + fp32 M-split
// accumulator margin (~16KB).
constexpr int64_t HL_UB_RESERVE = 96 * 1024;
// 方案三 (round2): fp32 element capacity of each HL compute plane (x1exp / diff), matching kernel HL_PLANE_ELEMS.
// Capping ubFactorR so (ubFactorR * MAlign) <= HL_PLANE_ELEMS guarantees the kernel never R-chunks internally.
constexpr int64_t HL_PLANE_ELEMS = 8192;
constexpr int64_t B4 = 4;
constexpr int64_t CAST_BUFFER_RATIO = 2;
constexpr int64_t SIMT_MIN_BYTE = 128;

struct CdistCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

struct DimConfig {
    int64_t* loopNum;
    int64_t* factor;
    int64_t* tailFactor;
    int64_t baseValue;
    std::function<int64_t(int64_t)> calcTotalElements;
};

class CdistTiling {
public:
    explicit CdistTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunCdistTiling();
    ge::graphStatus CheckParams();
    ge::graphStatus MergeBatchAxis();
    void DoTiling();
    void DoSimtTiling();
    void DoNormalTiling();
    void DoNormalBlockTiling();
    void SetDefaultBlockTiling();
    void DoNormalUbTiling();
    void SetDefaultUbTiling();
    void ProcessDimension(const DimConfig& config, int64_t availableUbElements, int64_t& findUbTilingIdx);
    ge::graphStatus SetTilingData();
    void PrintTilingData();

    // 方案2: M==1 Broadcast fast path selection + tiling (逐行照抄直调 cdist_host_tiling.cpp)。
    void DoBrcTiling();
    int64_t GetBrcBlockDim() const;
    void PrintBrcTilingData() const;

    // 方案三 (round2): M∈[2,256] 矢量胜 → broadcast+高层reduce 内核（Normal 形态 tiling），否则回退 SIMT。
    void CapUbFactorRForHL(); // cap ubFactorR so the HL kernel never R-chunks internally

private:
    gert::TilingContext* tilingContext_ = nullptr;
    CdistTilingData tilingData_;
    CdistBrcTilingData brcTilingData_; // 方案2: M==1 快路径 tiling
    int64_t use_broadcast_ = 0;        // 方案2: 命中 M==1 广播快路径标记
    int64_t use_reduce_hl_ = 0; // 方案三 (round2): 命中 M∈[2,256] 矢量胜 broadcast+高层reduce 路径标记
    gert::Shape x1Shape_;
    gert::Shape x2Shape_;
    gert::Shape yShape_;
    int64_t dtypeSize_ = 0;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t is_small_m_ = 0;
    int64_t notFoundUbTilingAxis_ = 1;
    int64_t B_ = 0;
    int64_t M_ = 0;
    int64_t P_ = 0;
    int64_t R_ = 0;
};
} // namespace optiling
