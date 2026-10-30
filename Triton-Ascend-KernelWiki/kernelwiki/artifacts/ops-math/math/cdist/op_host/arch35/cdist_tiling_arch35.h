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
#include "../../op_kernel/cdist_tiling_key.h"

namespace optiling {
constexpr int64_t WORK_SPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024);
constexpr int64_t MIN_DIM_LEN = 2;
constexpr int64_t M_SIZE = 256;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
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

private:
    gert::TilingContext* tilingContext_ = nullptr;
    CdistTilingData tilingData_;
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