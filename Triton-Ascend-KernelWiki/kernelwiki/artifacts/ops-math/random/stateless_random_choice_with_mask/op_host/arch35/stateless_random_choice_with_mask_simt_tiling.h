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
 * \file stateless_random_choice_with_mask_simt_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_STATELESS_RANDOM_CHOICE_WITH_MASK_SIMT_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_STATELESS_RANDOM_CHOICE_WITH_MASK_SIMT_TILING_H

#include <array>
#include <cstdint>
#include "platform/platform_info.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "util/platform_util.h"

namespace optiling {

constexpr int64_t INPUT_X_IDX = 0;
constexpr int64_t INPUT_X_MAX_DIM_NUM = 5;
constexpr int64_t INPUT_X_MIN_DIM_NUM = 1;
constexpr int64_t OUTPUT_Y_IDX = 0;
constexpr int64_t THREAD_NUM = 512;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t DCACHE_SIZE = 32768;
constexpr int64_t ALIGNMENT_32 = 32;
constexpr int64_t ALIGNMENT_256 = 256;
constexpr int64_t TWO = 2;
constexpr int64_t ROWS_NUM = 32;
const int32_t INPUT_SEED_IDX = 2;
const int32_t INPUT_OFFSET_IDX = 3;

struct StatelessRandomChoiceWithMaskCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

struct IndexTilingData {
    int64_t blockNum;
    int64_t normalCoreProNum;
    int64_t tailCOreProNum;
    int64_t m;
    int64_t n;
    int64_t seed;
    int64_t offset;
    int64_t outputSize;
};

class StatelessRandomChoiceWithMaskSimtTiling : public Ops::Base::TilingBaseClass {
public:
    explicit StatelessRandomChoiceWithMaskSimtTiling(gert::TilingContext* context)
        : Ops::Base::TilingBaseClass(context), opName_(context->GetNodeName())
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    ge::graphStatus ComputeCoreNum();
    ge::graphStatus SetTilingData();

private:
    const std::string opName_;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t inputSize_ = 0;
    int64_t blockNum_ = 0;
    int64_t tailThreadNum_ = 0;
    int64_t normalCoreProNum_ = 0;
    int64_t noZeroCalcCount_ = 0;
    int64_t noZeroWorkspaceSize_ = 0;
    int64_t randomWorkspaceSize_ = 0;
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t seed_ = 0;
    int64_t offset_ = 0;
    int32_t count_ = 0;
    uint32_t inputDim_ = 0;

    gert::Shape xShape_;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_STATELESS_RANDOM_CHOICE_WITH_MASK_SIMT_TILING_H