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
 * \file cross_tiling.h
 * \brief cross tiling header
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_H_

#include <cstdint>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "../../op_kernel/arch35/cross_struct.h"

namespace optiling {

struct CrossCompileInfo {
    int64_t coreNum = 0;
};

class CrossTiling : public Ops::Base::TilingBaseClass {
public:
    explicit CrossTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override { return true; }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    uint64_t GetTilingKey() const override { return 0; }
    ge::graphStatus GetWorkspaceSize() override { return ge::GRAPH_SUCCESS; }
    ge::graphStatus PostTiling() override;
    ge::graphStatus CheckBaseShapeAndAttrs();
    ge::graphStatus CheckBroadcastAndMergeShape();
    ge::graphStatus CalcStrideAndVectors();

private:
    const CrossCompileInfo* compileInfo_;
    CrossRegbaseTilingData tilingData_{};
    uint32_t blockDim_{1};
    int64_t totalVectors_{0};
    int64_t dimNum_{0};
    int64_t dim_{0};
    int64_t dimNum1_{0};
    int64_t dimNum2_{0};
    int64_t dimStride_{1};
    int64_t normalizedDim_{0};
    int64_t ySize_{0};
    int64_t x1Dims_[MAX_DIM] = {0};
    int64_t x2Dims_[MAX_DIM] = {0};
    int64_t x1Stride_[MAX_DIM] = {0};
    int64_t x2Stride_[MAX_DIM] = {0};
    int64_t mergedStride_[MAX_DIM] = {1};
    int64_t mergedShape_[MAX_DIM] = {1};
    int64_t yStride_[MAX_DIM] = {0};
};

} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CROSS_H_
