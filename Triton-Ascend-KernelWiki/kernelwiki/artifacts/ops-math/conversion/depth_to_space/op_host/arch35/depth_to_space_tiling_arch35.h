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
 * \file depth_to_space_tiling_arch35.h
 * \brief tiling for DepthToSpace
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DEPTH_TO_SPACE_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DEPTH_TO_SPACE_H

#include "conversion/transpose/op_host/arch35/transpose_tiling_arch35.h"

namespace optiling {

namespace DepthToSpace {

BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TransposeOpTilingData, transposeOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData);

struct DepthToSpaceCompileInfo {
    TransposeCompilerInfo transposeCompilerInfo;
};

ge::graphStatus DepthToSpaceTilingForAscendC(gert::TilingContext* context,
                                             const TransposeCompilerInfo* transposeCompileInfo);

class DepthToSpaceTiling {
public:
    explicit DepthToSpaceTiling(gert::TilingContext* context) : tilingContext_(context) {};
    ge::graphStatus ParametersVerifying();
    void ProcessShapeInfo(ShapeInfo& shapeInfo);

private:
    ge::graphStatus CheckFormatAndShape(int64_t xDimNum, int64_t yDimNum, ge::Format xFormat, ge::Format yFormat);
    ge::graphStatus CheckAttrValues(ge::Format xFormat);
    void SetPermAndOutShape(ShapeInfo& shapeInfo, const int64_t* perm);
    gert::TilingContext* tilingContext_ = nullptr;
    ParamInfo paramInfo_{};
    int64_t nhwcDcrPerm_[DIM_SIX] = {0, 1, 3, 2, 4, 5};
    int64_t nchwDcrPerm_[DIM_SIX] = {0, 3, 4, 1, 5, 2};
    int64_t crdPerm_[DIM_SIX] = {0, 1, 4, 2, 5, 3};
};
} // namespace DepthToSpace
} // namespace optiling
#endif
