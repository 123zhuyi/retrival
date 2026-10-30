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
 * \file confusion_transpose_d_tiling_arch35.h
 * \brief
 */

#ifndef CONFUSION_TRANSPOSE_D_TILING_ARCH35_H_
#define CONFUSION_TRANSPOSE_D_TILING_ARCH35_H_
#include <cmath>
#include <vector>
#include "register/op_impl_registry.h"
#include "op_host/tiling_base_class.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "../../../transpose/op_host/arch35/transpose_tiling_arch35.h"

using namespace std;
namespace optiling {

BEGIN_TILING_DATA_DEF(ConfusionTransposeDTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TransposeOpTilingData, transposeOpTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ConfusionTransposeD, ConfusionTransposeDTilingData);

struct ConfusionTransposeDCompileInfo {
    TransposeCompilerInfo transposeCompilerInfo;
};

ge::graphStatus ConfusionTransposeDTilingForAscendC(gert::TilingContext* context,
                                                    const TransposeCompilerInfo* transposeCompileInfo);

struct ConfusionTransposeDParamInfo {
    gert::Shape xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    gert::Shape yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    gert::Shape perm;
    gert::Shape shape;
    bool transpose_first;
};
constexpr int64_t SPLIT_NUM = 16;
constexpr int64_t MIN_DIM_ND = 1;
constexpr int64_t MAX_DIM_ND = 8;
constexpr int64_t MIN_DIM_NZ = 4;
constexpr int64_t MAX_DIM_NZ = 8;
constexpr int64_t MIN_DIM_NZ2ND = 2;
constexpr int64_t MAX_DIM_NZ2ND = 6;

class ConfusionTransposeDTiling {
public:
    explicit ConfusionTransposeDTiling(gert::TilingContext* context) : tilingContext_(context){};

    ge::graphStatus ParametersVerifying();
    void ProcessShapeInfo(ShapeInfo& shapeInfo);

private:
    ge::graphStatus GetParameters();
    ge::graphStatus ParametersVerifyingFormatAndDatatype();
    ge::graphStatus ParametersVerifyingInputAndOutput();
    ge::graphStatus ParametersVerifyingDimNd();
    ge::graphStatus ParametersVerifyingPerm();
    ge::graphStatus ParametersVerifyingProdAndPositive();

    void ProcessShapeInfoForNd(ShapeInfo& shapeInfo);

    ConfusionTransposeDParamInfo paramInfo_;
    gert::TilingContext* tilingContext_ = nullptr;
};

} // namespace optiling

#endif // CONFUSION_TRANSPOSE_D_TILING_H_