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
 * \file lerp_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_LERP_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_LERP_H_

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"

namespace optiling {
constexpr int64_t LERP_MAX_DIM_SIZE = 8;

BEGIN_TILING_DATA_DEF(LerpTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubOuter);
TILING_DATA_FIELD_DEF(int64_t, ubTail);
TILING_DATA_FIELD_DEF(int64_t, blockTail);
TILING_DATA_FIELD_DEF(int64_t, shapeLen);
TILING_DATA_FIELD_DEF(int64_t, ubSplitAxis);
TILING_DATA_FIELD_DEF(int64_t, dimProductBeforeUbInner);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, input0Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, input1Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, input2Dims);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, outputDims);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, input0Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, input1Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, input2Strides);
TILING_DATA_FIELD_DEF_ARR(int64_t, LERP_MAX_DIM_SIZE, outputStrides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lerp, LerpTilingData);

struct LerpCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class LerpTiling : public Ops::Base::TilingBaseClass {
   public:
    explicit LerpTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {
    }

   protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    std::string ToString(LerpTilingData& inputTilingData);

   private:
    uint64_t GetOpKey(ge::DataType startDtype, ge::DataType endDtype, ge::DataType weightDtype, ge::DataType yDtype);
    uint64_t GenerateTilingKey(uint64_t innerKey);
    std::map<uint64_t, Ops::Base::BroadcastComputeParams> GetComputeMap(uint64_t inputOpKey);

    LerpTilingData tilingData;
    uint64_t opKey;
    int64_t coreNum;
    int64_t ubSize;
    uint64_t blockNum;
    int64_t input0Dims[LERP_MAX_DIM_SIZE] = {0};
    int64_t input1Dims[LERP_MAX_DIM_SIZE] = {0};
    int64_t input2Dims[LERP_MAX_DIM_SIZE] = {0};
    int64_t outputDims[LERP_MAX_DIM_SIZE] = {0};
    int64_t input0Strides[LERP_MAX_DIM_SIZE] = {0};
    int64_t input1Strides[LERP_MAX_DIM_SIZE] = {0};
    int64_t input2Strides[LERP_MAX_DIM_SIZE] = {0};
    int64_t outputStrides[LERP_MAX_DIM_SIZE] = {0};
};

}  // namespace optiling

#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_LERP_H_
