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
 * \file fused_mul_add_n_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSED_MUL_ADD_N_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSED_MUL_ADD_N_H_

#include "atvoss/elewise/elewise_tiling.h"
#include "op_host/tiling_base_class.h"
#include "register/tilingdata_base.h"
#include "platform/platform_ascendc.h"

using namespace Ops::Base;

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedMulAddNTilingData)
TILING_DATA_FIELD_DEF(int64_t, dim0);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedMulAddN, FusedMulAddNTilingData);

struct FusedMulAddNCompileInfo {
    uint64_t coreNum{0};
    uint64_t ubSize{0};
};

class FusedMulAddNTiling : public Ops::Base::TilingBaseClass {
public:
    explicit FusedMulAddNTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
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
    std::string ToString(FusedMulAddNTilingData& tilingData);

private:
    void SetOpKey();
    uint64_t GetOpKey(
        ge::DataType inputX1Dtype, ge::DataType inputX2Dtype, ge::DataType inputX3Dtype, ge::DataType outputYDtype);
    uint64_t GenerateTilingKey(uint64_t innerKey);
    std::map<uint64_t, ComputeParams> GetComputeMap(uint64_t opKey);

    FusedMulAddNTilingData tilingData;
    uint64_t opKey{0};
    int64_t coreNum{0};
    int64_t ubSize{0};
    uint64_t blockNum{0};
    std::map<std::tuple<ge::DataType, ge::DataType, ge::DataType, ge::DataType>, uint64_t> opKeys;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSED_MUL_ADD_N_H_
