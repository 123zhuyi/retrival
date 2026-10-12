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
 * \file grouped_matmul_swiglu_quant_v2_basic_tiling.h
 * \brief
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_TILING_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_BASIC_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h"
#include "../../grouped_matmul_swiglu_quant_v2_host_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "../grouped_matmul_swiglu_quant_v2_tiling.h"

namespace optiling {
using namespace Ops::Transformer::OpTiling;
class GroupedMatmulSwigluQuantV2Tiling950 : public GroupedQbmmTiling {
public:
    explicit GroupedMatmulSwigluQuantV2Tiling950(gert::TilingContext *context) : GroupedQbmmTiling(context)
    {
        Reset();
    }
    ~GroupedMatmulSwigluQuantV2Tiling950() override = default;

    void Reset(gert::TilingContext *context) override
    {
        GroupedQbmmTiling::Reset(context);
        Reset();
    }

protected:
    // 1、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 2、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 3、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 4、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset() override;
    void SetKernelType();
    ge::graphStatus GetWorkspaceSize() override;

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    void PrintQuantParams() override;
    bool SetQuantModeForGMMSwigluQuant(const gert::Shape &wScaleShape, const gert::Shape &xScaleShape);
    bool CheckShapeForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape);
    bool CheckDtype();
    bool CheckDims() const;
    bool IsFp4(ge::DataType dtype) const;
    bool IsFp8(ge::DataType dtype) const;
    bool IsFp4Input() const;
    bool IsFp8Input();
    // add for pertoken quant mode
    bool AnalyzeAttrsPertoken();
    bool IsB8(ge::DataType dtype);
    bool CheckDtypePertoken();
    bool AnalyzeInputsPertoken();
    ge::graphStatus DoOpTilingPertoken();
    void PrintPertokenQuantParams();
    bool CheckCoreNum() const override;
    GMMSwigluQuantTilingDataParams tilingData_;

    const std::vector<ge::DataType> quantDtypeSupportList = {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E2M1};
};
} // namespace optiling

#endif