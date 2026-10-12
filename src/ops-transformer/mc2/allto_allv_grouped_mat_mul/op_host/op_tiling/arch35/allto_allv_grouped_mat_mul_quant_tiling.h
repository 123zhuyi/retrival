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
 * \file allto_allv_grouped_mat_mul_quant_tiling.h
 * \brief
 */
#ifndef ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H
#define ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H

#include "../allto_allv_grouped_mat_mul_tiling_base.h"

namespace optiling {
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t PERTENSOR_MODE = 1;
constexpr uint32_t SINGLE_GROUP_NUM = 1;
constexpr uint32_t GMM_ACT_TYPE_NONE = 0;
constexpr uint64_t DB_SIZE = 2UL;

class AlltoAllvGmmQuantTiling : public AlltoAllvGmmTilingBase {
public:
    explicit AlltoAllvGmmQuantTiling(gert::TilingContext *context) : AlltoAllvGmmTilingBase(context){
        tilingData = context->GetTilingData<QuantAlltoAllvGroupedMatmulTilingData>();
    };
    QuantAlltoAllvGroupedMatmulTilingData *tilingData;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckGmmDType() const;
    ge::graphStatus CheckMmDType() const;
    ge::graphStatus CheckQuantMode() const;
    ge::graphStatus SetHcclTiling() const;
    void SetGMMQuantParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData) const;
    void SetTilingArray(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const;
    void SetTilingParams(Mc2GroupedMatmulTilingData::GMMQuantTilingData &gmmQuantTilingData, uint64_t M, uint64_t N, uint64_t K) const;
    void PrintGMMQuantTilingData(const Mc2GroupedMatmulTilingData::GMMQuantTilingData &data) const;
    void PrintTaskTilingInfo(const MC2KernelTemplate::TaskTilingInfo& taskTilingInfo) const;
};
}  // namespace optiling
#endif  // ALLTO_ALLV_GROUPED_MATMUL_QUANT_TILING_H