/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_grouped_mat_mul_tiling.h
 * \brief
 */
#ifndef MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H
#define MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H

#include "allto_allv_grouped_mat_mul_tiling_base.h"

namespace optiling {
constexpr int64_t BEST_BASE_N = 256;
constexpr int32_t MAX_BASE_K = 128;

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15; // 15: 16 bytes up-align
    }
    return a & ~15; // ~15: 16 bytes down-align
}

static inline uint32_t Ceil(uint32_t a, uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

struct MMTilingParams {
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t *curBaseM;
    int32_t *curBaseK;
    int32_t *curBaseN;
};

struct SetMMTilingParams {
    matmul_tiling::DataType matmulDtype;
    int32_t curMaxM;
    int32_t curMaxK;
    int32_t curMaxN;
    int32_t curBaseM;
    int32_t curBaseN;
    int32_t type;
};

class AlltoAllvGmmNoQuantTiling : public AlltoAllvGmmTilingBase {
public:
    explicit AlltoAllvGmmNoQuantTiling(gert::TilingContext *context) : AlltoAllvGmmTilingBase(context){
        tilingData = context->GetTilingData<AlltoAllvGmmTilingData>();
    };

    AlltoAllvGmmTilingData *tilingData;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckDType() const;
    ge::graphStatus SetHcclTiling() const;
    ge::graphStatus CalMMTiling(MMTilingParams &params) const;
    ge::graphStatus SetMMTiling(SetMMTilingParams &params) const;
    void PrintMatmulTilingData(::TCubeTiling msg, const std::string &tilingType);
    void PrintCommonTilingInfo(AlltoAllvGmmCommonTilingInfo &commonTilingInfo) const;

    int32_t expertBaseM_;
    int32_t expertBaseN_;
    int32_t expertBaseK_;
    int32_t sharedExpertBaseM_;
    int32_t sharedExpertBaseN_;
    int32_t sharedExpertBaseK_;
};
} // namespace optiling
#endif // MC2_ALLTO_ALLV_GROUPED_MATMUL_TILING_STRUCT_H