/**

Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
/*!

\file grouped_bias_add_grad_tiling_arch35.h
\brief tiling for grouped_bias_add_grad arch35
*/
#ifndef GROUPED_BIAS_ADD_GRAD_TILING_RA_ARCH35_H
#define GROUPED_BIAS_ADD_GRAD_TILING_RA_ARCH35_H

#include "op_host/tiling_base_class.h"
#include "math/grouped_bias_add_grad/op_kernel/arch35/grouped_bias_add_grad_struct.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "grouped_bias_add_grad_tiling_arch35.h"

namespace optiling {

class GroupedBiasAddGradTilingArch35 : public Ops::Base::TilingBaseClass {
public:
    explicit GroupedBiasAddGradTilingArch35(
        gert::TilingContext* context, const GroupedBiasAddGradCompileInfoArch35* compileInfo)
        : Ops::Base::TilingBaseClass(context)
    {
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
        cacheLineSize_ = compileInfo->clSize; // 从平台获取的cacheLineSize
        blockSize_ = compileInfo->blockSize;  // 32B
        coreNumThreshold_ = compileInfo->coreNum / 2;
    }

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    // Input/Output info
    ge::graphStatus GetInputOutputInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus CheckInputOutput();

    // Tiling computation
    ge::graphStatus DetermineMode();
    ge::graphStatus DoCutHTiling(); // 模版2: 核数>32, 切分H轴
    ge::graphStatus DoCutGTiling(); // 模版3: 核数<=32, 切分G*H块

    // Helper functions
    int64_t AlignUp(int64_t value, int64_t alignment) const;
    int64_t AlignDown(int64_t value, int64_t alignment) const;
    int64_t GetComputeTypeSize() const; // bf16/fp16按float分配空间
    void PrintTilingInfo() const;

    // 分核优化：3331 vs 3322 场景选择
    void OptimizeCoreSplit(
        int64_t totalBlocks, int64_t coreNum, int64_t& blockFactor, int64_t& blockTailFactor, int64_t& normalCoreNum,
        int64_t& tailStartIndex) const;

private:
    // Input shape info
    int64_t gradYDimNum_ = 0;
    int64_t dimG_ = 0;  // group count (from group_idx shape)
    int64_t dimGB_ = 0; // grad_y dim0 (batch * group)
    int64_t dimH_ = 0;  // H dimension

    // Data types
    ge::DataType gradYDtype_ = ge::DT_FLOAT;
    ge::DataType groupIdxDtype_ = ge::DT_INT32;

    // Attributes
    int32_t groupIdxType_ = 0;

    // Platform info
    uint32_t coreNum_;
    uint64_t ubSize_;
    int64_t cacheLineSize_;
    int64_t blockSize_;

    int64_t coreNumThreshold_;

    // Tiling mode
    GroupedBiasAddGradTilingModeArch35 tilingMode_ = GroupedBiasAddGradTilingModeArch35::CUT_H_MODE;

    // Tiling data
    GroupedBiasAddGradCutHTilingData cutHTilingData_;
    GroupedBiasAddGradCutGTilingData cutGTilingData_;
    GroupedBiasAddGradEmptyTensorTilingData emptyTensorTilingData_;

    // Block info
    int64_t usedCoreNum_ = 0;
    int64_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // GROUPED_BIAS_ADD_GRAD_TILING_RA_ARCH35_H