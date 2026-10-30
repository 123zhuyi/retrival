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
 * \file grouped_bias_add_grad_tiling.h
 * \brief
 */

#ifndef GROUPED_BIAS_ADD_GRAD_TILING_H
#define GROUPED_BIAS_ADD_GRAD_TILING_H

#include "grouped_bias_add_grad_tiling_def.h"
#include "op_host/tiling_base_class.h"

namespace optiling {
namespace groupedbiasaddgrad {
// 自己定义需要的全局变量
struct BaseInfoParams {
    // attr
    int32_t groupIdxType{0};

    // platformInfo
    uint32_t vectorCoreNum{0};
    uint64_t ubSize{0};

    // shapeInfo
    int64_t gradYDimNum{0};
    int64_t dimG{0};
    int64_t dimC{0};
    int64_t dimH{0};
    int64_t dimGB{0};

    // dtype
    ge::DataType gradYInputDtype{ge::DT_FLOAT};
    ge::DataType groupIdxInputDtype{ge::DT_INT32};

    // optional
    uint32_t existGroupIdx{0};
    uint32_t performance{0};
};

struct SplitCoreParams {
    int64_t usedCoreNum{0};
    int64_t normalCoreNum{0};
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};

    int64_t baseH{0};
    int64_t baseC{0};

    int64_t loopCNum{0};
    int64_t wsUnitNum{0};
};

class GroupedBiasAddGradTiling
{
public:
    explicit GroupedBiasAddGradTiling(gert::TilingContext* context)
        : context_(context), nodeName_(context->GetNodeName())
    {}
    ~GroupedBiasAddGradTiling()
    {}
    GroupedBiasAddGradTilingData tilingData_;
    ge::graphStatus RunGroupedBiasAddGradTiling();
    ge::graphStatus GetInputInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus CheckOutput();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoTiling();
    ge::graphStatus DoSplitTiling();
    ge::graphStatus DoUnequalCPerformanceTiling();
    ge::graphStatus PostTiling();
    void DumpTilingInfo() const;
    uint64_t ComputeTiling(const std::vector<uint32_t>& args) const;
    uint64_t GetTilingKey() const;
    void SaveToTilingData();

protected:
    gert::TilingContext* context_ = nullptr;
    const ge::char_t* nodeName_;

private:
    BaseInfoParams baseInfoOp_;
    SplitCoreParams splitCoreOp_;
};
} // namespace groupedbiasaddgrad

struct GroupedBiasAddGradCompileInfo {
    uint64_t ubSize{0};
    int32_t coreNum{0};
};
} // namespace optiling
#endif // GROUPED_BIAS_ADD_GRAD_TILING_H