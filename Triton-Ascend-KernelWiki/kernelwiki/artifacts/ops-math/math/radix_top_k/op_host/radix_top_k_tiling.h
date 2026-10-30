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
 * \file radix_top_k_tiling.h
 * \brief
 */
#pragma once

#include <string>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "../op_kernel/radix_top_k_struct.h"
#include "../op_kernel/radix_top_k_tiling_key.h"

namespace optiling {
namespace RadixTopK {

class RadixTopKTiling : public Ops::Base::TilingBaseClass
{
public:
    explicit RadixTopKTiling(gert::TilingContext* context)
        : Ops::Base::TilingBaseClass(context), opName_(context->GetNodeName())
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    bool CalcLargeTilingParams(const uint64_t &dataNum);
    bool CalcTilingParams(const uint64_t &dataNum);
    void PrintTilingData();

    template<typename CheckUbFn>
    bool TryCalcTileDistribution(const uint64_t &dataNum, uint64_t startTileLen,
        uint64_t minTileLen, uint64_t step, CheckUbFn &&checkUb);

private:
    const std::string opName_;
    uint64_t ubSize_{0};
    uint64_t totalCoreNum_{0};
    ge::DataType xDType_{ge::DT_BF16};
    uint32_t xDtypeSize_{0};
    uint64_t xAlign_{0};
    bool isLargeShape_{false};
    RadixTopKTilingData tilingData_;

    void SetTilingData();
    void CalcTileDistribution(const uint64_t &dataNum, const uint64_t &formerTileLen);
};

} // namespace RadixTopK

struct RadixTopKCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling