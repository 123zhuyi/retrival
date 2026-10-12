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
 * \file stateless_randperm_tiling_arch35.h
 * \brief
 */
#ifndef STATELESS_RANDPERM_TILING_ARCH35_H
#define STATELESS_RANDPERM_TILING_ARCH35_H
#include <string>
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "../../op_kernel/stateless_randperm_key.h"
#include "../../op_kernel/stateless_randperm_struct.h"
#include "../../../../math/sort/op_kernel/arch35/sort_tiling_data.h"

namespace optiling{
static constexpr uint16_t INPUT_IDX_N = 0;
static constexpr uint16_t INPUT_IDX_SEED = 1;
static constexpr uint16_t INPUT_IDX_OFFSET = 2;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr uint16_t ATTR_IDX_LAYOUT = 0;
static constexpr uint16_t ATTR_IDX_DTYPE = 1;

struct StatelessRandpermCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

class StatelessRandpermTiling : public Ops::Base::TilingBaseClass {
public:
    explicit StatelessRandpermTiling(gert::TilingContext* context)
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

    void DumpTilingInfo() override;
private:
    StatelessRandpermTilingData* tilingData_;
    SortRegBaseTilingDataForRandperm* sortTilingData_;
    uint64_t tilingKey_;
    uint64_t tilingKeyForSort_;
    size_t workSpaceSize_;
    size_t workSpaceSizeForSort_;
    size_t workSpaceSizeForRandom_;
    const std::string opName_;
    int64_t ubSize_{0};
    int64_t totalCoreNum_{0};
    uint32_t needCoreNumForSort_{0};
    int64_t n_{0};
    uint64_t nIsInt32_{0};
    uint32_t key_[PHILOX_KEY_SIZE]{0};
    uint64_t offset_{0};
    uint32_t islandFactor_;
    uint32_t islandFactorTail_;
    uint32_t castFactor_;
    uint32_t castFactorTail_;
    int32_t randomBits_;
    uint64_t randomIsInt32_;
    std::vector<int32_t> subNs_;
    uint32_t subNSize_;
    ge::DataType attrOutDtype_;
    ge::DataType randomDtype_;
    int32_t randomType_;

    ge::graphStatus GetInputN();
    ge::graphStatus GetInputSeed();
    ge::graphStatus GetInputOffset();
    ge::graphStatus GetOutputY();
    ge::graphStatus GetAttrs();
    ge::graphStatus SortTilingBridge();
    void PhiloxRandomComputeBits();
    bool canUse32bitIndexing(int64_t len);
    void Int32IndexingSplit(int64_t len, std::vector<int32_t>& subBlocks, uint32_t& splitCount);
    void ThreadBlockNumCalc(uint32_t threadNum, uint32_t& factor, uint32_t& factorTail);
    void SetTilingData();
};
}
#endif
