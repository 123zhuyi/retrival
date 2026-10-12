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
 * \file random_standard_normal_v2_tiling_arch35.cpp
 * \brief
 */
#include <random>
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "random_standard_normal_v2_tiling_arch35.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "../../../random_common/op_host/arch35/random_tiling_arch35.h"

using namespace ge;
namespace optiling {

static constexpr uint16_t SPLIT_UB_NUM = 3;
static constexpr uint32_t DCACHE_SIZE = 128U * 1024U;
static constexpr uint16_t SIZE_OF_FLOAT = 4;

// ========== 仅需配置规则+字段映射函数 ==========
OpTilingConfig RandomStandardNormalV2Tiling::BuildOpConfig()
{
    OpTilingConfig config;
    config.inputCheckRules = {
        // 输入索引:  dtype列表，shapeSize，dim_num
        {0, {{ge::DT_INT32, ge::DT_INT64}, -1, {1}, nullptr}}, // shape
        {1, {{ge::DT_INT64}, 1, {}, nullptr}},               // offset
    };
    config.DcacheSize = DCACHE_SIZE;
    config.outputCheckRules = {// 输出索引:  dtype列表，shapeSize，dim_num
                               {0, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {1,2,3,4,5,6,7,8}, nullptr}}}; // y

    // 获取output_size：输入0(shape)的shapeSize
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        return RandomUtils::GetAndCheckOutputSize<0,0>(ctx, shapeSize);
    };

    // 获取key[2]：从attr1(seed) counter[4] attr(seed2)
    config.getKeyAndCounter = [](gert::TilingContext* ctx, uint32_t key[2], uint32_t counter[4]) -> ge::graphStatus {
        return RandomUtils::GetKeyAndCounter<1, 2>(ctx, key, counter);
    };

    config.getBufferNum = [](gert::TilingContext* ctx, int64_t& bufNum) -> ge::graphStatus {
        auto outDesc = ctx->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outDesc);

        bufNum = SPLIT_UB_NUM * SIZE_OF_FLOAT;
        return ge::GRAPH_SUCCESS;
    };
    //   auto perCoreHandleRandomAlign = Ops::Base::CeilAlign(perCoreHandleRandom, coreAlignFactor); 128对齐了

    config.isNeedSyncAll = true;
    return config;
}

// ========== 算子构造函数
RandomStandardNormalV2Tiling::RandomStandardNormalV2Tiling(gert::TilingContext* ctx)
    : RandomTilingArch35(ctx, BuildOpConfig())
{}

static ge::graphStatus TilingPrepare4RandomStandardNormalV2Tiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<RandomStandardNormalV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
        OP_LOGE(
            context, "RandomStandardNormalV2 GetHardwareInfo Failed, vectorCoreNum:%ld, ubSize:%ld.",
            compileInfo->totalCoreNum, compileInfo->ubSize),
        return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingRandomStandardNormalV2(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingRandomStandardNormalV2");
    RandomStandardNormalV2Tiling tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(RandomStandardNormalV2)
    .Tiling(TilingRandomStandardNormalV2)
    .TilingParse<RandomStandardNormalV2CompileInfo>(TilingPrepare4RandomStandardNormalV2Tiling)
    .TilingInputsDataDependency({0});
} // namespace optiling
