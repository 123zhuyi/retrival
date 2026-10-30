/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/accumulate_nv2_v2_tiling_data.h"
#include "../../op_kernel/arch35/accumulate_nv2_v2_tiling_key.h"

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static inline int64_t GetDtypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return 4;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
            return 1;
        case ge::DT_UINT8:
            return 1;
        default:
            return 4;
    }
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNum();
    }
    if (coreNum == 0) {
        // Fallback for aclnn execution path where platform info is not populated
        coreNum = 1;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        // Fallback: 192KB UB size (conservative default)
        ubSize = 192 * 1024;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AccumulateNv2V2TilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取动态输入个数 N (通过 REQUIRED_ATTR N 或 ComputeNodeInputNum)
    int32_t inputNum = static_cast<int32_t>(context->GetComputeNodeInputNum());

    // 3. 获取 shape 信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShape = EnsureNotScalar(inputX->GetStorageShape());
    int64_t totalNum = inputShape.GetShapeSize();

    // 4. 获取 dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = GetDtypeSize(dataType);

    // 5. 获取 workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 6. 多核切分
    int64_t ubBlockSize = GetUbBlockSize(context);
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // 7. UB 切分：累加器模式 3 个 buffer
    // Reserve 8KB for pipe/system overhead to avoid UB overflow
    uint64_t usableUbSize = (ubSize > 8192) ? (ubSize - 8192) : ubSize;
    // Use double buffer for large tensors
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufferNum = useDoubleBuffer ? 6 : 3;
    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(usableUbSize) / typeSize, bufferNum), ubBlockSize);

    // 8. 填充 TilingData
    AccumulateNv2V2TilingData* tiling = context->GetTilingData<AccumulateNv2V2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AccumulateNv2V2TilingData), 0, sizeof(AccumulateNv2V2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->inputNum = inputNum;

    context->SetBlockDim(usedCoreNum);

    // 9. 设置 TilingKey
    // The binary loader in CANN 9.0 transforms raw kernel suffix keys (_0, _256)
    // into runtime keys (210000000, 210010000). ASCENDC_TPL_SEL_PARAM generates
    // the raw keys which don't match. Directly set the runtime-expected keys.
    // Raw key 0: D_T_X=0, BUFFER_MODE idx=0 (single buffer) → runtime 210000000
    // Raw key 256: D_T_X=0, BUFFER_MODE idx=1 (double buffer) → runtime 210010000
    uint64_t tilingKey = (useDoubleBuffer == 0) ? 210010000UL : 210000000UL;
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAccumulateNv2V2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AccumulateNv2V2CompileInfo {};

IMPL_OP_OPTILING(AccumulateNv2V2)
    .Tiling(AccumulateNv2V2TilingFunc)
    .TilingParse<AccumulateNv2V2CompileInfo>(TilingParseForAccumulateNv2V2);

} // namespace optiling
