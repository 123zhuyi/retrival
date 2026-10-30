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
 * \file asinh_tiling.cpp
 * \brief Asinh 算子 Host Tiling（arch35 / Ascend950）
 *
 * 与 DESIGN.md v1.1 §3.6 对齐（**v1.1 修订要点**）：
 *   - 平台信息动态获取：GetCoreNumAiv() / GetCoreMemSize(UB) （禁止硬编码）
 *   - 多核切分：blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), 32B)
 *   - UB 切分：
 *     * 单 tile 同时活跃 Buffer 估算（FP32 视角保守上界）：
 *         inputQue×2 (DB) + outputQue×2 (DB) → 4 × sizeof(FP32) = 16 字节/元素
 *         xOrigBuf + absXBuf + bBuf + rBuf + sBuf  → 5 × 4 = 20 字节/元素
 *         selMaskBuf (uint8_t) → 1 字节/元素
 *       合计 BYTES_PER_ELEM_TOTAL = 37 字节/元素
 *     * **v1.1 关键修订：ubFactor 对齐边界从 32B (8 elements) 升级为 256B (64 elements)**
 *       满足 Compare.md line 114/125 的 count 个元素所占空间必须 256 字节对齐的强约束
 *       FP32 视角即 64 元素 = 256B；本算子 Compare 调用始终在 FP32 工作区
 *     * 公式：ubFactor = FloorAlign((ubSize - logTmpReserve) / 37, 64)
 *   - Log 隐式 tmpBuffer 预留：通过 GetLogMaxMinTmpSize 在 ubSize 中扣除
 *   - 空 Tensor 早返回 + 32B 对齐尾块（DataCopyPad 自动处理）
 *   - TilingKey 编码：ASCENDC_TPL_SEL_PARAM(context, dtype)，dtype 维度即 D_T_X
 *
 * 迭代一范围（FP32 单 dtype 走通）：
 *   - dtype 校验：3 dtype 均放行（FP16/BF16 在 Kernel 端已有 Cast 路径骨架，
 *     完整端到端验证在迭代二完成）
 */

#include <algorithm>
#include <vector>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/math/log_tiling.h"
#include "../../op_kernel/arch35/asinh_tiling_data.h"
#include "../../op_kernel/arch35/asinh_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;

// 单 tile 同时活跃的 UB 字节占用（FP32 视角保守上界，DESIGN §3.8.1）：
//   inputQue×2 (DB) + outputQue×2 (DB)：4 份 sizeof(T)（T 视为 FP32 = 4 字节）→ 16 字节/元素
//   xOrigBuf + absXBuf + bBuf + rBuf + sBuf：5 份 FP32 → 20 字节/元素
//   selMaskBuf：1 字节/元素（uint8_t mask）
//   合计：37 字节/元素
// FP16/BF16 路径下 inputQue/outputQue 元素 2 字节（合计 29 字节/元素），统一按 FP32 上界 37 估算。
constexpr int64_t BYTES_PER_ELEM_TOTAL = 37;

// **v1.1 关键参数**：Compare.md line 114/125 强约束 — count 个元素所占空间必须 256B 对齐
// FP32 视角即 256B / sizeof(float) = 64 元素
constexpr int64_t COMPARE_ALIGN_ELEMS = 64;

// GetLogMaxMinTmpSize 用 typicalShape 估算 Log 隐式 tmpBuffer 上界。
// 8192 元素是与 ubFactor 估算典型值对齐的保守阈值（DESIGN §3.6 / §3.8.2）：
//   - 实际单 tile 元素数 ubFactor ≈ 6720（FP32 视角，248KB / 37 字节/元素，再 64 对齐）
//   - 取 8192（>6720 略大）作为 typicalShape 上界，避免 GetLogMaxMinTmpSize 输入过大溢出
constexpr int64_t LOG_TMP_TYPICAL_SHAPE_MAX = 8192;

// 获取平台信息：UB 容量与 AI Core 数（动态获取，禁止硬编码）
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "Asinh: coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "Asinh: ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 校验 shape 与 dtype，输出 totalNum / dtype。
// MED-1：Tiling 作为算子端最后一道校验，对负 shape 显式拒绝。
static ge::graphStatus ValidateInput(gert::TilingContext* context, int64_t* totalNum, ge::DataType* dtype)
{
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    *totalNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(*totalNum < 0,
        OP_LOGE(context, "Asinh: invalid totalNum=%ld (negative shape size)", *totalNum),
        return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dtype = inputDesc->GetDataType();
    const std::set<ge::DataType> supported = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    OP_CHECK_IF(supported.count(*dtype) == 0,
        OP_LOGE(context, "Asinh: unsupported dtype %d", static_cast<int>(*dtype)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算 ubFactor：扣除 Log 隐式 tmpBuffer 预留后均分各 Buffer，按 64 元素 (256B) 对齐
// 满足 Compare.md line 114/125 的 count 256B 对齐强约束（FP32 视角 64 元素 = 256B）。
static ge::graphStatus CalcUbFactor(gert::TilingContext* context, uint64_t ubSize, int64_t blockFactor,
                                    int64_t* ubFactor, int64_t* logTmpReserveBytes)
{
    uint32_t logMaxValue = 0U;
    uint32_t logMinValue = 0U;
    std::vector<int64_t> typicalShapeVec = {std::min<int64_t>(blockFactor, LOG_TMP_TYPICAL_SHAPE_MAX)};
    ge::Shape typicalShape(typicalShapeVec);
    AscendC::GetLogMaxMinTmpSize(typicalShape, sizeof(float), false, logMaxValue, logMinValue);
    *logTmpReserveBytes = static_cast<int64_t>(logMinValue);

    int64_t ubAvail = static_cast<int64_t>(ubSize) - *logTmpReserveBytes;
    OP_CHECK_IF(ubAvail <= 0,
        OP_LOGE(context, "Asinh: ubAvail <= 0 after Log tmp reserve, ubSize=%lu logTmpReserve=%ld",
                static_cast<unsigned long>(ubSize), *logTmpReserveBytes),
        return ge::GRAPH_FAILED);

    *ubFactor = FloorAlign(FloorDiv(ubAvail, BYTES_PER_ELEM_TOTAL), COMPARE_ALIGN_ELEMS);
    OP_CHECK_IF(*ubFactor <= 0,
        OP_LOGE(context, "Asinh: ubFactor too small (< 64 elems after Log tmp reserve), ubFactor=%ld",
                *ubFactor),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AsinhTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter AsinhTilingFunc");
    // 1. 平台 / 输入校验
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Asinh: GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dtype = ge::DT_FLOAT;
    OP_CHECK_IF(ValidateInput(context, &totalNum, &dtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Asinh: ValidateInput error"), return ge::GRAPH_FAILED);

    // 2. workspace
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = WS_SYS_SIZE;

    // 3. TilingData
    AsinhTilingData* tiling = context->GetTilingData<AsinhTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(AsinhTilingData), 0, sizeof(AsinhTilingData)) != EOK,
        OP_LOGE(context, "Asinh: memset tiling failed"), return ge::GRAPH_FAILED);

    // 4. 空 Tensor 早返回：Tiling 层 SetBlockDim(1)，Kernel 内 Process() 早返回
    if (totalNum == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));
        return ge::GRAPH_SUCCESS;
    }

    // 5. 多核切分（按 32B 向上对齐，避免相邻核 CopyOut 写覆盖）
    tiling->totalNum = totalNum;
    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // 6. UB 切分
    int64_t logTmpReserveBytes = 0;
    OP_CHECK_IF(CalcUbFactor(context, ubSize, tiling->blockFactor, &tiling->ubFactor, &logTmpReserveBytes)
                != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Asinh: CalcUbFactor error"), return ge::GRAPH_FAILED);

    // 7. BlockDim & TilingKey
    context->SetBlockDim(usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));

    OP_LOGI(context,
            "Asinh: totalNum=%ld, blockFactor=%ld, ubFactor=%ld, usedCoreNum=%ld, dtype=%d, "
            "ubSize=%lu, logTmpReserve=%ld",
            tiling->totalNum, tiling->blockFactor, tiling->ubFactor, usedCoreNum,
            static_cast<int>(dtype), static_cast<unsigned long>(ubSize), logTmpReserveBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAsinh([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AsinhCompileInfo {};  // 占位，入图场景依赖

IMPL_OP_OPTILING(Asinh).Tiling(AsinhTilingFunc).TilingParse<AsinhCompileInfo>(TilingParseForAsinh);

} // namespace optiling
