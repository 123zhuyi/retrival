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
 * \file mod_tiling.cpp
 * \brief
 */

#include <cstdlib>
#include <cmath>
#include <cerrno>
#include <limits>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "register/tilingdata_base.h"

#include "../op_kernel/mod_tiling_data.h"
#include "../op_kernel/mod_tiling_key.h"
#include "torch_extension/tiling_utils_math.h"
#include "op_host/tiling_base_util.h"

using namespace ge;

namespace ModNs {

class ModTiling {
public:
    constexpr static int64_t MINIMUM_ELEMENT_PER_CORE = 1024;

    constexpr static int64_t DATA_BLOCK = 64;
    constexpr static int64_t RESERVERD_UB_SIZE = 1024;

    template <typename T>
    static void ModCommonTiling(T x, ModTilingData& tilingData, uint32_t coreNum, uint64_t ubSize, uint32_t ubDivider)
    {
        if (ubDivider == 0) {
            return;
        }

        int64_t elementCount = 1;

        for (uint16_t i = 0; i < TilingUtils::GetDimNum(x); i++) {
            elementCount *= TilingUtils::GetDim(x, i);
        }

        uint32_t blockDim = (elementCount + MINIMUM_ELEMENT_PER_CORE - 1) / MINIMUM_ELEMENT_PER_CORE;

        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
        if (blockDim == 0) {
            blockDim = 1;
        }

        uint32_t dataBlockSize = DATA_BLOCK;
        uint32_t usableUbSize = uint32_t(ubSize - RESERVERD_UB_SIZE - sizeof(ModTilingData)) / ubDivider;
        usableUbSize = usableUbSize / dataBlockSize * dataBlockSize;

        uint64_t perCoreDataCount = elementCount / blockDim;
        perCoreDataCount = perCoreDataCount / DATA_BLOCK * DATA_BLOCK;

        uint64_t tempTailDataCount = elementCount - perCoreDataCount * blockDim;
        uint64_t tailDataCoreNum = 0;
        uint64_t lastCoreDataCount = 0;

        tailDataCoreNum = tempTailDataCount / DATA_BLOCK;
        lastCoreDataCount = perCoreDataCount + tempTailDataCount % DATA_BLOCK;

        tilingData.usableUbSize = usableUbSize;
        tilingData.needCoreNum = blockDim;
        tilingData.totalDataCount = elementCount;
        tilingData.perCoreDataCount = perCoreDataCount;
        tilingData.tailDataCoreNum = tailDataCoreNum;
        tilingData.lastCoreDataCount = lastCoreDataCount;
    }
};

} // namespace ModNs

using namespace ModNs;

namespace optiling {

struct ModCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
// FP 路自适应仍无条件分配 5 个 fp32 工作块 (A1..A5，+20 B/elem，运行时任一 tile 可能走 AlgoA) -> 与 kernel
//   InitBuffers 锁步。每元素字节占用之和 (队列 bufferNum=2)：
//   FP32 = 队列 3*2*4=24 + tmp 4 + (ResQuot+ResRem+Zero+Inf+Nan) 5*4=20 + Mask 1 + A1..A5 5*4=20 = 69
//   FP16/BF16 = 队列 3*2*2=12 + tmp 4 + 20 + Mask 1 + A1..A5 20 + x1/x2FP32 8 = 65
//   INT32 路不走 AlgoA，保持 69
// int16 same-dtype 走整数域 naive，kernel USE_ALGO_A=false 不分配 A1..A5 (-20 B/elem)。实测 footprint = 队列
//   3*2*2=12 + tmp 4 + (ResQuot+ResRem+Zero+Inf+Nan) 20 + Mask 1 + x1/x2FP32 8 = 45。下调到 45 与 kernel 锁步。
constexpr uint32_t UB_DIVIDER_FP32 = 69;
constexpr uint32_t UB_DIVIDER_FP16 = 65;
constexpr uint32_t UB_DIVIDER_INT16 = 45;
constexpr uint32_t UB_DIVIDER_INT32 = 69;
// same-dtype fp32/fp16/bf16 的连续派发 (isInput2Scalar || isInput2SameShape) 走 kernel 精简核 (USE_LEAN_CONTIG)
//   —— 绕过 inf/nan/zero Select 收尾 + 常驻常量 + tmp -> perElem 69/65 -> 48。lockstep kernel InitFlatBuffers +
//   InitLeanWorkBuffers：fp32-native = flat(self 2*4 + other 2*4 + out 2*4)=24 + (w0..w5) 6*4=24 = 48；
//   fp16/bf16 = flat(2*2*3)=12 + w0..w5 24 + rF32 4 + selfF32 4 + otherF32 4 = 48。
//   ⚠️ 仅【连续派发】用此 divider；general broadcast (非融合) 仍走 ComputeCore (需 inf/nan 常量，footprint
//   69/65) -> 保持原 UB_DIVIDER_FP32/FP16。int16/int32 不精简 -> 保持原 divider。mod 仅注册 arch22
//   (ascend910b/ascend910_93 均 DAV_2201) -> 无非 arch22 kernel 消费此连续 tiling。
constexpr uint32_t UB_DIVIDER_FP32_LEAN = 48;
constexpr uint32_t UB_DIVIDER_FP16_LEAN = 48;

// 自适应路由阈值。默认 256 (远低于大商例：int16 满值域 32767 / |a|>1000 -> 所有大商 case 路由到 AlgoA)。
//   env FMOD_NAIVE_THRESH 覆盖供真机 sweep (生产默认 unset)。
constexpr float FMOD_NAIVE_THRESH_DEFAULT = 256.0f;
static inline float FmodNaiveThresh()
{
    // 环境变量在进程内不变，缓存为 function-local static (C++11 once-init 线程安全)，避免每次 tiling 都
    //   getenv/strtod。行为等价 (同一 env 值)。
    static const float cached = []() -> float {
        const char* s = std::getenv("FMOD_NAIVE_THRESH");
        if (s == nullptr || s[0] == '\0') {
            return FMOD_NAIVE_THRESH_DEFAULT;
        }
        errno = 0;
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        const float f = static_cast<float>(v);
        // strtod 溢出返 ±HUGE_VAL(±inf) 且置 errno=ERANGE；double->float 收窄也可能溢成 ±inf (如 "1e300" 是
        //   有限 double 却越 FLT_MAX -> +inf-float)。若漏拦，naiveThresh_=+inf -> kernel 内 |商|>naiveThresh_ 恒假
        //   -> AlgoA 大商精度路被静默禁用。故 unparsable(end==s) / 上溢下溢 (errno==ERANGE) / 非有限(±inf/NaN) /
        //   非正(<=0) 一律回退默认阈值 (与 env unset 行为一致)。
        if (end == s || errno == ERANGE || !std::isfinite(f) || !(f > 0.0f)) {
            return FMOD_NAIVE_THRESH_DEFAULT;
        }
        return f;
    }();
    return cached;
}

static ge::graphStatus TilingPrepare4ModTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ModCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->isRegbase = (Ops::Base::IsRegbaseSocVersion(context)) ? true : false;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->totalCoreNum <= 0 || compileInfo->ubSize <= 0),
                OP_LOGE(context, "Mod GetHardwareInfo Failed, vectorCoreNum:%d, ubSize:%ld.", compileInfo->totalCoreNum,
                        compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Get totalCoreNum:%d, ubSize:%ld", compileInfo->totalCoreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

// 单 dtype -> MOD_TPL_* 编译期 dtype 信号映射。
static uint32_t MapModDtype(ge::DataType dtype)
{
    if (dtype == ge::DataType::DT_FLOAT) {
        return MOD_TPL_FP32;
    } else if (dtype == ge::DataType::DT_FLOAT16) {
        return MOD_TPL_FP16;
    } else if (dtype == ge::DataType::DT_BF16) {
        return MOD_TPL_BF16;
    } else if (dtype == ge::DataType::DT_INT32) {
        return MOD_TPL_INT32;
    } else if (dtype == ge::DataType::DT_INT16) {
        return MOD_TPL_INT16;
    }
    return MOD_TPL_FP32;
}

// 按 (x1, x2, y) 三 dtype 各自映射 MOD_TPL_*。op_def 仅注册同 dtype 三元组。
static void SetTilingKeyParams(ge::DataType dtypeX1, ge::DataType dtypeX2, ge::DataType dtypeY, uint32_t& dTypeX1,
                               uint32_t& dTypeX2, uint32_t& dTypeY, uint32_t& ubDivider)
{
    dTypeX1 = MapModDtype(dtypeX1);
    dTypeX2 = MapModDtype(dtypeX2);
    dTypeY = MapModDtype(dtypeY);

    if (dTypeY == MOD_TPL_FP32) {
        ubDivider = UB_DIVIDER_FP32;
    } else if (dTypeY == MOD_TPL_INT32) {
        ubDivider = UB_DIVIDER_INT32;
    } else if (dTypeY == MOD_TPL_INT16) {
        // int16 same-dtype naive，不分配 A1..A5 -> 45 (与 kernel 锁步)。
        ubDivider = UB_DIVIDER_INT16;
    } else {
        // fp16 / bf16 same-dtype (复用 FP cast 路, adaptive 分配 A1..A5)
        ubDivider = UB_DIVIDER_FP16;
    }
}

static ge::graphStatus CheckModTilingContext(gert::TilingContext* tilingContext, const ModCompileInfo*& compileInfo,
                                             const gert::StorageShape*& shape, const gert::StorageShape*& otherShape)
{
    OP_CHECK_IF(tilingContext == nullptr, OP_LOGE("ModTiling", "tiling context is nullptr"), return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext, "Entering ModTilingForGe");
    compileInfo = reinterpret_cast<const ModCompileInfo*>(tilingContext->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);

    auto tempInputDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(tempInputDesc == nullptr, OP_LOGE(tilingContext, "InputDesc == nullptr"), return ge::GRAPH_FAILED);
    shape = tilingContext->GetInputShape(0);
    OP_CHECK_IF(shape == nullptr, OP_LOGE(tilingContext, "InputShape == nullptr"), return ge::GRAPH_FAILED);
    otherShape = tilingContext->GetInputShape(1);
    OP_CHECK_IF(otherShape == nullptr, OP_LOGE(tilingContext, "OtherShape == nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetInput2ShapeInfo(ModNs::ModTilingData* tilingData, const gert::Shape& input1StorageShape,
                               const gert::Shape& input2StorageShape)
{
    tilingData->isInput2Scalar = (input2StorageShape.GetShapeSize() == 1);
    tilingData->dimNum = static_cast<uint32_t>(input1StorageShape.GetDimNum());
    tilingData->isInput2SameShape = (input1StorageShape.GetDimNum() == input2StorageShape.GetDimNum());
    if (tilingData->isInput2SameShape) {
        for (uint32_t i = 0; i < tilingData->dimNum; ++i) {
            if (input1StorageShape.GetDim(i) != input2StorageShape.GetDim(i)) {
                tilingData->isInput2SameShape = false;
                break;
            }
        }
    }
    for (uint32_t i = 0; i < 8; ++i) {
        tilingData->input1Shape[i] = 1;
        tilingData->input2Shape[i] = 1;
        tilingData->input2Stride[i] = 0;
    }
    uint64_t stride = 1;
    const int64_t input1DimNum = static_cast<int64_t>(input1StorageShape.GetDimNum());
    const int64_t input2DimNum = static_cast<int64_t>(input2StorageShape.GetDimNum());
    for (int64_t i = static_cast<int64_t>(tilingData->dimNum) - 1; i >= 0; --i) {
        tilingData->input1Shape[i] = static_cast<uint64_t>(input1StorageShape.GetDim(i));
        const int64_t input2DimIndex = i - (input1DimNum - input2DimNum);
        if (input2DimIndex >= 0) {
            tilingData->input2Shape[i] = static_cast<uint64_t>(input2StorageShape.GetDim(input2DimIndex));
        }
        tilingData->input2Stride[i] = (tilingData->input2Shape[i] == 1) ? 0 : stride;
        stride *= tilingData->input2Shape[i];
    }
}

// 融合广播判定 (collapse-2D / 32B-INNER / 纯 OUTER|INNER 广播)。原单函数圈复杂度 35 + 104 行超阈，拆成
//   ①ModCollapseBroadcastSegments (右对齐 + 相邻同广播态 collapse) ②ModFusedBroadcastEligible (段数/mode/
//   32B 对齐 资格判定) ③ModComputeFusedTiling (ubFormer 估算 + 有界 fit 校验 + 写 tilingData) ④ModTryFusedBroadcast
//   (薄编排)。纯机械拆分，分支/退出条件不变 (唯一新增 = ③ 的有界 guard 上界)。

// 右对齐 other 到 self, 逐轴标注广播态 (isB=0 匹配 / isB=1 纯 other 广播), 相邻同态 collapse 成 <=MAX_RANK 段。
// self 侧广播 (od 既不等也不为 1) 或 rank 非法 -> 返回 false (不融合)。输出 segLen[]/segB[]/nseg (INNER 在末段)。
static bool ModCollapseBroadcastSegments(const gert::Shape& selfShape, const gert::Shape& otherShape, int64_t* segLen,
                                         int* segB, int& nseg)
{
    constexpr int64_t MAX_RANK = 8;
    const int64_t rs = static_cast<int64_t>(selfShape.GetDimNum());
    const int64_t ro = static_cast<int64_t>(otherShape.GetDimNum());
    if (rs <= 0 || rs > MAX_RANK) {
        return false;
    }
    int64_t sdim[MAX_RANK];
    int isB[MAX_RANK];
    for (int64_t i = 0; i < rs; ++i) {
        const int64_t sd = selfShape.GetDim(i);
        sdim[i] = sd;
        const int64_t fromBack = rs - 1 - i;
        const int64_t od = (fromBack < ro) ? otherShape.GetDim(ro - 1 - fromBack) : 1;
        if (od == sd) {
            isB[i] = 0;
        } else if (od == 1) {
            isB[i] = 1;
        } else {
            return false; // 既不等也不为 1 -> 非纯 other 广播 -> 通用路
        }
    }
    nseg = 0;
    for (int64_t i = 0; i < rs; ++i) {
        if (nseg > 0 && segB[nseg - 1] == isB[i]) {
            segLen[nseg - 1] *= sdim[i];
        } else {
            segB[nseg] = isB[i];
            segLen[nseg] = sdim[i];
            ++nseg;
        }
    }
    return true;
}

// 段数须 <=2 -> [OUTER, INNER]; 恰好 OUTER 广播 (mode 1) 或 INNER 广播 (mode 2)。0811 修复：不再要求
// INNER 32B 对齐——非对齐由 kernel 侧 padding 行布局吸收 (bcIpad 按 dtype 感知规则 ceil(inner*sizeof/32)*
// 32/sizeof，见 ModComputeFusedTiling)。命中填 mode/inner/outer 并返回 true。
static bool ModFusedBroadcastEligible(const gert::Shape& selfShape, const gert::Shape& otherShape, uint32_t dTypeY,
                                      uint32_t& mode, uint64_t& inner, uint64_t& outer)
{
    constexpr int64_t MAX_RANK = 8;
    int64_t segLen[MAX_RANK];
    int segB[MAX_RANK];
    int nseg = 0;
    if (!ModCollapseBroadcastSegments(selfShape, otherShape, segLen, segB, nseg)) {
        return false;
    }
    if (nseg == 0 || nseg > 2) {
        return false;
    }
    const int64_t innerDim = segLen[nseg - 1];
    const int innerIsB = segB[nseg - 1];
    int64_t outerDim = 1;
    int outerIsB = 0;
    if (nseg == 2) {
        outerDim = segLen[0];
        outerIsB = segB[0];
    }
    // 纯几何: 恰好 OUTER 广播 或 INNER 广播 (两轴都广播=标量 -> flat/通用; 都不广播=无广播 -> 通用路)。
    if (innerIsB == 0 && outerIsB == 1) {
        mode = 1; // OUTER (行) 广播
    } else if (innerIsB == 1 && outerIsB == 0) {
        mode = 2; // INNER (列) 广播
    } else {
        return false;
    }
    // 0811: 取消 32B 对齐门槛 (原 innerDim % (32/elemBytes) != 0 即拒)。非对齐 inner 由 padding 行布局吸收,
    //   dTypeY 参数保留 (签名兼容) 但不再用于对齐判定。
    (void)dTypeY;
    if (innerDim <= 0 || outerDim <= 0) {
        return false;
    }
    inner = static_cast<uint64_t>(innerDim);
    outer = static_cast<uint64_t>(outerDim);
    return true;
}

// 防御式实测 fit 校验 (镜像 kernel 分配)：总 UB = perElem*tileAligned + rawPerElem*rawAligned <= fusedUsable。
//   tile 侧按 padding 行步长 ipad 计 (tileElems = ubFormer*ipad)；raw 侧仍按原始 inner (OUTER 原始行) /
//   ubFormer (INNER per-row 标量) 计。命中 -> ubFormer 就绪返回 true；放不下 (INNER 过宽) -> false (通用路)。
//   原 `for(;;)` 改有界 for + guard<=64 安全上界 (ubFormer 估算已含 FUSED_RAW_SLOP 缓冲 -> 收敛 <~10 次，
//   64 绰绰有余)；保留原退出条件 (fit / ubFormer<=1)；guard 耗尽仍未 fit (分析上不可达) -> 保守 false。
static bool ModFitFusedUbFormer(uint32_t mode, uint64_t inner, uint64_t ipad, uint64_t fusedUsable, uint64_t perElem,
                                uint64_t rawPerElem, uint64_t& ubFormer)
{
    for (int guard = 0; guard < 64; ++guard) {
        const uint64_t tileElems = ubFormer * ipad;
        const uint64_t tileAligned = ((tileElems + 63U) / 64U) * 64U;
        const uint64_t rawElems = (mode == 1U) ? inner : ubFormer;
        uint64_t rawAligned = ((rawElems + 63U) / 64U) * 64U;
        if (rawAligned < 64U) {
            rawAligned = 64U;
        }
        const uint64_t total = perElem * tileAligned + rawPerElem * rawAligned;
        if (total <= fusedUsable) {
            return true;
        }
        if (ubFormer <= 1U) {
            return false; // 连 1 行都放不下 (INNER 过宽) -> 不融合, 通用路
        }
        --ubFormer;
    }
    return false; // guard 上界耗尽仍未 fit (分析上不可达) -> 保守不融合
}

// ubFormer 估算 + fit 校验 (ModFitFusedUbFormer) + 多核 blockFactor, 命中写 bcast* 字段 (kernel arch22 走融合广播)。
// 不改 SetBlockDim: 融合按 OUTER 行切到通用路 needCoreNum 批核 (越界核 coreRows_==0 空转) -> block dim 与通用/
// 非 arch22 回落路一致, 零回归。放不下 -> 直接 return (bcastFusedMode 保持调用方置的 0 -> 通用 ProcessBroadcast)。
static void ModComputeFusedTiling(uint32_t mode, uint64_t inner, uint64_t outer, uint64_t ubSize, uint32_t needCoreNum,
                                  uint32_t dTypeY, ModNs::ModTilingData* tilingData)
{
    constexpr uint64_t FUSED_RESERVED = 1024;
    constexpr uint64_t FUSED_RAW_SLOP = 8192; // 预留给原始 other 队列 + 对齐 slop (实测 fit 校验兜底)
    if (ubSize <= FUSED_RESERVED + sizeof(ModNs::ModTilingData)) {
        return;
    }
    const uint64_t fusedUsable = ubSize - FUSED_RESERVED - sizeof(ModNs::ModTilingData);
    // 0811 padding 行步长：ipad = ceil(inner*sizeof(dtype)/32)*32/sizeof(dtype) (fp32 8-elem / 2B 16-elem 单位)。
    //   与 kernel DataCopyPad blockCount 模式的自动 padding 落块规则 lockstep (UB 行首恒 32B 对齐 -> UB->UB
    //   行复制 / 逐行 Duplicate 合法)，任意 inner 可融合。ipad==inner 时与原 1D 平铺等价。
    const uint64_t elemBytes = (dTypeY == MOD_TPL_FP32) ? 4U : 2U; // same-dtype: x1/x2/y 同宽
    const uint64_t alignElems = 32U / elemBytes;
    const uint64_t ipad = (inner + alignElems - 1U) / alignElems * alignElems;
    // perElem = tileAligned-尺寸 buffer 的每元素字节 (与 kernel InitFusedBcastBuffers 精简核 lockstep)：
    //   fp32: self 2*4 + out 2*4 + otherF32 4 + (ResQuot/ResRem/A1..A4=w0..w5) 6*4 = 44
    //   2B  : self 2*2 + out 2*2 + otherF32 4 + w0..w5 24 + rF32(A5) 4 + selfF32 4 = 44
    //   (0811 起 2B 含 int16：NEED_FP32_IO_BUF 对 int16 为 true，selfF32/rF32/cdminF32 与 fp16/bf16 同套)
    const uint64_t perElem = 44U;
    const uint64_t rawPerElem = 8U;
    const uint64_t denom = perElem * ipad + rawPerElem; // +INNER-scalar raw queue 上界 (随行增长)
    uint64_t ubFormer = (fusedUsable > FUSED_RAW_SLOP) ? (fusedUsable - FUSED_RAW_SLOP) / denom : 0U;
    if (ubFormer < 1U) {
        ubFormer = 1U;
    }
    if (ubFormer > outer) {
        ubFormer = outer;
    }
    // 防御式实测 fit 校验 (有界 guard；见 ModFitFusedUbFormer)。放不下 -> 不融合，通用路。
    if (!ModFitFusedUbFormer(mode, inner, ipad, fusedUsable, perElem, rawPerElem, ubFormer)) {
        return;
    }
    // 多核: OUTER 行切到通用路的 needCoreNum 批核 (SetBlockDim 不变)。
    const uint64_t cores = (needCoreNum == 0U) ? 1U : static_cast<uint64_t>(needCoreNum);
    uint64_t blockFactor = (outer + cores - 1U) / cores;
    if (blockFactor < 1U) {
        blockFactor = 1U;
    }
    tilingData->bcastFusedMode = mode;
    tilingData->bcOuter = outer;
    tilingData->bcInner = inner;
    tilingData->bcUbFormer = ubFormer;
    tilingData->bcBlockFactor = blockFactor;
    tilingData->bcIpad = ipad;
}

// 薄编排 (仅对【同 dtype fp32/fp16/bf16/int16 + 通用广播 (非 scalar 非 same-shape)】调用)：资格判定命中则
// 计算 tiling；否则 bcastFusedMode 保持 0 -> 走通用 ProcessBroadcast (零回归)。mixed / int32 / 非 2D-collapse
// 几何 -> 全不融合。
static void ModTryFusedBroadcast(const gert::Shape& selfShape, const gert::Shape& otherShape, uint32_t dTypeY,
                                 uint64_t ubSize, uint32_t needCoreNum, ModNs::ModTilingData* tilingData)
{
    uint32_t mode = 0;
    uint64_t inner = 0;
    uint64_t outer = 0;
    if (!ModFusedBroadcastEligible(selfShape, otherShape, dTypeY, mode, inner, outer)) {
        return;
    }
    ModComputeFusedTiling(mode, inner, outer, ubSize, needCoreNum, dTypeY, tilingData);
}

// same-dtype fp32/fp16/bf16 的连续派发 (isInput2Scalar || isInput2SameShape) 走 kernel 精简核 -> 下调 divider
//   到 LEAN 值 (tile 更宽)。镜像 SetInput2ShapeInfo 的 isInput2Scalar/isInput2SameShape 分类 (kernel 消费同一
//   分类)。general broadcast / int16 / int32 -> 返回原 ubDivider 不变。
static uint32_t ModSelectContiguousLeanDivider(const gert::Shape& selfSh, const gert::Shape& otherSh, uint32_t dTypeX1,
                                               uint32_t dTypeX2, uint32_t dTypeY, uint32_t ubDivider)
{
    bool input2Scalar = (otherSh.GetShapeSize() == 1);
    bool sameShape = (selfSh.GetDimNum() == otherSh.GetDimNum());
    if (sameShape) {
        for (size_t i = 0; i < selfSh.GetDimNum(); ++i) {
            if (selfSh.GetDim(i) != otherSh.GetDim(i)) {
                sameShape = false;
                break;
            }
        }
    }
    const bool contiguous = input2Scalar || sameShape;
    const bool sameDtype = (dTypeX1 == dTypeX2) && (dTypeX1 == dTypeY);
    if (contiguous && sameDtype) {
        if (dTypeY == MOD_TPL_FP32) {
            return UB_DIVIDER_FP32_LEAN;
        } else if (dTypeY == MOD_TPL_FP16 || dTypeY == MOD_TPL_BF16) {
            return UB_DIVIDER_FP16_LEAN;
        }
    }
    return ubDivider;
}

static void SetFusedBroadcastTiling(const gert::Shape& selfShape, const gert::Shape& otherShape, uint32_t dTypeX1,
                                    uint32_t dTypeX2, uint32_t dTypeY, uint64_t ubSize,
                                    ModNs::ModTilingData* tilingData)
{
    tilingData->bcastFusedMode = 0;
    tilingData->bcOuter = 0;
    tilingData->bcInner = 0;
    tilingData->bcUbFormer = 0;
    tilingData->bcBlockFactor = 0;
    tilingData->bcIpad = 0;
    const bool generalBcast = !tilingData->isInput2Scalar && !tilingData->isInput2SameShape;
    const bool supportedDtype = dTypeY == MOD_TPL_FP32 || dTypeY == MOD_TPL_FP16 || dTypeY == MOD_TPL_BF16 ||
                                dTypeY == MOD_TPL_INT16;
    const bool sameDtypeFused = dTypeX1 == dTypeX2 && dTypeX1 == dTypeY && supportedDtype;
    if (generalBcast && sameDtypeFused) {
        ModTryFusedBroadcast(selfShape, otherShape, dTypeY, ubSize, tilingData->needCoreNum, tilingData);
    }
}

static ge::graphStatus ModTilingForGe(gert::TilingContext* tilingContext)
{
    const ModCompileInfo* compileInfo = nullptr;
    const gert::StorageShape* shape = nullptr;
    const gert::StorageShape* otherShape = nullptr;
    auto ret = CheckModTilingContext(tilingContext, compileInfo, shape, otherShape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ModNs::ModTilingData* tilingData = tilingContext->GetTilingData<ModNs::ModTilingData>();
    uint32_t D_T_X1, D_T_X2, D_T_Y, ubDivider;
    // 读 x1(0)/x2(1)/y(out 0) 三 dtype；op_def 保证三者同 dtype。
    auto x2Desc = tilingContext->GetInputDesc(1);
    OP_CHECK_IF(x2Desc == nullptr, OP_LOGE(tilingContext, "InputDesc(1) == nullptr"), return ge::GRAPH_FAILED);
    auto yDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE(tilingContext, "OutputDesc(0) == nullptr"), return ge::GRAPH_FAILED);
    ge::DataType dtype_x1 = tilingContext->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_x2 = x2Desc->GetDataType();
    ge::DataType dtype_y = yDesc->GetDataType();
    SetTilingKeyParams(dtype_x1, dtype_x2, dtype_y, D_T_X1, D_T_X2, D_T_Y, ubDivider);

    // same-dtype fp32/fp16/bf16 的连续派发 (scalar/same-shape) -> kernel 精简核 (USE_LEAN_CONTIG) -> 下调 divider
    //   到 48 (tile 更宽 -> tile 数更少)。分类须在 ModCommonTiling (消费 ubDivider) 前完成 (host divider 与 kernel
    //   精简核 lockstep，绝不错配溢出)。
    ubDivider = ModSelectContiguousLeanDivider(shape->GetStorageShape(), otherShape->GetStorageShape(), D_T_X1, D_T_X2,
                                               D_T_Y, ubDivider);

    ModNs::ModTiling::ModCommonTiling<gert::Shape>(shape->GetStorageShape(), *tilingData, compileInfo->totalCoreNum,
                                                   compileInfo->ubSize, ubDivider);
    SetInput2ShapeInfo(tilingData, shape->GetStorageShape(), otherShape->GetStorageShape());
    tilingData->naiveThresh = FmodNaiveThresh();

    // 融合广播：默认关；仅【同 dtype fp32/fp16/bf16/int16 + 通用广播】尝试资格判定 (0811 起 int16 入列、
    //   去 32B 对齐约束 -> padding 行布局)。命中则填 bcast* 字段 (kernel arch22 走融合广播)；否则 0 ->
    //   通用 ProcessBroadcast。mixed / int32 不进融合 (mixed promote-round 语义另案；int32>2^24 fp32 不精确)。
    SetFusedBroadcastTiling(shape->GetStorageShape(), otherShape->GetStorageShape(), D_T_X1, D_T_X2, D_T_Y,
                            static_cast<uint64_t>(compileInfo->ubSize), tilingData);

    tilingContext->SetBlockDim(tilingData->needCoreNum);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(D_T_X1, D_T_X2, D_T_Y);

    tilingContext->SetTilingKey(tilingKey);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = WORK_SPACE_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Mod).Tiling(ModTilingForGe).TilingParse<ModCompileInfo>(TilingPrepare4ModTiling);

} // namespace optiling
