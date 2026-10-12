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
 * \file mod.cpp
 * \brief Mod L0 implementation with same-dtype AiCore/AICPU dispatch and geometry-adaptive broadcasting.
 *
 * aclnn
 * normalizes cross-dtype inputs before this layer. Scalar and eligible fp broadcast geometries remain
 * in-kernel;
 * other broadcasts are materialized so the kernel can use its same-shape path.
 */

#include "mod.h"

#include <vector>
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/op_dfx.h"
// ⚠️ 与同目录 aclnn_fmod_tensor.cpp 一致的 9.0.0 头路径 (该文件已用此路径且编过);
//    8.5.0 分支该头在 "conversion/broadcast_to/op_host/op_api/broadcast_to.h" (推 8.5.0 分支时需改)。
#include "conversion/broadcast_to/op_api/broadcast_to.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(Mod);

static const std::initializer_list<op::DataType> AICPU_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_DOUBLE, op::DataType::DT_INT64, op::DataType::DT_INT8, op::DataType::DT_UINT8};

// fused-broadcast UB-row 预算 (与 host 融合判定同量级)。
static constexpr uint64_t MOD_UB_USABLE = 184U * 1024U; // == host fused usable 上界
static constexpr int64_t MOD_INNER_ALIGN_FP32 = 8;      // 32B / 4B(fp32) = 8 elems
static constexpr int64_t MOD_INNER_ALIGN_2B = 16;       // 32B / 2B(bf16/fp16) = 16 elems

// AICore supports BF16, FP16, FP32 and INT32/INT16. Other README dtypes fall back to AICPU.
const aclTensor* ModAiCore(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ModAiCore, input, other, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Mod, OP_INPUT(input, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ModAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."),
             return nullptr);
    return out;
}

// TF-AICPU 支持FLOAT64
const aclTensor* ModAiCpu(const aclTensor* input, const aclTensor* other, aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(ModAiCpu, input, other, out);

    static internal::AicpuTaskSpace space("Mod", ge::DEPEND_IN_SHAPE, true);
    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(Mod, OP_ATTR_NAMES(), OP_INPUT(input, other), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ModAiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return out;
}

// 发射选择: AICPU dtype (double/int64/int8/uint8) -> ModAiCpu (上游能力, 保留); 其余 -> ModAiCore。
// (dtype 分流抽成 helper 供物化/未物化两路共用，语义不变。)
static const aclTensor* ModLaunch(const aclTensor* input, const aclTensor* other, aclTensor* out,
                                  aclOpExecutor* executor)
{
    if (CheckType(input->GetDataType(), AICPU_DTYPE_SUPPORT_LIST)) {
        return ModAiCpu(input, other, out, executor);
    }
    return ModAiCore(input, other, out, executor);
}

// 融合资格判定 (与 host mod_tiling.cpp ModFusedBroadcastEligible 一致)：
//   dtype ∈ {fp32,bf16,fp16} + self.shape == out.shape + 纯单轴 other 广播 collapse 到 [OUTER,INNER] +
//   INNER 32B 对齐 (dtype-aware) + 一行入 UB tile。true => 保持 other 未广播 (host Path B 命中 -> 快)。
//   int16 (及其它) -> false (走物化)。
//   原单体 ModFusedEligible 拆成 3 个纯谓词 helper + 薄主判定 (语义不变)。

// dtype-aware INNER 32B 对齐元素数; fused-eligible dtype (fp32/fp16/bf16) 之外返 0 (== 原 fusedDtypeOk=false)。
static int64_t ModFusedInnerAlign(op::DataType dt)
{
    if (dt == op::DataType::DT_FLOAT) {
        return MOD_INNER_ALIGN_FP32;
    }
    if (dt == op::DataType::DT_FLOAT16 || dt == op::DataType::DT_BF16) {
        return MOD_INNER_ALIGN_2B;
    }
    return 0;
}

// self 须已 == out.shape (self 在本路从不广播); rank ∈ [1,8] 且与 out 同 rank。
static bool ModSelfShapeMatchesOut(const op::Shape& ss, const op::Shape& outs)
{
    const size_t rs = ss.GetDimNum();
    if (rs == 0 || rs > 8 || rs != outs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < rs; ++i) {
        if (ss.GetDim(i) != outs.GetDim(i)) {
            return false; // self must already be out.shape (self never broadcasts in this route)
        }
    }
    return true;
}

// 逐轴判 other 广播态 (0=equal, 1=other-dim-1), collapse 相邻同态 -> 至多 2 段; 仅纯单轴广播
//   (OUTER-only 或 INNER-only) 返 true 并把 collapse 后的 INNER 段长写回 innerDim。前置: rs ∈ [1,8]。
static bool ModCollapseBcastSegments(const op::Shape& ss, const op::Shape& os, int64_t& innerDim)
{
    const size_t rs = ss.GetDimNum();
    const size_t ro = os.GetDimNum();
    int isB[8];
    int64_t sdim[8];
    for (size_t i = 0; i < rs; ++i) {
        const int64_t sd = ss.GetDim(i);
        sdim[i] = sd;
        const size_t fromBack = rs - 1 - i;
        const int64_t od = (fromBack < ro) ? os.GetDim(ro - 1 - fromBack) : 1;
        if (od == sd) {
            isB[i] = 0;
        } else if (od == 1) {
            isB[i] = 1;
        } else {
            return false; // neither equal nor 1 -> not a pure other-broadcast
        }
    }
    // collapse adjacent same-state axes -> at most 2 segments
    int64_t segLen[8];
    int segB[8];
    int nseg = 0;
    for (size_t i = 0; i < rs; ++i) {
        if (nseg > 0 && segB[nseg - 1] == isB[i]) {
            segLen[nseg - 1] *= sdim[i];
        } else {
            segB[nseg] = isB[i];
            segLen[nseg] = sdim[i];
            ++nseg;
        }
    }
    if (nseg == 0 || nseg > 2) {
        return false;
    }
    innerDim = segLen[nseg - 1];
    const int innerIsB = segB[nseg - 1];
    const int outerIsB = (nseg == 2) ? segB[0] : 0;
    const bool pureOuter = (innerIsB == 0 && outerIsB == 1);
    const bool pureInner = (innerIsB == 1 && outerIsB == 0);
    return pureOuter || pureInner; // false: no broadcast, or both axes broadcast (SCALAR special-case)
}

static bool ModFusedEligible(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    const int64_t innerAlign = ModFusedInnerAlign(self->GetDataType());
    if (innerAlign == 0) {
        return false; // dtype ∉ {fp32,fp16,bf16}
    }
    const op::Shape& ss = self->GetViewShape();
    const op::Shape& os = other->GetViewShape();
    const op::Shape& outs = out->GetViewShape();
    if (!ModSelfShapeMatchesOut(ss, outs)) {
        return false;
    }
    int64_t innerDim = 0;
    if (!ModCollapseBcastSegments(ss, os, innerDim)) {
        return false;
    }
    if (innerDim <= 0 || (innerDim % innerAlign) != 0) {
        return false; // INNER must be 32B-aligned for the dtype (fp32 % 8 / bf16-fp16 % 16)
    }
    if (static_cast<uint64_t>(innerDim) * 52U > MOD_UB_USABLE) {
        return false; // one INNER row must fit a UB tile (52*INNER bytes/row upper bound)
    }
    return true;
}

// BCAST_FIX: SCALAR 特例 — other 为单元素 (any rank) 且 out > 1 元素 (真广播)。host 检出 1-element other 发
//   scalar 快路。self 须 == out.shape (scalar other 不广播 self)。
static bool ModScalarEligible(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    if (static_cast<int64_t>(other->GetViewShape().GetShapeSize()) != 1) {
        return false;
    }
    if (static_cast<int64_t>(out->GetViewShape().GetShapeSize()) <= 1) {
        return false; // 1-element out: 无需广播 (kernel same-shape 快路已处理)
    }
    const op::Shape& ss = self->GetViewShape();
    const op::Shape& outs = out->GetViewShape();
    if (ss.GetDimNum() != outs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < ss.GetDimNum(); ++i) {
        if (ss.GetDim(i) != outs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

// in-kernel 通用广播的 tile-run 长度 (镜像 kernel mod_copy_impl.h GetInput2ContiguousCopyCount 的 collapse
//   + mod_tiling.cpp SetInput2ShapeInfo 的 stride)。suffix 小 = tile 窄 = 退化 (物化更快)；suffix 大 = tile
//   宽 = in-kernel 已快 (物化只增 HBM 流量)。
static int64_t ModGeneralBcastSuffix(const aclTensor* self, const aclTensor* other, const aclTensor* out)
{
    (void)self;
    const op::Shape& ss = out->GetViewShape(); // == self.shape (self 在本路不广播)
    const op::Shape& os = other->GetViewShape();
    const int64_t rs = static_cast<int64_t>(ss.GetDimNum());
    const int64_t ro = static_cast<int64_t>(os.GetDimNum());
    if (rs <= 0 || rs > 8) {
        return 1;
    }
    int64_t i2sh[8];
    int64_t i2st[8];
    int64_t stride = 1;
    for (int64_t i = rs - 1; i >= 0; --i) {
        const int64_t fromBack = rs - 1 - i;
        const int64_t od = (fromBack < ro) ? os.GetDim(ro - 1 - fromBack) : 1;
        i2sh[i] = od;
        i2st[i] = (od == 1) ? 0 : stride;
        stride *= od;
    }
    bool sawRealDim = false;
    bool isConstantRun = false;
    int64_t suffixSize = 1;
    int64_t expectedStride = 1;
    for (int64_t i = rs - 1; i >= 0; --i) {
        if (ss.GetDim(i) == 1) {
            continue;
        }
        if (i2st[i] == 0 && !sawRealDim) {
            suffixSize *= ss.GetDim(i);
            isConstantRun = true;
            continue;
        }
        if (isConstantRun) {
            break;
        }
        if (i2st[i] != expectedStride) {
            break;
        }
        sawRealDim = true;
        suffixSize *= ss.GetDim(i);
        expectedStride *= i2sh[i];
    }
    (void)isConstantRun;
    return suffixSize;
}

// 是否物化 (非 scalar/fused/no-bcast 的通用广播)。物化仅在净赚时：
//   * fp-output (AlgoA compute-bound)：物化的 other HBM 被计算掩盖 -> 恒赚，恒物化。
//   * int16 same-dtype (naive MTE2-bound)：物化把 other HBM 翻倍。仅当 in-kernel 退化 (窄 tile、小 suffix)
//     才净赚；宽 int16 广播 in-kernel 已快 + cache 友好，物化反劣 -> int16 物化门控 suffix < INT16_DEGEN。
static bool ModShouldMaterialize(const aclTensor* self, const aclTensor* other, const aclTensor* out,
                                 op::DataType outDtype)
{
    // outDtype = Mod 核输出 dtype (int16 <=> same-dtype int16 naive/MTE2 route).
    // int16-output 退化 suffix 实测 ≤90 需物化；物化反劣者 suffix≈443311。阈值取中间稳态 32768。
    // (fp16²→int16 之类 same-input-fp 其 Mod-out=fp16 -> 走下方 fp 恒物化分支，与本 int16 门无关。)
    constexpr int64_t INT16_DEGEN = 32768;
    if (outDtype != op::DataType::DT_INT16) {
        return true; // fp-output (AlgoA compute-bound): 恒物化 (非 fused/非 scalar 广播)
    }
    return ModGeneralBcastSuffix(self, other, out) < INT16_DEGEN;
}

// out.shape as aclIntArray (物化目标 shape)。
static aclIntArray* ModOutShape(const aclTensor* out, aclOpExecutor* executor)
{
    const int64_t dimNum = static_cast<int64_t>(out->GetViewShape().GetDimNum());
    if (dimNum == 0) {
        int64_t one[1] = {1};
        return executor->AllocIntArray(one, 1);
    }
    std::vector<int64_t> dims(static_cast<size_t>(dimNum));
    for (int64_t i = 0; i < dimNum; ++i) {
        dims[static_cast<size_t>(i)] = (out->GetViewShape())[i];
    }
    return executor->AllocIntArray(dims.data(), dims.size());
}

static const aclTensor* ModMaterializedLaunch(const aclTensor* self, const aclTensor* other, aclTensor* out,
                                              aclOpExecutor* executor)
{
    auto outShape = ModOutShape(out, executor);
    if (outShape == nullptr) {
        return nullptr;
    }
    const aclTensor* selfB = self;
    const aclTensor* otherB = other;
    if (self->GetViewShape() != out->GetViewShape()) {
        selfB = l0op::BroadcastTo(self, outShape, executor);
        if (selfB == nullptr) {
            return nullptr;
        }
    }
    if (other->GetViewShape() != out->GetViewShape()) {
        otherB = l0op::BroadcastTo(other, outShape, executor);
        if (otherB == nullptr) {
            return nullptr;
        }
    }
    return ModLaunch(selfB, otherB, out, executor);
}

// Core same-dtype dispatch. Cross-dtype inputs have already been normalized by aclnn.
static const aclTensor* ModImpl(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    op::Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), other->GetViewShape(), broadcastShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Broadcast %s and %s failed.", op::ToString(self->GetViewShape()).GetString(),
                op::ToString(other->GetViewShape()).GetString());
        return nullptr;
    }
    auto out = executor->AllocTensor(broadcastShape, self->GetDataType());
    if (out == nullptr) {
        return nullptr;
    }

    // in-kernel-eligible (免物化): SCALAR 特例或 FUSED 广播 -> host 走 scalar / Path B 快路。
    const bool scalarEligible = ModScalarEligible(self, other, out);
    const bool fusedEligible = !scalarEligible && ModFusedEligible(self, other, out);

    if (scalarEligible || fusedEligible) {
        // self 已 == out.shape (两个 predicate 都要求); other 保持原 shape (未广播)。
        return ModLaunch(self, other, out, executor);
    }

    // 非退化的 general 广播 (fp 恒物化；int16 仅退化时物化，宽 int16 保持 in-kernel) -> 物化净劣时保持未广播
    //   (in-kernel 已快，零回归；见 ModShouldMaterialize)。no-bcast 亦落此路 (物化恒 true 但下方两个
    //   BroadcastTo 都跳过 -> 发射不变)。
    const bool materialize = ModShouldMaterialize(self, other, out, self->GetDataType());
    // DAV_2201 BroadcastTo has no AiCore int16 lane. Keep the existing suffix profitability decision, but only
    // materialize when that route can stay on AiCore; otherwise use the already-supported in-kernel broadcast.
    const bool int16BroadcastFallsBackToAiCpu = self->GetDataType() == op::DataType::DT_INT16;
    if (!materialize || int16BroadcastFallsBackToAiCpu) {
        return ModLaunch(self, other, out, executor);
    }

    // ---- ineligible: canonical BroadcastTo 物化 (kernel 见 same-shape 连续 = NONE 快路) ----
    // no-bcast (self/other 已 == out.shape) 自然落此路但两个 BroadcastTo 都跳过 -> 发射不变。
    return ModMaterializedLaunch(self, other, out, executor);
}

// AICORE with AICPU fallback; out dtype equals the normalized input dtype.
const aclTensor* Mod(const aclTensor* input, const aclTensor* other, aclOpExecutor* executor)
{
    L0_DFX(Mod, input, other);
    return ModImpl(input, other, executor);
}
} // namespace l0op
