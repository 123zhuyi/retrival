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
 * \file cdist_tiling.cpp
 * \brief
 */

#include <array>
#include <cmath>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "op_host/math_tiling_templates_registry.h"
#include "cdist_tiling_arch35.h"

namespace optiling {
// ============================================================
// M==1 broadcast 路由标定常量
//   broadcast 赢亏由 (plane, min(P,R), 对齐, B) 四维决定
// ============================================================
// ① OneDim：plane==1（P=R=1）输出塌一维，broadcast 快 ~11 倍
constexpr int64_t kOneDimPlane = 1;
// ② UB-BRC：P 或 R 任一超 64 → 分块处理任意大 plane
constexpr int64_t kUbBrcMinSide = 64;
// ③ 中小 plane (P,R<=64)：broadcast 净赢的平面起点
constexpr int64_t kBrcMidMinPlane = 256;
// ③ min(P,R) 下限：排除扁长劣化
constexpr int64_t kBrcMidMinSide = 9;
// ③ 非对齐 plane 拐点：plane%8!=0 时需 >=485
constexpr int64_t kBrcMidMinPlaneUnaligned = 485;
// ④ 过渡区补充 (plane 100~255 && 对齐8 && B>=1000)
constexpr int64_t kTransitionMinPlane = 100;
constexpr int64_t kTransitionMinBatch = 1000;

// ============================================================
// M∈(2,256) HL(ReduceRepeat) 路由标定常量
// ============================================================
// 场景A: R<=10 时 HL 写放大无法摊薄；扩展 R∈[11,15] & M<=150
constexpr int64_t kHlSmallR = 10;
constexpr int64_t kHlSmallREx = 15;
constexpr int64_t kHlSmallRExMaxM = 150;
// 场景B: p==2.0 且 M<24 时 MAlign 利用率低。
constexpr int64_t kHlP2SmallMMaxM = 24;
// 场景C: M<24 且 M*R<1000（小 MR 区域 HL 劣化）。
constexpr int64_t kHlSmallMRMaxM = 24;
constexpr int64_t kHlSmallMRMaxMR = 1000;
// 场景D: 66<M<70 且 R>=300 且 p==2 且 B<=112（M=67~69 连续劣化带）。
constexpr int64_t kHlMidMLo = 66;
constexpr int64_t kHlMidMHi = 70;
constexpr int64_t kHlMidMMinR = 300;
constexpr int64_t kHlMidMMaxB = 112;
// 场景E: p==1 且 M∈[17,22]（p=1 小 M 大面积劣化）。
constexpr int64_t kHlP1SmallMLo = 17;
constexpr int64_t kHlP1SmallMHi = 22;
// 场景F: 一般 p (p∉{1,2,inf}) 且 M<32 → HL 的 exp/log 链在 FTZ_FALSE 下失去对 SIMT 优势
constexpr int64_t kHlSmallMGenPMaxM = 32;
// 矢量启用主门槛：M>=32，或 M>=16 且 P*R<=262144。
constexpr int64_t kHlVectorMMin = 32;
constexpr int64_t kHlVectorMEx = 16;
constexpr int64_t kHlVectorMaxPR = 262144;

// ============================================================
// UB-BRC / multi-batch 通用常量
// ============================================================
// M 下限：M>=2 才进入 HL 矢量路径（M==1 走独立 broadcast 路由）。
constexpr int64_t kHlMMin = 2;
// 迭代收敛求 elemNum 的最大迭代次数（UB 容量与 aliveBig 的耦合求解）。
constexpr int64_t kUbConvergeMaxIter = 8;
// multi-batch 打包分支标识（brcBranch==2 表示每 tile 打包多个 batch）。
constexpr int64_t kBrcBranchMultiBatch = 2;
// multi-batch 启用门槛：P/R 任一侧上限（P<=64 且 R<=64 才打包，避免大 plane 撑满 UB）。
constexpr int64_t kMultiBatchMaxSide = 64;
// multi-batch 启用门槛：B 下限（B>=128 时打包才摊薄/并行收益，B 小回退 UB-BRC）。
constexpr int64_t kMultiBatchMinBatch = 128;
// multi-batch 收敛迭代上限（收缩 batchPerTile 直到 UB 装得下）。
constexpr int64_t kMultiBatchConvergeMaxIter = 8;
// multi-batch 生效下限：batchPerTile>=2 才有打包意义。
constexpr int64_t kMultiBatchMinBatchPerTile = 2;

ge::graphStatus CdistTiling::CheckParams()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start CheckParams.");
    auto x1 = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, x1);
    auto x1Desc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, x1Desc);
    ge::DataType dtype = x1Desc->GetDataType();
    dtypeSize_ = ge::GetSizeByDataType(dtype);
    auto x2 = tilingContext_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, x2);
    auto y = tilingContext_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, y);
    const gert::Shape& x1Shape = x1->GetStorageShape();
    const gert::Shape& x2Shape = x2->GetStorageShape();
    const gert::Shape& yShape = y->GetStorageShape();
    x1Shape_ = x1Shape;
    x2Shape_ = x2Shape;
    yShape_ = yShape;
    int64_t dimNum = x1Shape_.GetDimNum();
    OP_CHECK_IF(
        dimNum < MIN_DIM_LEN,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "x1", std::to_string(dimNum).c_str(),
                                                 "Input only supports at least 2D tensors"),
        return ge::GRAPH_FAILED);
    int64_t x2DimNum = x2Shape_.GetDimNum();
    if (x2DimNum != dimNum) {
        std::string reasonMsg = "The dim num of x1 and x2 must be the same, x1 got: " + std::to_string(dimNum) +
                                " ,x2 got: " + std::to_string(x2DimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "x2", std::to_string(x2DimNum).c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t yDimNum = yShape_.GetDimNum();
    if (yDimNum != dimNum) {
        std::string reasonMsg = "The dim num of input and output must be the same, x1 got: " + std::to_string(dimNum) +
                                " ,y got: " + std::to_string(yDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "y", std::to_string(yDimNum).c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t M1 = x1Shape_.GetDim(dimNum - 1);
    int64_t M2 = x2Shape_.GetDim(dimNum - 1);
    if (M1 != M2) {
        std::string reasonMsg = "The last dim of x1 and x2 must be the same, x1 got: " + std::to_string(M1) +
                                " ,x2 got: " + std::to_string(M2);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "x2", std::to_string(M2).c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    M_ = M1;
    P_ = x1Shape_.GetDim(dimNum - MIN_DIM_LEN);
    R_ = x2Shape_.GetDim(dimNum - MIN_DIM_LEN);
    if (P_ != yShape_.GetDim(dimNum - MIN_DIM_LEN) || R_ != yShape_.GetDim(dimNum - 1)) {
        std::string reasonMsg = "The last two dims of output are incorrect, output[-1] got: " +
                                std::to_string(yShape_.GetDim(dimNum - 1)) +
                                " ,output[-2] got: " + std::to_string(yShape_.GetDim(dimNum - MIN_DIM_LEN));
        std::string errDimMsg = "output[-1]: " + std::to_string(yShape_.GetDim(dimNum - 1)) +
                                " ,output[-2]: " + std::to_string(yShape_.GetDim(dimNum - MIN_DIM_LEN));
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "y", errDimMsg.c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    tilingData_.p = 2.0f;
    if (attrs->GetAttrNum() > 0) {
        const float* pAttr = attrs->GetAttrPointer<float>(0);
        tilingData_.p = pAttr == nullptr ? 2.0f : *pAttr;
        OP_CHECK_IF(tilingData_.p < 0,
                    OP_LOGE_WITH_INVALID_ATTR(tilingContext_->GetNodeName(), "p", std::to_string(tilingData_.p).c_str(),
                                              "greater than or equal to 0"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CdistTiling::MergeBatchAxis()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start MergeBatchAxis.");
    int64_t dimNum = x1Shape_.GetDimNum();
    B_ = 1;
    int64_t x2B = 1;
    int64_t yB = 1;
    for (int64_t i = 0; i < dimNum - MIN_DIM_LEN; i++) {
        B_ *= x1Shape_.GetDim(i);
        x2B *= x2Shape_.GetDim(i);
        yB *= yShape_.GetDim(i);
    }
    if (B_ != x2B || B_ != yB) {
        std::string reasonMsg = "The batch of input and output must be the same, but x1 got: " + std::to_string(B_) +
                                ", x2 got: " + std::to_string(B_) + ", y got: " + std::to_string(yB);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext_->GetNodeName(), "y", std::to_string(yB).c_str(),
                                                 reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void CdistTiling::DoSimtTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoSimtTiling.");
    int64_t totalElements = B_ * P_ * R_;
    int64_t minPerCoreElement = SIMT_MIN_BYTE / dtypeSize_;
    if (totalElements <= minPerCoreElement) {
        tilingData_.realCoreNum = 1;
        tilingData_.blockFactor = totalElements;
        tilingData_.blockTailFactor = totalElements;
        return;
    } else {
        int64_t minRequiredCores = Ops::Base::CeilDiv(totalElements, minPerCoreElement);
        int64_t usedCoreNum = std::min(coreNum_, minRequiredCores);
        if (usedCoreNum == 0) {
            return;
        }
        int64_t perCoreElement = totalElements / usedCoreNum;
        int64_t tailCoreElement = totalElements - perCoreElement * (usedCoreNum - 1);
        if (perCoreElement < minPerCoreElement) {
            perCoreElement = minPerCoreElement;
            tailCoreElement = totalElements - perCoreElement * (usedCoreNum - 1);
        }
        tilingData_.realCoreNum = usedCoreNum;
        tilingData_.blockFactor = perCoreElement;
        tilingData_.blockTailFactor = tailCoreElement;
    }
}

void CdistTiling::SetDefaultBlockTiling()
{
    tilingData_.blockMainNumB = B_;
    tilingData_.blockMainFactorB = 1;
    tilingData_.blockMainNumP = P_;
    tilingData_.blockMainFactorP = 1;
    tilingData_.blockMainNumR = R_;
    tilingData_.blockMainFactorR = 1;
}

void CdistTiling::DoNormalBlockTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoNormalBlockTiling.");
    // block tiling: B -> P -> R
    SetDefaultBlockTiling();
    int64_t currentCoreNum = coreNum_;
    int64_t remainingCoreNum = currentCoreNum / B_;
    if (currentCoreNum < B_) {
        tilingData_.blockMainNumB = B_ % currentCoreNum == 0 ? currentCoreNum : B_ % currentCoreNum;
        tilingData_.blockTailNumB = currentCoreNum - tilingData_.blockMainNumB;
        tilingData_.blockMainFactorB = tilingData_.blockTailNumB == 0 ? B_ / currentCoreNum : B_ / currentCoreNum + 1;
        tilingData_.blockTailFactorB = tilingData_.blockTailNumB == 0 ? 0 : tilingData_.blockMainFactorB - 1;
    }
    if (remainingCoreNum <= 1) {
        tilingData_.blockMainNumP = 1;
        tilingData_.blockMainFactorP = P_;
        tilingData_.blockMainNumR = 1;
        tilingData_.blockMainFactorR = R_;
        return;
    }
    currentCoreNum = remainingCoreNum;
    remainingCoreNum = currentCoreNum / P_;
    if (currentCoreNum < P_) {
        tilingData_.blockMainNumP = P_ % currentCoreNum == 0 ? currentCoreNum : P_ % currentCoreNum;
        tilingData_.blockTailNumP = currentCoreNum - tilingData_.blockMainNumP;
        tilingData_.blockMainFactorP = tilingData_.blockTailNumP == 0 ? P_ / currentCoreNum : P_ / currentCoreNum + 1;
        tilingData_.blockTailFactorP = tilingData_.blockTailNumP == 0 ? 0 : tilingData_.blockMainFactorP - 1;
    }
    if (remainingCoreNum <= 1) {
        tilingData_.blockMainNumR = 1;
        tilingData_.blockMainFactorR = R_;
        return;
    }
    currentCoreNum = remainingCoreNum;
    remainingCoreNum = currentCoreNum / R_;
    if (currentCoreNum < R_) {
        tilingData_.blockMainNumR = R_ % currentCoreNum == 0 ? currentCoreNum : R_ % currentCoreNum;
        tilingData_.blockTailNumR = currentCoreNum - tilingData_.blockMainNumR;
        tilingData_.blockMainFactorR = tilingData_.blockTailNumR == 0 ? R_ / currentCoreNum : R_ / currentCoreNum + 1;
        tilingData_.blockTailFactorR = tilingData_.blockTailNumR == 0 ? 0 : tilingData_.blockMainFactorR - 1;
    }
}

void CdistTiling::SetDefaultUbTiling()
{
    tilingData_.ubLoopNumM = 1;
    tilingData_.ubFactorM = M_;
    tilingData_.ubTailFactorM = 0;
    tilingData_.ubLoopNumR = 1;
    tilingData_.ubFactorR = tilingData_.blockMainFactorR;
    tilingData_.ubTailFactorR = 0;
    tilingData_.ubLoopNumP = 1;
    tilingData_.ubFactorP = tilingData_.blockMainFactorP;
    tilingData_.ubTailFactorP = 0;
    tilingData_.ubLoopNumB = 1;
    tilingData_.ubFactorB = tilingData_.blockMainFactorB;
    tilingData_.ubTailFactorB = 0;
}

void CdistTiling::ProcessDimension(const DimConfig& config, int64_t availableUbElements, int64_t& findUbTilingIdx)
{
    findUbTilingIdx++;
    int64_t totalElements = config.calcTotalElements(config.baseValue);
    if (totalElements > availableUbElements) {
        for (int64_t i = config.baseValue - 1; i >= 1; i--) {
            int64_t splitElements = config.calcTotalElements(i);
            if (splitElements <= availableUbElements) {
                SetDefaultUbTiling();
                *config.factor = i;
                *config.loopNum = Ops::Base::CeilDiv(config.baseValue, i);
                *config.tailFactor = config.baseValue % i;
                notFoundUbTilingAxis_ = 0;
                break;
            }
        }
    }
}

void CdistTiling::DoNormalUbTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoNormalUbTiling.");
    // Ub tiling: M -> R0 -> P0 -> B0
    int64_t availableUbElements = ubSize_ / dtypeSize_;
    int64_t blockElements = BLOCK_BYTES / dtypeSize_;
    int64_t B0 = tilingData_.blockMainFactorB;
    int64_t P0 = tilingData_.blockMainFactorP;
    int64_t R0 = tilingData_.blockMainFactorR;
    int64_t MBlockAlign = Ops::Base::CeilAlign(M_ * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
    int64_t R0BlockAlign = Ops::Base::CeilAlign(R0 * dtypeSize_, BLOCK_BYTES) / dtypeSize_;

    std::vector<DimConfig> configs;
    if (use_reduce_hl_ == 1) {
        // 方案三 (round3): HL 内核不再有 cast queue —— x1/x2/y 队列直接按 sizeof(float) 分配（fp16 的
        // T-typed staging 复用同一 buffer 的上半部，无额外 UB）。队列字节数 BUFFER_NUM*N*sizeof(float)
        // 换算回 T 元素单位即 BUFFER_NUM*fp32Ratio*N。
        const int64_t fp32Ratio = static_cast<int64_t>(sizeof(float)) / dtypeSize_; // fp16:2, fp32:1
        configs = {{&tilingData_.ubLoopNumM, &tilingData_.ubFactorM, &tilingData_.ubTailFactorM, M_,
                    [this, blockElements, fp32Ratio](int64_t i) {
                        int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                        return BUFFER_NUM * fp32Ratio * (iBlockAlign + iBlockAlign + blockElements) + blockElements;
                    }},
                   {&tilingData_.ubLoopNumR, &tilingData_.ubFactorR, &tilingData_.ubTailFactorR, R0,
                    [this, MBlockAlign, blockElements, fp32Ratio](int64_t i) {
                        int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                        return BUFFER_NUM * fp32Ratio * (MBlockAlign + i * MBlockAlign + iBlockAlign) + blockElements;
                    }},
                   {&tilingData_.ubLoopNumP, &tilingData_.ubFactorP, &tilingData_.ubTailFactorP, P0,
                    [this, MBlockAlign, R0BlockAlign, R0, blockElements, fp32Ratio](int64_t i) {
                        return BUFFER_NUM * fp32Ratio * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) +
                               blockElements;
                    }},
                   {&tilingData_.ubLoopNumB, &tilingData_.ubFactorB, &tilingData_.ubTailFactorB, B0,
                    [this, MBlockAlign, R0BlockAlign, R0, P0, blockElements, fp32Ratio](int64_t i) {
                        return BUFFER_NUM * fp32Ratio *
                                   (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                               blockElements;
                    }}};
    } else if (dtypeSize_ == B4) {
        configs = {{&tilingData_.ubLoopNumM, &tilingData_.ubFactorM, &tilingData_.ubTailFactorM, M_,
                    [this, blockElements](int64_t i) {
                        int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                        return BUFFER_NUM * (iBlockAlign + iBlockAlign + blockElements) + blockElements;
                    }},
                   {&tilingData_.ubLoopNumR, &tilingData_.ubFactorR, &tilingData_.ubTailFactorR, R0,
                    [this, MBlockAlign, blockElements](int64_t i) {
                        int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                        return BUFFER_NUM * (MBlockAlign + i * MBlockAlign + iBlockAlign) + blockElements;
                    }},
                   {&tilingData_.ubLoopNumP, &tilingData_.ubFactorP, &tilingData_.ubTailFactorP, P0,
                    [this, MBlockAlign, R0BlockAlign, R0, blockElements](int64_t i) {
                        return BUFFER_NUM * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) + blockElements;
                    }},
                   {&tilingData_.ubLoopNumB, &tilingData_.ubFactorB, &tilingData_.ubTailFactorB, B0,
                    [this, MBlockAlign, R0BlockAlign, R0, P0, blockElements](int64_t i) {
                        return BUFFER_NUM * (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                               blockElements;
                    }}};
    } else {
        configs = {
            {&tilingData_.ubLoopNumM, &tilingData_.ubFactorM, &tilingData_.ubTailFactorM, M_,
             [this, blockElements](int64_t i) {
                 int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                 return BUFFER_NUM * (iBlockAlign + iBlockAlign + blockElements) +
                        CAST_BUFFER_RATIO * (iBlockAlign + iBlockAlign + blockElements) + blockElements;
             }},
            {&tilingData_.ubLoopNumR, &tilingData_.ubFactorR, &tilingData_.ubTailFactorR, R0,
             [this, MBlockAlign, blockElements](int64_t i) {
                 int64_t iBlockAlign = Ops::Base::CeilAlign(i * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
                 return BUFFER_NUM * (MBlockAlign + i * MBlockAlign + iBlockAlign) +
                        CAST_BUFFER_RATIO * (MBlockAlign + i * MBlockAlign + iBlockAlign) + blockElements;
             }},
            {&tilingData_.ubLoopNumP, &tilingData_.ubFactorP, &tilingData_.ubTailFactorP, P0,
             [this, MBlockAlign, R0BlockAlign, R0, blockElements](int64_t i) {
                 return BUFFER_NUM * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) +
                        CAST_BUFFER_RATIO * (i * MBlockAlign + R0 * MBlockAlign + i * R0BlockAlign) + blockElements;
             }},
            {&tilingData_.ubLoopNumB, &tilingData_.ubFactorB, &tilingData_.ubTailFactorB, B0,
             [this, MBlockAlign, R0BlockAlign, R0, P0, blockElements](int64_t i) {
                 return BUFFER_NUM * (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                        CAST_BUFFER_RATIO * (i * P0 * MBlockAlign + i * R0 * MBlockAlign + i * P0 * R0BlockAlign) +
                        blockElements;
             }}};
    }

    SetDefaultUbTiling();
    int64_t findUbTilingIdx = 0;
    for (auto& config : configs) {
        if (notFoundUbTilingAxis_) {
            ProcessDimension(config, availableUbElements, findUbTilingIdx);
        }
    }
    switch (findUbTilingIdx) {
        case 1:
            tilingData_.ubLoopNumR = tilingData_.blockMainFactorR;
            tilingData_.ubFactorR = 1;
            tilingData_.ubLoopNumP = tilingData_.blockMainFactorP;
            tilingData_.ubFactorP = 1;
            tilingData_.ubLoopNumB = tilingData_.blockMainFactorB;
            tilingData_.ubFactorB = 1;
            break;
        case 2:
            tilingData_.ubLoopNumP = tilingData_.blockMainFactorP;
            tilingData_.ubFactorP = 1;
            tilingData_.ubLoopNumB = tilingData_.blockMainFactorB;
            tilingData_.ubFactorB = 1;
            break;
        case 3:
            tilingData_.ubLoopNumB = tilingData_.blockMainFactorB;
            tilingData_.ubFactorB = 1;
            break;
        default:
            break;
    }
}

void CdistTiling::DoNormalTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoNormalTiling.");
    DoNormalBlockTiling();
    tilingData_.realCoreNum = (tilingData_.blockMainNumB + tilingData_.blockTailNumB) *
                              (tilingData_.blockMainNumP + tilingData_.blockTailNumP) *
                              (tilingData_.blockMainNumR + tilingData_.blockTailNumR);
    DoNormalUbTiling();
}

void CdistTiling::DoTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start DoTiling.");
    // 方案2: M==1 命中 Broadcast 快路径（OneDim ④ / 静态 rank UB-BRC ③ / multi-batch）；其余走原 SIMT/Normal。
    if (M_ == 1) {
        // 路由谓词：
        //   ① OneDim (plane==1)：broadcast 快 ~11 倍
        //   ② UB-BRC (P>64 或 R>64)：分块处理任意大 plane
        //   ③ 中小 plane (P,R<=64)：plane>=256 且 min(P,R)>=9 后 broadcast 净赢；
        //      对齐 plane(plane%8==0) 直接放行；
        //      非对齐 plane 需 >=485；
        //      min(P,R)>=9 排除扁长劣化；
        //      plane<256 全输 <100:17%、100-255:26%，回退 SIMT。
        //   ④ 过渡区补充 (plane 100~255 && 对齐8 && B>=1000)。
        int64_t plane = P_ * R_;
        int64_t minPr = (P_ < R_) ? P_ : R_;
        bool planeAligned8 = (plane % 8 == 0);
        bool brcWins = (plane == kOneDimPlane) || (P_ > kUbBrcMinSide || R_ > kUbBrcMinSide) ||
                       (plane >= kBrcMidMinPlane && minPr >= kBrcMidMinSide &&
                        (planeAligned8 || plane >= kBrcMidMinPlaneUnaligned)) ||
                       (plane >= kTransitionMinPlane && planeAligned8 && B_ >= kTransitionMinBatch);
        if (brcWins) {
            use_broadcast_ = 1;
            is_small_m_ = 1; // fallback flag; not launched when use_broadcast_ set
            DoSimtTiling();  // compute SIMT tiling as safety fallback
            DoBrcTiling();
        } else {
            use_broadcast_ = 0;
            is_small_m_ = 1; // SIMT 标量路径 (KERNEL_MODE=1)
            DoSimtTiling();
        }
        return;
    }
    // 方案三 (round2 crossover): M∈[2,256] broadcast+高层 ReduceRepeat 归约内核（复用 Normal 的 block/UB 切分）
    // 而非 SIMT 标量路径 —— 仅在矢量胜出时。谓词按 msprof 实测标定：矢量 if M>=32 || (M>=16 && P*R<=262144)，
    // 否则（小 M 或 M∈[16,32) 且 P*R 巨大）走 SIMT。
    // M==1 走 Broadcast 快路径（上面已 return），M>256 仍走 Normal(VF)（下面 else）。
    bool isInf = std::isinf(tilingData_.p);
    // 路由谓词优化:
    // 场景A: R<=10 时 HL broadcast 写放大无法摊薄, SIMT 标量更优;
    //        扩展 R∈[11,15] & M<=150
    // 场景B: p==2.0 且 M<24 时 MAlign 利用率低, SIMT 更优
    // 场景C: M<24 且 M*R<1000 —— 小 MR 区域 HL 劣化
    // 场景D: 66<M<70 且 R>=300 且 p==2 且 B<=112 —— M=67~69 连续劣化带
    // 场景E: p==1 且 M∈[17,22] —— p=1 小 M 大面积劣化
    bool smallRBad = (R_ <= kHlSmallR) || (R_ <= kHlSmallREx && M_ <= kHlSmallRExMaxM);
    bool p2SmallMBad = (tilingData_.p == 2.0f) && (M_ < kHlP2SmallMMaxM);
    bool smallMRBad = (M_ < kHlSmallMRMaxM) && (M_ * R_ < kHlSmallMRMaxMR);
    bool midMBad = (M_ > kHlMidMLo && M_ < kHlMidMHi) && (R_ >= kHlMidMMinR) && (tilingData_.p == 2.0f) &&
                   (B_ <= kHlMidMMaxB);
    bool p1SmallMBad = (tilingData_.p == 1.0f) && (M_ >= kHlP1SmallMLo && M_ <= kHlP1SmallMHi);
    bool genpSmallMBad = (!isInf && tilingData_.p != 1.0f && tilingData_.p != 2.0f && M_ < kHlSmallMGenPMaxM);
    bool hlVectorWins = !isInf && !smallRBad && !p2SmallMBad && !smallMRBad && !midMBad && !p1SmallMBad &&
                        !genpSmallMBad &&
                        ((M_ >= kHlVectorMMin) || (M_ >= kHlVectorMEx && (P_ * R_) <= kHlVectorMaxPR));
    if (M_ >= kHlMMin && M_ <= M_SIZE && hlVectorWins) {
        is_small_m_ = 0;    // launch on the Normal-shaped tiling
        use_reduce_hl_ = 1; // but dispatch to the high-level reduce kernel (KERNEL_MODE=3)
        // Reserve UB for the two fixed HL fp32 compute planes (x1exp + diff, 4096 elems each = 32KB) plus
        // the fp32 M-split accumulator margin, so the extra planes never overflow UB on large P·R tiles.
        int64_t savedUb = ubSize_;
        ubSize_ = ubSize_ - HL_UB_RESERVE;
        if (ubSize_ < BLOCK_BYTES)
            ubSize_ = BLOCK_BYTES;
        DoNormalTiling();
        ubSize_ = savedUb;
        CapUbFactorRForHL();
    } else if (M_ <= M_SIZE) {
        is_small_m_ = 1;
        DoSimtTiling();
    } else {
        is_small_m_ = 0;
        DoNormalTiling();
    }
}

// 方案三 (round2): cap ubFactorR so the HL kernel processes each R UB-tile in ONE broadcast/reduce pass
// (no internal R-chunk). 逐行照抄直调 cdist_host_tiling.cpp CapUbFactorRForHL 363-385 行，只把裸接口换成
// register 侧同名成员（dtypeSize_ / tilingData_ 字段名与 Normal 一致）。
void CdistTiling::CapUbFactorRForHL()
{
    // MAlign used by the HL kernel for the broadcast/reduce plane: for ubLoopNumM==1 it's M_, else ubFactorM.
    int64_t mForAlign = (tilingData_.ubLoopNumM == 1) ? M_ : tilingData_.ubFactorM;
    int64_t mAlign = Ops::Base::CeilAlign(mForAlign * dtypeSize_, BLOCK_BYTES) / dtypeSize_;
    if (mAlign < 1)
        mAlign = 1;
    int64_t maxUbR = HL_PLANE_ELEMS / mAlign;
    // Align the cap down to a 32B row multiple so DataCopyPad / reduce strides stay block-aligned.
    int64_t alignEle = BLOCK_BYTES / dtypeSize_;
    maxUbR = (maxUbR / alignEle) * alignEle;
    if (maxUbR < alignEle)
        maxUbR = alignEle; // at least one aligned row-group

    int64_t curUbR = tilingData_.ubFactorR;
    if (curUbR <= maxUbR) {
        return; // already fits in one pass
    }
    // Shrink ubFactorR to maxUbR and recompute the R UB-loop over the per-core R factor (blockMainFactorR).
    int64_t r0 = tilingData_.blockMainFactorR;
    tilingData_.ubFactorR = maxUbR;
    tilingData_.ubLoopNumR = Ops::Base::CeilDiv(r0, maxUbR);
    tilingData_.ubTailFactorR = r0 % maxUbR; // 0 => last chunk is a full ubFactorR (kernel derives via subtraction)
}

// 方案2: 填充 Broadcast fast path tiling
void CdistTiling::DoBrcTiling()
{
    CdistBrcTilingData& t = brcTilingData_;
    t.B = B_;
    t.P = P_;
    t.R = R_;
    t.p = tilingData_.p;

    const int64_t alignEle = BLOCK_BYTES / dtypeSize_; // elements per 32B block (dtype T)

    if (P_ == 1 && R_ == 1) {
        // ---- OneDim (④): output collapses to 1D length B ----
        t.brcBranch = 0;
        int64_t dimLen = B_;
        // aliveBuf: x1(DB)+x2(DB)+out(DB) T-planes + fp32 scratch(x1f,x2f,yfp). Conservative => 6.
        const int64_t aliveBuf = 6;
        int64_t tileNum = (ubSize_ / (aliveBuf * dtypeSize_ * BUFFER_NUM));
        tileNum = (tileNum / alignEle) * alignEle;
        if (tileNum < alignEle)
            tileNum = alignEle;
        int64_t ubOuter = Ops::Base::CeilDiv(dimLen, tileNum);
        int64_t cn = coreNum_;
        if (ubOuter < cn)
            cn = (ubOuter < 1) ? 1 : ubOuter;
        t.dimLen = dimLen;
        t.tileNum = static_cast<int32_t>(tileNum);
        t.blockNum = static_cast<int32_t>(cn);
        return;
    }

    // ---- UB-BRC (③) static rank=2 : tile = (rowsP, R) plane per batch ----
    t.brcBranch = 1;
    // UB 必须容纳所有 InitBuffer 分配：
    //   qLoad1_(2*src1Ele*T) + qLoad2_(2*src2Ele*T) + bufSrc1_(src1Ele*4) + bufSrc2_(src2Ele*4)
    //   + bufX1Exp_(elemNum*4) + bufX2Exp_(elemNum*4) + qY_(2*elemNum*T)
    // 其中 src1Ele=AlignUp(ubFormerP,alignT), src2Ele=AlignUp(rSeg,alignT)，ubFormerP/rSeg 依赖 elemNum。
    // 用迭代收敛求 elemNum。
    const int64_t fp32Bytes = 4;
    int64_t alignFp = BLOCK_BYTES / fp32Bytes; // 8
    int64_t alignT = BLOCK_BYTES / dtypeSize_;
    // 大 buffer 的 fp32-plane 等效数：x1exp(1) + x2exp(1) + qY DB(2*dtypeSize/4)
    int64_t aliveBig = 2 + 2 * dtypeSize_ / fp32Bytes;
    // GetRuntimeUBSize() = get_shmem_sz() - 8KB，InitBuffer 可用 UB 须扣除系统保留区。
    constexpr int64_t UB_RESERVED_FOR_SYSTEM = 8 * 1024;
    int64_t usableUb = ubSize_ - UB_RESERVED_FOR_SYSTEM;
    if (usableUb < alignFp * aliveBig * fp32Bytes)
        usableUb = alignFp * aliveBig * fp32Bytes;
    int64_t elemNum = usableUb / (aliveBig * fp32Bytes);
    elemNum = (elemNum / alignFp) * alignFp;
    if (elemNum < alignFp)
        elemNum = alignFp;
    for (int iter = 0; iter < kUbConvergeMaxIter; ++iter) {
        int64_t rSegTmp, rowsPTmp;
        if (R_ <= elemNum) {
            rSegTmp = R_;
            rowsPTmp = elemNum / R_;
        } else {
            rowsPTmp = 1;
            rSegTmp = (elemNum / alignFp) * alignFp;
        }
        if (rowsPTmp < 1)
            rowsPTmp = 1;
        int64_t s1 = ((rowsPTmp + alignT - 1) / alignT) * alignT;
        int64_t s2 = ((rSegTmp + alignT - 1) / alignT) * alignT;
        int64_t bigBytes = aliveBig * elemNum * fp32Bytes;
        int64_t srcBytes = (2 * dtypeSize_ + fp32Bytes) * (s1 + s2);
        if (bigBytes + srcBytes <= usableUb)
            break;
        int64_t availForBig = usableUb - srcBytes;
        if (availForBig < alignFp * aliveBig * fp32Bytes) {
            elemNum = alignFp;
            break;
        }
        int64_t newElem = (availForBig / (aliveBig * fp32Bytes) / alignFp) * alignFp;
        if (newElem < alignFp)
            newElem = alignFp;
        if (newElem >= elemNum)
            break;
        elemNum = newElem;
    }

    // 缺陷② 修复：小 tile 场景（whole (P,R) plane 装得下 UB，且 P*R 小到可打包 >=2 个 batch）走
    // multi-batch(brcBranch==2)，把每 tile 只处理 1 batch 的 256 元素小 tile 打包成每 tile 多 batch
    {
        int64_t plane = P_ * R_;                   // 单 batch 输出元素数
        const int64_t alignFp32 = BLOCK_BYTES / 4; // fp32 32B = 8 元素
        // 打包需求：整块 (P,R) 装得下 UB（R<=elemNum 且 P*R<=elemNum），且能装 >=2 个 batch；
        // 且 plane 必须 32B(fp32 8 元素)对齐——否则逐 batch Broadcast 的 dst 局部偏移 j*plane 未对齐，
        // 非对齐 plane 回落 brcBranch==1 (原正确 UB-BRC 路径)。181 plane=256 对齐 → 走快路径。
        // B 极小(<128)时回退 UB-BRC：multi-batch 每 tile 打包 batchPerTile=min(elemNum/plane,B) 个 batch，
        // B 小意味着 plane 大(plane>=485 才进 broadcast)，UB 被少数大 plane 撑满，且 B 小时 multi-batch
        // 摊薄/并行收益消失(实测与 SIMT 持平)，回退 UB-BRC 更安全。
        if (plane > 0 && (plane % alignFp32 == 0) && R_ <= elemNum && plane * kMultiBatchMinBatchPerTile <= elemNum &&
            P_ <= kMultiBatchMaxSide && R_ <= kMultiBatchMaxSide && B_ >= kMultiBatchMinBatch) {
            int64_t batchPerTile = elemNum / plane; // 每 tile 打包的 batch 数
            if (batchPerTile > B_)
                batchPerTile = B_;
            // UB 检查：brcBranch==2 的 src buffer 随 batchPerTile 增长，需验证不超 UB
            for (int gIter = 0; gIter < kMultiBatchConvergeMaxIter && batchPerTile >= kMultiBatchMinBatchPerTile;
                 ++gIter) {
                int64_t g = batchPerTile;
                int64_t s1 = ((g * P_ + alignT - 1) / alignT) * alignT;
                int64_t s2 = ((g * R_ + alignT - 1) / alignT) * alignT;
                int64_t bigBytes2 = aliveBig * elemNum * fp32Bytes;
                int64_t srcBytes2 = (2 * dtypeSize_ + fp32Bytes) * (s1 + s2);
                if (bigBytes2 + srcBytes2 <= usableUb)
                    break;
                int64_t overshoot = bigBytes2 + srcBytes2 - usableUb;
                int64_t shrinkG = (overshoot / ((2 * dtypeSize_ + fp32Bytes) * (P_ + R_))) + 1;
                batchPerTile -= shrinkG;
            }
            if (batchPerTile >= kMultiBatchMinBatchPerTile) {
                t.brcBranch = kBrcBranchMultiBatch;
                t.B = B_;
                t.P = P_;
                t.R = R_;
                t.elemNum = elemNum;
                t.batchPerTile = batchPerTile;
                int64_t totalTiles2 = Ops::Base::CeilDiv(B_, batchPerTile);
                int64_t cn2 = coreNum_ < 1 ? 1 : coreNum_;
                int64_t blockFormer2 = Ops::Base::CeilDiv(totalTiles2, cn2);
                if (blockFormer2 < 1)
                    blockFormer2 = 1;
                int64_t coreNumT2 = Ops::Base::CeilDiv(totalTiles2, blockFormer2);
                int64_t blockTail2 = totalTiles2 - (coreNumT2 - 1) * blockFormer2;
                t.totalTiles = totalTiles2;
                t.blockFormerT = blockFormer2;
                t.coreNumT = coreNumT2;
                t.blockTailT = blockTail2;
                return;
            }
        }
    }

    int64_t rSeg, rOuter, rTail;
    int64_t ubFormerP, ubOuterP, ubTailP;
    if (R_ <= elemNum) {
        rSeg = R_;
        rOuter = 1;
        rTail = R_;
        int64_t rowsP = elemNum / R_;
        if (rowsP < 1)
            rowsP = 1;
        if (rowsP > P_)
            rowsP = P_;
        ubFormerP = rowsP;
        ubOuterP = Ops::Base::CeilDiv(P_, rowsP);
        ubTailP = P_ - (ubOuterP - 1) * rowsP;
    } else {
        ubFormerP = 1;
        ubOuterP = P_;
        ubTailP = 1;
        rSeg = (elemNum / alignFp) * alignFp;
        if (rSeg < alignFp)
            rSeg = alignFp;
        rOuter = Ops::Base::CeilDiv(R_, rSeg);
        rTail = R_ - (rOuter - 1) * rSeg;
    }
    t.elemNum = elemNum;
    t.ubFormerP = ubFormerP;
    t.ubOuterP = ubOuterP;
    t.ubTailP = ubTailP;
    t.rSeg = rSeg;
    t.rOuter = rOuter;
    t.rTail = rTail;

    int64_t totalTiles = B_ * ubOuterP * rOuter;
    int64_t cn = coreNum_ < 1 ? 1 : coreNum_;
    int64_t blockFormer = Ops::Base::CeilDiv(totalTiles, cn);
    if (blockFormer < 1)
        blockFormer = 1;
    int64_t coreNumT = Ops::Base::CeilDiv(totalTiles, blockFormer);
    int64_t blockTail = totalTiles - (coreNumT - 1) * blockFormer;
    t.totalTiles = totalTiles;
    t.blockFormerT = blockFormer;
    t.coreNumT = coreNumT;
    t.blockTailT = blockTail;
}

int64_t CdistTiling::GetBrcBlockDim() const
{
    if (brcTilingData_.brcBranch == 0) {
        return brcTilingData_.blockNum < 1 ? 1 : brcTilingData_.blockNum;
    }
    // brcBranch 1 & 2 both use coreNumT.
    return brcTilingData_.coreNumT < 1 ? 1 : brcTilingData_.coreNumT;
}

void CdistTiling::PrintBrcTilingData() const
{
    const CdistBrcTilingData& t = brcTilingData_;
    std::stringstream ss;
    ss << " brcBranch: " << t.brcBranch << " B: " << t.B << " P: " << t.P << " R: " << t.R << " p: " << t.p
       << " dimLen: " << t.dimLen << " tileNum: " << t.tileNum << " blockNum: " << t.blockNum
       << " elemNum: " << t.elemNum << " ubFormerP: " << t.ubFormerP << " ubOuterP: " << t.ubOuterP
       << " ubTailP: " << t.ubTailP << " rSeg: " << t.rSeg << " rOuter: " << t.rOuter << " rTail: " << t.rTail
       << " batchPerTile: " << t.batchPerTile << " totalTiles: " << t.totalTiles << " blockFormerT: " << t.blockFormerT
       << " coreNumT: " << t.coreNumT << " blockTailT: " << t.blockTailT;
    OP_LOGI(tilingContext_->GetNodeName(), "CdistBrcTilingData: %s", ss.str().c_str());
}

ge::graphStatus CdistTiling::RunCdistTiling()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start RunCdistTiling.");
    OP_CHECK_IF(CheckParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(), "RunCdistTiling check params failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(MergeBatchAxis() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(), "RunCdistTiling merge batch axis failed!"),
                return ge::GRAPH_FAILED);
    DoTiling();
    OP_CHECK_IF(SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext_->GetNodeName(), "RunCdistTiling failed to set tiling data!"),
                return ge::GRAPH_FAILED);
    if (use_broadcast_) {
        PrintBrcTilingData();
    } else {
        PrintTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CdistTiling::Init()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start init CdistTiling.");
    auto compileInfo = reinterpret_cast<const CdistCompileInfo*>(tilingContext_->GetCompileInfo());
    if (compileInfo != nullptr) {
        coreNum_ = compileInfo->coreNum;
        ubSize_ = compileInfo->ubSize;
    } else {
        // Fallback: nnopbase single-op path may not invoke TilingParse (compile info null). Read the
        // platform directly from the tiling context (same source TilingParse would use).
        auto platformInfo = tilingContext_->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = static_cast<int64_t>(ubSize);
    }
    OP_CHECK_IF((coreNum_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(tilingContext_->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_->GetNodeName(), "Init CdistTiling success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CdistTiling::SetTilingData()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Start SetTilingData.");
    auto ptrTilingData = tilingContext_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrTilingData);
    auto capSize = ptrTilingData->GetCapacity();
    void* ptrData = ptrTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrData);
    size_t* workspaces = tilingContext_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, workspaces);
    workspaces[0] = WORK_SPACE_SIZE;

    // 方案2: M==1 命中 Broadcast 快路径 → 写 CdistBrcTilingData + SetTilingKey(KERNEL_MODE=2)。
    if (use_broadcast_) {
        void* ptrBrc = static_cast<void*>(&brcTilingData_);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrBrc);
        OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrBrc, sizeof(brcTilingData_)) != 0,
                    OP_LOGE(tilingContext_->GetNodeName(), "Set brc tiling data failed!"), return ge::GRAPH_FAILED);
        ptrTilingData->SetDataSize(sizeof(brcTilingData_));
        tilingContext_->SetBlockDim(GetBrcBlockDim());
        const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(2)); // KERNEL_MODE == 2
        tilingContext_->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    // 原路径（M>1）：Normal(is_small_m_==0 -> KERNEL_MODE 0) / SIMT(is_small_m_==1 -> KERNEL_MODE 1)。
    tilingData_.B = B_;
    tilingData_.P = P_;
    tilingData_.R = R_;
    tilingData_.M = M_;
    void* ptrStruct = static_cast<void*>(&tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData_)) != 0,
                OP_LOGE(tilingContext_->GetNodeName(), "Set tiling data failed!"), return ge::GRAPH_FAILED);
    ptrTilingData->SetDataSize(sizeof(tilingData_));

    tilingContext_->SetBlockDim(tilingData_.realCoreNum);
    // 方案三 (round2): use_reduce_hl_ 命中 → 数据仍是 Normal 主结构 CdistTilingData（DoNormalTiling 已填好），
    // block dim 用 Normal 的 realCoreNum（同 mode0），仅 tilingKey 派发到 KERNEL_MODE=3 的高层 reduce 内核。
    // 与 use_broadcast_ 互斥（M==1 前者 return / M∈[2,256]矢量胜 use_reduce_hl_ / 其余 is_small_m_ 0|1）。
    const uint64_t tilingKey = use_reduce_hl_ ? GET_TPL_TILING_KEY(static_cast<uint64_t>(3)) :
                                                GET_TPL_TILING_KEY(static_cast<uint64_t>(is_small_m_));
    tilingContext_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

void CdistTiling::PrintTilingData()
{
    std::stringstream ss;
    ss << " realCoreNum: " << tilingData_.realCoreNum << " blockFactor: " << tilingData_.blockFactor
       << " blockTailFactor: " << tilingData_.blockTailFactor << " B: " << tilingData_.B << " P: " << tilingData_.P
       << " R: " << tilingData_.R << " M: " << tilingData_.M << " blockMainNumB: " << tilingData_.blockMainNumB
       << " blockTailNumB: " << tilingData_.blockTailNumB << " blockMainFactorB: " << tilingData_.blockMainFactorB
       << " blockTailFactorB: " << tilingData_.blockTailFactorB << " blockMainNumP: " << tilingData_.blockMainNumP
       << " blockTailNumP: " << tilingData_.blockTailNumP << " blockMainFactorP: " << tilingData_.blockMainFactorP
       << " blockTailFactorP: " << tilingData_.blockTailFactorP << " blockMainNumR: " << tilingData_.blockMainNumR
       << " blockTailNumR: " << tilingData_.blockTailNumR << " blockMainFactorR: " << tilingData_.blockMainFactorR
       << " blockTailFactorR: " << tilingData_.blockTailFactorR << " ubLoopNumB: " << tilingData_.ubLoopNumB
       << " ubFactorB: " << tilingData_.ubFactorB << " ubTailFactorB: " << tilingData_.ubTailFactorB
       << " ubLoopNumP: " << tilingData_.ubLoopNumP << " ubFactorP: " << tilingData_.ubFactorP
       << " ubTailFactorP: " << tilingData_.ubTailFactorP << " ubLoopNumR: " << tilingData_.ubLoopNumR
       << " ubFactorR: " << tilingData_.ubFactorR << " ubTailFactorR: " << tilingData_.ubTailFactorR
       << " ubLoopNumM: " << tilingData_.ubLoopNumM << " ubFactorM: " << tilingData_.ubFactorM
       << " ubTailFactorM: " << tilingData_.ubTailFactorM << " p: " << tilingData_.p;
    OP_LOGI(tilingContext_->GetNodeName(), "CdistTilingData: %s", ss.str().c_str());
}

static ge::graphStatus TilingParseForCdist([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingParseForCdist");
    OP_CHECK_IF(context == nullptr, OP_LOGE("TilingParseForCdist", "TilingParseContext is nullptr!"),
                return ge::GRAPH_FAILED);
    auto compileInfo = context->GetCompiledInfo<CdistCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context->GetNodeName(), "Get hardwareInfo failed, coreNum:%ld.", compileInfo->coreNum),
                return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE(context->GetNodeName(), "Get hardwareInfo failed, ubSize:%ld.", compileInfo->ubSize),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "Get coreNum:%ld, ubSize:%ld.", compileInfo->coreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CdistTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start CdistTilingFunc.");
    CdistTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunCdistTiling();
}

IMPL_OP_OPTILING(Cdist).Tiling(CdistTilingFunc).TilingParse<CdistCompileInfo>(TilingParseForCdist);
} // namespace optiling
