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
 * \file pad_v3_grad_mirror_normal.h
 * \brief PadV3Grad backward kernel for reflect/symmetric mode with small last dimension
 *
 * 反向切分策略说明:
 *   反向传播中，输入是 grad_y (大张量，对应正向的 inShape)，输出是 grad_x (小张量，对应正向的 outShape)。
 *
 *   切分策略 (与正向相反):
 *   - 找切分轴时: 按 inShape (grad_y) 累乘，找到第一个超过 bufferSize 的轴
 *   - 计算 ubFactor 时: 限制不超过 outShape[ubAxis_] (grad_x)
 *   - 计算 outTileSize 时: 用 inShape 计算实际搬运的数据量
 *   - 分核计算: 按 outShape (grad_x) 计算总 UB 块数
 *
 * axisNumInUb_ 含义说明:
 *   axisNumInUb_ = dimNum_ - ubAxis，表示从 ubAxis_ 到最后一维的维度数量。
 *   这决定了 UB 内数据的维度结构，但不决定哪些维度需要处理 padding 梯度累加。
 *
 *   所有有 padding 的维度都需要处理梯度累加：
 *   - W 、H、C 维度: 在 GradGatherProcess 中处理
 *
 *   实际 UB 内的数据大小由 tiling 参数决定:
 *   - ubAxis_: 切分的维度索引 (按 inShape 累乘确定)
 *   - ubFactor: 每次处理的数量 (限制不超过 outShape[ubAxis_])
 *   - inCopyLen_[i]: 各维度实际拷贝的长度 (按 inShape 计算)
 */

#ifndef PAD_V3_GRAD_MIRR_NORMAL_H_
#define PAD_V3_GRAD_MIRR_NORMAL_H_

#include "kernel_operator.h"
#include "pad_v3_grad_struct.h"
#include "pad_v3_grad_common.h"

namespace PadV3Grad {
using namespace AscendC;

constexpr static int64_t MIN_DIM_FOR_H_PAD = 2;
constexpr static int64_t MIN_DIM_FOR_C_PAD = 3;
constexpr static int64_t MIN_DIM_FOR_N_PAD = 4;
constexpr static int32_t MIN_DIM_FOR_D5_PAD = 5;

constexpr static int32_t CAST_SPACE_MULTIPLIER = 2;
constexpr static int32_t MIRROR_BOUNDARY_OFFSET_1 = 1;
constexpr static int32_t MIRROR_BOUNDARY_OFFSET_2 = 2;

struct PadGradNormalParam {
    uint32_t padWI; // grad_y W (padded)
    uint32_t padWO; // grad_x W (original)
    uint32_t padLeft;
    uint32_t padRight;
};

// 镜像位置列表，最多包含 2 个位置（上镜像和下镜像）
struct MirrorList {
    uint32_t mirrors[2]; // 镜像位置数组
    uint8_t count;       // 有效镜像位置数量
};

// 镜像条件结果，包含上下镜像的有效性和镜像位置
struct MirrorCondition {
    bool hasTop;           // 是否存在上/左镜像
    bool hasBottom;        // 是否存在下/右镜像
    uint32_t mirrorTop;    // 上镜像位置 (仅当 hasTop=true 时有效)
    uint32_t mirrorBottom; // 下镜像位置 (仅当 hasBottom=true 时有效)
};

template <typename T, uint8_t modeName>
class KernelPadV3GradMirrWithNormalWidth {
private:
    static constexpr uint32_t BLK_ELEMS = UB_BLOCK / sizeof(T);
    static constexpr uint32_t VL_ELEMS = VL_SIZE / sizeof(T);

    // axisNumInUb_: 从 ubAxis_ 到最后一维的维度数量 (2, 3, 4)
    // axisNumInUb_ = dimNum_ - ubAxis，决定 UB 内数据的维度结构
    // 注意: 这不决定哪些维度需要处理 padding，所有有 padding 的维度都需要处理
    uint32_t axisNumInUb_{0};
    static constexpr uint32_t MODE = (modeName == 2) ? 1 : 2; // modeName=2→reflect(1), otherwise→symmetric(2)
    GlobalTensor<T> input_;                                   // grad_y (padded gradient input)
    GlobalTensor<T> output_;                                  // grad_x (original gradient output)
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_; // 用于存储高维镜像累加结果和从 GM 读取的临时数据

    TPipe* pipe_ = nullptr;
    int64_t blockIdx_;
    uint32_t inTileSize_{0};
    uint32_t outTileSize_{0};
    uint32_t tmpBufTileSize_{0};
    uint8_t dimNum_{0};
    uint8_t ubAxis_{0};
    uint16_t modeOffset_{0}; // reflect=0, symmetric=1
    PadGradNormalParam padParam_;
    const PadV3GradACTilingData* tilingData_ = nullptr;
    bool has2DPadding{false};   // H 维度是否有 padding (dimNum_ >= 2)
    bool has3DPadding{false};   // C 维度是否有 padding (dimNum_ >= 3)
    bool has4DPadding{false};   // N 维度是否有 padding (dimNum_ >= 4)
    bool has5DPadding{false};   // 第5维是否有 padding (dimNum_ >= 5)
    uint32_t padWInLength_{0};  // grad_y W aligned
    uint32_t padWOutLength_{0}; // grad_x W aligned
    // inIndex_[i]: grad_y 中当前块的起始索引
    uint64_t inIndex_[PAD_GRAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};
    // inCopyLen_[i]: 各维度实际拷贝的长度，按 inShape (grad_y) 计算
    // UB 内完整维度: inCopyLen_[i] = inShape[i]
    // 切分轴: inCopyLen_[ubAxis_] = min(ubFactor, outShape[ubAxis_] - outIndex[ubAxis_])
    uint32_t inCopyLen_[PAD_GRAD_MAX_DIMS_NUM] = {1, 1, 1, 1, 1, 1, 1, 1};
    // outIndex_[i]: grad_x 中当前块的起始索引
    uint64_t outIndex_[PAD_GRAD_MAX_DIMS_NUM] = {0, 0, 0, 0, 0, 0, 0, 0};

public:
    __aicore__ inline KernelPadV3GradMirrWithNormalWidth(TPipe* pipe, const PadV3GradACTilingData* tilingData)
    {
        pipe_ = pipe;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR grad_y, GM_ADDR grad_x)
    {
        blockIdx_ = GetBlockIdx();
        inTileSize_ = tilingData_->outTileSize * sizeof(T);
        outTileSize_ = tilingData_->outTileSize * sizeof(T);
        if constexpr (IsSameType<T, PromoteDataT>::value) {
            tmpBufTileSize_ = tilingData_->outTileSize * sizeof(T);
        } else {
            tmpBufTileSize_ = CAST_SPACE_MULTIPLIER * tilingData_->outTileSize * sizeof(T);
        }
        input_.SetGlobalBuffer((__gm__ T*)grad_y);
        output_.SetGlobalBuffer((__gm__ T*)grad_x);

        pipe_->InitBuffer(inQueue_, BUFFER_NUM, inTileSize_);
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, outTileSize_);
        pipe_->InitBuffer(tmpBuf_, tmpBufTileSize_);

        dimNum_ = tilingData_->dimNum;
        ubAxis_ = tilingData_->ubAxis;
        modeOffset_ = MODE <= 1 ? 0 : 1;
        axisNumInUb_ = dimNum_ - ubAxis_;
        // grad_y W (padded) - aligned
        padWInLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->inShape[dimNum_ - 1]), BLK_ELEMS);
        // grad_x W (original) - aligned
        padWOutLength_ = CeilAlign(static_cast<uint32_t>(tilingData_->outShape[dimNum_ - 1]), BLK_ELEMS);

        padParam_.padWI = padWInLength_;  // grad_y W
        padParam_.padWO = padWOutLength_; // grad_x W
        padParam_.padLeft = static_cast<uint32_t>(tilingData_->leftPad[dimNum_ - 1]);
        padParam_.padRight = static_cast<uint32_t>(tilingData_->rightPad[dimNum_ - 1]);

        // UB 内完整维度使用 inShape (grad_y) 初始化拷贝长度
        // 因为 CopyIn 从 H=0 开始读取完整的 grad_y 行（含 padding 行）
        for (int8_t i = dimNum_ - 1; i > ubAxis_; i--) {
            inCopyLen_[i] = tilingData_->inShape[i];
        }
        // 检查 H 维度是否有 padding
        if (dimNum_ >= MIN_DIM_FOR_H_PAD) {
            has2DPadding = (tilingData_->leftPad[dimNum_ - MIN_DIM_FOR_H_PAD] > 0 ||
                            tilingData_->rightPad[dimNum_ - MIN_DIM_FOR_H_PAD] > 0);
        }
        // 检查 C 维度是否有 padding
        if (dimNum_ >= MIN_DIM_FOR_C_PAD) {
            has3DPadding = (tilingData_->leftPad[dimNum_ - MIN_DIM_FOR_C_PAD] > 0 ||
                            tilingData_->rightPad[dimNum_ - MIN_DIM_FOR_C_PAD] > 0);
        }
        // 检查 N 维度是否有 padding
        if (dimNum_ >= MIN_DIM_FOR_N_PAD) {
            has4DPadding = (tilingData_->leftPad[dimNum_ - MIN_DIM_FOR_N_PAD] > 0 ||
                            tilingData_->rightPad[dimNum_ - MIN_DIM_FOR_N_PAD] > 0);
        }
        // 检查第5维是否有 padding
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD) {
            has5DPadding = (tilingData_->leftPad[dimNum_ - MIN_DIM_FOR_D5_PAD] > 0 ||
                            tilingData_->rightPad[dimNum_ - MIN_DIM_FOR_D5_PAD] > 0);
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t ubFactor = tilingData_->ubFactor;
        uint64_t ubPerCount = tilingData_->ubPerCount;
        uint64_t ubTotalCount = tilingData_->ubTotalCount;

        uint32_t startIdx = blockIdx_ * ubPerCount;
        if (startIdx >= ubTotalCount) {
            return;
        }
        uint32_t endIdx = (blockIdx_ + 1L) * ubPerCount;
        endIdx = (endIdx < ubTotalCount ? endIdx : ubTotalCount);

        // 按 outShape (grad_x) 计算索引，因为分核是按 outShape 计算的
        for (uint32_t idx = startIdx; idx < endIdx; idx++) {
            uint32_t curIdx = idx;
            for (int8_t i = ubAxis_; i >= 0; i--) {
                uint64_t factor = tilingData_->outShape[i]; // use outShape (grad_x shape)
                if (i == ubAxis_) {
                    factor = CeilDiv(tilingData_->outShape[i], ubFactor);
                }
                outIndex_[i] = (i == ubAxis_ ? curIdx % factor * ubFactor : curIdx % factor);
                inIndex_[i] = outIndex_[i] + tilingData_->leftPad[i]; // corresponding position in grad_y
                curIdx = curIdx / factor;
            }
            // 切分轴: 实际拷贝长度 = min(ubFactor, outShape[ubAxis_] - outIndex[ubAxis_])
            inCopyLen_[ubAxis_] = outIndex_[ubAxis_] + ubFactor < tilingData_->outShape[ubAxis_] ?
                                      ubFactor :
                                      tilingData_->outShape[ubAxis_] - outIndex_[ubAxis_];
            ProcessOneStep();
        }
    }

private:
    __aicore__ inline void ProcessOneStep()
    {
        CopyIn();
        LocalTensor<T> dstLocal = outQueue_.AllocTensor<T>();
        GradGatherProcess(dstLocal);
        outQueue_.EnQue(dstLocal);
        dstLocal = outQueue_.DeQue<T>();
        CopyOut(dstLocal);
        outQueue_.FreeTensor(dstLocal);
    }

    __aicore__ inline void CopyIn()
    {
        // Calculate input address in grad_y (padded tensor)
        LocalTensor<T> inUbLocal = inQueue_.AllocTensor<T>();

        uint64_t inAddr = 0;
        for (uint8_t i = 0; i < dimNum_; i++) {
            inAddr += inIndex_[i] * tilingData_->inStride[i];
        }

        DataCopyPadExtParams<T> padExtParams{true, 0, 0, 0};
        DataCopyExtParams copyInParams;
        // 使用 inCopyLen_ (按 inShape 计算的长度) 来确定拷贝的行数
        copyInParams.blockCount = inCopyLen_[dimNum_ - CONST2];
        copyInParams.blockLen = tilingData_->inShape[dimNum_ - 1] * sizeof(T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        if (axisNumInUb_ == CONST3) {
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = inCopyLen_[dimNum_ - CONST3];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum_ - CONST3] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->inShape[dimNum_ - CONST2] * padParam_.padWI * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(inUbLocal, input_[inAddr], copyInParams, padExtParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else if (axisNumInUb_ == CONST4) {
            LoopModeParams loopParams;
            loopParams.loop1Size = inCopyLen_[dimNum_ - CONST3];
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum_ - CONST3] * sizeof(T);
            loopParams.loop1DstStride = tilingData_->inShape[dimNum_ - CONST2] * padParam_.padWI * sizeof(T);
            loopParams.loop2Size = inCopyLen_[dimNum_ - CONST4];
            loopParams.loop2SrcStride = tilingData_->inStride[dimNum_ - CONST4] * sizeof(T);
            loopParams.loop2DstStride = tilingData_->inShape[dimNum_ - CONST3] *
                                        tilingData_->inShape[dimNum_ - CONST2] * padParam_.padWI * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(inUbLocal, input_[inAddr], copyInParams, padExtParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else {
            DataCopyPad(inUbLocal, input_[inAddr], copyInParams, padExtParams);
        }
        inQueue_.EnQue(inUbLocal);
    }

    // 将 src (inQueue_) 中的数据拷贝/转换到 tmpBuf_
    // 对于 bfloat16_t/half 类型，使用 Cast 转换为 float32 存储
    // 对于 float32 类型，直接使用 Copy
    __aicore__ inline void CopyToTmpBuf(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum, uint32_t padWI)
    {
        LocalTensor<T> inUbLocal = inQueue_.DeQue<T>();
        uint32_t totalLen = dimHNum * padWI;

        if constexpr (IsSameType<T, PromoteDataT>::value) {
            // float32 → float32，直接使用 Copy
            Copy(tmpLocal, inUbLocal, totalLen);
        } else {
            // bfloat16_t/half → float32，使用 Cast
            Cast(tmpLocal, inUbLocal, RoundMode::CAST_NONE, totalLen);
        }
        inQueue_.FreeTensor(inUbLocal);
    }
    // W dimension gradient accumulation
    // For each position w in grad_x, accumulate from:
    // 1. Self position: grad_y[w + leftPad]
    // 2. Left pad mirror: grad_y[leftPad - 1 - w + modeOffset_] if applicable
    // 3. Right pad mirror: grad_y[2*outW + leftPad - 2 - w + modeOffset_] if applicable
    __aicore__ inline void GradGatherProcess(const LocalTensor<T>& dst)
    {
        __local_mem__ T* dstAddr = reinterpret_cast<__local_mem__ T*>(dst.GetPhyAddr());
        const uint32_t outW = tilingData_->outShape[dimNum_ - 1]; // grad_x W
        const uint32_t inW = tilingData_->inShape[dimNum_ - 1];   // grad_y W
        const uint32_t leftPad = padParam_.padLeft;
        const uint32_t rightPad = padParam_.padRight;
        const uint32_t padWI = padParam_.padWI;
        const uint32_t padWO = padParam_.padWO;
        const uint32_t padHW = tilingData_->outShape[dimNum_ - CONST2] * padWO;
        const uint32_t padCHW = (axisNumInUb_ > CONST3) ? tilingData_->outShape[dimNum_ - CONST3] * padHW : 0;

        // inCopyLen_ 现在是 inShape (CopyIn 拷贝的行数)
        // outShape 用于输出侧 (dst 写入的行数)
        const uint32_t dimNNum = (axisNumInUb_ < CONST4) ? 1 : inCopyLen_[dimNum_ - CONST4];
        const uint32_t dimCNum = (axisNumInUb_ < CONST3) ? 1 : inCopyLen_[dimNum_ - CONST3];
        const uint32_t dimHIn = inCopyLen_[dimNum_ - CONST2];             // inShape[H], src 侧行数
        const uint32_t dimHOut = tilingData_->outShape[dimNum_ - CONST2]; // outShape[H], dst 侧行数

        if (axisNumInUb_ == CONST2) {
            // axisNumInUb_=2 时的数据流：
            // 1. CopyIn 后，src 包含原始 grad_y 数据 (T 类型)
            // 2a. 主pad: 批量从 GM 读取纯高维镜像 (不含 H)，Cast+Add 到 tmpBuf_
            // 2b. 副pad: 逐行从 GM 读取 H 相关镜像，Cast+Add 到 tmpBuf_
            // 3. W 维度梯度累加：从 tmpBuf_ 读取，写入 dst
            LocalTensor<PromoteDataT> tmpLocal = tmpBuf_.Get<PromoteDataT>();

            // Step 1: 将 src (原始 grad_y) Cast/Copy 到 tmpLocal
            CopyToTmpBuf(tmpLocal, dimHIn, padWI);

            // Step 2a: 主pad — 批量从 GM 读取纯高维镜像 (C, N, D5 及其组合)
            GradAccumulateHighDimBulk_UB2(tmpLocal, dimHIn, padWI);

            // Step 2b: 副pad — 逐行处理 H 相关镜像 (H, C×H, N×H, N×C×H, D5×H 等)
            GradAccumulateHRelatedToTmpBuf(tmpLocal, dimHIn, padWI);

            // Step 3: W 维度梯度累加
            __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr());
            for (uint32_t h = 0; h < dimHIn; h++) {
                GradProcessLineFromTmpBuf(dstAddr + h * padWO, tmpAddr + h * padWI, outW, inW, leftPad, rightPad);
            }
        } else if (axisNumInUb_ == CONST3) {
            // axisNumInUb_=3: UB 内包含 inCopyLen_[C] × inShape[H] × inShape[W]
            // C 是切分轴，UB 内只有部分 C slice
            LocalTensor<PromoteDataT> tmpLocal = tmpBuf_.Get<PromoteDataT>();
            __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr());
            const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
            const uint32_t sliceSize = dimHIn * padWI; // 一个 C slice 在 tmpLocal 中的元素数

            // Step 1: 一次性 Cast 所有 C slice 数据到 tmpLocal
            CopyToTmpBuf(tmpLocal, dimCNum * dimHIn, padWI);

            // Step 2: 批量从 GM 累加高维镜像 (主pad + 副pad)
            GradAccumulateHighDimBulk(tmpLocal, dimCNum, dimHIn, padWI);

            // Step 3: H 镜像内部累加 (padding 行 → self 行)
            for (uint32_t c = 0; c < dimCNum; c++) {
                AccumulateHMirrorsInPlace(tmpLocal, c * sliceSize, padWI);
            }

            // Step 4: W 维度梯度累加，从 self 行读取，写入 dst
            for (uint32_t c = 0; c < dimCNum; c++) {
                for (uint32_t h = 0; h < dimHOut; h++) {
                    GradProcessLineFromTmpBuf(dstAddr + c * padHW + h * padWO,
                                              tmpAddr + c * sliceSize + (h + leftPadH) * padWI, outW, inW, leftPad,
                                              rightPad);
                }
            }
        } else if (axisNumInUb_ == CONST4) {
            // axisNumInUb_=4: UB 内包含 inCopyLen_[N] × inShape[C] × inShape[H] × inShape[W]
            // N 是切分轴，C/H/W 完整在 UB 内
            LocalTensor<PromoteDataT> tmpLocal = tmpBuf_.Get<PromoteDataT>();
            __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr());
            const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
            const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
            const uint32_t hSliceSize = dimHIn * padWI;       // 一个 H slice (一个 C plane)
            const uint32_t cSliceSize = dimCNum * hSliceSize; // 一个 N slice (所有 C planes)

            // Step 1: 一次性 Cast 所有 N×C×H 数据到 tmpLocal
            CopyToTmpBuf(tmpLocal, dimNNum * dimCNum * dimHIn, padWI);

            // Step 2: 批量从 GM 累加高维镜像 (主pad D5 + 副pad N)
            GradAccumulateHighDimBulk_UB4(tmpLocal, dimNNum, dimCNum, dimHIn, padWI);

            // Step 3: C 镜像内部累加 (对每个 N slice)
            for (uint32_t n = 0; n < dimNNum; n++) {
                AccumulateCMirrorsInPlace(tmpLocal[n * cSliceSize], dimHIn, padWI);
            }

            // Step 4: H 镜像内部累加 (对每个 N slice 的每个 self C slice)
            const uint32_t dimCOut = tilingData_->outShape[dimNum_ - CONST3];
            for (uint32_t n = 0; n < dimNNum; n++) {
                for (uint32_t c = 0; c < dimCOut; c++) {
                    AccumulateHMirrorsInPlace(tmpLocal, n * cSliceSize + (c + leftPadC) * hSliceSize, padWI);
                }
            }

            // Step 5: W 维度梯度累加，从 self 行读取，写入 dst
            for (uint32_t n = 0; n < dimNNum; n++) {
                for (uint32_t c = 0; c < dimCOut; c++) {
                    for (uint32_t h = 0; h < dimHOut; h++) {
                        GradProcessLineFromTmpBuf(
                            dstAddr + n * padCHW + c * padHW + h * padWO,
                            tmpAddr + n * cSliceSize + (c + leftPadC) * hSliceSize + (h + leftPadH) * padWI, outW, inW,
                            leftPad, rightPad);
                    }
                }
            }
        }
    }

    // axisNumInUb_=2 副pad: 只处理涉及 H 镜像的组合 (逐行从 GM 读取)
    // 主pad (纯高维镜像) 已由 GradAccumulateHighDimBulk_UB2 批量处理
    __aicore__ inline void GradAccumulateHRelatedToTmpBuf(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                          uint32_t padWI)
    {
        const uint32_t inW = tilingData_->inShape[dimNum_ - 1];
        const uint32_t globalHStart = outIndex_[dimNum_ - CONST2];
        __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr());

        for (uint32_t h = 0; h < dimHNum; h++) {
            uint32_t globalH = globalHStart + h;
            __local_mem__ PromoteDataT* lineAddr = tmpAddr + h * padWI;

            // 1. 纯 H 镜像
            if (has2DPadding) {
                ProcessHDimMirrorAtCurrentHighDim(lineAddr, globalH, inW);
            }

            // 2. C×H 组合 (需要 C 有 padding 且 H 有 padding)
            if (dimNum_ >= MIN_DIM_FOR_C_PAD && has3DPadding && has2DPadding) {
                ProcessCxHSubPad(lineAddr, globalH, inW);
            }

            // 3. N×H 相关组合 (N×H, N×C×H)
            if (dimNum_ >= MIN_DIM_FOR_N_PAD && has4DPadding) {
                ProcessNxHSubPad(lineAddr, globalH, inW);
            }

            // 4. D5×H 相关组合 (D5×H, D5×C×H, D5×N×H, D5×N×C×H)
            if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
                ProcessD5xHSubPad(lineAddr, globalH, inW);
            }
        }
    }

    // 副pad: C×H 组合 — 对每个 mirrorC 位置，处理 H 镜像
    __aicore__ inline void ProcessCxHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH, uint32_t inW)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        MirrorList cList = CollectMirrorPositions(globalC, tilingData_->outShape[dimNum_ - CONST3],
                                                  tilingData_->leftPad[dimNum_ - CONST3],
                                                  tilingData_->rightPad[dimNum_ - CONST3]);

        for (uint8_t ci = 0; ci < cList.count; ci++) {
            ProcessCxHCombinedMirror(lineAddr, globalH, cList.mirrors[ci], inW);
        }
    }

    // 副pad: N×H 相关组合 — 对每个 mirrorN 位置，处理 N×H 和 N×C×H
    __aicore__ inline void ProcessNxHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH, uint32_t inW)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        MirrorList nList = CollectMirrorPositions(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                  tilingData_->leftPad[dimNum_ - CONST4],
                                                  tilingData_->rightPad[dimNum_ - CONST4]);

        for (uint8_t ni = 0; ni < nList.count; ni++) {
            // N×H
            ProcessNxHCombinedMirror(lineAddr, globalH, nList.mirrors[ni], inW);
            // N×C×H
            if (has3DPadding) {
                ProcessNxCxHSubPad(lineAddr, globalH, nList.mirrors[ni], inW);
            }
        }
    }

    // 副pad: N×C×H 组合 — 对给定 mirrorN，检查 C 镜像条件，处理 N×C×H
    __aicore__ inline void ProcessNxCxHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH, uint32_t mirrorN,
                                              uint32_t inW)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        MirrorList cList = CollectMirrorPositions(globalC, tilingData_->outShape[dimNum_ - CONST3],
                                                  tilingData_->leftPad[dimNum_ - CONST3],
                                                  tilingData_->rightPad[dimNum_ - CONST3]);

        for (uint8_t ci = 0; ci < cList.count; ci++) {
            ProcessNxCxHCombinedMirror(lineAddr, globalH, mirrorN, cList.mirrors[ci], inW);
        }
    }

    // 副pad: D5×H 相关组合 — 对每个 mirrorD5，处理 D5×H, D5×C×H, D5×N×H, D5×N×C×H
    __aicore__ inline void ProcessD5xHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH, uint32_t inW)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        MirrorList d5List = CollectMirrorPositions(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                   tilingData_->leftPad[dimNum_ - CONST5],
                                                   tilingData_->rightPad[dimNum_ - CONST5]);

        for (uint8_t di = 0; di < d5List.count; di++) {
            // D5×H
            ProcessD5xHCombinedMirror(lineAddr, globalH, d5List.mirrors[di], inW);
            // D5×C×H
            if (has3DPadding) {
                ProcessD5xCxHSubPad(lineAddr, globalH, d5List.mirrors[di], inW);
            }
            // D5×N×H (含 D5×N×C×H)
            if (has4DPadding) {
                ProcessD5xNxHSubPad(lineAddr, globalH, d5List.mirrors[di], inW);
            }
        }
    }

    // D5×H 组合镜像: 给定 mirrorD5，检查 H 镜像条件
    __aicore__ inline void ProcessD5xHCombinedMirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                     uint32_t mirrorD5, uint32_t inW)
    {
        if (!has2DPadding) {
            return;
        }

        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];
        MirrorCondition condH = CalcMirrorCondition(globalH, tilingData_->outShape[dimNum_ - CONST2], leftPadH,
                                                    tilingData_->rightPad[dimNum_ - CONST2]);

        if (condH.hasTop && condH.mirrorTop < leftPadH) {
            CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, inIndex_[dimNum_ - CONST4],
                                               inIndex_[dimNum_ - CONST3], condH.mirrorTop, inW);
        }

        if (condH.hasBottom && condH.mirrorBottom >= leftPadH + tilingData_->outShape[dimNum_ - CONST2] &&
            condH.mirrorBottom < inH) {
            CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, inIndex_[dimNum_ - CONST4],
                                               inIndex_[dimNum_ - CONST3], condH.mirrorBottom, inW);
        }
    }

    // D5×C×H 副pad: 给定 mirrorD5，检查 C 镜像条件，对每个 mirrorC 处理 H 镜像
    __aicore__ inline void ProcessD5xCxHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                               uint32_t mirrorD5, uint32_t inW)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];

        MirrorList cList = CollectMirrorPositions(globalC, tilingData_->outShape[dimNum_ - CONST3],
                                                  tilingData_->leftPad[dimNum_ - CONST3],
                                                  tilingData_->rightPad[dimNum_ - CONST3]);
        MirrorCondition condH = CalcMirrorCondition(globalH, outH, leftPadH, tilingData_->rightPad[dimNum_ - CONST2]);

        for (uint8_t ci = 0; ci < cList.count; ci++) {
            if (condH.hasTop && condH.mirrorTop < leftPadH) {
                CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, inIndex_[dimNum_ - CONST4], cList.mirrors[ci],
                                                   condH.mirrorTop, inW);
            }
            if (condH.hasBottom && condH.mirrorBottom >= leftPadH + outH && condH.mirrorBottom < inH) {
                CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, inIndex_[dimNum_ - CONST4], cList.mirrors[ci],
                                                   condH.mirrorBottom, inW);
            }
        }
    }

    // D5×N×H 副pad (含 D5×N×C×H): 给定 mirrorD5，检查 N 镜像条件
    __aicore__ inline void ProcessD5xNxHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                               uint32_t mirrorD5, uint32_t inW)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];

        MirrorList nList = CollectMirrorPositions(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                  tilingData_->leftPad[dimNum_ - CONST4],
                                                  tilingData_->rightPad[dimNum_ - CONST4]);
        MirrorCondition condH = CalcMirrorCondition(globalH, outH, leftPadH, tilingData_->rightPad[dimNum_ - CONST2]);

        for (uint8_t ni = 0; ni < nList.count; ni++) {
            // D5×N×H
            if (condH.hasTop && condH.mirrorTop < leftPadH) {
                CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, nList.mirrors[ni], inIndex_[dimNum_ - CONST3],
                                                   condH.mirrorTop, inW);
            }
            if (condH.hasBottom && condH.mirrorBottom >= leftPadH + outH && condH.mirrorBottom < inH) {
                CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, nList.mirrors[ni], inIndex_[dimNum_ - CONST3],
                                                   condH.mirrorBottom, inW);
            }
            // D5×N×C×H
            if (has3DPadding) {
                ProcessD5xNxCxHSubPad(lineAddr, globalH, mirrorD5, nList.mirrors[ni], inW);
            }
        }
    }

    // D5×N×C×H 副pad: 给定 mirrorD5 和 mirrorN，检查 C 和 H 镜像条件
    __aicore__ inline void ProcessD5xNxCxHSubPad(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                 uint32_t mirrorD5, uint32_t mirrorN, uint32_t inW)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];

        MirrorList cList = CollectMirrorPositions(globalC, tilingData_->outShape[dimNum_ - CONST3],
                                                  tilingData_->leftPad[dimNum_ - CONST3],
                                                  tilingData_->rightPad[dimNum_ - CONST3]);
        MirrorCondition condH = CalcMirrorCondition(globalH, outH, leftPadH, tilingData_->rightPad[dimNum_ - CONST2]);

        for (uint8_t ci = 0; ci < cList.count; ci++) {
            if (condH.hasTop && condH.mirrorTop < leftPadH) {
                CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, mirrorN, cList.mirrors[ci], condH.mirrorTop,
                                                   inW);
            }
            if (condH.hasBottom && condH.mirrorBottom >= leftPadH + outH && condH.mirrorBottom < inH) {
                CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, mirrorN, cList.mirrors[ci], condH.mirrorBottom,
                                                   inW);
            }
        }
    }

    // H 维度镜像处理 (高维索引不变，只改变 H 索引)
    __aicore__ inline void ProcessHDimMirrorAtCurrentHighDim(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                             uint32_t inW)
    {
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t rightPadH = tilingData_->rightPad[dimNum_ - CONST2];

        // 上镜像
        bool hasTopMirror = (modeOffset_ == 0) ? (globalH > 0 && globalH <= leftPadH) : (globalH < leftPadH);
        if (hasTopMirror) {
            uint32_t mirrorH = leftPadH - modeOffset_ - globalH;
            // 使用当前高维索引，只改变 H
            CopyAndAddMirrorLineFromGM(lineAddr, mirrorH, inW);
        }

        // 下镜像
        bool hasBottomMirror = (modeOffset_ == 0) ?
                                   (rightPadH > 0 && globalH >= outH - rightPadH - 1 && globalH <= outH - 2) :
                                   (rightPadH > 0 && globalH >= outH - rightPadH);
        if (hasBottomMirror) {
            uint32_t mirrorH = leftPadH + 2 * outH - 2 + modeOffset_ - globalH;
            uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];
            CopyAndAddMirrorLineFromGM(lineAddr, mirrorH, inW);
        }
    }

    // C 维度镜像处理
    __aicore__ inline void ProcessCDimMirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH, uint32_t inW)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outC = tilingData_->outShape[dimNum_ - CONST3];
        const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
        const uint32_t rightPadC = tilingData_->rightPad[dimNum_ - CONST3];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];

        // C 轴上镜像
        bool hasTopMirrorC = (modeOffset_ == 0) ? (globalC > 0 && globalC <= leftPadC) : (globalC < leftPadC);
        if (hasTopMirrorC) {
            uint32_t mirrorC = leftPadC - modeOffset_ - globalC;
            // 从 GM 拷贝 grad_y[..., mirrorC, globalH+leftPadH, :] 并累加
            uint32_t hIdxInGradY = globalH + leftPadH;
            CopyAndAddMirrorLineFromGMWithC(lineAddr, mirrorC, hIdxInGradY, inW);

            // C×H 组合镜像
            if (has2DPadding) {
                ProcessCxHCombinedMirror(lineAddr, globalH, mirrorC, inW);
            }
        }

        // C 轴下镜像
        bool hasBottomMirrorC = (modeOffset_ == 0) ?
                                    (rightPadC > 0 && globalC >= outC - rightPadC - 1 && globalC <= outC - 2) :
                                    (rightPadC > 0 && globalC >= outC - rightPadC);
        if (hasBottomMirrorC) {
            uint32_t mirrorC = leftPadC + 2 * outC - 2 + modeOffset_ - globalC;
            uint32_t hIdxInGradY = globalH + leftPadH;
            CopyAndAddMirrorLineFromGMWithC(lineAddr, mirrorC, hIdxInGradY, inW);

            // C×H 组合镜像
            if (has2DPadding) {
                ProcessCxHCombinedMirror(lineAddr, globalH, mirrorC, inW);
            }
        }
    }

    // C×H 组合镜像处理
    __aicore__ inline void ProcessCxHCombinedMirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                    uint32_t mirrorC, uint32_t inW)
    {
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t rightPadH = tilingData_->rightPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];

        // H 上镜像 × C 镜像
        bool hasTopMirrorH = (modeOffset_ == 0) ? (globalH > 0 && globalH <= leftPadH) : (globalH < leftPadH);
        if (hasTopMirrorH) {
            uint32_t mirrorH = leftPadH - modeOffset_ - globalH;
            if (mirrorH < leftPadH) {
                CopyAndAddMirrorLineFromGMWithC(lineAddr, mirrorC, mirrorH, inW);
            }
        }

        // H 下镜像 × C 镜像
        bool hasBottomMirrorH = (modeOffset_ == 0) ?
                                    (rightPadH > 0 && globalH >= outH - rightPadH - 1 && globalH <= outH - 2) :
                                    (rightPadH > 0 && globalH >= outH - rightPadH);
        if (hasBottomMirrorH) {
            uint32_t mirrorH = leftPadH + 2 * outH - 2 + modeOffset_ - globalH;
            if (mirrorH >= leftPadH + outH && mirrorH < inH) {
                CopyAndAddMirrorLineFromGMWithC(lineAddr, mirrorC, mirrorH, inW);
            }
        }
    }

    // N×H 组合镜像
    __aicore__ inline void ProcessNxHCombinedMirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                    uint32_t mirrorN, uint32_t inW)
    {
        if (!has2DPadding)
            return;

        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t rightPadH = tilingData_->rightPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];

        bool hasTopMirrorH = (modeOffset_ == 0) ? (globalH > 0 && globalH <= leftPadH) : (globalH < leftPadH);
        if (hasTopMirrorH) {
            uint32_t mirrorH = leftPadH - modeOffset_ - globalH;
            if (mirrorH < leftPadH) {
                CopyAndAddMirrorLineFromGMWithNC(lineAddr, mirrorN, inIndex_[dimNum_ - CONST3], mirrorH, inW);
            }
        }

        bool hasBottomMirrorH = (modeOffset_ == 0) ?
                                    (rightPadH > 0 && globalH >= outH - rightPadH - 1 && globalH <= outH - 2) :
                                    (rightPadH > 0 && globalH >= outH - rightPadH);
        if (hasBottomMirrorH) {
            uint32_t mirrorH = leftPadH + 2 * outH - 2 + modeOffset_ - globalH;
            if (mirrorH >= leftPadH + outH && mirrorH < inH) {
                CopyAndAddMirrorLineFromGMWithNC(lineAddr, mirrorN, inIndex_[dimNum_ - CONST3], mirrorH, inW);
            }
        }
    }

    // N×C 组合镜像 (包含 N×C×H)
    __aicore__ inline void ProcessNxCCombinedMirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                    uint32_t mirrorN, uint32_t inW)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outC = tilingData_->outShape[dimNum_ - CONST3];
        const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
        const uint32_t rightPadC = tilingData_->rightPad[dimNum_ - CONST3];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];

        // N×C 上镜像
        bool hasTopMirrorC = (modeOffset_ == 0) ? (globalC > 0 && globalC <= leftPadC) : (globalC < leftPadC);
        if (hasTopMirrorC) {
            uint32_t mirrorC = leftPadC - modeOffset_ - globalC;
            uint32_t hIdxInGradY = globalH + leftPadH;
            CopyAndAddMirrorLineFromGMWithNC(lineAddr, mirrorN, mirrorC, hIdxInGradY, inW);

            // N×C×H 组合镜像
            if (has2DPadding) {
                ProcessNxCxHCombinedMirror(lineAddr, globalH, mirrorN, mirrorC, inW);
            }
        }

        // N×C 下镜像
        bool hasBottomMirrorC = (modeOffset_ == 0) ?
                                    (rightPadC > 0 && globalC >= outC - rightPadC - 1 && globalC <= outC - 2) :
                                    (rightPadC > 0 && globalC >= outC - rightPadC);
        if (hasBottomMirrorC) {
            uint32_t mirrorC = leftPadC + 2 * outC - 2 + modeOffset_ - globalC;
            uint32_t hIdxInGradY = globalH + leftPadH;
            CopyAndAddMirrorLineFromGMWithNC(lineAddr, mirrorN, mirrorC, hIdxInGradY, inW);

            // N×C×H 组合镜像
            if (has2DPadding) {
                ProcessNxCxHCombinedMirror(lineAddr, globalH, mirrorN, mirrorC, inW);
            }
        }
    }

    // N×C×H 组合镜像
    __aicore__ inline void ProcessNxCxHCombinedMirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH,
                                                      uint32_t mirrorN, uint32_t mirrorC, uint32_t inW)
    {
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t rightPadH = tilingData_->rightPad[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];

        bool hasTopMirrorH = (modeOffset_ == 0) ? (globalH > 0 && globalH <= leftPadH) : (globalH < leftPadH);
        if (hasTopMirrorH) {
            uint32_t mirrorH = leftPadH - modeOffset_ - globalH;
            if (mirrorH < leftPadH) {
                CopyAndAddMirrorLineFromGMWithNC(lineAddr, mirrorN, mirrorC, mirrorH, inW);
            }
        }

        bool hasBottomMirrorH = (modeOffset_ == 0) ?
                                    (rightPadH > 0 && globalH >= outH - rightPadH - 1 && globalH <= outH - 2) :
                                    (rightPadH > 0 && globalH >= outH - rightPadH);
        if (hasBottomMirrorH) {
            uint32_t mirrorH = leftPadH + 2 * outH - 2 + modeOffset_ - globalH;
            if (mirrorH >= leftPadH + outH && mirrorH < inH) {
                CopyAndAddMirrorLineFromGMWithNC(lineAddr, mirrorN, mirrorC, mirrorH, inW);
            }
        }
    }

    // 第5维镜像处理 (简化版，只处理单维度镜像)
    __aicore__ inline void ProcessDim5Mirror(__local_mem__ PromoteDataT* lineAddr, uint32_t globalH, uint32_t inW)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        const uint32_t outD5 = tilingData_->outShape[dimNum_ - CONST5];
        const uint32_t leftPadD5 = tilingData_->leftPad[dimNum_ - CONST5];
        const uint32_t rightPadD5 = tilingData_->rightPad[dimNum_ - CONST5];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];

        // 第5维上镜像
        bool hasTopMirrorD5 = (modeOffset_ == 0) ? (globalD5 > 0 && globalD5 <= leftPadD5) : (globalD5 < leftPadD5);
        if (hasTopMirrorD5) {
            uint32_t mirrorD5 = leftPadD5 - modeOffset_ - globalD5;
            uint32_t hIdxInGradY = globalH + leftPadH;
            CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, inIndex_[dimNum_ - CONST4],
                                               inIndex_[dimNum_ - CONST3], hIdxInGradY, inW);
        }

        // 第5维下镜像
        bool hasBottomMirrorD5 = (modeOffset_ == 0) ?
                                     (rightPadD5 > 0 && globalD5 >= outD5 - rightPadD5 - 1 && globalD5 <= outD5 - 2) :
                                     (rightPadD5 > 0 && globalD5 >= outD5 - rightPadD5);
        if (hasBottomMirrorD5) {
            uint32_t mirrorD5 = leftPadD5 + 2 * outD5 - 2 + modeOffset_ - globalD5;
            uint32_t hIdxInGradY = globalH + leftPadH;
            CopyAndAddMirrorLineFromGMWithD5NC(lineAddr, mirrorD5, inIndex_[dimNum_ - CONST4],
                                               inIndex_[dimNum_ - CONST3], hIdxInGradY, inW);
        }
    }

    // 从 GM 拷贝镜像行到 srcLocal (inQueue_)，然后整行累加到目标行
    // 使用当前高维索引 (inIndex_)，只改变 H 索引
    __aicore__ inline void CopyAndAddMirrorLineFromGM(__local_mem__ PromoteDataT* dstLineAddr, uint32_t hIdx,
                                                      uint32_t inW)
    {
        // 计算 GM 地址：使用当前高维索引，只改变 H
        uint64_t mirrorAddr = 0;
        for (uint8_t i = 0; i < dimNum_ - CONST2; i++) {
            mirrorAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        mirrorAddr += hIdx * tilingData_->inStride[dimNum_ - CONST2];

        CopyAndAddLineFromGMAddr(dstLineAddr, mirrorAddr, inW);
    }

    // 从 GM 拷贝镜像行，指定 C 索引
    __aicore__ inline void CopyAndAddMirrorLineFromGMWithC(__local_mem__ PromoteDataT* dstLineAddr, uint32_t cIdx,
                                                           uint32_t hIdx, uint32_t inW)
    {
        uint64_t mirrorAddr = 0;
        // 高于 C 的维度使用 inIndex_
        for (uint8_t i = 0; i < dimNum_ - CONST3; i++) {
            mirrorAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        // C 维度使用指定的 cIdx
        mirrorAddr += cIdx * tilingData_->inStride[dimNum_ - CONST3];
        // H 维度
        mirrorAddr += hIdx * tilingData_->inStride[dimNum_ - CONST2];

        CopyAndAddLineFromGMAddr(dstLineAddr, mirrorAddr, inW);
    }

    // 从 GM 拷贝镜像行，指定 N 和 C 索引
    __aicore__ inline void CopyAndAddMirrorLineFromGMWithNC(__local_mem__ PromoteDataT* dstLineAddr, uint32_t nIdx,
                                                            uint32_t cIdx, uint32_t hIdx, uint32_t inW)
    {
        uint64_t mirrorAddr = 0;
        // 高于 N 的维度使用 inIndex_
        for (uint8_t i = 0; i < dimNum_ - CONST4; i++) {
            mirrorAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        // N 维度
        mirrorAddr += nIdx * tilingData_->inStride[dimNum_ - CONST4];
        // C 维度
        mirrorAddr += cIdx * tilingData_->inStride[dimNum_ - CONST3];
        // H 维度
        mirrorAddr += hIdx * tilingData_->inStride[dimNum_ - CONST2];

        CopyAndAddLineFromGMAddr(dstLineAddr, mirrorAddr, inW);
    }

    // 从 GM 拷贝镜像行，指定第5维、N 和 C 索引
    __aicore__ inline void CopyAndAddMirrorLineFromGMWithD5NC(__local_mem__ PromoteDataT* dstLineAddr, uint32_t d5Idx,
                                                              uint32_t nIdx, uint32_t cIdx, uint32_t hIdx, uint32_t inW)
    {
        uint64_t mirrorAddr = 0;
        // 第5维
        mirrorAddr += d5Idx * tilingData_->inStride[dimNum_ - CONST5];
        // N 维度
        mirrorAddr += nIdx * tilingData_->inStride[dimNum_ - CONST4];
        // C 维度
        mirrorAddr += cIdx * tilingData_->inStride[dimNum_ - CONST3];
        // H 维度
        mirrorAddr += hIdx * tilingData_->inStride[dimNum_ - CONST2];

        CopyAndAddLineFromGMAddr(dstLineAddr, mirrorAddr, inW);
    }

    // tmpLocal 内两行 PromoteDataT (float32) 相加: dstLine[i] += srcLine[i]
    __aicore__ inline void AddLocalLineInTmpBuf(__local_mem__ PromoteDataT* dstLineAddr,
                                                __local_mem__ PromoteDataT* srcLineAddr, uint32_t inW)
    {
        constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(PromoteDataT);
        constexpr uint32_t BL_ELEMS_FLOAT = UB_BLOCK / sizeof(T);
        uint32_t inWAlign = CeilAlign(inW, BL_ELEMS_FLOAT);
        uint16_t loopCount = CeilDiv(inWAlign, VL_ELEMS_FLOAT);
        uint32_t remainLen = inW;

        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::RegTensor<PromoteDataT> dstReg;
            AscendC::Reg::RegTensor<PromoteDataT> srcReg;

            for (uint16_t i = 0; i < loopCount; i++) {
                mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                AscendC::Reg::DataCopy(dstReg, dstLineAddr + i * VL_ELEMS_FLOAT);
                AscendC::Reg::DataCopy(srcReg, srcLineAddr + i * VL_ELEMS_FLOAT);
                AscendC::Reg::Add(dstReg, dstReg, srcReg, mask);
                AscendC::Reg::DataCopy(dstLineAddr + i * VL_ELEMS_FLOAT, dstReg, mask);
            }
        }
    }
    // tmpLocal 内两行 PromoteDataT (float32) 相加: dstLine[i] = dstLine[i] + srcLine[i] + srcLine[j]
    __aicore__ inline void AddLocalUpAndDownLineInTmpBuf(__local_mem__ PromoteDataT* dstLineAddr,
                                                         __local_mem__ PromoteDataT* srcUpLineAddr,
                                                         __local_mem__ PromoteDataT* srcDownLineAddr, uint32_t inW)
    {
        constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(PromoteDataT);
        constexpr uint32_t BL_ELEMS_FLOAT = UB_BLOCK / sizeof(T);
        uint32_t inWAlign = CeilAlign(inW, BL_ELEMS_FLOAT);
        uint16_t loopCount = CeilDiv(inWAlign, VL_ELEMS_FLOAT);
        uint32_t remainLen = inW;

        __VEC_SCOPE__
        {
            AscendC::Reg::MaskReg mask;
            AscendC::Reg::RegTensor<PromoteDataT> dstReg;
            AscendC::Reg::RegTensor<PromoteDataT> srcUpReg;
            AscendC::Reg::RegTensor<PromoteDataT> srcDownReg;

            for (uint16_t i = 0; i < loopCount; i++) {
                mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                AscendC::Reg::DataCopy(dstReg, dstLineAddr + i * VL_ELEMS_FLOAT);
                AscendC::Reg::DataCopy(srcUpReg, srcUpLineAddr + i * VL_ELEMS_FLOAT);
                AscendC::Reg::DataCopy(srcDownReg, srcDownLineAddr + i * VL_ELEMS_FLOAT);
                AscendC::Reg::Add(dstReg, dstReg, srcUpReg, mask);
                AscendC::Reg::Add(dstReg, dstReg, srcDownReg, mask);
                AscendC::Reg::DataCopy(dstLineAddr + i * VL_ELEMS_FLOAT, dstReg, mask);
            }
        }
    }
    // axisNumInUb_>=3 时，H 数据完整在 tmpLocal 中，在 tmpLocal 内部做 H 镜像累加
    // 将 padding 行累加到对应的 self 行: tmpLocal[(h+leftPadH)*padWI] += tmpLocal[mirrorH*padWI]
    // __aicore__ inline void AccumulateHMirrorsInPlace(
    //     const LocalTensor<PromoteDataT>& tmpLocal, uint32_t padWI)
    __aicore__ inline void AccumulateHMirrorsInPlace(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t cOffset,
                                                     uint32_t padWI)
    {
        const uint32_t outH = tilingData_->outShape[dimNum_ - CONST2];
        const uint32_t inH = tilingData_->inShape[dimNum_ - CONST2];
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t rightPadH = tilingData_->rightPad[dimNum_ - CONST2];
        const uint32_t inW = tilingData_->inShape[dimNum_ - 1];

        if (!has2DPadding) {
            return;
        }

        __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr()) +
                                              cOffset;

        for (uint32_t h = 0; h < outH; h++) {
            __local_mem__ PromoteDataT* selfAddr = tmpAddr + (h + leftPadH) * padWI;
            bool hasTopMirror = (modeOffset_ == 0) ? (h > 0 && h <= leftPadH) : (h < leftPadH);
            bool hasBottomMirror = (modeOffset_ == 0) ? (rightPadH > 0 && h >= outH - rightPadH - 1 && h <= outH - 2) :
                                                        (rightPadH > 0 && h >= outH - rightPadH);
            // 上镜像: padding 行 mirrorH 映射到 self 行 h
            if (hasTopMirror && !hasBottomMirror) {
                uint32_t mirrorH = leftPadH - modeOffset_ - h;
                AddLocalLineInTmpBuf(selfAddr, tmpAddr + mirrorH * padWI, inW);
            }
            // 下镜像: padding 行 mirrorH 映射到 self 行 h
            if (hasBottomMirror && !hasTopMirror) {
                uint32_t mirrorH = leftPadH + 2 * outH - 2 + modeOffset_ - h;
                AddLocalLineInTmpBuf(selfAddr, tmpAddr + mirrorH * padWI, inW);
            }
            // 上、下镜像: padding 行 mirrorUpH、mirrorDownH 映射到 self 行 h
            if (hasBottomMirror && hasTopMirror) {
                uint32_t mirrorUpH = leftPadH - modeOffset_ - h;
                uint32_t mirrorDownH = leftPadH + 2 * outH - 2 + modeOffset_ - h;
                AddLocalUpAndDownLineInTmpBuf(selfAddr, tmpAddr + mirrorUpH * padWI, tmpAddr + mirrorDownH * padWI,
                                              inW);
            }
        }
    }

    // axisNumInUb_>=4 时，C 数据完整在 tmpLocal 中，在 tmpLocal 内部做 C 镜像累加
    // 将 padding C slice 累加到对应的 self C slice
    // tmpLocal 指向单个 N slice 的起始位置，布局: inShape[C] × inShape[H] × padWI
    __aicore__ inline void AccumulateCMirrorsInPlace(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHIn,
                                                     uint32_t padWI)
    {
        const uint32_t outC = tilingData_->outShape[dimNum_ - CONST3];
        const uint32_t inC = tilingData_->inShape[dimNum_ - CONST3];
        const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
        const uint32_t rightPadC = tilingData_->rightPad[dimNum_ - CONST3];
        const uint32_t cSliceSize = dimHIn * padWI; // 一个 C slice 的元素数

        if (!has3DPadding) {
            return;
        }

        __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr());

        for (uint32_t c = 0; c < outC; c++) {
            __local_mem__ PromoteDataT* selfAddr = tmpAddr + (c + leftPadC) * cSliceSize;

            bool hasTopMirror = (modeOffset_ == 0) ? (c > 0 && c <= leftPadC) : (c < leftPadC);
            bool hasBottomMirror = (modeOffset_ == 0) ? (rightPadC > 0 && c >= outC - rightPadC - 1 && c <= outC - 2) :
                                                        (rightPadC > 0 && c >= outC - rightPadC);
            // C 上镜像: padding C slice mirrorC 映射到 self C slice c
            if (hasTopMirror && !hasBottomMirror) {
                uint32_t mirrorC = leftPadC - modeOffset_ - c;
                AddLocalLineInTmpBuf(selfAddr, tmpAddr + mirrorC * cSliceSize, cSliceSize);
            }

            // C 下镜像: padding C slice mirrorC 映射到 self C slice c
            if (hasBottomMirror && !hasTopMirror) {
                uint32_t mirrorC = leftPadC + 2 * outC - 2 + modeOffset_ - c;
                AddLocalLineInTmpBuf(selfAddr, tmpAddr + mirrorC * cSliceSize, cSliceSize);
            }
            // C 上、下镜像: padding C slice mirrorUpC、mirrorDownC 映射到 self C slice c
            if (hasTopMirror && hasBottomMirror) {
                uint32_t mirrorUpC = leftPadC - modeOffset_ - c;
                uint32_t mirrorDownC = leftPadC + 2 * outC - 2 + modeOffset_ - c;
                AddLocalUpAndDownLineInTmpBuf(selfAddr, tmpAddr + mirrorUpC * cSliceSize,
                                              tmpAddr + mirrorDownC * cSliceSize, cSliceSize);
            }
        }
    }

    // ========== 主pad 函数: 高维镜像不改变 C 范围，读取整块 ==========

    // 主pad N 镜像: 读取 mirrorN 位置的整块 dimCNum × dimHIn × padWI
    __aicore__ inline void ProcessMainPadN(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimCNum, uint32_t dimHIn,
                                           uint32_t padWI)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        MirrorCondition condN = CalcMirrorCondition(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                    tilingData_->leftPad[dimNum_ - CONST4],
                                                    tilingData_->rightPad[dimNum_ - CONST4]);

        // 计算基础 GM 地址 (不含 N 维度)
        uint64_t baseGmAddr = 0;
        for (uint8_t i = 0; i < dimNum_ - CONST4; i++) {
            baseGmAddr += inIndex_[i] * tilingData_->inStride[i];
        }
        baseGmAddr += inIndex_[dimNum_ - CONST3] * tilingData_->inStride[dimNum_ - CONST3];

        if (condN.hasTop) {
            uint64_t gmAddr = baseGmAddr + condN.mirrorTop * tilingData_->inStride[dimNum_ - CONST4];
            CopyAndAddBlockFromGM(tmpLocal, gmAddr, dimCNum, dimHIn, padWI);
        }

        if (condN.hasBottom) {
            uint64_t gmAddr = baseGmAddr + condN.mirrorBottom * tilingData_->inStride[dimNum_ - CONST4];
            CopyAndAddBlockFromGM(tmpLocal, gmAddr, dimCNum, dimHIn, padWI);
        }
    }

    // 主pad D5 镜像: 读取 mirrorD5 位置的整块
    __aicore__ inline void ProcessMainPadD5(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimCNum,
                                            uint32_t dimHIn, uint32_t padWI)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        MirrorCondition condD5 = CalcMirrorCondition(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                     tilingData_->leftPad[dimNum_ - CONST5],
                                                     tilingData_->rightPad[dimNum_ - CONST5]);

        uint64_t baseGmAddr = inIndex_[dimNum_ - CONST4] * tilingData_->inStride[dimNum_ - CONST4] +
                              inIndex_[dimNum_ - CONST3] * tilingData_->inStride[dimNum_ - CONST3];

        if (condD5.hasTop) {
            uint64_t gmAddr = condD5.mirrorTop * tilingData_->inStride[dimNum_ - CONST5] + baseGmAddr;
            CopyAndAddBlockFromGM(tmpLocal, gmAddr, dimCNum, dimHIn, padWI);
        }

        if (condD5.hasBottom) {
            uint64_t gmAddr = condD5.mirrorBottom * tilingData_->inStride[dimNum_ - CONST5] + baseGmAddr;
            CopyAndAddBlockFromGM(tmpLocal, gmAddr, dimCNum, dimHIn, padWI);
        }
    }

    // 主pad N×D5 组合镜像: mirrorN × mirrorD5 位置的整块
    __aicore__ inline void ProcessMainPadNxD5(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimCNum,
                                              uint32_t dimHIn, uint32_t padWI)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];

        MirrorList nList = CollectMirrorPositions(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                  tilingData_->leftPad[dimNum_ - CONST4],
                                                  tilingData_->rightPad[dimNum_ - CONST4]);
        MirrorList d5List = CollectMirrorPositions(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                   tilingData_->leftPad[dimNum_ - CONST5],
                                                   tilingData_->rightPad[dimNum_ - CONST5]);

        // N×D5 组合
        for (uint8_t ni = 0; ni < nList.count; ni++) {
            for (uint8_t di = 0; di < d5List.count; di++) {
                uint64_t gmAddr = d5List.mirrors[di] * tilingData_->inStride[dimNum_ - CONST5] +
                                  nList.mirrors[ni] * tilingData_->inStride[dimNum_ - CONST4] +
                                  inIndex_[dimNum_ - CONST3] * tilingData_->inStride[dimNum_ - CONST3];
                CopyAndAddBlockFromGM(tmpLocal, gmAddr, dimCNum, dimHIn, padWI);
            }
        }
    }

    // ========== axisNumInUb_=4 主pad: D5 镜像不改变 N 范围，读取整块 dimNNum × dimCNum × dimHIn ==========

    __aicore__ inline void ProcessMainPadD5_UB4(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimNNum,
                                                uint32_t dimCNum, uint32_t dimHIn, uint32_t padWI)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        MirrorCondition condD5 = CalcMirrorCondition(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                     tilingData_->leftPad[dimNum_ - CONST5],
                                                     tilingData_->rightPad[dimNum_ - CONST5]);

        uint64_t baseGmAddr = inIndex_[dimNum_ - CONST4] * tilingData_->inStride[dimNum_ - CONST4];

        if (condD5.hasTop) {
            uint64_t gmAddr = condD5.mirrorTop * tilingData_->inStride[dimNum_ - CONST5] + baseGmAddr;
            CopyAndAddFullBlockFromGM(tmpLocal, gmAddr, dimNNum, dimCNum, dimHIn, padWI);
        }

        if (condD5.hasBottom) {
            uint64_t gmAddr = condD5.mirrorBottom * tilingData_->inStride[dimNum_ - CONST5] + baseGmAddr;
            CopyAndAddFullBlockFromGM(tmpLocal, gmAddr, dimNNum, dimCNum, dimHIn, padWI);
        }
    }

    // ========== axisNumInUb_=4 副pad: N 维度镜像，逐 N slice 读取 dimCNum × dimHIn ==========

    __aicore__ inline void ProcessSubPadN_UB4(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t globalN,
                                              uint32_t dimCNum, uint32_t dimHIn, uint32_t padWI, uint32_t dstOffset)
    {
        MirrorCondition condN = CalcMirrorCondition(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                    tilingData_->leftPad[dimNum_ - CONST4],
                                                    tilingData_->rightPad[dimNum_ - CONST4]);

        // 计算基础 GM 地址 (不含 N 维度)
        uint64_t baseGmAddr = 0;
        for (uint8_t i = 0; i < dimNum_ - CONST4; i++) {
            baseGmAddr += inIndex_[i] * tilingData_->inStride[i];
        }

        // N 上镜像
        if (condN.hasTop) {
            uint64_t gmAddr = baseGmAddr + condN.mirrorTop * tilingData_->inStride[dimNum_ - CONST4];
            CopyAndAddBlockFromGM(tmpLocal[dstOffset], gmAddr, dimCNum, dimHIn, padWI);
            if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
                ProcessSubPadD5xN_UB4(tmpLocal, condN.mirrorTop, dimCNum, dimHIn, padWI, dstOffset);
            }
        }

        // N 下镜像
        if (condN.hasBottom) {
            uint64_t gmAddr = baseGmAddr + condN.mirrorBottom * tilingData_->inStride[dimNum_ - CONST4];
            CopyAndAddBlockFromGM(tmpLocal[dstOffset], gmAddr, dimCNum, dimHIn, padWI);
            if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
                ProcessSubPadD5xN_UB4(tmpLocal, condN.mirrorBottom, dimCNum, dimHIn, padWI, dstOffset);
            }
        }
    }

    // 副pad D5×N 组合: mirrorD5 位置的 mirrorN slice
    __aicore__ inline void ProcessSubPadD5xN_UB4(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t mirrorN,
                                                 uint32_t dimCNum, uint32_t dimHIn, uint32_t padWI, uint32_t dstOffset)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        MirrorCondition condD5 = CalcMirrorCondition(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                     tilingData_->leftPad[dimNum_ - CONST5],
                                                     tilingData_->rightPad[dimNum_ - CONST5]);
        if (condD5.hasTop) {
            uint64_t gmAddr = condD5.mirrorTop * tilingData_->inStride[dimNum_ - CONST5] +
                              mirrorN * tilingData_->inStride[dimNum_ - CONST4];
            CopyAndAddBlockFromGM(tmpLocal[dstOffset], gmAddr, dimCNum, dimHIn, padWI);
        }
        if (condD5.hasBottom) {
            uint64_t gmAddr = condD5.mirrorBottom * tilingData_->inStride[dimNum_ - CONST5] +
                              mirrorN * tilingData_->inStride[dimNum_ - CONST4];
            CopyAndAddBlockFromGM(tmpLocal[dstOffset], gmAddr, dimCNum, dimHIn, padWI);
        }
    }

    // ========== 副pad 函数: C 维度镜像，逐 C slice 读取 plane ==========

    // 副pad C 镜像: 对单个 C slice，读取 mirrorC 位置的 plane
    // 同时处理 C 镜像位置的高维镜像 (N×C, D5×C, N×D5×C)
    __aicore__ inline void ProcessSubPadC(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t globalC, uint32_t dimHIn,
                                          uint32_t padWI, uint32_t dstOffset)
    {
        MirrorList cList = CollectMirrorPositions(globalC, tilingData_->outShape[dimNum_ - CONST3],
                                                  tilingData_->leftPad[dimNum_ - CONST3],
                                                  tilingData_->rightPad[dimNum_ - CONST3]);

        for (uint8_t ci = 0; ci < cList.count; ci++) {
            uint32_t mirrorC = cList.mirrors[ci];
            uint64_t gmAddr = CalcGMAddrWithC(mirrorC);

            CopyAndAddPlaneFromGM(tmpLocal, gmAddr, dimHIn, padWI, dstOffset);

            if (dimNum_ >= MIN_DIM_FOR_N_PAD && has4DPadding) {
                ProcessSubPadNxC(tmpLocal, mirrorC, dimHIn, padWI, dstOffset);
            }
            if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
                ProcessSubPadD5xC(tmpLocal, mirrorC, dimHIn, padWI, dstOffset);
            }
            if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has4DPadding && has5DPadding) {
                ProcessSubPadNxD5xC(tmpLocal, mirrorC, dimHIn, padWI, dstOffset);
            }
        }
    }

    // 副pad N×C: mirrorN 位置的 mirrorC plane
    __aicore__ inline void ProcessSubPadNxC(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t mirrorC,
                                            uint32_t dimHIn, uint32_t padWI, uint32_t dstOffset)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        MirrorList nList = CollectMirrorPositions(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                  tilingData_->leftPad[dimNum_ - CONST4],
                                                  tilingData_->rightPad[dimNum_ - CONST4]);

        for (uint8_t ni = 0; ni < nList.count; ni++) {
            uint64_t gmAddr = CalcGMAddrWithNC(nList.mirrors[ni], mirrorC);
            CopyAndAddPlaneFromGM(tmpLocal, gmAddr, dimHIn, padWI, dstOffset);
        }
    }

    // 副pad D5×C: mirrorD5 位置的 mirrorC plane
    __aicore__ inline void ProcessSubPadD5xC(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t mirrorC,
                                             uint32_t dimHIn, uint32_t padWI, uint32_t dstOffset)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        MirrorList d5List = CollectMirrorPositions(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                   tilingData_->leftPad[dimNum_ - CONST5],
                                                   tilingData_->rightPad[dimNum_ - CONST5]);

        for (uint8_t di = 0; di < d5List.count; di++) {
            uint64_t gmAddr = CalcGMAddrWithD5NC(d5List.mirrors[di], inIndex_[dimNum_ - CONST4], mirrorC);
            CopyAndAddPlaneFromGM(tmpLocal, gmAddr, dimHIn, padWI, dstOffset);
        }
    }

    // 副pad N×D5×C: mirrorN × mirrorD5 位置的 mirrorC plane
    __aicore__ inline void ProcessSubPadNxD5xC(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t mirrorC,
                                               uint32_t dimHIn, uint32_t padWI, uint32_t dstOffset)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];

        MirrorList nList = CollectMirrorPositions(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                  tilingData_->leftPad[dimNum_ - CONST4],
                                                  tilingData_->rightPad[dimNum_ - CONST4]);
        MirrorList d5List = CollectMirrorPositions(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                   tilingData_->leftPad[dimNum_ - CONST5],
                                                   tilingData_->rightPad[dimNum_ - CONST5]);

        // N×D5×C 组合
        for (uint8_t ni = 0; ni < nList.count; ni++) {
            for (uint8_t di = 0; di < d5List.count; di++) {
                uint64_t gmAddr = CalcGMAddrWithD5NC(d5List.mirrors[di], nList.mirrors[ni], mirrorC);
                CopyAndAddPlaneFromGM(tmpLocal, gmAddr, dimHIn, padWI, dstOffset);
            }
        }
    }

    // axisNumInUb_=3 专用: 批量从 GM 累加高维镜像到 tmpLocal
    // 主pad: 高维镜像不改变 C 范围，读取整块 dimCNum × dimHIn × padWI
    // 副pad: C 维度镜像，逐 C slice 读取 1 × dimHIn × padWI plane
    __aicore__ inline void GradAccumulateHighDimBulk(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimCNum,
                                                     uint32_t dimHIn, uint32_t padWI)
    {
        const uint32_t sliceSize = dimHIn * padWI;

        // ========== 主pad: 高维镜像不改变 C 范围 ==========
        if (dimNum_ >= MIN_DIM_FOR_N_PAD && has4DPadding) {
            ProcessMainPadN(tmpLocal, dimCNum, dimHIn, padWI);
        }
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
            ProcessMainPadD5(tmpLocal, dimCNum, dimHIn, padWI);
        }
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has4DPadding && has5DPadding) {
            ProcessMainPadNxD5(tmpLocal, dimCNum, dimHIn, padWI);
        }

        // ========== 副pad: C 维度镜像 ==========
        if (has3DPadding) {
            uint64_t savedOutC = outIndex_[dimNum_ - CONST3];
            uint64_t savedInC = inIndex_[dimNum_ - CONST3];

            for (uint32_t c = 0; c < dimCNum; c++) {
                uint32_t globalC = savedOutC + c;
                outIndex_[dimNum_ - CONST3] = globalC;
                inIndex_[dimNum_ - CONST3] = savedInC + c;
                uint32_t dstOffset = c * sliceSize;

                ProcessSubPadC(tmpLocal, globalC, dimHIn, padWI, dstOffset);
            }

            outIndex_[dimNum_ - CONST3] = savedOutC;
            inIndex_[dimNum_ - CONST3] = savedInC;
        }
    }

    // axisNumInUb_=4 专用: 批量从 GM 累加高维镜像到 tmpLocal
    // tmpLocal 布局: dimNNum × dimCNum × dimHIn × padWI
    // 主pad: D5 镜像不改变 N 范围，读取整块 dimNNum × dimCNum × dimHIn × padWI
    // 副pad: N 维度镜像，逐 N slice 读取 dimCNum × dimHIn × padWI
    // ========== axisNumInUb_=2 主pad: 批量从 GM 读取纯高维镜像 (不含 H) ==========
    __aicore__ inline void GradAccumulateHighDimBulk_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                         uint32_t padWI)
    {
        const uint32_t leftPadH = tilingData_->leftPad[dimNum_ - CONST2];
        const uint32_t hStartInGradY = outIndex_[dimNum_ - CONST2] + leftPadH;

        // C 镜像 (dimNum_ >= 3)
        if (dimNum_ >= MIN_DIM_FOR_C_PAD && has3DPadding) {
            ProcessMainPadC_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }

        // N 镜像 (dimNum_ >= 4)
        if (dimNum_ >= MIN_DIM_FOR_N_PAD && has4DPadding) {
            ProcessMainPadN_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }

        // D5 镜像 (dimNum_ >= 5)
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
            ProcessMainPadD5_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }

        // C×N 组合 (dimNum_ >= 4)
        if (dimNum_ >= MIN_DIM_FOR_N_PAD && has3DPadding && has4DPadding) {
            ProcessMainPadCxN_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }

        // C×D5 组合 (dimNum_ >= 5)
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has3DPadding && has5DPadding) {
            ProcessMainPadCxD5_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }

        // N×D5 组合 (dimNum_ >= 5)
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has4DPadding && has5DPadding) {
            ProcessMainPadNxD5_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }

        // C×N×D5 组合 (dimNum_ >= 5)
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has3DPadding && has4DPadding && has5DPadding) {
            ProcessMainPadCxNxD5_UB2(tmpLocal, dimHNum, padWI, hStartInGradY);
        }
    }

    // UB2 主pad: C 镜像 — 批量读 dimHNum 行
    __aicore__ inline void ProcessMainPadC_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                               uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outC = tilingData_->outShape[dimNum_ - CONST3];
        const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
        const uint32_t rightPadC = tilingData_->rightPad[dimNum_ - CONST3];

        MirrorCondition condC = CalcMirrorCondition(globalC, outC, leftPadC, rightPadC);
        if (condC.hasTop) {
            uint64_t gmAddr = CalcGMAddrWithC(condC.mirrorTop) +
                              hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
            CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
        }
        if (condC.hasBottom) {
            uint64_t gmAddr = CalcGMAddrWithC(condC.mirrorBottom) +
                              hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
            CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
        }
    }

    // UB2 主pad: N 镜像 — 批量读 dimHNum 行
    __aicore__ inline void ProcessMainPadN_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                               uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t outN = tilingData_->outShape[dimNum_ - CONST4];
        const uint32_t leftPadN = tilingData_->leftPad[dimNum_ - CONST4];
        const uint32_t rightPadN = tilingData_->rightPad[dimNum_ - CONST4];

        MirrorCondition condN = CalcMirrorCondition(globalN, outN, leftPadN, rightPadN);
        if (condN.hasTop) {
            uint64_t gmAddr = CalcGMAddrWithNC(condN.mirrorTop, inIndex_[dimNum_ - CONST3]) +
                              hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
            CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
        }
        if (condN.hasBottom) {
            uint64_t gmAddr = CalcGMAddrWithNC(condN.mirrorBottom, inIndex_[dimNum_ - CONST3]) +
                              hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
            CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
        }
    }

    // UB2 主pad: D5 镜像 — 批量读 dimHNum 行
    __aicore__ inline void ProcessMainPadD5_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        const uint32_t outD5 = tilingData_->outShape[dimNum_ - CONST5];
        const uint32_t leftPadD5 = tilingData_->leftPad[dimNum_ - CONST5];
        const uint32_t rightPadD5 = tilingData_->rightPad[dimNum_ - CONST5];

        MirrorCondition condD5 = CalcMirrorCondition(globalD5, outD5, leftPadD5, rightPadD5);
        if (condD5.hasTop) {
            uint64_t gmAddr = CalcGMAddrWithD5NC(condD5.mirrorTop, inIndex_[dimNum_ - CONST4],
                                                 inIndex_[dimNum_ - CONST3]) +
                              hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
            CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
        }
        if (condD5.hasBottom) {
            uint64_t gmAddr = CalcGMAddrWithD5NC(condD5.mirrorBottom, inIndex_[dimNum_ - CONST4],
                                                 inIndex_[dimNum_ - CONST3]) +
                              hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
            CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
        }
    }

    // UB2 主pad: C×N 组合 — 对每个 mirrorN，检查 C 镜像条件
    __aicore__ inline void ProcessMainPadCxN_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                 uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t outN = tilingData_->outShape[dimNum_ - CONST4];
        const uint32_t leftPadN = tilingData_->leftPad[dimNum_ - CONST4];
        const uint32_t rightPadN = tilingData_->rightPad[dimNum_ - CONST4];
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outC = tilingData_->outShape[dimNum_ - CONST3];
        const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
        const uint32_t rightPadC = tilingData_->rightPad[dimNum_ - CONST3];

        MirrorList nList = CollectMirrorPositions(globalN, outN, leftPadN, rightPadN);
        MirrorList cList = CollectMirrorPositions(globalC, outC, leftPadC, rightPadC);

        for (uint8_t ni = 0; ni < nList.count; ni++) {
            for (uint8_t ci = 0; ci < cList.count; ci++) {
                uint64_t gmAddr = CalcGMAddrWithNC(nList.mirrors[ni], cList.mirrors[ci]) +
                                  hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
                CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
            }
        }
    }

    // UB2 主pad: C×D5 组合 — 对每个 mirrorD5，检查 C 镜像条件
    __aicore__ inline void ProcessMainPadCxD5_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                  uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        const uint32_t outD5 = tilingData_->outShape[dimNum_ - CONST5];
        const uint32_t leftPadD5 = tilingData_->leftPad[dimNum_ - CONST5];
        const uint32_t rightPadD5 = tilingData_->rightPad[dimNum_ - CONST5];
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];
        const uint32_t outC = tilingData_->outShape[dimNum_ - CONST3];
        const uint32_t leftPadC = tilingData_->leftPad[dimNum_ - CONST3];
        const uint32_t rightPadC = tilingData_->rightPad[dimNum_ - CONST3];

        MirrorList d5List = CollectMirrorPositions(globalD5, outD5, leftPadD5, rightPadD5);
        MirrorList cList = CollectMirrorPositions(globalC, outC, leftPadC, rightPadC);

        for (uint8_t di = 0; di < d5List.count; di++) {
            for (uint8_t ci = 0; ci < cList.count; ci++) {
                uint64_t gmAddr = CalcGMAddrWithD5NC(d5List.mirrors[di], inIndex_[dimNum_ - CONST4],
                                                     cList.mirrors[ci]) +
                                  hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
                CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
            }
        }
    }

    // UB2 主pad: N×D5 组合 — 对每个 mirrorD5，检查 N 镜像条件
    __aicore__ inline void ProcessMainPadNxD5_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                  uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        const uint32_t outD5 = tilingData_->outShape[dimNum_ - CONST5];
        const uint32_t leftPadD5 = tilingData_->leftPad[dimNum_ - CONST5];
        const uint32_t rightPadD5 = tilingData_->rightPad[dimNum_ - CONST5];
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t outN = tilingData_->outShape[dimNum_ - CONST4];
        const uint32_t leftPadN = tilingData_->leftPad[dimNum_ - CONST4];
        const uint32_t rightPadN = tilingData_->rightPad[dimNum_ - CONST4];

        MirrorList d5List = CollectMirrorPositions(globalD5, outD5, leftPadD5, rightPadD5);
        MirrorList nList = CollectMirrorPositions(globalN, outN, leftPadN, rightPadN);

        for (uint8_t di = 0; di < d5List.count; di++) {
            for (uint8_t ni = 0; ni < nList.count; ni++) {
                uint64_t gmAddr = CalcGMAddrWithD5NC(d5List.mirrors[di], nList.mirrors[ni],
                                                     inIndex_[dimNum_ - CONST3]) +
                                  hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
                CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
            }
        }
    }

    // UB2 主pad: C×N×D5 组合 — 对每个 mirrorD5×mirrorN，检查 C 镜像条件
    __aicore__ inline void ProcessMainPadCxNxD5_UB2(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimHNum,
                                                    uint32_t padWI, uint32_t hStartInGradY)
    {
        const uint32_t globalD5 = outIndex_[dimNum_ - CONST5];
        const uint32_t globalN = outIndex_[dimNum_ - CONST4];
        const uint32_t globalC = outIndex_[dimNum_ - CONST3];

        // 使用辅助函数收集各维度镜像位置
        MirrorList d5List = CollectMirrorPositions(globalD5, tilingData_->outShape[dimNum_ - CONST5],
                                                   tilingData_->leftPad[dimNum_ - CONST5],
                                                   tilingData_->rightPad[dimNum_ - CONST5]);
        MirrorList nList = CollectMirrorPositions(globalN, tilingData_->outShape[dimNum_ - CONST4],
                                                  tilingData_->leftPad[dimNum_ - CONST4],
                                                  tilingData_->rightPad[dimNum_ - CONST4]);
        MirrorList cList = CollectMirrorPositions(globalC, tilingData_->outShape[dimNum_ - CONST3],
                                                  tilingData_->leftPad[dimNum_ - CONST3],
                                                  tilingData_->rightPad[dimNum_ - CONST3]);

        // 三重循环处理 D5×N×C 组合
        for (uint8_t di = 0; di < d5List.count; di++) {
            for (uint8_t ni = 0; ni < nList.count; ni++) {
                for (uint8_t ci = 0; ci < cList.count; ci++) {
                    uint64_t gmAddr = CalcGMAddrWithD5NC(d5List.mirrors[di], nList.mirrors[ni], cList.mirrors[ci]) +
                                      hStartInGradY * tilingData_->inStride[dimNum_ - CONST2];
                    CopyAndAddRowsFromGM(tmpLocal, gmAddr, dimHNum, padWI);
                }
            }
        }
    }

    __aicore__ inline void GradAccumulateHighDimBulk_UB4(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t dimNNum,
                                                         uint32_t dimCNum, uint32_t dimHIn, uint32_t padWI)
    {
        const uint32_t nSliceSize = dimCNum * dimHIn * padWI;

        // ========== 主pad: D5 镜像不改变 N 范围 ==========
        if (dimNum_ >= MIN_DIM_FOR_D5_PAD && has5DPadding) {
            ProcessMainPadD5_UB4(tmpLocal, dimNNum, dimCNum, dimHIn, padWI);
        }

        // ========== 副pad: N 维度镜像 ==========
        if (has4DPadding) {
            uint64_t savedOutN = outIndex_[dimNum_ - CONST4];
            uint64_t savedInN = inIndex_[dimNum_ - CONST4];

            for (uint32_t n = 0; n < dimNNum; n++) {
                uint32_t globalN = savedOutN + n;
                outIndex_[dimNum_ - CONST4] = globalN;
                inIndex_[dimNum_ - CONST4] = savedInN + n;
                uint32_t dstOffset = n * nSliceSize;

                ProcessSubPadN_UB4(tmpLocal, globalN, dimCNum, dimHIn, padWI, dstOffset);
            }

            outIndex_[dimNum_ - CONST4] = savedOutN;
            inIndex_[dimNum_ - CONST4] = savedInN;
        }
    }

    // 通用的从 GM 地址拷贝并累加一行数据
    __aicore__ inline void CopyAndAddLineFromGMAddr(__local_mem__ PromoteDataT* dstLineAddr, uint64_t gmAddr,
                                                    uint32_t inW)
    {
        // Step 1: 从 GM 拷贝镜像行到 srcLocal
        LocalTensor<T> inUbLocal = inQueue_.AllocTensor<T>();
        uint32_t inWAlign = CeilAlign(inW, BLK_ELEMS);
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = inW * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padExtParams{true, 0, 0, 0};
        DataCopyPad(inUbLocal, input_[gmAddr], copyParams, padExtParams);
        inQueue_.EnQue(inUbLocal);

        // Step 2: 使用 VF 指令进行 Cast 和累加
        LocalTensor<T> inUbLocal2 = inQueue_.DeQue<T>();
        constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(PromoteDataT);
        __local_mem__ T* srcAddr = reinterpret_cast<__local_mem__ T*>(inUbLocal2.GetPhyAddr());
        uint16_t loopCount = CeilDiv(inWAlign, VL_ELEMS_FLOAT);

        if constexpr (IsSameType<T, PromoteDataT>::value) {
            // float 类型：直接累加，无需 Cast
            __VEC_SCOPE__
            {
                uint32_t remainLen = inW;
                AscendC::Reg::MaskReg mask;
                AscendC::Reg::RegTensor<PromoteDataT> dstReg;
                AscendC::Reg::RegTensor<PromoteDataT> srcReg;

                for (uint16_t i = 0; i < loopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                    AscendC::Reg::DataCopy(dstReg, dstLineAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::DataCopy(srcReg, srcAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::Add(dstReg, dstReg, srcReg, mask);
                    AscendC::Reg::DataCopy(dstLineAddr + i * VL_ELEMS_FLOAT, dstReg, mask);
                }
            }
        } else {
            // float16/bfloat16 类型：先 Cast 再累加
            __VEC_SCOPE__
            {
                uint32_t remainLen = inW;
                AscendC::Reg::MaskReg mask;
                AscendC::Reg::RegTensor<PromoteDataT> dstReg;
                AscendC::Reg::RegTensor<PromoteDataT> srcCastReg;
                AscendC::Reg::RegTensor<T> srcReg;

                for (uint16_t i = 0; i < loopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                    // 使用 DIST_UNPACK_B16 模式读取 B16 类型数据到寄存器
                    AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg,
                                                                                       srcAddr + i * VL_ELEMS_FLOAT);
                    // Cast 到 PromoteDataT (float)
                    AscendC::Reg::Cast<PromoteDataT, T, CAST_TRAIT_0>(srcCastReg, srcReg, mask);
                    // 读取目标数据并累加
                    AscendC::Reg::DataCopy(dstReg, dstLineAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::Add(dstReg, dstReg, srcCastReg, mask);
                    AscendC::Reg::DataCopy(dstLineAddr + i * VL_ELEMS_FLOAT, dstReg, mask);
                }
            }
        }
        inQueue_.FreeTensor(inUbLocal2);
    }

    // 将 srcLocal (T 类型) Cast+Add 到 tmpLocal (PromoteDataT 类型)
    // 用于批量 GM 读取后的累加操作
    __aicore__ inline void AddSrcLocalToTmpLocal(const LocalTensor<PromoteDataT>& tmpLocal, uint32_t totalLen)
    {
        LocalTensor<T> inUbLocal = inQueue_.DeQue<T>();
        constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(PromoteDataT);
        uint16_t loopCount = CeilDiv(totalLen, VL_ELEMS_FLOAT);
        __local_mem__ PromoteDataT* tmpAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpLocal.GetPhyAddr());
        __local_mem__ T* srcAddr = reinterpret_cast<__local_mem__ T*>(inUbLocal.GetPhyAddr());

        if constexpr (IsSameType<T, PromoteDataT>::value) {
            __VEC_SCOPE__
            {
                uint32_t remainLen = totalLen;
                AscendC::Reg::MaskReg mask;
                AscendC::Reg::RegTensor<PromoteDataT> dstReg;
                AscendC::Reg::RegTensor<PromoteDataT> srcReg;
                for (uint16_t i = 0; i < loopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                    AscendC::Reg::DataCopy(dstReg, tmpAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::DataCopy(srcReg, srcAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::Add(dstReg, dstReg, srcReg, mask);
                    AscendC::Reg::DataCopy(tmpAddr + i * VL_ELEMS_FLOAT, dstReg, mask);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                uint32_t remainLen = totalLen;
                AscendC::Reg::MaskReg mask;
                AscendC::Reg::RegTensor<PromoteDataT> dstReg;
                AscendC::Reg::RegTensor<PromoteDataT> srcCastReg;
                AscendC::Reg::RegTensor<T> srcReg;
                for (uint16_t i = 0; i < loopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                    AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg,
                                                                                       srcAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::Cast<PromoteDataT, T, CAST_TRAIT_0>(srcCastReg, srcReg, mask);
                    AscendC::Reg::DataCopy(dstReg, tmpAddr + i * VL_ELEMS_FLOAT);
                    AscendC::Reg::Add(dstReg, dstReg, srcCastReg, mask);
                    AscendC::Reg::DataCopy(tmpAddr + i * VL_ELEMS_FLOAT, dstReg, mask);
                }
            }
        }
        inQueue_.FreeTensor(inUbLocal);
    }

    // GM 地址计算: 高维用 inIndex_, C 用指定值, H=0
    __aicore__ inline uint64_t CalcGMAddrWithC(uint32_t cIdx)
    {
        uint64_t addr = 0;
        for (uint8_t i = 0; i < dimNum_ - CONST3; i++) {
            addr += inIndex_[i] * tilingData_->inStride[i];
        }
        addr += cIdx * tilingData_->inStride[dimNum_ - CONST3];
        return addr;
    }

    // GM 地址计算: 指定 N, C, H=0
    __aicore__ inline uint64_t CalcGMAddrWithNC(uint32_t nIdx, uint32_t cIdx)
    {
        uint64_t addr = 0;
        for (uint8_t i = 0; i < dimNum_ - CONST4; i++) {
            addr += inIndex_[i] * tilingData_->inStride[i];
        }
        addr += nIdx * tilingData_->inStride[dimNum_ - CONST4];
        addr += cIdx * tilingData_->inStride[dimNum_ - CONST3];
        return addr;
    }

    // GM 地址计算: 指定 D5, N, C, H=0
    __aicore__ inline uint64_t CalcGMAddrWithD5NC(uint32_t d5Idx, uint32_t nIdx, uint32_t cIdx)
    {
        uint64_t addr = d5Idx * tilingData_->inStride[dimNum_ - CONST5] +
                        nIdx * tilingData_->inStride[dimNum_ - CONST4] + cIdx * tilingData_->inStride[dimNum_ - CONST3];
        return addr;
    }

    // 从 GM 批量读取整块 (dimNNum × dimCNum × dimHIn × padWI) 到 srcLocal，Cast+Add 到 tmpLocal
    // 用于 axisNumInUb_=4 主pad: D5 镜像不改变 N 范围，使用 2-level LoopMode
    __aicore__ inline void CopyAndAddFullBlockFromGM(const LocalTensor<PromoteDataT>& tmpLocal, uint64_t gmBaseAddr,
                                                     uint32_t dimNNum, uint32_t dimCNum, uint32_t dimHIn,
                                                     uint32_t padWI)
    {
        LocalTensor<T> inUbLocal = inQueue_.AllocTensor<T>();

        uint32_t inW = tilingData_->inShape[dimNum_ - 1];
        DataCopyExtParams copyParams;
        copyParams.blockCount = dimHIn;
        copyParams.blockLen = inW * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padExtParams{true, 0, 0, 0};

        LoopModeParams loopParams;
        loopParams.loop1Size = dimCNum;
        loopParams.loop1SrcStride = tilingData_->inStride[dimNum_ - CONST3] * sizeof(T);
        loopParams.loop1DstStride = dimHIn * padWI * sizeof(T);
        loopParams.loop2Size = dimNNum;
        loopParams.loop2SrcStride = tilingData_->inStride[dimNum_ - CONST4] * sizeof(T);
        loopParams.loop2DstStride = dimCNum * dimHIn * padWI * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad(inUbLocal, input_[gmBaseAddr], copyParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        inQueue_.EnQue(inUbLocal);

        uint32_t totalLen = dimNNum * dimCNum * dimHIn * padWI;
        AddSrcLocalToTmpLocal(tmpLocal, totalLen);
    }

    // 从 GM 批量读取整块 (dimCNum × dimHIn × padWI) 到 srcLocal，Cast+Add 到 tmpLocal
    // 用于主pad: 高维镜像不改变 C 范围
    __aicore__ inline void CopyAndAddBlockFromGM(const LocalTensor<PromoteDataT>& tmpLocal, uint64_t gmBaseAddr,
                                                 uint32_t dimCNum, uint32_t dimHIn, uint32_t padWI)
    {
        LocalTensor<T> inUbLocal = inQueue_.AllocTensor<T>();

        uint32_t inW = tilingData_->inShape[dimNum_ - 1];
        DataCopyExtParams copyParams;
        copyParams.blockCount = dimHIn;
        copyParams.blockLen = inW * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padExtParams{true, 0, 0, 0};

        if (dimCNum > 1) {
            LoopModeParams loopParams;
            loopParams.loop2Size = 1;
            loopParams.loop1Size = dimCNum;
            loopParams.loop1SrcStride = tilingData_->inStride[dimNum_ - CONST3] * sizeof(T);
            loopParams.loop1DstStride = dimHIn * padWI * sizeof(T);
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyPad(inUbLocal, input_[gmBaseAddr], copyParams, padExtParams);
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        } else {
            DataCopyPad(inUbLocal, input_[gmBaseAddr], copyParams, padExtParams);
        }
        inQueue_.EnQue(inUbLocal);

        uint32_t totalLen = dimCNum * dimHIn * padWI;
        AddSrcLocalToTmpLocal(tmpLocal, totalLen);
    }

    // 从 GM 读取单个 C plane (1 × dimHIn × padWI) 到 srcLocal，Cast+Add 到 tmpLocal 的指定偏移
    // 用于副pad: C 维度镜像
    __aicore__ inline void CopyAndAddPlaneFromGM(
        const LocalTensor<PromoteDataT>& tmpLocal, /*const LocalTensor<T>& srcLocal,*/
        uint64_t gmBaseAddr, uint32_t dimHIn, uint32_t padWI, uint32_t dstOffset)
    {
        LocalTensor<T> inUbLocal = inQueue_.AllocTensor<T>();

        uint32_t inW = tilingData_->inShape[dimNum_ - 1];
        DataCopyExtParams copyParams;
        copyParams.blockCount = dimHIn;
        copyParams.blockLen = inW * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padExtParams{true, 0, 0, 0};
        DataCopyPad(inUbLocal, input_[gmBaseAddr], copyParams, padExtParams);
        inQueue_.EnQue(inUbLocal);

        uint32_t totalLen = dimHIn * padWI;
        AddSrcLocalToTmpLocal(tmpLocal[dstOffset], totalLen);
    }

    // 从 GM 读取 dimHNum 连续 H 行到 srcLocal，Cast+Add 到 tmpLocal
    // 用于 axisNumInUb_=2 主pad: 高维镜像不改变 H 范围，批量读取 dimHNum 行
    __aicore__ inline void CopyAndAddRowsFromGM(const LocalTensor<PromoteDataT>& tmpLocal, uint64_t gmBaseAddr,
                                                uint32_t dimHNum, uint32_t padWI)
    {
        LocalTensor<T> inUbLocal = inQueue_.AllocTensor<T>();

        uint32_t inW = tilingData_->inShape[dimNum_ - 1];
        DataCopyExtParams copyParams;
        copyParams.blockCount = dimHNum;
        copyParams.blockLen = inW * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T> padExtParams{true, 0, 0, 0};
        DataCopyPad(inUbLocal, input_[gmBaseAddr], copyParams, padExtParams);
        inQueue_.EnQue(inUbLocal);

        uint32_t totalLen = dimHNum * padWI;
        AddSrcLocalToTmpLocal(tmpLocal, totalLen);
    }

    // W 维度梯度累加，从 tmpBuf_ 读取累加后的数据 (仅 axisNumInUb_=2 时使用)
    // tmpAddr: PromoteDataT 类型 (float)，包含原始 grad_y 数据 + 高维镜像累加结果
    // dstAddr: 输出 grad_x (T 类型)
    // 使用 VF 指令和 gather 进行向量化处理
    __aicore__ inline void GradProcessLineFromTmpBuf(__local_mem__ T* dstAddr, __local_mem__ PromoteDataT* tmpAddr,
                                                     uint32_t outW, uint32_t inW, uint32_t leftPad, uint32_t rightPad)
    {
        constexpr uint32_t VL_ELEMS_FLOAT = VL_SIZE / sizeof(PromoteDataT);
        // using IdxType = uint32_t;  // float 对应 uint32_t 索引

        // 计算左 pad 有效范围 (需要累加左镜像的输出位置)
        // reflect (modeOffset_=0): w in [1, leftPad], 即 w = 1, 2, ..., leftPad，共 leftPad 个元素
        // symmetric (modeOffset_=1): w in [0, leftPad), 即 w = 0, 1, ..., leftPad-1，共 leftPad 个元素
        uint32_t leftStart = (modeOffset_ == 0) ? 1 : 0;
        uint32_t leftEnd = (modeOffset_ == 0) ? (leftPad + 1) : leftPad; // 不包含 leftEnd

        // 计算右 pad 有效范围 (需要累加右镜像的输出位置)
        // reflect (modeOffset_=0): w in [outW - rightPad - 1, outW - 2], 共 rightPad 个元素
        // symmetric (modeOffset_=1): w in [outW - rightPad, outW - 1], 共 rightPad 个元素
        uint32_t rightStart = (rightPad > 0) ? ((modeOffset_ == 0) ? (outW - rightPad - 1) : (outW - rightPad)) : outW;
        uint32_t rightEnd = (rightPad > 0) ? ((modeOffset_ == 0) ? (outW - 1) : outW) : outW; // 不包含 rightEnd

        // ========== 阶段1: 在 tmpAddr 上原地完成左右 pad 的 gather 累加 ==========

        // ========== 1a. 处理左 pad 区域 ==========
        if (leftPad > 0 && leftEnd > leftStart) {
            uint32_t leftLen = leftEnd - leftStart;
            uint16_t leftMainLoops = leftLen / VL_ELEMS_FLOAT;
            uint16_t leftTailLen = leftLen % VL_ELEMS_FLOAT;

            uint32_t mainMaskLen = VL_ELEMS_FLOAT;
            uint32_t tailMaskLen = leftTailLen;
            __local_mem__ PromoteDataT* srcStartAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(
                tmpAddr + leftStart + leftPad);
            __local_mem__ PromoteDataT* dstStartAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(
                tmpAddr + leftStart + leftPad);
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<PromoteDataT> selfReg;
                AscendC::Reg::RegTensor<PromoteDataT> mirrorReg;
                AscendC::Reg::RegTensor<PromoteDataT> resultReg;
                AscendC::Reg::RegTensor<uint32_t> idxReg;
                AscendC::Reg::RegTensor<int32_t> arangeReg;
                AscendC::Reg::RegTensor<uint32_t> baseIdxReg;
                AscendC::Reg::UnalignReg uSrc;
                AscendC::Reg::UnalignReg uDst;

                AscendC::Reg::MaskReg maskMain = AscendC::Reg::UpdateMask<PromoteDataT>(mainMaskLen);
                AscendC::Reg::MaskReg maskTail = AscendC::Reg::UpdateMask<PromoteDataT>(tailMaskLen);

                AscendC::Reg::Arange(arangeReg, 0);
                idxReg = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(arangeReg);
                AscendC::Reg::DataCopyUnAlignPre(uSrc, srcStartAddr);

                for (uint16_t i = 0; i < leftMainLoops; i++) {
                    uint32_t curStart = leftStart + i * VL_ELEMS_FLOAT;
                    AscendC::Reg::DataCopyUnAlign(selfReg, uSrc, srcStartAddr, VL_ELEMS_FLOAT);

                    uint32_t baseIdx = static_cast<uint32_t>(leftPad - modeOffset_ - curStart);
                    AscendC::Reg::Duplicate(baseIdxReg, baseIdx);
                    AscendC::Reg::Sub(baseIdxReg, baseIdxReg, idxReg, maskMain);
                    AscendC::Reg::DataCopyGather(mirrorReg, tmpAddr, baseIdxReg, maskMain);

                    AscendC::Reg::Add(resultReg, selfReg, mirrorReg, maskMain);
                    AscendC::Reg::DataCopyUnAlign(dstStartAddr, resultReg, uDst, VL_ELEMS_FLOAT);
                }
                uint32_t curStart = leftStart + leftMainLoops * VL_ELEMS_FLOAT;
                AscendC::Reg::DataCopyUnAlign(selfReg, uSrc, srcStartAddr, leftTailLen);

                uint32_t baseIdx = static_cast<uint32_t>(leftPad - modeOffset_ - curStart);
                AscendC::Reg::Duplicate(baseIdxReg, baseIdx);
                AscendC::Reg::Sub(baseIdxReg, baseIdxReg, idxReg, maskTail);
                AscendC::Reg::DataCopyGather(mirrorReg, tmpAddr, baseIdxReg, maskTail);

                AscendC::Reg::Add(resultReg, selfReg, mirrorReg, maskTail);
                AscendC::Reg::DataCopyUnAlign(dstStartAddr, resultReg, uDst, leftTailLen);
                AscendC::Reg::DataCopyUnAlignPost(dstStartAddr, uDst, 0);
            }
        }

        // ========== 1b. 处理右 pad 区域 ==========
        if (rightPad > 0 && rightEnd > rightStart) {
            uint32_t rightLen = rightEnd - rightStart;
            uint16_t rightMainLoops = rightLen / VL_ELEMS_FLOAT;
            uint16_t rightTailLen = rightLen % VL_ELEMS_FLOAT;

            uint32_t mainMaskLen = VL_ELEMS_FLOAT;
            uint32_t tailMaskLen = rightTailLen;
            __local_mem__ PromoteDataT* srcStartAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(
                tmpAddr + rightStart + leftPad);
            __local_mem__ PromoteDataT* dstStartAddr = reinterpret_cast<__local_mem__ PromoteDataT*>(
                tmpAddr + rightStart + leftPad);
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<PromoteDataT> selfReg;
                AscendC::Reg::RegTensor<PromoteDataT> mirrorReg;
                AscendC::Reg::RegTensor<PromoteDataT> resultReg;
                AscendC::Reg::RegTensor<uint32_t> idxReg;
                AscendC::Reg::RegTensor<int32_t> arangeReg;
                AscendC::Reg::RegTensor<uint32_t> baseIdxReg;
                AscendC::Reg::UnalignReg uSrc;
                AscendC::Reg::UnalignReg uDst;

                AscendC::Reg::MaskReg maskMain = AscendC::Reg::UpdateMask<PromoteDataT>(mainMaskLen);
                AscendC::Reg::MaskReg maskTail = AscendC::Reg::UpdateMask<PromoteDataT>(tailMaskLen);

                AscendC::Reg::Arange(arangeReg, 0);
                idxReg = reinterpret_cast<AscendC::Reg::RegTensor<uint32_t>&>(arangeReg);
                AscendC::Reg::DataCopyUnAlignPre(uSrc, srcStartAddr);

                for (uint16_t i = 0; i < rightMainLoops; i++) {
                    uint32_t curStart = rightStart + i * VL_ELEMS_FLOAT;
                    AscendC::Reg::DataCopyUnAlign(selfReg, uSrc, srcStartAddr, VL_ELEMS_FLOAT);

                    uint32_t baseIdx = static_cast<uint32_t>(2 * outW + leftPad - 2 + modeOffset_ - curStart);
                    AscendC::Reg::Duplicate(baseIdxReg, baseIdx);
                    AscendC::Reg::Sub(baseIdxReg, baseIdxReg, idxReg, maskMain);
                    AscendC::Reg::DataCopyGather(mirrorReg, tmpAddr, baseIdxReg, maskMain);

                    AscendC::Reg::Add(resultReg, selfReg, mirrorReg, maskMain);
                    AscendC::Reg::DataCopyUnAlign(dstStartAddr, resultReg, uDst, VL_ELEMS_FLOAT);
                }
                uint32_t curStart = rightStart + rightMainLoops * VL_ELEMS_FLOAT;
                AscendC::Reg::DataCopyUnAlign(selfReg, uSrc, srcStartAddr, rightTailLen);

                uint32_t baseIdx = static_cast<uint32_t>(2 * outW + leftPad - 2 + modeOffset_ - curStart);
                AscendC::Reg::Duplicate(baseIdxReg, baseIdx);
                AscendC::Reg::Sub(baseIdxReg, baseIdxReg, idxReg, maskTail);
                AscendC::Reg::DataCopyGather(mirrorReg, tmpAddr, baseIdxReg, maskTail);

                AscendC::Reg::Add(resultReg, selfReg, mirrorReg, maskTail);
                AscendC::Reg::DataCopyUnAlign(dstStartAddr, resultReg, uDst, rightTailLen);
                AscendC::Reg::DataCopyUnAlignPost(dstStartAddr, uDst, 0);
            }
        }

        // ========== 阶段2: 从 tmpAddr+leftPad 非对齐读 outW 个元素，对齐搬出到 dstAddr ==========

        uint16_t outLoopCount = CeilDiv(outW, VL_ELEMS_FLOAT);
        __local_mem__ PromoteDataT* srcAddr2 = reinterpret_cast<__local_mem__ PromoteDataT*>(tmpAddr + leftPad);
        __local_mem__ T* dstAddr2 = reinterpret_cast<__local_mem__ T*>(dstAddr);

        if constexpr (IsSameType<T, PromoteDataT>::value) {
            // float: 非对齐读 → DataCopy 对齐写
            __VEC_SCOPE__
            {
                uint32_t remainLen = outW;
                AscendC::Reg::MaskReg mask;
                AscendC::Reg::RegTensor<PromoteDataT> dataReg;
                AscendC::Reg::UnalignReg uSrc;
                AscendC::Reg::DataCopyUnAlignPre(uSrc, srcAddr2);
                for (uint16_t i = 0; i < outLoopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                    AscendC::Reg::DataCopyUnAlign(dataReg, uSrc, srcAddr2, VL_ELEMS_FLOAT);
                    AscendC::Reg::DataCopy(dstAddr2 + i * VL_ELEMS_FLOAT, dataReg, mask);
                }
            }
        } else {
            // fp16/bf16: 非对齐读 → Cast(fp32→fp16) → DataCopy<DIST_PACK_B32> 对齐写
            __VEC_SCOPE__
            {
                uint32_t remainLen = outW;
                AscendC::Reg::MaskReg mask;
                AscendC::Reg::RegTensor<PromoteDataT> dataReg;
                AscendC::Reg::RegTensor<T> outReg;
                AscendC::Reg::UnalignReg uSrc;
                AscendC::Reg::DataCopyUnAlignPre(uSrc, srcAddr2);
                for (uint16_t i = 0; i < outLoopCount; i++) {
                    mask = AscendC::Reg::UpdateMask<PromoteDataT>(remainLen);
                    AscendC::Reg::DataCopyUnAlign(dataReg, uSrc, srcAddr2, VL_ELEMS_FLOAT);
                    AscendC::Reg::Cast<T, PromoteDataT, CAST_TRAIT_1>(outReg, dataReg, mask);
                    AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dstAddr2 + i * VL_ELEMS_FLOAT,
                                                                                      outReg, mask);
                }
            }
        }
    }

    // 计算单维度的镜像条件
    // globalPos: 当前在输出空间的位置
    // outDimSize: 输出维度大小
    // leftPad: 左/上 padding 大小
    // rightPad: 右/下 padding 大小
    // modeOffset_: 0 表示 reflect 模式，1 表示 symmetric 模式
    __aicore__ inline MirrorCondition CalcMirrorCondition(uint32_t globalPos, uint32_t outDimSize, uint32_t leftPad,
                                                          uint32_t rightPad)
    {
        MirrorCondition cond;
        // 上/左镜像条件
        cond.hasTop = (modeOffset_ == 0) ? (globalPos > 0 && globalPos <= leftPad) : (globalPos < leftPad);
        // 下/右镜像条件
        cond.hasBottom = (modeOffset_ == 0) ?
                             (rightPad > 0 && globalPos >= outDimSize - rightPad - MIRROR_BOUNDARY_OFFSET_1 &&
                              globalPos <= outDimSize - MIRROR_BOUNDARY_OFFSET_2) :
                             (rightPad > 0 && globalPos >= outDimSize - rightPad);
        // 镜像位置计算
        cond.mirrorTop = leftPad - modeOffset_ - globalPos;
        cond.mirrorBottom = leftPad + MIRROR_BOUNDARY_OFFSET_2 * outDimSize - MIRROR_BOUNDARY_OFFSET_2 + modeOffset_ -
                            globalPos;
        return cond;
    }

    // 收集单维度的所有镜像位置
    __aicore__ inline MirrorList CollectMirrorPositions(uint32_t globalPos, uint32_t outDimSize, uint32_t leftPad,
                                                        uint32_t rightPad)
    {
        MirrorList list;
        list.count = 0;
        MirrorCondition cond = CalcMirrorCondition(globalPos, outDimSize, leftPad, rightPad);
        if (cond.hasTop) {
            list.mirrors[list.count++] = cond.mirrorTop;
        }
        if (cond.hasBottom) {
            list.mirrors[list.count++] = cond.mirrorBottom;
        }
        return list;
    }

    __aicore__ inline void CopyOut(const LocalTensor<T>& outUbLocal)
    {
        // Calculate output address in grad_x
        uint64_t outAddr = 0;
        for (uint8_t i = 0; i < dimNum_; i++) {
            outAddr += outIndex_[i] * tilingData_->outStride[i];
        }

        uint32_t blockCount = inCopyLen_[ubAxis_];
        for (uint8_t i = ubAxis_ + 1; i < dimNum_ - 1; i++) {
            blockCount = blockCount * tilingData_->outShape[i];
        }

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = blockCount;
        copyOutParams.blockLen = tilingData_->outShape[dimNum_ - 1] * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        DataCopyPad(output_[outAddr], outUbLocal, copyOutParams);
    }
};
} // namespace PadV3Grad
#endif
