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
 * \file tile_with_axis.h
 * \brief TileWithAxis 算子 Kernel 类定义（arch35 架构）
 *
 * 设计依据: DESIGN.md v1.6 3.6 节
 *
 * 核心设计 (迭代二: 全 18 TilingKey, NDDMA CopyIn):
 *   - 3D 模型: inShape=[outerDim,1,rowLength], outShape=[outerDim,tiles,rowLength]
 *   - CopyIn ubAxis=0: 3D NDDMA (dim0=rowLength, dim1=tiles, dim2=ubFactor)
 *   - CopyIn ubAxis=1: 2D NDDMA (dim0=rowLength, dim1=ubFactor 实际 tileCount)
 *   - CopyIn ubAxis=2: DataCopyPad blockCount=1
 *   - CopyOut: DataCopyPad blockCount=1 (所有 ubAxis 统一)
 *   - CopyIn 中必须按 block 实际长度覆写 loopSize (末块问题)
 *   - 同步链: MTE2_MTE3 / MTE3_MTE2
 *   - TilingData 通过 const TilingData* td_ 指针访问
 *   - TBuf Ping-Pong 双缓冲
 *
 * 与 kernel_implementation.md 强制要求对照:
 *   1. Layout 必须是成员变量 (layout_)
 *   2. stride 在 Init 中一次性算完 (InitLayout)
 *   3. 禁止在 CopyIn/CopyOut 中重复算偏移 (UpdateLayoutByIdx)
 *   4. sizeof(T) 直接内联在 NDDMA/DataCopyPad 参数中
 *   5. TilingData 通过 const 指针访问 (td_)
 *   6. CopyOut 使用 DataCopyPad 3 参数形式
 */

#ifndef TILE_WITH_AXIS_H
#define TILE_WITH_AXIS_H

#include "kernel_operator.h"
#include "tile_with_axis_tiling_data.h"

using namespace AscendC;

// CeilDiv: ceil(a / b) for positive integers
template <typename IntT>
__aicore__ inline IntT CeilDiv(IntT a, IntT b) { return (a + b - 1) / b; }

// NDDMA 维度: 3 维 (ubAxis=0 用满 3 维; ubAxis=1 退化为 2 维; ubAxis=2 不使用)
static constexpr uint8_t NDDMA_DIM = 3;
static constexpr int AXIS_COUNT = 3;

struct AxisLayout {
    int64_t inStart;      // 输入起始坐标
    int64_t outStart;     // 输出起始坐标
    int64_t length;       // 搬运长度
    int64_t inStride;     // 输入 stride (InitLayout 一次性计算)
    int64_t outStride;    // 输出 stride (InitLayout 一次性计算)
};

struct TileLayout {
    AxisLayout axes[AXIS_COUNT];
    int64_t inOffset;     // = SUM(inStart[axis] * inStride[axis])
    int64_t outOffset;    // = SUM(outStart[axis] * outStride[axis])
};

template <typename T, int UB_AXIS>
class TileWithAxisKernel {
public:
    // ================================================================
    // Init: 一次性完成 stride 计算、buffer 申请、NDDMA 参数预计算
    // ================================================================
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 const TileWithAxisTilingData* tilingData)
    {
        td_ = tilingData;

        xGm_.SetGlobalBuffer((__gm__ T*)x);
        yGm_.SetGlobalBuffer((__gm__ T*)y);

        pipe_.InitBuffer(ubBuffer_, td_->bufferSize * 2);

        InitLayout();
        InitNddmaParams();

        uint64_t blockIdx   = GetBlockIdx();
        startBlock_ = blockIdx * td_->perCoreCount;
        endBlock_   = min(startBlock_ + td_->perCoreCount, td_->totalCount);
    }

    // ================================================================
    // Process: Ping-Pong 主循环 + 完整同步链
    // ================================================================
    __aicore__ inline void Process()
    {
        for (uint64_t idx = startBlock_; idx < endBlock_; idx++) {
            uint64_t bufOffset = (idx % 2) * (td_->bufferSize / sizeof(T));
            auto ubBuf = ubBuffer_.Get<T>();

            UpdateLayoutByIdx(idx);
            CopyIn(ubBuf, bufOffset);
            InsertSync<HardEvent::MTE2_MTE3>();
            CopyOut(ubBuf, bufOffset);
            InsertSync<HardEvent::MTE3_MTE2>();
        }
    }

private:
    // ================================================================
    // 成员变量
    // ================================================================
    const TileWithAxisTilingData* td_;
    TileLayout layout_;
    TBuf<TPosition::VECCALC> ubBuffer_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    TPipe pipe_;

    uint64_t startBlock_;
    uint64_t endBlock_;

    // 预计算常量
    int64_t rowInLength_;                 // = axisDim * innerDim = rowLength
    int64_t rowOutLength_;                // = tiles * rowLength

    // NDDMA 预计算参数 (仅 ubAxis=0/1 使用)
    AscendC::NdDmaLoopInfo<NDDMA_DIM> nddmaLoopInfo_;
    int64_t nddmaBlockLen_;               // NDDMA 每次连续搬运的元素数
    int64_t nddmaRepeat_;                 // NDDMA dim1 重复次数

    // ================================================================
    // InitLayout: 基于 3D 模型 inShape=[outerDim,1,rowLength] outShape=[outerDim,tiles,rowLength]
    // ================================================================
    __aicore__ inline void InitLayout()
    {
        // 输入 stride: inShape = [outerDim, 1, rowLength]
        layout_.axes[2].inStride  = 1;                                         // rowLength 维 stride=1
        layout_.axes[1].inStride  = td_->inShape[2];                           // 1 维 stride = rowLength
        layout_.axes[0].inStride  = td_->inShape[1] * td_->inShape[2];         // outer 维 stride = 1 * rowLength

        // 输出 stride: outShape = [outerDim, tiles, rowLength]
        layout_.axes[2].outStride = 1;                                         // rowLength 维 stride=1
        layout_.axes[1].outStride = td_->outShape[2];                          // tiles 维 stride = rowLength
        layout_.axes[0].outStride = td_->outShape[1] * td_->outShape[2];       // outer 维 stride = tiles * rowLength

        rowInLength_  = td_->rowLength;
        rowOutLength_ = td_->tiles * td_->rowLength;
    }

    // ================================================================
    // InitNddmaParams: 根据 ubAxis 预计算 NDDMA 参数
    //   ubAxis=0: 3 维 (dim0=连续行, dim1=tiles 重复, dim2=多 outer 行)
    //   ubAxis=1: 2 维退化 (dim0=连续行, dim1=ubFactor 重复, dim2=1)
    //   ubAxis=2: 不使用 NDDMA
    // ================================================================
    __aicore__ inline void InitNddmaParams()
    {
        if constexpr (UB_AXIS > 1) {
            return;  // UB_AXIS=2 不使用 NDDMA
        }
        // ---- 公共部分 ----
        nddmaBlockLen_ = rowInLength_;
        nddmaRepeat_   = td_->ubFactor;  // UB_AXIS=1 默认值
        nddmaLoopInfo_.loopSrcStride[0] = 1;
        nddmaLoopInfo_.loopSrcStride[1] = 0;
        nddmaLoopInfo_.loopSrcStride[2] = 0;
        nddmaLoopInfo_.loopDstStride[0] = 1;
        nddmaLoopInfo_.loopDstStride[1] = nddmaBlockLen_;
        nddmaLoopInfo_.loopDstStride[2] = 0;
        nddmaLoopInfo_.loopSize[0] = nddmaBlockLen_;
        nddmaLoopInfo_.loopSize[1] = nddmaRepeat_;
        nddmaLoopInfo_.loopSize[2] = 1;
        nddmaLoopInfo_.loopLpSize[0] = 0; nddmaLoopInfo_.loopLpSize[1] = 0; nddmaLoopInfo_.loopLpSize[2] = 0;
        nddmaLoopInfo_.loopRpSize[0] = 0; nddmaLoopInfo_.loopRpSize[1] = 0; nddmaLoopInfo_.loopRpSize[2] = 0;
        // ---- UB_AXIS=0 覆盖 dim2 参数（注意 loopSize[1] 需同步 nddmaRepeat_）----
        if constexpr (UB_AXIS == 0) {
            nddmaRepeat_   = td_->tiles;
            nddmaLoopInfo_.loopSrcStride[2] = nddmaBlockLen_;
            nddmaLoopInfo_.loopDstStride[2] = nddmaBlockLen_ * nddmaRepeat_;
            nddmaLoopInfo_.loopSize[1] = nddmaRepeat_;  // 同步为 tiles
            nddmaLoopInfo_.loopSize[2] = td_->ubFactor;
        }
    }

    // ================================================================
    // UpdateLayoutByIdx: 派发到对应 axis 的子函数
    // ================================================================
    __aicore__ inline void UpdateLayoutByIdx(uint64_t blockIdx)
    {
        if constexpr (UB_AXIS == 0) {
            UpdateLayoutAxis0(blockIdx);
        } else if constexpr (UB_AXIS == 1) {
            UpdateLayoutAxis1(blockIdx);
        } else {
            UpdateLayoutAxis2(blockIdx);
        }
    }

    // ---- 切 outerDim (outShape=[outerDim,tiles,rowLength]) ----
    __aicore__ inline void UpdateLayoutAxis0(uint64_t blockIdx)
    {
        int64_t outerStart = static_cast<int64_t>(td_->ubFactor) * static_cast<int64_t>(blockIdx);
        int64_t outerCount = min(
            static_cast<int64_t>(td_->ubFactor),
            td_->outShape[0] - outerStart);

        layout_.axes[0].inStart  = outerStart;
        layout_.axes[0].outStart = outerStart;
        layout_.axes[0].length   = outerCount;
        layout_.axes[1].length   = td_->outShape[1];  // tiles
        layout_.axes[2].length   = td_->outShape[2];  // rowLength

        layout_.inOffset   = outerStart * layout_.axes[0].inStride;
        layout_.outOffset  = outerStart * layout_.axes[0].outStride;

        ubLen_      = outerCount * rowOutLength_;
        outGmBase_  = layout_.outOffset;
        inGmBase_   = layout_.inOffset;
    }

    // ---- 切 tiles (outShape=[outerDim,tiles,rowLength]) ----
    __aicore__ inline void UpdateLayoutAxis1(uint64_t blockIdx)
    {
        int64_t tileBlocksPerOuter = CeilDiv(
            td_->outShape[1], static_cast<int64_t>(td_->ubFactor));
        int64_t outerIdx  = static_cast<int64_t>(blockIdx) / tileBlocksPerOuter;
        int64_t tileStart = static_cast<int64_t>(td_->ubFactor)
                          * (static_cast<int64_t>(blockIdx) % tileBlocksPerOuter);
        int64_t tileCount = min(
            static_cast<int64_t>(td_->ubFactor),
            td_->outShape[1] - tileStart);

        layout_.axes[0].inStart  = outerIdx;
        layout_.axes[0].outStart = outerIdx;
        layout_.axes[0].length   = 1;
        layout_.axes[1].outStart = tileStart;
        layout_.axes[1].length   = tileCount;
        layout_.axes[2].length   = td_->outShape[2];  // rowLength

        layout_.outOffset = outerIdx  * layout_.axes[0].outStride
                          + tileStart * layout_.axes[1].outStride;

        // 输入: 中间维恒为 1, tile 维 = 0
        layout_.inOffset = outerIdx * layout_.axes[0].inStride;

        ubLen_      = tileCount * td_->outShape[2];
        outGmBase_  = layout_.outOffset;
        inGmBase_   = layout_.inOffset;
    }

    // ---- 切 rowLength (outShape=[outerDim,tiles,rowLength])
    //      每 block = 1 outer 行 x 1 tile x ubFactor 个连续元素 ----
    __aicore__ inline void UpdateLayoutAxis2(uint64_t blockIdx)
    {
        int64_t rowBlocksPerTile = CeilDiv(
            td_->outShape[2], static_cast<int64_t>(td_->ubFactor));
        int64_t blocksPerOuter  = td_->outShape[1] * rowBlocksPerTile;
        int64_t outerIdx    = static_cast<int64_t>(blockIdx) / blocksPerOuter;
        int64_t remainder   = static_cast<int64_t>(blockIdx) % blocksPerOuter;
        int64_t tileIdx     = remainder / rowBlocksPerTile;
        int64_t innerBlock  = remainder % rowBlocksPerTile;
        int64_t innerStart  = static_cast<int64_t>(td_->ubFactor) * innerBlock;
        int64_t innerCount  = min(
            static_cast<int64_t>(td_->ubFactor),
            td_->outShape[2] - innerStart);

        // 输入: 中间维=1 (tile=0)
        layout_.axes[0].inStart  = outerIdx;
        layout_.axes[0].outStart = outerIdx;
        layout_.axes[0].length   = 1;
        layout_.axes[1].outStart = tileIdx;
        layout_.axes[1].inStart  = 0;            // 输入中间维=1
        layout_.axes[1].length   = 1;            // 仅 1 个 tile
        layout_.axes[2].inStart  = innerStart;
        layout_.axes[2].outStart = innerStart;
        layout_.axes[2].length   = innerCount;

        layout_.inOffset  = outerIdx   * layout_.axes[0].inStride
                          + innerStart * layout_.axes[2].inStride;
        layout_.outOffset = outerIdx   * layout_.axes[0].outStride
                          + tileIdx    * layout_.axes[1].outStride
                          + innerStart * layout_.axes[2].outStride;

        ubLen_      = innerCount;
        outGmBase_  = layout_.outOffset;
        inGmBase_   = layout_.inOffset;
    }

    // ================================================================
    // CopyIn: 搬运数据 GM->UB
    //   ubAxis=0: 3 维 NDDMA -- 单次调用搬 ubFactor 行 x tiles 份
    //   ubAxis=1: 2 维 NDDMA -- 单次调用重复 ubFactor 次
    //   ubAxis=2: DataCopyPad -- blockCount=1 连续搬运
    // ================================================================
    __aicore__ inline void CopyIn(const LocalTensor<T>& ubLocal, uint64_t bufOffset)
    {
        if constexpr (UB_AXIS == 0 || UB_AXIS == 1) {
            AscendC::NdDmaLoopInfo<NDDMA_DIM> loopInfo = nddmaLoopInfo_;
            if constexpr (UB_AXIS == 0) {
                loopInfo.loopSize[2] = layout_.axes[0].length;
            } else {
                loopInfo.loopSize[1] = layout_.axes[1].length;
            }
            NdDmaDci();
            AscendC::NdDmaParams<T, NDDMA_DIM> params = {loopInfo, 0};
            AscendC::DataCopy<T, NDDMA_DIM>(
                ubLocal[bufOffset],
                xGm_[inGmBase_],
                params);
        } else {
            DataCopyExtParams copyIn;
            copyIn.blockCount = 1;
            copyIn.blockLen   = static_cast<uint32_t>(ubLen_ * sizeof(T));
            copyIn.srcStride  = 0;
            copyIn.dstStride  = 0;
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(ubLocal[bufOffset],
                       xGm_[inGmBase_],
                       copyIn, padParams);
        }
    }

    // ================================================================
    // CopyOut: DataCopyPad UB->GM (简单连续写)
    // 输出 block 在 output GM 上连续，blockCount=1 单指令完成
    // ================================================================
    __aicore__ inline void CopyOut(const LocalTensor<T>& ubLocal, uint64_t bufOffset)
    {
        DataCopyExtParams copyOut{
            1, static_cast<uint32_t>(ubLen_ * sizeof(T)), 0, 0, 0
        };
        DataCopyPad(yGm_[outGmBase_],
                    ubLocal[bufOffset],
                    copyOut);
    }

    // ================================================================
    // InsertSync: SetFlag/WaitFlag 同步 (模板参数消除 switch-case)
    // 同步链: COPY_IN(MTE2) -> MTE2_MTE3 -> COPY_OUT(MTE3) -> MTE3_MTE2
    // ================================================================
    template <HardEvent EVENT>
    __aicore__ inline void InsertSync()
    {
        event_t eventID = static_cast<event_t>(pipe_.FetchEventID(EVENT));
        SetFlag<EVENT>(eventID);
        WaitFlag<EVENT>(eventID);
    }

    // ================================================================
    // 每 Block 运行时变量
    // ================================================================
    int64_t ubLen_;       // 本 Block 输出元素数
    int64_t outGmBase_;   // 输出 GM 基址
    int64_t inGmBase_;    // 输入 GM 基址
};

#endif // TILE_WITH_AXIS_H
