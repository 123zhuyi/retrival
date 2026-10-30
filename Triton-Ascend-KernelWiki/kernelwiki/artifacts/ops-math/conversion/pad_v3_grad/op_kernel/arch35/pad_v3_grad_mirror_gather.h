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
 * @file pad_v3_grad_mirror_gather.h
 * @brief PadV3Grad Gather 模式 kernel 实现（支持 Reflect/Symmetric 模式）
 *
 * 进入条件：
 * 1. 输入尾轴小：outShape[-1] * sizeof(dtype) <= 128B
 * 2. 维度小于等于五维
 * 3. UB 切分轴为 -2 到 -4（即 UB_AXES = 2/3/4）
 *
 * 核心思想：
 * - 使用 Gather 按索引读取数据，累加后写回
 * - 将 gradOutput 的值"折叠"回 gradInput，处理多对一映射
 *
 * 数据流：
 * GM(input) -> CopyIn+Cast -> tempMidBuf(float32) -> Gather折叠 -> outputBuf -> Cast+CopyOut -> GM(output)
 *
 * 主要处理阶段：
 * 1. CopyIn: 搬入中间区域数据到 tempMidBuf
 * 2. FoldOuterAxis: 外轴折叠（3^n 遍历所有组合）
 * 3. ProcessUbAxisPad: 切分轴 pad 处理
 * 4. FoldInnerAxis: 内轴折叠（W/HW/CHW 轴折叠）
 * 5. CopyOut: 结果搬出到 GM
 */

#ifndef PAD_V3_GRAD_MIRROR_GATHER_H_
#define PAD_V3_GRAD_MIRROR_GATHER_H_

#include "kernel_operator.h"
#include "pad_v3_grad_struct.h"
#include "pad_v3_grad_common.h"

namespace PadV3Grad {
using namespace AscendC;

/**
 * @brief PadV3Grad Gather类
 *
 * 用于处理  Reflect/Symmetric 模式，当输入尾轴较小时
 * （outShape[-1] * sizeof(dtype) <= 128B），使用 VF Gather 方式进行高效处理。
 *
 * 核心思想：
 * - 将 gradOutput 的值"折叠"回 gradInput，多个位置可能映射到同一输出位置
 * - 使用 VF Gather 按索引读取数据，累加后写回
 *
 * 数据流：
 * GM(input) -> CopyIn+Cast -> tempMidBuf(float32) -> Gather折叠 -> outputBuf -> Cast+CopyOut -> GM(output)
 *
 * 支持的 UB 切分模式：
 * - UB_AXES = 2: 切分在 H 轴，内部折叠 W 轴
 * - UB_AXES = 3: 切分在 C 轴，内部折叠 H+W 轴
 * - UB_AXES = 4: 切分在 N 轴，内部折叠 C+H+W 轴
 *
 * @tparam T        数据类型（float16, bfloat16, float32）
 * @tparam modeName 模式名称（2=Reflect, 3=Symmetric）
 */
template <typename T, uint8_t modeName>
class PadV3GradGather {
private:
    // ========== 类型定义 ==========
    using GatherIdxType = uint32_t;  ///< Gather 索引类型（需与 PromoteDataT 匹配）
    using GatherRangeType = int32_t; ///< 索引计算类型（支持负数）

    // ========== 编译期常量 ==========
    constexpr static uint64_t VL_COMPUTE_CNT = VL_SIZE / sizeof(PromoteDataT); ///< 单次计算元素数
    constexpr static uint64_t VL_RANGE_CNT = VL_SIZE / sizeof(GatherIdxType);  ///< 单次索引元素数

    constexpr static int64_t PAD_GRAD_MAX_DIMS_NUM = 8; ///< 最大支持维度数
    constexpr static uint64_t MAX_DIM = 5;              ///< 实际使用的最大维度
    constexpr static int64_t ZERO_PAD_CNT = 8;          ///< 零填充元素数（用于无效索引指向）

    constexpr static bool IS_REFLECT = modeName == 2; ///< 是否为 Reflect 模式

    /**
     * @brief 外轴映射信息结构体
     *
     * 存储当前位置在某个外轴上可能映射到的三个输入位置索引
     */
    struct AxisMappingInfo {
        int64_t leftMapIdx;  ///< 左 pad 映射索引（-1 表示无映射）
        int64_t midIdx;      ///< 中间区域索引
        int64_t rightMapIdx; ///< 右 pad 映射索引（-1 表示无映射）
    };

private:
    TPipe pipe_;
    const PadV3GradACTilingData* tdPtr_ = nullptr;

    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;

    // ========== UB 缓冲区 ==========
    TQue<QuePosition::VECIN, 1> inputQue_;
    TQue<QuePosition::VECOUT, 1> outputQue_;
    TBuf<> tempMidBuf_;
    TBuf<> tempPadBuf_;

    TBuf<> leftPadIdx_;
    TBuf<> midIdx_;
    TBuf<> rightPadIdx_;

    TBuf<> leftPadOffset_;
    TBuf<> midOffset_;
    TBuf<> rightPadOffset_;

    // ========== 运行时参数 ==========
    int64_t blockIdx_{0}; ///< 当前核索引
    int64_t dimNum_{0};   ///< 维度数
    int64_t ubAxis_{0};   ///< UB 切分轴
    int64_t outTileSize_{0};
    int64_t additionTileSize_{0};
    int64_t innerSize_{0}; ///< 切分轴内层大小（inStride[ubAxis]）
    int64_t outerSize_{0}; ///< 切分轴外层大小
    int64_t outputSize_{0};
    int64_t posIdx[PAD_GRAD_MAX_DIMS_NUM] = {0}; ///< 当前位置索引

    // ========== Tiling 参数缓存 ==========
    int64_t ubFactor_{0};
    int64_t ubPerCount_{0};
    int64_t ubTotalCount_{0};
    int64_t inShape_[PAD_GRAD_MAX_DIMS_NUM] = {0};
    int64_t outShape_[PAD_GRAD_MAX_DIMS_NUM] = {0};
    int64_t inStride_[PAD_GRAD_MAX_DIMS_NUM] = {0};
    int64_t outStride_[PAD_GRAD_MAX_DIMS_NUM] = {0};
    int64_t leftPad_[PAD_GRAD_MAX_DIMS_NUM] = {0};
    int64_t rightPad_[PAD_GRAD_MAX_DIMS_NUM] = {0};

    uint64_t UB_AXES = 0;

    /**
     * @brief 初始化 Tiling 数组参数
     */
    __aicore__ inline void InitTilingArrays(const PadV3GradACTilingData* tilingData)
    {
        for (int64_t i = 0; i < PAD_GRAD_MAX_DIMS_NUM; i++) {
            inShape_[i] = static_cast<int64_t>(tilingData->inShape[i]);
            outShape_[i] = static_cast<int64_t>(tilingData->outShape[i]);
            inStride_[i] = static_cast<int64_t>(tilingData->inStride[i]);
            outStride_[i] = static_cast<int64_t>(tilingData->outStride[i]);
            leftPad_[i] = tilingData->leftPad[i];
            rightPad_[i] = tilingData->rightPad[i];
        }
    }

public:
    __aicore__ inline PadV3GradGather() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ComputePositionIndices(int64_t* posIdx, int64_t curIdx, int64_t ubFactor);
    __aicore__ inline void ProcessOneFactor(int64_t* posIdx, int64_t factor);

    // 通用 Cast 操作（T -> PromoteDataT）
    __aicore__ inline void CastToPromote(__local_mem__ T* srcAddr, __local_mem__ PromoteDataT* dstAddr, int64_t count,
                                         bool addMode);

    // 数据搬入与 cast
    __aicore__ inline void CopyInAndCast(int64_t gmOffset, int64_t copySize, __local_mem__ PromoteDataT* dstAddr,
                                         bool addMode);

    // 外轴折叠（通用）
    __aicore__ inline void ComputeAxisMappings(int64_t* posIdx, AxisMappingInfo* mappings);
    __aicore__ inline void FoldOuterAxis(int64_t* posIdx, int64_t ubAxisPos, int64_t count,
                                         __local_mem__ PromoteDataT* dstAddr);
    __aicore__ inline int64_t ComputeGmOffsetForAxis(int64_t* posIdx, int64_t axis, int64_t axisInIdx);

    // 切分轴 pad 处理
    __aicore__ inline void ProcessUbAxisPad(int64_t* posIdx, int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                            LocalTensor<PromoteDataT>& padLocal);

    // 通用 Gather+Add+Store/Scatter 操作
    __aicore__ inline void GatherAddStore(__local_mem__ PromoteDataT* srcAddr, __local_mem__ PromoteDataT* dstAddr,
                                          __local_mem__ GatherRangeType* idxAddr, uint64_t dstOffset,
                                          uint64_t curElements);
    __aicore__ inline void GatherAddScatter(__local_mem__ PromoteDataT* dataAddr,
                                            __local_mem__ GatherRangeType* srcIdxAddr,
                                            __local_mem__ GatherRangeType* dstIdxAddr, uint64_t curElements);

    // 切分轴内折叠
    __aicore__ inline void FoldInnerAxis(int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                         LocalTensor<PromoteDataT>& outputLocal);
    __aicore__ inline void FoldWAxis(int64_t factor, int64_t width, LocalTensor<PromoteDataT>& midLocal,
                                     LocalTensor<PromoteDataT>& outputLocal);
    __aicore__ inline void GenerateFoldWAxisIndices(uint16_t rowsPerBatch, uint16_t innerWidth, int64_t width,
                                                    int64_t leftPadW, int64_t rightPadW, GatherRangeType rowWidthOffset,
                                                    __local_mem__ GatherRangeType* midIdxAddr,
                                                    __local_mem__ GatherRangeType* leftIdxAddr,
                                                    __local_mem__ GatherRangeType* rightIdxAddr,
                                                    __local_mem__ GatherRangeType* midOffsetAddr,
                                                    __local_mem__ GatherRangeType* leftOffsetAddr,
                                                    __local_mem__ GatherRangeType* rightOffsetAddr);
    __aicore__ inline void FoldHWAxis(int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                      LocalTensor<PromoteDataT>& outputLocal);
    __aicore__ inline void FoldCHWAxis(int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                       LocalTensor<PromoteDataT>& outputLocal);

    // ========== FoldCHWAxis 辅助函数 ==========
    __aicore__ inline void FoldWAxisForCHWAxis(int64_t factor, int64_t outC, uint64_t hwSize, int64_t width,
                                               int64_t leftPadC, int64_t leftPadH, int64_t leftPadW, int64_t rightPadW,
                                               int64_t outH, int64_t outW, __local_mem__ PromoteDataT* midAddr,
                                               __local_mem__ PromoteDataT* outputAddr,
                                               __local_mem__ GatherRangeType* midIdxAddr,
                                               __local_mem__ GatherRangeType* leftIdxAddr,
                                               __local_mem__ GatherRangeType* rightIdxAddr);

    // ========== FoldHWAxis 辅助函数 ==========
    /**
     * @brief 生成 W 轴折叠的索引数组（mid/left/right）和偏移数组
     *
     * 索引映射公式说明（以 outW=3, leftPadW=1, rightPadW=1 为例）：
     *
     * 输入数据布局（width=5）: [L0, M0, M1, M2, R0]  (L=left pad, M=mid, R=right pad)
     * 输出数据布局（outW=3）:  [O0, O1, O2]
     *
     * 对于 reflect 模式：
     *   - mid:   O[col] <- M[col]，即 idx = rowOffset + leftPadW + col
     *   - left:  O[0] <- L[0] 映射到 M[1]，即 idx = rowOffset + leftPadW - col = rowOffset + 1 - 0 = rowOffset + 1
     *   - right: O[2] <- R[0] 映射到 M[1]，即 idx = rowOffset + leftPadW + outW - 2 - rightCol
     *            rightCol=0 时，idx = rowOffset + 1 + 3 - 2 - 0 = rowOffset + 2
     *
     * 对于 symmetric 模式：
     *   - mid:   同上
     *   - left:  O[0] <- L[0] 映射到 M[0]，即 idx = rowOffset + leftPadW - 1 - col
     *   - right: O[2] <- R[0] 映射到 M[2]，即 idx = rowOffset + leftPadW + outW - 1 - rightCol
     *
     * @param cOffset      当前 C 通道的偏移量 (c * hwSize)
     * @param rowsPerBatch 每批处理的行数
     * @param outW         输出宽度
     * @param width        输入宽度
     * @param leftPadW     左 pad 宽度
     * @param rightPadW    右 pad 宽度
     * @param leftPadH     上 pad 高度（用于计算行偏移）
     * @param batchOffset  每批行的偏移量 (rowsPerBatch * width)
     * @param midIdxAddr   mid 索引数组地址
     * @param leftIdxAddr  left 索引数组地址
     * @param rightIdxAddr right 索引数组地址
     * @param midOffsetAddr   mid 偏移数组地址
     * @param leftOffsetAddr  left 偏移数组地址
     * @param rightOffsetAddr right 偏移数组地址
     */
    __aicore__ inline void GenerateWAxisIndices(uint64_t cOffset, uint16_t rowsPerBatch, uint16_t outW, uint16_t width,
                                                int64_t leftPadW, int64_t rightPadW, int64_t leftPadH,
                                                GatherRangeType batchOffset, __local_mem__ GatherRangeType* midIdxAddr,
                                                __local_mem__ GatherRangeType* leftIdxAddr,
                                                __local_mem__ GatherRangeType* rightIdxAddr,
                                                __local_mem__ GatherRangeType* midOffsetAddr,
                                                __local_mem__ GatherRangeType* leftOffsetAddr,
                                                __local_mem__ GatherRangeType* rightOffsetAddr);

    // 跨 C 通道生成 W 轴索引
    __aicore__ inline void GenerateCrossChannelWAxisIndices(uint32_t totalRows, uint16_t outW, uint16_t outH,
                                                            int64_t width, uint64_t hwSize, int64_t leftPadW,
                                                            int64_t rightPadW, int64_t leftPadH,
                                                            __local_mem__ GatherRangeType* midIdxAddr,
                                                            __local_mem__ GatherRangeType* leftIdxAddr,
                                                            __local_mem__ GatherRangeType* rightIdxAddr);

    // 由 padRow 计算 H 轴 pad 折叠的 srcRowOffset / dstRowOffset
    __aicore__ inline void CalcHAxisPadRowOffset(int64_t padRow, int64_t padCount, bool isLeftPad, int64_t leftPadH,
                                                 int64_t outH, int64_t width, int64_t& srcRowOffset,
                                                 int64_t& dstRowOffset)
    {
        int64_t srcRow;
        int64_t dstRow;
        if (isLeftPad) {
            srcRow = padCount - 1 - padRow;
            dstRow = IS_REFLECT ? (2 * leftPadH - srcRow) : (2 * leftPadH - 1 - srcRow);
        } else {
            srcRow = leftPadH + outH + padRow;
            dstRow = IS_REFLECT ? (leftPadH + outH - 2 - padRow) : (leftPadH + outH - 1 - padRow);
        }
        srcRowOffset = srcRow * width;
        dstRowOffset = dstRow * width;
    }

    /**
     * @brief 折叠 H 轴的 pad 数据（GatherAddScatter 方式）
     *
     * H 轴折叠映射公式说明（以 height=5, leftPadH=1, rightPadH=1, outH=3 为例）：
     *
     * 输入数据布局: [Row0(L), Row1(M0), Row2(M1), Row3(M2), Row4(R)]
     *
     * 对于 reflect 模式：
     *   - left pad:  srcRow=0 映射到 dstRow=2 (2*leftPadH - srcRow = 2*1 - 0 = 2)
     *   - right pad: srcRow=4 映射到 dstRow=2 (leftPadH + outH - 2 - row = 1 + 3 - 2 - 0 = 2)
     *
     * 对于 symmetric 模式：
     *   - left pad:  srcRow=0 映射到 dstRow=1 (2*leftPadH - 1 - srcRow = 2*1 - 1 - 0 = 1)
     *   - right pad: srcRow=4 映射到 dstRow=3 (leftPadH + outH - 1 - row = 1 + 3 - 1 - 0 = 3)
     *
     * @param factor     C 通道数量
     * @param hwSize     单个 C 通道的 H*W 大小
     * @param width      输入宽度
     * @param padCount   pad 行数（leftPadH 或 rightPadH）
     * @param isLeftPad  是否为 left pad（true=left, false=right）
     * @param leftPadH   上 pad 高度
     * @param outH       输出高度
     * @param midAddr    数据地址
     * @param srcIdxAddr 源索引数组地址
     * @param dstIdxAddr 目标索引数组地址
     */
    __aicore__ inline void FoldHAxisPad(int64_t factor, uint64_t hwSize, int64_t width, int64_t padCount,
                                        bool isLeftPad, int64_t leftPadH, int64_t outH, int64_t nFactor,
                                        uint64_t nStride, __local_mem__ PromoteDataT* baseAddr,
                                        __local_mem__ GatherRangeType* srcIdxAddr,
                                        __local_mem__ GatherRangeType* dstIdxAddr);

    /**
     * @brief 折叠 C 轴的 pad 数据（GatherAddScatter 方式）
     *
     * C 轴折叠映射公式说明（以 cHeight=5, leftPadC=1, rightPadC=1, outC=3 为例）：
     *
     * 输入数据布局: [C0(L), C1(M0), C2(M1), C3(M2), C4(R)]，每个 C 包含 H*W 个元素
     *
     * 对于 reflect 模式：
     *   - left pad:  srcC=0 映射到 dstC=2 (2*leftPadC - srcC)
     *   - right pad: srcC=4 映射到 dstC=2 (leftPadC + outC - 2 - cIdx)
     *
     * 对于 symmetric 模式：
     *   - left pad:  srcC=0 映射到 dstC=1 (2*leftPadC - 1 - srcC)
     *   - right pad: srcC=4 映射到 dstC=3 (leftPadC + outC - 1 - cIdx)
     *
     * @param hwSize      单个 C 通道的 H*W 大小
     * @param padCount    pad 通道数（leftPadC 或 rightPadC）
     * @param isLeftPad   是否为 left pad（true=left, false=right）
     * @param leftPadC    左 pad 通道数
     * @param outC        输出通道数
     * @param midAddr     数据地址
     * @param srcIdxAddr  源索引数组地址
     * @param dstIdxAddr  目标索引数组地址
     */
    __aicore__ inline void FoldCAxisPad(uint64_t hwSize, int64_t padCount, bool isLeftPad, int64_t leftPadC,
                                        int64_t outC, int64_t factor, uint64_t nStride,
                                        __local_mem__ PromoteDataT* baseAddr, __local_mem__ GatherRangeType* srcIdxAddr,
                                        __local_mem__ GatherRangeType* dstIdxAddr);

    // 输出
    __aicore__ inline void CopyOutToGm(int64_t* posIdx, int64_t factor);
};

// ========== Init & Process ==========

/**
 * @brief 初始化
 *
 * 1. 从 tilingData 中读取切分参数（维度、切分轴、factor 等）
 * 2. 初始化输入/输出 GlobalTensor
 * 3. 分配 UB 缓冲区（输入队列、输出队列、中间缓冲、索引缓冲）
 *
 * UB 空间布局：
 * - inputQue_:      双 buffer，存放从 GM 搬入的原始数据
 * - outputQue_:     双 buffer，存放待搬出的结果数据
 * - tempMidBuf_:    中间累加缓冲，前 ZERO_PAD_CNT 元素为零填充
 * - tempPadBuf_:    pad 数据临时缓冲
 * - leftPadIdx_/midIdx_/rightPadIdx_:    索引缓冲区
 * - leftPadOffset_/midOffset_/rightPadOffset_: 偏移缓冲区
 *
 * @param x          输入数据的 GM 地址（gradOutput）
 * @param y          输出数据的 GM 地址（gradInput）
 * @param tilingData Tiling 参数结构体指针
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::Init(GM_ADDR x, GM_ADDR y, const PadV3GradACTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    tdPtr_ = tilingData;

    // 初始化基本参数
    dimNum_ = tdPtr_->dimNum;
    ubAxis_ = tdPtr_->ubAxis;
    ubFactor_ = tdPtr_->ubFactor;
    ubPerCount_ = tdPtr_->ubPerCount;
    ubTotalCount_ = tdPtr_->ubTotalCount;
    outTileSize_ = tdPtr_->outTileSize;
    additionTileSize_ = tdPtr_->additionTileSize;

    // 初始化数组参数
    InitTilingArrays(tilingData);

    // 计算派生参数
    innerSize_ = inStride_[ubAxis_];
    outerSize_ = outShape_[0];
    for (int64_t i = 1; i <= ubAxis_; i++) {
        outerSize_ *= outShape_[i];
    }
    outputSize_ = outStride_[ubAxis_];
    UB_AXES = dimNum_ - ubAxis_;

    inputGm_.SetGlobalBuffer((__gm__ T*)x);
    outputGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_.InitBuffer(inputQue_, BUFFER_NUM, outTileSize_ * sizeof(T));
    pipe_.InitBuffer(outputQue_, BUFFER_NUM, outTileSize_ * sizeof(PromoteDataT));
    pipe_.InitBuffer(tempMidBuf_, (outTileSize_ + ZERO_PAD_CNT) * sizeof(PromoteDataT));
    pipe_.InitBuffer(tempPadBuf_, outTileSize_ * sizeof(PromoteDataT));

    // 存储索引及偏移
    pipe_.InitBuffer(leftPadIdx_, VL_SIZE);
    pipe_.InitBuffer(midIdx_, VL_SIZE);
    pipe_.InitBuffer(rightPadIdx_, VL_SIZE);

    pipe_.InitBuffer(leftPadOffset_, VL_SIZE);
    pipe_.InitBuffer(midOffset_, VL_SIZE);
    pipe_.InitBuffer(rightPadOffset_, VL_SIZE);
}

/**
 * @brief 处理入口
 *
 * 处理流程：
 * 1. 根据 blockIdx 计算当前核负责的任务范围 [startIdx, endIdx)
 * 2. 遍历每个任务单元，计算其在各维度上的位置索引
 * 3. 计算实际处理的 factor 数量（处理尾块时可能小于 ubFactor）
 * 4. 调用 ProcessOneFactor 处理单个任务单元
 *
 * 多核并行策略：
 * - 总任务数 = ubTotalCount_
 * - 每核处理 ubPerCount_ 个任务
 * - 第 blockIdx 核处理 [blockIdx * ubPerCount, (blockIdx+1) * ubPerCount) 范围
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::Process()
{
    int64_t startIdx = blockIdx_ * ubPerCount_;
    if (startIdx >= ubTotalCount_) {
        return;
    }

    int64_t endIdx = (blockIdx_ + 1) * ubPerCount_;
    endIdx = (endIdx < ubTotalCount_ ? endIdx : ubTotalCount_);

    // 按 outShape 计算索引，与 tiling 逻辑保持一致
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t curIdx = idx;

        // 计算位置索引
        ComputePositionIndices(posIdx, curIdx, ubFactor_);

        // 计算当前块实际处理的 factor 数量
        int64_t curFactor = posIdx[ubAxis_] + ubFactor_ < outShape_[ubAxis_] ? ubFactor_ :
                                                                               outShape_[ubAxis_] - posIdx[ubAxis_];

        if (curFactor > 0) {
            // 大于0，开始处理
            ProcessOneFactor(posIdx, curFactor);
        }
    }
}

/**
 * @brief 计算任务索引在各维度上的位置
 *
 * 将一维任务索引 curIdx 转换为多维位置索引 posIdx[]。
 * 计算方式类似于将线性索引转换为多维坐标，但切分轴的处理有所不同：
 * - 普通轴：posIdx[i] = curIdx % outShape[i]
 * - 切分轴：posIdx[ubAxis] = (curIdx % factor) * ubFactor
 *   其中 factor = CeilDiv(outShape[ubAxis], ubFactor)
 *
 * @param posIdx   输出参数，存储计算得到的各维度位置索引
 * @param curIdx   当前任务的一维索引
 * @param ubFactor 切分轴上每次处理的份数
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::ComputePositionIndices(int64_t* posIdx, int64_t curIdx,
                                                                            int64_t ubFactor)
{
    // 从 ubAxis_ 到 0 计算各维度索引
    for (int64_t i = ubAxis_; i >= 0; i--) {
        int64_t factor = outShape_[i];
        if (i == ubAxis_) {
            // 切分轴：用 CeilDiv(outShape[ubAxis], ubFactor) 作为该维度的 factor
            factor = (outShape_[i] + ubFactor - 1) / ubFactor;
        }
        posIdx[i] = (i == ubAxis_ ? curIdx % factor * ubFactor : curIdx % factor);
        curIdx = curIdx / factor;
    }
}

/**
 * @brief 处理单个任务单元（一次 UB 切分）
 *
 * 核心处理函数，完成一个切分块的完整处理流程：
 *
 * Step 1: 搬入中间区域数据
 *   - 计算 GM 偏移（加上各轴的 leftPad）
 *   - 将数据搬入 tempMidBuf 并 Cast 到 float32
 *
 * Step 2: 外轴折叠（FoldOuterAxis）
 *   - 对 ubAxis 之前的轴，遍历所有 3^n 种组合
 *   - 将映射到当前块的 pad 数据累加到 tempMidBuf
 *
 * Step 3: 切分轴 pad 处理（ProcessUbAxisPad）
 *   - 处理切分轴上的 leftPad 和 rightPad
 *   - 使用 Gather+Add+Store 方式累加
 *
 * Step 4: 内轴折叠（FoldInnerAxis）
 *   - 根据 UB_AXES 调用对应的折叠函数（FoldWAxis/FoldHWAxis/FoldCHWAxis）
 *   - 将结果输出到 outputLocal
 *
 * Step 5: 搬出到 GM（CopyOutToGm）
 *   - 原地 Cast 回原类型
 *   - 非对齐搬出到 GM
 *
 * @param posIdx 当前任务在各维度上的位置索引
 * @param factor 当前任务实际处理的切分轴份数
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::ProcessOneFactor(int64_t* posIdx, int64_t factor)
{
    LocalTensor<PromoteDataT> midLocal = tempMidBuf_.Get<PromoteDataT>();
    LocalTensor<PromoteDataT> padLocal = tempPadBuf_.Get<PromoteDataT>();

    // 初始化 tempMidBuf 前 ZERO_PAD_CNT 个元素为 0，为折叠W轴时做准备
    AscendC::Duplicate<PromoteDataT>(midLocal, 0, ZERO_PAD_CNT);

    __local_mem__ PromoteDataT* midAddr = (__local_mem__ PromoteDataT*)midLocal.GetPhyAddr();

    // Step 1: 搬入中间区域数据
    int64_t gmOffset = 0;
    for (int64_t i = 0; i <= ubAxis_; i++) {
        int64_t inIdx = posIdx[i] + leftPad_[i];
        gmOffset += inIdx * inStride_[i];
    }

    int64_t copySize = factor * innerSize_;
    CopyInAndCast(gmOffset, copySize, midAddr + ZERO_PAD_CNT, false);

    // Step 2: 外轴折叠
    FoldOuterAxis(posIdx, posIdx[ubAxis_] + leftPad_[ubAxis_], factor, midAddr + ZERO_PAD_CNT);

    // Step 3: 处理切分轴上的 pad
    ProcessUbAxisPad(posIdx, factor, midLocal, padLocal);

    // Step 4: 切分轴内折叠 + 输出
    // outputLocal 使用 PromoteDataT 类型，搬出前再进行原地 cast
    LocalTensor<PromoteDataT> outputLocal = outputQue_.AllocTensor<PromoteDataT>();
    FoldInnerAxis(factor, midLocal, outputLocal);

    outputQue_.EnQue<PromoteDataT>(outputLocal);
    CopyOutToGm(posIdx, factor);
}

/**
 * @brief 数据类型转换（T -> PromoteDataT）并可选累加
 *
 * 将源数据从原始类型 T 转换为计算类型 PromoteDataT（float32），
 * 支持直接赋值或累加两种模式。使用 VF 方式处理，
 * 每次处理 VL_COMPUTE_CNT 个元素。
 *
 * @param srcAddr 源数据地址（类型 T）
 * @param dstAddr 目标数据地址（类型 PromoteDataT）
 * @param count   处理的元素数量
 * @param addMode false: dstAddr = Cast(srcAddr)
 *                true:  dstAddr = dstAddr + Cast(srcAddr)
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::CastToPromote(__local_mem__ T* srcAddr,
                                                                   __local_mem__ PromoteDataT* dstAddr, int64_t count,
                                                                   bool addMode)
{
    uint32_t vlCount = VL_COMPUTE_CNT;

    __VEC_SCOPE__
    {
        uint32_t sregCount = count;
        uint16_t loops = (count + vlCount - 1) / vlCount;
        for (uint16_t i = 0; i < loops; i++) {
            uint32_t offset = i * vlCount;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(sregCount);

            AscendC::Reg::RegTensor<PromoteDataT> newReg;
            if constexpr (IsSameType<PromoteDataT, T>::value) {
                AscendC::Reg::DataCopy(newReg, srcAddr + offset);
            } else {
                AscendC::Reg::RegTensor<T> srcReg;
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg, srcAddr + offset);
                AscendC::Reg::Cast<PromoteDataT, T, CAST_TRAIT_0>(newReg, srcReg, mask);
            }

            if (addMode) {
                AscendC::Reg::RegTensor<PromoteDataT> existReg;
                AscendC::Reg::DataCopy(existReg, dstAddr + offset);
                AscendC::Reg::RegTensor<PromoteDataT> resultReg;

                AscendC::Reg::Add(resultReg, existReg, newReg, mask);

                AscendC::Reg::DataCopy(dstAddr + offset, resultReg, mask);
            } else {
                AscendC::Reg::DataCopy(dstAddr + offset, newReg, mask);
            }
        }
    }
}

/**
 * @brief 从 GM 搬入数据并转换类型
 *
 * 完成 GM -> UB 的数据搬运，并将数据从原始类型 T 转换为计算类型 PromoteDataT。
 * 使用 DataCopyPad 实现非对齐搬入。
 *
 * 数据流: GM[gmOffset] -> inputQue_ -> CastToPromote -> dstAddr
 *
 * @param gmOffset GM 起始偏移（元素单位）
 * @param copySize 搬入的元素数量
 * @param dstAddr  目标 UB 地址
 * @param addMode  false: 直接赋值  true: 累加到已有数据
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::CopyInAndCast(int64_t gmOffset, int64_t copySize,
                                                                   __local_mem__ PromoteDataT* dstAddr, bool addMode)
{
    LocalTensor<T> inputLocal = inputQue_.AllocTensor<T>();
    uint32_t dataSize = static_cast<uint32_t>(copySize) * sizeof(T);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParams = {1u, dataSize, 0, 0, 0};
    DataCopyPad(inputLocal, inputGm_[gmOffset], copyParams, padParams);

    inputQue_.EnQue<T>(inputLocal);
    inputLocal = inputQue_.DeQue<T>();

    __local_mem__ T* srcAddr = (__local_mem__ T*)inputLocal.GetPhyAddr();

    // VF中cast并copy/cast到midLocal，addMode判断是否是加到midLocal
    CastToPromote(srcAddr, dstAddr, copySize, addMode);

    inputQue_.FreeTensor(inputLocal);
}

/**
 * @brief 计算外轴各维度的映射关系
 *
 * 对于每个外轴（ubAxis 之前的轴），计算当前位置 posIdx[axis] 在输入 shape 中
 * 可能映射到的三个位置：中间区域、左 pad 区域、右 pad 区域。
 *
 * 映射公式（Reflect 模式）：
 *   - mid:   curPos + leftPad
 *   - left:  leftPad - curPos  (条件: 1 <= curPos <= leftPad)
 *   - right: leftPad + outShape + i  (条件: 0 <= i < rightPad)
 *
 * 映射公式（Symmetric 模式）：
 *   - mid:   curPos + leftPad
 *   - left:  leftPad - 1 - curPos  (条件: curPos < leftPad)
 *   - right: leftPad + outShape + i  (条件: 0 <= i < rightPad)
 *
 * @param posIdx   当前位置在 outShape 上的索引
 * @param mappings 输出参数，存储各轴的映射信息（-1 表示无映射）
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::ComputeAxisMappings(int64_t* posIdx, AxisMappingInfo* mappings)
{
    /*
        计算当前处理块在外轴inputShape中的映射块的位置
        posIdx：outShape上的索引位置
        -1表示无映射关系
    */
    for (int64_t axis = 0; axis < ubAxis_; axis++) {
        int64_t curPos = posIdx[axis];
        int64_t leftPadLen = leftPad_[axis];
        int64_t rightPadLen = rightPad_[axis];
        int64_t outAxisSize = outShape_[axis];

        // 1.mid数据
        mappings[axis].midIdx = curPos + leftPadLen;

        // 2.left pad数据
        mappings[axis].leftMapIdx = -1;
        if constexpr (modeName == 2) {
            // Reflect
            if (curPos >= 1 && curPos <= leftPadLen) {
                mappings[axis].leftMapIdx = leftPadLen - curPos;
            }
        } else {
            // Symmetric
            if (curPos < leftPadLen) {
                mappings[axis].leftMapIdx = leftPadLen - 1 - curPos;
            }
        }

        // 3.right pad数据
        mappings[axis].rightMapIdx = -1;
        if constexpr (modeName == 2) {
            // Reflect
            int64_t i = outAxisSize - 2 - curPos;
            if (i >= 0 && i < rightPadLen) {
                mappings[axis].rightMapIdx = leftPadLen + outAxisSize + i;
            }
        } else {
            // Symmetric
            int64_t i = outAxisSize - 1 - curPos;
            if (i >= 0 && i < rightPadLen) {
                mappings[axis].rightMapIdx = leftPadLen + outAxisSize + i;
            }
        }
    }
}

/**
 * @brief 外轴折叠处理
 *
 * 对于 ubAxis 之前的所有轴（外轴），遍历所有可能的映射组合（3^n 种），
 * 将映射到当前数据块的 pad 数据累加到 dstAddr。
 *
 * 算法思路：
 * 1. 每个外轴有 3 种状态：mid(0)、left(1)、right(2)
 * 2. 遍历所有 3^ubAxis 种组合（combo = 0 到 3^ubAxis - 1）
 * 3. 解码 combo 得到各轴状态：states[axis] = combo / 3^axis % 3
 * 4. 跳过全 mid 组合（已在 Step1 处理）和无效组合（某轴无映射）
 * 5. 计算对应的 GM 偏移并搬入累加
 *
 * @param posIdx    当前位置在 outShape 上的索引
 * @param ubAxisPos 切分轴在 inShape 上的位置（含 leftPad 偏移）
 * @param count     当前处理的元素数量
 * @param dstAddr   目标 UB 地址（累加目标）
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldOuterAxis(int64_t* posIdx, int64_t ubAxisPos, int64_t count,
                                                                   __local_mem__ PromoteDataT* dstAddr)
{
    if (ubAxis_ == 0) {
        // 切分到第0轴，没有外轴，直接返回
        return;
    }

    // 单次处理总量
    uint64_t dataSize = count * innerSize_;

    // 外轴需要折叠的区域
    AxisMappingInfo mappings[PAD_GRAD_MAX_DIMS_NUM];
    ComputeAxisMappings(posIdx, mappings);

    int64_t totalCombinations = 1; // 循环总数, 3^n，n为外轴数量，范围为0-27
    for (int64_t i = 0; i < ubAxis_; i++) {
        totalCombinations *= 3;
    }

    for (int64_t combo = 0; combo < totalCombinations; combo++) {
        int64_t states[PAD_GRAD_MAX_DIMS_NUM];
        int64_t temp = combo;

        bool allMid = true; // 当前块所在的位置，不处理
        bool validCombo = true;

        for (int64_t axis = ubAxis_ - 1; axis >= 0; axis--) {
            /*
                解码：0：mid，1：left，2：right
            */
            states[axis] = temp % 3;
            temp /= 3;
            if (states[axis] != 0) {
                allMid = false;
            }
        }

        if (allMid) {
            continue;
        }

        int64_t gmOffset = 0;
        for (int64_t axis = 0; axis < ubAxis_; axis++) {
            int64_t axisIdx;
            if (states[axis] == 0) {
                axisIdx = mappings[axis].midIdx;
            } else if (states[axis] == 1) {
                if (mappings[axis].leftMapIdx < 0) {
                    validCombo = false;
                    break;
                }
                axisIdx = mappings[axis].leftMapIdx;
            } else {
                if (mappings[axis].rightMapIdx < 0) {
                    validCombo = false;
                    break;
                }
                axisIdx = mappings[axis].rightMapIdx;
            }
            gmOffset += axisIdx * inStride_[axis];
        }

        if (!validCombo) {
            continue;
        }

        // 计算需要折叠的gm地址
        gmOffset += ubAxisPos * innerSize_;

        // 循环累加到dstAddr
        CopyInAndCast(gmOffset, dataSize, dstAddr, true);
    }
}

/**
 * @brief 计算指定轴使用自定义索引时的 GM 偏移
 *
 * 与普通偏移计算不同，指定轴使用 axisInIdx 而非 posIdx[axis] + leftPad[axis]。
 * 用于处理切分轴 pad 时，需要指定特定轴的输入索引。
 *
 * @param posIdx    当前位置在 outShape 上的索引
 * @param axis      使用自定义索引的轴
 * @param axisInIdx 该轴在 inShape 上的索引
 * @return          计算得到的 GM 偏移（元素单位）
 */
template <typename T, uint8_t modeName>
__aicore__ inline int64_t PadV3GradGather<T, modeName>::ComputeGmOffsetForAxis(int64_t* posIdx, int64_t axis,
                                                                               int64_t axisInIdx)
{
    int64_t gmOffset = 0;
    for (int64_t i = 0; i <= ubAxis_; i++) {
        if (i == axis) {
            gmOffset += axisInIdx * inStride_[i];
        } else {
            gmOffset += (posIdx[i] + leftPad_[i]) * inStride_[i];
        }
    }
    return gmOffset;
}

/**
 * @brief Gather + Add + Store 操作
 *
 * 按索引从源地址 gather 数据，加到目标地址后存回。
 * 支持非对齐的目标地址访问。
 *
 * 操作公式: dstAddr[dstOffset + i] += srcAddr[idxAddr[i]]  (i = 0..curElements-1)
 *
 * @param srcAddr     源数据地址（gather 的数据源）
 * @param dstAddr     目标数据地址（累加结果存放位置）
 * @param idxAddr     索引数组地址（gather 的索引）
 * @param dstOffset   目标地址偏移（元素单位）
 * @param curElements 处理的元素数量
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::GatherAddStore(__local_mem__ PromoteDataT* srcAddr,
                                                                    __local_mem__ PromoteDataT* dstAddr,
                                                                    __local_mem__ GatherRangeType* idxAddr,
                                                                    uint64_t dstOffset, uint64_t curElements)
{
    uint64_t vlCount = VL_COMPUTE_CNT;
    __local_mem__ GatherIdxType* idxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(idxAddr);

    __VEC_SCOPE__
    {
        uint32_t sregCurElements = curElements;
        uint16_t loops = (curElements + vlCount - 1) / vlCount;
        uint32_t lastElemets = curElements - (loops - 1) * vlCount;

        dstAddr = dstAddr + dstOffset;

        __local_mem__ PromoteDataT* copyOutDstAddr = dstAddr;

        AscendC::Reg::UnalignRegForLoad uSrc;
        AscendC::Reg::UnalignRegForStore uDst;

        AscendC::Reg::LoadUnAlignPre(uSrc, dstAddr);

        for (uint16_t i = 0; i < static_cast<uint16_t>(loops - 1); i++) {
            uint32_t offset = i * vlCount;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherIdxType>(sregCurElements);
            AscendC::Reg::RegTensor<GatherIdxType> idxReg;
            AscendC::Reg::DataCopy(idxReg, idxAddrU + offset);

            AscendC::Reg::RegTensor<PromoteDataT> gatherReg;
            AscendC::Reg::DataCopyGather(gatherReg, srcAddr, idxReg, mask);

            AscendC::Reg::RegTensor<PromoteDataT> dstReg;
            AscendC::Reg::LoadUnAlign(dstReg, uSrc, dstAddr, vlCount);

            AscendC::Reg::RegTensor<PromoteDataT> resultReg;
            AscendC::Reg::Add(resultReg, dstReg, gatherReg, mask);

            AscendC::Reg::StoreUnAlign(copyOutDstAddr, resultReg, uDst, vlCount);
        }

        // 最后一块非对齐
        uint32_t offset = (loops - 1) * vlCount;
        uint32_t tempLastElements = lastElemets;

        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherIdxType>(lastElemets);

        AscendC::Reg::RegTensor<GatherIdxType> idxReg;
        AscendC::Reg::DataCopy(idxReg, idxAddrU + offset);

        AscendC::Reg::RegTensor<PromoteDataT> gatherReg;
        AscendC::Reg::DataCopyGather(gatherReg, srcAddr, idxReg, mask);

        AscendC::Reg::RegTensor<PromoteDataT> dstReg;
        AscendC::Reg::LoadUnAlign(dstReg, uSrc, dstAddr, tempLastElements);

        AscendC::Reg::RegTensor<PromoteDataT> resultReg;
        AscendC::Reg::Add(resultReg, dstReg, gatherReg, mask);

        AscendC::Reg::StoreUnAlign(copyOutDstAddr, resultReg, uDst, tempLastElements);

        AscendC::Reg::StoreUnAlignPost(copyOutDstAddr, uDst, 0);
    }
}

/**
 * @brief Gather + Add + Scatter 操作
 *
 * 按源索引 gather 数据，加上目标索引位置的数据，结果 scatter 回目标位置。
 * 用于同一 buffer 内不同位置数据的累加（如 H 轴折叠）。
 *
 * 操作公式: dataAddr[dstIdxAddr[i]] += dataAddr[srcIdxAddr[i]]  (i = 0..curElements-1)
 *
 * @param dataAddr    数据地址（同时作为源和目标）
 * @param srcIdxAddr  源索引数组地址
 * @param dstIdxAddr  目标索引数组地址
 * @param curElements 处理的元素数量
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::GatherAddScatter(__local_mem__ PromoteDataT* dataAddr,
                                                                      __local_mem__ GatherRangeType* srcIdxAddr,
                                                                      __local_mem__ GatherRangeType* dstIdxAddr,
                                                                      uint64_t curElements)
{
    uint64_t vlCount = VL_COMPUTE_CNT;
    __local_mem__ GatherIdxType* srcIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(srcIdxAddr);
    __local_mem__ GatherIdxType* dstIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(dstIdxAddr);

    __VEC_SCOPE__
    {
        uint16_t loops = (curElements + vlCount - 1) / vlCount;
        for (uint16_t i = 0; i < loops; i++) {
            uint32_t offset = i * vlCount;
            uint64_t remaining = curElements - offset;
            uint32_t curCount = static_cast<uint32_t>(Std::min(vlCount, remaining));

            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(curCount);

            AscendC::Reg::RegTensor<GatherIdxType> srcIdxReg;
            AscendC::Reg::DataCopy(srcIdxReg, srcIdxAddrU + offset);

            AscendC::Reg::RegTensor<PromoteDataT> srcDataReg;
            AscendC::Reg::DataCopyGather(srcDataReg, dataAddr, srcIdxReg, mask);

            AscendC::Reg::RegTensor<GatherIdxType> dstIdxReg;
            AscendC::Reg::DataCopy(dstIdxReg, dstIdxAddrU + offset);

            AscendC::Reg::RegTensor<PromoteDataT> dstDataReg;
            AscendC::Reg::DataCopyGather(dstDataReg, dataAddr, dstIdxReg, mask);

            AscendC::Reg::RegTensor<PromoteDataT> resultReg;
            AscendC::Reg::Add(resultReg, srcDataReg, dstDataReg, mask);

            AscendC::Reg::DataCopyScatter(dataAddr, resultReg, dstIdxReg, mask);
        }
    }
}

/**
 * @brief 处理切分轴上的 pad 数据
 *
 * 处理当前任务块在切分轴（ubAxis）上与 leftPad/rightPad 区域的重叠部分。
 * 将 pad 区域的数据折叠（累加）到对应的中间区域。
 *
 * 处理流程：
 * 1. 计算 leftPad 重叠区域：
 *    - 确定重叠范围 [leftOverlapStart, leftOverlapStart + leftPadCount - 1]
 *    - 计算映射的 GM 偏移（反向映射）
 *    - 搬入数据到 padLocal，进行外轴折叠
 *    - 使用 GatherAddStore 翻转累加到 midLocal
 *
 * 2. 计算 rightPad 重叠区域：
 *    - 类似 leftPad，但映射公式不同
 *
 * @param posIdx   当前位置在 outShape 上的索引
 * @param factor   当前处理的切分轴份数
 * @param midLocal 中间缓冲区（已包含中间区域数据）
 * @param padLocal pad 数据临时缓冲区
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::ProcessUbAxisPad(int64_t* posIdx, int64_t factor,
                                                                      LocalTensor<PromoteDataT>& midLocal,
                                                                      LocalTensor<PromoteDataT>& padLocal)
{
    int64_t leftPad = leftPad_[ubAxis_];
    int64_t rightPad = rightPad_[ubAxis_];
    int64_t innerLen = outShape_[ubAxis_];
    int64_t curPos = posIdx[ubAxis_];

    __local_mem__ PromoteDataT* midAddr = (__local_mem__ PromoteDataT*)midLocal.GetPhyAddr() + ZERO_PAD_CNT;
    __local_mem__ PromoteDataT* padAddr = (__local_mem__ PromoteDataT*)padLocal.GetPhyAddr();

    // 计算 W 轴宽度和 H 轴高度（用于大 innerSize_ 情况）
    int64_t width = inShape_[dimNum_ - 1];

    // innerSize_ = H * W 或 = W（此时 heightPerInner = 1），一定能整除，计算出多少行
    int64_t heightPerInner = innerSize_ / width;

    // 每次能处理多少个 W 行（基于索引缓冲区容量）
    int64_t rowsPerBatch = VL_RANGE_CNT / width;
    if (rowsPerBatch < 1)
        rowsPerBatch = 1; // 不会出现

    // 1. 处理 left pad
    int64_t leftPadCount = 0;
    int64_t leftOverlapStart = 0; // 需要处理的输出位置起始
    if constexpr (modeName == 2) {
        // Reflect 模式：输出位置 [1, leftPad] 需要接收 left pad 折叠
        // 检查当前块 [curPos, curPos + factor - 1] 是否与 [1, leftPad] 有交集
        int64_t endPos = curPos + factor - 1;
        if (endPos >= 1 && curPos <= leftPad) {
            leftOverlapStart = Std::max(curPos, static_cast<int64_t>(1));
            int64_t leftOverlapEnd = Std::min(endPos, leftPad);
            leftPadCount = leftOverlapEnd - leftOverlapStart + 1;
        }
    } else {
        // Symmetric
        if (curPos < leftPad) {
            leftPadCount = Std::min(factor, leftPad - curPos);
            leftOverlapStart = curPos;
        }
    }

    if (leftPadCount > 0) {
        // 输出位置 [leftOverlapStart, leftOverlapStart + leftPadCount - 1] 映射到输入索引
        // reflect: 输出 j -> 输入 (leftPad - j)
        //          输出 leftOverlapStart + leftPadCount - 1 -> 输入 leftPad - (leftOverlapStart + leftPadCount - 1)
        //          所以 mapStartPos = leftPad - leftOverlapStart - leftPadCount + 1
        // symmetric: 输出 j -> 输入 (leftPad - 1 - j)
        //          所以 mapStartPos = leftPad - 1 - (leftOverlapStart + leftPadCount - 1) = leftPad - leftOverlapStart
        //          - leftPadCount
        int64_t mapStartPos;
        if constexpr (modeName == 2) {
            // Reflect
            mapStartPos = leftPad - leftOverlapStart - leftPadCount + 1;
        } else {
            // Symmetric
            mapStartPos = leftPad - leftOverlapStart - leftPadCount;
        }
        int64_t gmOffset = ComputeGmOffsetForAxis(posIdx, ubAxis_, mapStartPos);
        int64_t copySize = leftPadCount * innerSize_;

        // 搬入第一份数据到 padLocal
        CopyInAndCast(gmOffset, copySize, padAddr, false);

        // 外轴折叠累加到 padLocal
        FoldOuterAxis(posIdx, mapStartPos, leftPadCount, padAddr);

        // 翻转 gather 并累加到 midLocal
        LocalTensor<GatherRangeType> idxLocal = leftPadIdx_.Get<GatherRangeType>();
        __local_mem__ GatherRangeType* idxAddr = (__local_mem__ GatherRangeType*)idxLocal.GetPhyAddr();

        // 对每个切分轴上的 pad 行进行翻转
        int64_t offsetLen = rowsPerBatch * width;
        for (int64_t padRow = 0; padRow < leftPadCount; padRow++) {
            int64_t srcPadRow = leftPadCount - 1 - padRow; // 翻转后的源行

            // 在 H 轴方向循环（当 innerSize_ > VL_RANGE_CNT 时需要分批）
            // 计算索引
            __VEC_SCOPE__
            {
                uint32_t idxLen = static_cast<uint32_t>(rowsPerBatch * width);
                int32_t idxStart = static_cast<int32_t>(srcPadRow * innerSize_);
                AscendC::Reg::RegTensor<GatherRangeType> leftIdxReg;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(idxLen);

                AscendC::Reg::Arange(leftIdxReg, idxStart);
                AscendC::Reg::DataCopy(idxAddr, leftIdxReg, mask);
            }

            for (int64_t hStart = 0; hStart < heightPerInner; hStart += rowsPerBatch) {
                int64_t curHRows = Std::min(rowsPerBatch, heightPerInner - hStart);
                uint64_t curElements = static_cast<uint64_t>(curHRows * width);

                // 偏移索引
                if (hStart > 0) {
                    // 第一次不需要偏移
                    AscendC::Adds(idxLocal, idxLocal, static_cast<PromoteDataT>(offsetLen),
                                  static_cast<uint32_t>(offsetLen));
                }

                // Gather + Add + Store
                uint64_t dstOffset = (leftOverlapStart - curPos + padRow) * innerSize_ + hStart * width;

                GatherAddStore(padAddr, midAddr, idxAddr, dstOffset, curElements);
            }
        }
    }

    // 2. 处理 right pad
    int64_t rightPadCount = 0;
    int64_t rightStartInFactor = 0;
    if constexpr (modeName == 2) {
        // reflect 模式：输出位置 [outShape - rightPad, outShape - 2] 受 right pad 影响
        int64_t rightValidStart = innerLen - rightPad - 1;
        int64_t rightValidEnd = innerLen - 2;

        if (rightValidStart < 0)
            rightValidStart = 0;
        if (rightValidEnd < rightValidStart)
            rightValidEnd = rightValidStart - 1; // 无效范围
        int64_t factorEnd = curPos + factor - 1;

        if (factorEnd >= rightValidStart && curPos <= rightValidEnd) {
            int64_t overlapStart = Std::max(curPos, rightValidStart);
            int64_t overlapEnd = Std::min(factorEnd, rightValidEnd);
            rightPadCount = overlapEnd - overlapStart + 1;
            rightStartInFactor = overlapStart - curPos;
        }
    } else {
        // symmetric 模式：输出位置 [outShape - rightPad, outShape - 1] 受 right pad 影响
        int64_t rightValidStart = innerLen - rightPad;
        int64_t rightValidEnd = innerLen - 1;

        if (rightValidStart < 0)
            rightValidStart = 0;
        int64_t factorEnd = curPos + factor - 1;

        if (factorEnd >= rightValidStart && curPos <= rightValidEnd) {
            int64_t overlapStart = Std::max(curPos, rightValidStart);
            int64_t overlapEnd = Std::min(factorEnd, rightValidEnd);
            rightPadCount = overlapEnd - overlapStart + 1;
            rightStartInFactor = overlapStart - curPos;
        }
    }

    if (rightPadCount > 0) {
        int64_t actualPos = curPos + rightStartInFactor;
        // 输出位置 [actualPos, actualPos + rightPadCount - 1] 映射到输入索引
        // reflect: 输出 j -> 输入 (leftPad + 2*outShape - 2 - j)
        //          输出 actualPos -> 输入 (leftPad + 2*innerLen - 2 - actualPos)，这是最大输入索引
        //          需要搬入 rightPadCount 块，从最大输入索引开始递减
        // symmetric: 输出 j -> 输入 (leftPad + 2*outShape - 1 - j)
        int64_t mapStartPos;
        if constexpr (modeName == 2) {
            // Reflect
            // 最大输入索引是 actualPos 对应的，从这里开始搬入 rightPadCount 块
            mapStartPos = leftPad + 2 * innerLen - 2 - actualPos - rightPadCount + 1;
        } else {
            // Symmetric
            mapStartPos = leftPad + 2 * innerLen - 1 - actualPos - rightPadCount + 1;
        }

        int64_t gmOffset = ComputeGmOffsetForAxis(posIdx, ubAxis_, mapStartPos);
        int64_t copySize = rightPadCount * innerSize_;

        // 搬入第一份数据到 padLocal
        CopyInAndCast(gmOffset, copySize, padAddr, false);

        // 外轴折叠累加到 padLocal
        FoldOuterAxis(posIdx, mapStartPos, rightPadCount, padAddr);

        // 翻转 gather 并累加到 midLocal
        LocalTensor<GatherRangeType> idxLocal = rightPadIdx_.Get<GatherRangeType>();
        __local_mem__ GatherRangeType* idxAddr = (__local_mem__ GatherRangeType*)idxLocal.GetPhyAddr();

        int64_t offsetLen = rowsPerBatch * width;
        for (int64_t padRow = 0; padRow < rightPadCount; padRow++) {
            int64_t srcPadRow = rightPadCount - 1 - padRow;

            // 计算索引
            __VEC_SCOPE__
            {
                uint32_t idxLen = static_cast<uint32_t>(rowsPerBatch * width);
                int32_t idxStart = static_cast<int32_t>(srcPadRow * innerSize_);
                AscendC::Reg::RegTensor<GatherRangeType> rightIdxReg;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(idxLen);

                AscendC::Reg::Arange(rightIdxReg, idxStart);
                AscendC::Reg::DataCopy(idxAddr, rightIdxReg, mask);
            }

            for (int64_t hStart = 0; hStart < heightPerInner; hStart += rowsPerBatch) {
                int64_t curHRows = Std::min(rowsPerBatch, heightPerInner - hStart);
                uint64_t curElements = static_cast<uint64_t>(curHRows * width);

                // 偏移索引
                if (hStart > 0) {
                    // 第一次不需要偏移
                    AscendC::Adds(idxLocal, idxLocal, static_cast<PromoteDataT>(offsetLen),
                                  static_cast<uint32_t>(offsetLen));
                }

                uint64_t dstOffset = (rightStartInFactor + padRow) * innerSize_ + hStart * width;
                GatherAddStore(padAddr, midAddr, idxAddr, dstOffset, curElements);
            }
        }
    }

    // 等待索引使用结束
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
}

/**
 * @brief 切分轴内折叠
 *
 * 根据 UB 内包含的轴数（UB_AXES）调用对应的折叠函数：
 *   - UB_AXES = 2: 折叠 W 轴（调用 FoldWAxis）
 *   - UB_AXES = 3: 折叠 H+W 轴（调用 FoldHWAxis）
 *   - UB_AXES = 4: 折叠 C+H+W 轴（调用 FoldCHWAxis）
 *
 * 折叠完成后，结果数据存放在 outputLocal 中，等待搬出。
 *
 * @param factor      当前处理的切分轴份数
 * @param midLocal    中间缓冲区（包含待折叠的数据）
 * @param outputLocal 输出缓冲区（存放折叠结果）
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldInnerAxis(int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                                                   LocalTensor<PromoteDataT>& outputLocal)
{
    if (UB_AXES == CONST2) {
        // 切分在 H 轴，FoldWAxis 直接输出到 outputLocal
        int64_t width = inShape_[dimNum_ - 1];
        FoldWAxis(factor, width, midLocal, outputLocal);
    } else if (UB_AXES == CONST3) {
        // 切分在 C 轴，先折叠 H（写回 midLocal），再折叠 W 并输出
        FoldHWAxis(factor, midLocal, outputLocal);
    } else if (UB_AXES == CONST4) {
        // 切分在N轴，先折叠C、H轴（写回 midLocal），再折叠W轴并输出
        FoldCHWAxis(factor, midLocal, outputLocal);
    }
}

/**
 * @brief 折叠 W 轴（最内层轴）
 *
 * 将输入数据的 W 轴（宽度）方向上的 left pad、mid、right pad 三个区域
 * 合并输出到 outputLocal。这是 UB_AXES=2 时的折叠函数。
 *
 * 算法思路：
 * 1. 计算索引缓冲区（256B）最多能容纳的行数 rowsPerBatch
 * 2. 生成三组索引数组（mid/left/right）和对应的偏移数组
 *    - midIdx: 指向中间区域数据
 *    - leftIdx: 指向 left pad 映射位置（无映射的位置指向 ZERO_PAD）
 *    - rightIdx: 指向 right pad 映射位置
 * 3. 循环处理所有行：
 *    - 按索引 gather 三个区域的数据
 *    - 三个数据相加得到输出
 *    - 存储到 outputLocal
 *    - 更新索引（累加偏移）
 *
 * @param factor      行数（H 维度大小）
 * @param width       输入宽度（含 pad）
 * @param midLocal    中间缓冲区
 * @param outputLocal 输出缓冲区
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldWAxis(int64_t factor, int64_t width,
                                                               LocalTensor<PromoteDataT>& midLocal,
                                                               LocalTensor<PromoteDataT>& outputLocal)
{
    if (factor <= 0) {
        return;
    }

    int64_t leftPadW = leftPad_[dimNum_ - 1];
    int64_t rightPadW = rightPad_[dimNum_ - 1];
    uint32_t innerWidth = static_cast<uint32_t>(outShape_[dimNum_ - 1]);

    __local_mem__ PromoteDataT* midAddr = (__local_mem__ PromoteDataT*)midLocal.GetPhyAddr();
    __local_mem__ PromoteDataT* outputAddr = (__local_mem__ PromoteDataT*)outputLocal.GetPhyAddr();

    // 获取三个索引缓冲区（每个 256B）
    LocalTensor<GatherRangeType> leftIdxLocal = leftPadIdx_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> midIdxLocal = midIdx_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> rightIdxLocal = rightPadIdx_.Get<GatherRangeType>();

    __local_mem__ GatherRangeType* leftIdxAddr = (__local_mem__ GatherRangeType*)leftIdxLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* midIdxAddr = (__local_mem__ GatherRangeType*)midIdxLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* rightIdxAddr = (__local_mem__ GatherRangeType*)rightIdxLocal.GetPhyAddr();

    // 1. 计算一次能处理多少行（基于 innerWidth）
    int64_t maxElements = VL_RANGE_CNT;
    int64_t rowsPerBatch = maxElements / innerWidth;
    if (rowsPerBatch < 1)
        rowsPerBatch = 1;

    GatherRangeType rowWidthOffset = static_cast<GatherRangeType>(rowsPerBatch * width); // 每批行的偏移

    // 存储索引及偏移
    LocalTensor<GatherRangeType> leftOffsetLocal = leftPadOffset_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> midOffsetLocal = midOffset_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> rightOffsetLocal = rightPadOffset_.Get<GatherRangeType>();

    __local_mem__ GatherRangeType* leftOffsetAddr = (__local_mem__ GatherRangeType*)leftOffsetLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* midOffsetAddr = (__local_mem__ GatherRangeType*)midOffsetLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* rightOffsetAddr = (__local_mem__ GatherRangeType*)rightOffsetLocal.GetPhyAddr();

    // 2. 生成索引和偏移数组（只生成一次，基于前 rowsPerBatch 行）
    GenerateFoldWAxisIndices(static_cast<uint16_t>(rowsPerBatch), static_cast<uint16_t>(innerWidth), width, leftPadW,
                             rightPadW, rowWidthOffset, midIdxAddr, leftIdxAddr, rightIdxAddr, midOffsetAddr,
                             leftOffsetAddr, rightOffsetAddr);

    // 3. 循环处理所有行
    __local_mem__ GatherIdxType* midIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midIdxAddr);
    __local_mem__ GatherIdxType* leftIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftIdxAddr);
    __local_mem__ GatherIdxType* rightIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightIdxAddr);

    __local_mem__ GatherIdxType* midOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midOffsetAddr);
    __local_mem__ GatherIdxType* leftOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftOffsetAddr);
    __local_mem__ GatherIdxType* rightOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightOffsetAddr);

    int64_t totalLoops = (factor + rowsPerBatch - 1) / rowsPerBatch;
    uint32_t curElements = static_cast<uint32_t>(rowsPerBatch * innerWidth);

    // 尾块行数和元素数
    int64_t tailRows = factor - (totalLoops - 1) * rowsPerBatch;
    uint32_t curTailElements = static_cast<uint32_t>(tailRows * innerWidth);

    __VEC_SCOPE__
    {
        __local_mem__ PromoteDataT* tempOutputAddr = outputAddr;

        // 加载索引到 RegTensor（只加载一次）
        AscendC::Reg::RegTensor<GatherIdxType> midIdxReg;
        AscendC::Reg::DataCopy(midIdxReg, midIdxAddrU);

        AscendC::Reg::RegTensor<GatherIdxType> leftIdxReg;
        AscendC::Reg::DataCopy(leftIdxReg, leftIdxAddrU);

        AscendC::Reg::RegTensor<GatherIdxType> rightIdxReg;
        AscendC::Reg::DataCopy(rightIdxReg, rightIdxAddrU);

        // 加载偏移到 RegTensor
        AscendC::Reg::RegTensor<GatherIdxType> midOffsetReg;
        AscendC::Reg::DataCopy(midOffsetReg, midOffsetAddrU);

        AscendC::Reg::RegTensor<GatherIdxType> leftOffsetReg;
        AscendC::Reg::DataCopy(leftOffsetReg, leftOffsetAddrU);

        AscendC::Reg::RegTensor<GatherIdxType> rightOffsetReg;
        AscendC::Reg::DataCopy(rightOffsetReg, rightOffsetAddrU);

        AscendC::Reg::UnalignRegForStore ureg1; // 非对齐搬出
        uint32_t elementsPerBatch = static_cast<uint32_t>(rowsPerBatch * innerWidth);
        AscendC::Reg::MaskReg idxMask = AscendC::Reg::UpdateMask<GatherIdxType>(elementsPerBatch);

        // 处理完整批次
        for (uint16_t loop = 0; loop < static_cast<uint16_t>(totalLoops - 1); loop++) {
            uint32_t tempCurElements = curElements;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(tempCurElements);

            // Gather 三个区域的数据
            AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
            AscendC::Reg::DataCopyGather(midDataReg, midAddr, midIdxReg, mask);

            AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
            AscendC::Reg::DataCopyGather(leftDataReg, midAddr, leftIdxReg, mask);

            AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
            AscendC::Reg::DataCopyGather(rightDataReg, midAddr, rightIdxReg, mask);

            // 三个数据相加
            AscendC::Reg::RegTensor<PromoteDataT> sumReg;
            AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
            AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);

            // 存储到 outputAddr
            AscendC::Reg::StoreUnAlign(tempOutputAddr, sumReg, ureg1, curElements);

            // 更新索引（为下一次循环准备）
            AscendC::Reg::Add(midIdxReg, midIdxReg, midOffsetReg, idxMask);
            AscendC::Reg::Add(leftIdxReg, leftIdxReg, leftOffsetReg, idxMask);
            AscendC::Reg::Add(rightIdxReg, rightIdxReg, rightOffsetReg, idxMask);
        }

        // 处理尾块
        uint32_t tempCurElements = curTailElements;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(tempCurElements);

        // Gather 三个区域的数据
        AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
        AscendC::Reg::DataCopyGather(midDataReg, midAddr, midIdxReg, mask);

        AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
        AscendC::Reg::DataCopyGather(leftDataReg, midAddr, leftIdxReg, mask);

        AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
        AscendC::Reg::DataCopyGather(rightDataReg, midAddr, rightIdxReg, mask);

        // 三个数据相加
        AscendC::Reg::RegTensor<PromoteDataT> sumReg;
        AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
        AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);

        // 存储到 outputAddr
        AscendC::Reg::StoreUnAlign(tempOutputAddr, sumReg, ureg1, curTailElements);
        AscendC::Reg::StoreUnAlignPost(tempOutputAddr, ureg1, 0);
    }
}

// ========== FoldWAxis 辅助函数实现 ==========

/**
 * @brief 生成 FoldWAxis 所需的 mid/left/right 索引数组和偏移数组
 *
 * 基于前 rowsPerBatch 行生成一次索引，后续通过累加偏移复用：
 *   - midIdx: 中间区域索引，公式 rowOffset + leftPadW + col
 *   - leftIdx: 左 pad 映射位置（Reflect: col∈(0,leftPadW]，Symmetric: col∈[0,leftPadW)），其余为 0 指向 ZERO_PAD
 *   - rightIdx: 右 pad 映射位置（Reflect: col∈[innerWidth-rightPadW-1,innerWidth-1)，Symmetric:
 * col∈[innerWidth-rightPadW,innerWidth)），其余为 0
 *   - 偏移数组：有效映射位置为 rowsPerBatch * width，无效位置为 0（保持指向 ZERO_PAD）
 *
 * @param rowsPerBatch   每批处理的行数
 * @param innerWidth     输出宽度 (outShape[W])
 * @param width          输入宽度 (leftPadW + innerWidth + rightPadW)
 * @param leftPadW       左 pad 宽度
 * @param rightPadW      右 pad 宽度
 * @param rowWidthOffset 每批行的偏移量 (rowsPerBatch * width)
 * @param midIdxAddr     mid 索引数组地址
 * @param leftIdxAddr    left 索引数组地址
 * @param rightIdxAddr   right 索引数组地址
 * @param midOffsetAddr  mid 偏移数组地址
 * @param leftOffsetAddr left 偏移数组地址
 * @param rightOffsetAddr right 偏移数组地址
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::GenerateFoldWAxisIndices(
    uint16_t rowsPerBatch, uint16_t innerWidth, int64_t width, int64_t leftPadW, int64_t rightPadW,
    GatherRangeType rowWidthOffset, __local_mem__ GatherRangeType* midIdxAddr,
    __local_mem__ GatherRangeType* leftIdxAddr, __local_mem__ GatherRangeType* rightIdxAddr,
    __local_mem__ GatherRangeType* midOffsetAddr, __local_mem__ GatherRangeType* leftOffsetAddr,
    __local_mem__ GatherRangeType* rightOffsetAddr)
{
    __VEC_SCOPE__
    {
        uint32_t vLen = rowsPerBatch * innerWidth;
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

        // x = [0, 1, ..., vLen-1]，批内线性位置
        AscendC::Reg::RegTensor<GatherRangeType> xReg;
        AscendC::Reg::Arange(xReg, static_cast<GatherRangeType>(0));

        // row = x / innerWidth，col = x % innerWidth（始终用 Div，避免运行时分支）
        AscendC::Reg::RegTensor<GatherRangeType> rowReg, colReg, tmpReg;
        AscendC::Reg::RegTensor<GatherRangeType> divWidthReg;
        AscendC::Reg::Duplicate(divWidthReg, static_cast<GatherRangeType>(innerWidth), mask);
        AscendC::Reg::Div(rowReg, xReg, divWidthReg, mask);
        AscendC::Reg::Muls(tmpReg, rowReg, static_cast<GatherRangeType>(innerWidth), mask);
        AscendC::Reg::Sub(colReg, xReg, tmpReg, mask);

        // chanHBaseReg = row * width + (ZERO_PAD_CNT + leftPadW)
        // 三个索引的公共基：midIdx = chanHBase + col，leftIdx/rightIdx 从此派生，
        // 避免重复计算 row * width。
        AscendC::Reg::RegTensor<GatherRangeType> chanHBaseReg;
        AscendC::Reg::Muls(chanHBaseReg, rowReg, static_cast<GatherRangeType>(width), mask);
        AscendC::Reg::RegTensor<GatherRangeType> constBaseReg;
        AscendC::Reg::Duplicate(constBaseReg, static_cast<GatherRangeType>(ZERO_PAD_CNT + leftPadW), mask);
        AscendC::Reg::Add(chanHBaseReg, chanHBaseReg, constBaseReg, mask);

        // ===== midIdx = chanHBase + col（恒有效）=====
        AscendC::Reg::RegTensor<GatherRangeType> midIdxReg;
        AscendC::Reg::Add(midIdxReg, chanHBaseReg, colReg, mask);
        AscendC::Reg::DataCopy(midIdxAddr, midIdxReg, mask);

        // ===== leftIdx =====
        // Reflect:   leftIdx = chanHBase - col，有效 col ∈ [1, leftPadW]
        // Symmetric: leftIdx = chanHBase - 1 - col，有效 col ∈ [0, leftPadW-1]
        // leftPadW=0 时：两个 Compare 联合清零全部位置，效果等价于 Duplicate(0)，
        //              无需单独处理 leftPadW==0 分支。
        AscendC::Reg::RegTensor<GatherRangeType> leftIdxReg;
        if constexpr (IS_REFLECT) {
            AscendC::Reg::Sub(leftIdxReg, chanHBaseReg, colReg, mask);
            // 清零 col < 1（col == 0）
            AscendC::Reg::RegTensor<GatherRangeType> oneReg;
            AscendC::Reg::Duplicate(oneReg, static_cast<GatherRangeType>(1), mask);
            AscendC::Reg::MaskReg colLt1Mask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colLt1Mask, oneReg, colReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                leftIdxReg, static_cast<GatherRangeType>(0), colLt1Mask);
            // 清零 col > leftPadW（leftPadW=0 时 GT(col,0) 清零所有 col>0，与上行联合清零全部）
            AscendC::Reg::RegTensor<GatherRangeType> leftPWReg;
            AscendC::Reg::Duplicate(leftPWReg, static_cast<GatherRangeType>(leftPadW), mask);
            AscendC::Reg::MaskReg colGtLPWMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colGtLPWMask, colReg, leftPWReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                leftIdxReg, static_cast<GatherRangeType>(0), colGtLPWMask);
        } else {
            // chanHBase - col - 1
            AscendC::Reg::Sub(leftIdxReg, chanHBaseReg, colReg, mask);
            AscendC::Reg::RegTensor<GatherRangeType> oneReg;
            AscendC::Reg::Duplicate(oneReg, static_cast<GatherRangeType>(1), mask);
            AscendC::Reg::Sub(leftIdxReg, leftIdxReg, oneReg, mask);
            // 清零 col > leftPadW-1（leftPadW=0 时值为 -1，GT 对所有 col≥0 为真，全部清零）
            AscendC::Reg::RegTensor<GatherRangeType> lpwM1Reg;
            AscendC::Reg::Duplicate(lpwM1Reg, static_cast<GatherRangeType>(leftPadW - 1), mask);
            AscendC::Reg::MaskReg colGeLPWMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colGeLPWMask, colReg, lpwM1Reg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                leftIdxReg, static_cast<GatherRangeType>(0), colGeLPWMask);
        }
        AscendC::Reg::DataCopy(leftIdxAddr, leftIdxReg, mask);

        // ===== rightIdx =====
        // Reflect:   rightIdx = chanHBase + (2*innerWidth-2) - col，有效 col ∈ [innerWidth-rightPadW-1, innerWidth-2]
        // Symmetric: rightIdx = chanHBase + (2*innerWidth-1) - col，有效 col ∈ [innerWidth-rightPadW, innerWidth-1]
        // rightPadW=0 时：Compare 条件覆盖全部位置，全部清零，无需单独分支。
        AscendC::Reg::RegTensor<GatherRangeType> rightIdxReg;
        if constexpr (IS_REFLECT) {
            AscendC::Reg::RegTensor<GatherRangeType> rightConstReg;
            AscendC::Reg::Duplicate(rightConstReg,
                                    static_cast<GatherRangeType>(2 * static_cast<int64_t>(innerWidth) - 2), mask);
            AscendC::Reg::Add(rightIdxReg, chanHBaseReg, rightConstReg, mask);
            AscendC::Reg::Sub(rightIdxReg, rightIdxReg, colReg, mask);
            // 清零 col < innerWidth - rightPadW - 1
            AscendC::Reg::RegTensor<GatherRangeType> lowBoundReg;
            AscendC::Reg::Duplicate(
                lowBoundReg, static_cast<GatherRangeType>(static_cast<int64_t>(innerWidth) - rightPadW - 1), mask);
            AscendC::Reg::MaskReg colLtLowMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colLtLowMask, lowBoundReg, colReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                rightIdxReg, static_cast<GatherRangeType>(0), colLtLowMask);
            // 清零 col > innerWidth - 2
            AscendC::Reg::RegTensor<GatherRangeType> highM1Reg;
            AscendC::Reg::Duplicate(highM1Reg, static_cast<GatherRangeType>(static_cast<int64_t>(innerWidth) - 2),
                                    mask);
            AscendC::Reg::MaskReg colGtHighMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colGtHighMask, colReg, highM1Reg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                rightIdxReg, static_cast<GatherRangeType>(0), colGtHighMask);
        } else {
            AscendC::Reg::RegTensor<GatherRangeType> rightConstReg;
            AscendC::Reg::Duplicate(rightConstReg,
                                    static_cast<GatherRangeType>(2 * static_cast<int64_t>(innerWidth) - 1), mask);
            AscendC::Reg::Add(rightIdxReg, chanHBaseReg, rightConstReg, mask);
            AscendC::Reg::Sub(rightIdxReg, rightIdxReg, colReg, mask);
            // 清零 col < innerWidth - rightPadW（rightPadW=0 时为 innerWidth，GT 对全部 col 为真，全部清零）
            AscendC::Reg::RegTensor<GatherRangeType> lowBound2Reg;
            AscendC::Reg::Duplicate(lowBound2Reg,
                                    static_cast<GatherRangeType>(static_cast<int64_t>(innerWidth) - rightPadW), mask);
            AscendC::Reg::MaskReg colLtLow2Mask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colLtLow2Mask, lowBound2Reg, colReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                rightIdxReg, static_cast<GatherRangeType>(0), colLtLow2Mask);
        }
        AscendC::Reg::DataCopy(rightIdxAddr, rightIdxReg, mask);

        // ===== 偏移数组 =====
        // midOffset: 所有位置均为 rowWidthOffset（mid 索引始终有效）
        AscendC::Reg::RegTensor<GatherRangeType> midOffsetReg;
        AscendC::Reg::Duplicate(midOffsetReg, static_cast<GatherRangeType>(rowWidthOffset), mask);
        AscendC::Reg::DataCopy(midOffsetAddr, midOffsetReg, mask);

        // leftOffset: 有效位置（leftIdx > 0）为 rowWidthOffset，ZERO_PAD 位置（leftIdx == 0）为 0
        AscendC::Reg::RegTensor<GatherRangeType> zeroReg;
        AscendC::Reg::Duplicate(zeroReg, static_cast<GatherRangeType>(0), mask);
        AscendC::Reg::RegTensor<GatherRangeType> leftOffsetReg;
        AscendC::Reg::Duplicate(leftOffsetReg, static_cast<GatherRangeType>(0), mask);
        AscendC::Reg::MaskReg leftOffMask;
        AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(leftOffMask, leftIdxReg, zeroReg, mask);
        AscendC::Reg::Duplicate(leftOffsetReg, static_cast<GatherRangeType>(rowWidthOffset), leftOffMask);
        AscendC::Reg::DataCopy(leftOffsetAddr, leftOffsetReg, mask);

        // rightOffset: 有效位置（rightIdx > 0）为 rowWidthOffset，ZERO_PAD 位置为 0
        AscendC::Reg::RegTensor<GatherRangeType> rightOffsetReg;
        AscendC::Reg::Duplicate(rightOffsetReg, static_cast<GatherRangeType>(0), mask);
        AscendC::Reg::MaskReg rightOffMask;
        AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(rightOffMask, rightIdxReg, zeroReg, mask);
        AscendC::Reg::Duplicate(rightOffsetReg, static_cast<GatherRangeType>(rowWidthOffset), rightOffMask);
        AscendC::Reg::DataCopy(rightOffsetAddr, rightOffsetReg, mask);
    }
}

// ========== GenerateCrossChannelWAxisIndices 实现 ==========
/**
 * @brief 跨 C 通道生成 W 轴折叠索引（VF 实现）
 *
 * 对 totalRows 行（可跨多个 C 通道）的每列生成 mid/left/right 三组索引。
 * 批内位置 x = globalRow * outW + col，其中：
 *   globalRow = c * outH + r（c 为相对 C 通道号，r 为 H 行号）
 *
 * chanHBase = c * hwSize + (leftPadH + r) * width + ZERO_PAD_CNT + leftPadW
 * midIdx    = chanHBase + col
 * leftIdx   = chanHBase - col [- 1]，有效范围外清零
 * rightIdx  = chanHBase + (2*outW-2 or 2*outW-1) - col，有效范围外清零
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::GenerateCrossChannelWAxisIndices(
    uint32_t totalRows, uint16_t outW, uint16_t outH, int64_t width, uint64_t hwSize, int64_t leftPadW,
    int64_t rightPadW, int64_t leftPadH, __local_mem__ GatherRangeType* midIdxAddr,
    __local_mem__ GatherRangeType* leftIdxAddr, __local_mem__ GatherRangeType* rightIdxAddr)
{
    __VEC_SCOPE__
    {
        uint32_t vLen = totalRows * static_cast<uint32_t>(outW);
        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

        // x = [0, 1, ..., vLen-1]
        AscendC::Reg::RegTensor<GatherRangeType> xReg;
        AscendC::Reg::Arange(xReg, static_cast<GatherRangeType>(0));

        // 第一层分解：globalRow = x / outW，col = x % outW
        AscendC::Reg::RegTensor<GatherRangeType> globalRowReg, colReg, tmpReg;
        AscendC::Reg::RegTensor<GatherRangeType> outWReg;
        AscendC::Reg::Duplicate(outWReg, static_cast<GatherRangeType>(outW), mask);
        AscendC::Reg::Div(globalRowReg, xReg, outWReg, mask);
        AscendC::Reg::Muls(tmpReg, globalRowReg, static_cast<GatherRangeType>(outW), mask);
        AscendC::Reg::Sub(colReg, xReg, tmpReg, mask);

        // 第二层分解：c = globalRow / outH，r = globalRow % outH
        AscendC::Reg::RegTensor<GatherRangeType> cReg, rReg;
        AscendC::Reg::RegTensor<GatherRangeType> outHReg;
        AscendC::Reg::Duplicate(outHReg, static_cast<GatherRangeType>(outH), mask);
        AscendC::Reg::Div(cReg, globalRowReg, outHReg, mask);
        AscendC::Reg::Muls(tmpReg, cReg, static_cast<GatherRangeType>(outH), mask);
        AscendC::Reg::Sub(rReg, globalRowReg, tmpReg, mask);

        // chanHBase = c * hwSize + (leftPadH + r) * width + ZERO_PAD_CNT + leftPadW
        AscendC::Reg::RegTensor<GatherRangeType> chanHBaseReg;
        AscendC::Reg::Muls(chanHBaseReg, cReg, static_cast<GatherRangeType>(hwSize), mask);
        AscendC::Reg::Muls(tmpReg, rReg, static_cast<GatherRangeType>(width), mask);
        AscendC::Reg::Add(chanHBaseReg, chanHBaseReg, tmpReg, mask);
        AscendC::Reg::RegTensor<GatherRangeType> constBaseReg;
        AscendC::Reg::Duplicate(constBaseReg, static_cast<GatherRangeType>(leftPadH * width + ZERO_PAD_CNT + leftPadW),
                                mask);
        AscendC::Reg::Add(chanHBaseReg, chanHBaseReg, constBaseReg, mask);

        // ===== midIdx = chanHBase + col =====
        AscendC::Reg::RegTensor<GatherRangeType> midIdxReg;
        AscendC::Reg::Add(midIdxReg, chanHBaseReg, colReg, mask);
        AscendC::Reg::DataCopy(midIdxAddr, midIdxReg, mask);

        // ===== leftIdx（与 GenerateFoldWAxisIndices 相同的清零逻辑）=====
        // 注意：清零时必须使用 MERGING 模式，仅将指定位置置 0，保留其余有效索引
        AscendC::Reg::RegTensor<GatherRangeType> leftIdxReg;
        if constexpr (IS_REFLECT) {
            AscendC::Reg::Sub(leftIdxReg, chanHBaseReg, colReg, mask);
            AscendC::Reg::RegTensor<GatherRangeType> oneReg;
            AscendC::Reg::Duplicate(oneReg, static_cast<GatherRangeType>(1), mask);
            AscendC::Reg::MaskReg colLt1Mask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colLt1Mask, oneReg, colReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                leftIdxReg, static_cast<GatherRangeType>(0), colLt1Mask);
            AscendC::Reg::RegTensor<GatherRangeType> leftPWReg;
            AscendC::Reg::Duplicate(leftPWReg, static_cast<GatherRangeType>(leftPadW), mask);
            AscendC::Reg::MaskReg colGtLPWMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colGtLPWMask, colReg, leftPWReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                leftIdxReg, static_cast<GatherRangeType>(0), colGtLPWMask);
        } else {
            AscendC::Reg::Sub(leftIdxReg, chanHBaseReg, colReg, mask);
            AscendC::Reg::RegTensor<GatherRangeType> oneReg;
            AscendC::Reg::Duplicate(oneReg, static_cast<GatherRangeType>(1), mask);
            AscendC::Reg::Sub(leftIdxReg, leftIdxReg, oneReg, mask);
            AscendC::Reg::RegTensor<GatherRangeType> lpwM1Reg;
            AscendC::Reg::Duplicate(lpwM1Reg, static_cast<GatherRangeType>(leftPadW - 1), mask);
            AscendC::Reg::MaskReg colGeLPWMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colGeLPWMask, colReg, lpwM1Reg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                leftIdxReg, static_cast<GatherRangeType>(0), colGeLPWMask);
        }
        AscendC::Reg::DataCopy(leftIdxAddr, leftIdxReg, mask);

        // ===== rightIdx（与 GenerateFoldWAxisIndices 相同的清零逻辑）=====
        // 注意：清零时必须使用 MERGING 模式，仅将指定位置置 0，保留其余有效索引
        AscendC::Reg::RegTensor<GatherRangeType> rightIdxReg;
        if constexpr (IS_REFLECT) {
            AscendC::Reg::RegTensor<GatherRangeType> rightConstReg;
            AscendC::Reg::Duplicate(rightConstReg, static_cast<GatherRangeType>(2 * static_cast<int64_t>(outW) - 2),
                                    mask);
            AscendC::Reg::Add(rightIdxReg, chanHBaseReg, rightConstReg, mask);
            AscendC::Reg::Sub(rightIdxReg, rightIdxReg, colReg, mask);
            AscendC::Reg::RegTensor<GatherRangeType> lowBoundReg;
            AscendC::Reg::Duplicate(lowBoundReg,
                                    static_cast<GatherRangeType>(static_cast<int64_t>(outW) - rightPadW - 1), mask);
            AscendC::Reg::MaskReg colLtLowMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colLtLowMask, lowBoundReg, colReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                rightIdxReg, static_cast<GatherRangeType>(0), colLtLowMask);
            AscendC::Reg::RegTensor<GatherRangeType> highM1Reg;
            AscendC::Reg::Duplicate(highM1Reg, static_cast<GatherRangeType>(static_cast<int64_t>(outW) - 2), mask);
            AscendC::Reg::MaskReg colGtHighMask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colGtHighMask, colReg, highM1Reg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                rightIdxReg, static_cast<GatherRangeType>(0), colGtHighMask);
        } else {
            AscendC::Reg::RegTensor<GatherRangeType> rightConstReg;
            AscendC::Reg::Duplicate(rightConstReg, static_cast<GatherRangeType>(2 * static_cast<int64_t>(outW) - 1),
                                    mask);
            AscendC::Reg::Add(rightIdxReg, chanHBaseReg, rightConstReg, mask);
            AscendC::Reg::Sub(rightIdxReg, rightIdxReg, colReg, mask);
            AscendC::Reg::RegTensor<GatherRangeType> lowBound2Reg;
            AscendC::Reg::Duplicate(lowBound2Reg, static_cast<GatherRangeType>(static_cast<int64_t>(outW) - rightPadW),
                                    mask);
            AscendC::Reg::MaskReg colLtLow2Mask;
            AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(colLtLow2Mask, lowBound2Reg, colReg, mask);
            AscendC::Reg::Duplicate<GatherRangeType, Reg::MaskMergeMode::MERGING, GatherRangeType>(
                rightIdxReg, static_cast<GatherRangeType>(0), colLtLow2Mask);
        }
        AscendC::Reg::DataCopy(rightIdxAddr, rightIdxReg, mask);
    }
}

// ========== FoldHWAxis 辅助函数实现 ==========

/**
 * @brief 生成 W 轴折叠的索引数组
 *
 * 索引映射公式（以 outW=3, leftPadW=1, rightPadW=1, width=5 为例）：
 *
 * 输入行布局: [idx0, idx1, idx2, idx3, idx4] 对应 [L0, M0, M1, M2, R0]
 * 输出列:     [col0, col1, col2]
 *
 * reflect 模式映射：
 *   col=0 (left pad区): midIdx=1, leftIdx=2 (反射到M1), rightIdx=0
 *   col=1 (mid区):      midIdx=2, leftIdx=0, rightIdx=0
 *   col=2 (right pad区): midIdx=3, leftIdx=0, rightIdx=2 (反射到M1)
 *
 * symmetric 模式映射：
 *   col=0 (left pad区): midIdx=1, leftIdx=1 (对称到M0), rightIdx=0
 *   col=1 (mid区):      midIdx=2, leftIdx=0, rightIdx=0
 *   col=2 (right pad区): midIdx=3, leftIdx=0, rightIdx=3 (对称到M2)
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::GenerateWAxisIndices(
    uint64_t cOffset, uint16_t rowsPerBatch, uint16_t outW, uint16_t width, int64_t leftPadW, int64_t rightPadW,
    int64_t leftPadH, GatherRangeType batchOffset, __local_mem__ GatherRangeType* midIdxAddr,
    __local_mem__ GatherRangeType* leftIdxAddr, __local_mem__ GatherRangeType* rightIdxAddr,
    __local_mem__ GatherRangeType* midOffsetAddr, __local_mem__ GatherRangeType* leftOffsetAddr,
    __local_mem__ GatherRangeType* rightOffsetAddr)
{
    for (uint16_t r = 0; r < rowsPerBatch; r++) {
        // rowOffset = 当前 C 通道偏移 + (leftPadH + r) 行的起始位置 + ZERO_PAD_CNT
        // 例如: cOffset=0, leftPadH=1, r=0, width=5 => rowOffset = 0 + 1*5 + 8 = 13
        uint64_t rowOffset = cOffset + (leftPadH + r) * width + ZERO_PAD_CNT;

        for (uint16_t col = 0; col < outW; col++) {
            uint32_t idx = r * outW + col;

            // ===== midIdx: 中间区域索引 =====
            // 公式: rowOffset + leftPadW + col
            // 例如: rowOffset=13, leftPadW=1, col=0 => midIdx = 13 + 1 + 0 = 14
            midIdxAddr[idx] = static_cast<GatherRangeType>(rowOffset + leftPadW + col);
            midOffsetAddr[idx] = batchOffset; // 每批行偏移

            // ===== leftIdx: 左 pad 映射位置 =====
            // Reflect: col > 0 && col <= leftPadW 时有映射（M[1]到M[leftPadW]接收left pad）
            // Symmetric: col < leftPadW 时有映射（M[0]到M[leftPadW-1]接收left pad）
            bool hasLeftMapping = false;
            if constexpr (modeName == 2) {
                // Reflect
                hasLeftMapping = (leftPadW > 0 && col > 0 && col <= leftPadW);
            } else {
                // Symmetric
                hasLeftMapping = (leftPadW > 0 && col < leftPadW);
            }

            if (hasLeftMapping) {
                if constexpr (modeName == 2) {
                    // reflect: col=1 映射到 leftPadW-1 位置的 left pad 值
                    // 公式: rowOffset + leftPadW - col
                    leftIdxAddr[idx] = static_cast<GatherRangeType>(rowOffset + leftPadW - col);
                } else {
                    // symmetric: col=0 映射到 leftPadW-1 位置的 left pad 值
                    // 公式: rowOffset + leftPadW - 1 - col
                    leftIdxAddr[idx] = static_cast<GatherRangeType>(rowOffset + leftPadW - 1 - col);
                }
                leftOffsetAddr[idx] = batchOffset;
            } else {
                leftIdxAddr[idx] = 0;    // 指向 ZERO_PAD 区域
                leftOffsetAddr[idx] = 0; // 偏移为 0，保持指向 ZERO_PAD
            }

            // ===== rightIdx: 右 pad 映射位置 =====
            // Reflect: col >= outW - rightPadW - 1 && col < outW - 1 时有映射
            // Symmetric: col >= outW - rightPadW 时有映射
            // 正确公式: rightIdx = rowOffset + leftPadW + outW + rightPadW - 1 - rightCol
            //         = rowOffset + width - 1 - rightCol (width = leftPadW + outW + rightPadW)
            bool hasRightMapping = false;
            if constexpr (modeName == 2) {
                // Reflect
                hasRightMapping = (rightPadW > 0 && col >= outW - rightPadW - 1 && col < outW - 1);
            } else {
                // Symmetric
                hasRightMapping = (rightPadW > 0 && col >= outW - rightPadW);
            }

            if (hasRightMapping) {
                // reflect 和 symmetric 模式下公式相同，都是指向 right pad 区域
                uint32_t rightCol;
                if constexpr (modeName == 2) { // Reflect
                    rightCol = col - (outW - rightPadW - 1);
                } else { // Symmetric
                    rightCol = col - (outW - rightPadW);
                }
                rightIdxAddr[idx] = static_cast<GatherRangeType>(rowOffset + leftPadW + outW + rightPadW - 1 -
                                                                 rightCol);
                rightOffsetAddr[idx] = batchOffset;
            } else {
                rightIdxAddr[idx] = 0;
                rightOffsetAddr[idx] = 0;
            }
        }
    }
}

/**
 * @brief 折叠 H 轴的 pad 数据
 *
 * H 轴折叠将 pad 区域的数据累加到对应的 mid 区域
 *
 * 映射公式（以 height=5, leftPadH=1, rightPadH=1, outH=3 为例）：
 *
 * 输入行布局: [Row0, Row1, Row2, Row3, Row4] 对应 [L0, M0, M1, M2, R0]
 *
 * Left pad 折叠 (reflect):
 *   srcRow=0 (L0) -> dstRow = 2*leftPadH - srcRow = 2*1 - 0 = 2 (M1)
 *
 * Left pad 折叠 (symmetric):
 *   srcRow=0 (L0) -> dstRow = 2*leftPadH - 1 - srcRow = 2*1 - 1 - 0 = 1 (M0)
 *
 * Right pad 折叠 (reflect):
 *   row=0 (R0, srcRow=4) -> dstRow = leftPadH + outH - 2 - row = 1 + 3 - 2 - 0 = 2 (M1)
 *
 * Right pad 折叠 (symmetric):
 *   row=0 (R0, srcRow=4) -> dstRow = leftPadH + outH - 1 - row = 1 + 3 - 1 - 0 = 3 (M2)
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldHAxisPad(int64_t factor, uint64_t hwSize, int64_t width,
                                                                  int64_t padCount, bool isLeftPad, int64_t leftPadH,
                                                                  int64_t outH, int64_t nFactor, uint64_t nStride,
                                                                  __local_mem__ PromoteDataT* baseAddr,
                                                                  __local_mem__ GatherRangeType* srcIdxAddr,
                                                                  __local_mem__ GatherRangeType* dstIdxAddr)
{
    if (padCount <= 0 || nFactor <= 0) {
        return;
    }

    // elemPerN: 每个 N 贡献的元素数（factor 个 C 通道 × width 个 W 列）
    int64_t elemPerN = factor * width;

    // ===== 优化路径：N 轴批处理 =====
    // 条件：有多个 N，且每个 N 的元素数 ≤ VL_RANGE_CNT（即一次 VF 可处理多个 N）
    if (nFactor > 1 && elemPerN <= static_cast<int64_t>(VL_RANGE_CNT)) {
        int64_t nPerVF = static_cast<int64_t>(VL_RANGE_CNT) / elemPerN;
        if (nPerVF < 1)
            nPerVF = 1;

        for (int64_t padRow = 0; padRow < padCount; ++padRow) {
            int64_t srcRowOffset, dstRowOffset;
            CalcHAxisPadRowOffset(padRow, padCount, isLeftPad, leftPadH, outH, width, srcRowOffset, dstRowOffset);

            for (int64_t nBase = 0; nBase < nFactor; nBase += nPerVF) {
                int64_t curN = Std::min<int64_t>(nPerVF, nFactor - nBase);
                int64_t totalElems = curN * elemPerN;

                if (nBase == 0) {
                    // 首批：三级分解 x → (nInBatch, c, col)
                    // srcIdx = nInBatch*nStride + c*hwSize + srcRowOffset + col
                    // dstIdx = nInBatch*nStride + c*hwSize + dstRowOffset + col
                    __VEC_SCOPE__
                    {
                        uint32_t vLen = static_cast<uint32_t>(totalElems);
                        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

                        AscendC::Reg::RegTensor<GatherRangeType> xReg;
                        AscendC::Reg::Arange(xReg, static_cast<GatherRangeType>(0));

                        // N 级分解：nInBatch = x / elemPerN, xi = x % elemPerN
                        AscendC::Reg::RegTensor<GatherRangeType> nInBatchReg, xiReg, elemPerNReg, tmpReg;
                        AscendC::Reg::Duplicate(elemPerNReg, static_cast<GatherRangeType>(elemPerN), mask);
                        AscendC::Reg::Div(nInBatchReg, xReg, elemPerNReg, mask);
                        AscendC::Reg::Muls(tmpReg, nInBatchReg, static_cast<GatherRangeType>(elemPerN), mask);
                        AscendC::Reg::Sub(xiReg, xReg, tmpReg, mask);

                        // C 级分解：c = xi / width, col = xi % width
                        AscendC::Reg::RegTensor<GatherRangeType> cReg, colReg, wReg;
                        AscendC::Reg::Duplicate(wReg, static_cast<GatherRangeType>(width), mask);
                        AscendC::Reg::Div(cReg, xiReg, wReg, mask);
                        AscendC::Reg::Muls(tmpReg, cReg, static_cast<GatherRangeType>(width), mask);
                        AscendC::Reg::Sub(colReg, xiReg, tmpReg, mask);

                        // N 轴偏移（src 和 dst 共享）
                        AscendC::Reg::RegTensor<GatherRangeType> nOffsetReg;
                        AscendC::Reg::Muls(nOffsetReg, nInBatchReg, static_cast<GatherRangeType>(nStride), mask);

                        // srcIdx = nOffset + c*hwSize + srcRowOffset + col
                        AscendC::Reg::RegTensor<GatherRangeType> srcIdxReg;
                        AscendC::Reg::Muls(srcIdxReg, cReg, static_cast<GatherRangeType>(hwSize), mask);
                        AscendC::Reg::Add(srcIdxReg, srcIdxReg, nOffsetReg, mask);
                        AscendC::Reg::RegTensor<GatherRangeType> srcBaseReg;
                        AscendC::Reg::Duplicate(srcBaseReg, static_cast<GatherRangeType>(srcRowOffset), mask);
                        AscendC::Reg::Add(srcIdxReg, srcIdxReg, srcBaseReg, mask);
                        AscendC::Reg::Add(srcIdxReg, srcIdxReg, colReg, mask);

                        // dstIdx = nOffset + c*hwSize + dstRowOffset + col
                        AscendC::Reg::RegTensor<GatherRangeType> dstIdxReg;
                        AscendC::Reg::Muls(dstIdxReg, cReg, static_cast<GatherRangeType>(hwSize), mask);
                        AscendC::Reg::Add(dstIdxReg, dstIdxReg, nOffsetReg, mask);
                        AscendC::Reg::RegTensor<GatherRangeType> dstBaseReg;
                        AscendC::Reg::Duplicate(dstBaseReg, static_cast<GatherRangeType>(dstRowOffset), mask);
                        AscendC::Reg::Add(dstIdxReg, dstIdxReg, dstBaseReg, mask);
                        AscendC::Reg::Add(dstIdxReg, dstIdxReg, colReg, mask);

                        AscendC::Reg::DataCopy(srcIdxAddr, srcIdxReg, mask);
                        AscendC::Reg::DataCopy(dstIdxAddr, dstIdxReg, mask);
                    }
                } else {
                    // 后续批：src 和 dst 同向递增 nPerVF*nStride（N 轴两端方向一致）
                    GatherRangeType nStep = static_cast<GatherRangeType>(nPerVF * static_cast<int64_t>(nStride));
                    __VEC_SCOPE__
                    {
                        uint32_t vLen = static_cast<uint32_t>(totalElems);
                        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);
                        AscendC::Reg::RegTensor<GatherRangeType> srcIdxReg, dstIdxReg, stepReg;
                        AscendC::Reg::DataCopy(srcIdxReg, srcIdxAddr);
                        AscendC::Reg::DataCopy(dstIdxReg, dstIdxAddr);
                        AscendC::Reg::Duplicate(stepReg, nStep, mask);
                        AscendC::Reg::Add(srcIdxReg, srcIdxReg, stepReg, mask);
                        AscendC::Reg::Add(dstIdxReg, dstIdxReg, stepReg, mask);
                        AscendC::Reg::DataCopy(srcIdxAddr, srcIdxReg, mask);
                        AscendC::Reg::DataCopy(dstIdxAddr, dstIdxReg, mask);
                    }
                }
                GatherAddScatter(baseAddr + ZERO_PAD_CNT, srcIdxAddr, dstIdxAddr, static_cast<uint64_t>(totalElems));
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
            }
        }
    } else {
        // ===== 回退路径：C 轴分批，N 循环在内部 =====
        // 适用于：nFactor==1（FoldHWAxis 调用），或 elemPerN > VL_RANGE_CNT（C 通道很多）
        int64_t factorsPerVF = static_cast<int64_t>(VL_RANGE_CNT) / width;
        if (factorsPerVF < 1)
            factorsPerVF = 1;

        for (int64_t n = 0; n < nFactor; n++) {
            __local_mem__ PromoteDataT* nBaseAddr = baseAddr + n * static_cast<int64_t>(nStride);

            for (int64_t padRow = 0; padRow < padCount; ++padRow) {
                int64_t srcRowOffset, dstRowOffset;
                CalcHAxisPadRowOffset(padRow, padCount, isLeftPad, leftPadH, outH, width, srcRowOffset, dstRowOffset);

                int64_t remainC = factor;
                int64_t cBase = 0;

                // 首批：用 VF 生成索引
                int64_t curC = Std::min<int64_t>(factorsPerVF, remainC);
                int64_t curElements = curC * width;

                __VEC_SCOPE__
                {
                    uint32_t vLen = static_cast<uint32_t>(curElements);
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

                    AscendC::Reg::RegTensor<GatherRangeType> xReg;
                    AscendC::Reg::Arange(xReg, static_cast<GatherRangeType>(0));

                    AscendC::Reg::RegTensor<GatherRangeType> cReg, colReg, tmpReg, wReg;
                    AscendC::Reg::Duplicate(wReg, static_cast<GatherRangeType>(width), mask);
                    AscendC::Reg::Div(cReg, xReg, wReg, mask);
                    AscendC::Reg::Muls(tmpReg, cReg, static_cast<GatherRangeType>(width), mask);
                    AscendC::Reg::Sub(colReg, xReg, tmpReg, mask);

                    AscendC::Reg::RegTensor<GatherRangeType> absCReg;
                    AscendC::Reg::Duplicate(absCReg, static_cast<GatherRangeType>(cBase), mask);
                    AscendC::Reg::Add(absCReg, absCReg, cReg, mask);

                    AscendC::Reg::RegTensor<GatherRangeType> srcIdxReg, dstIdxReg;
                    AscendC::Reg::Muls(srcIdxReg, absCReg, static_cast<GatherRangeType>(hwSize), mask);
                    AscendC::Reg::Muls(dstIdxReg, absCReg, static_cast<GatherRangeType>(hwSize), mask);

                    AscendC::Reg::RegTensor<GatherRangeType> srcBaseReg, dstBaseReg;
                    AscendC::Reg::Duplicate(srcBaseReg, static_cast<GatherRangeType>(srcRowOffset), mask);
                    AscendC::Reg::Duplicate(dstBaseReg, static_cast<GatherRangeType>(dstRowOffset), mask);

                    AscendC::Reg::Add(srcIdxReg, srcIdxReg, srcBaseReg, mask);
                    AscendC::Reg::Add(dstIdxReg, dstIdxReg, dstBaseReg, mask);
                    AscendC::Reg::Add(srcIdxReg, srcIdxReg, colReg, mask);
                    AscendC::Reg::Add(dstIdxReg, dstIdxReg, colReg, mask);

                    AscendC::Reg::DataCopy(srcIdxAddr, srcIdxReg, mask);
                    AscendC::Reg::DataCopy(dstIdxAddr, dstIdxReg, mask);
                }

                GatherAddScatter(nBaseAddr + ZERO_PAD_CNT, srcIdxAddr, dstIdxAddr, static_cast<uint64_t>(curElements));
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);

                remainC -= curC;
                cBase += curC;
                const int64_t cStep = factorsPerVF * static_cast<int64_t>(hwSize);

                while (remainC > 0) {
                    curC = Std::min<int64_t>(factorsPerVF, remainC);
                    curElements = curC * width;

                    __VEC_SCOPE__
                    {
                        uint32_t vLen = static_cast<uint32_t>(curElements);
                        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

                        AscendC::Reg::RegTensor<GatherRangeType> stepReg;
                        AscendC::Reg::Duplicate(stepReg, static_cast<GatherRangeType>(cStep), mask);

                        AscendC::Reg::RegTensor<GatherRangeType> srcIdxReg, dstIdxReg;
                        AscendC::Reg::DataCopy(srcIdxReg, srcIdxAddr);
                        AscendC::Reg::DataCopy(dstIdxReg, dstIdxAddr);

                        AscendC::Reg::Add(srcIdxReg, srcIdxReg, stepReg, mask);
                        AscendC::Reg::Add(dstIdxReg, dstIdxReg, stepReg, mask);

                        AscendC::Reg::DataCopy(srcIdxAddr, srcIdxReg, mask);
                        AscendC::Reg::DataCopy(dstIdxAddr, dstIdxReg, mask);
                    }

                    GatherAddScatter(nBaseAddr + ZERO_PAD_CNT, srcIdxAddr, dstIdxAddr,
                                     static_cast<uint64_t>(curElements));
                    event_t nextEventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                    SetFlag<HardEvent::V_S>(nextEventId);
                    WaitFlag<HardEvent::V_S>(nextEventId);

                    remainC -= curC;
                    cBase += curC;
                }
            }
        }
    }
}

/**
 * @brief 折叠 C 轴的 pad 数据（支持跨 N 批处理）
 *
 * 将 C 轴的 pad 通道数据累加到对应的 mid 通道。每个通道包含 hwSize 个元素，
 * 通过 GatherAddScatter 实现同一 buffer 内不同通道间的数据累加。
 *
 * 映射公式（Reflect 模式）：
 *   - left pad:  srcC -> dstC = 2*leftPadC - srcC
 *   - right pad: cIdx -> dstC = leftPadC + outC - 2 - cIdx
 *
 * 映射公式（Symmetric 模式）：
 *   - left pad:  srcC -> dstC = 2*leftPadC - 1 - srcC
 *   - right pad: cIdx -> dstC = leftPadC + outC - 1 - cIdx
 *
 * 优化策略（当 padCount*hwSize <= VL_RANGE_CNT 时）：
 *   引入两层批处理：
 *   1. C pad 批：cPadsPerVF = VL_RANGE_CNT / hwSize，一批处理多个 pad C 通道
 *   2. N 批：nPerVF = VL_RANGE_CNT / (padCount*hwSize)，一批处理多个 N
 *   两层合并后，一次 VF 同时处理 nPerVF 个 N 的所有 padCount 个 pad C 通道。
 *   N 间的 srcIdx/dstIdx 增量方向相同（均 +nStride），后续批次增量更新简单。
 *
 * @param hwSize      单个 C 通道的 H*W 大小
 * @param padCount    pad 通道数（leftPadC 或 rightPadC）
 * @param isLeftPad   是否为 left pad（true=left, false=right）
 * @param leftPadC    左 pad 通道数
 * @param outC        输出通道数
 * @param factor      N 轴的数量
 * @param nStride     相邻 N 之间的元素步长（= cHeight * hwSize）
 * @param baseAddr    全局 base 地址（midAddr + ZERO_PAD_CNT，不含 N 偏移）
 * @param srcIdxAddr  源索引数组地址
 * @param dstIdxAddr  目标索引数组地址
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldCAxisPad(uint64_t hwSize, int64_t padCount, bool isLeftPad,
                                                                  int64_t leftPadC, int64_t outC, int64_t factor,
                                                                  uint64_t nStride,
                                                                  __local_mem__ PromoteDataT* baseAddr,
                                                                  __local_mem__ GatherRangeType* srcIdxAddr,
                                                                  __local_mem__ GatherRangeType* dstIdxAddr)
{
    if (padCount <= 0 || factor <= 0)
        return;

    // padElem: 每个 N 的所有 pad C 通道元素总数
    int64_t padElem = padCount * static_cast<int64_t>(hwSize);

    if (padElem <= static_cast<int64_t>(VL_RANGE_CNT)) {
        // ===== 优化路径：一次 VF 同时处理多个 N 的所有 pad C 通道 =====
        // nPerVF: 一次 VF 能处理几个 N（每个 N 贡献 padElem 个元素）
        int64_t nPerVF = static_cast<int64_t>(VL_RANGE_CNT) / padElem;
        if (nPerVF < 1)
            nPerVF = 1;

        // srcC0 / dstC0: padBase=0, nBase=0 时的初始通道号
        int64_t srcC0, dstC0;
        if (isLeftPad) {
            srcC0 = 0;
            dstC0 = (modeName == 2) ? (2 * leftPadC) : (2 * leftPadC - 1);
        } else {
            srcC0 = leftPadC + outC;
            dstC0 = (modeName == 2) ? (leftPadC + outC - 2) : (leftPadC + outC - 1);
        }

        for (int64_t nBase = 0; nBase < factor; nBase += nPerVF) {
            int64_t curN = Std::min<int64_t>(nPerVF, factor - nBase);
            int64_t totalElems = curN * padElem;

            if (nBase == 0) {
                // 第一批：Arange(0) + 向量分解，一次性生成所有 curN 个 N 的完整索引
                // 元素序号 x = [0..totalElems-1]，分解为 (nInBatch, ci, e)：
                //   nInBatch = x / padElem         — 在当前批内第几个 N
                //   xi       = x % padElem         — 该 N 内的序号
                //   ci       = xi / hwSize          — 该 N 内第几个 pad C 通道
                //   e        = xi % hwSize          — 通道内偏移
                //
                // srcIdx = (nBase + nInBatch) * nStride + srcC0*hwSize + xi
                //          （srcC 随 ci 递增，xi = ci*hwSize+e，直接用 Arange 生成 srcC0*hwSize+xi 部分）
                // dstIdx = (nBase + nInBatch) * nStride + (dstC0 - ci) * hwSize + e
                __VEC_SCOPE__
                {
                    uint32_t vLen = static_cast<uint32_t>(totalElems);
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

                    AscendC::Reg::RegTensor<GatherRangeType> xReg;
                    AscendC::Reg::Arange(xReg, static_cast<GatherRangeType>(0));

                    // ── N 层分解 ──
                    AscendC::Reg::RegTensor<GatherRangeType> nInBatchReg, xiReg, padElemReg, tmpReg;
                    AscendC::Reg::Duplicate(padElemReg, static_cast<GatherRangeType>(padElem), mask);
                    AscendC::Reg::Div(nInBatchReg, xReg, padElemReg, mask);
                    AscendC::Reg::Muls(tmpReg, nInBatchReg, static_cast<GatherRangeType>(padElem), mask);
                    AscendC::Reg::Sub(xiReg, xReg, tmpReg, mask); // xi = x % padElem

                    // ── C 层分解 ──
                    AscendC::Reg::RegTensor<GatherRangeType> ciReg, eReg, hwReg;
                    AscendC::Reg::Duplicate(hwReg, static_cast<GatherRangeType>(hwSize), mask);
                    AscendC::Reg::Div(ciReg, xiReg, hwReg, mask);
                    AscendC::Reg::Muls(tmpReg, ciReg, static_cast<GatherRangeType>(hwSize), mask);
                    AscendC::Reg::Sub(eReg, xiReg, tmpReg, mask); // e = xi % hwSize

                    // ── N 轴偏移（两者共享）──
                    // nOffset_reg = nInBatch * nStride  （nBase=0 时不含常量偏移）
                    AscendC::Reg::RegTensor<GatherRangeType> nOffsetReg;
                    AscendC::Reg::Muls(nOffsetReg, nInBatchReg, static_cast<GatherRangeType>(nStride), mask);

                    // ── srcIdx ──
                    // srcIdx = nOffset + srcC0*hwSize + xi
                    //        = nOffset + (srcC0*hwSize + xi)     [其中 srcC0*hwSize+xi 是 Arange 部分]
                    // 直接：srcIdx = nOffset + Duplicate(srcC0*hwSize) + xiReg
                    AscendC::Reg::RegTensor<GatherRangeType> srcIdxReg;
                    AscendC::Reg::Duplicate(srcIdxReg,
                                            static_cast<GatherRangeType>(srcC0 * static_cast<int64_t>(hwSize)), mask);
                    AscendC::Reg::Add(srcIdxReg, srcIdxReg, xiReg, mask);
                    AscendC::Reg::Add(srcIdxReg, srcIdxReg, nOffsetReg, mask);

                    // ── dstIdx ──
                    // dstC = dstC0 - ci
                    // dstIdx = nOffset + dstC * hwSize + e
                    AscendC::Reg::RegTensor<GatherRangeType> dstCReg, dstIdxReg;
                    AscendC::Reg::Duplicate(dstCReg, static_cast<GatherRangeType>(dstC0), mask);
                    AscendC::Reg::Sub(dstCReg, dstCReg, ciReg, mask);
                    AscendC::Reg::Muls(dstIdxReg, dstCReg, static_cast<GatherRangeType>(hwSize), mask);
                    AscendC::Reg::Add(dstIdxReg, dstIdxReg, eReg, mask);
                    AscendC::Reg::Add(dstIdxReg, dstIdxReg, nOffsetReg, mask);

                    AscendC::Reg::DataCopy(srcIdxAddr, srcIdxReg, mask);
                    AscendC::Reg::DataCopy(dstIdxAddr, dstIdxReg, mask);
                }
            } else {
                // 后续批次：N 轴移动，srcIdx 和 dstIdx 均加 nPerVF * nStride（方向相同）
                GatherRangeType nStep = static_cast<GatherRangeType>(nPerVF * static_cast<int64_t>(nStride));

                __VEC_SCOPE__
                {
                    uint32_t vLen = static_cast<uint32_t>(totalElems);
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

                    AscendC::Reg::RegTensor<GatherRangeType> srcIdxReg, dstIdxReg, stepReg;
                    AscendC::Reg::DataCopy(srcIdxReg, srcIdxAddr);
                    AscendC::Reg::DataCopy(dstIdxReg, dstIdxAddr);
                    AscendC::Reg::Duplicate(stepReg, nStep, mask);
                    AscendC::Reg::Add(srcIdxReg, srcIdxReg, stepReg, mask);
                    AscendC::Reg::Add(dstIdxReg, dstIdxReg, stepReg, mask);
                    AscendC::Reg::DataCopy(srcIdxAddr, srcIdxReg, mask);
                    AscendC::Reg::DataCopy(dstIdxAddr, dstIdxReg, mask);
                }
            }

            GatherAddScatter(baseAddr, srcIdxAddr, dstIdxAddr, static_cast<uint64_t>(totalElems));

            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
        }
    } else {
        // ===== 回退路径：padElem > VL_RANGE_CNT，每个 N 逐 pad C 通道分批处理 =====
        int64_t elementsPerVF = VL_RANGE_CNT;

        LocalTensor<GatherRangeType> srcIdxLocal = leftPadIdx_.Get<GatherRangeType>();
        LocalTensor<GatherRangeType> dstIdxLocal = midIdx_.Get<GatherRangeType>();

        for (int64_t n = 0; n < factor; n++) {
            __local_mem__ PromoteDataT* nMidAddr = baseAddr + n * static_cast<int64_t>(nStride);

            for (int64_t cIdx = 0; cIdx < padCount; cIdx++) {
                int64_t srcC, dstC;
                if (isLeftPad) {
                    srcC = cIdx;
                    dstC = (modeName == 2) ? (2 * leftPadC - srcC) : (2 * leftPadC - 1 - srcC);
                } else {
                    srcC = leftPadC + outC + cIdx;
                    dstC = (modeName == 2) ? (leftPadC + outC - 2 - cIdx) : (leftPadC + outC - 1 - cIdx);
                }

                uint64_t srcOffset = static_cast<uint64_t>(srcC) * hwSize;
                uint64_t dstOffset = static_cast<uint64_t>(dstC) * hwSize;

                // 生成初始索引（覆盖首批 VL_RANGE_CNT 个元素）
                __VEC_SCOPE__
                {
                    uint32_t vLen = VL_RANGE_CNT;
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

                    AscendC::Reg::RegTensor<GatherRangeType> srcReg;
                    AscendC::Reg::Arange(srcReg, static_cast<GatherRangeType>(srcOffset));
                    AscendC::Reg::DataCopy(srcIdxAddr, srcReg, mask);

                    AscendC::Reg::RegTensor<GatherRangeType> dstReg;
                    AscendC::Reg::Arange(dstReg, static_cast<GatherRangeType>(dstOffset));
                    AscendC::Reg::DataCopy(dstIdxAddr, dstReg, mask);
                }

                // 分批处理 hwSize 个元素，后续批次 Adds 递增索引
                for (uint64_t startElem = 0; startElem < hwSize; startElem += elementsPerVF) {
                    uint32_t curElements = Std::min(static_cast<uint64_t>(elementsPerVF), hwSize - startElem);

                    if (startElem > 0) {
                        AscendC::Adds(srcIdxLocal, srcIdxLocal, static_cast<PromoteDataT>(elementsPerVF),
                                      elementsPerVF);
                        AscendC::Adds(dstIdxLocal, dstIdxLocal, static_cast<PromoteDataT>(elementsPerVF),
                                      elementsPerVF);
                    }

                    GatherAddScatter(nMidAddr, srcIdxAddr, dstIdxAddr, curElements);

                    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                    SetFlag<HardEvent::V_S>(eventId);
                    WaitFlag<HardEvent::V_S>(eventId);
                }
            }
        }
    }
}

// ========== FoldHWAxis 主函数 ==========
/**
 * @brief 折叠 H、W 两根轴的 pad 数据并输出
 *
 * 处理流程：
 * 1. 折叠 H 轴 left pad：将上方 pad 区域数据累加到对应的 mid 区域
 * 2. 折叠 H 轴 right pad：将下方 pad 区域数据累加到对应的 mid 区域
 * 3. 折叠 W 轴并输出：对每行进行 left/mid/right 三区域 gather、相加、输出
 *
 * 数据布局示例（C=1, H=5, W=5, leftPadH=1, rightPadH=1, leftPadW=1, rightPadW=1）：
 *
 * 输入 (5x5):          输出 (3x3):
 * [L00 L01 L02 L03 L04]
 * [M00 M01 M02 M03 M04]  ->  [O00 O01 O02]
 * [M10 M11 M12 M13 M14]  ->  [O10 O11 O12]
 * [M20 M21 M22 M23 M24]  ->  [O20 O21 O22]
 * [R00 R01 R02 R03 R04]
 *
 * 其中 L=上pad行, M=中间行, R=下pad行
 * 每行内: 第0列=左pad, 第1-3列=中间, 第4列=右pad
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldHWAxis(int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                                                LocalTensor<PromoteDataT>& outputLocal)
{
    if (factor <= 0) {
        return;
    }

    // ===== 1. 提取维度参数 =====
    // 切分在 C 轴时，C 轴的 pad 已经在 ProcessUbAxisPad 中处理过了
    // 这里只需要处理 H 轴和 W 轴的折叠
    int64_t height = inShape_[dimNum_ - 2];                  // 输入高度（含 pad）
    int64_t width = inShape_[dimNum_ - 1];                   // 输入宽度（含 pad）
    uint64_t hwSize = static_cast<uint64_t>(height * width); // 单个 C 通道的大小

    int64_t leftPadH = leftPad_[dimNum_ - 2];   // 上 pad 行数
    int64_t rightPadH = rightPad_[dimNum_ - 2]; // 下 pad 行数
    int64_t leftPadW = leftPad_[dimNum_ - 1];   // 左 pad 列数
    int64_t rightPadW = rightPad_[dimNum_ - 1]; // 右 pad 列数

    int64_t outH = outShape_[dimNum_ - 2]; // 输出高度
    int64_t outW = outShape_[dimNum_ - 1]; // 输出宽度

    // ===== 2. 获取数据地址 =====
    __local_mem__ PromoteDataT* midAddr = (__local_mem__ PromoteDataT*)midLocal.GetPhyAddr();
    __local_mem__ PromoteDataT* outputAddr = (__local_mem__ PromoteDataT*)outputLocal.GetPhyAddr();

    // 获取索引缓冲区（用于 H 轴折叠和 W 轴折叠）
    LocalTensor<GatherRangeType> leftIdxLocal = leftPadIdx_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> midIdxLocal = midIdx_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> rightIdxLocal = rightPadIdx_.Get<GatherRangeType>();

    __local_mem__ GatherRangeType* leftIdxAddr = (__local_mem__ GatherRangeType*)leftIdxLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* midIdxAddr = (__local_mem__ GatherRangeType*)midIdxLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* rightIdxAddr = (__local_mem__ GatherRangeType*)rightIdxLocal.GetPhyAddr();

    // ===== 3. 折叠 H 轴 pad =====
    // 对每个 C 通道（factor 个）折叠 H 轴的 pad（nFactor=1 退化为原逻辑）
    FoldHAxisPad(factor, hwSize, width, leftPadH, true, leftPadH, outH, 1, 0, midAddr, leftIdxAddr, midIdxAddr);
    FoldHAxisPad(factor, hwSize, width, rightPadH, false, leftPadH, outH, 1, 0, midAddr, leftIdxAddr, midIdxAddr);

    // ===== 4. 折叠 W 轴并输出（跨 C 通道优化版）=====
    // 优化：一次 VF 同时处理多个 C 通道的所有行，减少循环次数
    // 对于 H*W 很小的场景（如 outH=2, outW=3），原来每次 VF 只处理 6 元素（利用率 9%）
    // 优化后每次 VF 处理 cPerVF 个 C 通道的所有行，利用率大幅提升

    // 4.1 计算跨 C 通道的批次参数
    // totalRowsPerVF: 一次 VF 能处理的最大行数（基于 VL 寄存器容量）
    int64_t totalRowsPerVF = static_cast<int64_t>(Std::min(VL_RANGE_CNT, VL_COMPUTE_CNT)) / outW;
    if (totalRowsPerVF < 1)
        totalRowsPerVF = 1;

    // cPerVF: 每次 VF 跨越的完整 C 通道数（对齐到 outH 整数倍，保证批次边界落在 C 通道边界）
    int64_t cPerVF = totalRowsPerVF / outH;
    int64_t actualRowsPerVF;
    GatherIdxType batchDelta; // 批次间索引增量（对有效位置）

    if (cPerVF >= 1) {
        // 跨 C 通道模式：每批处理 cPerVF 个完整 C 通道
        actualRowsPerVF = cPerVF * outH;
        batchDelta = static_cast<GatherIdxType>(cPerVF * static_cast<int64_t>(hwSize));
    } else {
        // 退化模式：outH > totalRowsPerVF，单 C 通道内多行批次
        actualRowsPerVF = totalRowsPerVF;
        batchDelta = static_cast<GatherIdxType>(actualRowsPerVF * width);
    }

    // 4.2 用 VF 生成跨 C 通道的初始索引（两次 Div，一次覆盖所有 actualRowsPerVF 行）
    GenerateCrossChannelWAxisIndices(static_cast<uint32_t>(actualRowsPerVF), static_cast<uint16_t>(outW),
                                     static_cast<uint16_t>(outH), width, hwSize, leftPadW, rightPadW, leftPadH,
                                     midIdxAddr, leftIdxAddr, rightIdxAddr);

    // ===== 5. 循环处理所有 C 通道的所有行 =====
    uint32_t elementsPerBatch = static_cast<uint32_t>(actualRowsPerVF * outW);

    if (cPerVF >= 1) {
        // ===== 5A. 跨 C 通道模式：每批处理 cPerVF 个完整 C 通道 =====
        // batchDelta = cPerVF * hwSize，批次边界恰好对齐在 C 通道边界，偏移正确
        int64_t totalRows = factor * outH;
        int64_t totalBatches = (totalRows + actualRowsPerVF - 1) / actualRowsPerVF;
        uint16_t totalBatchesU = static_cast<uint16_t>(totalBatches);
        int64_t tailRows = totalRows - static_cast<int64_t>(totalBatches - 1) * actualRowsPerVF;
        uint32_t tailElements = static_cast<uint32_t>(tailRows * outW);

        __VEC_SCOPE__
        {
            __local_mem__ PromoteDataT* sregOutAddr = outputAddr;
            __local_mem__ PromoteDataT* sregMiddAddr = midAddr;
            uint32_t tempElementsPerBatch = elementsPerBatch;
            AscendC::Reg::MaskReg idxMask = AscendC::Reg::UpdateMask<GatherIdxType>(elementsPerBatch);

            __local_mem__ GatherIdxType* midIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midIdxAddr);
            __local_mem__ GatherIdxType* leftIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftIdxAddr);
            __local_mem__ GatherIdxType* rightIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightIdxAddr);

            AscendC::Reg::UnalignRegForStore ureg1;

            AscendC::Reg::RegTensor<GatherIdxType> midIdxReg;
            AscendC::Reg::DataCopy(midIdxReg, midIdxAddrU);

            AscendC::Reg::RegTensor<GatherIdxType> leftIdxReg;
            AscendC::Reg::DataCopy(leftIdxReg, leftIdxAddrU);

            AscendC::Reg::RegTensor<GatherIdxType> rightIdxReg;
            AscendC::Reg::DataCopy(rightIdxReg, rightIdxAddrU);

            AscendC::Reg::RegTensor<GatherIdxType> midBatchOffsetReg;
            AscendC::Reg::Duplicate(midBatchOffsetReg, batchDelta, idxMask);

            AscendC::Reg::RegTensor<GatherIdxType> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<GatherIdxType>(0), idxMask);

            AscendC::Reg::RegTensor<GatherIdxType> leftBatchOffsetReg;
            AscendC::Reg::Duplicate(leftBatchOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::MaskReg leftValidMask;
            AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(leftValidMask, leftIdxReg, zeroReg, idxMask);
            AscendC::Reg::Duplicate<GatherIdxType, Reg::MaskMergeMode::MERGING, GatherIdxType>(
                leftBatchOffsetReg, batchDelta, leftValidMask);

            AscendC::Reg::RegTensor<GatherIdxType> rightBatchOffsetReg;
            AscendC::Reg::Duplicate(rightBatchOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::MaskReg rightValidMask;
            AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(rightValidMask, rightIdxReg, zeroReg, idxMask);
            AscendC::Reg::Duplicate<GatherIdxType, Reg::MaskMergeMode::MERGING, GatherIdxType>(
                rightBatchOffsetReg, batchDelta, rightValidMask);

            // 处理完整批次（不含尾块，VEC_SCOPE 内循环变量必须使用 uint16_t）
            for (uint16_t batch = 0; batch < static_cast<uint16_t>(totalBatchesU - 1); batch++) {
                uint32_t maskElementsPerBatch = tempElementsPerBatch;
                AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(maskElementsPerBatch);

                AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
                AscendC::Reg::DataCopyGather(midDataReg, sregMiddAddr, midIdxReg, mask);

                AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
                AscendC::Reg::DataCopyGather(leftDataReg, sregMiddAddr, leftIdxReg, mask);

                AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
                AscendC::Reg::DataCopyGather(rightDataReg, sregMiddAddr, rightIdxReg, mask);

                AscendC::Reg::RegTensor<PromoteDataT> sumReg;
                AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
                AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);
                AscendC::Reg::StoreUnAlign(sregOutAddr, sumReg, ureg1, tempElementsPerBatch);

                AscendC::Reg::Add(midIdxReg, midIdxReg, midBatchOffsetReg, idxMask);
                AscendC::Reg::Add(leftIdxReg, leftIdxReg, leftBatchOffsetReg, idxMask);
                AscendC::Reg::Add(rightIdxReg, rightIdxReg, rightBatchOffsetReg, idxMask);
            }

            // 处理尾块（最后一个批次，行数可能不满 actualRowsPerVF）
            uint32_t tempTailElements = tailElements;
            AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(tailElements);
            AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
            AscendC::Reg::DataCopyGather(midDataReg, sregMiddAddr, midIdxReg, mask);
            AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
            AscendC::Reg::DataCopyGather(leftDataReg, sregMiddAddr, leftIdxReg, mask);
            AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
            AscendC::Reg::DataCopyGather(rightDataReg, sregMiddAddr, rightIdxReg, mask);

            AscendC::Reg::RegTensor<PromoteDataT> sumReg;
            AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
            AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);
            AscendC::Reg::StoreUnAlign(sregOutAddr, sumReg, ureg1, tempTailElements);
            AscendC::Reg::StoreUnAlignPost(sregOutAddr, ureg1, 0);
        }
    } else {
        // ===== 5B. 退化模式：outH > totalRowsPerVF，单 C 通道内分多行批次 =====
        //
        // 【Bug 根因】：不能用统一 batchDelta = actualRowsPerVF * width 做平坦循环！
        // 当批次跨越 C 通道边界时，跨通道步长 = hwSize（含 H pad 行），
        // 而行批次步长 = actualRowsPerVF * width。
        // 当 leftPadH > 0 或 rightPadH > 0 时，hwSize ≠ outH * width，两者不一致。
        //
        // 示例（outH=3, actualRowsPerVF=2, leftPadH=1, height=5, hwSize=5*width）：
        //   平坦 batch1 位置1 = C0-r1-base + 2*width = (leftPadH+3)*width + ZERO_PAD_CNT + leftPadW
        //   正确应为 C1-r0 = 1*hwSize + leftPadH*width + ZERO_PAD_CNT + leftPadW = 6*width + ...
        //   两者不等 → 索引错误 → 精度问题。
        //
        // 【修复】：嵌套循环，外层遍历 C 通道（步长 hwSize），内层遍历行批次（步长 actualRowsPerVF*width）
        // 与 FoldWAxisForCHWAxis 采用相同结构，批次不跨 C 通道边界。

        int64_t loopsPerC = (outH + actualRowsPerVF - 1) / actualRowsPerVF;
        uint16_t loopsPerCU = static_cast<uint16_t>(loopsPerC);
        uint16_t factorU = static_cast<uint16_t>(factor);
        int64_t tailRowsInC = outH - (loopsPerC - 1) * actualRowsPerVF;
        uint32_t tailElementsInC = static_cast<uint32_t>(tailRowsInC * outW);
        GatherIdxType rowBatchDelta = static_cast<GatherIdxType>(actualRowsPerVF * width);
        GatherIdxType cChanDelta = static_cast<GatherIdxType>(hwSize);

        // 保存原始元素计数，UpdateMask 会修改其参数，StoreUnAlign 需要使用未修改的值
        uint32_t storeElemPerBatch = elementsPerBatch;
        uint32_t storeTailElemInC = tailElementsInC;

        __VEC_SCOPE__
        {
            __local_mem__ PromoteDataT* sregMiddAddr = midAddr;
            // UpdateMask 会修改参数，用临时副本传入
            uint32_t idxMaskCount = storeElemPerBatch;
            AscendC::Reg::MaskReg idxMask = AscendC::Reg::UpdateMask<GatherIdxType>(idxMaskCount);

            __local_mem__ GatherIdxType* midIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midIdxAddr);
            __local_mem__ GatherIdxType* leftIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftIdxAddr);
            __local_mem__ GatherIdxType* rightIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightIdxAddr);

            AscendC::Reg::UnalignRegForStore ureg1;

            // 加载 C0 基准索引（GenerateCrossChannelWAxisIndices 生成 C0-r0..r(actualRowsPerVF-1)）
            AscendC::Reg::RegTensor<GatherIdxType> baseMidIdxReg;
            AscendC::Reg::DataCopy(baseMidIdxReg, midIdxAddrU);
            AscendC::Reg::RegTensor<GatherIdxType> baseLeftIdxReg;
            AscendC::Reg::DataCopy(baseLeftIdxReg, leftIdxAddrU);
            AscendC::Reg::RegTensor<GatherIdxType> baseRightIdxReg;
            AscendC::Reg::DataCopy(baseRightIdxReg, rightIdxAddrU);

            AscendC::Reg::RegTensor<GatherIdxType> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<GatherIdxType>(0), idxMask);

            // 行批次偏移（同 C 通道内向下推进 actualRowsPerVF 行）
            AscendC::Reg::RegTensor<GatherIdxType> midRowOffsetReg;
            AscendC::Reg::Duplicate(midRowOffsetReg, rowBatchDelta, idxMask);

            AscendC::Reg::MaskReg leftValidMask;
            AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(leftValidMask, baseLeftIdxReg, zeroReg, idxMask);
            AscendC::Reg::RegTensor<GatherIdxType> leftRowOffsetReg;
            AscendC::Reg::Duplicate(leftRowOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(leftRowOffsetReg, rowBatchDelta, leftValidMask);

            AscendC::Reg::MaskReg rightValidMask;
            AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(rightValidMask, baseRightIdxReg, zeroReg, idxMask);
            AscendC::Reg::RegTensor<GatherIdxType> rightRowOffsetReg;
            AscendC::Reg::Duplicate(rightRowOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(rightRowOffsetReg, rowBatchDelta, rightValidMask);

            // C 通道偏移（步长 hwSize，含 H pad 行；ZERO_PAD 位置保持 0）
            AscendC::Reg::RegTensor<GatherIdxType> midCOffsetReg;
            AscendC::Reg::Duplicate(midCOffsetReg, cChanDelta, idxMask);

            AscendC::Reg::RegTensor<GatherIdxType> leftCOffsetReg;
            AscendC::Reg::Duplicate(leftCOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(leftCOffsetReg, cChanDelta, leftValidMask);

            AscendC::Reg::RegTensor<GatherIdxType> rightCOffsetReg;
            AscendC::Reg::Duplicate(rightCOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(rightCOffsetReg, cChanDelta, rightValidMask);

            // 工作寄存器（行批次循环内使用）
            AscendC::Reg::RegTensor<GatherIdxType> midIdxReg;
            AscendC::Reg::RegTensor<GatherIdxType> leftIdxReg;
            AscendC::Reg::RegTensor<GatherIdxType> rightIdxReg;

            // 外层循环：遍历每个 C 通道
            for (uint16_t c = 0; c < factorU; c++) {
                // 每个 C 通道开始时，从基准复制到工作寄存器
                midIdxReg = baseMidIdxReg;
                leftIdxReg = baseLeftIdxReg;
                rightIdxReg = baseRightIdxReg;

                // 内层循环：处理当前 C 通道内的完整行批次
                for (uint16_t loop = 0; loop < static_cast<uint16_t>(loopsPerCU - 1); loop++) {
                    // UpdateMask 会修改参数，用临时副本传入；StoreUnAlign 用 storeElemPerBatch
                    uint32_t tempLoopElem = storeElemPerBatch;
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(tempLoopElem);

                    AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
                    AscendC::Reg::DataCopyGather(midDataReg, sregMiddAddr, midIdxReg, mask);
                    AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
                    AscendC::Reg::DataCopyGather(leftDataReg, sregMiddAddr, leftIdxReg, mask);
                    AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
                    AscendC::Reg::DataCopyGather(rightDataReg, sregMiddAddr, rightIdxReg, mask);

                    AscendC::Reg::RegTensor<PromoteDataT> sumReg;
                    AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
                    AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);
                    AscendC::Reg::StoreUnAlign(outputAddr, sumReg, ureg1, storeElemPerBatch);

                    // 行批次内推进（同 C 通道，步长 actualRowsPerVF * width）
                    AscendC::Reg::Add(midIdxReg, midIdxReg, midRowOffsetReg, idxMask);
                    AscendC::Reg::Add(leftIdxReg, leftIdxReg, leftRowOffsetReg, idxMask);
                    AscendC::Reg::Add(rightIdxReg, rightIdxReg, rightRowOffsetReg, idxMask);
                }

                // 当前 C 通道的尾块（最后一批行，可能不满 actualRowsPerVF）
                {
                    // UpdateMask 会修改参数，用临时副本传入；StoreUnAlign 用 storeTailElemInC
                    uint32_t tempTailElem = storeTailElemInC;
                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(tempTailElem);
                    AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
                    AscendC::Reg::DataCopyGather(midDataReg, sregMiddAddr, midIdxReg, mask);
                    AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
                    AscendC::Reg::DataCopyGather(leftDataReg, sregMiddAddr, leftIdxReg, mask);
                    AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
                    AscendC::Reg::DataCopyGather(rightDataReg, sregMiddAddr, rightIdxReg, mask);

                    AscendC::Reg::RegTensor<PromoteDataT> sumReg;
                    AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
                    AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);
                    AscendC::Reg::StoreUnAlign(outputAddr, sumReg, ureg1, storeTailElemInC);
                }
                AscendC::Reg::StoreUnAlignPost(outputAddr, ureg1, 0);

                // 更新基准到下一个 C 通道（步长 hwSize，不跨 H pad 行混淆）
                if (c < static_cast<uint16_t>(factorU - 1)) {
                    AscendC::Reg::Add(baseMidIdxReg, baseMidIdxReg, midCOffsetReg, idxMask);
                    AscendC::Reg::Add(baseLeftIdxReg, baseLeftIdxReg, leftCOffsetReg, idxMask);
                    AscendC::Reg::Add(baseRightIdxReg, baseRightIdxReg, rightCOffsetReg, idxMask);
                }
            }
        }
    }
}

// ========== FoldWAxisForCHWAxis 辅助函数 ==========
/**
 * @brief 折叠 W 轴并输出（用于 FoldCHWAxis）
 *
 * 对每个有效通道（factor * outC 个），折叠 W 轴并输出到 outputAddr。
 * N×C 批处理：
 * - 将 N 和 C 展平为一维 ncTotal = factor * outC，以 elemPerNC = outH * outW 为单位分批
 * - ncPerVF = VL_RANGE_CNT / elemPerNC，一次 VF 同时处理多个 (n,c)
 * - 首批：用 GenerateWAxisIndices 生成 (n=0,c=0) 的行模板，
 *         再用向量分解 x → (ncInBatch, row, col) 扩展为多个 (n,c) 的完整索引
 * - 后续批：索引整体递增 ncPerVF * ncStride（mid 恒增，left/right ZERO_PAD 位保持 0）
 * - 当 elemPerNC > VL_RANGE_CNT 时回退：nc 标量循环 + 行批次 VF（保留原逻辑）
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldWAxisForCHWAxis(
    int64_t factor, int64_t outC, uint64_t hwSize, int64_t width, int64_t leftPadC, int64_t leftPadH, int64_t leftPadW,
    int64_t rightPadW, int64_t outH, int64_t outW, __local_mem__ PromoteDataT* midAddr,
    __local_mem__ PromoteDataT* outputAddr, __local_mem__ GatherRangeType* midIdxAddr,
    __local_mem__ GatherRangeType* leftIdxAddr, __local_mem__ GatherRangeType* rightIdxAddr)
{
    // 获取偏移缓冲区
    LocalTensor<GatherRangeType> leftOffsetLocal = leftPadOffset_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> midOffsetLocal = midOffset_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> rightOffsetLocal = rightPadOffset_.Get<GatherRangeType>();

    __local_mem__ GatherRangeType* leftOffsetAddr = (__local_mem__ GatherRangeType*)leftOffsetLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* midOffsetAddr = (__local_mem__ GatherRangeType*)midOffsetLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* rightOffsetAddr = (__local_mem__ GatherRangeType*)rightOffsetLocal.GetPhyAddr();

    __local_mem__ GatherIdxType* midOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midOffsetAddr);
    __local_mem__ GatherIdxType* leftOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftOffsetAddr);
    __local_mem__ GatherIdxType* rightOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightOffsetAddr);

    // 每个 (n,c) 对应的输出元素数
    int64_t elemPerNC = outH * outW;
    // n 轴步长：整个 C 维度（含 pad）的元素数
    int64_t nStride = static_cast<int64_t>(leftPadC + outC + rightPad_[dimNum_ - 3]) * static_cast<int64_t>(hwSize);

    // ===== 优化路径：N×C 批处理（elemPerNC <= VL_RANGE_CNT）=====
    // 索引公式（c 为输出 C 通道号 0~outC-1，对应输入第 leftPadC+c 个通道）：
    //   midIdx   = n*nStride + c*hwSize + baseOffset + row*width + col
    //   baseOffset = leftPadC*hwSize + leftPadH*width + ZERO_PAD_CNT + leftPadW
    //   leftIdx  = midIdx - 2*col         (reflect, 0 < col <= leftPadW)
    //            = midIdx - 2*col - 1     (symmetric, col < leftPadW)
    //            = 0                      (invalid)
    //   rightIdx = midIdx + 2*(outW-1-col)     (reflect, outW-1-rightPadW <= col < outW-1)
    //            = midIdx + 2*(outW-1-col) + 1 (symmetric, col >= outW-rightPadW)
    //            = 0                           (invalid)
    if (elemPerNC <= static_cast<int64_t>(VL_RANGE_CNT)) {
        int64_t ncPerVF = static_cast<int64_t>(VL_RANGE_CNT) / elemPerNC;
        if (ncPerVF < 1)
            ncPerVF = 1;

        int64_t ncTotal = factor * outC;
        int64_t curNC = Std::min<int64_t>(ncPerVF, ncTotal);
        int64_t totalElem = curNC * elemPerNC;

        // baseOffset：(n=0,c=0,row=0,col=0) 时 midIdx 中去掉 col 后的常量部分
        // c 是输出 C 通道号（0~outC-1），对应输入中第 leftPadC+c 个通道，
        // 所以需要加上 leftPadC*hwSize 作为 C 轴基础偏移
        GatherRangeType baseOffset = static_cast<GatherRangeType>(
            static_cast<int64_t>(leftPadC) * static_cast<int64_t>(hwSize) + static_cast<int64_t>(leftPadH) * width +
            ZERO_PAD_CNT + leftPadW);

        GatherRangeType nStrideR = static_cast<GatherRangeType>(nStride);
        GatherRangeType hwSizeR = static_cast<GatherRangeType>(hwSize);
        GatherRangeType widthR = static_cast<GatherRangeType>(width);

        GatherIdxType midStep = 0;
        int64_t cInc = 0;
        if (ncTotal > ncPerVF) {
            int64_t nInc = ncPerVF / outC;
            cInc = ncPerVF % outC;
            midStep = static_cast<GatherIdxType>(nInc * nStride + cInc * static_cast<int64_t>(hwSize));
        }

        __VEC_SCOPE__
        {
            __local_mem__ PromoteDataT* sregOutputAddr = outputAddr;

            AscendC::Reg::UnalignRegForStore ureg1;

            uint32_t vLen = static_cast<uint32_t>(totalElem);
            uint32_t tmpLen = vLen;
            uint32_t maskLen = vLen;

            // ===== 首批：全用 GatherRangeType 计算索引 =====
            AscendC::Reg::MaskReg fullMask = AscendC::Reg::UpdateMask<GatherRangeType>(vLen);

            // x → (ncInBatch, row, col) 三级分解
            AscendC::Reg::RegTensor<GatherRangeType> xReg;
            AscendC::Reg::Arange(xReg, static_cast<GatherRangeType>(0));

            AscendC::Reg::RegTensor<GatherRangeType> ncInBatchReg;
            AscendC::Reg::RegTensor<GatherRangeType> xRemReg;
            AscendC::Reg::RegTensor<GatherRangeType> ePerNCReg;
            AscendC::Reg::RegTensor<GatherRangeType> tmpReg;

            AscendC::Reg::Duplicate(ePerNCReg, static_cast<GatherRangeType>(elemPerNC), fullMask);
            AscendC::Reg::Div(ncInBatchReg, xReg, ePerNCReg, fullMask);
            AscendC::Reg::Muls(tmpReg, ncInBatchReg, static_cast<GatherRangeType>(elemPerNC), fullMask);
            AscendC::Reg::Sub(xRemReg, xReg, tmpReg, fullMask);

            AscendC::Reg::RegTensor<GatherRangeType> rowReg;
            AscendC::Reg::RegTensor<GatherRangeType> colReg;
            AscendC::Reg::RegTensor<GatherRangeType> outWReg;

            AscendC::Reg::Duplicate(outWReg, static_cast<GatherRangeType>(outW), fullMask);
            AscendC::Reg::Div(rowReg, xRemReg, outWReg, fullMask);
            AscendC::Reg::Muls(tmpReg, rowReg, static_cast<GatherRangeType>(outW), fullMask);
            AscendC::Reg::Sub(colReg, xRemReg, tmpReg, fullMask);

            // ncOffset = (ncInBatch/outC)*nStride + (ncInBatch%outC)*hwSize
            AscendC::Reg::RegTensor<GatherRangeType> nIdxReg;
            AscendC::Reg::RegTensor<GatherRangeType> cIdxReg;
            AscendC::Reg::RegTensor<GatherRangeType> outCReg;
            AscendC::Reg::RegTensor<GatherRangeType> ncOffsetReg;

            AscendC::Reg::Duplicate(outCReg, static_cast<GatherRangeType>(outC), fullMask);
            AscendC::Reg::Div(nIdxReg, ncInBatchReg, outCReg, fullMask);
            AscendC::Reg::Muls(tmpReg, nIdxReg, static_cast<GatherRangeType>(outC), fullMask);
            AscendC::Reg::Sub(cIdxReg, ncInBatchReg, tmpReg, fullMask);

            AscendC::Reg::RegTensor<GatherRangeType> nPartReg;
            AscendC::Reg::RegTensor<GatherRangeType> cPartReg;

            AscendC::Reg::Muls(nPartReg, nIdxReg, nStrideR, fullMask);
            AscendC::Reg::Muls(cPartReg, cIdxReg, hwSizeR, fullMask);
            AscendC::Reg::Add(ncOffsetReg, nPartReg, cPartReg, fullMask);

            // row 偏移
            AscendC::Reg::RegTensor<GatherRangeType> rowOffsetReg;

            AscendC::Reg::Muls(rowOffsetReg, rowReg, widthR, fullMask);

            // midIdx = ncOffset + baseOffset + rowOffset + col
            AscendC::Reg::RegTensor<GatherRangeType> midIdxR;

            AscendC::Reg::Duplicate(midIdxR, baseOffset, fullMask);
            AscendC::Reg::Add(midIdxR, midIdxR, ncOffsetReg, fullMask);
            AscendC::Reg::Add(midIdxR, midIdxR, rowOffsetReg, fullMask);
            AscendC::Reg::Add(midIdxR, midIdxR, colReg, fullMask);

            AscendC::Reg::RegTensor<GatherRangeType> zeroFull;

            AscendC::Reg::Duplicate(zeroFull, static_cast<GatherRangeType>(0), fullMask);

            // ===== leftIdx =====
            AscendC::Reg::RegTensor<GatherRangeType> leftIdxR;
            AscendC::Reg::RegTensor<GatherRangeType> col2Reg;

            AscendC::Reg::Muls(col2Reg, colReg, static_cast<GatherRangeType>(2), fullMask);

            if (leftPadW > 0) {
                AscendC::Reg::RegTensor<GatherRangeType> leftPadWFull;

                AscendC::Reg::Duplicate(leftPadWFull, static_cast<GatherRangeType>(leftPadW), fullMask);

                if constexpr (IS_REFLECT) {
                    // leftIdx = midIdx - 2*col，valid: 0 < col <= leftPadW
                    AscendC::Reg::Sub(leftIdxR, midIdxR, col2Reg, fullMask);
                    // col == 0 → 置 0
                    AscendC::Reg::MaskReg eqZeroMask;
                    AscendC::Reg::Compare<GatherRangeType, CMPMODE::EQ>(eqZeroMask, colReg, zeroFull, fullMask);
                    AscendC::Reg::Select(leftIdxR, zeroFull, leftIdxR, eqZeroMask);
                    // col > leftPadW → 置 0
                    AscendC::Reg::MaskReg gtLPWMask;
                    AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(gtLPWMask, colReg, leftPadWFull, fullMask);
                    AscendC::Reg::Select(leftIdxR, zeroFull, leftIdxR, gtLPWMask);
                } else {
                    // leftIdx = midIdx - 2*col - 1，valid: col < leftPadW
                    AscendC::Reg::Sub(leftIdxR, midIdxR, colReg, fullMask);
                    AscendC::Reg::Adds(leftIdxR, leftIdxR, static_cast<GatherRangeType>(-1), fullMask);
                    // col >= leftPadW → 置 0
                    AscendC::Reg::MaskReg geLPWMask;
                    AscendC::Reg::Compare<GatherRangeType, CMPMODE::GE>(geLPWMask, colReg, leftPadWFull, fullMask);
                    AscendC::Reg::Select(leftIdxR, zeroFull, leftIdxR, geLPWMask);
                }
            } else {
                AscendC::Reg::Duplicate(leftIdxR, static_cast<GatherRangeType>(0), fullMask);
            }

            // ===== rightIdx =====
            AscendC::Reg::RegTensor<GatherRangeType> rightIdxR;
            if (rightPadW > 0) {
                // 公共部分：outW-1-col
                AscendC::Reg::RegTensor<GatherRangeType> outW1Reg;
                AscendC::Reg::RegTensor<GatherRangeType> outW1ColReg;
                AscendC::Reg::RegTensor<GatherRangeType> rightBase2Reg;

                AscendC::Reg::Duplicate(outW1Reg, static_cast<GatherRangeType>(outW - 1), fullMask);
                AscendC::Reg::Sub(outW1ColReg, outW1Reg, colReg, fullMask);
                AscendC::Reg::Muls(rightBase2Reg, outW1ColReg, static_cast<GatherRangeType>(2), fullMask);
                // rightIdx = midIdx + 2*(outW-1-col)（两种模式共同部分）
                AscendC::Reg::Add(rightIdxR, midIdxR, rightBase2Reg, fullMask);

                if constexpr (IS_REFLECT) {
                    // valid: outW-1-rightPadW <= col < outW-1
                    AscendC::Reg::RegTensor<GatherRangeType> threshLow, threshHigh;
                    AscendC::Reg::Duplicate(threshLow, static_cast<GatherRangeType>(outW - 1 - rightPadW), fullMask);
                    AscendC::Reg::Duplicate(threshHigh, static_cast<GatherRangeType>(outW - 2), fullMask);
                    // col < threshLow → 置 0
                    AscendC::Reg::MaskReg ltMask;
                    AscendC::Reg::Compare<GatherRangeType, CMPMODE::LT>(ltMask, colReg, threshLow, fullMask);
                    AscendC::Reg::Select(rightIdxR, zeroFull, rightIdxR, ltMask);
                    // col > threshHigh（即 col >= outW-1）→ 置 0
                    AscendC::Reg::MaskReg gtMask;
                    AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(gtMask, colReg, threshHigh, fullMask);
                    AscendC::Reg::Select(rightIdxR, zeroFull, rightIdxR, gtMask);
                } else {
                    // symmetric: +1，valid: col >= outW-rightPadW
                    AscendC::Reg::Adds(rightIdxR, rightIdxR, static_cast<GatherRangeType>(1), fullMask);
                    AscendC::Reg::RegTensor<GatherRangeType> threshR;
                    AscendC::Reg::Duplicate(threshR, static_cast<GatherRangeType>(outW - rightPadW), fullMask);
                    // col < threshR → 置 0
                    AscendC::Reg::MaskReg ltMask;
                    AscendC::Reg::Compare<GatherRangeType, CMPMODE::LT>(ltMask, colReg, threshR, fullMask);
                    AscendC::Reg::Select(rightIdxR, zeroFull, rightIdxR, ltMask);
                }
            } else {
                AscendC::Reg::Duplicate(rightIdxR, static_cast<GatherRangeType>(0), fullMask);
            }

            // 存到 GatherRangeType* 内存缓冲区
            AscendC::Reg::DataCopy(midIdxAddr, midIdxR, fullMask);
            AscendC::Reg::DataCopy(leftIdxAddr, leftIdxR, fullMask);
            AscendC::Reg::DataCopy(rightIdxAddr, rightIdxR, fullMask);

            // reinterpret 为 GatherIdxType*，加载为 GatherIdxType 寄存器后做数据 Gather
            __local_mem__ GatherIdxType* midIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midIdxAddr);
            __local_mem__ GatherIdxType* leftIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftIdxAddr);
            __local_mem__ GatherIdxType* rightIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightIdxAddr);

            AscendC::Reg::RegTensor<GatherIdxType> midIdxFull;
            AscendC::Reg::RegTensor<GatherIdxType> leftIdxFull;
            AscendC::Reg::RegTensor<GatherIdxType> rightIdxFull;

            AscendC::Reg::DataCopy(midIdxFull, midIdxAddrU);
            AscendC::Reg::DataCopy(leftIdxFull, leftIdxAddrU);
            AscendC::Reg::DataCopy(rightIdxFull, rightIdxAddrU);

            AscendC::Reg::MaskReg dataMask = AscendC::Reg::UpdateMask<PromoteDataT>(maskLen);

            AscendC::Reg::RegTensor<PromoteDataT> midD;
            AscendC::Reg::RegTensor<PromoteDataT> leftD;
            AscendC::Reg::RegTensor<PromoteDataT> rightD;
            AscendC::Reg::RegTensor<PromoteDataT> sumD;

            AscendC::Reg::DataCopyGather(midD, midAddr, midIdxFull, dataMask);
            AscendC::Reg::DataCopyGather(leftD, midAddr, leftIdxFull, dataMask);
            AscendC::Reg::DataCopyGather(rightD, midAddr, rightIdxFull, dataMask);

            AscendC::Reg::Add(sumD, midD, leftD, dataMask);
            AscendC::Reg::Add(sumD, sumD, rightD, dataMask);

            AscendC::Reg::StoreUnAlign(sregOutputAddr, sumD, ureg1, tmpLen);
            if (ncTotal <= ncPerVF) {
                AscendC::Reg::StoreUnAlignPost(sregOutputAddr, ureg1, 0);
            }

            // ===== 后续批次：cInc==0 用增量更新，cInc!=0 用完整重计算 =====
            if (ncTotal > ncPerVF) {
                int64_t ncBase = ncPerVF;
                if (cInc == 0) {
                    // cInc==0：每批次 n 均匀递增，增量更新索引即可
                    AscendC::Reg::RegTensor<GatherIdxType> curMidIdx;
                    curMidIdx = midIdxFull;

                    AscendC::Reg::RegTensor<GatherIdxType> curLeftIdx;
                    curLeftIdx = leftIdxFull;

                    AscendC::Reg::RegTensor<GatherIdxType> curRightIdx;
                    curRightIdx = rightIdxFull;

                    uint32_t fullVLen = static_cast<uint32_t>(curNC * elemPerNC);
                    uint32_t idxMaskLen = fullVLen;

                    AscendC::Reg::MaskReg fullMaskInc = AscendC::Reg::UpdateMask<GatherIdxType>(idxMaskLen);

                    AscendC::Reg::RegTensor<GatherIdxType> zeroFull2;
                    AscendC::Reg::Duplicate(zeroFull2, static_cast<GatherIdxType>(0), fullMaskInc);

                    AscendC::Reg::MaskReg lValid2;
                    AscendC::Reg::MaskReg rValid2;

                    AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(lValid2, curLeftIdx, zeroFull2, fullMaskInc);
                    AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(rValid2, curRightIdx, zeroFull2, fullMaskInc);

                    AscendC::Reg::RegTensor<GatherIdxType> midStepReg;
                    AscendC::Reg::RegTensor<GatherIdxType> leftStepReg;
                    AscendC::Reg::RegTensor<GatherIdxType> rightStepReg;

                    uint32_t sregStep = midStep;

                    AscendC::Reg::Duplicate(midStepReg, sregStep, fullMaskInc);
                    AscendC::Reg::Duplicate(leftStepReg, static_cast<GatherIdxType>(0), fullMaskInc);
                    AscendC::Reg::Duplicate(rightStepReg, static_cast<GatherIdxType>(0), fullMaskInc);

                    AscendC::Reg::Duplicate<GatherIdxType, Reg::MaskMergeMode::ZEROING, GatherIdxType>(
                        leftStepReg, sregStep, lValid2);
                    AscendC::Reg::Duplicate<GatherIdxType, Reg::MaskMergeMode::ZEROING, GatherIdxType>(
                        rightStepReg, sregStep, rValid2);

                    while (ncBase < ncTotal) {
                        int64_t curNC2 = Std::min<int64_t>(ncPerVF, ncTotal - ncBase);
                        int64_t totalElem2 = curNC2 * elemPerNC;
                        uint32_t vLen2 = static_cast<uint32_t>(totalElem2);
                        uint32_t batchMaskLen = vLen2;
                        uint32_t tmpLen2 = vLen2;

                        AscendC::Reg::MaskReg batchMask = AscendC::Reg::UpdateMask<GatherIdxType>(batchMaskLen);
                        AscendC::Reg::MaskReg dataMask2 = AscendC::Reg::UpdateMask<PromoteDataT>(vLen2);

                        AscendC::Reg::Add(curMidIdx, curMidIdx, midStepReg, batchMask);
                        AscendC::Reg::Add(curLeftIdx, curLeftIdx, leftStepReg, lValid2);
                        AscendC::Reg::Add(curRightIdx, curRightIdx, rightStepReg, rValid2);

                        AscendC::Reg::RegTensor<PromoteDataT> midD2;
                        AscendC::Reg::RegTensor<PromoteDataT> leftD2;
                        AscendC::Reg::RegTensor<PromoteDataT> rightD2;
                        AscendC::Reg::RegTensor<PromoteDataT> sumD2;

                        AscendC::Reg::DataCopyGather(midD2, midAddr, curMidIdx, dataMask2);
                        AscendC::Reg::DataCopyGather(leftD2, midAddr, curLeftIdx, dataMask2);
                        AscendC::Reg::DataCopyGather(rightD2, midAddr, curRightIdx, dataMask2);

                        AscendC::Reg::Add(sumD2, midD2, leftD2, dataMask2);
                        AscendC::Reg::Add(sumD2, sumD2, rightD2, dataMask2);

                        AscendC::Reg::StoreUnAlign(sregOutputAddr, sumD2, ureg1, tmpLen2);
                        if (ncBase + ncPerVF >= ncTotal) {
                            AscendC::Reg::StoreUnAlignPost(sregOutputAddr, ureg1, 0);
                        }
                        ncBase += ncPerVF;
                    }
                } else {
                    // cInc!=0：批次跨 N 轴边界，每批次完整重计算索引
                    while (ncBase < ncTotal) {
                        int64_t curNC2 = Std::min<int64_t>(ncPerVF, ncTotal - ncBase);
                        int64_t totalElem2 = curNC2 * elemPerNC;
                        uint32_t vLen2 = static_cast<uint32_t>(totalElem2);
                        uint32_t tempVLen21 = vLen2;
                        uint32_t tempVLen22 = vLen2;

                        AscendC::Reg::MaskReg fullMask2 = AscendC::Reg::UpdateMask<GatherRangeType>(tempVLen21);
                        AscendC::Reg::MaskReg dataMask2 = AscendC::Reg::UpdateMask<PromoteDataT>(tempVLen22);

                        // 重新计算当前批次的索引（x → ncInBatch → n,c → row,col → midIdx/leftIdx/rightIdx）
                        AscendC::Reg::RegTensor<GatherRangeType> xReg2;
                        AscendC::Reg::Arange(xReg2, static_cast<GatherRangeType>(0));

                        // ncInBatch = x / elemPerNC（当前批次内的 nc 索引）
                        AscendC::Reg::RegTensor<GatherRangeType> ncInBatchReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> xRemReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> ePerNCReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> tmpReg2;

                        AscendC::Reg::Duplicate(ePerNCReg2, static_cast<GatherRangeType>(elemPerNC), fullMask2);
                        AscendC::Reg::Div(ncInBatchReg2, xReg2, ePerNCReg2, fullMask2);
                        AscendC::Reg::Muls(tmpReg2, ncInBatchReg2, static_cast<GatherRangeType>(elemPerNC), fullMask2);
                        AscendC::Reg::Sub(xRemReg2, xReg2, tmpReg2, fullMask2);

                        // 全局 nc 索引 = ncBase + ncInBatch
                        AscendC::Reg::RegTensor<GatherRangeType> ncGlobalReg;
                        AscendC::Reg::Adds(ncGlobalReg, ncInBatchReg2, static_cast<GatherRangeType>(ncBase), fullMask2);

                        // row = xRem / outW, col = xRem % outW
                        AscendC::Reg::RegTensor<GatherRangeType> rowReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> colReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> outWReg2;

                        AscendC::Reg::Duplicate(outWReg2, static_cast<GatherRangeType>(outW), fullMask2);
                        AscendC::Reg::Div(rowReg2, xRemReg2, outWReg2, fullMask2);
                        AscendC::Reg::Muls(tmpReg2, rowReg2, static_cast<GatherRangeType>(outW), fullMask2);
                        AscendC::Reg::Sub(colReg2, xRemReg2, tmpReg2, fullMask2);

                        // n = ncGlobal / outC, c = ncGlobal % outC
                        AscendC::Reg::RegTensor<GatherRangeType> nIdxReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> cIdxReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> outCReg2;

                        AscendC::Reg::Duplicate(outCReg2, static_cast<GatherRangeType>(outC), fullMask2);
                        AscendC::Reg::Div(nIdxReg2, ncGlobalReg, outCReg2, fullMask2);
                        AscendC::Reg::Muls(tmpReg2, nIdxReg2, static_cast<GatherRangeType>(outC), fullMask2);
                        AscendC::Reg::Sub(cIdxReg2, ncGlobalReg, tmpReg2, fullMask2);

                        // ncOffset = n*nStride + c*hwSize
                        AscendC::Reg::RegTensor<GatherRangeType> nPartReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> cPartReg2;
                        AscendC::Reg::RegTensor<GatherRangeType> ncOffsetReg2;

                        AscendC::Reg::Muls(nPartReg2, nIdxReg2, nStrideR, fullMask2);
                        AscendC::Reg::Muls(cPartReg2, cIdxReg2, hwSizeR, fullMask2);
                        AscendC::Reg::Add(ncOffsetReg2, nPartReg2, cPartReg2, fullMask2);

                        // rowOffset = row * width
                        AscendC::Reg::RegTensor<GatherRangeType> rowOffsetReg2;
                        AscendC::Reg::Muls(rowOffsetReg2, rowReg2, widthR, fullMask2);

                        // midIdx = ncOffset + baseOffset + rowOffset + col
                        AscendC::Reg::RegTensor<GatherRangeType> midIdxR2;
                        AscendC::Reg::Duplicate(midIdxR2, baseOffset, fullMask2);
                        AscendC::Reg::Add(midIdxR2, midIdxR2, ncOffsetReg2, fullMask2);
                        AscendC::Reg::Add(midIdxR2, midIdxR2, rowOffsetReg2, fullMask2);
                        AscendC::Reg::Add(midIdxR2, midIdxR2, colReg2, fullMask2);

                        AscendC::Reg::RegTensor<GatherRangeType> zeroFull2;
                        AscendC::Reg::Duplicate(zeroFull2, static_cast<GatherRangeType>(0), fullMask2);

                        // 计算 leftIdx 和 rightIdx（与首批逻辑相同）
                        AscendC::Reg::RegTensor<GatherRangeType> leftIdxR2;
                        AscendC::Reg::RegTensor<GatherRangeType> rightIdxR2;
                        AscendC::Reg::RegTensor<GatherRangeType> col2Reg2;

                        AscendC::Reg::Muls(col2Reg2, colReg2, static_cast<GatherRangeType>(2), fullMask2);

                        if (leftPadW > 0) {
                            AscendC::Reg::RegTensor<GatherRangeType> leftPadWFull2;
                            AscendC::Reg::Duplicate(leftPadWFull2, static_cast<GatherRangeType>(leftPadW), fullMask2);
                            if constexpr (IS_REFLECT) {
                                AscendC::Reg::Sub(leftIdxR2, midIdxR2, col2Reg2, fullMask2);
                                AscendC::Reg::MaskReg eqZeroMask2;
                                AscendC::Reg::Compare<GatherRangeType, CMPMODE::EQ>(eqZeroMask2, colReg2, zeroFull2,
                                                                                    fullMask2);
                                AscendC::Reg::Select(leftIdxR2, zeroFull2, leftIdxR2, eqZeroMask2);
                                AscendC::Reg::MaskReg gtLPWMask2;
                                AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(gtLPWMask2, colReg2, leftPadWFull2,
                                                                                    fullMask2);
                                AscendC::Reg::Select(leftIdxR2, zeroFull2, leftIdxR2, gtLPWMask2);
                            } else {
                                AscendC::Reg::Sub(leftIdxR2, midIdxR2, col2Reg2, fullMask2);
                                AscendC::Reg::Adds(leftIdxR2, leftIdxR2, static_cast<GatherRangeType>(-1), fullMask2);
                                AscendC::Reg::MaskReg geLPWMask2;
                                AscendC::Reg::Compare<GatherRangeType, CMPMODE::GE>(geLPWMask2, colReg2, leftPadWFull2,
                                                                                    fullMask2);
                                AscendC::Reg::Select(leftIdxR2, zeroFull2, leftIdxR2, geLPWMask2);
                            }
                        } else {
                            AscendC::Reg::Duplicate(leftIdxR2, static_cast<GatherRangeType>(0), fullMask2);
                        }

                        if (rightPadW > 0) {
                            AscendC::Reg::RegTensor<GatherRangeType> outW1Reg2;
                            AscendC::Reg::RegTensor<GatherRangeType> outW1ColReg2;
                            AscendC::Reg::RegTensor<GatherRangeType> rightBase2Reg2;

                            AscendC::Reg::Duplicate(outW1Reg2, static_cast<GatherRangeType>(outW - 1), fullMask2);
                            AscendC::Reg::Sub(outW1ColReg2, outW1Reg2, colReg2, fullMask2);
                            AscendC::Reg::Muls(rightBase2Reg2, outW1ColReg2, static_cast<GatherRangeType>(2),
                                               fullMask2);
                            AscendC::Reg::Add(rightIdxR2, midIdxR2, rightBase2Reg2, fullMask2);

                            if constexpr (IS_REFLECT) {
                                AscendC::Reg::RegTensor<GatherRangeType> threshLow2;
                                AscendC::Reg::RegTensor<GatherRangeType> threshHigh2;

                                AscendC::Reg::Duplicate(threshLow2, static_cast<GatherRangeType>(outW - 1 - rightPadW),
                                                        fullMask2);
                                AscendC::Reg::Duplicate(threshHigh2, static_cast<GatherRangeType>(outW - 2), fullMask2);
                                AscendC::Reg::MaskReg ltMask2;
                                AscendC::Reg::Compare<GatherRangeType, CMPMODE::LT>(ltMask2, colReg2, threshLow2,
                                                                                    fullMask2);
                                AscendC::Reg::Select(rightIdxR2, zeroFull2, rightIdxR2, ltMask2);
                                AscendC::Reg::MaskReg gtMask2;
                                AscendC::Reg::Compare<GatherRangeType, CMPMODE::GT>(gtMask2, colReg2, threshHigh2,
                                                                                    fullMask2);
                                AscendC::Reg::Select(rightIdxR2, zeroFull2, rightIdxR2, gtMask2);
                            } else {
                                AscendC::Reg::Adds(rightIdxR2, rightIdxR2, static_cast<GatherRangeType>(1), fullMask2);
                                AscendC::Reg::RegTensor<GatherRangeType> threshR2;
                                AscendC::Reg::Duplicate(threshR2, static_cast<GatherRangeType>(outW - rightPadW),
                                                        fullMask2);
                                AscendC::Reg::MaskReg ltMask2;
                                AscendC::Reg::Compare<GatherRangeType, CMPMODE::LT>(ltMask2, colReg2, threshR2,
                                                                                    fullMask2);
                                AscendC::Reg::Select(rightIdxR2, zeroFull2, rightIdxR2, ltMask2);
                            }
                        } else {
                            AscendC::Reg::Duplicate(rightIdxR2, static_cast<GatherRangeType>(0), fullMask2);
                        }

                        // 转换为 GatherIdxType 并执行 Gather
                        AscendC::Reg::DataCopy(midIdxAddr, midIdxR2, fullMask2);
                        AscendC::Reg::DataCopy(leftIdxAddr, leftIdxR2, fullMask2);
                        AscendC::Reg::DataCopy(rightIdxAddr, rightIdxR2, fullMask2);

                        __local_mem__ GatherIdxType* midIdxAddrU2 = reinterpret_cast<__local_mem__ GatherIdxType*>(
                            midIdxAddr);
                        __local_mem__ GatherIdxType* leftIdxAddrU2 = reinterpret_cast<__local_mem__ GatherIdxType*>(
                            leftIdxAddr);
                        __local_mem__ GatherIdxType* rightIdxAddrU2 = reinterpret_cast<__local_mem__ GatherIdxType*>(
                            rightIdxAddr);

                        AscendC::Reg::RegTensor<GatherIdxType> midIdxFull2;
                        AscendC::Reg::RegTensor<GatherIdxType> leftIdxFull2;
                        AscendC::Reg::RegTensor<GatherIdxType> rightIdxFull2;

                        AscendC::Reg::DataCopy(midIdxFull2, midIdxAddrU2);
                        AscendC::Reg::DataCopy(leftIdxFull2, leftIdxAddrU2);
                        AscendC::Reg::DataCopy(rightIdxFull2, rightIdxAddrU2);

                        AscendC::Reg::RegTensor<PromoteDataT> midD2;
                        AscendC::Reg::RegTensor<PromoteDataT> leftD2;
                        AscendC::Reg::RegTensor<PromoteDataT> rightD2;
                        AscendC::Reg::RegTensor<PromoteDataT> sumD2;

                        AscendC::Reg::DataCopyGather(midD2, midAddr, midIdxFull2, dataMask2);
                        AscendC::Reg::DataCopyGather(leftD2, midAddr, leftIdxFull2, dataMask2);
                        AscendC::Reg::DataCopyGather(rightD2, midAddr, rightIdxFull2, dataMask2);

                        AscendC::Reg::Add(sumD2, midD2, leftD2, dataMask2);
                        AscendC::Reg::Add(sumD2, sumD2, rightD2, dataMask2);

                        AscendC::Reg::StoreUnAlign(sregOutputAddr, sumD2, ureg1, static_cast<uint32_t>(vLen2));
                        if (ncBase + ncPerVF >= ncTotal) {
                            AscendC::Reg::StoreUnAlignPost(sregOutputAddr, ureg1, 0);
                        }
                        ncBase += ncPerVF;
                    }
                } // end else (cInc != 0)
            } // end if (ncTotal > ncPerVF)
        }
    } else {
        // ===== 回退路径：elemPerNC > VL_RANGE_CNT，行批次 VF + nc 标量循环 =====
        int64_t rowsPerBatch = static_cast<int64_t>(VL_RANGE_CNT) / outW;
        if (rowsPerBatch < 1)
            rowsPerBatch = 1;

        uint32_t elementsPerBatch = static_cast<uint32_t>(rowsPerBatch * outW);
        GatherRangeType batchOffset = static_cast<GatherRangeType>(rowsPerBatch * width);
        int64_t totalLoops = (outH + rowsPerBatch - 1) / rowsPerBatch;

        uint64_t cOffset0 = static_cast<uint64_t>(leftPadC) * hwSize;
        GenerateWAxisIndices(cOffset0, static_cast<uint16_t>(rowsPerBatch), static_cast<uint16_t>(outW),
                             static_cast<uint16_t>(width), leftPadW, rightPadW, leftPadH, batchOffset, midIdxAddr,
                             leftIdxAddr, rightIdxAddr, midOffsetAddr, leftOffsetAddr, rightOffsetAddr);

        __VEC_SCOPE__
        {
            AscendC::Reg::UnalignRegForStore ureg1;
            uint32_t maskLen = elementsPerBatch;
            AscendC::Reg::MaskReg idxMask = AscendC::Reg::UpdateMask<GatherIdxType>(maskLen);

            __local_mem__ GatherIdxType* midIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midIdxAddr);
            __local_mem__ GatherIdxType* leftIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(leftIdxAddr);
            __local_mem__ GatherIdxType* rightIdxAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(rightIdxAddr);

            __local_mem__ GatherIdxType* midOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(midOffsetAddr);
            __local_mem__ GatherIdxType* leftOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(
                leftOffsetAddr);
            __local_mem__ GatherIdxType* rightOffsetAddrU = reinterpret_cast<__local_mem__ GatherIdxType*>(
                rightOffsetAddr);

            AscendC::Reg::RegTensor<GatherIdxType> baseMidIdxReg;
            AscendC::Reg::RegTensor<GatherIdxType> baseLeftIdxReg;
            AscendC::Reg::RegTensor<GatherIdxType> baseRightIdxReg;

            AscendC::Reg::DataCopy(baseMidIdxReg, midIdxAddrU);
            AscendC::Reg::DataCopy(baseLeftIdxReg, leftIdxAddrU);
            AscendC::Reg::DataCopy(baseRightIdxReg, rightIdxAddrU);

            AscendC::Reg::RegTensor<GatherIdxType> midOffsetReg;
            AscendC::Reg::RegTensor<GatherIdxType> leftOffsetReg;
            AscendC::Reg::RegTensor<GatherIdxType> rightOffsetReg;

            AscendC::Reg::DataCopy(midOffsetReg, midOffsetAddrU);
            AscendC::Reg::DataCopy(leftOffsetReg, leftOffsetAddrU);
            AscendC::Reg::DataCopy(rightOffsetReg, rightOffsetAddrU);

            GatherIdxType hwSizeU = static_cast<GatherIdxType>(hwSize);
            GatherIdxType nStride_ = static_cast<GatherIdxType>(nStride);

            AscendC::Reg::RegTensor<GatherIdxType> zeroReg;
            AscendC::Reg::Duplicate(zeroReg, static_cast<GatherIdxType>(0), idxMask);

            AscendC::Reg::RegTensor<GatherIdxType> midCChanOffsetReg;
            AscendC::Reg::Duplicate(midCChanOffsetReg, hwSizeU, idxMask);

            AscendC::Reg::MaskReg leftValidMask, rightValidMask;
            AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(leftValidMask, baseLeftIdxReg, zeroReg, idxMask);
            AscendC::Reg::Compare<GatherIdxType, CMPMODE::GT>(rightValidMask, baseRightIdxReg, zeroReg, idxMask);

            AscendC::Reg::RegTensor<GatherIdxType> leftCChanOffsetReg, rightCChanOffsetReg;
            AscendC::Reg::Duplicate(leftCChanOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(rightCChanOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(leftCChanOffsetReg, hwSizeU, leftValidMask);
            AscendC::Reg::Duplicate(rightCChanOffsetReg, hwSizeU, rightValidMask);

            AscendC::Reg::RegTensor<GatherIdxType> midNOffsetReg, leftNOffsetReg, rightNOffsetReg;
            AscendC::Reg::Duplicate(midNOffsetReg, nStride_, idxMask);
            AscendC::Reg::Duplicate(leftNOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(rightNOffsetReg, static_cast<GatherIdxType>(0), idxMask);
            AscendC::Reg::Duplicate(leftNOffsetReg, nStride_, leftValidMask);
            AscendC::Reg::Duplicate(rightNOffsetReg, nStride_, rightValidMask);

            AscendC::Reg::RegTensor<GatherIdxType> nMidIdxReg, nLeftIdxReg, nRightIdxReg;
            AscendC::Reg::RegTensor<GatherIdxType> midIdxReg, leftIdxReg, rightIdxReg;

            for (uint16_t n = 0; n < static_cast<uint16_t>(factor); n++) {
                nMidIdxReg = baseMidIdxReg;
                nLeftIdxReg = baseLeftIdxReg;
                nRightIdxReg = baseRightIdxReg;

                for (uint16_t c = 0; c < static_cast<uint16_t>(outC); c++) {
                    midIdxReg = nMidIdxReg;
                    leftIdxReg = nLeftIdxReg;
                    rightIdxReg = nRightIdxReg;

                    for (uint16_t loop = 0; loop < static_cast<uint16_t>(totalLoops - 1); loop++) {
                        uint32_t curElements = elementsPerBatch;
                        uint32_t tmpLen = curElements;

                        AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(curElements);

                        AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
                        AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
                        AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
                        AscendC::Reg::RegTensor<PromoteDataT> sumReg;

                        AscendC::Reg::DataCopyGather(midDataReg, midAddr, midIdxReg, mask);
                        AscendC::Reg::DataCopyGather(leftDataReg, midAddr, leftIdxReg, mask);
                        AscendC::Reg::DataCopyGather(rightDataReg, midAddr, rightIdxReg, mask);

                        AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
                        AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);

                        AscendC::Reg::StoreUnAlign(outputAddr, sumReg, ureg1, tmpLen);

                        AscendC::Reg::Add(midIdxReg, midIdxReg, midOffsetReg, idxMask);
                        AscendC::Reg::Add(leftIdxReg, leftIdxReg, leftOffsetReg, idxMask);
                        AscendC::Reg::Add(rightIdxReg, rightIdxReg, rightOffsetReg, idxMask);
                    }

                    // 尾块
                    int64_t curRows = outH - (totalLoops - 1) * rowsPerBatch;
                    uint32_t curElements = static_cast<uint32_t>(curRows * outW);
                    uint32_t tmpLen = curElements;

                    AscendC::Reg::MaskReg mask = AscendC::Reg::UpdateMask<PromoteDataT>(curElements);

                    AscendC::Reg::RegTensor<PromoteDataT> midDataReg;
                    AscendC::Reg::RegTensor<PromoteDataT> leftDataReg;
                    AscendC::Reg::RegTensor<PromoteDataT> rightDataReg;
                    AscendC::Reg::RegTensor<PromoteDataT> sumReg;

                    AscendC::Reg::DataCopyGather(midDataReg, midAddr, midIdxReg, mask);
                    AscendC::Reg::DataCopyGather(leftDataReg, midAddr, leftIdxReg, mask);
                    AscendC::Reg::DataCopyGather(rightDataReg, midAddr, rightIdxReg, mask);

                    AscendC::Reg::Add(sumReg, midDataReg, leftDataReg, mask);
                    AscendC::Reg::Add(sumReg, sumReg, rightDataReg, mask);

                    AscendC::Reg::StoreUnAlign(outputAddr, sumReg, ureg1, tmpLen);
                    AscendC::Reg::StoreUnAlignPost(outputAddr, ureg1, 0);

                    if (c < static_cast<uint16_t>(outC) - 1) {
                        AscendC::Reg::Add(nMidIdxReg, nMidIdxReg, midCChanOffsetReg, idxMask);
                        AscendC::Reg::Add(nLeftIdxReg, nLeftIdxReg, leftCChanOffsetReg, idxMask);
                        AscendC::Reg::Add(nRightIdxReg, nRightIdxReg, rightCChanOffsetReg, idxMask);
                    }
                }

                if (n < static_cast<uint16_t>(factor) - 1) {
                    AscendC::Reg::Add(baseMidIdxReg, baseMidIdxReg, midNOffsetReg, idxMask);
                    AscendC::Reg::Add(baseLeftIdxReg, baseLeftIdxReg, leftNOffsetReg, idxMask);
                    AscendC::Reg::Add(baseRightIdxReg, baseRightIdxReg, rightNOffsetReg, idxMask);
                }
            }
        }
    }
}

/**
 * @brief 折叠 C+H+W 三根轴
 *
 * 处理 UB_AXES=4 的情况，依次折叠 C、H、W 三根轴。
 *
 * 处理流程：
 * 1. 折叠 C 轴 pad：
 *    - 对每个 N（factor 个），调用 FoldCAxisPad 处理 leftPadC 和 rightPadC
 *    - 将 C 轴 pad 数据累加到对应的有效 C 通道
 *
 * 2. 折叠 H 轴 pad：
 *    - 对每个 N，调用 FoldHAxisPad 处理 leftPadH 和 rightPadH
 *    - 将 H 轴 pad 数据累加到对应的有效 H 行
 *
 * 3. 折叠 W 轴并输出：
 *    - 调用 FoldWAxisForCHWAxis 处理所有通道的 W 轴折叠
 *    - 结果输出到 outputLocal
 *
 * @param factor      N 维度大小
 * @param midLocal    中间缓冲区
 * @param outputLocal 输出缓冲区
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::FoldCHWAxis(int64_t factor, LocalTensor<PromoteDataT>& midLocal,
                                                                 LocalTensor<PromoteDataT>& outputLocal)
{
    if (factor <= 0) {
        return;
    }

    // ===== 1. 提取维度参数 =====
    int64_t cHeight = inShape_[dimNum_ - 3];                 // C维度（含 pad）
    int64_t height = inShape_[dimNum_ - 2];                  // 输入高度（含 pad）
    int64_t width = inShape_[dimNum_ - 1];                   // 输入宽度（含 pad）
    uint64_t hwSize = static_cast<uint64_t>(height * width); // 单个 C 通道的大小

    int64_t leftPadC = leftPad_[dimNum_ - 3];
    int64_t rightPadC = rightPad_[dimNum_ - 3];
    int64_t leftPadH = leftPad_[dimNum_ - 2];
    int64_t rightPadH = rightPad_[dimNum_ - 2];
    int64_t leftPadW = leftPad_[dimNum_ - 1];
    int64_t rightPadW = rightPad_[dimNum_ - 1];

    int64_t outC = outShape_[dimNum_ - 3];
    int64_t outH = outShape_[dimNum_ - 2];
    int64_t outW = outShape_[dimNum_ - 1];

    // ===== 2. 获取数据地址 =====
    __local_mem__ PromoteDataT* midAddr = (__local_mem__ PromoteDataT*)midLocal.GetPhyAddr();
    __local_mem__ PromoteDataT* outputAddr = (__local_mem__ PromoteDataT*)outputLocal.GetPhyAddr();

    // 获取索引及偏移缓冲区
    LocalTensor<GatherRangeType> leftIdxLocal = leftPadIdx_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> dstIdxLocal = midIdx_.Get<GatherRangeType>();
    LocalTensor<GatherRangeType> rightIdxLocal = rightPadIdx_.Get<GatherRangeType>();

    __local_mem__ GatherRangeType* leftIdxAddr = (__local_mem__ GatherRangeType*)leftIdxLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* midIdxAddr = (__local_mem__ GatherRangeType*)dstIdxLocal.GetPhyAddr();
    __local_mem__ GatherRangeType* rightIdxAddr = (__local_mem__ GatherRangeType*)rightIdxLocal.GetPhyAddr();

    // ===== 3. 折叠 C 轴 pad =====
    // N 轴已合并进 FoldCAxisPad：当 padCount*hwSize <= VL_RANGE_CNT 时一次 VF 处理多个 N
    uint64_t cNStride = static_cast<uint64_t>(cHeight) * hwSize;
    __local_mem__ PromoteDataT* cBaseAddr = midAddr + ZERO_PAD_CNT;

    FoldCAxisPad(hwSize, leftPadC, true, leftPadC, outC, factor, cNStride, cBaseAddr, leftIdxAddr, midIdxAddr);
    FoldCAxisPad(hwSize, rightPadC, false, leftPadC, outC, factor, cNStride, cBaseAddr, leftIdxAddr, midIdxAddr);

    // ===== 4. 折叠 H 轴 pad =====
    // N 轴已合并进 FoldHAxisPad：当 outC*width <= VL_RANGE_CNT 时一次 VF 处理多个 N
    uint64_t hNStride = static_cast<uint64_t>(cHeight) * hwSize;
    __local_mem__ PromoteDataT* hBaseAddr = midAddr + leftPadC * hwSize;

    FoldHAxisPad(outC, hwSize, width, leftPadH, true, leftPadH, outH, factor, hNStride, hBaseAddr, leftIdxAddr,
                 midIdxAddr);
    FoldHAxisPad(outC, hwSize, width, rightPadH, false, leftPadH, outH, factor, hNStride, hBaseAddr, leftIdxAddr,
                 midIdxAddr);

    // ===== 5. 折叠 W 轴并输出 =====
    FoldWAxisForCHWAxis(factor, outC, hwSize, width, leftPadC, leftPadH, leftPadW, rightPadW, outH, outW, midAddr,
                        outputAddr, midIdxAddr, leftIdxAddr, rightIdxAddr);
}

/**
 * @brief 将结果数据搬出到 GM
 *
 * 完成 UB -> GM 的数据搬运，如果原始数据类型 T 与计算类型 PromoteDataT 不同，
 * 需要先进行原地 Cast。使用 DataCopyPad 实现非对齐搬出。
 *
 * 数据流: outputLocal -> Cast(if needed) -> GM[gmOffset]
 *
 * @param posIdx 当前位置在 outShape 上的索引（用于计算 GM 偏移）
 * @param factor 当前处理的切分轴份数
 */
template <typename T, uint8_t modeName>
__aicore__ inline void PadV3GradGather<T, modeName>::CopyOutToGm(int64_t* posIdx, int64_t factor)
{
    LocalTensor<PromoteDataT> outputLocal = outputQue_.DeQue<PromoteDataT>();

    // 计算GM地址
    int64_t gmOffset = 0;
    for (int64_t i = 0; i <= ubAxis_; i++) {
        gmOffset += posIdx[i] * outStride_[i];
    }
    uint64_t copySize = factor * outputSize_ * sizeof(T);
    uint32_t dataSize = static_cast<uint32_t>(copySize);
    DataCopyExtParams copyParams = {1u, dataSize, 0, 0, 0};

    // 原地 cast（如果 T != PromoteDataT 且需要 cast）并搬出
    if constexpr (!IsSameType<PromoteDataT, T>::value) {
        // 原地 cast: PromoteDataT -> T
        int64_t totalElements = factor * outputSize_;
        LocalTensor<T> outputLocalT = outputLocal.template ReinterpretCast<T>();
        AscendC::Cast(outputLocalT, outputLocal, RoundMode::CAST_RINT, totalElements);

        // 等待 Vector 完成
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        DataCopyPad(outputGm_[gmOffset], outputLocalT, copyParams);
    } else {
        DataCopyPad(outputGm_[gmOffset], outputLocal, copyParams);
    }
    outputQue_.FreeTensor(outputLocal);
}

} // namespace PadV3Grad

#endif // PAD_V3_GRAD_MIRROR_GATHER_H_
