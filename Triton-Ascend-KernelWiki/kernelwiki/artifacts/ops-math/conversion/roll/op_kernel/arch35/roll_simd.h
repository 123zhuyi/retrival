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
 * \file roll_simd.h
 * \brief
 */

#ifndef __ROLL_SIMD_BLOCK_MOVE_H__
#define __ROLL_SIMD_BLOCK_MOVE_H__

#include "kernel_operator.h"
#include "roll_struct.h"
#include "op_kernel/platform_util.h"

namespace Roll {
using namespace AscendC;
constexpr int64_t ALIGN_NUM = 32;
template <typename T>
class RollSimd {
    struct MoveParam {
        int64_t moveNum = 0;
        int64_t inputIndex[4] = {0};
        int64_t blockCount[4] = {0};
        int64_t blockLen[4] = {0};
        int64_t srcStride[4] = {0};
        int64_t dstStride[4] = {0};
        int64_t outputIndex[4] = {0};
    };

public:
    __aicore__ inline RollSimd(TPipe* pipe, const RollTilingData* tiling) : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyIn(int64_t index, int64_t count, int64_t loop);
    __aicore__ inline void CopyOut(int64_t index, int64_t count, int64_t loop);
    __aicore__ inline void setMoveParam(
        int64_t index, int64_t intputIndex, int64_t blockCount, int64_t blockLen, int64_t srcStride,
        int64_t dstStride, int64_t outputindex);
    __aicore__ inline void updataInputIndex(int64_t inputIndex, int64_t& outputIndex);
    __aicore__ inline void ComputeOutIndex(int64_t inputIndex, int64_t& outputIndex, int64_t& count);
    __aicore__ inline void Process();

private:
    const RollTilingData* tilingData_;
    TPipe* pipe_;

    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> xInQue_;
    MoveParam moveparam;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    int64_t curCoreBaseIndex_ = 0;
    int64_t perCoreEmelents_ = 0;
    int64_t curCoreElements_ = 0;
    int64_t blockIdx_ = 0;
    int64_t inputIndices_[MAX_DIM_NUM] = {0};
    bool isBlockMove = false;
};

template <typename T>
__aicore__ inline void RollSimd<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    perCoreEmelents_ = tilingData_->perCoreElements;
    if (blockIdx_ == GetBlockNum() - 1) {
        curCoreElements_ = tilingData_->lastCoreElements;
    } else {
        curCoreElements_ = tilingData_->perCoreElements;
    }
    curCoreBaseIndex_ = perCoreEmelents_ * blockIdx_;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    pipe_->InitBuffer(xInQue_, BUF_NUM, tilingData_->maxElements * sizeof(T));
}

template <typename T>
__aicore__ inline void RollSimd<T>::CopyIn(int64_t index, int64_t count, int64_t loop)
{
    if (isBlockMove) {
        // 块搬运，利用moveparam参数
        LocalTensor<T> xTensor = xInQue_.AllocTensor<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = moveparam.blockCount[loop];
        dataCopyParams.blockLen = moveparam.blockLen[loop] * sizeof(T);
        dataCopyParams.srcStride = moveparam.srcStride[loop] * sizeof(T);
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xTensor, xGm_[moveparam.inputIndex[loop]], dataCopyParams, dataCopyPadParams);
        xInQue_.EnQue<T>(xTensor);
    } else {
        LocalTensor<T> xTensor = xInQue_.AllocTensor<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = count * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xTensor, xGm_[index], dataCopyParams, dataCopyPadParams);
        xInQue_.EnQue<T>(xTensor);
    }
}

template <typename T>
__aicore__ inline void RollSimd<T>::CopyOut(int64_t index, int64_t count, int64_t loop)
{
    if (isBlockMove) { // 块搬出
        LocalTensor<T> xTensor = xInQue_.DeQue<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = moveparam.blockCount[loop];
        dataCopyParams.blockLen = moveparam.blockLen[loop] * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = moveparam.srcStride[loop] * sizeof(T);
        DataCopyPad(yGm_[moveparam.outputIndex[loop]], xTensor, dataCopyParams);
        xInQue_.FreeTensor<T>(xTensor);
    } else {
        LocalTensor<T> xTensor = xInQue_.DeQue<T>();
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = count * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(yGm_[index], xTensor, dataCopyParams);
        xInQue_.FreeTensor<T>(xTensor);
    }
}

template <typename T>
__aicore__ inline void RollSimd<T>::updataInputIndex(int64_t inputIndex, int64_t& outputIndex)
{
    outputIndex = 0;
    for (int64_t dim = 0; dim < tilingData_->dimNum; dim++) {
        inputIndices_[dim] = inputIndex / tilingData_->strides[dim];
        inputIndex = inputIndex % tilingData_->strides[dim];
        outputIndex +=
            (inputIndices_[dim] + tilingData_->shifts[dim]) % tilingData_->shapes[dim] * tilingData_->strides[dim];
    }
}

template <typename T>
__aicore__ inline void RollSimd<T>::setMoveParam(
    int64_t index, int64_t inputindex, int64_t blockCount, int64_t blockLen, int64_t srcStride, int64_t dstStride,
    int64_t outputindex)
{
    moveparam.inputIndex[index] = inputindex;
    moveparam.blockCount[index] = blockCount;
    moveparam.blockLen[index] = blockLen;
    moveparam.srcStride[index] = srcStride;
    moveparam.dstStride[index] = dstStride;
    moveparam.outputIndex[index] = outputindex;
}

template <typename T>
__aicore__ inline void RollSimd<T>::ComputeOutIndex(int64_t inputIndex, int64_t& outputIndex, int64_t& count)
{
    updataInputIndex(inputIndex, outputIndex);
    if (tilingData_->dimNum == 1) {
        count = tilingData_->shapes[tilingData_->dimNum - 1] - inputIndices_[tilingData_->dimNum - 1];
        if (count > tilingData_->shifts[tilingData_->dimNum - 1]) {
            count -= tilingData_->shifts[tilingData_->dimNum - 1];
        }
        isBlockMove = false;
    } else if (tilingData_->shifts[tilingData_->dimNum - 1] != 0) {
        // 考虑搬运后的UB对齐
        int64_t perUbMaxElements = Std::min(curCoreElements_, tilingData_->maxElements); // 单次UB最大搬运量
        // W轴需要shift，优化
        if (inputIndices_[tilingData_->dimNum - 1] != 0) {
            // W轴的index不为0，说明从中间开始的，只搬运这一行
            count = Std::min(
                tilingData_->shapes[tilingData_->dimNum - 1] - inputIndices_[tilingData_->dimNum - 1],
                perUbMaxElements);
            int64_t shiftIndex =
                tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->shifts[tilingData_->dimNum - 1];
            int64_t count1 = shiftIndex - inputIndices_[tilingData_->dimNum - 1];
            int64_t count2 = inputIndices_[tilingData_->dimNum - 1] + count - shiftIndex;
            if (shiftIndex > inputIndices_[tilingData_->dimNum - 1] &&
                shiftIndex < (inputIndices_[tilingData_->dimNum - 1] + count)) { // 分两块搬运
                moveparam.moveNum = 2;
                setMoveParam(0, inputIndex, 1, count1, 0, 0, outputIndex);
                inputIndex += moveparam.blockLen[0];
                int64_t tempOutputIndex = 0;
                updataInputIndex(inputIndex, tempOutputIndex);
                setMoveParam(1, inputIndex, 1, count2, 0, 0, tempOutputIndex);
            } else {
                // 只搬运一块
                moveparam.moveNum = 1;
                setMoveParam(0, inputIndex, 1, count, 0, 0, outputIndex);
            }
        } else {
            // W轴的index从0开始 hLen个 W
            int64_t wAlign = (tilingData_->shapes[tilingData_->dimNum - 1] * sizeof(T) + ALIGN_NUM - 1) / ALIGN_NUM * ALIGN_NUM / sizeof(T);
            int64_t hLen = tilingData_->maxElements / wAlign;
            hLen = Std::min(hLen, curCoreElements_ / tilingData_->shapes[tilingData_->dimNum - 1]);
            if (hLen == 0) {
                // 不足一个 W，处理最后一块
                count = perUbMaxElements;
                int64_t shiftIndex =
                    tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->shifts[tilingData_->dimNum - 1];
                int64_t count1 = shiftIndex - inputIndices_[tilingData_->dimNum - 1];
                int64_t count2 = count - count1;
                if (shiftIndex > inputIndices_[tilingData_->dimNum - 1] &&
                    shiftIndex < (inputIndices_[tilingData_->dimNum - 1] + count)) {
                    // 分两块搬运
                    moveparam.moveNum = 2;
                    setMoveParam(0, inputIndex, 1, count1, 0, 0, outputIndex);
                    inputIndex += moveparam.blockLen[0];
                    int64_t tempOutputIndex = 0;
                    updataInputIndex(inputIndex, tempOutputIndex);
                    setMoveParam(1, inputIndex, 1, count2, 0, 0, tempOutputIndex);
                } else {
                    moveparam.moveNum = 1;
                    setMoveParam(0, inputIndex, 1, count, 0, 0, outputIndex);
                }
            } else {
                int64_t curHindex = inputIndices_[tilingData_->dimNum - 2];
                hLen = Std::min(hLen, tilingData_->shapes[tilingData_->dimNum - 2] - curHindex);
                int64_t hShift =
                    tilingData_->shapes[tilingData_->dimNum - 2] - tilingData_->shifts[tilingData_->dimNum - 2];
                if (hShift > curHindex && hShift < (curHindex + hLen)) {
                    // shift轴在中间
                    moveparam.moveNum = 4;
                    int64_t hCount1 = hShift - curHindex;
                    int64_t hCount2 = hLen - hCount1;
                    int64_t wLen1 =
                        tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->shifts[tilingData_->dimNum - 1];
                    int64_t wLen2 = tilingData_->shifts[tilingData_->dimNum - 1];
                    setMoveParam(0, inputIndex, hCount1, wLen1, wLen2, 0, outputIndex);
                    inputIndex += moveparam.blockLen[0];
                    outputIndex -= tilingData_->shifts[tilingData_->dimNum - 1];
                    setMoveParam(1, inputIndex, hCount1, wLen2, wLen1, 0, outputIndex);
                    inputIndex += (tilingData_->shapes[tilingData_->dimNum - 1] * hCount1 - moveparam.blockLen[0]);
                    updataInputIndex(inputIndex, outputIndex);
                    setMoveParam(2, inputIndex, hCount2, wLen1, wLen2, 0, outputIndex);
                    inputIndex += wLen1;
                    outputIndex -= tilingData_->shifts[tilingData_->dimNum - 1];
                    setMoveParam(3, inputIndex, hCount2, wLen2, wLen1, 0, outputIndex);
                } else {
                    // 两组参数直接搬完
                    moveparam.moveNum = 2;
                    setMoveParam(
                        0, inputIndex, hLen,
                        tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->shifts[tilingData_->dimNum - 1],
                        tilingData_->shifts[tilingData_->dimNum - 1], 0, outputIndex);
                    inputIndex += moveparam.blockLen[0];
                    outputIndex -= tilingData_->shifts[tilingData_->dimNum - 1];
                    setMoveParam(
                        1, inputIndex, hLen, tilingData_->shifts[tilingData_->dimNum - 1],
                        tilingData_->shapes[tilingData_->dimNum - 1] - tilingData_->shifts[tilingData_->dimNum - 1], 0,
                        outputIndex);
                }
                // 更新count
                count = hLen * tilingData_->shapes[tilingData_->dimNum - 1];
            }
        }
        isBlockMove = true;
    } else {
        // 只shiftH
        count = tilingData_->shapes[tilingData_->dimNum - 2] - inputIndices_[tilingData_->dimNum - 2];
        if (count > tilingData_->shifts[tilingData_->dimNum - 2]) {
            count = count - tilingData_->shifts[tilingData_->dimNum - 2];
        }
        count = count * tilingData_->shapes[tilingData_->dimNum - 1] - inputIndices_[tilingData_->dimNum - 1];
        isBlockMove = false;
    }
}

template <typename T>
__aicore__ inline void RollSimd<T>::Process()
{
    int64_t inputIndex = curCoreBaseIndex_;
    while (curCoreElements_ > 0) {
        int64_t outIndex = 0;
        int64_t count = 0;
        ComputeOutIndex(inputIndex, outIndex, count);
        if (isBlockMove) {
            int64_t loops = moveparam.moveNum;
            for (int64_t loop = 0; loop < loops; loop++) {
                CopyIn(0, 0, loop);
                CopyOut(0, 0, loop);
            }
        } else {
            if (count > curCoreElements_) {
                count = curCoreElements_;
            }
            int64_t loops = (count + tilingData_->maxElements - 1) / tilingData_->maxElements;
            int64_t perLoopCount = (count + loops - 1) / loops;
            int64_t lastLoopCount = count - (loops - 1) * perLoopCount;
            for (int64_t loop = 0; loop < loops - 1; loop++) {
                CopyIn(inputIndex + loop * perLoopCount, perLoopCount, 0);
                CopyOut(outIndex + loop * perLoopCount, perLoopCount, 0);
            }
            CopyIn(inputIndex + (loops - 1) * perLoopCount, lastLoopCount, 0);
            CopyOut(outIndex + (loops - 1) * perLoopCount, lastLoopCount, 0);
        }
        inputIndex += count;
        curCoreElements_ -= count;
    }
}

} // namespace Roll
#endif // __ROLL_SIMD_H__