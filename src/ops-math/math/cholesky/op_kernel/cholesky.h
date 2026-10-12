/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHOLESKY_H
#define CHOLESKY_H

#include "kernel_operator.h"

using namespace AscendC;

namespace Cholesky {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BASIC_BLOCK = 32;

template <typename T>
class Cholesky {
public:
    __aicore__ inline Cholesky(){};
    __aicore__ inline void InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData,
                                    TPipe* pipe);
    __aicore__ inline void InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const CholeskyTilingData* tilingData,
                                    TPipe* pipe);
    __aicore__ inline void ProcessTril();
    __aicore__ inline void ProcessTriu();

private:
    __aicore__ inline void PIPE_V_S();
    __aicore__ inline void PIPE_MTE2_V();
    __aicore__ inline void PIPE_MTE3_S();
    __aicore__ inline void PIPE_V_MTE3();
    __aicore__ inline void GetTilingData(const CholeskyTilingData* tilingData);
    __aicore__ inline void FirstColumn(uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void SecondToNColumn(uint32_t index, uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void FirstRow(uint64_t offsetPrefix, uint64_t offset);
    __aicore__ inline void SecondToNRow(uint32_t index, uint64_t offsetPrefix, uint64_t offset);

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        return b == 0 ? a : (a + b - 1) / b;
    }

private:
    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t matSizeN_ = 0;
    uint64_t matrixNumCount_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t blockNum_ = 0;
    T inv_sqrt_A11_ = 0.0f; // 存储缩放因子，避免重复计算和直接访问GM内存
    T ZERO = 0.0f;

    TQue<QuePosition::VECIN, BUFFER_NUM> matAQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> matLeftQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> matRightQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> matResultQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> matLQueue;

    GlobalTensor<T> matAGM;
    GlobalTensor<T> outGM;

    // 辅助函数声明
    __aicore__ inline void ProcessColumnDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                   LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                   uint32_t index, uint64_t offset, uint32_t blockStart,
                                                   uint32_t count);

    __aicore__ inline void ProcessRowDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                uint32_t index, uint64_t offset, uint32_t blockStart, uint32_t count);

    __aicore__ inline T ComputeScaleFactor(LocalTensor<T>& matLLocal, uint64_t offsetPrefix, uint32_t index);
};

template <typename T>
__aicore__ inline void Cholesky<T>::PIPE_V_S()
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::PIPE_MTE2_V()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

template <typename T>
__aicore__ inline void Cholesky<T>::PIPE_MTE3_S()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

template <typename T>
__aicore__ inline void Cholesky<T>::PIPE_V_MTE3()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

template <typename T>
__aicore__ inline void Cholesky<T>::GetTilingData(const CholeskyTilingData* tilingData)
{
    matSizeN_ = tilingData->matSizeN;
    matrixNumCount_ = tilingData->matrixNumCount;
    blockSize_ = tilingData->blockSize;
    blockNum_ = tilingData->blockNum;
}

template <typename T>
__aicore__ inline void Cholesky<T>::InitTril(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                             const CholeskyTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    GetTilingData(tilingData);

    matAGM.SetGlobalBuffer((__gm__ T*)self, matSizeN_ * matSizeN_);
    outGM.SetGlobalBuffer((__gm__ T*)out, matSizeN_ * matSizeN_);

    // 使用分块大小计算buffer，减少UB内存使用
    uint64_t columnBufferSize = blockSize_ * BASIC_BLOCK;
    uint64_t rowBufferSize = CeilDiv(blockSize_ * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK;

    pipe->InitBuffer(matAQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matLeftQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matRightQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matResultQueue, BUFFER_NUM, rowBufferSize);
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTril()
{
    if (blockIdx_ < blockDim_) {
        auto loopTimes = matrixNumCount_ / blockDim_;
        for (uint64_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint64_t offsetPrefix = blockIdx_ + blockDim_ * loopIndex;
            if (offsetPrefix < matrixNumCount_) {
                uint64_t offset = offsetPrefix * matSizeN_ * matSizeN_;
                FirstColumn(offsetPrefix, offset);
                for (uint32_t index = 1; index < matSizeN_; index++) {
                    SecondToNColumn(index, offsetPrefix, offset);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstColumn(uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();

    // 核内分块处理，每次处理blockSize大小的数据
    for (uint32_t blockStart = 0; blockStart < matSizeN_; blockStart += blockSize_) {
        uint32_t count = (matSizeN_ - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - blockStart);

        DataCopyExtParams copyParamsMatALocal{static_cast<uint16_t>(count), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matALocal, matAGM[offset + blockStart * matSizeN_], copyParamsMatALocal, padParamsMatALocal);
        PIPE_MTE2_V();

        // 只在处理第一个元素时计算平方根并存储缩放因子
        if (blockStart == 0) {
            T A11 = matALocal.GetValue(0);
            if (matrixNumCount_ > 1) {
                ascendc_assert(A11 > 0.0f,
                               "(Batch element %d): The factorization could not be completed because the input is not "
                               "positive-definite (the leading minor of order 1 is not positive-definite).\n",
                               offsetPrefix);
            } else {
                ascendc_assert(A11 > 0.0f,
                               "The factorization could not be completed because the input is not positive-definite "
                               "(the leading minor of order 1 is not positive-definite).\n");
            }
            inv_sqrt_A11_ = T(1 / sqrt(A11));
            Muls(matALocal, matALocal, inv_sqrt_A11_, count * BASIC_BLOCK / sizeof(T));
        } else {
            // 直接使用之前计算好的缩放因子，避免访问GM内存
            Muls(matALocal, matALocal, inv_sqrt_A11_, count * BASIC_BLOCK / sizeof(T));
        }

        PIPE_V_MTE3();
        DataCopyExtParams dataCopyOutParams{static_cast<uint16_t>(count), sizeof(T), 0,
                                            static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0};
        DataCopyPad(outGM[offset + blockStart * matSizeN_], matALocal, dataCopyOutParams);
        PIPE_MTE3_S();
    }

    matAQueue.FreeTensor(matALocal);
}

// 辅助函数：处理SecondToNColumn中的点积计算部分
template <typename T>
__aicore__ inline void Cholesky<T>::ProcessColumnDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                            LocalTensor<T>& matRightLocal,
                                                            LocalTensor<T>& matResultLocal, uint32_t index,
                                                            uint64_t offset, uint32_t blockStart, uint32_t count)
{
    // 计算左侧分块的数量
    uint32_t leftBlockNum = (index + blockSize_ - 1) / blockSize_;

    for (uint32_t leftBlockIdx = 0; leftBlockIdx < leftBlockNum; leftBlockIdx++) {
        // 计算leftlocal当前块的起始位置和大小，分块大小为blockSize_
        uint32_t leftBlockStart = leftBlockIdx * blockSize_;
        uint32_t leftBlockSize = (index - leftBlockStart) > blockSize_ ? blockSize_ : (index - leftBlockStart);

        // 搬运当前块的matLeftLocal数据
        DataCopyExtParams copyParamsLeftLocal{1, static_cast<uint32_t>(sizeof(T) * leftBlockSize), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsLeftLocal{false, 0, 0, 0};
        DataCopyPad(matLeftLocal, outGM[offset + index * matSizeN_ + leftBlockStart], copyParamsLeftLocal,
                    padParamsLeftLocal);

        // 对当前块中的每一行，分块搬运matRightLocal并计算
        for (uint32_t row_in_block = 0; row_in_block < count; row_in_block++) {
            uint32_t row_below_pivot = blockStart + row_in_block;

            // 搬运当前块的matRightLocal数据
            DataCopyPad(matRightLocal, outGM[offset + (index + row_below_pivot) * matSizeN_ + leftBlockStart],
                        copyParamsLeftLocal, padParamsLeftLocal);
            PIPE_MTE2_V();

            // 计算当前块的点积并累加结果
            Mul(matResultLocal, matLeftLocal, matRightLocal, leftBlockSize);
            ReduceSum<T>(matResultLocal, matResultLocal, matResultLocal, leftBlockSize);

            // 将当前块的结果累加到matLLocal中
            T currentSum = matResultLocal.GetValue(0);
            T existingSum = matLLocal.GetValue(row_in_block * BASIC_BLOCK / sizeof(T));
            matLLocal.SetValue(row_in_block * BASIC_BLOCK / sizeof(T), existingSum + currentSum);
            PIPE_V_S();
        }
    }
}

// 辅助函数：计算缩放因子并进行正定性检查
template <typename T>
__aicore__ inline T Cholesky<T>::ComputeScaleFactor(LocalTensor<T>& matLLocal, uint64_t offsetPrefix, uint32_t index)
{
    T b1 = matLLocal.GetValue(0);
    PIPE_V_S();
    if (matrixNumCount_ > 1) {
        ascendc_assert(b1 > 0.0f,
                       "(Batch element %d): The factorization could not be completed because the input is not "
                       "positive-definite (the leading minor of order %d is not positive-definite).\n",
                       offsetPrefix, index + 1);
    } else {
        ascendc_assert(b1 > 0.0f,
                       "The factorization could not be completed because the input is not positive-definite (the "
                       "leading minor of order %d is not positive-definite).\n",
                       index + 1);
    }

    // 计算缩放因子
    return T(1 / sqrt(b1));
}

// 辅助函数：处理SecondToNRow中的点积计算部分
template <typename T>
__aicore__ inline void Cholesky<T>::ProcessRowDotProduct(LocalTensor<T>& matLLocal, LocalTensor<T>& matLeftLocal,
                                                         LocalTensor<T>& matRightLocal, LocalTensor<T>& matResultLocal,
                                                         uint32_t index, uint64_t offset, uint32_t blockStart,
                                                         uint32_t count)
{
    // 计算左侧分块的数量
    uint32_t leftBlockNum = (index + blockSize_ - 1) / blockSize_;

    for (uint32_t leftBlockIdx = 0; leftBlockIdx < leftBlockNum; leftBlockIdx++) {
        // 计算leftlocal当前块的起始位置和大小，分块大小为blockSize_
        uint32_t leftBlockStart = leftBlockIdx * blockSize_;
        uint32_t leftBlockSize = (index - leftBlockStart) > blockSize_ ? blockSize_ : (index - leftBlockStart);

        // 搬运当前块的matLeftLocal数据
        DataCopyExtParams copyParamsLeftLocal{static_cast<uint16_t>(leftBlockSize), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsLeftLocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matLeftLocal, outGM[offset + index + leftBlockStart * matSizeN_], copyParamsLeftLocal,
                    padParamsLeftLocal);

        // 对当前块中的每一列，分块搬运matRightLocal并计算
        for (uint32_t col_in_block = 0; col_in_block < count; col_in_block++) {
            uint32_t column_right_pivot = blockStart + col_in_block;

            // 搬运当前块的matRightLocal数据
            DataCopyPad(matRightLocal, outGM[offset + (index + column_right_pivot) + leftBlockStart * matSizeN_],
                        copyParamsLeftLocal, padParamsLeftLocal);
            PIPE_MTE2_V();

            // 计算当前块的点积并累加结果
            Mul(matResultLocal, matLeftLocal, matRightLocal, leftBlockSize * BASIC_BLOCK / sizeof(T));
            ReduceSum<T>(matResultLocal, matResultLocal, matResultLocal, leftBlockSize * BASIC_BLOCK / sizeof(T));

            // 将当前块的结果累加到matLLocal中
            T currentSum = matResultLocal.GetValue(0);
            T existingSum = matLLocal.GetValue(col_in_block);
            matLLocal.SetValue(col_in_block, existingSum + currentSum);
            PIPE_V_S();
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNColumn(uint32_t index, uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();
    LocalTensor<T> matLLocal = matLQueue.AllocTensor<T>();
    LocalTensor<T> matLeftLocal = matLeftQueue.AllocTensor<T>();
    LocalTensor<T> matRightLocal = matRightQueue.AllocTensor<T>();
    LocalTensor<T> matResultLocal = matResultQueue.AllocTensor<T>();

    // 存储当前列的缩放因子，所有分块共享同一个缩放因子
    T column_scale_factor = 0.0f;
    bool scale_factor_computed = false;

    // 对当前列的所有元素进行分块处理
    for (uint32_t blockStart = 0; blockStart < (matSizeN_ - index); blockStart += blockSize_) {
        // 计算当前块的大小
        uint32_t count = (matSizeN_ - index - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - index - blockStart);

        // 1. 先搬运当前块的count个A元素进来
        DataCopyExtParams copyParamsMatALocal{static_cast<uint16_t>(count), sizeof(T),
                                              static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{true, 0, BASIC_BLOCK / sizeof(T) - 1, 0};
        DataCopyPad(matALocal, matAGM[offset + index * matSizeN_ + index + blockStart * matSizeN_], copyParamsMatALocal,
                    padParamsMatALocal);
        PIPE_MTE2_V();

        // 2. 初始化当前块的L结果为0
        Duplicate(matLLocal, ZERO, count * BASIC_BLOCK / sizeof(T));

        // 3. 调用辅助函数处理点积计算
        ProcessColumnDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, blockStart,
                                count);
        PipeBarrier<PIPE_ALL>();

        // 4. 执行计算操作
        Sub(matLLocal, matALocal, matLLocal, count * BASIC_BLOCK / sizeof(T));

        // 只在第一次分块时计算缩放因子和进行正定性检查
        if (blockStart == 0) {
            column_scale_factor = ComputeScaleFactor(matLLocal, offsetPrefix, index);
            scale_factor_computed = true;
        }

        // 对当前块的所有元素应用同一个缩放因子
        Muls(matLLocal, matLLocal, column_scale_factor, count * BASIC_BLOCK / sizeof(T));

        // 5. 最后得到count个L元素并搬出
        PIPE_V_MTE3();
        DataCopyExtParams dataCopyOutParams{static_cast<uint16_t>(count), sizeof(T), 0,
                                            static_cast<uint32_t>((matSizeN_ - 1) * sizeof(T)), 0};
        DataCopyPad(outGM[offset + index * matSizeN_ + index + blockStart * matSizeN_], matLLocal, dataCopyOutParams);
        PIPE_MTE3_S();
    }

    // 释放张量资源
    matResultQueue.FreeTensor(matResultLocal);
    matRightQueue.FreeTensor(matRightLocal);
    matLeftQueue.FreeTensor(matLeftLocal);
    matLQueue.FreeTensor(matLLocal);
    matAQueue.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::InitTriu(GM_ADDR self, GM_ADDR out, GM_ADDR workspace,
                                             const CholeskyTilingData* tilingData, TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    GetTilingData(tilingData);

    matAGM.SetGlobalBuffer((__gm__ T*)self, matSizeN_ * matSizeN_);
    outGM.SetGlobalBuffer((__gm__ T*)out, matSizeN_ * matSizeN_);

    // 使用分块大小计算buffer，减少UB内存使用
    uint32_t columnBufferSize = blockSize_ * BASIC_BLOCK;
    uint32_t rowBufferSize = CeilDiv(blockSize_ * sizeof(T), BASIC_BLOCK) * BASIC_BLOCK;

    pipe->InitBuffer(matAQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLQueue, BUFFER_NUM, rowBufferSize);
    pipe->InitBuffer(matLeftQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matRightQueue, BUFFER_NUM, columnBufferSize);
    pipe->InitBuffer(matResultQueue, BUFFER_NUM, columnBufferSize);
}

template <typename T>
__aicore__ inline void Cholesky<T>::ProcessTriu()
{
    if (blockIdx_ < blockDim_) {
        auto loopTimes = matrixNumCount_ / blockDim_;
        for (uint64_t loopIndex = 0; loopIndex <= loopTimes; loopIndex++) {
            uint64_t offsetPrefix = blockIdx_ + blockDim_ * loopIndex;
            if (offsetPrefix < matrixNumCount_) {
                uint64_t offset = offsetPrefix * matSizeN_ * matSizeN_;
                FirstRow(offsetPrefix, offset);
                for (uint32_t index = 1; index < matSizeN_; index++) {
                    SecondToNRow(index, offsetPrefix, offset);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void Cholesky<T>::FirstRow(uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();

    // 核内分块处理，每次处理blockSize大小的数据
    for (uint32_t blockStart = 0; blockStart < matSizeN_; blockStart += blockSize_) {
        uint32_t count = (matSizeN_ - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - blockStart);

        DataCopyExtParams copyParamsMatALocal{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM[offset + blockStart], copyParamsMatALocal, padParamsMatALocal);
        PIPE_MTE2_V();

        // 只在处理第一个元素时计算平方根并存储缩放因子
        if (blockStart == 0) {
            T A11_sqrt = matALocal.GetValue(0);
            if (matrixNumCount_ > 1) {
                ascendc_assert(A11_sqrt > 0.0f,
                               "(Batch element %d): The factorization could not be completed because the input is not "
                               "positive-definite (the leading minor of order 1 is not positive-definite).\n",
                               offsetPrefix);
            } else {
                ascendc_assert(A11_sqrt > 0.0f,
                               "The factorization could not be completed because the input is not positive-definite "
                               "(the leading minor of order 1 is not positive-definite).\n");
            }
            inv_sqrt_A11_ = T(1 / sqrt(A11_sqrt));
            Muls(matALocal, matALocal, inv_sqrt_A11_, count);
        } else {
            // 使用之前存储的缩放因子，避免重复计算和直接访问GM内存
            Muls(matALocal, matALocal, inv_sqrt_A11_, count);
        }

        // 搬出当前块的结果
        PIPE_V_MTE3();
        DataCopyExtParams dataCopyOutParams{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPad(outGM[offset + blockStart], matALocal, dataCopyOutParams);
        PIPE_MTE3_S();
    }

    matAQueue.FreeTensor(matALocal);
}

template <typename T>
__aicore__ inline void Cholesky<T>::SecondToNRow(uint32_t index, uint64_t offsetPrefix, uint64_t offset)
{
    LocalTensor<T> matALocal = matAQueue.AllocTensor<T>();
    LocalTensor<T> matLLocal = matLQueue.AllocTensor<T>();
    LocalTensor<T> matLeftLocal = matLeftQueue.AllocTensor<T>();
    LocalTensor<T> matRightLocal = matRightQueue.AllocTensor<T>();
    LocalTensor<T> matResultLocal = matResultQueue.AllocTensor<T>();

    // 存储当前行的缩放因子，所有分块共享同一个缩放因子
    T row_scale_factor = 0.0f;
    bool scale_factor_computed = false;

    // 对当前行的所有元素进行分块处理
    for (uint32_t blockStart = 0; blockStart < (matSizeN_ - index); blockStart += blockSize_) {
        // 计算当前块的大小
        uint32_t count = (matSizeN_ - index - blockStart) > blockSize_ ? blockSize_ : (matSizeN_ - index - blockStart);

        // 1. 先搬运当前块的count个A元素进来
        DataCopyExtParams copyParamsMatALocal{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPadExtParams<T> padParamsMatALocal{false, 0, 0, 0};
        DataCopyPad(matALocal, matAGM[offset + index * matSizeN_ + index + blockStart], copyParamsMatALocal,
                    padParamsMatALocal);
        PIPE_MTE2_V();

        // 2. 初始化当前块的L结果为0
        Duplicate(matLLocal, ZERO, count);

        // 3. 调用辅助函数处理点积计算
        ProcessRowDotProduct(matLLocal, matLeftLocal, matRightLocal, matResultLocal, index, offset, blockStart, count);
        PipeBarrier<PIPE_ALL>();

        // 4. 执行计算操作
        Sub(matLLocal, matALocal, matLLocal, count);

        // 只在第一次分块时计算缩放因子和进行正定性检查
        if (blockStart == 0) {
            row_scale_factor = ComputeScaleFactor(matLLocal, offsetPrefix, index);
            scale_factor_computed = true;
        }

        // 对当前块的所有元素应用同一个缩放因子
        Muls(matLLocal, matLLocal, row_scale_factor, count);

        // 5. 最后得到count个L元素并搬出
        PIPE_V_MTE3();
        DataCopyExtParams dataCopyOutParams{1, static_cast<uint32_t>(sizeof(T) * count), 0, 0, 0};
        DataCopyPad(outGM[offset + index * matSizeN_ + index + blockStart], matLLocal, dataCopyOutParams);
        PIPE_MTE3_S();
    }

    // 释放张量资源
    matResultQueue.FreeTensor(matResultLocal);
    matRightQueue.FreeTensor(matRightLocal);
    matLeftQueue.FreeTensor(matLeftLocal);
    matLQueue.FreeTensor(matLLocal);
    matAQueue.FreeTensor(matALocal);
}

} // namespace Cholesky
#endif
