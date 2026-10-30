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
 * \file transpose021.h
 * \brief
 */

#ifndef ASCEND_TRANSPOSE021_H
#define ASCEND_TRANSPOSE021_H

#include "transpose_v2.h"

namespace TransposeV2 {
template <typename T, bool ISFP32 = false>
class Transpose021
{
public:
    __aicore__ inline Transpose021(AscendC::TPipe* p) : pipe(p)
    {}
    __aicore__ inline void Process(__gm__ uint8_t* src, __gm__ uint8_t* dst,
                                   const TransposeV2TilingData* __restrict__ tiling_data)
    {
        tasksPerCore = tiling_data->tasksPerCore;
        tasksTail = tiling_data->tasksTail;
        inputH = tiling_data->inputH;
        inputW = tiling_data->inputW;
        inputH16Align = tiling_data->inputH16Align;
        inputWAlign = tiling_data->inputWAlign;
        hOnce = tiling_data->hOnce;
        tasksOnceMax = tiling_data->tasksOnceMax;
        transLoop = tiling_data->transLoop;
        repeatH = tiling_data->repeatH;
        doubleBuffer = tiling_data->doubleBuffer;

        uint64_t blockIdx = AscendC::GetBlockIdx();
        uint64_t startOffset = blockIdx * tasksPerCore * inputH * inputW;
        if (blockIdx < tasksTail) {
            tasksPerCore++;
            startOffset += blockIdx * inputH * inputW;
        } else {
            startOffset += tasksTail * inputH * inputW;
        }
        srcGlobal.SetGlobalBuffer((__gm__ T*)src + startOffset);
        dstGlobal.SetGlobalBuffer((__gm__ T*)dst + startOffset);

        typeSize = sizeof(T);
        block = BLOCK_SIZE / typeSize;
        tasksOnce = tasksOnceMax;
        loop = tasksPerCore / tasksOnceMax;
        tail = tasksPerCore % tasksOnceMax;

        pipe->InitBuffer(queIn, doubleBuffer, MAX_UB_SIZE / 2 / doubleBuffer);  // 2 is buffer num
        pipe->InitBuffer(queOut, doubleBuffer, MAX_UB_SIZE / 2 / doubleBuffer); // 2 is buffer num
        if (inputW == inputWAlign && inputH == inputH16Align) {
            ProcessOp<true, true>();
        } else if (inputW == inputWAlign && inputH != inputH16Align) {
            ProcessOp<true, false>();
        } else if (inputW != inputWAlign && inputH == inputH16Align) {
            ProcessOp<false, true>();
        } else {
            ProcessOp<false, false>();
        }
    }

    template <bool ISWALIGN = false, bool ISHALIGN = false>
    __aicore__ inline void ProcessOp()
    {
        unPadHNum = tasksOnce * inputWAlign / block;
        for (uint64_t i = 0; i < loop; ++i) {
            CopyIn(i);
            Compute<ISWALIGN, ISHALIGN>();
            CopyOut<ISHALIGN>(i);
        }
        if (tail > 0) {
            tasksOnce = tail;
            transLoop = GetAlign(tasksOnce * inputH, hOnce) / hOnce;
            unPadHNum = tasksOnce * inputWAlign / block;
            CopyIn(loop);
            Compute<ISWALIGN, ISHALIGN>();
            CopyOut<ISHALIGN>(loop);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t index)
    {
        AscendC::LocalTensor<T> srcLocal = queIn.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParamsIn{
            1, static_cast<uint32_t>(tasksOnce * inputH * inputW * typeSize), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(srcLocal, srcGlobal[index * tasksOnceMax * inputH * inputW], copyParamsIn, padParams);
        queIn.EnQue(srcLocal);
    }

    template <bool ISWALIGN = false, bool ISHALIGN = false>
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> srcLocal = queIn.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = queOut.AllocTensor<T>();
        if constexpr (ISWALIGN) {
            ReshapeCompute(srcLocal, dstLocal);
            AscendC::PipeBarrier<PIPE_V>();
            TransposeCompute(dstLocal, srcLocal);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (ISHALIGN) {
                VECINToVECOUT(srcLocal, dstLocal);
            } else {
                UnPadHCompute(srcLocal, dstLocal);
            }
        } else {
            PadWCompute(srcLocal, dstLocal);
            AscendC::PipeBarrier<PIPE_V>();
            ReshapeCompute(dstLocal, srcLocal);
            AscendC::PipeBarrier<PIPE_V>();
            TransposeCompute(srcLocal, dstLocal);
            AscendC::PipeBarrier<PIPE_V>();
            UnPadCompute(dstLocal, srcLocal);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (ISHALIGN) {
                VECINToVECOUT(srcLocal, dstLocal);
            } else {
                UnPadHCompute(srcLocal, dstLocal);
            }
        }
        queOut.EnQue(dstLocal);
        queIn.FreeTensor(srcLocal);
    }

    template <bool ISHALIGN = false>
    __aicore__ inline void CopyOut(uint64_t index)
    {
        AscendC::LocalTensor<T> dstLocal = queOut.DeQue<T>();
        if constexpr (ISHALIGN) {
            AscendC::DataCopyExtParams copyParamsOut{
                1, static_cast<uint32_t>(tasksOnce * inputW * inputH * typeSize), 0, 0, 0};
            AscendC::DataCopyPad(dstGlobal[index * tasksOnceMax * inputW * inputH], dstLocal, copyParamsOut);
        } else {
            uint32_t blockLen = unPadHNum * inputH * typeSize;
            uint16_t blockCount = (tasksOnce * inputW * inputH * typeSize) / blockLen;
            uint32_t tailBlock = (tasksOnce * inputW * inputH * typeSize) % blockLen;
            uint32_t srcStride = unPadHNum * (inputH16Align - inputH) * typeSize / BLOCK_SIZE;
            AscendC::DataCopyExtParams copyParamsOut{blockCount, blockLen, srcStride, 0, 0};

            AscendC::DataCopyPad(dstGlobal[index * tasksOnceMax * inputW * inputH], dstLocal, copyParamsOut);
            if (tailBlock > 0) {
                copyParamsOut.blockCount = 1;
                copyParamsOut.blockLen = tailBlock;
                uint64_t srcOffset = blockCount * unPadHNum * inputH16Align;
                uint64_t dstOffset = index * tasksOnceMax * inputW * inputH + blockCount * blockLen / typeSize;
                AscendC::DataCopyPad(dstGlobal[dstOffset], dstLocal[srcOffset], copyParamsOut);
            }
        }
        queOut.FreeTensor(dstLocal);
    }

    // [transLoop, TRANS_BLOCK * block, inputW] -> [transLoop, TRANS_BLOCK * block, inputWAlign]
    __aicore__ inline void PadWCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        TransposeForPadW1(srcLocal, dstLocal);
        AscendC::PipeBarrier<PIPE_V>();
        PadW(dstLocal, srcLocal);
        AscendC::PipeBarrier<PIPE_V>();
        TransposeForPadW2(srcLocal, dstLocal);
    }

    // [tasksOnce, inputH, inputWAlign] -> [inputH16Align, tasksOnce, inputWAlign]
    __aicore__ inline void ReshapeCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams{
            static_cast<uint16_t>(inputH), static_cast<uint16_t>(inputWAlign * typeSize / BLOCK_SIZE), 0,
            static_cast<uint16_t>((inputWAlign * (tasksOnce - 1)) * typeSize / BLOCK_SIZE)};
        for (uint64_t i = 0; i < tasksOnce; ++i) {
            AscendC::DataCopy(dstLocal[i * inputWAlign], srcLocal[i * inputH * inputWAlign], copyParams);
        }
    }

    // [inputH16Align, tasksOnce, inputWAlign] -> [tasksOnce, inputWAlign, inputH16Align]
    __aicore__ inline void TransposeCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = tasksOnce * inputWAlign / block;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : inputH16Align;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < repeatH; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            for (int j = 0; j < TRANS_BLOCK; ++j) {
                uint64_t offset = i * tasksOnce * inputWAlign * TRANS_BLOCK + j * inputWAlign * tasksOnce;
                srcLocalList[j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            uint64_t dstLocalList[TRANS_BLOCK];
            for (uint64_t j = 0; j < block; ++j) {
                for (uint64_t k = 0; k < TRANS_BLOCK / block; ++k) {
                    uint64_t offset = i * TRANS_BLOCK + j * inputH16Align + k * block;
                    dstLocalList[j * (TRANS_BLOCK / block) + k] =
                        reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    // [tasksOnce, inputWAlign, inputH16Align] -> [tasksOnce, inputWAlign, inputH16Align](脏数据后移)
    __aicore__ inline void UnPadCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams{
            static_cast<uint16_t>(tasksOnce), static_cast<uint16_t>(inputW * inputH16Align * typeSize / BLOCK_SIZE),
            static_cast<uint16_t>((inputWAlign - inputW) * inputH16Align * typeSize / BLOCK_SIZE), 0};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    // [tasksOnce, inputWAlign, inputH16Align] -> [block, tasksOnce * inputWAlign / block * inputH16Align](脏数据后移)
    __aicore__ inline void UnPadHCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        if (unPadHNum * inputH16Align / TRANS_BLOCK == 1) {
            VECINToVECOUT(srcLocal, dstLocal);
        } else {
            TransposeForUnPadH1(srcLocal, dstLocal);
            AscendC::PipeBarrier<PIPE_V>();
            UnPadH(dstLocal, srcLocal);
            AscendC::PipeBarrier<PIPE_V>();
            TransposeForUnPadH2(srcLocal, dstLocal);
        }
    }

    // [tasksOnce, inputWAlign, inputH16Align] -> [block, tasksOnce * inputWAlign / block * inputH16Align]
    // -> [tasksOnce * inputWAlign / block * inputH16Align, block]
    __aicore__ inline void TransposeForUnPadH1(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = unPadHNum * inputH16Align / TRANS_BLOCK;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 2;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        uint64_t srcLocalList[TRANS_BLOCK];
        for (uint64_t i = 0; i < TRANS_BLOCK / block; ++i) {
            for (uint64_t j = 0; j < block; ++j) {
                uint64_t offset = i * block + j * unPadHNum * inputH16Align;
                srcLocalList[i * block + j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
        }
        uint64_t dstLocalList[TRANS_BLOCK];
        if constexpr (ISFP32) {
            for (uint64_t i = 0; i < TRANS_BLOCK; i += 2) { // 2 is stride
                uint64_t offset = (i / 2) * block;          // 2 is stride
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
            for (uint64_t i = 1; i < TRANS_BLOCK; i += 2) { // 2 is stride
                uint64_t offset = (i / 2 + block) * block;  // 2 is stride
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
        } else {
            transDataParams.srcRepStride = 1;
            for (uint64_t i = 0; i < TRANS_BLOCK; ++i) {
                uint64_t offset = i * block;
                dstLocalList[i] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
        }
        AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
    }

    // [tasksOnce * inputWAlign / block * inputH16Align, block]
    // -> [tasksOnce * inputWAlign / block * inputH16Align, block](脏数据后移)
    __aicore__ inline void UnPadH(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams{
            static_cast<uint16_t>(unPadHNum), static_cast<uint16_t>(block * inputH * typeSize / BLOCK_SIZE),
            static_cast<uint16_t>((inputH16Align - inputH) * block * typeSize / BLOCK_SIZE), 0};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    // [tasksOnce * inputWAlign / block * inputH16Align(脏数据后移), block]
    // -> [block, tasksOnce * inputWAlign / block * inputH16Align](脏数据后移)
    __aicore__ inline void TransposeForUnPadH2(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = unPadHNum * inputH16Align / TRANS_BLOCK;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : 2;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        uint64_t srcLocalList[TRANS_BLOCK];
        for (uint64_t i = 0; i < TRANS_BLOCK; ++i) {
            uint64_t offset = i * block;
            srcLocalList[i] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
        }

        uint64_t dstLocalList[TRANS_BLOCK];
        for (uint64_t i = 0; i < block; ++i) {
            for (uint64_t j = 0; j < TRANS_BLOCK / block; ++j) {
                uint64_t offset = i * unPadHNum * inputH16Align + block * j;
                dstLocalList[i * (TRANS_BLOCK / block) + j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
        }
        if constexpr (!ISFP32) {
            transDataParams.dstRepStride = 1;
        }
        AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
    }

    // [transLoop, TRANS_BLOCK * block, inputWAlign] -> [transLoop, block * inputWAlign, TRANS_BLOCK]
    __aicore__ inline void TransposeForPadW1(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = inputW;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < transLoop; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                uint64_t offset = i * TRANS_BLOCK * block * inputW + block * inputW * j;
                srcLocalList[j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            uint64_t dstLocalList[TRANS_BLOCK];
            for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                uint64_t offset = i * TRANS_BLOCK * block * inputW + block * j;
                dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    // [transLoop, block * inputW, TRANS_BLOCK] -> [transLoop, block * inputWAlign, TRANS_BLOCK]
    __aicore__ inline void PadW(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams{
            static_cast<uint16_t>(transLoop * block),
            static_cast<uint16_t>(inputW * TRANS_BLOCK * typeSize / BLOCK_SIZE), 0,
            static_cast<uint16_t>((inputWAlign - inputW) * TRANS_BLOCK * typeSize / BLOCK_SIZE)};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    // [transLoop, block * inputWAlign, TRANS_BLOCK] -> [transLoop, TRANS_BLOCK * block, inputWAlign]
    __aicore__ inline void TransposeForPadW2(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = inputWAlign;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : 1;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < transLoop; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            for (int j = 0; j < TRANS_BLOCK / block; ++j) {
                for (uint64_t k = 0; k < block; ++k) {
                    uint64_t offset = i * block * inputWAlign * TRANS_BLOCK + j * block + TRANS_BLOCK * k;
                    srcLocalList[j * block + k] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                }
            }
            uint64_t dstLocalList[TRANS_BLOCK];
            if constexpr (ISFP32) {
                for (uint64_t j = 0; j < TRANS_BLOCK; j += 2) { // 2 is stride
                    uint64_t offset =
                        i * block * inputWAlign * TRANS_BLOCK + (j / 2) * block * inputWAlign; // 2 is stride
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
                for (uint64_t j = 1; j < TRANS_BLOCK; j += 2) { // 2 is stride
                    uint64_t offset =
                        i * block * inputWAlign * TRANS_BLOCK + (j / 2 + block) * block * inputWAlign; // 2 is stride
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            } else {
                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) { // 2 is stride
                    uint64_t offset = i * block * inputWAlign * TRANS_BLOCK + j * block * inputWAlign;
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    __aicore__ inline void VECINToVECOUT(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams = {
            1, static_cast<uint16_t>(GetAlign(tasksOnce * inputW * inputH16Align * typeSize, BLOCK_SIZE) / BLOCK_SIZE),
            0, 0};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    __aicore__ inline uint64_t GetAlign(uint64_t len, uint64_t size)
    {
        return size == 0U ? 0 : (len + size - 1) / size * size;
    }

private:
    int32_t typeSize{0};
    int32_t block{0};
    uint64_t tasksPerCore{0};
    uint64_t tasksTail{0};
    uint64_t inputH{0};
    uint64_t inputW{0};
    uint64_t inputH16Align{0};
    uint64_t inputWAlign{0};
    uint64_t hOnce{0};
    uint64_t loop{0};
    uint64_t tail{0};
    uint64_t hOnceMax{0};
    uint64_t tasksOnceMax{0};
    uint64_t tasksOnce{0};
    uint64_t transLoop{0};
    uint64_t repeatH{0};
    uint64_t doubleBuffer{2};
    uint64_t unPadHNum{0};

    AscendC::TPipe* pipe{nullptr};
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queOut;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
};
} // namespace TransposeV2
#endif
