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
 * \file transpose102.h
 * \brief
 */

#ifndef ASCEND_TRANSPOSE102_H
#define ASCEND_TRANSPOSE102_H

#include "transpose_v2.h"

namespace TransposeV2 {
template <typename T, bool ISFP32 = false>
class Transpose102
{
public:
    __aicore__ inline Transpose102(AscendC::TPipe* p) : pipe(p)
    {}

    __aicore__ inline void Init102(__gm__ uint8_t* src, __gm__ uint8_t* dst,
                                   const TransposeV2TilingData* __restrict__ tiling_data)
    {
        InitCommon(tiling_data);
        if (subMode == 0) {
            srcGlobal.SetGlobalBuffer((__gm__ T*)src + startIdx * dim2Len * dim3Len);
            dstGlobal.SetGlobalBuffer((__gm__ T*)dst + startIdx * dim3Len);
            dim1Tasks = tasksPerCore;
            dim2Tasks = dim2Len;
        } else {
            srcGlobal.SetGlobalBuffer((__gm__ T*)src + startIdx * dim3Len);
            dstGlobal.SetGlobalBuffer((__gm__ T*)dst + startIdx * dim1Len * dim3Len);
            dim1Tasks = dim1Len;
            dim2Tasks = tasksPerCore;
        }
        dim1Once = dim1OnceMax;
        dim2Once = dim2OnceMax;
        dim1Loop = dim1Tasks / dim1OnceMax;
        dim1Tail = dim1Tasks % dim1OnceMax;
        dim2Loop = dim2Tasks / dim2OnceMax;
        dim2Tail = dim2Tasks % dim2OnceMax;
    }

    __aicore__ inline void Process102CopyMode(uint64_t dim0Idx)
    {
        if (subMode == 0) {
            ProcessSubMode0(dim0Idx);
        } else {
            ProcessSubMode1(dim0Idx);
        }
    }

    __aicore__ inline void ProcessTransMode(uint64_t dim0Idx)
    {
        dim2OnceAlign = GetAlign(dim2Once, TRANS_BLOCK);
        dim1OnceAlign = GetAlign(dim1Once, TRANS_BLOCK);
        for (uint64_t j = 0; j < dim1Loop; ++j) {
            dim2Once = dim2OnceMax;
            dim2OnceAlign = GetAlign(dim2Once, TRANS_BLOCK);
            for (uint64_t k = 0; k < dim2Loop; ++k) {
                CopyInTransMode(dim0Idx, j, k);
                ComputeTransMode();
                CopyOutTransMode(dim0Idx, j, k);
            }
            if (dim2Tail > 0) {
                dim2Once = dim2Tail;
                dim2OnceAlign = GetAlign(dim2Once, TRANS_BLOCK);
                CopyInTransMode(dim0Idx, j, dim2Loop);
                ComputeTransMode();
                CopyOutTransMode(dim0Idx, j, dim2Loop);
            }
        }
        if (dim1Tail > 0) {
            dim1Once = dim1Tail;
            dim1OnceAlign = GetAlign(dim1Once, TRANS_BLOCK);
            dim2Once = dim2OnceMax;
            dim2OnceAlign = GetAlign(dim2Once, TRANS_BLOCK);
            for (uint64_t k = 0; k < dim2Loop; ++k) {
                CopyInTransMode(dim0Idx, dim1Loop, k);
                ComputeTransMode();
                CopyOutTransMode(dim0Idx, dim1Loop, k);
            }
            if (dim2Tail > 0) {
                dim2Once = dim2Tail;
                dim2OnceAlign = GetAlign(dim2Once, TRANS_BLOCK);
                CopyInTransMode(dim0Idx, dim1Loop, dim2Loop);
                ComputeTransMode();
                CopyOutTransMode(dim0Idx, dim1Loop, dim2Loop);
            }
        }
    }

protected:
    __aicore__ inline void InitCommon(const TransposeV2TilingData* __restrict__ tiling_data)
    {
        dim1Len = tiling_data->dim1Len;
        dim2Len = tiling_data->dim2Len;
        dim3Len = tiling_data->dim3Len;
        dim3LenAlign = tiling_data->dim3LenAlign;
        tasksPerCore = tiling_data->tasksPerCore;
        tailCore = tiling_data->tailCore;
        subMode = tiling_data->subMode;
        dim1OnceMax = tiling_data->dim1OnceMax;
        dim2OnceMax = tiling_data->dim2OnceMax;
        doubleBuffer = tiling_data->doubleBuffer;

        typeSize = sizeof(T);
        block = BLOCK_SIZE / typeSize;
        uint64_t blockId = AscendC::GetBlockIdx();
        startIdx = tasksPerCore * blockId;
        if (blockId < tailCore) {
            tasksPerCore++;
            startIdx += blockId;
        } else {
            startIdx += tailCore;
        }
        pipe->InitBuffer(queIn, doubleBuffer, MAX_UB_SIZE / 2 / doubleBuffer);  // 2 is buffer num
        pipe->InitBuffer(queOut, doubleBuffer, MAX_UB_SIZE / 2 / doubleBuffer); // 2 is buffer num
    }

    /* ******************************************COPY MODE********************************************************** */
    __aicore__ inline void ProcessSubMode0(uint64_t dim0Idx)
    {
        for (uint64_t j = 0; j < dim1Loop; ++j) {
            dim2Once = dim2OnceMax;
            for (uint64_t k = 0; k < dim2Loop; ++k) {
                CopyInSubMode0(dim0Idx, j, k);
                ComputeCopyMode();
                CopyOutSubMode0(dim0Idx, j, k);
            }
            if (dim2Tail > 0) {
                dim2Once = dim2Tail;
                CopyInSubMode0(dim0Idx, j, dim2Loop);
                ComputeCopyMode();
                CopyOutSubMode0(dim0Idx, j, dim2Loop);
            }
        }
        if (dim1Tail > 0) {
            dim1Once = dim1Tail;
            dim2Once = dim2OnceMax;
            for (uint64_t k = 0; k < dim2Loop; ++k) {
                CopyInSubMode0(dim0Idx, dim1Loop, k);
                ComputeCopyMode();
                CopyOutSubMode0(dim0Idx, dim1Loop, k);
            }
            if (dim2Tail > 0) {
                dim2Once = dim2Tail;
                CopyInSubMode0(dim0Idx, dim1Loop, dim2Loop);
                ComputeCopyMode();
                CopyOutSubMode0(dim0Idx, dim1Loop, dim2Loop);
            }
        }
    }

    __aicore__ inline void ProcessSubMode1(uint64_t dim0Idx)
    {
        for (uint64_t j = 0; j < dim2Loop; ++j) {
            dim1Once = dim1OnceMax;
            for (uint64_t k = 0; k < dim1Loop; ++k) {
                CopyInSubMode1(dim0Idx, j, k);
                ComputeCopyMode();
                CopyOutSubMode1(dim0Idx, j, k);
            }
            if (dim1Tail > 0) {
                dim1Once = dim1Tail;
                CopyInSubMode1(dim0Idx, j, dim1Loop);
                ComputeCopyMode();
                CopyOutSubMode1(dim0Idx, j, dim1Loop);
            }
        }
        if (dim2Tail > 0) {
            dim2Once = dim2Tail;
            dim1Once = dim1OnceMax;
            for (uint64_t k = 0; k < dim1Loop; ++k) {
                CopyInSubMode1(dim0Idx, dim2Loop, k);
                ComputeCopyMode();
                CopyOutSubMode1(dim0Idx, dim2Loop, k);
            }
            if (dim1Tail > 0) {
                dim1Once = dim1Tail;
                CopyInSubMode1(dim0Idx, dim2Loop, dim1Loop);
                ComputeCopyMode();
                CopyOutSubMode1(dim0Idx, dim2Loop, dim1Loop);
            }
        }
    }

    __aicore__ inline void CopyInSubMode0(uint64_t dim0Idx, uint64_t taskIdx, uint64_t blockIdx)
    {
        AscendC::LocalTensor<T> srcLocal = queIn.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(dim1Once * dim2Once), static_cast<uint32_t>(dim3Len * typeSize), 0, 0, 0};
        if (dim3Len == dim3LenAlign) {
            copyParams = {1, static_cast<uint32_t>(dim1Once * dim2Once * dim3Len * typeSize), 0, 0, 0};
        }
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint64_t offset = (dim0Idx * dim1Len * dim2Len * dim3Len) + (taskIdx * dim1OnceMax * dim2Len * dim3Len) +
                          (blockIdx * dim2OnceMax * dim3Len);
        AscendC::DataCopyPad(srcLocal, srcGlobal[offset], copyParams, padParams);
        queIn.EnQue(srcLocal);
    }

    __aicore__ inline void ComputeCopyMode()
    {
        AscendC::LocalTensor<T> srcLocal = queIn.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = queOut.AllocTensor<T>();
        if (dim2Once >= dim1Once) {
            AscendC::DataCopyParams copyParams{
                static_cast<uint16_t>(dim2Once), static_cast<uint16_t>(dim3LenAlign * typeSize / BLOCK_SIZE), 0,
                static_cast<uint16_t>((dim1Once - 1) * dim3LenAlign * typeSize / BLOCK_SIZE)};
            for (uint64_t i = 0; i < dim1Once; ++i) {
                uint64_t offsetSrc = i * dim2Once * dim3LenAlign;
                uint64_t offsetDst = i * dim3LenAlign;
                AscendC::DataCopy(dstLocal[offsetDst], srcLocal[offsetSrc], copyParams);
            }
        } else {
            AscendC::DataCopyParams copyParams{
                static_cast<uint16_t>(dim1Once), static_cast<uint16_t>(dim3LenAlign * typeSize / BLOCK_SIZE),
                static_cast<uint16_t>((dim2Once - 1) * dim3LenAlign * typeSize / BLOCK_SIZE), 0};
            for (uint64_t i = 0; i < dim2Once; ++i) {
                uint64_t offsetSrc = i * dim3LenAlign;
                uint64_t offsetDst = i * dim1Once * dim3LenAlign;
                AscendC::DataCopy(dstLocal[offsetDst], srcLocal[offsetSrc], copyParams);
            }
        }
        queOut.EnQue(dstLocal);
        queIn.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOutSubMode0(uint64_t dim0Idx, uint64_t taskIdx, uint64_t blockIdx)
    {
        AscendC::LocalTensor<T> dstLocal = queOut.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(dim2Once), static_cast<uint32_t>(dim1Once * dim3Len * typeSize), 0,
            static_cast<uint32_t>((dim1Len - dim1Once) * dim3Len * typeSize), 0};
        uint64_t offset = (dim0Idx * dim1Len * dim2Len * dim3Len) + (taskIdx * dim1OnceMax * dim3Len) +
                          (blockIdx * dim2OnceMax * dim1Len * dim3Len);
        AscendC::DataCopyPad(dstGlobal[offset], dstLocal, copyParams);
        queOut.FreeTensor(dstLocal);
    }

    __aicore__ inline void CopyInSubMode1(uint64_t dim0Idx, uint64_t taskIdx, uint64_t blockIdx)
    {
        AscendC::LocalTensor<T> srcLocal = queIn.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(dim1Once), static_cast<uint32_t>(dim2Once * dim3Len * typeSize),
            static_cast<uint32_t>((dim2Len - dim2Once) * dim3Len * typeSize), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint64_t offset = (dim0Idx * dim1Len * dim2Len * dim3Len) + (taskIdx * dim2OnceMax * dim3Len) +
                          (blockIdx * dim1OnceMax * dim2Len * dim3Len);
        AscendC::DataCopyPad(srcLocal, srcGlobal[offset], copyParams, padParams);
        queIn.EnQue(srcLocal);
    }

    __aicore__ inline void CopyOutSubMode1(uint64_t dim0Idx, uint64_t taskIdx, uint64_t blockIdx)
    {
        AscendC::LocalTensor<T> dstLocal = queOut.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(dim1Once * dim2Once), static_cast<uint32_t>(dim3Len * typeSize), 0, 0, 0};
        if (dim3Len == dim3LenAlign) {
            copyParams = {1, static_cast<uint32_t>(dim1Once * dim2Once * dim3Len * typeSize), 0, 0, 0};
        }
        uint64_t offset = (dim0Idx * dim1Len * dim2Len * dim3Len) + (taskIdx * dim2OnceMax * dim1Len * dim3Len) +
                          (blockIdx * dim1OnceMax * dim3Len);
        AscendC::DataCopyPad(dstGlobal[offset], dstLocal, copyParams);
        queOut.FreeTensor(dstLocal);
    }

    /* ******************************************TRANS MODE********************************************************* */
    // [dim1Once, dim2Once, dim3Len]
    __aicore__ inline void CopyInTransMode(uint64_t dim0Idx, uint64_t dim1Idx, uint64_t dim2Idx)
    {
        AscendC::LocalTensor<T> srcLocal = queIn.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(dim1Once), static_cast<uint32_t>(dim2Once * dim3Len * typeSize),
            static_cast<uint32_t>((dim2Len - dim2Once) * dim3Len * typeSize), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        uint64_t offset = (dim1Idx * dim1OnceMax * dim2Len * dim3Len) + (dim2Idx * dim2OnceMax * dim3Len);
        AscendC::DataCopyPad(srcLocal, srcGlobal[offset], copyParams, padParams);
        queIn.EnQue(srcLocal);
    }

    // [dim1OnceAlign, dim2Once, dim3Len] -> [dim2Once, dim1OnceAlign * dim3LenAlign]
    __aicore__ inline void ComputeTransMode()
    {
        AscendC::LocalTensor<T> srcLocal = queIn.DeQue<T>();
        AscendC::LocalTensor<T> dstLocal = queOut.AllocTensor<T>();

        TransposeCompute1(srcLocal, dstLocal);
        AscendC::PipeBarrier<PIPE_V>();
        TransposeCompute2(dstLocal, srcLocal);
        AscendC::PipeBarrier<PIPE_V>();
        TransposeCompute3(srcLocal, dstLocal);
        AscendC::PipeBarrier<PIPE_V>();
        UnpadCompute(dstLocal, srcLocal);
        AscendC::PipeBarrier<PIPE_V>();
        if (dim1Once != 1 && dim3Len != dim3LenAlign) {
            UnPadHCompute(srcLocal, dstLocal);
        } else {
            VECINToVECOUT(srcLocal, dstLocal);
        }
        queOut.EnQue(dstLocal);
        queIn.FreeTensor(srcLocal);
    }

    // [dim2Once, dim1Once, dim3LenAlign]
    __aicore__ inline void CopyOutTransMode(uint64_t dim0Idx, uint64_t dim1Idx, uint64_t dim2Idx)
    {
        AscendC::LocalTensor<T> dstLocal = queOut.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(dim2Once), static_cast<uint32_t>(dim1Once * dim3Len * typeSize),
            static_cast<uint32_t>(dim1Once * (dim3LenAlign - dim3Len) / block),
            static_cast<uint32_t>((dim1Len - dim1Once) * dim3Len * typeSize), 0};
        uint64_t offset = (dim1Idx * dim1OnceMax * dim3Len) + (dim2Idx * dim2OnceMax * dim1Len * dim3Len);
        AscendC::DataCopyPad(dstGlobal[offset], dstLocal, copyParams);
        queOut.FreeTensor(dstLocal);
    }

    // [dim1OnceAlign, (dim2Once * dim3Len)Align] -> [(dim2Once * dim3Len)Align, dim1OnceAlign]
    __aicore__ inline void TransposeCompute1(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint64_t transNum = GetAlign(dim2Once * dim3Len, block);
        uint8_t repeatTimes = transNum / block;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : dim1OnceAlign;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < dim1OnceAlign / TRANS_BLOCK; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                uint64_t offset = i * TRANS_BLOCK * transNum + j * transNum;
                srcLocalList[j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            uint64_t dstLocalList[TRANS_BLOCK];
            if constexpr (ISFP32) {
                for (uint64_t j = 0; j < block; ++j) {
                    for (uint64_t k = 0; k < 2; ++k) { // 2 is stride
                        uint64_t offset = i * TRANS_BLOCK + j * dim1OnceAlign + k * block;
                        // 2 is stride
                        dstLocalList[j * 2 + k] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                    }
                }
            } else {
                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                    uint64_t offset = i * TRANS_BLOCK + j * dim1OnceAlign;
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    // [dim2Once, dim3Len, dim1OnceAlign] -> [dim3Len, dim2Once, dim1OnceAlign]
    __aicore__ inline void TransposeCompute2(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        // [64*8, 16]  -> [16(8), 64, 16]
        AscendC::DataCopyParams copyParams = {
            static_cast<uint16_t>(dim2Once), static_cast<uint16_t>(dim1OnceAlign / block),
            static_cast<uint16_t>((dim3Len - 1) * dim1OnceAlign / block), 0};
        for (uint64_t i = 0; i < dim3Len; ++i) {
            AscendC::DataCopy(dstLocal[i * dim2Once * dim1OnceAlign], srcLocal[i * dim1OnceAlign], copyParams);
        }
    }

    // [dim3LenAlign, dim2Once, dim1OnceAlign] -> [dim2Once * dim1OnceAlign, dim3LenAlign]
    __aicore__ inline void TransposeCompute3(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        // [16(8), 64, 16]  -> [64, 16, 16(8)]
        uint8_t repeatTimes = dim2Once * dim1OnceAlign / TRANS_BLOCK;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : dim3LenAlign * (TRANS_BLOCK / block);
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : TRANS_BLOCK / block;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < dim3LenAlign / block; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            uint64_t dstLocalList[TRANS_BLOCK];
            if constexpr (ISFP32) {
                // srcLocalList
                for (int j = 0; j < 2; ++j) { // 2 is stride
                    for (int64_t k = 0; k < block; ++k) {
                        uint64_t offset =
                            i * block * dim2Once * dim1OnceAlign + j * block + k * dim2Once * dim1OnceAlign;
                        srcLocalList[j * block + k] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                    }
                }
                // dstLocalList
                for (int64_t j = 0; j < TRANS_BLOCK; j += 2) {            // 2 is stride
                    uint64_t offset = i * block + (j / 2) * dim3LenAlign; // 2 is stride
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
                for (int64_t j = 1; j < TRANS_BLOCK; j += 2) {                    // 2 is stride
                    uint64_t offset = i * block + (j / 2 + block) * dim3LenAlign; // 2 is stride
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            } else {
                // srcLocalList
                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                    uint64_t offset = i * block * dim2Once * dim1OnceAlign + j * dim2Once * dim1OnceAlign;
                    srcLocalList[j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                }
                // dstLocalList
                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                    uint64_t offset = i * TRANS_BLOCK + j * dim3LenAlign;
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    // [dim2Once, dim1OnceAlign, dim3LenAlign] -> [dim2Once, dim1Once, dim3LenAlign]
    __aicore__ inline void UnpadCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams = {
            static_cast<uint16_t>(dim2Once), static_cast<uint16_t>(dim1Once * dim3LenAlign / block),
            static_cast<uint16_t>((dim1OnceAlign - dim1Once) * dim3LenAlign / block), 0};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    // [dim2Once, dim1Once, dim3LenAlign] -> [dim2Once, dim1Once * dim3LenAlign](脏数据后移)
    __aicore__ inline void UnPadHCompute(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        TransposeForUnPadH1(srcLocal, dstLocal);
        AscendC::PipeBarrier<PIPE_V>();
        UnPadH(dstLocal, srcLocal);
        AscendC::PipeBarrier<PIPE_V>();
        TransposeForUnPadH2(srcLocal, dstLocal);
    }

    // [dim2OnceAlign, dim1Once, dim3LenAlign] -> [dim1Once * dim3LenAlign, dim2OnceAlign]
    __aicore__ inline void TransposeForUnPadH1(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = dim1Once * dim3LenAlign / block;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : dim2OnceAlign;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : 1;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < dim2OnceAlign / TRANS_BLOCK; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                uint64_t offset = i * TRANS_BLOCK * dim1Once * dim3LenAlign + j * dim1Once * dim3LenAlign;
                srcLocalList[j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
            }
            uint64_t dstLocalList[TRANS_BLOCK];
            if constexpr (ISFP32) {
                for (uint64_t j = 0; j < block; ++j) {
                    for (uint64_t k = 0; k < 2; ++k) { // 2 is stride
                        uint64_t offset = i * TRANS_BLOCK + j * dim2OnceAlign + k * block;
                        // 2 is stride
                        dstLocalList[j * 2 + k] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                    }
                }
            } else {
                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                    uint64_t offset = i * TRANS_BLOCK + j * dim2OnceAlign;
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    // [dim1Once * dim3LenAlign, dim2OnceAlign] -> [dim1Once * dim3LenAlign, dim2OnceAlign](脏数据后移)
    __aicore__ inline void UnPadH(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams = {
            static_cast<uint16_t>(dim1Once), static_cast<uint16_t>(dim3Len * dim2OnceAlign / block),
            static_cast<uint16_t>((dim3LenAlign - dim3Len) * dim2OnceAlign / block), 0};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    // [dim1Once * dim3LenAlign, dim2OnceAlign](脏数据后移) -> [dim2OnceAlign, dim1Once * dim3LenAlign](脏数据后移)
    __aicore__ inline void TransposeForUnPadH2(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        uint8_t repeatTimes = dim1Once * dim3LenAlign / block;
        uint16_t dstRepStride = repeatTimes == 1 ? 0 : 1;
        uint16_t srcRepStride = repeatTimes == 1 ? 0 : dim2OnceAlign;
        AscendC::TransDataTo5HDParams transDataParams{false, false, repeatTimes, dstRepStride, srcRepStride};
        for (uint64_t i = 0; i < dim2OnceAlign / TRANS_BLOCK; ++i) {
            uint64_t srcLocalList[TRANS_BLOCK];
            uint64_t dstLocalList[TRANS_BLOCK];
            if constexpr (ISFP32) {
                // srcLocalList
                for (int j = 0; j < 2; ++j) { // 2 is stride
                    for (int64_t k = 0; k < block; ++k) {
                        uint64_t offset = i * TRANS_BLOCK + j * block + k * dim2OnceAlign;
                        srcLocalList[j * block + k] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                    }
                }
                // dstLocalList
                for (int64_t j = 0; j < TRANS_BLOCK; j += 2) { // 2 is stride
                    uint64_t offset =
                        i * TRANS_BLOCK * dim1Once * dim3LenAlign + (j / 2) * dim1Once * dim3LenAlign; // 2 is stride
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
                for (int64_t j = 1; j < TRANS_BLOCK; j += 2) { // 2 is stride
                    uint64_t offset = i * TRANS_BLOCK * dim1Once * dim3LenAlign +
                                      (j / 2 + block) * dim1Once * dim3LenAlign; // 2 is stride
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            } else {
                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                    uint64_t offset = i * TRANS_BLOCK + j * dim2OnceAlign;
                    srcLocalList[j] = reinterpret_cast<uint64_t>(srcLocal[offset].GetPhyAddr());
                }

                for (uint64_t j = 0; j < TRANS_BLOCK; ++j) {
                    uint64_t offset = i * TRANS_BLOCK * dim1Once * dim3LenAlign + j * dim1Once * dim3LenAlign;
                    dstLocalList[j] = reinterpret_cast<uint64_t>(dstLocal[offset].GetPhyAddr());
                }
            }
            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    // [dim2Once, dim1Once, dim3LenAlign] -> [dim2Once, dim1Once, dim3LenAlign]
    __aicore__ inline void VECINToVECOUT(AscendC::LocalTensor<T>& srcLocal, AscendC::LocalTensor<T>& dstLocal)
    {
        AscendC::DataCopyParams copyParams = {
            1, static_cast<uint16_t>(dim2Once * dim1Once * dim3LenAlign / block), 0, 0};
        AscendC::DataCopy(dstLocal, srcLocal, copyParams);
    }

    __aicore__ inline uint64_t GetAlign(uint64_t len, uint64_t size)
    {
        return size == 0U ? 0 : (len + size - 1) / size * size;
    }

protected:
    int32_t typeSize{0};
    int32_t block{0};
    uint64_t startIdx{0};

    uint64_t dim1Len{0};
    uint64_t dim2Len{0};
    uint64_t dim3Len{0};
    uint64_t dim3LenAlign{0};
    uint64_t tasksPerCore{0};
    uint64_t tailCore{0};
    uint64_t subMode{0};
    uint64_t dim1Once{0};
    uint64_t dim1OnceMax{0};
    uint64_t dim1OnceAlign{0};
    uint64_t dim2Once{0};
    uint64_t dim2OnceMax{0};
    uint64_t dim2OnceAlign{0};
    uint64_t dim1Loop{0};
    uint64_t dim1Tail{0};
    uint64_t dim2Loop{0};
    uint64_t dim2Tail{0};
    uint64_t dim1Tasks{0};
    uint64_t dim2Tasks{0};
    uint64_t doubleBuffer{2};

    AscendC::TPipe* pipe{nullptr};
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> queOut;
    AscendC::GlobalTensor<T> srcGlobal;
    AscendC::GlobalTensor<T> dstGlobal;
};
} // namespace TransposeV2
#endif
