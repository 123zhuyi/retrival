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
 * \file transpose_transdata_5hd.h
 * \brief TilingKey=10007 VCONV_TRANSPOSE 策略实现
 *
 * 适用场景：DAV_5102(Ascend950) + 2D perm=[1,0] + 16bit 数据类型(fp16/bf16) + shape[0]>5
 *
 * 核心机制：使用 TransDataTo5HD 硬件指令实现2D矩阵转置
 *   TransDataTo5HD 以16行为单位进行行列互换，每次处理16×16 的数据块。
 *
 * R/C 维切分模式：
 *   - IsRSplit：按行（R维）切分 → 每个核负责部分行
 *   - IsRCSplit：同时按行列切分 → 每个核负责部分行+部分列
 *   - 否则：按列（C维）切分 → 每个核负责部分列
 *
 * 数据流：
 *   1. BaseCopyIn: DataCopyPad(GM → UB)
 *   2. BaseCompute:
 *      - r >= c → ComputeRConv: 以行为主序遍历列，TransDataTo5HD按16行一组
 *      - r < c → ComputeCConv: 以列为主序遍历行
 *   3. BaseCopyOut: DataCopyPad(UB → GM)
 *
 * TransDataTo5HD 参数说明：
 *   - repeatTimes = r/16（或 c/16）：重复次数
 *   - srcRepStride：源重复步长（每处理完16行后，源地址的偏移）
 *   - dstRepStride：目标重复步长（每处理完16行后，目标地址的偏移）
 *   - srcList/dstList：16个 LocalTensor 地址列表，硬件按列表顺序读取/写入
 *
 * 仅支持 int16_t 模板参数（对应 fp16/bf16 的 sizeof==2）
 */

#ifndef KERNEL_TRANSPOSE_TRANSDATA_5HD_H_
#define KERNEL_TRANSPOSE_TRANSDATA_5HD_H_

#include <type_traits>
#include "op_kernel/platform_util.h"
#include "transpose_base.h"

namespace Transpose {
using namespace AscendC;
static constexpr int64_t TRANSELEM = 16; ///< TransDataTo5HD 每次处理的元素数（16×16块转置）

/**
 * @brief VCONV_TRANSPOSE 策略类，使用 TransDataTo5HD 实现2D矩阵转置
 *
 * 仅实例化 int16_t 模板（对应 fp16/bf16）。TransDataTo5HD 硬件指令
 * 以16行为单位进行行列互换，是2D转置的高效硬件加速方案。
 *
 * @tparam T 数据元素类型（实际仅使用 int16_t）
 */
template <typename T>
class KernelTransDataTo5HD {
public:
    __aicore__ inline KernelTransDataTo5HD(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeVCONVTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetCopyInParams(uint32_t process);
    __aicore__ inline void SetRCSplitCopyInParams(uint32_t process);
    __aicore__ inline void SetRSplitCopyInParams(uint32_t process);
    __aicore__ inline void SetCSplitCopyInParams(uint32_t process);
    __aicore__ inline void BaseCopyIn(uint32_t process);
    __aicore__ inline void SetBasePocessData(uint32_t process, bool rSplit, bool rcSplit);
    __aicore__ inline void BaseProcess(uint32_t process, uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign);
    __aicore__ inline void BaseCompute(uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign);
    __aicore__ inline void ComputeRConv(int r, int c, int cAlign);
    __aicore__ inline void ComputeCConv(int r, int c, int rAlign);
    __aicore__ inline void SetCopyOutParams(uint32_t process);
    __aicore__ inline void SetRCSplitCopyOutParams(uint32_t process);
    __aicore__ inline void SetRSplitCopyOutParams(uint32_t process);
    __aicore__ inline void SetCSplitCopyOutParams(uint32_t process);
    __aicore__ inline void BaseCopyOut(uint32_t process);

private:
    const TransposeVCONVTilingData* tiling_ = nullptr;
    TQue<TPosition::VECIN, 1> inQueueSrc;
    TQue<TPosition::VECOUT, 1> outQueueDst;
    GlobalTensor<T> srcGlobal, dstGlobal;
    int64_t fullCoreNum = 0;
    int64_t blockIdx_ = 0;
    uint64_t loopCount = 0;
    int64_t copyInCoreOffset = 0;
    int64_t copyOutCoreOffset = 0;
    int64_t gmOffset = 0;
    int64_t ubOffset = 0;
    bool isRsplit = false;
    bool isRCSplit = false;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams Params;
};

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeVCONVTilingData* tilingData,
                                                     TPipe* pipe)
{
    tiling_ = tilingData;
    blockIdx_ = GetBlockIdx();
    srcGlobal.SetGlobalBuffer((__gm__ T*)x);
    dstGlobal.SetGlobalBuffer((__gm__ T*)y);
    fullCoreNum = tiling_->UsedCoreNum - 1;
    loopCount = (blockIdx_ <= fullCoreNum - 1) ? tiling_->MainCoreLoopCount : tiling_->TailCoreLoopCount;
    isRsplit = tiling_->IsRSplit;
    isRCSplit = tiling_->IsRCSplit;
    pipe->InitBuffer(inQueueSrc, BUFFER_NUM, tiling_->AvailableUbSize);
    pipe->InitBuffer(outQueueDst, BUFFER_NUM, tiling_->AvailableUbSize);
}

/**
 * @brief R维主序计算：以行为主序遍历列，TransDataTo5HD按16行一组转置
 *
 * 当 r >= c 时选择此路径。每次 TransDataTo5HD 处理16行×c列的数据，
 * repeatTimes = r/16 表示共需要重复 r/16 次处理。
 *
 * @param r       实际行数（可能不等于 rAlign）
 * @param c       实际列数
 * @param cAlign  列数对齐值（用于 srcRepStride 计算）
 */
template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::ComputeRConv(int r, int c, int cAlign)
{
    LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = r / TRANSELEM;
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : cAlign * TRANSELEM;

    // TransDataTo5HD times
    for (int j = 0; j < cAlign; j++) {
        LocalTensor<T> srcLocalList[16];
        for (int i = 0; i < 16; i++) {
            srcLocalList[i] = srcLocal[i * c + j * TRANSELEM];
        }

        LocalTensor<T> dstLocalList[16];
        for (int i = 0; i < 16; i++) {
            dstLocalList[i] = dstLocal[i * r + j * TRANSELEM * r];
        }
        TransDataTo5HD(dstLocalList, srcLocalList, transDataParams);
    }
    outQueueDst.EnQue<T>(dstLocal);
    inQueueSrc.FreeTensor(srcLocal);
}

/**
 * @brief C维主序计算：以列为主序遍历行，TransDataTo5HD按16列一组转置
 *
 * 当 r < c 时选择此路径。每次 TransDataTo5HD 处理c列×16行的数据，
 * repeatTimes = c/16 表示共需要重复 c/16 次处理。
 *
 * @param r       实际行数
 * @param c       实际列数（可能不等于 cAlign）
 * @param rAlign  行数对齐值（用于 dstRepStride 计算）
 */
template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::ComputeCConv(int r, int c, int rAlign)
{
    LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
    LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();

    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = c / TRANSELEM;
    transDataParams.dstRepStride = transDataParams.repeatTimes == 1 ? 0 : rAlign * TRANSELEM;
    transDataParams.srcRepStride = transDataParams.repeatTimes == 1 ? 0 : 1;

    // TransDataTo5HD times
    for (int j = 0; j < rAlign; j++) {
        LocalTensor<T> srcLocalList[16];
        for (int i = 0; i < 16; i++) {
            srcLocalList[i] = srcLocal[i * c + j * TRANSELEM * c];
        }

        LocalTensor<T> dstLocalList[16];
        for (int i = 0; i < 16; i++) {
            dstLocalList[i] = dstLocal[i * r + j * TRANSELEM];
        }
        TransDataTo5HD(dstLocalList, srcLocalList, transDataParams);
    }
    outQueueDst.EnQue<T>(dstLocal);
    inQueueSrc.FreeTensor(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::Process()
{
    if (blockIdx_ >= tiling_->UsedCoreNum) {
        return;
    }
    if (blockIdx_ < fullCoreNum) {
        for (int i = 0; i < loopCount; i++) {
            if (i < loopCount - 1) {
                SetBasePocessData(i, isRsplit, isRCSplit);
                BaseProcess(i, tiling_->rUbSplitPara.MainCoreUbFactor, tiling_->cUbSplitPara.MainCoreUbFactor,
                            tiling_->rUbSplitPara.MainCoreUbAlignFactor, tiling_->cUbSplitPara.MainCoreUbAlignFactor);
            } else {
                SetBasePocessData(i, isRsplit, isRCSplit);
                BaseProcess(i, tiling_->rUbSplitPara.MainCoreTailUbFactor, tiling_->cUbSplitPara.MainCoreTailUbFactor,
                            tiling_->rUbSplitPara.MainCoreTailUbAlignFactor,
                            tiling_->cUbSplitPara.MainCoreTailUbAlignFactor);
            }
        }
    } else {
        for (int i = 0; i < loopCount; i++) {
            if (i < loopCount - 1) {
                SetBasePocessData(i, isRsplit, isRCSplit);
                BaseProcess(i, tiling_->rUbSplitPara.TailCoreUbFactor, tiling_->cUbSplitPara.TailCoreUbFactor,
                            tiling_->rUbSplitPara.TailCoreUbAlignFactor, tiling_->cUbSplitPara.TailCoreUbAlignFactor);
            } else {
                SetBasePocessData(i, isRsplit, isRCSplit);
                BaseProcess(i, tiling_->rUbSplitPara.TailCoreTailUbFactor, tiling_->cUbSplitPara.TailCoreTailUbFactor,
                            tiling_->rUbSplitPara.TailCoreTailUbAlignFactor,
                            tiling_->cUbSplitPara.TailCoreTailUbAlignFactor);
            }
        }
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetBasePocessData(uint32_t process, bool rSplit, bool rcSplit)
{
    if (rSplit) {
        if (rcSplit) {
            copyInCoreOffset = blockIdx_ * tiling_->CLen * tiling_->rSplitPara.BlockFactor;
            gmOffset = process * tiling_->cUbSplitPara.MainCoreUbFactor;
            copyOutCoreOffset = blockIdx_ * tiling_->rUbSplitPara.MainCoreUbFactor;
            ubOffset = process * tiling_->RLen * tiling_->cUbSplitPara.MainCoreUbFactor;
        } else {
            copyInCoreOffset = blockIdx_ * tiling_->CLen * tiling_->rSplitPara.BlockFactor;
            gmOffset = process * tiling_->rUbSplitPara.MainCoreUbFactor * tiling_->CLen;
            copyOutCoreOffset = blockIdx_ * tiling_->rSplitPara.BlockFactor;
            ubOffset = process * tiling_->rUbSplitPara.MainCoreUbFactor;
        }
    } else {
        copyInCoreOffset = blockIdx_ * tiling_->cSplitPara.BlockFactor;
        gmOffset = process * tiling_->cUbSplitPara.MainCoreUbFactor;
        copyOutCoreOffset = blockIdx_ * tiling_->RLen * tiling_->cSplitPara.BlockFactor;
        ubOffset = process * tiling_->RLen * tiling_->cUbSplitPara.MainCoreUbFactor;
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::BaseProcess(uint32_t process, uint32_t r, uint32_t c, uint32_t rAlign,
                                                            uint32_t cAlign)
{
    BaseCopyIn(process);
    BaseCompute(r, c, rAlign, cAlign);
    BaseCopyOut(process);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::BaseCopyIn(uint32_t process)
{
    LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
    SetCopyInParams(process);
    DataCopyPad(srcLocal, srcGlobal[gmOffset + copyInCoreOffset], Params, padParams);
    inQueueSrc.EnQue(srcLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::BaseCompute(uint32_t r, uint32_t c, uint32_t rAlign, uint32_t cAlign)
{
    if (r >= c) {
        ComputeRConv(r, c, cAlign);
    } else {
        ComputeCConv(r, c, rAlign);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::BaseCopyOut(uint32_t process)
{
    LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
    SetCopyOutParams(process);
    DataCopyPad(dstGlobal[copyOutCoreOffset + ubOffset], dstLocal, Params);
    outQueueDst.FreeTensor(dstLocal);
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetCopyInParams(uint32_t process)
{
    if (isRCSplit) {
        SetRCSplitCopyInParams(process);
    } else if (isRsplit) {
        SetRSplitCopyInParams(process);
    } else {
        SetCSplitCopyInParams(process);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetRCSplitCopyInParams(uint32_t process)
{
    if (blockIdx_ < fullCoreNum) {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->rUbSplitPara.MainCoreUbFactor;
            Params.blockLen = tiling_->cUbSplitPara.MainCoreUbFactor * sizeof(T);
            Params.srcStride = (tiling_->CLen - tiling_->cUbSplitPara.MainCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = tiling_->rUbSplitPara.MainCoreTailUbFactor;
            Params.blockLen = (tiling_->CLen - process * tiling_->cUbSplitPara.MainCoreUbFactor) * sizeof(T);
            Params.srcStride = (process * tiling_->cUbSplitPara.MainCoreUbFactor) * sizeof(T);
        }
    } else {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->RLen - blockIdx_ * tiling_->rSplitPara.BlockFactor;
            Params.blockLen = tiling_->cUbSplitPara.TailCoreUbFactor * sizeof(T);
            Params.srcStride = (tiling_->CLen - tiling_->cUbSplitPara.TailCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = tiling_->RLen - blockIdx_ * tiling_->rSplitPara.BlockFactor;
            Params.blockLen = (tiling_->CLen - process * tiling_->cUbSplitPara.TailCoreUbFactor) * sizeof(T);
            Params.srcStride = (process * tiling_->cUbSplitPara.TailCoreUbFactor) * sizeof(T);
        }
    }
    Params.dstStride = 0;
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetRSplitCopyInParams(uint32_t process)
{
    if (tiling_->CLen % TRANSELEM == 0) {
        Params.blockCount = 1;
        Params.srcStride = 0;
        Params.dstStride = 0;
        if (blockIdx_ < fullCoreNum) {
            if (process < loopCount - 1) {
                Params.blockLen = tiling_->rUbSplitPara.MainCoreUbFactor * tiling_->cUbSplitPara.MainCoreUbFactor *
                                  sizeof(T);
            } else {
                Params.blockLen = tiling_->rUbSplitPara.MainCoreTailUbFactor *
                                  tiling_->cUbSplitPara.MainCoreTailUbFactor * sizeof(T);
            }
        } else {
            if (process < loopCount - 1) {
                Params.blockLen = tiling_->rUbSplitPara.TailCoreUbFactor * tiling_->cUbSplitPara.TailCoreUbFactor *
                                  sizeof(T);
            } else {
                Params.blockLen = (tiling_->RLen - (blockIdx_ * tiling_->rSplitPara.BlockFactor +
                                                    process * tiling_->rUbSplitPara.TailCoreUbFactor)) *
                                  tiling_->cUbSplitPara.TailCoreTailUbFactor * sizeof(T);
            }
        }
    } else {
        if (blockIdx_ < fullCoreNum) {
            if (process < loopCount - 1) {
                Params.blockCount = tiling_->rUbSplitPara.MainCoreUbFactor;
            } else {
                Params.blockCount = tiling_->rUbSplitPara.MainCoreTailUbFactor;
            }
        } else {
            if (process < loopCount - 1) {
                Params.blockCount = tiling_->rUbSplitPara.TailCoreUbFactor;
            } else {
                Params.blockCount = tiling_->RLen - (blockIdx_ * tiling_->rSplitPara.BlockFactor +
                                                     process * tiling_->rUbSplitPara.TailCoreUbFactor);
            }
        }
        Params.blockLen = tiling_->CLen * sizeof(T);
        Params.srcStride = 0;
        Params.dstStride = 0;
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetCSplitCopyInParams(uint32_t process)
{
    if (blockIdx_ < fullCoreNum) {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->RLen;
            Params.blockLen = tiling_->cUbSplitPara.MainCoreUbFactor * sizeof(T);
            Params.srcStride = (tiling_->CLen - tiling_->cUbSplitPara.MainCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = tiling_->RLen;
            Params.blockLen = tiling_->cUbSplitPara.MainCoreTailUbFactor * sizeof(T);
            Params.srcStride = (tiling_->CLen - tiling_->cUbSplitPara.MainCoreTailUbFactor) * sizeof(T);
        }
    } else {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->RLen;
            Params.blockLen = tiling_->cUbSplitPara.TailCoreUbFactor * sizeof(T);
            Params.srcStride = (tiling_->CLen - tiling_->cUbSplitPara.TailCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = tiling_->RLen;
            Params.blockLen = (tiling_->CLen - blockIdx_ * tiling_->cSplitPara.BlockFactor -
                               process * tiling_->cUbSplitPara.TailCoreUbFactor) *
                              sizeof(T);
            Params.srcStride = (blockIdx_ * tiling_->cSplitPara.BlockFactor +
                                process * tiling_->cUbSplitPara.TailCoreUbFactor) *
                               sizeof(T);
        }
    }
    Params.dstStride = 0;
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetCopyOutParams(uint32_t process)
{
    if (isRCSplit) {
        SetRCSplitCopyOutParams(process);
    } else if (isRsplit) {
        SetRSplitCopyOutParams(process);
    } else {
        SetCSplitCopyOutParams(process);
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetRCSplitCopyOutParams(uint32_t process)
{
    if (blockIdx_ < fullCoreNum) {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->cUbSplitPara.MainCoreUbFactor;
            Params.blockLen = tiling_->rUbSplitPara.MainCoreUbFactor * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (tiling_->RLen - tiling_->rUbSplitPara.MainCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = (tiling_->CLen - process * tiling_->cUbSplitPara.MainCoreUbFactor);
            Params.blockLen = tiling_->rUbSplitPara.MainCoreTailUbFactor * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (tiling_->RLen - tiling_->rUbSplitPara.MainCoreTailUbFactor) * sizeof(T);
        }
    } else {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->cUbSplitPara.TailCoreUbFactor;
            Params.blockLen = (tiling_->RLen - blockIdx_ * tiling_->rSplitPara.BlockFactor) * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (blockIdx_ * tiling_->rSplitPara.BlockFactor) * sizeof(T);
        } else {
            Params.blockCount = (tiling_->CLen - process * tiling_->cUbSplitPara.TailCoreUbFactor);
            Params.blockLen = (tiling_->RLen - blockIdx_ * tiling_->rSplitPara.BlockFactor) * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (blockIdx_ * tiling_->rSplitPara.BlockFactor) * sizeof(T);
        }
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetRSplitCopyOutParams(uint32_t process)
{
    if (blockIdx_ < fullCoreNum) {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->CLen;
            Params.blockLen = tiling_->rUbSplitPara.MainCoreUbFactor * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (tiling_->RLen - tiling_->rUbSplitPara.MainCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = tiling_->CLen;
            Params.blockLen = tiling_->rUbSplitPara.MainCoreTailUbFactor * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (tiling_->RLen - tiling_->rUbSplitPara.MainCoreTailUbFactor) * sizeof(T);
        }
    } else {
        if (process < loopCount - 1) {
            Params.blockCount = tiling_->CLen;
            Params.blockLen = tiling_->rUbSplitPara.TailCoreUbFactor * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (tiling_->RLen - tiling_->rUbSplitPara.TailCoreUbFactor) * sizeof(T);
        } else {
            Params.blockCount = tiling_->CLen;
            Params.blockLen = (tiling_->RLen - blockIdx_ * tiling_->rSplitPara.BlockFactor -
                               process * tiling_->rUbSplitPara.TailCoreUbFactor) *
                              sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = (blockIdx_ * tiling_->rSplitPara.BlockFactor +
                                process * tiling_->rUbSplitPara.TailCoreUbFactor) *
                               sizeof(T);
        }
    }
}

template <typename T>
__aicore__ inline void KernelTransDataTo5HD<T>::SetCSplitCopyOutParams(uint32_t process)
{
    if (blockIdx_ < fullCoreNum) {
        if (process < loopCount - 1) {
            Params.blockCount = (tiling_->RLen % TRANSELEM == 0) ? 1 : tiling_->cUbSplitPara.MainCoreUbFactor;
            Params.blockLen = (tiling_->RLen % TRANSELEM == 0) ?
                                  tiling_->rUbSplitPara.MainCoreUbFactor * tiling_->cUbSplitPara.MainCoreUbFactor *
                                      sizeof(T) :
                                  tiling_->RLen * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = 0;
        } else {
            Params.blockCount = (tiling_->RLen % TRANSELEM == 0) ? 1 : tiling_->cUbSplitPara.MainCoreTailUbFactor;
            Params.blockLen = (tiling_->RLen % TRANSELEM == 0) ?
                                  tiling_->rUbSplitPara.MainCoreTailUbFactor *
                                      tiling_->cUbSplitPara.MainCoreTailUbFactor * sizeof(T) :
                                  tiling_->RLen * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = 0;
        }
    } else {
        if (process < loopCount - 1) {
            Params.blockCount = (tiling_->RLen % TRANSELEM == 0) ? 1 : tiling_->cUbSplitPara.TailCoreUbFactor;
            Params.blockLen = (tiling_->RLen % TRANSELEM == 0) ?
                                  tiling_->rUbSplitPara.TailCoreUbFactor * tiling_->cUbSplitPara.TailCoreUbFactor *
                                      sizeof(T) :
                                  tiling_->RLen * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = 0;
        } else {
            Params.blockCount = (tiling_->RLen % TRANSELEM == 0) ?
                                    1 :
                                    (tiling_->CLen - blockIdx_ * tiling_->cSplitPara.BlockFactor -
                                     process * tiling_->cUbSplitPara.TailCoreUbFactor);
            Params.blockLen = (tiling_->RLen % TRANSELEM == 0) ?
                                  tiling_->rUbSplitPara.TailCoreTailUbFactor *
                                      (tiling_->CLen - blockIdx_ * tiling_->cSplitPara.BlockFactor -
                                       process * tiling_->cUbSplitPara.TailCoreUbFactor) *
                                      sizeof(T) :
                                  tiling_->RLen * sizeof(T);
            Params.srcStride = 0;
            Params.dstStride = 0;
        }
    }
}

} // namespace Transpose
#endif
