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
 * \file quant_reduce_scatter_mte.h
 * \brief quant_reduce_scatter mte通信kernel代码逻辑
 */

#ifndef QUANT_REDUCE_SCATTER_MTE_H
#define QUANT_REDUCE_SCATTER_MTE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/pad/broadcast.h"
#include "kernel_tiling/kernel_tiling.h"
#include "quant_reduce_scatter_tiling_data.h"
#include "utils.h"
#include "mte_comm.h"
#include "vec_comp.h"

namespace QuantReduceScatterImpl {

using namespace QuantMTECommImpl;
using namespace VectorComputeImpl;
using namespace AscendC;

// 之后可修改成从tiling侧获取数据切块大小
constexpr static uint32_t X_PRE_BLOCK_NUM = 1024U;  // 当前一次搬运一个x数据块，x dtype为 8bit 时对应 1024个x数据. 对于fp4需要另外算
constexpr static uint64_t MX_SCALES_LAST_DIM = 2U; // MX量化scales最后一维的大小

template<TemplateTypeClass>
class QuantReduceScatterMte {
public:
    __aicore__ inline QuantReduceScatterMte() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR output,
                                TPipe *pipe, const QuantReduceScatterTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ClearSumTensor();
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curXOffset, uint64_t curScaleOffset);
    __aicore__ inline void ExecuteReduceScatter();

    MTECommunication<TemplateType> mteComm_; // MTE 通信相关实现
    VectorCompute<TemplateType> vecComp_; // vector 计算相关实现

    GlobalTensor<XType> remoteWinXTensor_;
    GlobalTensor<ScalesType> remoteWinScaleTensor_;
    LocalTensor<float> sumTensor_;

    TQue<QuePosition::VECIN, 1> xInQueue_, scaleInQue; // 用于读数据和反量化求和的通算并行
    TBuf<> sumBuf_; // 用于Reduce_sum 求和

    uint64_t xSize_{0};
    uint64_t totalWinSize_{0};
    uint64_t tailXNums_{0};
    uint32_t totalBlockNums_{0};
    uint64_t alignedXSize_{0};
    // reduceScater进行all2all过程，数据要按卡数进行均分，以下为计算切分相关的参数
    uint64_t scaleSize_{0};
    uint64_t xSliceSize_{0};
    uint64_t xSliceSizeNums_{0};
    uint64_t scaleSliceNums_{0};
};

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Init(GM_ADDR x, GM_ADDR scales,
    GM_ADDR output, TPipe *tPipe, const QuantReduceScatterTilingData *tilingData)
{
    // 初始化HcclContext
    mteComm_.InitHcclContext();

    /* quant_reduce_scatter自己的数据 */
    auto&& tiliingDatainfo = tilingData->quantReduceScatterTilingInfo;
    totalWinSize_ = tiliingDatainfo.totalWinSize;
    xSize_ = tiliingDatainfo.bs * tiliingDatainfo.hiddenSize * sizeof(XType);
    scaleSize_ = tiliingDatainfo.bs * tiliingDatainfo.scaleHiddenSize * sizeof(ScalesType);
    // 对于mx的scale是三维，最后一维为2，总scales的数据量需要再乘以2
    if constexpr(AscendC::IsSameType<ScalesType, fp8_e8m0_t>::value) {
        scaleSize_ *= MX_SCALES_LAST_DIM;
    }
    xSliceSize_ = xSize_ / (mteComm_.hcclContext_->rankDim ); // all2all过程，数据需要按卡数均分
    xSliceSizeNums_ = xSliceSize_ / sizeof(XType); // 按卡均分后每片x数据的个数
    scaleSliceNums_ = scaleSize_ / (mteComm_.hcclContext_->rankDim * sizeof(ScalesType)); // all2all过程，每张卡需要的scale数据个数
    tailXNums_ = BlockAlignMod(xSliceSizeNums_, X_PRE_BLOCK_NUM); // 计算分卡后每片的最后一个数据块的大小
    totalBlockNums_ = CeilDiv(xSliceSize_, X_BLOCK_BYTES); // 1/rank 数据需要搬运的总块数
    mteComm_.round_ = totalBlockNums_ / tiliingDatainfo.aivNum;  // 计算数据分核搬运需要的轮次数
    mteComm_.tailBlockNums_ = totalBlockNums_ % tiliingDatainfo.aivNum; // 搬运的尾块数
    mteComm_.ComputeTailAivId(tiliingDatainfo.aivNum); // 计算最后一个核的id
    tPipe->Reset();
    tPipe->InitBuffer(xInQueue_, BUFFER_NUM, X_BLOCK_BYTES); // 每次拷贝 1024B x; 128 * 8
    tPipe->InitBuffer(scaleInQue, BUFFER_NUM, UB_ALIGN_BYTES); // 每次拷贝 32B scale；4 * 8
    tPipe->InitBuffer(sumBuf_, X_PRE_BLOCK_NUM * sizeof(float)); // 用于Reduce_sum 求和，1024 * 4 = 4k
    sumTensor_ = sumBuf_.Get<float>();

    // 设置quant_reduce_scatter切块大小
    mteComm_.SetBlockSize(X_PRE_BLOCK_NUM, tiliingDatainfo.aivNum, tailXNums_);
    vecComp_.SetBlockSize(X_PRE_BLOCK_NUM);  

    // 公共MTE搬运参数计算
    mteComm_.InitParams();

    // 初始化tPipe的各种buffer
    mteComm_.InitBuffer(tPipe);
    vecComp_.InitBuffer(tPipe);

    // 初始化GM上的Tensor，包括Win区
    mteComm_.InitGMTensor(x, scales, output, xSize_, totalWinSize_);
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ReadDataBlockReduceSum(uint64_t curXOffset,
    uint64_t curScaleOffset)
{
    /* 读取 x 从 Win -> UB */
    LocalTensor<XType> xTmpTensor = xInQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor, remoteWinXTensor_[curXOffset], X_PRE_BLOCK_NUM);
    xInQueue_.EnQue(xTmpTensor);
    xTmpTensor = xInQueue_.DeQue<XType>();

    /* 读取 scale 从 Win -> UB */
    LocalTensor<ScalesType> scaleTmpTensor = scaleInQue.AllocTensor<ScalesType>();
    DataCopy(scaleTmpTensor, remoteWinScaleTensor_[curScaleOffset], mteComm_.scaleNumsPerBlcok_);
    scaleInQue.EnQue(scaleTmpTensor);
    scaleTmpTensor = scaleInQue.DeQue<ScalesType>();

    /* 反量化计算与ReduceSum求和 */
    vecComp_.DequantReduceSum(xTmpTensor, scaleTmpTensor, sumTensor_); 
    xInQueue_.FreeTensor(xTmpTensor);
    scaleInQue.FreeTensor(scaleTmpTensor);
}


template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ClearSumTensor()
{
    Duplicate<float>(sumTensor_, (float)0.0, X_PRE_BLOCK_NUM); // sumTensor 清零
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::ExecuteReduceScatter()
{   
    // 读状态位，软同步
    mteComm_.ReadStatus(); 
    // 遍历需要搬运的数据块
    for (uint64_t curBlock = 0; curBlock < mteComm_.assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = mteComm_.xOffset_ + curBlock * X_PRE_BLOCK_NUM;
        uint64_t curScaleOffset = mteComm_.scaleOffset_ + curBlock * mteComm_.scaleNumsPerBlcok_;
        ClearSumTensor(); // sumTensor 清零
        // 遍历每张卡，读取其Win区的数据，采取错卡序读取，从自己卡上读起
        /* rank0: [0,1,2]; rank1: [1,2,0]; rank2: [2,0,1]*/
        uint32_t startRankId = mteComm_.hcclContext_->rankId;
        for (uint32_t i = 0; i < mteComm_.hcclContext_->rankDim; ++i) {
            uint32_t remoteRankId = (startRankId + i) % mteComm_.hcclContext_->rankDim;

            // 获取对端Win区中数据区相关的地址
            GM_ADDR remoteDataSpaceGm = mteComm_.GetWinDataAddrGm(remoteRankId, mteComm_.winBufferFlags_);

            remoteWinXTensor_.SetGlobalBuffer((__gm__ XType*)remoteDataSpaceGm);
            remoteWinScaleTensor_.SetGlobalBuffer((__gm__ ScalesType*)(remoteDataSpaceGm + xSize_));

            // 读取对端对应地址的 x 和 scale数据，进行反量化和求和
            uint64_t curRankXOffset = curXOffset + mteComm_.hcclContext_->rankId * xSliceSizeNums_;
            uint64_t curRankScaleOffset = curScaleOffset + mteComm_.hcclContext_->rankId * scaleSliceNums_;
            ReadDataBlockReduceSum(curRankXOffset, curRankScaleOffset); // ReduceScatter过程，all2all仅需与rankId相关的数据，加上本卡偏移
        }

        // 将计算好的数据拷贝到输出tensor
        uint32_t copyBlockNum = X_PRE_BLOCK_NUM;
        if ((mteComm_.aivId_ ==  mteComm_.lastAivId_) && (curBlock == mteComm_.assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_; // 检测是否为最后的尾块搬运（即最后一个核的最后一个数据块）
        }
        mteComm_.CopyResultToOutput(curXOffset, sumTensor_, copyBlockNum);
    }
}

template <TemplateTypeClass>
__aicore__ inline void QuantReduceScatterMte<TemplateType>::Process()
{
    // 纯AIV过程
    if ASCEND_IS_AIC {
        return;
    }

    // 一次性拷贝完所有数据到本地卡win区
    mteComm_.template CopyDataToWin<true>(xSliceSizeNums_, scaleSliceNums_);
    // 写入状态到状态区
    mteComm_.WriteStatusToWin();
    // 执行ReduceScatter过程：等待状态区同步，读取数据并进行反量化ReduceSum
    ExecuteReduceScatter();
}
} // QuantReduceScatterImpl
#endif  // QUANT_REDUCE_SCATTER_MTE_H