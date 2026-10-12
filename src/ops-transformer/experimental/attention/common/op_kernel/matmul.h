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
 * \file matmul.h
 * \brief
 */
#ifndef MATMUL_H
#define MATMUL_H
#include "buffers_policy.h"
using namespace AscendC;
namespace fa_base_matmul {

constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16; // 一个分形512B,16*16个fp16
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16; //  一个分形512B,height方向为16个element
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32; //  一个分形512B,weight方向为32B
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
static constexpr uint32_t MMAD_MN_SIZE_10 = 10;
static constexpr uint8_t LOAD3D_STRIDE_W = 1;
static constexpr uint8_t LOAD3D_STRIDE_H = 1;
static constexpr uint8_t LOAD3D_FILTER_W = 1;
static constexpr uint8_t LOAD3D_FILTER_H = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_W = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_H = 1;
static constexpr uint32_t K_STEP_ALIGN_BASE = 2;
static constexpr uint32_t M_STEP_ALIGN_BASE = 2;

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // 默认值为true， true：在L1切K轴的场景中，表示首轮K
    uint32_t unitFlag = 0;  // 0：disable: 不配置unitFlag
                            // 2：enable: 行为在切K接口中（MatmulK），会将mmadParams.unitFlag设置为 0b10
                            // 3：enable: 行为在切K接口中（MatmulK），在k的最后一轮循环，会将mmadParams.unitFlag设置为 0b11
                            // 外部使用时，在外层k循环的最后一轮将该参数配置为3
    uint32_t realM = 0; // bmm2以s1realsize为M轴，不赋值时不影响现有代码逻辑
};

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Aoffset, uint32_t kSplitSize,
                                     uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = mSplitSize / LOAD3D_L1W_SIZE; // 源操作数height
        loadData3DParams.l1W = LOAD3D_L1W_SIZE; // 源操作数weight
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

        loadData3DParams.mExtension = mSplitSize; // 在目的操作数height维度的传输长度
        loadData3DParams.kExtension = kSplitSize; // 在目的操作数width维度的传输长度
        loadData3DParams.mStartPt = 0; // 卷积核在目的操作数width维度的起点
        loadData3DParams.kStartPt = 0; // 卷积核在目的操作数height维度的起点
        loadData3DParams.strideW = 1; // 卷积核在源操作数width维度滑动的步长
        loadData3DParams.strideH = 1; // 卷积核在源操作数height维度滑动的步长
        loadData3DParams.filterW = 1; // 卷积核width
        loadData3DParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
        loadData3DParams.filterH = 1; // 卷积核height
        loadData3DParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
        loadData3DParams.dilationFilterW = 1; // 卷积核width膨胀系数
        loadData3DParams.dilationFilterH = 1; // 卷积核height膨胀系数
        loadData3DParams.enTranspose = 0; // 是否启用转置功能，对整个目标矩阵进行转置
        loadData3DParams.fMatrixCtrl = 0;
        loadData3DParams.channelSize = kSplitSize; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
        LoadData<T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams);
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // 分型矩阵ID，表明搬运起始位置为源操作数中第0个分型
        loadData2DParams.repeatTimes = (kSplitSize / ONE_FRACTAL_H_ELEMENT) * (mmParam.singleM /
                                       (ONE_FRACTAL_W_BYTE / sizeof(T))); // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.srcStride = 1; // 相邻迭代间，源操作数前一个分型和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.dstGap = 0; // 相邻迭代间，目的操作数前一个分型的结束地址和后一个分型起始地址的间隔（单位512B）
        loadData2DParams.ifTranspose = true;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParams);
    }
}

// L1→L0B + 切K/切N/全载
template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                     const MMParam& mmParam, uint64_t L1Boffset, uint32_t kSplitSize,
                                     uint32_t nSplitSize)
{
    if constexpr (BL == ABLayout::KN) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = kSplitSize / LOAD3D_L1W_SIZE; // 源操作数height
        loadData3DParams.l1W = LOAD3D_L1W_SIZE; // 源操作数weight=16，目的height=l1H*L1W
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果
 
        loadData3DParams.mExtension = kSplitSize; // 在目的操作数height维度的传输长度
        loadData3DParams.kExtension = nSplitSize; // 在目的操作数width维度的传输长度
        loadData3DParams.mStartPt = 0; // 卷积核在目的操作数width维度的起点
        loadData3DParams.kStartPt = 0; // 卷积核在目的操作数height维度的起点
        loadData3DParams.strideW = LOAD3D_STRIDE_W;
        loadData3DParams.strideH = LOAD3D_STRIDE_H;
        loadData3DParams.filterW = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
        loadData3DParams.filterH = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W; // 卷积核width膨胀系数
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H; // 卷积核height膨胀系数
        loadData3DParams.enTranspose = 1; // 是否启用转置功能
        loadData3DParams.fMatrixCtrl = 0; // 使用FMATRIX_LEFT还是使用FMATRIX_RIGHT，=0使用FMATRIX_LEFT，=1使用FMATRIX_RIGHT 1
        loadData3DParams.channelSize = nSplitSize; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
        LoadData<T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams);
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = (nSplitSize + (ONE_FRACTAL_H_ELEMENT - 1)) / ONE_FRACTAL_H_ELEMENT *
                                       (kSplitSize / (ONE_FRACTAL_W_BYTE / sizeof(T))); // 迭代次数，每个迭代可以处理512B数据
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParams);
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKPP(const LocalTensor<A> &aL1Tensor,
                                 const LocalTensor<B> &bL1Tensor,
                                 BuffersPolicyDB<BufferType::L0A> &aL0BuffsDb,
                                 BuffersPolicyDB<BufferType::L0B> &bL0BuffsDb,
                                 const LocalTensor<C> &cL0Tensor,
                                 const MMParam &param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
    uint64_t L1Aoffset = AlignUp(param.singleM, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    uint64_t L1Boffset = AlignUp(param.singleN, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    for (uint32_t k = 0; k < kLoops; k++) {
        if (k == kLoops - 1) {
            kSplitSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
            kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
        }
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // mte1等Matmul:上一轮matmul完成后才能搬运新数据到L0A
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, kSplitSizeAlign, param.singleM);
        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSizeAlign, param.singleN);

        l0aBuffer.Set<HardEvent::MTE1_M>(); // mte1搬运完后，通知可以开始matmul
        l0aBuffer.Wait<HardEvent::MTE1_M>(); // matmul等mte1：L0A数据搬运完成后才能开始matmul

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        mmadParams.n = param.singleN;
        mmadParams.k = kSplitSize;
        if (mmadParams.m == 1) { //m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = (param.isOutKFisrt == true) && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                  UNITFLAG_EN_OUTER_LAST : UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
    
        if ((mmadParams.m / FP16_ONE_FRACTAL_ELEMENT) * (mmadParams.n / FP16_ONE_FRACTAL_ELEMENT) < MMAD_MN_SIZE_10) {
            AscendC::PipeBarrier<PIPE_M>();
        }
   
        l0aBuffer.Set<HardEvent::M_MTE1>(); // matmul完成后，通知mte1可以开始搬运新数据到L0A
    }
}
}
#endif