/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file quant_grouped_matmul.h
 * \brief
 */
 
#ifndef MC2_QUANT_GROUPED_MATMUL_H
#define MC2_QUANT_GROUPED_MATMUL_H

#include "kernel_operator.h"

#if __has_include("../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h")
#include "../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h"
#else
#include "../../../../3rd/grouped_matmul/op_kernel/gqmm_cube_on_the_fly.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {
constexpr uint64_t GROUP_LIST_INDEX = 0;

template <typename TilingDataType, typename GmmTilingDataType, class xType, class wType, class scaleType, class yType,
    CubeFormat wFormat, bool aTrans, bool bTrans, bool isLocal, bool isA2avGmm>
class QuantGroupedMatmul {
public:
    __aicore__ inline void Init(GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR xScaleGM, GM_ADDR weightScaleGM, GM_ADDR yGM,
        GM_ADDR workspaceGM, const TilingDataType *tilingData, const GmmTilingDataType *gmmTilingData,
        TILING_TYPE *gmmArrayAddrIn, TPipe *tPipe, bool isA2avGmmFlag)
    {
        if ASCEND_IS_AIV {
            return ;
        }
        xGM_ = xGM;
        wGM_ = weightGM;
        xScaleGM_ = xScaleGM;
        weightScaleGM_ = weightScaleGM;
        yGM_ = yGM;
        tilingData_ = tilingData;
        tPipe_ = tPipe;
        workspaceGM_ = workspaceGM;
        gmmTilingData_ = gmmTilingData;
        gmmArrayAddrIn_ = gmmArrayAddrIn;

        expertNumInOneRank_ = tilingData_->taskTilingInfo.e;
        epWorldSize_ = tilingData_->taskTilingInfo.epWorldSize;
        h1_ = tilingData_->taskTilingInfo.H1;
        n1_ = tilingData_->taskTilingInfo.N1;
        bs_ = tilingData_->taskTilingInfo.BS;
        a_ = tilingData_->taskTilingInfo.A;

        uint64_t permuteOutSize = tilingData_->isPermuteOut ? 0 : (a_ * h1_);
        // 将 permuteOutSize 对齐到 512 字节
        const uint64_t tensorListSize = 512;
        if (permuteOutSize % tensorListSize != 0) {
            permuteOutSize = (permuteOutSize + tensorListSize - 1) & ~(tensorListSize - 1);
        }
        uint64_t groupListSize = sizeof(int64_t) * expertNumInOneRank_; // GMM计算所需的groupList GM空间大小
        if (isA2avGmmFlag) {
            groupListGm_ = tilingData_->isPermuteOut ? workspaceGM_ : workspaceGM_ + permuteOutSize;
        } else {
            groupListGm_ = workspaceGM_;
        }
        
        ptrTableBase_ = groupListGm_ + groupListSize;
        xGlobalBuffer_.SetGlobalBuffer((__gm__ xType *)this->xGM_);
        wGlobalBuffer_.SetGlobalBuffer((__gm__ wType *)this->wGM_);
        yGlobalBuffer_.SetGlobalBuffer((__gm__ yType *)this->yGM_);
        groupListGlobalBuffer_.SetGlobalBuffer((__gm__ int64_t *)groupListGm_);
        xScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)xScaleGM);
        wScaleGlobalBuffer_.SetGlobalBuffer((__gm__ scaleType *)weightScaleGM);

        const auto *opCnt = isA2avGmmFlag ? &tilingData_->taskTilingInfo.recvCnt[0] : &tilingData_->taskTilingInfo.sendCnt[0];
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            for (uint32_t i = 0U; i < epWorldSize_; i++) {
                expertTokenNum_[e] += static_cast<uint64_t>(opCnt[e + i * expertNumInOneRank_]);
            }
        }
    }

    __aicore__ inline void Process(uint32_t expertIdx)
    {
        if ASCEND_IS_AIV {
            return ;
        }
        if (expertTokenNum_[expertIdx] == 0) {
            return ;
        }
        this->UpdateAddr(expertIdx);
        // 3. 计算偏移后地址
        __gm__ uint8_t *xAddr = reinterpret_cast<__gm__ uint8_t *>(xGM_);
        __gm__ uint8_t *yAddr = reinterpret_cast<__gm__ uint8_t *>(yGM_);
        __gm__ uint8_t *wAddr = reinterpret_cast<__gm__ uint8_t *>(wGM_);
        // 4. 构建 GetTensorAddr 指针表
        GM_ADDR xPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(xAddr), 0);
        GM_ADDR wPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(wAddr), 1);
        GM_ADDR scaleBPtr = BuildPtrTable(xScaleGM_, 2);
        GM_ADDR yPtr = BuildPtrTable(reinterpret_cast<GM_ADDR>(yAddr), 3); 

        uint64_t groupListToken = isLocal ? bs_ : expertTokenNum_[expertIdx];
        groupListGlobalBuffer_.SetValue(GROUP_LIST_INDEX, groupListToken);
        AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
            AscendC::DcciDst::CACHELINE_OUT>(groupListGlobalBuffer_);
        Mc2GroupedMatmul::Mc2GmmASWKernel<xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans> gmmASWKernel;
        tPipe_->Reset();
        gmmASWKernel.Init(xPtr, wPtr, nullptr, scaleBPtr, groupListGm_, weightScaleGM_, yPtr, workspaceGM_,
            &gmmTilingData_->gmmQuantParams, &gmmTilingData_->mmTilingData, gmmArrayAddrIn_, tPipe_);
        gmmASWKernel.Process();
    }

    __aicore__ inline void End() {
        if ASCEND_IS_AIV {
            return ;
        }
    }

protected:
    __aicore__ inline void UpdateAddr(uint32_t expertIdx)
    {
        xGM_ = (GM_ADDR)xGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * h1_);
        wGM_ = (GM_ADDR)wGlobalBuffer_.GetPhyAddr(expertIdx * h1_ * n1_);
        yGM_ = (GM_ADDR)yGlobalBuffer_.GetPhyAddr(expertTokenOffset_ * n1_);
        expertTokenOffset_ += expertTokenNum_[expertIdx];
    }

    /**
     * 在 workspace 指针表区域构建 GetTensorAddr 所需的双重间接指针
     * slotIdx: 0=x, 1=weight, 2=scaleB, 3=y
     * 返回指向指针表条目的地址
     */
    __aicore__ inline GM_ADDR BuildPtrTable(GM_ADDR dataAddr, uint32_t slotIdx)
    {
        // 每个 slot 占 16 bytes (2 * uint64_t)
        __gm__ uint64_t *slot = reinterpret_cast<__gm__ uint64_t *>(
            reinterpret_cast<__gm__ uint8_t *>(ptrTableBase_) + slotIdx * 16);
        slot[0] = sizeof(uint64_t);  // byteOffset
        slot[1] = reinterpret_cast<uint64_t>(dataAddr);  // 实际数据地址
        return reinterpret_cast<GM_ADDR>(slot);
    }

private:
    using biasType = float;

    GM_ADDR xGM_;
    GM_ADDR wGM_;
    GM_ADDR xScaleGM_;
    GM_ADDR weightScaleGM_;
    GM_ADDR yGM_;
    GM_ADDR groupListGm_;
    GM_ADDR workspaceGM_;
    GM_ADDR ptrTableBase_ = nullptr;
    GlobalTensor<xType> xGlobalBuffer_;
    GlobalTensor<wType> wGlobalBuffer_;
    GlobalTensor<scaleType> xScaleGlobalBuffer_;
    GlobalTensor<scaleType> wScaleGlobalBuffer_;
    GlobalTensor<yType> yGlobalBuffer_;
    GlobalTensor<int64_t> groupListGlobalBuffer_;
    const TilingDataType *tilingData_;
    TPipe *tPipe_;
    uint64_t expertTokenNum_[32] = {0};
    uint64_t expertTokenOffset_ = 0;
    uint64_t expertNumInOneRank_ = 0;
    uint64_t epWorldSize_ = 0;
    uint64_t h1_;
    uint64_t n1_;
    uint64_t bs_;
    uint64_t a_;
    const GmmTilingDataType *gmmTilingData_;
    TILING_TYPE *gmmArrayAddrIn_;
};
} // namespace MC2KernelTemplate
#endif
// MC2_QUANT_GROUPED_MATMUL_H