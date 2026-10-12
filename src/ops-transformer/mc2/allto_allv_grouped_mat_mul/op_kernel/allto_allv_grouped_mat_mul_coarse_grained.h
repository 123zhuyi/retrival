/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_allv_grouped_mat_mul_coarse_grained.h
 * \brief
 */
#ifndef ALL_TO_ALL_V_GROUPED_MAT_MUL_H
#define ALL_TO_ALL_V_GROUPED_MAT_MUL_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "allto_allv_gmm.h"
#include "lib/matmul_intf.h"
#include "allto_allv_grouped_mat_mul_tiling.h"
namespace AscendC {
using namespace ALLTO_ALLV_GMM;

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
class AlltoAllvGmmCoarseGrained
{
public:
    __aicore__ inline AlltoAllvGmmCoarseGrained()
    {}
    __aicore__ inline void Init(
        GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
        GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
        GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM, const AlltoAllvGmmTilingData* tilingData,
        __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, false>;
    using gmmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, IsTranGmmW>;
    using mmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, IsTranMmW>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DataType>;
    using gmmType = MMImplType<aType, gmmBType, cType, biasType, CFG_MDL>;
    using mmType = MMImplType<aType, mmBType, cType, biasType, CFG_MDL>;

    __aicore__ inline void CalcMatmul();
    __aicore__ inline void HcclAlltoAllvPrepare();
    __aicore__ inline void HcclAlltoAllvExec();
    __aicore__ inline void HcclFinalize();

    GM_ADDR gmmxGM_ = nullptr;
    GM_ADDR gmmwGM_ = nullptr;
    GM_ADDR sendCntsGM_ = nullptr;
    GM_ADDR recvCntsGM_ = nullptr;
    GM_ADDR mmxGM_ = nullptr;
    GM_ADDR mmwGM_ = nullptr;
    GM_ADDR gmmyGM_ = nullptr;
    GM_ADDR mmyGM_ = nullptr;
    GM_ADDR permuteOutGM_ = nullptr;
    const AlltoAllvGmmTilingData* tilingData_ = nullptr;
    uint32_t rankId_ = 0U;             // 当前卡ID
    uint32_t rankDim_ = 8U;            // 通信域内卡的数量
    uint32_t expertNumInOneRank_ = 0U; // 单卡上面的专家个数
    uint32_t expertNumAll_ = 0U;       // 通信域内专家个数

    // alltoall 流程数据结构
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl_;
#else
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
#endif
    HcclDataType hcclDataType_ = HCCL_DATA_TYPE_FP16;
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};

    GlobalTensor<DataType> gmmxGMTensor_;
    GlobalTensor<DataType> permutedGMTensor_;

    uint64_t axisH1_;
    uint64_t axisN1_;

    typename gmmType::MT gmm_;
    typename mmType::MT mm_;
};

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorOptionalGM, GM_ADDR recvCountsTensorOptionalGM,
    GM_ADDR mmxOptionalGM, GM_ADDR mmweightOptionalGM, GM_ADDR gmmyGM, GM_ADDR mmyOptionalGM,
    GM_ADDR permuteOutOptionalGM, GM_ADDR workspaceGM, GM_ADDR contextGM, const AlltoAllvGmmTilingData* tilingData,
    __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling, TPipe* tPipe)
{
    gmmxGM_ = gmmxGM;
    gmmwGM_ = gmmweightGM;
    sendCntsGM_ = sendCountsTensorOptionalGM;
    recvCntsGM_ = recvCountsTensorOptionalGM;
    mmxGM_ = mmxOptionalGM;
    mmwGM_ = mmweightOptionalGM;
    gmmyGM_ = gmmyGM;
    mmyGM_ = mmyOptionalGM;
    tilingData_ = tilingData;
    permuteOutGM_ = tilingData_->commonTilingInfo.isPermuteOut ? permuteOutOptionalGM : workspaceGM;

    const void *hcclInitTilingV2 = &(tilingData_->hcclInitTiling);
    uint64_t hcclCcTilingOffset = offsetof(AlltoAllvGmmTilingData, alltoAllvCcTiling);
    hccl_.InitV2(contextGM, hcclInitTilingV2);
    hccl_.SetCcTilingV2(hcclCcTilingOffset); 
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    expertNumAll_ = expertNumInOneRank_ * rankDim_;

    axisH1_ = tilingData_->commonTilingInfo.H1;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    if constexpr (AscendC::IsSameType<DataType, bfloat16_t>::value) {
        hcclDataType_ = HCCL_DATA_TYPE_BFP16;
    }

    gmmxGMTensor_.SetGlobalBuffer((__gm__ DataType*)this->gmmxGM_);
    permutedGMTensor_.SetGlobalBuffer((__gm__ DataType*)this->permuteOutGM_);
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::Process()
{
    HcclAlltoAllvPrepare();
    if (tilingData_->commonTilingInfo.isNeedMM) {
        CalcMatmul();
        SyncAll<false>();
    }
    HcclAlltoAllvExec();
    SyncAll<false>();
    HcclFinalize();
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::CalcMatmul()
{
    mm_.Init(&(tilingData_->mmTilingData));
    GMMCompute<mmType> computeOp(mm_);
    computeOp.Init(mmxGM_, mmwGM_, mmyGM_);
    GMMProcess<decltype(computeOp)> mmOp(computeOp);
    mmOp.Init(
        tilingData_->mmTilingData.baseM, tilingData_->mmTilingData.baseN, tilingData_->commonTilingInfo.aicCoreNum);

    uint64_t mmInOffset[1] = {0};
    uint64_t mmOutOffset[1] = {0};
    uint64_t mmWeightOffset[1] = {0};
    uint32_t tokenNum[1] = {(uint32_t)(tilingData_->commonTilingInfo.BS)};
    if ASCEND_IS_AIC {
        mmOp.Process(
            this->rankId_, tilingData_->commonTilingInfo.H2, tilingData_->commonTilingInfo.N2, mmInOffset, mmOutOffset,
            tokenNum, 1, 0);
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::HcclAlltoAllvPrepare()
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }
        const auto* sendCnt = &tilingData_->aicpuTiling.sendCnt[0];
        const auto* recvCnt = &tilingData_->aicpuTiling.recvCnt[0];
        uint64_t alltoAllvRecvOffsetLastSum = 0UL;
        uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
        uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
        for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
            // 计算sendcnts/recvcnts
            for (uint32_t i = 0U; i < rankDim_; i++) {
                alltoAllvSendCnt[i] = static_cast<uint64_t>(sendCnt[i * expertNumInOneRank_ + e]) * axisH1_;
                alltoAllvRecvCnt[i] = static_cast<uint64_t>(recvCnt[i * expertNumInOneRank_ + e]) * axisH1_;
            }

            // 计算sendoffs: 本卡上专家数偏移+前面卡token数的和 -- 维护每张卡每个专家的总长度
            alltoAllvSendOffset[0] = 0UL;
            for (uint32_t j = 0U; j < e; j++) { // 0卡上的sendOffset
                alltoAllvSendOffset[0U] += static_cast<uint64_t>(sendCnt[j]) * axisH1_;
            }
            for (uint32_t i = 1U; i < rankDim_; i++) {
                alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1U];
                for (uint32_t j = 0U; j < expertNumInOneRank_; j++) {
                    // 后面的每张卡的sendOffset/等同于前一张卡的sendOffset+后面expertNumInOneRank_个sendCnt之和
                    alltoAllvSendOffset[i] +=
                        static_cast<uint64_t>(sendCnt[e + (i - 1U) * expertNumInOneRank_ + j]) * axisH1_;
                }
            }
            for (uint32_t i = 0U; i < rankDim_; i++) {
                if ((e == 0U) && (i == 0U)) {
                    alltoAllvRecvOffset[i] = 0UL;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[0];
                } else {
                    alltoAllvRecvOffset[i] = alltoAllvRecvOffsetLastSum;
                    alltoAllvRecvOffsetLastSum += alltoAllvRecvCnt[i];
                }
            }
            alltoAllvHandleId_[e] = hccl_.AlltoAllV<true>(
                (__gm__ uint8_t*)this->gmmxGMTensor_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
                (__gm__ uint8_t*)this->permutedGMTensor_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset,
                hcclDataType_);
        }
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::HcclAlltoAllvExec()
{
    gmm_.Init(&(tilingData_->gmmTilingData));
    GMMCompute<gmmType> computeOp(gmm_);
    computeOp.Init(permuteOutGM_, gmmwGM_, gmmyGM_);
    GMMProcess<decltype(computeOp)> gmmOp(computeOp);
    gmmOp.Init(
        tilingData_->gmmTilingData.baseM, tilingData_->gmmTilingData.baseN, tilingData_->commonTilingInfo.aicCoreNum);

    auto* recvCnt = &tilingData_->aicpuTiling.recvCnt[0];

    uint64_t mmInOffset[2] = {0};
    uint64_t mmOutOffset[2] = {0};
    uint64_t mmWeightOffset[2] = {0};
    uint32_t tokenNum[2] = {0};
    uint32_t processNum = 1;

    uint64_t expertoffset[MAX_HANDLE_ID_NUM] = {0UL};
    uint64_t gmmTokennum[MAX_HANDLE_ID_NUM] = {0UL};
    for (uint32_t e = 0U; e < expertNumInOneRank_; e++) {
        uint64_t curTokenNum = 0;
        for (uint32_t i = 0U; i < rankDim_; i++) {
            curTokenNum += static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]);
        }
        gmmTokennum[e] = curTokenNum; // 本专家的tokenNum
    }
    for (uint64_t i = 0; i < expertNumInOneRank_; i++) {
        if (i >= 1) {
            expertoffset[i] = expertoffset[i - 1] + gmmTokennum[i - 1] * axisH1_;
        }
    }
    for (uint32_t e = 0U; e < this->expertNumInOneRank_; e++) {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                hccl_.Wait(alltoAllvHandleId_[e]);
            }
        }
        SyncAll<false>();
        mmInOffset[0] = expertoffset[e];
        mmOutOffset[0] = expertoffset[e] / axisH1_ * axisN1_;
        mmWeightOffset[0] = e * axisH1_ * axisN1_;
        tokenNum[0] = gmmTokennum[e];

        if ASCEND_IS_AIC {
            if (tokenNum[0] != 0) {
                gmmOp.Process(this->rankId_, axisH1_, axisN1_, mmInOffset, mmOutOffset, tokenNum, processNum, e);
            }
        }
    }
}

template <typename DataType, bool IsNeedMM, bool IsTranGmmW, bool IsTranMmW>
__aicore__ inline void AlltoAllvGmmCoarseGrained<DataType, IsNeedMM, IsTranGmmW, IsTranMmW>::HcclFinalize()
{
    if ASCEND_IS_AIV {
        hccl_.Finalize();
    }
}

} // namespace AscendC
#endif
