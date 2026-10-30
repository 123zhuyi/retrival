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
 * \file all_gather_quant_bmm.h
 * \brief
 */
#ifndef ALL_GATHER_QUANT_BMM_H
#define ALL_GATHER_QUANT_BMM_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../common.h"
#include "lib/hccl/hccl.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_cube_on_the_fly.h"
#include "../../common/new_mc2_mm/kernel/mc2_quant_batch_matmul.h"
#include "all_gather_matmul_tiling_arch35.h"

/**
 * 1、依赖tiling结构QuantBatchMatmulV3TilingData
 * 2、依赖新matmul接口
 * 3、低精度类型定义 (json)
 * 4、AllgatherMatmulTilingData结构体, tilingdata + taildata (QuantBatchMatmulV3TilingData)
 */
namespace AllGatherMatmulImpl {
using namespace AscendC;

template <typename AType, typename BType, typename BiasType, typename X2ScaleType, typename CType, bool ATrans,
          bool BTrans>
class AllGatherQuantBmm {
public:
    __aicore__ inline AllGatherQuantBmm() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM1, GM_ADDR dequantGM2, GM_ADDR biasGM,
                                GM_ADDR scaleGM, GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR contextGM,
                                Mc2Tiling::AllGatherMatmulTilingDataFp8* tilingData, __gm__ void* mc2InitTiling, 
                                __gm__ void* mc2CcTiling, TPipe* tPipe);
    __aicore__ inline void Process();
private:
    __aicore__ inline void AllGatherCommit();               /* allgather通信 */
    __aicore__ inline void QuantMatmulCompute();            /* 计算入口 */
    __aicore__ inline void QuantMatmulLocalCompute();       /* 本地块计算 */

    __aicore__ inline void QuantMatmulGatherCompute();      /* 远端计算 */
    __aicore__ inline void PostProcess();                   /* 后处理 */
    __aicore__ inline void MatmulKernelComputeGather(GM_ADDR aGM,
                                                     DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qMmv3Tiling,
                                                     uint32_t count, bool isLast, bool isTail);
private:
    Mc2Tiling::AllGatherMatmulTilingDataFp8* tilingData_;
    TPipe *tPipe_;
    GM_ADDR cGM_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR dequantGM1_;
    GM_ADDR localDequantGM1_;
    GM_ADDR gatherDequantGM1_;
    GM_ADDR dequantGM2_;
    GM_ADDR dequantGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR gatherOut_;
    GM_ADDR context_;
    AscendC::HcclDataType dataType_;
    uint8_t debugMode_;
    uint32_t rankId_;

    HcclHandle handleList_[MAX_HANDLE_WITH_SCALE1];             /* hccl handle */
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hcclAllgather_;        /* CCU */
    uint64_t preCoreNum_ = 0;
};

/**
 * @brief allgather初始化
 * @param [in] aGM: 左矩阵地址
 * @param [in] bGM: 右矩阵地址
 * @param [in] dequantGM1: 左矩阵dequant参数地址
 * @param [in] dequantGM2: 右矩阵dequant参数地址
 * @param [in] biasGM: bias矩阵地址
 * @param [in] scaleGM: scale quant参数地址
 * @param [in] cGM: 输出矩阵地址
 * @param [in] gatherOut:gatherOut地址: 若有指定地址则用指定地址，若没有指定地址则使用workspacek空间
 * @param [in] workspaceGM: workspace结构：16M头 + nd2nz空间 + mgmcFloat空间 + gather空间 + bias
 * @param [in] contextGM: context信息
 * @param [in] tilingData: 切分参数
 * @param [in] tPipe: tipe对象，matmul接口调用入参
 * @return void
*/
template <typename AType, typename BType, typename BiasType, typename X2ScaleType, typename CType, bool ATrans,
          bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType, X2ScaleType, CType, ATrans, BTrans>::Init(
                                GM_ADDR aGM, GM_ADDR bGM, GM_ADDR dequantGM1,
                                GM_ADDR dequantGM2, GM_ADDR biasGM, GM_ADDR scaleGM, GM_ADDR cGM,
                                GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR contextGM,
                                Mc2Tiling::AllGatherMatmulTilingDataFp8* tilingData, __gm__ void* mc2InitTiling, 
                                __gm__ void* mc2CcTiling, TPipe* tPipe) {
    auto &&cfg = tilingData->param;

    context_ = contextGM;
    hcclAllgather_.Init(context_, mc2InitTiling);
    hcclAllgather_.SetCcTiling(mc2CcTiling);
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    biasGM_ = biasGM;
    dequantGM1_ = dequantGM1;
    localDequantGM1_ = dequantGM1;
    dequantGM2_ = dequantGM2;
    dequantGM_ = scaleGM;
    workspaceGM_ = workspaceGM;
    debugMode_ = tilingData_->debugMode;
    dataType_ = static_cast<AscendC::HcclDataType>(tilingData_->dataType);
    rankId_ = ((__gm__ HcclCombinOpParam*)context_)->rankId;
    // 若有指定gatherout地址则用指定地址，若未指定gatherOut地址则为workspace空间
    if ((cfg.gatherLen != 0) || (!gatherOut)) {
        gatherOut_ = workspaceGM_;
        // MXType场景下，gather的存储顺序是gatherScale1、gatherX1
        if constexpr (DequantBmm::IsMxType<X2ScaleType>()) {
            // 留出scales1偏移
            uint64_t gatherScale1Offset =
                DequantBmm::Align(CeilDiv(tilingData_->quantBmmv3TileTiling.matmulTiling.Ka, MX_BLOCK_SIZE), EVEN_ALIGN) *
                static_cast<uint64_t>(cfg.rankM) * sizeof(X2ScaleType) * static_cast<uint64_t>(cfg.rankDim);
            gatherDequantGM1_ = workspaceGM_;
            gatherOut_ = workspaceGM_ + gatherScale1Offset;
        } 
    } else {
        gatherOut_ = gatherOut;
    }
}

/**
 * @brief allgather通信流程
 * @return void
*/
template <typename AType, typename BType, typename BiasType, typename X2ScaleType, typename CType, bool ATrans,
          bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType, X2ScaleType,
                                         CType, ATrans, BTrans>::AllGatherCommit() {
    auto&& cfg = tilingData_->param;
    uint32_t tileCnt = cfg.tileCnt;                                     // 整块个数
    uint32_t tailCnt = cfg.tailCnt;                                     // 尾块个数
    uint32_t tilingM = tilingData_->quantBmmv3TileTiling.matmulTiling.M / (cfg.rankDim - 1); // 整块M轴大小
    uint32_t tilingKa = tilingData_->quantBmmv3TileTiling.matmulTiling.Ka;  // 整块K轴大小
    uint32_t tailM = tilingData_->quantBmmv3TailTiling.matmulTiling.M / (cfg.rankDim - 1);   // 尾块M轴大小


    uint64_t tileSendCount = static_cast<uint64_t>(tilingM) * static_cast<uint64_t>(tilingKa);    // 整块总共搬运数据个数
    uint64_t tailSendCount = static_cast<uint64_t>(tailM) * static_cast<uint64_t>(tilingKa);      // 尾块总共搬运数据个数
    uint64_t stride = tileSendCount * static_cast<uint64_t>(tileCnt) +
                      tailSendCount * static_cast<uint64_t>(tailCnt); // 不同rank在数据块内的地址偏移
    uint8_t repeat = 1;                                               // 搬运重复次数
    uint64_t scalesCount = DequantBmm::Align(CeilDiv(tilingKa, MX_BLOCK_SIZE), EVEN_ALIGN) *
                           static_cast<uint64_t>(cfg.rankM); // scale1总共搬运数据个数

    GM_ADDR aGM = aGM_;
    GM_ADDR gatherOut = gatherOut_;

    if constexpr (DequantBmm::IsMxType<X2ScaleType>()) {
        // 全量scales1通信
        uint64_t scaleStride = scalesCount;
        handleList_[SCALE1_HANDLE_IDX] =
            hcclAllgather_.AllGather<true>(dequantGM1_, gatherDequantGM1_, scalesCount, dataType_, scaleStride, repeat);
    }

    for (uint32_t i = 0U; i < tileCnt; i++) {
        handleList_[i] = hcclAllgather_.AllGather<true>(aGM, gatherOut, tileSendCount, dataType_, stride, repeat);
        // 刷新aGM和目的地址
        aGM += tileSendCount * sizeof(AType);
        gatherOut += tileSendCount * sizeof(AType);
    }

    // 此时aGM和gatherOut为尾块开始地址
    for (uint32_t i = 0U; i < tailCnt; i++) {
        handleList_[tileCnt + i] = hcclAllgather_.AllGather<true>
                                   (aGM, gatherOut, tailSendCount, dataType_, stride, repeat);
        aGM += tailSendCount * sizeof(AType);
        gatherOut += tailSendCount * sizeof(AType);
    }
}

/**
 * @brief 本地块计算流程
 * @return void
*/
template <typename AType, typename BType, typename BiasType, typename X2ScaleType, typename CType, bool ATrans,
          bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType,
                                         X2ScaleType, CType, ATrans, BTrans>::QuantMatmulLocalCompute() {
    auto &&cfg = tilingData_->param;
    auto &&qBMMLocalTiling = tilingData_->quantBmmv3LocalTiling;     // QuantBatchMatmulV3TilingData 本地核切分tiling参数
    uint32_t tilingKa = tilingData_->quantBmmv3TileTiling.matmulTiling.Ka;  // 整块K轴大小

    MatMulASWKernel<
        AType, BType, X2ScaleType, BiasType, CType, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, ATrans, BTrans>
        op;
    GM_ADDR workspaceGM = workspaceGM_;
    // 本卡位置与rankId一致，C矩阵本卡矩阵首地址为rankid*cSize
    GM_ADDR cGM = cGM_ + static_cast<uint64_t>(rankId_) * static_cast<uint64_t>(cfg.rankM) *
                             static_cast<uint64_t>(qBMMLocalTiling.matmulTiling.N) *
                             sizeof(CType); // 卡上原始数据大小rankM
    if constexpr (DequantBmm::IsMxType<X2ScaleType>()) {
        // Workspace偏移留出scale目的地址
        workspaceGM += DequantBmm::Align(CeilDiv(tilingKa, MX_BLOCK_SIZE), EVEN_ALIGN) *
                       static_cast<uint64_t>(cfg.rankM) * sizeof(X2ScaleType) * static_cast<uint64_t>(cfg.rankDim);
    }
    op.Init(aGM_, bGM_, biasGM_, dequantGM2_, localDequantGM1_, cGM, workspaceGM, &qBMMLocalTiling, tPipe_);
    op.Process();
}

/**
 * @brief 其他片计算流程
 * @param count: 切分块数，isTail为true时，应传入tail尾块数，isTail为false时，应传入tile整块数
 * @param qMmv3Tiling: 核切分信息，isTail为true时，应传入tail切分信息，isTail为false时，应传入tile切分信息
 * @param isTail: 当前块是否为尾块
 * @return void
*/
template <typename AType, typename BType, typename BiasType, typename X2ScaleType,
          typename CType, bool ATrans, bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType, X2ScaleType,
                                         CType, ATrans, BTrans>::MatmulKernelComputeGather(
    GM_ADDR aGM, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams& qMmv3Tiling, uint32_t count, bool isLast, bool isTail)
{
    auto&& cfg = tilingData_->param;
    cfg.rankID = rankId_;
    if (GetBlockIdx() >= qMmv3Tiling.matmulTiling.usedCoreNum) {
        uint32_t shift = isTail ? cfg.tileCnt : 0;
        if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
            for (uint32_t i = 0; i < count; i++) {
                // 所有核Prepare，因此需要所有核做Wait
                hcclAllgather_.Wait(handleList_[i + shift]);
            }
        }
        return;
    }

    GetTPipePtr() -> Destroy();
    GetTPipePtr() -> Init();
    Mc2MatmulV3::Mc2QuantBatchMatmulASWKernel<
        AType, BType, X2ScaleType, BiasType, CType, CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, ATrans, BTrans> mmv3;
    GM_ADDR workspaceGM = workspaceGM_;
    if constexpr (DequantBmm::IsMxType<X2ScaleType>()) {
        // Workspace偏移留出scale1目的地址
        workspaceGM += DequantBmm::Align(CeilDiv(qMmv3Tiling.matmulTiling.Ka, MX_BLOCK_SIZE), EVEN_ALIGN) *
                       static_cast<uint64_t>(cfg.rankM) * sizeof(X2ScaleType) * static_cast<uint64_t>(cfg.rankDim);
        mmv3.Init(
            aGM, bGM_, biasGM_, dequantGM2_, gatherDequantGM1_, cGM_, workspaceGM, &qMmv3Tiling, GetTPipePtr(), cfg,
            isTail, true, preCoreNum_);
    } else {
        mmv3.Init(
            aGM, bGM_, biasGM_, dequantGM2_, dequantGM1_, cGM_, workspaceGM, &qMmv3Tiling, GetTPipePtr(), cfg, isTail,
            true, preCoreNum_);
    }

    uint32_t shift = isTail ? cfg.tileCnt : 0;
    for (uint32_t i = 0; i < count; i++) {
        GetTPipePtr() -> Destroy();
        GetTPipePtr() -> Init();
        // 开始计算前确认是否搬运完成，debug模式只做本片内计算，则不需要判断搬运状态
        if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
            hcclAllgather_.Wait(handleList_[i + shift]);
        }

        mmv3.UpdateSlice(i, isTail);
        mmv3.Process(isLast && (i == (count - 1)));
    }
    preCoreNum_ = mmv3.GetPreCoreNum();
    mmv3.MmEnd();
}

/**
 * @brief allgather quant bmm 计算流程
 * @return void
*/
template <typename AType, typename BType, typename BiasType,
          typename X2ScaleType, typename CType, bool ATrans, bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType,
                                         X2ScaleType, CType, ATrans, BTrans>::QuantMatmulCompute() {
    auto &&cfg = tilingData_->param;
    // 计算本片的块
    // 未用到的AIC核不需要参与计算
    if (GetBlockIdx() < tilingData_->quantBmmv3LocalTiling.matmulTiling.usedCoreNum) {
        QuantMatmulLocalCompute();
    }
    Mc2SyncAll<Mc2CoreType::ON_CUBE>();

    // 只做本卡计算时，aGM_从本地获取；计算远端卡数据时，aGM_为gather来的数据
    if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
        aGM_ = gatherOut_;
        if constexpr (DequantBmm::IsMxType<X2ScaleType>()) {
            // 计算其他片数据前先wait scale1
            hcclAllgather_.Wait(handleList_[SCALE1_HANDLE_IDX]);
        }
    }
    QuantMatmulGatherCompute();
}

template <
    typename AType, typename BType, typename BiasType, typename X2ScaleType, typename CType, bool ATrans, bool BTrans>
__aicore__ inline void
AllGatherQuantBmm<AType, BType, BiasType, X2ScaleType, CType, ATrans, BTrans>::QuantMatmulGatherCompute()
{
    auto &&cfg = tilingData_->param;
    // 处理gather主块
    if (debugMode_ == MC2_DEBUG_ONLY_CUBE) {
        MatmulKernelComputeGather(aGM_, tilingData_->quantBmmv3TileTiling,
                                  cfg.tileCnt, cfg.tailM ? false : true, false);
    } else {
        MatmulKernelComputeGather(gatherOut_, tilingData_->quantBmmv3TileTiling,
                                  cfg.tileCnt, cfg.tailM ? false : true, false);
    }

    if (cfg.tailM) {
        if (debugMode_ == MC2_DEBUG_ONLY_CUBE) {
            DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams tileTilingData = tilingData_->quantBmmv3TailTiling;
            uint64_t tileM = tileTilingData.matmulTiling.M;
            uint64_t tileN = tileTilingData.matmulTiling.N;
            uint64_t tileK = tileTilingData.matmulTiling.Ka;
            auto tailAGM = aGM_ + tileM * tileK * sizeof(AType) * (uint64_t)(cfg.tileCnt);
            MatmulKernelComputeGather(tailAGM, tilingData_->quantBmmv3TailTiling, cfg.tailCnt, true, true);
        } else {
            MatmulKernelComputeGather(gatherOut_, tilingData_->quantBmmv3TailTiling, cfg.tailCnt, true, true);
        }
    }
}

/**
 * @brief allgather quant bmm 后处理流程
 * @return void
*/
template <typename AType, typename BType, typename BiasType,
          typename X2ScaleType, typename CType, bool ATrans, bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType, X2ScaleType, CType, ATrans, BTrans>::PostProcess() {
    // 在最后一次计算完成后，单核清除状态位置，避免核间读写依赖
    // 防止某一个核执行过快清空RcvCnt, 增加全核同步
    Mc2SyncAll<Mc2CoreType::ON_CUBE>();
    if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
        hcclAllgather_.Finalize();
    }
}

/**
 * @brief allgather quant bmm主流程：通信+计算整块+计算尾块+后处理
 * @return void
*/
template <typename AType, typename BType, typename BiasType,
          typename X2ScaleType, typename CType, bool ATrans, bool BTrans>
__aicore__ inline void AllGatherQuantBmm<AType, BType, BiasType, X2ScaleType, CType, ATrans, BTrans>::Process() {
    // AIV核不进行通信和计算
    if ASCEND_IS_AIC {
        // 通信
        // debugmode 为 MC2_DEBUG_ONLY_CUBE时，只做本片计算不做通信
        if (debugMode_ != MC2_DEBUG_ONLY_CUBE) {
            AllGatherCommit();
        }
        // 计算
        QuantMatmulCompute();
        // 后处理：AIC全核同步+终止hcclserver
        PostProcess();
    }
}

}

#endif  // ALL_GATHER_QUANT_BMM_H
