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
 * \file matmul_all_reduce_quant_pertoken_comm_int8.h
 * \brief
 */
#ifndef MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_H
#define MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "../common.h"

#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_online_dynamic.h"
#include "matmul_all_reduce_add_x3.h"
#include "matmul_all_reduce_quant_perchannel.h"
#include "matmul_all_reduce_quant_mul_cast.h"
#include "matmul_all_reduce_dequant_perchannel.h"

namespace MatmulAllReduceImpl {
constexpr uint32_t MAX_HANDLE_ID_NUM = 16;
constexpr uint32_t NUM_TWO_PERTOKEN = 2;

using namespace AscendC;
using namespace MatmulAllReduceQuantMulCastImpl;
using namespace MatmulAllReduceQuantPerchannelImpl;
using namespace MatmulAllReduceDequantPerchannelImpl;
template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
class MatmulAllReduceQuantPertokenCommInt8
{
public:
    __aicore__ inline MatmulAllReduceQuantPertokenCommInt8()
    {}
    __aicore__ inline void Init(
        GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenScaleGM,
        GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
        Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InnerProcess(
        MmType& mmOp, uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams* mmTiling, uint32_t isAdd, uint32_t needUbBuffer,
        uint32_t padM, bool isTailFlag);
    __aicore__ inline void PrepareInit();
    __aicore__ inline uint32_t SendCountCheck(uint32_t prepareIndex);

    Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilingData_;
    TPipe* tPipe_;
    GM_ADDR aGM_;
    GM_ADDR bGM_;
    GM_ADDR biasGM_;
    GM_ADDR addGM_;
    GM_ADDR dequantScaleGM_;
    GM_ADDR pertokenScaleGM_;
    GM_ADDR commQuantScale1GM_;
    GM_ADDR commQuantScale2GM_;
    GM_ADDR cGM_;
    GM_ADDR workspaceGM_;
    GM_ADDR outGM_;
    GM_ADDR tempBuffWinOrGM_;
    GM_ADDR tempBuffGM_;
    GM_ADDR tempBuffDequantGM_;
    GM_ADDR reduceScatterInGM_;
    GM_ADDR reduceScatterOutGM_;
    GM_ADDR allGatherInGM_;
    GM_ADDR allGatherOutGM_;
    Hccl<HCCL_SERVER_TYPE_CCU> hccl_;
    bool notifyFlag_{false};

    // 仅在0核上使用
    AscendC::HcclHandle reduceScatterHandleId_[MAX_HANDLE_ID_NUM] = {0};
    AscendC::HcclHandle allGatherHandleId_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR reduceScatterSendGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allGatherSendGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR reduceScatterRecvGM_[MAX_HANDLE_ID_NUM] = {0};
    GM_ADDR allGatherRecvGM_[MAX_HANDLE_ID_NUM] = {0};
    int reduceScatterCommitIdx_ = 0;
    int reduceScatterWaitIdx_ = 0;
    int allGatherCommitIdx_ = 0;
    int allGatherWaitIdx_ = 0;
    // 所有核
    bool isSendTileFlag_ = false;
    uint32_t tilePadM_ = 0;
    uint32_t tailPadM_ = 0;
    uint32_t tilePadDataCnt_ = 0;
    uint32_t tailPadDataCnt_ = 0;
};

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR addGM, GM_ADDR dequantScaleGM, GM_ADDR pertokenScaleGM,
    GM_ADDR commQuantScale1GM, GM_ADDR commQuantScale2GM, GM_ADDR cGM, GM_ADDR workspaceGM,
    Mc2Tiling::QuantMatmulAllReduceTilingDataA5* tilingData, TPipe* tPipe)
{
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());
    OOMInit(context);
    hccl_.InitV2(GetHcclContext<0>(), tilingData);
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTiling));
    hccl_.SetCcTilingV2(offsetof(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, mc2CcTilingComm));
    cGM_ = cGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    aGM_ = aGM;
    bGM_ = bGM;
    biasGM_ = biasGM;
    addGM_ = addGM;
    dequantScaleGM_ = dequantScaleGM;
    pertokenScaleGM_ = pertokenScaleGM;
    commQuantScale1GM_ = commQuantScale1GM;
    commQuantScale2GM_ = commQuantScale2GM;
    workspaceGM_ = workspaceGM;
    outGM_ = cGM;

    reduceScatterInGM_ = workspaceGM_ + tilingData_->param.commWorkSpaceSize;     // reduceScatter输入
    allGatherInGM_ = reduceScatterInGM_ + tilingData_->param.commInt8WorkSpace;   // allGather输入
    allGatherOutGM_ = allGatherInGM_ + tilingData_->param.commInt8WorkSpace;      // allGather输出
    reduceScatterOutGM_ = allGatherOutGM_ + tilingData_->param.commInt8WorkSpace; // reduceScatter输出

    if ((GetBlockIdx() == 0) && (g_coreType == AscendC::AIV)) { // V核的0核下发通信任务
        notifyFlag_ = true;
    }
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline uint32_t MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType>::SendCountCheck(
    uint32_t prepareIndex)
{
    uint32_t sendCount = tilePadDataCnt_ / tilingData_->param.rankDim;
    if (prepareIndex >= tilingData_->param.tileCnt) {
        sendCount = tailPadDataCnt_ / tilingData_->param.rankDim;
    }
    return sendCount;
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType>::PrepareInit()
{
    auto&& mc2Tiling = tilingData_->param;
    uint32_t rankNum = mc2Tiling.rankDim;
    uint32_t tileM = tilingData_->tilematmulTiling.matmulTiling.M; // 头块 pad 大小更新
    tilePadM_ = tileM;
    if ((tileM % rankNum) != 0) { // 按照 M 来 pad
        tilePadM_ += rankNum - (tileM % rankNum);
    }
    uint32_t tailM = tilingData_->tailmatmulTiling.matmulTiling.M;
    tailPadM_ = tailM;
    if ((tailM % rankNum) != 0) {
        tailPadM_ += rankNum - (tailM % rankNum);
    }

    tilePadDataCnt_ = tilePadM_ * tilingData_->tilematmulTiling.matmulTiling.N; // 一个MM头块的数据个数
    tailPadDataCnt_ = tailPadM_ * tilingData_->tailmatmulTiling.matmulTiling.N; // 一个MM尾块的数据个数
    const uint64_t tempBufOffsetTilePadSingle = tilePadDataCnt_ / rankNum;
    const uint64_t tempBufOffsetTailPadSingle = tailPadDataCnt_ / rankNum;

    for (uint32_t i = 0U; i < mc2Tiling.tileCnt; ++i) { // 头块偏移
        const uint64_t indexOffsetTile = i * tilePadDataCnt_;
        reduceScatterSendGM_[i] = reduceScatterInGM_ + indexOffsetTile * sizeof(int8_t);
        reduceScatterRecvGM_[i] = reduceScatterOutGM_ + indexOffsetTile * sizeof(float);
        allGatherSendGM_[i] = allGatherInGM_ + indexOffsetTile * sizeof(int8_t);
        allGatherRecvGM_[i] = allGatherOutGM_ + indexOffsetTile * sizeof(int8_t);
    }
    for (uint32_t i = 0U; i < mc2Tiling.tailCnt; ++i) { // 尾块偏移
        const uint64_t indexOffsetTail = mc2Tiling.tileCnt * tilePadDataCnt_ + i * tailPadDataCnt_;
        reduceScatterSendGM_[mc2Tiling.tileCnt + i] = reduceScatterInGM_ + indexOffsetTail * sizeof(int8_t);
        reduceScatterRecvGM_[mc2Tiling.tileCnt + i] = reduceScatterOutGM_ + indexOffsetTail * sizeof(float);
        allGatherSendGM_[mc2Tiling.tileCnt + i] = allGatherInGM_ + indexOffsetTail * sizeof(int8_t);
        allGatherRecvGM_[mc2Tiling.tileCnt + i] = allGatherOutGM_ + indexOffsetTail * sizeof(int8_t);
    }

    if (g_coreType == AscendC::AIV) {
        uint32_t nowReduceScatterIdx = 0U;
        uint32_t nowAllGatherIdx = 0U;
        uint32_t numN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) / NUM_TWO_PERTOKEN;
        uint32_t numReN = (mc2Tiling.tileCnt + mc2Tiling.tailCnt) % NUM_TWO_PERTOKEN;
        for (uint32_t i = 0U; i < numN; ++i) { // 按总核数下发
            reduceScatterHandleId_[nowReduceScatterIdx] = hccl_.ReduceScatter<false>(
                reduceScatterSendGM_[nowReduceScatterIdx], reduceScatterRecvGM_[nowReduceScatterIdx],
                SendCountCheck(nowReduceScatterIdx), AscendC::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
            nowReduceScatterIdx++;
            reduceScatterHandleId_[nowReduceScatterIdx] = hccl_.ReduceScatter<false>(
                reduceScatterSendGM_[nowReduceScatterIdx], reduceScatterRecvGM_[nowReduceScatterIdx],
                SendCountCheck(nowReduceScatterIdx), AscendC::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
            nowReduceScatterIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.AllGather<false>(
                allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.AllGather<false>(
                allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }

        if (numReN != 0U) { // 余数下发
            reduceScatterHandleId_[nowReduceScatterIdx] = hccl_.ReduceScatter<false>(
                reduceScatterSendGM_[nowReduceScatterIdx], reduceScatterRecvGM_[nowReduceScatterIdx],
                SendCountCheck(nowReduceScatterIdx), AscendC::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
            nowReduceScatterIdx++;
            allGatherHandleId_[nowAllGatherIdx] = hccl_.AllGather<false>(
                allGatherSendGM_[nowAllGatherIdx], allGatherRecvGM_[nowAllGatherIdx], SendCountCheck(nowAllGatherIdx),
                AscendC::HCCL_DATA_TYPE_INT8, 0);
            nowAllGatherIdx++;
        }
    }
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType>::InnerProcess(
    MmType& mmOp, uint32_t tileCnt, DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams* mmTiling, uint32_t isAdd, uint32_t needUbBuffer,
    uint32_t padM, bool isTailFlag)
{
    const uint64_t aOffset = CalcShapeOffset(sizeof(xType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.Ka);
    const uint64_t cOffset = CalcShapeOffset(sizeof(YType), mmTiling->matmulTiling.M, mmTiling->matmulTiling.N);
    const uint64_t tempOffset = CalcShapeOffset(sizeof(int8_t), padM, mmTiling->matmulTiling.N);
    const uint64_t pertokenOffset = sizeof(float) * mmTiling->matmulTiling.M;
    for (uint32_t i = 0U; i < tileCnt; ++i) {
        tPipe_->Reset();
        mmOp.Init(
            aGM_, bGM_, dequantScaleGM_, nullptr, biasGM_, pertokenScaleGM_, cGM_, workspaceGM_, mmTiling, tPipe_);
        mmOp.Process();
        SyncAll<false>();
        if (isAdd) {
            MatmulAllReduceAddX3Kernel<YType>(
                cGM_, addGM_, cOffset / sizeof(YType), tilingData_->param.addX3UbCnt, tPipe_);
            addGM_ += cOffset;
            SyncAll<false>();
        }
        // bf16->int8 存在 reduceScatterInGM_
        MatmulAllReduceQuantPerchannelCommInt8<YType>(
            cGM_, commQuantScale1GM_, reduceScatterInGM_, tPipe_, mmTiling->matmulTiling.N, mmTiling->matmulTiling.M);
        SyncAll<false>();
        if (g_coreType == AscendC::AIV) {
            hccl_.Commit(reduceScatterHandleId_[reduceScatterCommitIdx_]);
            reduceScatterCommitIdx_++;
        }
        if (g_coreType == AscendC::AIV && isSendTileFlag_) { // 需要同步前一块
            if (notifyFlag_) {
                hccl_.Wait(reduceScatterHandleId_[reduceScatterWaitIdx_]);
            }
            SyncAll();
            if (isTailFlag && (i == 0U)) {
                MatmulAllReduceQuantMulCastCommInt8<YType>(
                    reduceScatterOutGM_, commQuantScale1GM_, commQuantScale2GM_, allGatherInGM_, tilePadM_,
                    mmTiling->matmulTiling.N, tPipe_, hccl_);
                reduceScatterOutGM_ += tilePadM_ * mmTiling->matmulTiling.N * sizeof(float);
                allGatherInGM_ += tilePadM_ * mmTiling->matmulTiling.N * sizeof(int8_t);
            } else {
                MatmulAllReduceQuantMulCastCommInt8<YType>(
                    reduceScatterOutGM_, commQuantScale1GM_, commQuantScale2GM_, allGatherInGM_, padM,
                    mmTiling->matmulTiling.N, tPipe_, hccl_);
                reduceScatterOutGM_ += padM * mmTiling->matmulTiling.N * sizeof(float);
                allGatherInGM_ += padM * mmTiling->matmulTiling.N * sizeof(int8_t);
            }
            SyncAll();
            hccl_.Commit(allGatherHandleId_[reduceScatterWaitIdx_]);
            reduceScatterWaitIdx_++;
        }
        // 避免MatmulAllReduceQuantMulCastCommInt8的vec计算和quant_batch_matmul_v3的vec计算发生数据覆盖，确保流水正常
        SyncAll<false>();
        isSendTileFlag_ = true;
        aGM_ += aOffset;
        cGM_ += cOffset;                  // 偏原始大小
        reduceScatterInGM_ += tempOffset; // 偏移 padM*N 大小
        pertokenScaleGM_ += pertokenOffset;
    }
}

template <typename xType, typename WType, typename YType, class MmType, Mc2CoreType CoreType>
__aicore__ inline void MatmulAllReduceQuantPertokenCommInt8<xType, WType, YType, MmType, CoreType>::Process()
{
    auto&& mc2Tiling = tilingData_->param;

    // reducescatter allgather Prepare + InnerProcess
    PrepareInit();
    MmType opTile;
    InnerProcess(
        opTile, mc2Tiling.tileCnt, &tilingData_->tilematmulTiling, mc2Tiling.isAdd, mc2Tiling.needUbBuffer, tilePadM_,
        false);
    if (mc2Tiling.tailM != 0U) {
        MmType opTail;
        InnerProcess(
            opTail, mc2Tiling.tailCnt, &tilingData_->tailmatmulTiling, mc2Tiling.isAdd, mc2Tiling.needUbBuffer,
            tailPadM_, true);
    }

    // 最后一次的 reduceScatter 任务没有 wait+mul/cast，以及最后一次的 allgather 任务没有下发
    if (g_coreType == AscendC::AIV) {
        hccl_.Wait(reduceScatterHandleId_[reduceScatterWaitIdx_]);
        SyncAll();
        uint32_t padM = tilePadM_;
        uint32_t lastN = tilingData_->tilematmulTiling.matmulTiling.N;
        if (mc2Tiling.tailM != 0U) {
            padM = tailPadM_;
            lastN = tilingData_->tailmatmulTiling.matmulTiling.N;
        }
        MatmulAllReduceQuantMulCastCommInt8<YType>(
            reduceScatterOutGM_, commQuantScale1GM_, commQuantScale2GM_, allGatherInGM_, padM, lastN, tPipe_, hccl_);
        SyncAll();
        hccl_.Commit(allGatherHandleId_[reduceScatterWaitIdx_]);
        reduceScatterWaitIdx_++;
    }

    if (g_coreType == AscendC::AIV) {
        // wait所有的allGather任务 + dequant
        const uint64_t outGmTileOffset =
            tilingData_->tilematmulTiling.matmulTiling.M * tilingData_->tilematmulTiling.matmulTiling.N * sizeof(YType);
        const uint64_t outGmTailOffset =
            tilingData_->tailmatmulTiling.matmulTiling.M * tilingData_->tailmatmulTiling.matmulTiling.N * sizeof(YType);
        for (uint32_t i = 0U; i < (mc2Tiling.tileCnt + mc2Tiling.tailCnt); ++i) { // 尾块偏移
            hccl_.Wait(allGatherHandleId_[i]);
            SyncAll();
            if (i < mc2Tiling.tileCnt) {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(
                    allGatherOutGM_, commQuantScale2GM_, outGM_, tPipe_, tilingData_->tilematmulTiling.matmulTiling.N,
                    tilingData_->tilematmulTiling.matmulTiling.M);
                allGatherOutGM_ += tilePadDataCnt_ * sizeof(int8_t);
                outGM_ += outGmTileOffset;
                SyncAll();
            } else {
                MatmulAllReduceDequantPerchannelCommInt8<YType>(
                    allGatherOutGM_, commQuantScale2GM_, outGM_, tPipe_, tilingData_->tailmatmulTiling.matmulTiling.N,
                    tilingData_->tailmatmulTiling.matmulTiling.M);
                allGatherOutGM_ += tailPadDataCnt_ * sizeof(int8_t);
                outGM_ += outGmTailOffset;
                SyncAll();
            }
        }
        hccl_.Finalize();
    }
}

#define INVOKE_BATCH_MATMUL_QUANT_PERTOKEN_COMM_INT8_IMPL(templateClass, coreType, scaleType, isATrans, isBTrans, ...)            \
    do {                                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(Mc2Tiling::QuantMatmulAllReduceTilingDataA5, tilingData, tilingGM);                \
        MC2GmAddrs addrs = {aGM, bGM, biasGM, addGM, cGM, workspaceGM, cGM};                                           \
        QuantGmAddrs quantAddrs = {nullptr, nullptr, nullptr, dequantGM, pertokenGM};                                  \
        using OpType = templateClass<                                                                                  \
            DTYPE_X1, DTYPE_X2, scaleType, DTYPE_BIAS, float, DTYPE_Y, X1_FORMAT, X2_FORMAT, Y_FORMAT, isATrans, isBTrans, \
            DTYPE_LOC_LOCAL, Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock, MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG>;                  \
        MatmulAllReduceQuantPertokenCommInt8<DTYPE_X1, DTYPE_X2, DTYPE_Y, OpType, coreType> op;                        \
        op.Init(                                                                                                       \
            aGM, bGM, biasGM, addGM, dequantGM, pertokenGM, commQuantScale1GM, commQuantScale2GM, cGM, userWS,         \
            &tilingData, &tPipe);                                                                                      \
        op.Process();                                                                                                  \
        tPipe.Destroy();                                                                                               \
    } while (0)
} // namespace MatmulAllReduceImpl
#endif // MATMUL_ALL_REDUCE_QUANT_PERTOKEN_COMM_INT8_H