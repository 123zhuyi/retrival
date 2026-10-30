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
 * \file stateless_random_normal_v2.h
 * \brief
 */

#ifndef STATELESS_RANDOM_CHOICE_WITH_MASK
#define STATELESS_RANDOM_CHOICE_WITH_MASK

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace StatelessRandomChoiceWithMask {
using namespace AscendC;

constexpr uint32_t RIGHT_SHIFT = 32;
constexpr uint16_t QUE_NUM = 2;
constexpr uint16_t ALG_KEY_SIZE = 2;
constexpr uint16_t ALG_COUNTER_SIZE = 4;
constexpr uint16_t MAX_INPUT_DIM = 5;
constexpr int64_t CORE_ALIGN_SIZE = 512;
constexpr uint16_t CORE_THREAD_NUM = 512;
constexpr uint16_t RESULT_ELEMENT_CNT = 4;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t PER_LOOP_ROWS = 32;
constexpr int64_t ALIGNMENT_32 = 32;
constexpr int64_t PrefixUbSize = 20 * 1024; // 每次计算前缀和使用的ub为8k，并行预留20k
static constexpr int64_t Y_LENGTH_ADDR = 1;
static constexpr int64_t Y_DIM_ADDR = 2;
static constexpr int64_t MASK_DIM_ADDR = 9;
static constexpr int64_t MASK_LENGTH_ADDR = 10;

class StatelessRandomChoiceWithMask {
public:
    __aicore__ inline StatelessRandomChoiceWithMask(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR count, GM_ADDR y, GM_ADDR mask, GM_ADDR shapeOut, GM_ADDR workspace,
                                const StatelessRandomChoiceWithMaskSimtTilingData* __restrict tilingData, TPipe* pipe);
    __aicore__ inline void InitKeyAndCounter();
    __aicore__ inline void GenRandomData(int64_t outputNonzeroCount);
    __aicore__ inline void Process();
    __aicore__ inline void FisherYatesShuffleAndSplit(int64_t outputNonzeroCount, int64_t outputLength);
    __aicore__ inline void Skip(const uint64_t count);
    __aicore__ inline void CopyRandomToWorkspace(const int64_t offset, const int64_t count);
    __aicore__ inline void CopyInCount(const int64_t offset, const int64_t count);
    __aicore__ inline void ComputePrefixSum(const int64_t rows, const int64_t cols, const int64_t maxValue);
    __aicore__ inline void CopyOutCount(const int64_t offset, const int64_t count);

private:
    TPipe* pipe_;
    GlobalTensor<bool> inputGM_;
    GlobalTensor<int32_t> countGM_;
    GlobalTensor<int32_t> outputGM_;
    GlobalTensor<bool> maskGM_;
    GlobalTensor<uint64_t> outShapeGM_;

    GlobalTensor<int64_t> workspaceNoZeroCount_;
    GlobalTensor<int32_t> workSpaceOutput_;
    GlobalTensor<uint32_t> workspaceRandomData_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queOut_;
    TBuf<QuePosition::VECCALC> xBuf_;
    const StatelessRandomChoiceWithMaskSimtTilingData* tiling_;
    int64_t threadProNum_ = 0;
    uint32_t blockIdx_ = 0;
    int32_t count_ = 0;
    uint32_t key_[ALG_KEY_SIZE] = {0};
    uint32_t counter_[ALG_COUNTER_SIZE] = {0};
};

__simt_vf__ LAUNCH_BOUND(CORE_THREAD_NUM) __aicore__
    inline void SimtComputeNonZeroCount(__gm__ bool* inputGM, __gm__ volatile int64_t* workspaceNonZeroCount,
                                        int64_t inputSize, int64_t perThreadCalcCount, int64_t count)
{
    int64_t threadOffset = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t startIdx = threadOffset * perThreadCalcCount;
    int64_t endIdx = startIdx + perThreadCalcCount;
    if (startIdx + perThreadCalcCount > inputSize) {
        endIdx = inputSize;
    }

    int64_t zeroCount = 0;
    for (int64_t inputIdx = startIdx; inputIdx < endIdx; inputIdx++) {
        if (count > 0 && zeroCount >= count) {
            break;
        }
        if (inputGM[inputIdx] == true) {
            zeroCount++;
        }
    }
    workspaceNonZeroCount[0] = 0;
    workspaceNonZeroCount[threadOffset + 1] = zeroCount;
}

__simt_vf__ LAUNCH_BOUND(1) __aicore__
    inline void SimtPrefixSum(__gm__ int64_t* workspaceNonZeroCount, int32_t totalPart)
{
    if (threadIdx.x != 0) {
        return;
    }
    for (int64_t idx = 1; idx <= totalPart; idx++) {
        workspaceNonZeroCount[idx] = workspaceNonZeroCount[idx] + workspaceNonZeroCount[idx - 1];
    }
}

__simt_vf__ LAUNCH_BOUND(CORE_THREAD_NUM) __aicore__
    inline void SimtPadOutput(__gm__ volatile int32_t* outputGM, __gm__ volatile bool* maskGM, int64_t blockNum,
                              uint32_t inputDim, int64_t outputLength, int64_t outputNonzeroCount)
{
    int64_t threadOffset = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint64_t idx = outputNonzeroCount + threadOffset; idx < outputLength; idx += blockNum * blockDim.x) {
        for (uint64_t dim = 0; dim < inputDim; dim++) {
            outputGM[idx * inputDim + dim] = 0;
        }
        maskGM[idx] = false;
    }
}

__simt_vf__ LAUNCH_BOUND(CORE_THREAD_NUM) __aicore__
    inline void SimtCalcOutput(__gm__ volatile bool* inputGM, __gm__ volatile int64_t* workspaceNonZeroCount,
                               __gm__ volatile int32_t* outputGM, __gm__ volatile bool* maskGM, int64_t blockNum,
                               int64_t outputLength, int64_t outputNonzeroCount, int64_t perThreadCalcCount)
{
    int64_t threadOffset = blockIdx.x * blockDim.x + threadIdx.x;

    int64_t preIdx = workspaceNonZeroCount[threadOffset];
    int64_t curIdx = workspaceNonZeroCount[threadOffset + 1];
    int64_t idx = threadOffset * perThreadCalcCount;
    int64_t zeroCount = 0;
    while (preIdx < curIdx && preIdx < outputNonzeroCount) {
        if (inputGM[idx] == true) {
            outputGM[preIdx] = idx;
            maskGM[preIdx] = true;
            preIdx++;
        }
        idx++;
    }
}

__simt_vf__ LAUNCH_BOUND(CORE_THREAD_NUM) __aicore__
    inline void SplitDim(__gm__ volatile int32_t* srcGM, __gm__ volatile int32_t* dstGM, int64_t outputNonzeroCount,
                         int64_t blockNum, int64_t inputDim, uint32_t m0, uint32_t s0, uint32_t m1, uint32_t s1,
                         uint32_t m2, uint32_t s2, uint32_t m3, uint32_t s3, uint32_t m4, uint32_t s4, uint32_t c0,
                         uint32_t c1, uint32_t c2, uint32_t c3)
{
    uint32_t m[5] = {m0, m1, m2, m3, m4};
    uint32_t s[5] = {s0, s1, s2, s3, s4};
    uint32_t c[5] = {c0, c1, c2, c3, 1};

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < outputNonzeroCount; idx += blockNum * blockDim.x) {
        uint32_t realIdx = srcGM[idx];
        for (int64_t dim = 0; dim < inputDim; dim++) {
            dstGM[idx * inputDim + dim] = Simt::UintDiv(realIdx, m[dim], s[dim]);
            realIdx -= dstGM[idx * inputDim + dim] * c[dim];
        }
    }
}

__simt_vf__ LAUNCH_BOUND(1) __aicore__
    inline void SimtFisherYatesShuffle(__gm__ uint32_t* workspaceRandomData, __gm__ volatile int32_t* outputGM,
                                       __gm__ uint64_t* outShapeGM, int64_t inputDim, int64_t outputNonzeroCount,
                                       int64_t outputCount)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int32_t idx = outputNonzeroCount - 1; idx > 0; idx--) {
        uint32_t randomIdx = workspaceRandomData[idx] % static_cast<uint32_t>(idx);
        int32_t tmpIndex = outputGM[idx];
        outputGM[idx] = outputGM[randomIdx];
        outputGM[randomIdx] = tmpIndex;
    }
    outShapeGM[0] = 0x80000002;
    outShapeGM[Y_LENGTH_ADDR] = outputCount;
    outShapeGM[Y_DIM_ADDR] = inputDim;
    outShapeGM[MASK_DIM_ADDR] = 1;
    outShapeGM[MASK_LENGTH_ADDR] = outputCount;
}

__aicore__ inline void StatelessRandomChoiceWithMask::Init(
    GM_ADDR x, GM_ADDR count, GM_ADDR y, GM_ADDR mask, GM_ADDR shapeOut, GM_ADDR workspace,
    const StatelessRandomChoiceWithMaskSimtTilingData* __restrict tilingData, TPipe* pipe)
{
    tiling_ = tilingData;
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    if (blockIdx_ > tiling_->blockNum) {
        return;
    }
    threadProNum_ = Ops::Base::CeilDiv(tiling_->normalCoreProNum, static_cast<int64_t>(CORE_THREAD_NUM));

    inputGM_.SetGlobalBuffer((__gm__ bool*)x);
    countGM_.SetGlobalBuffer((__gm__ int32_t*)count);
    outputGM_.SetGlobalBuffer((__gm__ int32_t*)y);
    maskGM_.SetGlobalBuffer((__gm__ bool*)mask);
    outShapeGM_.SetGlobalBuffer((__gm__ uint64_t*)shapeOut);

    workspaceNoZeroCount_.SetGlobalBuffer((__gm__ int64_t*)workspace);
    workSpaceOutput_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                     tiling_->noZeroWorkspaceSize * sizeof(int64_t) / sizeof(int32_t));
    workspaceRandomData_.SetGlobalBuffer((__gm__ uint32_t*)workspace +
                                         tiling_->noZeroWorkspaceSize * sizeof(int64_t) / sizeof(int32_t) +
                                         tiling_->randomWorkspaceSize);

    count_ = countGM_.GetValue(0);
    pipe_->InitBuffer(queOut_, DOUBLE_BUFFER, (tiling_->ubSize - PrefixUbSize) / DOUBLE_BUFFER);
    pipe_->InitBuffer(xBuf_, PrefixUbSize);
}

__aicore__ inline void StatelessRandomChoiceWithMask::InitKeyAndCounter()
{
    key_[0] = static_cast<uint32_t>(tiling_->seed);
    key_[1] = static_cast<uint32_t>(tiling_->seed >> RIGHT_SHIFT);
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = static_cast<uint32_t>(tiling_->offset);
    counter_[3] = static_cast<uint32_t>(tiling_->offset >> RIGHT_SHIFT);
}

__aicore__ inline void StatelessRandomChoiceWithMask::GenRandomData(int64_t outputNonzeroCount)
{
    if (outputNonzeroCount <= 0) {
        return;
    }
    InitKeyAndCounter();
    auto valueSize = sizeof(uint32_t);
    int64_t perCoreHandleRandomAlign = Ops::Base::CeilAlign(Ops::Base::CeilDiv(outputNonzeroCount, tiling_->blockNum),
                                                            CORE_ALIGN_SIZE);
    int64_t simdBlockNum = Ops::Base::CeilDiv(outputNonzeroCount, perCoreHandleRandomAlign);
    if (blockIdx_ >= simdBlockNum) {
        return;
    }

    auto alignFactor = Ops::Base::GetUbBlockSize() / valueSize;
    int64_t ubFactor = Ops::Base::FloorAlign((tiling_->ubSize - PrefixUbSize) / (valueSize * DOUBLE_BUFFER),
                                             alignFactor);
    int64_t blockFactor = Ops::Base::CeilDiv(perCoreHandleRandomAlign, ubFactor);
    int64_t tailUbFactor = perCoreHandleRandomAlign - (blockFactor - 1) * ubFactor;
    auto tailCoreHandleRandom = outputNonzeroCount - (simdBlockNum - 1) * perCoreHandleRandomAlign;
    if (blockIdx_ == simdBlockNum - 1) {
        blockFactor = Ops::Base::CeilDiv(tailCoreHandleRandom, ubFactor);
        tailUbFactor = tailCoreHandleRandom - (blockFactor - 1) * ubFactor;
    }
    int64_t blockOffSet = blockIdx_ * perCoreHandleRandomAlign;

    int64_t groupCnt = blockOffSet / RESULT_ELEMENT_CNT;
    Skip(groupCnt);
    for (auto idx = 0; idx < blockFactor; idx++) {
        int64_t currUbTilingSize = ubFactor;
        if (idx == blockFactor - 1) {
            currUbTilingSize = tailUbFactor;
        }

        LocalTensor<uint32_t> philoxRes = queOut_.AllocTensor<uint32_t>();

        PhiloxRandom<10>(philoxRes, {key_[0], key_[1]}, {counter_[0], counter_[1], counter_[2], counter_[3]},
                         currUbTilingSize);
        queOut_.EnQue(philoxRes);
        CopyRandomToWorkspace(blockOffSet + idx * ubFactor, currUbTilingSize);
        groupCnt = currUbTilingSize / RESULT_ELEMENT_CNT;
        Skip(groupCnt);
    }
}

__aicore__ inline void StatelessRandomChoiceWithMask::FisherYatesShuffleAndSplit(int64_t outputNonzeroCount,
                                                                                 int64_t outputLength)
{
    auto outGM = outputGM_;
    if (tiling_->inputDim != 1) {
        outGM = workSpaceOutput_;
    }
    asc_vf_call<SimtCalcOutput>(dim3{CORE_THREAD_NUM}, (__gm__ volatile bool*)(inputGM_.GetPhyAddr()),
                                (__gm__ volatile int64_t*)(workspaceNoZeroCount_.GetPhyAddr()),
                                (__gm__ volatile int32_t*)(outGM.GetPhyAddr()),
                                (__gm__ volatile bool*)(maskGM_.GetPhyAddr()), tiling_->blockNum, outputLength,
                                outputNonzeroCount, threadProNum_);
    asc_vf_call<SimtPadOutput>(dim3{CORE_THREAD_NUM}, (__gm__ volatile int32_t*)(outputGM_.GetPhyAddr()),
                               (__gm__ volatile bool*)(maskGM_.GetPhyAddr()), tiling_->blockNum, tiling_->inputDim,
                               outputLength, outputNonzeroCount);
    GenRandomData(outputNonzeroCount);
    SyncAll();
    // 6、洗牌算法 - 单核、单线程，后续性能考虑，可改为多线程
    asc_vf_call<SimtFisherYatesShuffle>(
        dim3{1}, (__gm__ uint32_t*)(workspaceRandomData_.GetPhyAddr()), (__gm__ volatile int32_t*)(outGM.GetPhyAddr()),
        (__gm__ uint64_t*)(outShapeGM_.GetPhyAddr()), tiling_->inputDim, outputNonzeroCount, outputLength);

    if (tiling_->inputDim != 1) {
        SyncAll();
        uint32_t strides[MAX_INPUT_DIM] = {1, 1, 1, 1, 1};
        uint32_t m[MAX_INPUT_DIM] = {0, 0, 0, 0, 0};
        uint32_t s[MAX_INPUT_DIM] = {0, 0, 0, 0, 0};
        for (int32_t idx = tiling_->inputDim - 2; idx >= 0; idx--) {
            strides[idx] = strides[idx + 1] * tiling_->inputShape[idx + 1];
        }
        for (int32_t idx = 0; idx < tiling_->inputDim; idx++) {
            GetUintDivMagicAndShift(m[idx], s[idx], strides[idx]);
        }
        // 如果是多轴情况，需要做拆轴
        asc_vf_call<SplitDim>(dim3{CORE_THREAD_NUM}, (__gm__ volatile int32_t*)(workSpaceOutput_.GetPhyAddr()),
                              (__gm__ volatile int32_t*)(outputGM_.GetPhyAddr()), outputNonzeroCount, tiling_->blockNum,
                              tiling_->inputDim, m[0], s[0], m[1], s[1], m[2], s[2], m[3], s[3], m[4], s[4], strides[0],
                              strides[1], strides[2], strides[3]);
    }
}

__aicore__ inline void StatelessRandomChoiceWithMask::Process()
{
    if (blockIdx_ >= tiling_->blockNum) {
        return;
    }
    if (tiling_->inputSize == 0) {
        __gm__ uint64_t* outShapeGM = (__gm__ uint64_t*)(outShapeGM_.GetPhyAddr());
        outShapeGM[0] = 0x80000002;
        outShapeGM[Y_LENGTH_ADDR] = 0;
        outShapeGM[Y_DIM_ADDR] = tiling_->inputDim;
        outShapeGM[MASK_DIM_ADDR] = 1;
        outShapeGM[MASK_LENGTH_ADDR] = 0;
        return;
    }

    asc_vf_call<SimtComputeNonZeroCount>(dim3{CORE_THREAD_NUM}, (__gm__ bool*)(inputGM_.GetPhyAddr()),
                                         (__gm__ volatile int64_t*)(workspaceNoZeroCount_.GetPhyAddr()),
                                         tiling_->inputSize, threadProNum_, count_);
    SyncAll();

    int64_t colsTotal = tiling_->m;
    int64_t rowsTotal = tiling_->n;
    int64_t loopCount = rowsTotal / ALIGNMENT_32;
    if (blockIdx_ == 0) {
        int64_t maxValue = 0;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t offset = i * ALIGNMENT_32 * ALIGNMENT_32;
            CopyInCount(offset, ALIGNMENT_32 * ALIGNMENT_32);
            ComputePrefixSum(ALIGNMENT_32, ALIGNMENT_32, maxValue);
            LocalTensor<int64_t> xLocal = xBuf_.Get<int64_t>();
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
            maxValue = xLocal.GetValue(ALIGNMENT_32 * ALIGNMENT_32 - 1);
            CopyOutCount(offset, ALIGNMENT_32 * ALIGNMENT_32);
        }
    }
    SyncAll();
    int64_t nonzeroCount = workspaceNoZeroCount_.GetValue(tiling_->noZeroCalcCount);
    if (count_ == 0) {
        count_ = nonzeroCount;
    }
    int64_t outputLength = count_;
    int64_t outputNonzeroCount = count_;
    if (nonzeroCount < count_) {
        outputNonzeroCount = nonzeroCount;
    }

    FisherYatesShuffleAndSplit(outputNonzeroCount, outputLength);
}

__aicore__ inline void StatelessRandomChoiceWithMask::Skip(const uint64_t count)
{
    const uint32_t countLo = static_cast<uint32_t>(count);
    uint32_t countHi = static_cast<uint32_t>(count >> 32);

    counter_[0] += countLo;
    if (counter_[0] < countLo) {
        ++countHi;
    }
    counter_[1] += countHi;
    if (counter_[1] < countHi) {
        if (++counter_[2] == 0) {
            ++counter_[3];
        }
    }
}

__aicore__ inline void StatelessRandomChoiceWithMask::CopyInCount(const int64_t offset, const int64_t count)
{
    LocalTensor<int64_t> countLocal = xBuf_.Get<int64_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(count * sizeof(int64_t));
    DataCopyPadExtParams<int64_t> padParams = {false, 0, 0, 0};
    DataCopyPad(countLocal, workspaceNoZeroCount_[offset], copyParams, padParams);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
}

__aicore__ inline void StatelessRandomChoiceWithMask::ComputePrefixSum(const int64_t rows, const int64_t cols,
                                                                       const int64_t maxValue)
{
    LocalTensor<int64_t> countLocal = xBuf_.Get<int64_t>();
    __local_mem__ int64_t* countLocalAddr = (__local_mem__ int64_t*)countLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<uint64_t> sequence;
        AscendC::Reg::RegTensor<uint64_t> index;
        AscendC::Reg::Arange((AscendC::Reg::RegTensor<int64_t>&)(sequence), 0);
        AscendC::Reg::RegTensor<int64_t> reg0;
        AscendC::Reg::RegTensor<int64_t> reg1;
        AscendC::Reg::RegTensor<int64_t> tmpReg;

        AscendC::Reg::MaskReg pregFull = AscendC::Reg::CreateMask<int64_t, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::Duplicate(tmpReg, 0, pregFull);
        AscendC::Reg::Muls(sequence, sequence, cols, pregFull);

        for (uint16_t j = 0; j < static_cast<uint16_t>(cols); j++) {
            AscendC::Reg::Adds(index, sequence, (uint64_t)j, pregFull);
            AscendC::Reg::DataCopyGather(reg0, countLocalAddr, index, pregFull);
            AscendC::Reg::Add(reg1, reg0, tmpReg, pregFull);
            AscendC::Reg::Copy(tmpReg, reg1, pregFull);
            AscendC::Reg::DataCopyScatter(countLocalAddr, reg1, index, pregFull);
        }

        AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

        AscendC::Reg::DataCopy(reg1, countLocalAddr);
        AscendC::Reg::Adds(reg1, reg1, maxValue, pregFull);
        AscendC::Reg::DataCopy(countLocalAddr, reg1, pregFull);
        __local_mem__ int64_t* countAddr = countLocalAddr;
        for (uint16_t i = 1; i < static_cast<uint16_t>(rows); i++) {
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::SCALAR_LOAD>();
            int64_t max = countLocalAddr[i * cols - 1];
            int64_t maxValue1 = max;
            countAddr += ALIGNMENT_32;
            AscendC::Reg::DataCopy(reg1, countAddr);
            AscendC::Reg::Adds(reg1, reg1, maxValue1, pregFull);
            AscendC::Reg::DataCopy(countAddr, reg1, pregFull);
        }
    }
}

__aicore__ inline void StatelessRandomChoiceWithMask::CopyOutCount(const int64_t offset, const int64_t count)
{
    LocalTensor<int64_t> countLocal = xBuf_.Get<int64_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(count * sizeof(int64_t));
    DataCopyPad(workspaceNoZeroCount_[offset], countLocal, copyParams);
}

__aicore__ inline void StatelessRandomChoiceWithMask::CopyRandomToWorkspace(const int64_t offset, const int64_t count)
{
    LocalTensor<uint32_t> yOutput = queOut_.DeQue<uint32_t>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(count * sizeof(uint32_t));
    DataCopyPad(workspaceRandomData_[offset], yOutput, copyParams);
    queOut_.FreeTensor(yOutput);
}
} // namespace StatelessRandomChoiceWithMask

#endif // STATELESS_RANDOM_CHOICE_WITH_MASK
