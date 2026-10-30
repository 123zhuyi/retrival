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
 * \file reduce_var_sch.h
 * \brief reduce var schedule
 */

#ifndef _REDUCE_VAR_SCH_H_
#define _REDUCE_VAR_SCH_H_
#include "atvoss/reduce/reduce_sch_aux_util.h"
#include "reduce_var_welford.h"
#include "reduce_var_twopass.h"

namespace ReduceOpTmpl
{
template <typename DataType, typename PromoteDataType, bool isContiguous, uint32_t PatternID, uint32_t LoopARCount,
          uint32_t LoopInnerARCount, bool isStd = false>
class ReduceVarSch
{
public:
    constexpr static Ops::Base::ReduceOpTmpl::ReduceSchLoopInfo SchLoopInfo =
        Ops::Base::ReduceOpTmpl::GetSchLoopInfo<PatternID, LoopARCount, LoopInnerARCount>();
    using Pattern = typename Ops::Base::ReduceOpTmpl::__reducePattern::GetPattern<SchLoopInfo.patternID>::T;
    using InnerPattern = typename Ops::Base::ReduceOpTmpl::__reducePattern::GetPattern<SchLoopInfo.innerPatternID>::T;
    constexpr static int32_t Dim = Pattern::Dim;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = Ops::Base::GetVRegSize() / sizeof(PromoteDataType);
    constexpr static int32_t VL_LENGTH_B = Ops::Base::GetVRegSize();
    constexpr static uint64_t BLOCK_SIZE_BYTE = Ops::Base::GetUbBlockSize();
    constexpr static int32_t POST_BUF_SIZE = 8 * 1024;
    constexpr static int32_t FLOAT32_INF = 0x7F800000;  // inf
    constexpr static uint32_t MAX_NDDMA_DIM = 4;
    constexpr static uint32_t WELFORD_GROUP_NUM = 8;
    constexpr static uint32_t MAX_INNER_A = 512;  // Bytes, 按PromoteDataType计算
    constexpr static uint32_t MAX_INNER_A_NUM = MAX_INNER_A / sizeof(PromoteDataType);
    constexpr static uint32_t GROUP_CACHE_BUF_SIZE = (WELFORD_GROUP_NUM + 1) * MAX_INNER_A;

private:
    TPipe* pipe_ = nullptr;
    TBuf<> buf_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    GlobalTensor<DataType> inputGM_;
    GlobalTensor<DataType> varGM_;
    GlobalTensor<DataType> meanGM_;
    GlobalTensor<PromoteDataType> workspace_;

    LocalTensor<PromoteDataType> tMeanTensor_;  // welford cache var buf
    LocalTensor<PromoteDataType> tVarTensor_;   // welford cache mean buf
    LocalTensor<PromoteDataType> tDichAddTensor_;
    LocalTensor<PromoteDataType> tCountTensor_;
    LocalTensor<PromoteDataType> tGroupMeanTensor_;  // welford group cache mean buf
    LocalTensor<PromoteDataType> tGroupVarTensor_;   // welford group cache var buf

    const ReduceVarTilingData* tiling_ = nullptr;
    const ReduceVarTilingDataStru* tilingOp_ = nullptr;

private:
    uint64_t lastRAxisLen_ = 0;
    uint64_t lastRAxisLenAlign_ = 0;

    int64_t blockIdx_ = 0;
    int64_t basicBlockLen_ = 0;

    uint64_t loopAStartIndex_ = 0;
    uint64_t loopAEndIndex_ = 0;
    uint64_t loopAAxisStep_ = 0;
    uint64_t ubFactorA_ = 0;

    uint64_t loopRStartIndex_ = 0;
    uint64_t loopREndIndex_ = 0;
    uint64_t lastRAxisNum_ = 0;
    uint64_t loopRAxisStep_ = 0;
    uint64_t splitRAxisTail_ = 0;  // R轴的ub切分的尾块
    uint64_t ubFactorR_ = 0;

    int64_t rCount_ = 0;             // r loop count
    int64_t lastReduceTailR_ = 0;    // ubRfactor存在尾块时, welford尾块的长度
    uint32_t loopLastRCnt_ = 1;      // 单次ub内，总共包含多少个lastR
    uint32_t loopWelfTailRCnt_ = 1;  // 单次ub内，welford尾块包含多少个lastR

    uint64_t aOutBurstLen_ = 0;
    uint64_t aOutNBurst_ = 0;

    uint32_t rCntGroupWelford_[WELFORD_GROUP_NUM + 1] = {0};
    int32_t rCntGroupIdx_ = 0;
    int32_t dstGroupGroupIdx_ = 0;
    bool isInvert_ = false;

    struct {
        uint64_t start = 0;
        uint64_t stride = 1;  // 拷贝步长
    } iterAddr_[Dim];

public:
    __aicore__ inline explicit ReduceVarSch(const ReduceVarTilingData* tiling)
    {
        tiling_ = tiling;
        tilingOp_ = &(tiling_->reduceOpTiling);
    };

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR workspace)
    {
        pipe_ = pipeIn;
        blockIdx_ = GetBlockIdx();
        basicBlockLen_ = tilingOp_->basicBlock;

        inputGM_.SetGlobalBuffer((__gm__ DataType*)x);
        varGM_.SetGlobalBuffer((__gm__ DataType*)var);
        if (mean != nullptr) {
            meanGM_.SetGlobalBuffer((__gm__ DataType*)mean);
        }
        workspace_.SetGlobalBuffer((__gm__ PromoteDataType*)workspace);

        pipe_->InitBuffer(inputQueue_, BUFFER_NUM, basicBlockLen_);
        // resultBlock 是按fp32预留的, 为单个mean/var的大小，mean + var, 开db
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, tilingOp_->resultBlock * Ops::Base::ReduceOpTmpl::CONST2);

        int64_t totalBufSize = 0;
        int64_t tmpCacheBufNum = 0;
        if constexpr (IsSameType<PromoteDataType, DataType>::value) {
            // dichotomyAddBuf_ 后续可以优化成只有一半
            //               meanBuf_         varBuf_      dichotomyAddBuf_    tCountBuff_
            totalBufSize = basicBlockLen_ + basicBlockLen_ + basicBlockLen_ + basicBlockLen_ / Ops::Base::ReduceOpTmpl::CONST2;
            tmpCacheBufNum = basicBlockLen_ / sizeof(PromoteDataType);
        } else {
            totalBufSize = (basicBlockLen_ + basicBlockLen_ + basicBlockLen_) * Ops::Base::ReduceOpTmpl::CONST2 + basicBlockLen_;
            tmpCacheBufNum = basicBlockLen_ * Ops::Base::ReduceOpTmpl::CONST2 / sizeof(PromoteDataType);
        }

        totalBufSize += GROUP_CACHE_BUF_SIZE * Ops::Base::ReduceOpTmpl::CONST2;  // CONST2: groupMeanBuf_ + groupVarBuf_
        pipe_->InitBuffer(buf_, totalBufSize);

        tMeanTensor_ = buf_.Get<PromoteDataType>();
        tVarTensor_ = tMeanTensor_[tmpCacheBufNum];
        tDichAddTensor_ = tVarTensor_[tmpCacheBufNum];
        tCountTensor_ = tDichAddTensor_[tmpCacheBufNum];
        tGroupMeanTensor_ = tCountTensor_[tmpCacheBufNum / Ops::Base::ReduceOpTmpl::CONST2];
        tGroupVarTensor_ = tGroupMeanTensor_[GROUP_CACHE_BUF_SIZE / sizeof(PromoteDataType)];

        for (uint64_t i = 0; i < Dim; i++) {
            iterAddr_[i].stride = tilingOp_->shape[i];
        }

        lastRAxisLen_ = tilingOp_->shape[Dim - 1];
        lastRAxisLenAlign_ = Ops::Base::CeilAlign(lastRAxisLen_, (BLOCK_SIZE_BYTE / sizeof(DataType)));
    }

    __aicore__ inline void ReInitWelfordGroups()
    {
        for (int32_t i = 0; i < WELFORD_GROUP_NUM + 1; i++) {
            rCntGroupWelford_[i] = 0;
        }

        rCntGroupIdx_ = 0;
        dstGroupGroupIdx_ = 0;
        isInvert_ = false;
    }

    __aicore__ inline void Process()
    {
        if constexpr (Ops::Base::ReduceOpTmpl::IsBlockCutA<&SchLoopInfo>()) {
            ProcessNormal();
        } else {
            ProcessGroupPhase1();
            SyncAll();
            ProcessGroupPhase2();
        }
    }

    __aicore__ inline void ProcessNormal()
    {
        SetLoopRangeNormal();
        rCount_ = tilingOp_->factorRCntPerCore;
        for (uint64_t i = loopAStartIndex_; i < loopAEndIndex_; i++) {
            CalcIterA<SchLoopInfo.loopACount>(i);
            IterateInnerA<0, SchLoopInfo.loopInnerACount>();
        }
    }

    __aicore__ inline void ProcessGroupPhase1()
    {
        SetLoopRangeGroup();
        rCount_ = loopREndIndex_ - loopRStartIndex_;
        IterateInnerA<0, SchLoopInfo.loopInnerACount>();
    }

    __aicore__ inline void ProcessGroupPhase2()
    {
        ubFactorA_ = ELEMENT_ONE_REPEAT_COMPUTE;
        ubFactorR_ = tilingOp_->groupR;

        int32_t blockIdx = GetBlockIdx();
        int64_t factorATotalCnt = Ops::Base::CeilDiv(tilingOp_->outSize, static_cast<uint64_t>(ubFactorA_));
        int64_t factorACntPerCore = Ops::Base::CeilDiv(factorATotalCnt, static_cast<int64_t>(tilingOp_->coreNum));

        int64_t loopAStartIndex = blockIdx * factorACntPerCore;
        int64_t loopAEndIndex = loopAStartIndex + factorACntPerCore;
        if (unlikely(loopAStartIndex > factorATotalCnt)) {
            loopAStartIndex = factorATotalCnt;
        }
        if (unlikely(loopAEndIndex > factorATotalCnt)) {
            loopAEndIndex = factorATotalCnt;
        }

        int64_t loopACnt = loopAEndIndex - loopAStartIndex;
        for (int64_t i = 0; i < loopACnt; i++) {
            ComputeWelfordPhase2(i, factorACntPerCore);
        }
    }

    __aicore__ inline void SetLoopRangeNormal()
    {
        int32_t blockId = GetBlockIdx();
        loopAStartIndex_ = blockId * tilingOp_->factorACntPerCore;
        loopAEndIndex_ = loopAStartIndex_ + tilingOp_->factorACntPerCore;
        if (unlikely(loopAEndIndex_ > tilingOp_->factorATotalCnt)) {
            loopAEndIndex_ = tilingOp_->factorATotalCnt;
        }
        constexpr int32_t aAxisIdx = SchLoopInfo.loopACount - 1;
        constexpr int32_t aAxis = SchLoopInfo.loopAAxis[aAxisIdx];
        loopAAxisStep_ = Ops::Base::CeilDiv(tilingOp_->shape[aAxis], tilingOp_->ubFactorA);

        // R轴可以全载时 loopInnerRCount 为 0, ubFactorR 为 1
        if constexpr (SchLoopInfo.loopInnerRCount > 0) {
            constexpr int32_t rAxisIdx = SchLoopInfo.loopInnerRCount - 1;
            constexpr int32_t rAxis = SchLoopInfo.loopInnerRAxis[rAxisIdx];
            lastRAxisNum_ = tilingOp_->shape[rAxis];
            loopRAxisStep_ = Ops::Base::CeilDiv(lastRAxisNum_, tilingOp_->ubFactorR);
            splitRAxisTail_ = tilingOp_->shape[rAxis] % tilingOp_->ubFactorR;
        }

        ubFactorA_ = tilingOp_->ubFactorA;
        ubFactorR_ = tilingOp_->ubFactorR;
    }

    template <int32_t LoopAIdx>
    __aicore__ inline void CalcIterA(uint64_t step)
    {
        if constexpr (LoopAIdx != 0) {
            constexpr auto axis = SchLoopInfo.loopAAxis[LoopAIdx - 1];
            if constexpr (LoopAIdx == SchLoopInfo.loopACount) {
                // 切分轴
                auto cur = step % loopAAxisStep_;
                iterAddr_[axis].start = cur * ubFactorA_;
                iterAddr_[axis].stride = tilingOp_->shape[axis] - iterAddr_[axis].start;
                if (likely(iterAddr_[axis].stride >= ubFactorA_)) {
                    iterAddr_[axis].stride = ubFactorA_;
                }

                if constexpr (LoopAIdx > 0) {
                    CalcIterA<LoopAIdx - 1>(step / loopAAxisStep_);
                }
            } else {
                iterAddr_[axis].start = step % tilingOp_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalcIterA<LoopAIdx - 1>(step / tilingOp_->shape[axis]);
            }
        }
    }

    template <int32_t start = 0, int32_t end = 0>
    __aicore__ inline void IterateInnerA()
    {
        if constexpr (start == end) {
            Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim> shape;
            if constexpr (SchLoopInfo.loopRCount == 0) {
                if constexpr (SchLoopInfo.loopInnerRCount == 0) {
                    // R轴全载
                    ComputeTwoPass(shape);
                } else {
                    ComputeWelford<false>(shape);
                }
                CopyOut(shape);
            } else {
                // group reduce第一阶段输出的是M2的值, 不输出方差
                ComputeWelford<true>(shape);
                CopyOutGroup(shape);
            }
        } else {
            constexpr int32_t axis = SchLoopInfo.loopInnerAAxis[start];
            uint64_t shape = tilingOp_->shape[axis];
            if constexpr (start + 1 == end) {  // 为最内轴
                uint64_t loopSize = shape / ubFactorA_;
                uint64_t tail = shape - loopSize * ubFactorA_;
                iterAddr_[axis].start = 0;
                iterAddr_[axis].stride = ubFactorA_;

                for (uint64_t i = 0; i < loopSize; i++) {  // 整块
                    IterateInnerA<start + 1, end>();
                    iterAddr_[axis].start += ubFactorA_;
                }

                if (tail) {
                    iterAddr_[axis].stride = shape - iterAddr_[axis].start;
                    IterateInnerA<start + 1, end>();
                }
            } else {
                for (uint64_t i = 0; i < shape; i++) {
                    iterAddr_[axis].start = i;
                    IterateInnerA<start + 1, end>();
                }
            }
        }
    }

    __aicore__ inline void ComputeTwoPass(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM> view;
        bool calcShape = true;

        CopyInX(0, view, shape, calcShape);

        LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
        __local_mem__ DataType* xLocal = (__local_mem__ DataType*)inputUb.GetPhyAddr();
        __local_mem__ float* dichotomyAddAddr = (__local_mem__ float*)tDichAddTensor_.GetPhyAddr();

        LocalTensor<DataType> outMeanTensor = outQueue_.AllocTensor<DataType>();
        LocalTensor<DataType> outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(DataType)];
        __local_mem__ DataType* outMeanAddr = (__local_mem__ DataType*)outMeanTensor.GetPhyAddr();
        __local_mem__ DataType* outVarAddr = (__local_mem__ DataType*)outVarTensor.GetPhyAddr();

        float varScale = tiling_->correctionInvalid == 1 ? (*((float*)&FLOAT32_INF)) : tiling_->varFactor;
        float meanScale = tiling_->meanFactor;

        if constexpr (!InnerPattern::TailA) {
            if (lastRAxisLen_ % (BLOCK_SIZE_BYTE / sizeof(DataType)) == 0 || Dim == Ops::Base::ReduceOpTmpl::CONST2) {
                uint32_t realRLen = (Dim == Ops::Base::ReduceOpTmpl::CONST2) ? (uint32_t)lastRAxisLen_ : (uint32_t)shape.value[1];
                VFMeanVarTwoPassAR<DataType, isStd>(xLocal, dichotomyAddAddr, outMeanAddr, outVarAddr, shape.value[0],
                                                    shape.value[1], realRLen, varScale);
            } else {
                VFMeanVarTwoPassARPad<DataType, isStd>(xLocal, dichotomyAddAddr, outMeanAddr, outVarAddr,
                                                       shape.value[0], shape.value[1], lastRAxisLen_ * loopLastRCnt_,
                                                       varScale, lastRAxisLen_, lastRAxisLenAlign_, loopLastRCnt_);
            }
        } else {
            VFMeanVarTwoPassRA<DataType, isStd>(xLocal, tDichAddTensor_, tMeanTensor_, tVarTensor_, outMeanTensor,
                                                outVarTensor, shape.value[1], shape.value[0], varScale);
        }

        inputQueue_.FreeTensor(inputUb);
        outQueue_.EnQue(outMeanTensor);
    }

    __aicore__ inline bool CheckTailWelford(uint64_t loopIdx)
    {
        if constexpr (SchLoopInfo.loopRCount > 0) {
            uint64_t idx = loopIdx + loopRStartIndex_;
            uint64_t cur = idx % loopRAxisStep_;
            uint64_t start = cur * ubFactorR_;
            constexpr auto axis = SchLoopInfo.loopRAxis[SchLoopInfo.loopRCount - 1];
            // 特例: R场景，前63个核处理128个数, 最后一个核处理 16个数，这里会把最后一个核的16当做尾块，其实是整块
            if (tilingOp_->shape[axis] - start < ubFactorR_) {
                return true;
            }
        } else {
            // normal 场景， R轴全载时，loopRAxisStep_ = 0
            if (splitRAxisTail_ != 0 && loopRAxisStep_ != 0 && (loopIdx % loopRAxisStep_ == loopRAxisStep_ - 1)) {
                return true;
            }
        }

        return false;
    }

    __aicore__ inline void WelfordUpdate(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
        LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor, int64_t& count, int64_t& tailsNum)
    {
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM> view;
        __local_mem__ float* meanBufAddr = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* varBufAddr = (__local_mem__ float*)tVarTensor.GetPhyAddr();

        float scale = 1.0f;
        uint32_t processNum = 0;
        bool hasTail = false;
        bool calcShape = true;
        // 可能存在多个尾块的场景，如整块、尾块、整块、尾块..., 可以先做整块的update，再做尾块的update
        for (int64_t i = 0; i < rCount_; i++) {
            if (CheckTailWelford(i) == true) {
                hasTail = true;
                continue;
            }
            CopyInX(i, view, shape, calcShape);

            LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
            __local_mem__ DataType* xLocal = (__local_mem__ DataType*)inputUb.GetPhyAddr();

            count = count + 1;
            scale = static_cast<float>(1.0) / static_cast<float>(count);
            processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
            // RA 和 AR 更新的逻辑一样，不做区分
            if (count == 1) {
                // 第一次更新时，需要将tmp mean和tmp var清0
                VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
            } else {
                VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
            }
            inputQueue_.FreeTensor(inputUb);
        }

        if (hasTail == true) {
            for (int64_t i = 0; i < rCount_; i++) {
                if (CheckTailWelford(i) == false) {
                    continue;
                }
                CopyInX(i, view, shape, calcShape);
                LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
                __local_mem__ DataType* xLocal = (__local_mem__ DataType*)inputUb.GetPhyAddr();

                count = count + 1;
                tailsNum = tailsNum + 1;
                scale = static_cast<float>(1.0) / static_cast<float>(count);
                if constexpr (!InnerPattern::TailA) {
                    if (count == 1) {
                        processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                        VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    } else {
                        VFWelfordParallelUpdateARWithTail(
                            xLocal, meanBufAddr, varBufAddr, static_cast<uint32_t>(shape.value[0]),
                            static_cast<uint32_t>(shape.value[1]), static_cast<uint32_t>(lastReduceTailR_), scale);
                    }
                } else {
                    if (count == 1) {
                        processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                        VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    } else {
                        processNum = static_cast<uint32_t>(lastReduceTailR_ * shape.value[1]);
                        VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    }
                }
                inputQueue_.FreeTensor(inputUb);
            }
        }
    }

    __aicore__ inline void WelfordUpdateGroups(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
        LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor, int64_t& count, int64_t& tailsNum)
    {
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM> view;
        __local_mem__ float* meanBufAddr = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* varBufAddr = (__local_mem__ float*)tVarTensor.GetPhyAddr();

        uint32_t processNum = 0;
        bool hasTail = false;
        bool calcShape = true;
        uint32_t updateCycleCnt = 0;
        // 可能存在多个尾块的场景，如整块、尾块、整块、尾块..., 可以先做整块的update，再做尾块的update
        for (int64_t i = 0; i < rCount_; i++) {
            if (CheckTailWelford(i) == true) {
                hasTail = true;
                continue;
            }
            CopyInX(i, view, shape, calcShape);

            LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
            __local_mem__ DataType* xLocal = (__local_mem__ DataType*)inputUb.GetPhyAddr();

            count = count + 1;
            updateCycleCnt = updateCycleCnt + 1;
            float scale = static_cast<float>(1.0) / static_cast<float>(updateCycleCnt);
            processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
            // RA 和 AR 更新的逻辑一样，不做区分
            if (updateCycleCnt == 1) {
                // 第一次更新时，需要将tmp mean和tmp var清0
                VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
            } else {
                VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
            }
            inputQueue_.FreeTensor(inputUb);

            if (updateCycleCnt == WELFORD_GROUP_NUM) {
                VFWelfordParallelFinalizeGroups(shape, false, tMeanTensor, tVarTensor, tGroupMeanTensor_,
                                                tGroupVarTensor_, updateCycleCnt);
                updateCycleCnt = 0;
            }
        }

        if (updateCycleCnt != 0) {
            VFWelfordParallelFinalizeGroups(shape, false, tMeanTensor, tVarTensor, tGroupMeanTensor_, tGroupVarTensor_,
                                            updateCycleCnt);
            updateCycleCnt = 0;
        }

        if (hasTail == true) {
            for (int64_t i = 0; i < rCount_; i++) {
                if (CheckTailWelford(i) == false) {
                    continue;
                }
                CopyInX(i, view, shape, calcShape);
                LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
                __local_mem__ DataType* xLocal = (__local_mem__ DataType*)inputUb.GetPhyAddr();

                count = count + 1;
                updateCycleCnt = updateCycleCnt + 1;
                tailsNum = tailsNum + 1;
                float scale = static_cast<float>(1.0) / static_cast<float>(updateCycleCnt);
                if constexpr (!InnerPattern::TailA) {
                    if (updateCycleCnt == 1) {
                        processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                        VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    } else {
                        VFWelfordParallelUpdateARWithTail(
                            xLocal, meanBufAddr, varBufAddr, static_cast<uint32_t>(shape.value[0]),
                            static_cast<uint32_t>(shape.value[1]), static_cast<uint32_t>(lastReduceTailR_), scale);
                    }
                } else {
                    if (updateCycleCnt == 1) {
                        processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                        VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    } else {
                        processNum = static_cast<uint32_t>(lastReduceTailR_ * shape.value[1]);
                        VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    }
                }
                inputQueue_.FreeTensor(inputUb);

                if (updateCycleCnt == WELFORD_GROUP_NUM) {
                    VFWelfordParallelFinalizeGroups(shape, true, tMeanTensor, tVarTensor, tGroupMeanTensor_,
                                                    tGroupVarTensor_, updateCycleCnt);
                    updateCycleCnt = 0;
                }
            }

            if (updateCycleCnt != 0) {
                VFWelfordParallelFinalizeGroups(shape, true, tMeanTensor, tVarTensor, tGroupMeanTensor_,
                                                tGroupVarTensor_, updateCycleCnt);
                updateCycleCnt = 0;
            }
        }
    }

    __aicore__ inline void VFWelfordParallelFinalizeGroups(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
                                                           bool isTail,
                                                           LocalTensor<float>& tMeanTensor,
                                                           LocalTensor<float>& tVarTensor,
                                                           LocalTensor<float>& groupMeanTensor,
                                                           LocalTensor<float>& groupVarTensor, uint32_t updateCycleCnt)
    {
        __local_mem__ float* meanBufAddr = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* varBufAddr = (__local_mem__ float*)tVarTensor.GetPhyAddr();

        __local_mem__ float* groupMeanBufAddr =
            (__local_mem__ float*)groupMeanTensor.GetPhyAddr() + rCntGroupIdx_ * MAX_INNER_A_NUM;
        __local_mem__ float* groupVarBufAddr =
            (__local_mem__ float*)groupVarTensor.GetPhyAddr() + rCntGroupIdx_ * MAX_INNER_A_NUM;

        // LocalTensor<float> tDichTensor = dichotomyAddBuf_.Get<float>();
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)tDichAddTensor_.GetPhyAddr();

        if constexpr (!InnerPattern::TailA) {
            uint32_t aNum = static_cast<uint32_t>(shape.value[0]);
            uint32_t rNum = static_cast<uint32_t>(shape.value[1]);
            uint32_t rStride = static_cast<uint32_t>(shape.value[1]);

            // AR isTail == true
            // 1. ub切尾轴时  r实际长度 = lastReduceTailR_ = splitRAxisTail_;
            // 2. ub不切尾轴但尾轴本身是对齐的: r实际长度 = lastRAxisLen_ * loopWelfTailRCnt_
            // 3. ub不切尾轴且尾轴本身非对齐的: r实际长度 = lastRAxisLen_ * loopWelfTailRCnt_
            bool isLastAlign = ((Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) ||
                                (tilingOp_->shape[Dim - 1] % (BLOCK_SIZE_BYTE / sizeof(DataType)) == 0));
            if (isLastAlign) {
                if (isTail) {
                    rNum = (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) ? splitRAxisTail_
                                                                                   : lastRAxisLen_ * loopWelfTailRCnt_;
                }
                rCntGroupWelford_[rCntGroupIdx_] = static_cast<uint32_t>(rNum * updateCycleCnt);
                rCntGroupIdx_ = isInvert_ ? (rCntGroupIdx_ - 1) : (rCntGroupIdx_ + 1);

                // 没有尾块, 没有pad, meanScale = (1.0f * updateCycleCnt) / (1.0 * rNum * updateCycleCnt)
                float meanScale =
                    (rNum == 0) ? 1.0f : (1.0f * updateCycleCnt) / static_cast<float>(rNum * updateCycleCnt);

                // 不带pad
                VFWelfordParallelFinalizeARAlign<float, isStd, true>(meanBufAddr, varBufAddr, dichotomyAddLocal,
                                                                     groupMeanBufAddr, groupVarBufAddr, aNum, rNum,
                                                                     rStride, 1.0f, meanScale, updateCycleCnt);
            } else {
                int64_t realR = lastRAxisLen_ * loopLastRCnt_;
                int64_t lastRLoops = loopLastRCnt_;
                if (isTail) {
                    rNum = lastReduceTailR_;
                    realR = lastRAxisLen_ * loopWelfTailRCnt_;
                    lastRLoops = loopWelfTailRCnt_;
                }
                rCntGroupWelford_[rCntGroupIdx_] = static_cast<uint32_t>(realR * updateCycleCnt);
                rCntGroupIdx_ = isInvert_ ? (rCntGroupIdx_ - 1) : (rCntGroupIdx_ + 1);

                // 没有尾块, 有pad, meanScale = updateCycleCnt / (realR * updateCycleCnt)
                float meanScale = (realR == 0) ? 1.0f : 1.0f / static_cast<float>(realR);
                // 带pad
                VFWelfordParallelFinalizeARAlignPad<float, isStd, true>(
                    meanBufAddr, varBufAddr, dichotomyAddLocal, groupMeanBufAddr, groupVarBufAddr, aNum, rNum, rStride,
                    1.0f, meanScale, updateCycleCnt, lastRAxisLen_, lastRAxisLenAlign_, lastRLoops);
            }
        } else {
            // dichotomyAddLocal RA场景下空间分配
            __local_mem__ float* tmpCountLocal = (__local_mem__ float*)tCountTensor_.GetPhyAddr();
            uint32_t RNum = isTail ? lastReduceTailR_ : shape.value[0];
            uint32_t ANum = shape.value[1];

            int64_t tailRNum = 0;
            int64_t addCnt = updateCycleCnt;
            int64_t addTailCnt = updateCycleCnt;  // welford 累加次数, addTailCnt >= addCnt
            CaculateCountBuf(tmpCountLocal, RNum, tailRNum, addCnt, addTailCnt);

            float meanScale = (updateCycleCnt * RNum == 0) ? 1.0f : 1.0f / static_cast<float>(updateCycleCnt * RNum);
            LocalTensor<float> dstGroupMean = groupMeanTensor[rCntGroupIdx_ * MAX_INNER_A_NUM];
            LocalTensor<float> dstGroupVar = groupVarTensor[rCntGroupIdx_ * MAX_INNER_A_NUM];
            VFWelfordFinalizeRA<float, isStd, true>(RNum, ANum, tMeanTensor, tVarTensor, tmpCountLocal, dstGroupMean,
                                                    dstGroupVar, dichotomyAddLocal, meanScale, 1.0f);

            rCntGroupWelford_[rCntGroupIdx_] = static_cast<uint32_t>(RNum * updateCycleCnt);
            rCntGroupIdx_ = isInvert_ ? (rCntGroupIdx_ - 1) : (rCntGroupIdx_ + 1);
        }

        if ((isInvert_ && rCntGroupIdx_ <= 0) || (!isInvert_ && rCntGroupIdx_ >= WELFORD_GROUP_NUM)) {
            // RA finalize
            int64_t totalCnt = 0;
            __local_mem__ float* tmpCountLocal = (__local_mem__ float*)tCountTensor_.GetPhyAddr();
            int32_t startIdx = isInvert_ ? 1 : 0;
            int32_t endIdx = isInvert_ ? (WELFORD_GROUP_NUM + 1) : WELFORD_GROUP_NUM;

            uint32_t RNum = WELFORD_GROUP_NUM;
            // 优化：改成A的实际长度，并且 VFWelfordFinalizeRA 增加一个aStride的入参
            uint32_t ANum = MAX_INNER_A_NUM;
            int32_t dstIdx = isInvert_ ? 0 : WELFORD_GROUP_NUM;

            Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_S>(HardEvent::V_S);

            for (int32_t idx = startIdx; idx < endIdx; idx++) {
                float reduceCnt = static_cast<float>(rCntGroupWelford_[idx]);
                int32_t bufIdx = isInvert_ ? (idx - 1) : idx;
                tCountTensor_.SetValue(bufIdx, reduceCnt);
                totalCnt += rCntGroupWelford_[idx];
            }

            Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::S_V>(HardEvent::S_V);

            float meanScale = (totalCnt == 0) ? 1.0f : 1.0f / static_cast<float>(totalCnt);
            LocalTensor<float> srcGroupMean = groupMeanTensor[startIdx * MAX_INNER_A_NUM];
            LocalTensor<float> srcGroupVar = groupVarTensor[startIdx * MAX_INNER_A_NUM];
            LocalTensor<float> dstGroupMean = groupMeanTensor[dstIdx * MAX_INNER_A_NUM];
            LocalTensor<float> dstGroupVar = groupVarTensor[dstIdx * MAX_INNER_A_NUM];
            VFWelfordFinalizeRA<float, isStd, true>(RNum, ANum, srcGroupMean, srcGroupVar, tmpCountLocal, dstGroupMean,
                                                    dstGroupVar, dichotomyAddLocal, meanScale, 1.0f);

            rCntGroupWelford_[dstIdx] = totalCnt;
            rCntGroupIdx_ = isInvert_ ? 1 : (WELFORD_GROUP_NUM - 1);
            isInvert_ = (!isInvert_);
        }
    }

    template <typename T, bool isM2Out = false>
    __aicore__ inline void WelfordFinalizeGroup()
    {
        LocalTensor<T> outMeanTensor = outQueue_.AllocTensor<T>();
        LocalTensor<T> outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(T)];
        __local_mem__ T* outMeanAddr = (__local_mem__ T*)outMeanTensor.GetPhyAddr();
        __local_mem__ T* outVarAddr = (__local_mem__ T*)outVarTensor.GetPhyAddr();

        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)tDichAddTensor_.GetPhyAddr();

        float varScale = tiling_->varFactor;
        if (tiling_->correctionInvalid == 1) {
            varScale = *((float*)&FLOAT32_INF);
        }

        float meanScale = tiling_->meanFactor;
        if constexpr (SchLoopInfo.loopRCount > 0) {
            meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tilingOp_->groupR]);
        }

        // RA finalize
        int64_t totalRCnt = 0;
        __local_mem__ float* tmpCountLocal = (__local_mem__ float*)tCountTensor_.GetPhyAddr();
        int32_t startIdx = isInvert_ ? (rCntGroupIdx_ + 1) : 0;
        int32_t endIdx = isInvert_ ? (WELFORD_GROUP_NUM + 1) : rCntGroupIdx_;
        uint32_t RNum = endIdx - startIdx;
        // 优化: 改成A的实际长度，并且 VFWelfordFinalizeRA 增加一个aStride的入参
        uint32_t ANum = MAX_INNER_A_NUM;

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_S>(HardEvent::V_S);

        for (int32_t idx = startIdx; idx < endIdx; idx++) {
            float reduceCnt = static_cast<float>(rCntGroupWelford_[idx]);
            int32_t bufIdx = isInvert_ ? (idx - rCntGroupIdx_ - 1) : idx;
            tCountTensor_.SetValue(bufIdx, reduceCnt);
            totalRCnt += rCntGroupWelford_[idx];
        }

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::S_V>(HardEvent::S_V);

        LocalTensor<float> srcGroupMean = tGroupMeanTensor_[startIdx * MAX_INNER_A_NUM];
        LocalTensor<float> srcGroupVar = tGroupVarTensor_[startIdx * MAX_INNER_A_NUM];
        VFWelfordFinalizeRA<T, isStd, isM2Out>(RNum, ANum, srcGroupMean, srcGroupVar, tmpCountLocal, outMeanTensor,
                                               outVarTensor, dichotomyAddLocal, meanScale, varScale);

        outQueue_.EnQue(outMeanTensor);
    }

    template <typename T, bool isM2Out = false>
    __aicore__ inline void WelfordFinalize(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
        int64_t count, int64_t tailsNum, LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor)
    {
        __local_mem__ float* dichotomyAddLocal = (__local_mem__ float*)tDichAddTensor_.GetPhyAddr();
        __local_mem__ float* meanBufAddr = (__local_mem__ float*)tMeanTensor.GetPhyAddr();
        __local_mem__ float* varBufAddr = (__local_mem__ float*)tVarTensor.GetPhyAddr();

        LocalTensor<T> outMeanTensor = outQueue_.AllocTensor<T>();
        LocalTensor<T> outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(T)];
        __local_mem__ T* outMeanAddr = (__local_mem__ T*)outMeanTensor.GetPhyAddr();
        __local_mem__ T* outVarAddr = (__local_mem__ T*)outVarTensor.GetPhyAddr();

        float varScale = tiling_->varFactor;
        if (tiling_->correctionInvalid == 1) {
            varScale = *((float*)&FLOAT32_INF);
        }

        if constexpr (!InnerPattern::TailA) {
            uint32_t aNum = static_cast<uint32_t>(shape.value[0]);
            uint32_t rNum = static_cast<uint32_t>(shape.value[1]);
            uint32_t rStride = static_cast<uint32_t>(shape.value[1]);
            // group
            // reduce存在此场景，如果ub切分的R.o全部用来开多核,则groupR的最后一个核，计算得到的全是尾块，但要当整块来处理
            if (count == tailsNum) {
                tailsNum = 0;
                rNum = static_cast<uint32_t>(lastReduceTailR_);
            }

            bool isLastAlign = ((Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) ||
                                (tilingOp_->shape[Dim - 1] % (BLOCK_SIZE_BYTE / sizeof(DataType)) == 0));
            float meanScale = tiling_->meanFactor;
            if (tailsNum == 0) {
                meanScale = (float)count * tiling_->meanFactor;  // 无尾块场景, meanscale需要乘以count
                if constexpr (SchLoopInfo.loopRCount > 0) {
                    meanScale =
                        float(count) / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tilingOp_->groupR]);
                }
                if (isLastAlign) {
                    // 不带pad
                    VFWelfordParallelFinalizeARAlign<T, isStd, isM2Out>(meanBufAddr, varBufAddr, dichotomyAddLocal,
                                                                        outMeanAddr, outVarAddr, aNum, rNum, rStride,
                                                                        varScale, meanScale, count);
                } else {
                    // 带pad
                    VFWelfordParallelFinalizeARAlignPad<T, isStd, isM2Out>(
                        meanBufAddr, varBufAddr, dichotomyAddLocal, outMeanAddr, outVarAddr, aNum, rNum, rStride,
                        varScale, meanScale, count, lastRAxisLen_, lastRAxisLenAlign_, loopLastRCnt_);
                }
            } else if (isLastAlign) {
                meanScale = tiling_->meanFactor;
                if constexpr (SchLoopInfo.loopRCount > 0) {
                    meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tilingOp_->groupR]);
                }
                VFWelfordParallelFinalizeARNonAlign<T, isStd, isM2Out>(
                    meanBufAddr, varBufAddr, dichotomyAddLocal, outMeanAddr, outVarAddr, aNum, rNum, rStride, varScale,
                    meanScale, count - tailsNum, count, lastReduceTailR_);
            } else {
                meanScale = tiling_->meanFactor;
                if constexpr (SchLoopInfo.loopRCount > 0) {
                    meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tilingOp_->groupR]);
                }
                VFWelfordParallelFinalizeARNonAlignPad<T, isStd, isM2Out>(
                    tMeanTensor, tVarTensor, tDichAddTensor_, outMeanTensor, outVarTensor, aNum, rNum, rStride,
                    varScale, meanScale, count - tailsNum, count, lastReduceTailR_, loopWelfTailRCnt_, lastRAxisLen_,
                    lastRAxisLenAlign_, loopLastRCnt_);
            }
        } else {
            // dichotomyAddLocal RA场景下空间分配
            __local_mem__ float* tmpCountLocal = (__local_mem__ float*)tCountTensor_.GetPhyAddr();
            uint32_t RNum = shape.value[0];
            uint32_t ANum = shape.value[1];

            if (count == tailsNum) {
                tailsNum = 0;
            }
            int64_t tailRNum = (tailsNum == 0) ? 0 : lastReduceTailR_;
            int64_t addCnt = count - tailsNum;
            int64_t addTailCnt = count;  // welford 累加次数, addTailCnt >= addCnt
            CaculateCountBuf(tmpCountLocal, RNum, tailRNum, addCnt, addTailCnt);

            float meanScale = tiling_->meanFactor;
            if constexpr (SchLoopInfo.loopRCount > 0) {
                meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tilingOp_->groupR]);
            }
            VFWelfordFinalizeRA<T, isStd, isM2Out>(RNum, ANum, tMeanTensor, tVarTensor, tmpCountLocal, outMeanTensor,
                                                   outVarTensor, dichotomyAddLocal, meanScale, varScale);
        }

        outQueue_.EnQue(outMeanTensor);
    }

    template <bool isM2Out = false>
    __aicore__ inline void ComputeWelford(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        if (rCount_ > WELFORD_GROUP_NUM) {
            ReInitWelfordGroups();
            int64_t count = 0;
            int64_t tailsNum = 0;
            WelfordUpdateGroups(shape, tMeanTensor_, tVarTensor_, count, tailsNum);
            if constexpr (isM2Out == true) {
                WelfordFinalizeGroup<PromoteDataType, isM2Out>();
            } else {
                WelfordFinalizeGroup<DataType, isM2Out>();
            }
        } else {
            int64_t count = 0;
            int64_t tailsNum = 0;
            WelfordUpdate(shape, tMeanTensor_, tVarTensor_, count, tailsNum);
            if constexpr (isM2Out == true) {
                WelfordFinalize<PromoteDataType, isM2Out>(shape, count, tailsNum, tMeanTensor_, tVarTensor_);
            } else {
                WelfordFinalize<DataType, isM2Out>(shape, count, tailsNum, tMeanTensor_, tVarTensor_);
            }
        }
    }

    __aicore__ inline void ComputeWelfordPhase2(int64_t loopAIdx, int64_t factorACntPerCore)
    {
        int32_t blockIdx = GetBlockIdx();
        uint64_t startWs = (blockIdx * factorACntPerCore + loopAIdx) * ubFactorA_;
        int64_t realAnum = tilingOp_->outSize - static_cast<int64_t>(startWs);
        if (realAnum <= 0) {
            return;
        }
        if (likely(realAnum > ubFactorA_)) {
            realAnum = ubFactorA_;
        }

        __local_mem__ float* dichotomyAddAddr = (__local_mem__ float*)tDichAddTensor_.GetPhyAddr();
        LocalTensor<DataType> outMeanTensor = outQueue_.AllocTensor<DataType>();
        LocalTensor<DataType> outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(DataType)];

        uint64_t asize = Ops::Base::CeilAlign(tilingOp_->outSize, static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE));
        uint64_t varOffset = static_cast<uint64_t>(tiling_->workSpaceSize) / sizeof(PromoteDataType);
        float varScale = (tiling_->correctionInvalid == 1) ? (*((float*)&FLOAT32_INF)) : tiling_->varFactor;

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_S>(HardEvent::V_S);
        __local_mem__ float* groupCountBufAddr = (__local_mem__ float*)tCountTensor_.GetPhyAddr();
        for (int i = 0; i < tilingOp_->groupR; i++) {
            float reduceCnt = static_cast<float>(tiling_->reduceCntEachGroupR[i]);
            tCountTensor_.SetValue(i, reduceCnt);
        }

        DataCopyPadExtParams<PromoteDataType> padParams{true, 0, 0, static_cast<PromoteDataType>(0.0)};
        DataCopyExtParams copyInParams = {1, 1, 0, 0, 0};
        copyInParams.blockCount = tilingOp_->groupR;
        copyInParams.blockLen = ubFactorA_ * sizeof(PromoteDataType);
        copyInParams.srcStride = (asize - ubFactorA_) * sizeof(PromoteDataType);

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPad(tMeanTensor_, workspace_[startWs], copyInParams, padParams);
        DataCopyPad(tVarTensor_, workspace_[varOffset + startWs], copyInParams, padParams);

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::S_V>(HardEvent::S_V);

        VFWelfordFinalizeRA<DataType, isStd, false>(
            static_cast<uint32_t>(tilingOp_->groupR), static_cast<uint32_t>(ubFactorA_), tMeanTensor_, tVarTensor_,
            groupCountBufAddr, outMeanTensor, outVarTensor, dichotomyAddAddr, tiling_->meanFactor, varScale);

        outQueue_.EnQue(outMeanTensor);
        outMeanTensor = outQueue_.DeQue<DataType>();
        outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(DataType)];

        DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = realAnum * sizeof(DataType);

        DataCopyPad(varGM_[startWs], outVarTensor, copyOutParams);
        if (tiling_->isMeanOut) {
            DataCopyPad(meanGM_[startWs], outMeanTensor, copyOutParams);
        }

        outQueue_.FreeTensor(outMeanTensor);
    }

    __aicore__ inline void SetLoopRangeGroup()
    {
        int32_t blockId = GetBlockIdx();
        loopRStartIndex_ = blockId / tilingOp_->groupR * tilingOp_->factorRTotalCnt +
                           blockId % tilingOp_->groupR * tilingOp_->factorRCntPerCore;
        loopREndIndex_ = loopRStartIndex_ + tilingOp_->factorRCntPerCore;
        uint64_t maxRCnt = (blockId / tilingOp_->groupR + 1) * tilingOp_->factorRTotalCnt;
        uint64_t totalCnt = tilingOp_->factorATotalCnt * tilingOp_->factorRTotalCnt;
        maxRCnt = maxRCnt > totalCnt ? totalCnt : maxRCnt;
        if (unlikely(loopRStartIndex_ > maxRCnt)) {
            loopRStartIndex_ = maxRCnt;
        }
        if (unlikely(loopREndIndex_ > maxRCnt)) {
            loopREndIndex_ = maxRCnt;
        }

        constexpr int32_t rAxisIdx = SchLoopInfo.loopRCount - 1;
        constexpr int32_t rAxis = SchLoopInfo.loopRAxis[rAxisIdx];
        loopRAxisStep_ = Ops::Base::CeilDiv(tilingOp_->shape[rAxis], tilingOp_->ubFactorR);  // 切分轴Rfactor的个数
        splitRAxisTail_ = tilingOp_->shape[rAxis] % tilingOp_->ubFactorR;

        if constexpr (SchLoopInfo.loopACount > 0) {
            constexpr int32_t aAxisIdx = SchLoopInfo.loopACount - 1;
            constexpr int32_t aAxis = SchLoopInfo.loopAAxis[aAxisIdx];
            loopAAxisStep_ = Ops::Base::CeilDiv(tilingOp_->shape[aAxis], tilingOp_->ubFactorA);
        }

        ubFactorA_ = tilingOp_->ubFactorA;
        ubFactorR_ = tilingOp_->ubFactorR;
    }

    __aicore__ inline void CopyInX(
        int64_t index, Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
        Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape, bool& calcShape)
    {
        LocalTensor<DataType> inputTensor = inputQueue_.AllocTensor<DataType>();

        if constexpr (SchLoopInfo.loopRCount > 0) {
            CalcIterR<SchLoopInfo.loopRCount>(index + loopRStartIndex_);
        } else {
            CalcInnerIterR<SchLoopInfo.loopInnerRCount>(index);
        }

        CalcCopyInParam(view);

        if (calcShape) {
            CalcInnerShape(view, shape);
            calcShape = false;
        }

        CopyIn(view, inputTensor);
        inputQueue_.EnQue(inputTensor);
    }

    template <int32_t LoopRIdx>
    __aicore__ inline void CalcIterR(uint64_t step)
    {
        uint64_t temp = step;
        if constexpr (LoopRIdx != 0) {
            for (int32_t idx = SchLoopInfo.loopRCount - 1; idx > -1; --idx) {
                if (idx == SchLoopInfo.loopRCount - 1) {
                    constexpr auto axis = SchLoopInfo.loopRAxis[SchLoopInfo.loopRCount - 1];
                    auto cur = temp % loopRAxisStep_;
                    iterAddr_[axis].start = cur * ubFactorR_;
                    iterAddr_[axis].stride = tilingOp_->shape[axis] - iterAddr_[axis].start;
                    if (likely(iterAddr_[axis].stride >= ubFactorR_)) {
                        iterAddr_[axis].stride = ubFactorR_;
                    }
                    temp = temp / loopRAxisStep_;
                } else {
                    auto axis = SchLoopInfo.loopRAxis[idx];
                    if (Ops::Base::ReduceOpTmpl::IsLoopSpliteAAxis<&SchLoopInfo>(axis)) {
                        auto cur = temp % loopAAxisStep_;
                        iterAddr_[axis].start = cur * ubFactorA_;
                        iterAddr_[axis].stride = tilingOp_->shape[axis] - iterAddr_[axis].start;
                        if (likely(iterAddr_[axis].stride >= ubFactorA_)) {
                            iterAddr_[axis].stride = ubFactorA_;
                        }
                        temp = temp / loopAAxisStep_;
                    } else {
                        iterAddr_[axis].start = temp % tilingOp_->shape[axis];
                        iterAddr_[axis].stride = 1;
                        temp = temp / tilingOp_->shape[axis];
                    }
                }
            }
        }
    }

    template <int32_t LoopInnerRIdx>
    __aicore__ inline void CalcInnerIterR(uint64_t basicBlockIdx)
    {
        if constexpr (LoopInnerRIdx != 0) {
            constexpr auto axis = SchLoopInfo.loopInnerRAxis[LoopInnerRIdx - 1];
            if constexpr (LoopInnerRIdx == SchLoopInfo.loopInnerRCount) {
                // 最内层循环
                auto cur = basicBlockIdx % loopRAxisStep_;
                iterAddr_[axis].start = cur * ubFactorR_;
                iterAddr_[axis].stride = tilingOp_->shape[axis] - iterAddr_[axis].start;
                if (likely(iterAddr_[axis].stride >= ubFactorR_)) {
                    iterAddr_[axis].stride = ubFactorR_;
                }
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / loopRAxisStep_);
            } else {
                iterAddr_[axis].start = basicBlockIdx % tilingOp_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / tilingOp_->shape[axis]);
            }
        }
    }

    __aicore__ inline void CalcCopyInParam(Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < Dim; i++) {
            addrOffset += iterAddr_[i].start * tilingOp_->stride[i];
        }

        constexpr static auto burstLenAxis = Dim - 1;  // 获取搬运的最内轴的循环轴
        view.addr = addrOffset;                        // 搬运地址
        view.axis[0].repeat = Ops::Base::ReduceOpTmpl::GetBurstLen<&SchLoopInfo, burstLenAxis>(iterAddr_, tilingOp_);
        view.axisSize = 1;  // 一次搬运时的循环轴个数

        if constexpr (burstLenAxis > 0) {
            int32_t axis = burstLenAxis;
            for (int32_t i = 1; i < Dim; i++) {
                view.axisSize = i + 1;
                view.axis[i].repeat = Ops::Base::ReduceOpTmpl::GetRepeatStride<&SchLoopInfo>(
                    axis - 1, iterAddr_, tilingOp_, view.axis[i].srcStride);
                view.axis[i].idx = axis - 1;
                view.axis[i].isAxisA = Ops::Base::ReduceOpTmpl::IsAxisA<Pattern::FirstA>(view.axis[i].idx);
                if (view.axis[i].idx <= 0) {
                    break;
                }
                axis = view.axis[i].idx;
            }
        }
    }

    __aicore__ inline void CalcInnerShapeLastR(Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
                                               Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        int64_t value = Ops::Base::CeilAlign(view.axis[0].repeat, BLOCK_SIZE_BYTE / sizeof(DataType));
        lastReduceTailR_ = value;  // 不切最后一根轴时，取对齐后的大小
        if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) {
            value = Ops::Base::CeilAlign(ubFactorR_, BLOCK_SIZE_BYTE / sizeof(DataType));
            lastReduceTailR_ = splitRAxisTail_;  // 尾轴reduce且ub切分最后一根轴,则取实际的大小,不做对齐
        }
        loopLastRCnt_ = 1;
        loopWelfTailRCnt_ = 1;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(view.axis[i].idx)) {
                    value = value * ubFactorR_;
                    lastReduceTailR_ = lastReduceTailR_ * splitRAxisTail_;
                    loopWelfTailRCnt_ = loopWelfTailRCnt_ * splitRAxisTail_;
                } else {
                    value = value * view.axis[i].repeat;
                    lastReduceTailR_ = lastReduceTailR_ * view.axis[i].repeat;
                    loopWelfTailRCnt_ = loopWelfTailRCnt_ * view.axis[i].repeat;
                }
                loopLastRCnt_ = loopLastRCnt_ * view.axis[i].repeat;
            }
        }
        shape.value[InnerPattern::Dim - 1] = value;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
            }
        }
        shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2] = value / shape.value[InnerPattern::Dim - 1];
    }

    __aicore__ inline void CalcInnerShapeLastA(Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
                                               Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        int64_t value = tiling_->useNddma == 1
                            ? view.axis[0].repeat
                            : Ops::Base::CeilAlign(view.axis[0].repeat, BLOCK_SIZE_BYTE / sizeof(DataType));
        aOutBurstLen_ = view.axis[0].repeat;
        if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) {
            value =
                tiling_->useNddma == 1 ? ubFactorA_ : Ops::Base::CeilAlign(ubFactorA_, BLOCK_SIZE_BYTE / sizeof(DataType));
            aOutBurstLen_ = ubFactorA_;
        }

        aOutNBurst_ = 1;
        lastReduceTailR_ = 1;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
                aOutNBurst_ = aOutNBurst_ * view.axis[i].repeat;
            }
        }
        if (tiling_->useNddma == 1) {
            aOutNBurst_ = 1;
            aOutBurstLen_ = value;
            value = Ops::Base::CeilAlign(static_cast<uint64_t>(value), BLOCK_SIZE_BYTE / sizeof(DataType));
        }
        shape.value[InnerPattern::Dim - 1] = value;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
                if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(view.axis[i].idx)) {
                    lastReduceTailR_ = lastReduceTailR_ * splitRAxisTail_;
                } else {
                    lastReduceTailR_ = lastReduceTailR_ * view.axis[i].repeat;
                }
            }
        }
        shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2] = value / shape.value[InnerPattern::Dim - 1];
    }

    __aicore__ inline void CalcInnerShape(Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
                                          Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        if constexpr (!InnerPattern::TailA) {
            CalcInnerShapeLastR(view, shape);
        } else {
            CalcInnerShapeLastA(view, shape);
        }
    }

    __aicore__ inline void CopyInWithNddma(const Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
                                           LocalTensor<DataType>& ubTensor)
    {
        static constexpr MultiCopyConfig config = {false};
        MultiCopyParams<DataType, MAX_NDDMA_DIM> paramsMain;
        paramsMain.constantValue = 0;
        paramsMain.loopInfo.loopSize[0] = view.axis[0].repeat;
        paramsMain.loopInfo.loopSrcStride[0] = 1;
        paramsMain.loopInfo.loopDstStride[0] = 1;
        paramsMain.loopInfo.loopLpSize[0] = 0;
        paramsMain.loopInfo.loopRpSize[0] = 0;
        for (uint32_t i = 1; i < MAX_NDDMA_DIM; i++) {
            paramsMain.loopInfo.loopSize[i] = view.axis[i].repeat;
            paramsMain.loopInfo.loopSrcStride[i] = view.axis[i].srcStride;
            paramsMain.loopInfo.loopDstStride[i] = view.axis[i].dstStride;
            paramsMain.loopInfo.loopLpSize[i] = 0;
            paramsMain.loopInfo.loopRpSize[i] = 0;
        }

        DataCopy<DataType, MAX_NDDMA_DIM, config>(ubTensor, inputGM_[view.addr], paramsMain);
    }

    __aicore__ inline void CopyIn(const Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
                                  LocalTensor<DataType>& ubTensor)
    {
        if (tiling_->useNddma == 1) {
            CopyInWithNddma(view, ubTensor);
            return;
        }

        DataCopyPadExtParams<DataType> padParams{true, 0, 0, static_cast<DataType>(0.0)};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = view.axis[1].repeat;
        copyInParams.blockLen = view.axis[0].repeat * sizeof(DataType);
        copyInParams.srcStride = (view.axis[1].srcStride - view.axis[0].repeat) * sizeof(DataType);
        copyInParams.dstStride = (view.axis[1].dstStride - view.axis[0].repeat) * sizeof(DataType) /
                                 BLOCK_SIZE_BYTE;  // unit block(32byte) "gap"
        LoopModeParams loopParams;
        loopParams.loop1Size = view.axis[2].repeat; // 2: the second-to-last dim
        loopParams.loop1SrcStride = view.axis[2].srcStride * sizeof(DataType); // 2: the second-to-last dim
        loopParams.loop1DstStride = view.axis[2].dstStride * sizeof(DataType); // 2: the second-to-last dim
        loopParams.loop2Size = view.axis[3].repeat; // 3: the third-to-last dim
        loopParams.loop2SrcStride = view.axis[3].srcStride * sizeof(DataType); // 3: the third-to-last dim
        loopParams.loop2DstStride = view.axis[3].dstStride * sizeof(DataType); // 3: the third-to-last dim

        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        for (int32_t i = 0; i < view.axis[Ops::Base::ReduceOpTmpl::CONST7].repeat; i++) {
            for (int32_t j = 0; j < view.axis[Ops::Base::ReduceOpTmpl::CONST6].repeat; j++) {
                for (int32_t k = 0; k < view.axis[Ops::Base::ReduceOpTmpl::CONST5].repeat; k++) {
                    for (int32_t l = 0; l < view.axis[Ops::Base::ReduceOpTmpl::CONST4].repeat; l++) {
                        int64_t dstStride = i * view.axis[Ops::Base::ReduceOpTmpl::CONST7].dstStride +
                                            j * view.axis[Ops::Base::ReduceOpTmpl::CONST6].dstStride +
                                            k * view.axis[Ops::Base::ReduceOpTmpl::CONST5].dstStride +
                                            l * view.axis[Ops::Base::ReduceOpTmpl::CONST4].dstStride;
                        int64_t srcStride = i * view.axis[Ops::Base::ReduceOpTmpl::CONST7].srcStride +
                                            j * view.axis[Ops::Base::ReduceOpTmpl::CONST6].srcStride +
                                            k * view.axis[Ops::Base::ReduceOpTmpl::CONST5].srcStride +
                                            l * view.axis[Ops::Base::ReduceOpTmpl::CONST4].srcStride;
                        DataCopyPad(ubTensor[dstStride], inputGM_[view.addr + srcStride], copyInParams, padParams);
                    }
                }
            }
        }
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }

    __aicore__ inline void CopyOut(const Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        constexpr int32_t axis = Pattern::FirstA ? 0 : 1;
        uint64_t addrOffset = 0;
        for (int32_t i = axis; i < Dim; i += Ops::Base::ReduceOpTmpl::CONST2) {
            addrOffset += iterAddr_[i].start * tilingOp_->dstStride[i];
        }

        LocalTensor<DataType> outMeanTensor = outQueue_.DeQue<DataType>();
        LocalTensor<DataType> outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(DataType)];

        DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};

        if constexpr (Pattern::TailA) {
            copyOutParams.blockCount = aOutNBurst_;
            copyOutParams.blockLen = aOutBurstLen_ * sizeof(DataType);
        } else {
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = shape.value[0] * sizeof(DataType);
        }

        DataCopyPad(varGM_[addrOffset], outVarTensor, copyOutParams);
        if (tiling_->isMeanOut) {
            DataCopyPad(meanGM_[addrOffset], outMeanTensor, copyOutParams);
        }

        outQueue_.FreeTensor(outMeanTensor);
    }

    __aicore__ inline void CopyOutGroup(const Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        LocalTensor<PromoteDataType> outMeanTensor = outQueue_.DeQue<PromoteDataType>();
        LocalTensor<PromoteDataType> outVarTensor = outMeanTensor[tilingOp_->resultBlock / sizeof(PromoteDataType)];

        // CopyOut As RA Pattern
        int32_t blockId = GetBlockIdx();
        DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};

        int32_t innerA = Ops::Base::ReduceOpTmpl::CaculateInnerA<&SchLoopInfo, Pattern::TailA, Pattern::Dim>(iterAddr_);
        if constexpr (Pattern::TailA) {
            copyOutParams.blockLen = aOutBurstLen_ * sizeof(PromoteDataType);
            copyOutParams.blockCount = aOutNBurst_;
            uint64_t withPadNum = Ops::Base::CeilAlign(aOutBurstLen_, BLOCK_SIZE_BYTE / sizeof(DataType));
            copyOutParams.srcStride = (withPadNum - aOutBurstLen_) * sizeof(PromoteDataType) / BLOCK_SIZE_BYTE;
        } else {
            copyOutParams.blockLen = shape.value[0] * sizeof(PromoteDataType);
            copyOutParams.blockCount = 1;
        }
        int32_t axis = Pattern::FirstA ? 0 : 1;
        if constexpr (SchLoopInfo.loopACount > 0) {
            axis = SchLoopInfo.loopAAxis[SchLoopInfo.loopACount - 1];
        }

        uint64_t addrOffset = 0;
        if constexpr (SchLoopInfo.loopInnerACount > 0) {
            for (int32_t i = axis; i < Dim; i += Ops::Base::ReduceOpTmpl::CONST2) {
                addrOffset += iterAddr_[i].start * tilingOp_->dstStride[i];
            }
        }

        uint64_t aSize = Ops::Base::CeilAlign(tilingOp_->outSize, static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE));
        uint64_t axisStep = SchLoopInfo.loopACount > 0 ? loopAAxisStep_ : 1;
        uint64_t addr =
            (blockId % tilingOp_->groupR) * aSize +                                         // group offset
            (blockId / (tilingOp_->groupR * axisStep)) * tilingOp_->shape[axis] * innerA +  // all A Axis offset
            (blockId / tilingOp_->groupR % axisStep) * ubFactorA_ * innerA +                // split A Axis offset
            addrOffset;                                                                     // innerA offset

        uint64_t varOffset = static_cast<uint64_t>(tiling_->workSpaceSize) / sizeof(PromoteDataType);
        DataCopyPad(workspace_[addr], outMeanTensor, copyOutParams);
        DataCopyPad(workspace_[addr + varOffset], outVarTensor, copyOutParams);

        outQueue_.FreeTensor(outMeanTensor);
    }
};
}  // namespace ReduceOpTmpl
#endif  // _REDUCE_VAR_SCH_H_