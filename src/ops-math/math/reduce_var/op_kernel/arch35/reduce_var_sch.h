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
#include "reduce_var_struct.h"
#include "reduce_var_welford.h"
#include "reduce_var_twopass.h"

namespace ReduceOpTmpl {
// view.axis[] / invDstStride[] 中的维度下标（第 5~8 维），避免魔数
constexpr int32_t AXIS_DIM4 = 4;
constexpr int32_t AXIS_DIM5 = 5;
constexpr int32_t AXIS_DIM6 = 6;
constexpr int32_t AXIS_DIM7 = 7;
// 张量维数（Dim 比较分支），避免魔数
constexpr int32_t DIM3 = 3;
constexpr int32_t DIM4 = 4;
constexpr int32_t DIM5 = 5;

template <typename DataType, typename PromoteDataType, bool batchInvariant, uint32_t PatternID, uint32_t LoopARCount,
          uint32_t LoopInnerARCount, bool isStd = false>
class ReduceVarSch {
public:
    constexpr static Ops::Base::ReduceOpTmpl::ReduceSchLoopInfo
        SchLoopInfo = Ops::Base::ReduceOpTmpl::GetSchLoopInfo<PatternID, LoopARCount, LoopInnerARCount>();
    using Pattern = typename Ops::Base::ReduceOpTmpl::__reducePattern::GetPattern<SchLoopInfo.patternID>::T;
    using InnerPattern = typename Ops::Base::ReduceOpTmpl::__reducePattern::GetPattern<SchLoopInfo.innerPatternID>::T;
    constexpr static int32_t Dim = Pattern::Dim;
    constexpr static int32_t BUFFER_NUM = 2;
    constexpr static int32_t ELEMENT_ONE_REPEAT_COMPUTE = Ops::Base::GetVRegSize() / sizeof(PromoteDataType);
    constexpr static int32_t VL_LENGTH_B = Ops::Base::GetVRegSize();
    constexpr static uint64_t BLOCK_SIZE_BYTE = Ops::Base::GetUbBlockSize();
    constexpr static int32_t POST_BUF_SIZE = 8 * 1024;
    constexpr static int32_t FLOAT32_INF = 0x7F800000; // inf
    constexpr static uint32_t WELFORD_GROUP_NUM = 8;
    constexpr static uint32_t MAX_INNER_A = 512; // Bytes, 按PromoteDataType计算
    constexpr static uint32_t MAX_INNER_A_NUM = MAX_INNER_A / sizeof(PromoteDataType);
    constexpr static uint32_t GROUP_CACHE_BUF_SIZE = (WELFORD_GROUP_NUM + 1) * MAX_INNER_A;
    // NDDMA 多维搬运的循环层数：3/4/5 分别对应 Dim==3/4/5(或退化外层后内层为 5) 时的内层 NDDMA 维度数
    constexpr static int32_t NDDMA_LOOP_DIM3 = 3;
    constexpr static int32_t NDDMA_LOOP_DIM4 = 4;
    constexpr static int32_t NDDMA_LOOP_DIM5 = 5;

private:
    TPipe* pipe_ = nullptr;
    TBuf<> buf_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    GlobalTensor<DataType> inputGM_;
    GlobalTensor<DataType> varGM_;
    GlobalTensor<DataType> meanGM_;
    GlobalTensor<PromoteDataType> workspace_;

    LocalTensor<PromoteDataType> tMeanTensor_; // welford cache var buf
    LocalTensor<PromoteDataType> tVarTensor_;  // welford cache mean buf
    LocalTensor<PromoteDataType> tDichAddTensor_;
    LocalTensor<PromoteDataType> tCountTensor_;
    LocalTensor<PromoteDataType> tGroupMeanTensor_; // welford group cache mean buf
    LocalTensor<PromoteDataType> tGroupVarTensor_;  // welford group cache var buf

    const ReduceVarTilingData* tiling_ = nullptr;

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
    uint64_t splitRAxisTail_ = 0; // R轴的ub切分的尾块
    uint64_t ubFactorR_ = 0;

    int64_t rCount_ = 0;            // r loop count
    int64_t lastReduceTailR_ = 0;   // ubRfactor存在尾块时, welford尾块的长度
    int64_t lastReduceMainR_ = 0;   // welford主块的真实R长度（TailA invert finalize 用）
    uint32_t loopLastRCnt_ = 1;     // 单次ub内，总共包含多少个lastR
    uint32_t loopWelfTailRCnt_ = 1; // 单次ub内，welford尾块包含多少个lastR
    // NDDMA 转置模板 (CopyInWithNddmaInvert) 的行 stride（主块几何）。
    // 尾块 R 变短不能改变 slab 行 stride，否则主尾块 lane 对不齐；
    // 每次 Welford 调用的首次 CopyInX（必为主块或全尾块场景的首块）在 CalcInnerShape 时刷新
    uint32_t invOtherAlign_ = 0;

    uint64_t aOutBurstLen_ = 0;
    uint64_t aOutNBurst_ = 0;

    uint32_t rCntGroupWelford_[WELFORD_GROUP_NUM + 1] = {0};
    int32_t rCntGroupIdx_ = 0;
    int32_t dstGroupGroupIdx_ = 0;
    bool isInvert_ = false;

    // NDDMA 转置模板 (CopyInWithNddmaInvert) 下 UB 内真实数据数（不含 pad）
    // outer==1 短路分支：ubRealABundle_ = 转置后 slab 行数，ubRealRBundle_ = 每行真实数据数
    //   （!TailA: 行=R, 每行=A；TailA: 行=A, 每行=R）
    // 多维分支：ubRealABundle_ = 所有 A 轴 repeat 乘积，ubRealRBundle_ = 所有 R 轴 repeat 乘积
    // shape.value[1] 是含 pad 的 inner row stride（otherAlign），不能当真实 R 长度用
    uint32_t ubRealABundle_ = 0;
    uint32_t ubRealRBundle_ = 0;

    struct {
        uint64_t start = 0;
        uint64_t stride = 1; // 拷贝步长
    } iterAddr_[Dim];

public:
    __aicore__ inline explicit ReduceVarSch(const ReduceVarTilingData* tiling) { tiling_ = tiling; };

    __aicore__ inline void Init(TPipe* pipeIn, GM_ADDR x, GM_ADDR var, GM_ADDR mean, GM_ADDR workspace)
    {
        pipe_ = pipeIn;
        blockIdx_ = GetBlockIdx();
        basicBlockLen_ = tiling_->basicBlock;

        inputGM_.SetGlobalBuffer((__gm__ DataType*)x);
        varGM_.SetGlobalBuffer((__gm__ DataType*)var);
        if (mean != nullptr) {
            meanGM_.SetGlobalBuffer((__gm__ DataType*)mean);
        }
        workspace_.SetGlobalBuffer((__gm__ PromoteDataType*)workspace);

        pipe_->InitBuffer(inputQueue_, BUFFER_NUM, basicBlockLen_);
        // resultBlock 是按fp32预留的, 为单个mean/var的大小，mean + var, 开db
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, tiling_->resultBlock * Ops::Base::ReduceOpTmpl::CONST2);

        int64_t totalBufSize = 0;
        int64_t tmpCacheBufNum = 0;
        if constexpr (IsSameType<PromoteDataType, DataType>::value) {
            // dichotomyAddBuf_ 后续可以优化成只有一半
            //               meanBuf_         varBuf_      dichotomyAddBuf_    tCountBuff_
            totalBufSize = basicBlockLen_ + basicBlockLen_ + basicBlockLen_ +
                           basicBlockLen_ / Ops::Base::ReduceOpTmpl::CONST2;
            tmpCacheBufNum = basicBlockLen_ / sizeof(PromoteDataType);
        } else {
            totalBufSize = (basicBlockLen_ + basicBlockLen_ + basicBlockLen_) * Ops::Base::ReduceOpTmpl::CONST2 +
                           basicBlockLen_;
            tmpCacheBufNum = basicBlockLen_ * Ops::Base::ReduceOpTmpl::CONST2 / sizeof(PromoteDataType);
        }

        totalBufSize += GROUP_CACHE_BUF_SIZE * Ops::Base::ReduceOpTmpl::CONST2; // CONST2: groupMeanBuf_ + groupVarBuf_
        pipe_->InitBuffer(buf_, totalBufSize);

        tMeanTensor_ = buf_.Get<PromoteDataType>();
        tVarTensor_ = tMeanTensor_[tmpCacheBufNum];
        tDichAddTensor_ = tVarTensor_[tmpCacheBufNum];
        tCountTensor_ = tDichAddTensor_[tmpCacheBufNum];
        tGroupMeanTensor_ = tCountTensor_[tmpCacheBufNum / Ops::Base::ReduceOpTmpl::CONST2];
        tGroupVarTensor_ = tGroupMeanTensor_[GROUP_CACHE_BUF_SIZE / sizeof(PromoteDataType)];

        for (uint64_t i = 0; i < Dim; i++) {
            iterAddr_[i].stride = tiling_->shape[i];
        }

        lastRAxisLen_ = tiling_->shape[Dim - 1];
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
        rCount_ = tiling_->factorRCntPerCore;
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
        ubFactorR_ = tiling_->groupR;

        int32_t blockIdx = GetBlockIdx();
        int64_t factorATotalCnt = Ops::Base::CeilDiv(tiling_->outSize, static_cast<uint64_t>(ubFactorA_));
        int64_t factorACntPerCore = Ops::Base::CeilDiv(factorATotalCnt, static_cast<int64_t>(tiling_->coreNum));

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
        loopAStartIndex_ = blockId * tiling_->factorACntPerCore;
        loopAEndIndex_ = loopAStartIndex_ + tiling_->factorACntPerCore;
        if (unlikely(loopAEndIndex_ > tiling_->factorATotalCnt)) {
            loopAEndIndex_ = tiling_->factorATotalCnt;
        }
        constexpr int32_t aAxisIdx = SchLoopInfo.loopACount - 1;
        constexpr int32_t aAxis = SchLoopInfo.loopAAxis[aAxisIdx];
        loopAAxisStep_ = Ops::Base::CeilDiv(tiling_->shape[aAxis], tiling_->ubFactorA);

        if constexpr (SchLoopInfo.loopInnerRCount > 0) {
            constexpr int32_t rAxisIdx = SchLoopInfo.loopInnerRCount - 1;
            constexpr int32_t rAxis = SchLoopInfo.loopInnerRAxis[rAxisIdx];
            lastRAxisNum_ = tiling_->shape[rAxis];
            loopRAxisStep_ = Ops::Base::CeilDiv(lastRAxisNum_, tiling_->ubFactorR);
            splitRAxisTail_ = tiling_->shape[rAxis] % tiling_->ubFactorR;
        }

        ubFactorA_ = tiling_->ubFactorA;
        ubFactorR_ = tiling_->ubFactorR;
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
                iterAddr_[axis].stride = tiling_->shape[axis] - iterAddr_[axis].start;
                if (likely(iterAddr_[axis].stride >= ubFactorA_)) {
                    iterAddr_[axis].stride = ubFactorA_;
                }

                if constexpr (LoopAIdx > 0) {
                    CalcIterA<LoopAIdx - 1>(step / loopAAxisStep_);
                }
            } else {
                iterAddr_[axis].start = step % tiling_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalcIterA<LoopAIdx - 1>(step / tiling_->shape[axis]);
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
            uint64_t shape = tiling_->shape[axis];
            if constexpr (start + 1 == end) { // 为最内轴
                uint64_t loopSize = shape / ubFactorA_;
                uint64_t tail = shape - loopSize * ubFactorA_;
                iterAddr_[axis].start = 0;
                iterAddr_[axis].stride = ubFactorA_;

                for (uint64_t i = 0; i < loopSize; i++) { // 整块
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
        __ubuf__ DataType* xLocal = (__ubuf__ DataType*)inputUb.GetPhyAddr();
        __ubuf__ float* dichotomyAddAddr = (__ubuf__ float*)tDichAddTensor_.GetPhyAddr();

        LocalTensor<DataType> outMeanTensor = outQueue_.AllocTensor<DataType>();
        LocalTensor<DataType> outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(DataType)];
        __ubuf__ DataType* outMeanAddr = (__ubuf__ DataType*)outMeanTensor.GetPhyAddr();
        __ubuf__ DataType* outVarAddr = (__ubuf__ DataType*)outVarTensor.GetPhyAddr();

        float varScale = tiling_->correctionInvalid == 1 ? (*((float*)&FLOAT32_INF)) : tiling_->varFactor;
        float meanScale = tiling_->meanFactor;

        if constexpr (!InnerPattern::TailA) {
            if (tiling_->isInvert == 1) {
                // AR→RA: UB data is transposed, call RA VF with swapped shape
                VFMeanVarTwoPassRA<DataType, isStd>(xLocal, tDichAddTensor_, tMeanTensor_, tVarTensor_, outMeanTensor,
                                                    outVarTensor, shape.value[1], shape.value[0], varScale);
            } else {
                bool useARBranch = (tiling_->useNddma == 1) ||
                                   (lastRAxisLen_ % (BLOCK_SIZE_BYTE / sizeof(DataType)) == 0) ||
                                   (Dim == Ops::Base::ReduceOpTmpl::CONST2);
                if (useARBranch) {
                    uint32_t realRLen;
                    if (Dim == Ops::Base::ReduceOpTmpl::CONST2) {
                        realRLen = (uint32_t)lastRAxisLen_;
                    } else if (tiling_->useNddma == 1) {
                        // NDDMA: shape.value[1] 是尾 pad 后的总长, 真数据数 = lastR * loopCnt
                        realRLen = (uint32_t)(lastRAxisLen_ * loopLastRCnt_);
                    } else {
                        realRLen = (uint32_t)shape.value[1];
                    }
                    VFMeanVarTwoPassAR<DataType, isStd>(xLocal, dichotomyAddAddr, outMeanAddr, outVarAddr,
                                                        shape.value[0], shape.value[1], realRLen, varScale);
                } else {
                    VFMeanVarTwoPassARPad<DataType, isStd>(
                        xLocal, dichotomyAddAddr, outMeanAddr, outVarAddr, shape.value[0], shape.value[1],
                        lastRAxisLen_ * loopLastRCnt_, varScale, lastRAxisLen_, lastRAxisLenAlign_, loopLastRCnt_);
                }
            }
        } else {
            if (tiling_->isInvert == 1) {
                // RA→AR: UB data is transposed, call AR VF with swapped shape
                uint32_t realRLen = ubRealRBundle_;
                VFMeanVarTwoPassAR<DataType, isStd>(xLocal, dichotomyAddAddr, outMeanAddr, outVarAddr, shape.value[0],
                                                    shape.value[1], realRLen, varScale);
            } else {
                VFMeanVarTwoPassRA<DataType, isStd>(xLocal, tDichAddTensor_, tMeanTensor_, tVarTensor_, outMeanTensor,
                                                    outVarTensor, shape.value[1], shape.value[0], varScale);
            }
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
            if (tiling_->shape[axis] - start < ubFactorR_) {
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
                                         LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor,
                                         int64_t& count, int64_t& tailsNum)
    {
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM> view;
        __ubuf__ float* meanBufAddr = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* varBufAddr = (__ubuf__ float*)tVarTensor.GetPhyAddr();

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
            __ubuf__ DataType* xLocal = (__ubuf__ DataType*)inputUb.GetPhyAddr();

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
                __ubuf__ DataType* xLocal = (__ubuf__ DataType*)inputUb.GetPhyAddr();
                count = count + 1;
                tailsNum = tailsNum + 1;
                scale = static_cast<float>(1.0) / static_cast<float>(count);
                if constexpr (!InnerPattern::TailA) {
                    if (count == 1) {
                        processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                        VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    } else if (tiling_->useNddma == 1 && tiling_->isInvert == 1) {
                        // !TailA + isInvert: slab 已转置为 [R 行 × A 列]，尾块 = R 行数变少，
                        // 平坦更新前 lastReduceTailR_ 行（pad 列 x=0，mean/var 保持 0 不被污染）
                        processNum = static_cast<uint32_t>(lastReduceTailR_ * shape.value[1]);
                        VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
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
                        if (tiling_->useNddma == 1 && tiling_->isInvert == 1) {
                            // 同 WelfordUpdateGroups: slab 几何主尾一致，尾块只更新每行真实 lane
                            VFWelfordParallelUpdateARWithTail(xLocal, meanBufAddr, varBufAddr, ubRealABundle_,
                                                              static_cast<uint32_t>(shape.value[1]), ubRealRBundle_,
                                                              scale);
                        } else {
                            processNum = static_cast<uint32_t>(lastReduceTailR_ * shape.value[1]);
                            VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                        }
                    }
                }
                inputQueue_.FreeTensor(inputUb);
            }
        }
    }

    __aicore__ inline void WelfordUpdateGroups(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
                                               LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor,
                                               int64_t& count, int64_t& tailsNum)
    {
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM> view;
        bool hasTail = false;
        bool calcShape = true;
        uint32_t updateCycleCnt = 0;

        WelfordUpdateGroupsMainLoop(shape, view, calcShape, tMeanTensor, tVarTensor, count, updateCycleCnt, hasTail);
        if (updateCycleCnt != 0) {
            VFWelfordParallelFinalizeGroups(shape, false, tMeanTensor, tVarTensor, tGroupMeanTensor_, tGroupVarTensor_,
                                            updateCycleCnt);
            updateCycleCnt = 0;
        }
        if (hasTail == true) {
            WelfordUpdateGroupsTailLoop(shape, view, calcShape, tMeanTensor, tVarTensor, count, tailsNum,
                                        updateCycleCnt);
        }
    }

    // 主块（非尾块）welford 累加循环：整块优先 update，达到 WELFORD_GROUP_NUM 触发 finalize
    __aicore__ inline void WelfordUpdateGroupsMainLoop(
        Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view, bool& calcShape,
        LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor, int64_t& count, uint32_t& updateCycleCnt,
        bool& hasTail)
    {
        __ubuf__ float* meanBufAddr = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* varBufAddr = (__ubuf__ float*)tVarTensor.GetPhyAddr();

        uint32_t processNum = 0;
        // 可能存在多个尾块的场景，如整块、尾块、整块、尾块..., 可以先做整块的update，再做尾块的update
        for (int64_t i = 0; i < rCount_; i++) {
            if (CheckTailWelford(i) == true) {
                hasTail = true;
                continue;
            }
            CopyInX(i, view, shape, calcShape);

            LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
            __ubuf__ DataType* xLocal = (__ubuf__ DataType*)inputUb.GetPhyAddr();

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
    }

    // 尾块 welford 累加循环：处理 CheckTailWelford 为 true 的块，按 TailA/isInvert 分支选择 update API
    __aicore__ inline void WelfordUpdateGroupsTailLoop(
        Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view, bool& calcShape,
        LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor, int64_t& count, int64_t& tailsNum,
        uint32_t& updateCycleCnt)
    {
        __ubuf__ float* meanBufAddr = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* varBufAddr = (__ubuf__ float*)tVarTensor.GetPhyAddr();

        uint32_t processNum = 0;
        for (int64_t i = 0; i < rCount_; i++) {
            if (CheckTailWelford(i) == false) {
                continue;
            }
            CopyInX(i, view, shape, calcShape);
            LocalTensor<DataType> inputUb = inputQueue_.DeQue<DataType>();
            __ubuf__ DataType* xLocal = (__ubuf__ DataType*)inputUb.GetPhyAddr();

            count = count + 1;
            updateCycleCnt = updateCycleCnt + 1;
            tailsNum = tailsNum + 1;
            float scale = static_cast<float>(1.0) / static_cast<float>(updateCycleCnt);
            if constexpr (!InnerPattern::TailA) {
                if (updateCycleCnt == 1) {
                    processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                    VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                } else {
                    uint32_t tailAWithTail = (tiling_->useNddma == 1 && tiling_->isInvert == 1) ?
                                                 ubRealABundle_ :
                                                 static_cast<uint32_t>(shape.value[0]);
                    uint32_t tailRealTail = (tiling_->useNddma == 1 && tiling_->isInvert == 1) ?
                                                ubRealRBundle_ :
                                                static_cast<uint32_t>(lastReduceTailR_);
                    VFWelfordParallelUpdateARWithTail(xLocal, meanBufAddr, varBufAddr, tailAWithTail,
                                                      static_cast<uint32_t>(shape.value[1]), tailRealTail, scale);
                }
            } else {
                if (updateCycleCnt == 1) {
                    processNum = static_cast<uint32_t>(shape.value[0] * shape.value[1]);
                    VFWelfordParallelUpdateWithInit(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                } else {
                    if (tiling_->useNddma == 1 && tiling_->isInvert == 1) {
                        VFWelfordParallelUpdateARWithTail(xLocal, meanBufAddr, varBufAddr, ubRealABundle_,
                                                          static_cast<uint32_t>(shape.value[1]), ubRealRBundle_, scale);
                    } else {
                        processNum = static_cast<uint32_t>(lastReduceTailR_ * shape.value[1]);
                        VFWelfordParallelUpdate(xLocal, meanBufAddr, varBufAddr, processNum, scale);
                    }
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
            VFWelfordParallelFinalizeGroups(shape, true, tMeanTensor, tVarTensor, tGroupMeanTensor_, tGroupVarTensor_,
                                            updateCycleCnt);
            updateCycleCnt = 0;
        }
    }

    __aicore__ inline void VFWelfordParallelFinalizeGroups(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
                                                           bool isTail, LocalTensor<float>& tMeanTensor,
                                                           LocalTensor<float>& tVarTensor,
                                                           LocalTensor<float>& groupMeanTensor,
                                                           LocalTensor<float>& groupVarTensor, uint32_t updateCycleCnt)
    {
        __ubuf__ float* meanBufAddr = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* varBufAddr = (__ubuf__ float*)tVarTensor.GetPhyAddr();

        int32_t entryGroupIdx = rCntGroupIdx_;
        __ubuf__ float* groupMeanBufAddr = (__ubuf__ float*)groupMeanTensor.GetPhyAddr() +
                                           entryGroupIdx * MAX_INNER_A_NUM;
        __ubuf__ float* groupVarBufAddr = (__ubuf__ float*)groupVarTensor.GetPhyAddr() +
                                          entryGroupIdx * MAX_INNER_A_NUM;

        __ubuf__ float* dichotomyAddLocal = (__ubuf__ float*)tDichAddTensor_.GetPhyAddr();

        if constexpr (!InnerPattern::TailA) {
            FinalizeGroupsNonTailA(shape, isTail, updateCycleCnt, entryGroupIdx, meanBufAddr, varBufAddr,
                                   groupMeanBufAddr, groupVarBufAddr, dichotomyAddLocal, tMeanTensor, tVarTensor,
                                   groupMeanTensor, groupVarTensor);
        } else {
            FinalizeGroupsTailA(shape, isTail, updateCycleCnt, meanBufAddr, varBufAddr, dichotomyAddLocal, tMeanTensor,
                                tVarTensor, groupMeanTensor, groupVarTensor);
        }

        FinalizeGroupsConsolidate(dichotomyAddLocal, groupMeanTensor, groupVarTensor);
    }

    // !TailA 分支：AR pattern，按 isLastAlign / isInvert 决定 finalize 调用与 rCntGroupWelford_ 记账
    __aicore__ inline void FinalizeGroupsNonTailA(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape, bool isTail,
                                                  uint32_t updateCycleCnt, int32_t entryGroupIdx,
                                                  __ubuf__ float* meanBufAddr, __ubuf__ float* varBufAddr,
                                                  __ubuf__ float* groupMeanBufAddr, __ubuf__ float* groupVarBufAddr,
                                                  __ubuf__ float* dichotomyAddLocal, LocalTensor<float>& tMeanTensor,
                                                  LocalTensor<float>& tVarTensor, LocalTensor<float>& groupMeanTensor,
                                                  LocalTensor<float>& groupVarTensor)
    {
        uint32_t aNum = static_cast<uint32_t>(shape.value[0]);
        uint32_t rNum = static_cast<uint32_t>(shape.value[1]);
        uint32_t rStride = static_cast<uint32_t>(shape.value[1]);
        bool isLastAlign = ((Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) ||
                            (tiling_->useNddma == 1) ||
                            (tiling_->shape[Dim - 1] % (BLOCK_SIZE_BYTE / sizeof(DataType)) == 0));
        if (isLastAlign) {
            if (isTail) {
                rNum = (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) ?
                           splitRAxisTail_ :
                           lastRAxisLen_ * loopWelfTailRCnt_;
            } else if (tiling_->useNddma == 1) {
                rNum = static_cast<uint32_t>(lastRAxisLen_ * loopLastRCnt_);
            }
            if (tiling_->isInvert == 1) {
                aNum = ubRealABundle_;
                rNum = ubRealRBundle_;
            }

            rCntGroupWelford_[rCntGroupIdx_] = static_cast<uint32_t>(rNum * updateCycleCnt);
            rCntGroupIdx_ = isInvert_ ? (rCntGroupIdx_ - 1) : (rCntGroupIdx_ + 1);

            float meanScale = (rNum == 0) ? 1.0f : (1.0f * updateCycleCnt) / static_cast<float>(rNum * updateCycleCnt);

            if (tiling_->isInvert == 1) {
                __ubuf__ float* tmpCountLocal = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
                uint32_t RNumRA = rNum;
                uint32_t ANumRA = rStride;
                int64_t tailRNumRA = 0;
                int64_t addCntRA = updateCycleCnt;
                int64_t addTailCntRA = updateCycleCnt;
                CaculateCountBuf(tmpCountLocal, RNumRA, tailRNumRA, addCntRA, addTailCntRA);
                float meanScaleRA = (updateCycleCnt * RNumRA == 0) ? 1.0f :
                                                                     1.0f / static_cast<float>(updateCycleCnt * RNumRA);
                LocalTensor<float> dstGroupMeanRA = groupMeanTensor[entryGroupIdx * MAX_INNER_A_NUM];
                LocalTensor<float> dstGroupVarRA = groupVarTensor[entryGroupIdx * MAX_INNER_A_NUM];
                VFWelfordFinalizeRA<float, isStd, true>(RNumRA, ANumRA, tMeanTensor, tVarTensor, tmpCountLocal,
                                                        dstGroupMeanRA, dstGroupVarRA, dichotomyAddLocal, meanScaleRA,
                                                        1.0f);
            } else {
                VFWelfordParallelFinalizeARAlign<float, isStd, true>(meanBufAddr, varBufAddr, dichotomyAddLocal,
                                                                     groupMeanBufAddr, groupVarBufAddr, aNum, rNum,
                                                                     rStride, 1.0f, meanScale, updateCycleCnt);
            }
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
    }

    // TailA 分支：RA pattern，按 isInvert 选择对齐 finalize 或 RA finalize
    __aicore__ inline void FinalizeGroupsTailA(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape, bool isTail,
                                               uint32_t updateCycleCnt, __ubuf__ float* meanBufAddr,
                                               __ubuf__ float* varBufAddr, __ubuf__ float* dichotomyAddLocal,
                                               LocalTensor<float>& tMeanTensor, LocalTensor<float>& tVarTensor,
                                               LocalTensor<float>& groupMeanTensor, LocalTensor<float>& groupVarTensor)
    {
        if (tiling_->isInvert == 1) {
            uint32_t aNum = static_cast<uint32_t>(shape.value[0]);
            uint32_t rNum = ubRealRBundle_;
            uint32_t rStride = static_cast<uint32_t>(shape.value[1]);
            LocalTensor<float> dstGroupMeanInv = groupMeanTensor[rCntGroupIdx_ * MAX_INNER_A_NUM];
            LocalTensor<float> dstGroupVarInv = groupVarTensor[rCntGroupIdx_ * MAX_INNER_A_NUM];
            __ubuf__ float* dstGroupMeanAddrInv = (__ubuf__ float*)dstGroupMeanInv.GetPhyAddr();
            __ubuf__ float* dstGroupVarAddrInv = (__ubuf__ float*)dstGroupVarInv.GetPhyAddr();

            rCntGroupWelford_[rCntGroupIdx_] = rNum * updateCycleCnt;
            rCntGroupIdx_ = isInvert_ ? (rCntGroupIdx_ - 1) : (rCntGroupIdx_ + 1);

            float meanScale = (rNum == 0) ? 1.0f : (1.0f * updateCycleCnt) / static_cast<float>(rNum * updateCycleCnt);

            VFWelfordParallelFinalizeARAlign<float, isStd, true>(meanBufAddr, varBufAddr, dichotomyAddLocal,
                                                                 dstGroupMeanAddrInv, dstGroupVarAddrInv, aNum, rNum,
                                                                 rStride, 1.0f, meanScale, updateCycleCnt);
        } else {
            // dichotomyAddLocal RA场景下空间分配
            __ubuf__ float* tmpCountLocal = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
            uint32_t RNum = isTail ? lastReduceTailR_ : shape.value[0];
            uint32_t ANum = shape.value[1];

            int64_t tailRNum = 0;
            int64_t addCnt = updateCycleCnt;
            int64_t addTailCnt = updateCycleCnt; // welford 累加次数, addTailCnt >= addCnt
            CaculateCountBuf(tmpCountLocal, RNum, tailRNum, addCnt, addTailCnt);

            float meanScale = (updateCycleCnt * RNum == 0) ? 1.0f : 1.0f / static_cast<float>(updateCycleCnt * RNum);
            LocalTensor<float> dstGroupMean = groupMeanTensor[rCntGroupIdx_ * MAX_INNER_A_NUM];
            LocalTensor<float> dstGroupVar = groupVarTensor[rCntGroupIdx_ * MAX_INNER_A_NUM];
            VFWelfordFinalizeRA<float, isStd, true>(RNum, ANum, tMeanTensor, tVarTensor, tmpCountLocal, dstGroupMean,
                                                    dstGroupVar, dichotomyAddLocal, meanScale, 1.0f);

            rCntGroupWelford_[rCntGroupIdx_] = static_cast<uint32_t>(RNum * updateCycleCnt);
            rCntGroupIdx_ = isInvert_ ? (rCntGroupIdx_ - 1) : (rCntGroupIdx_ + 1);
        }
    }

    // 组归约收尾：group buffer 写满后，将 WELFORD_GROUP_NUM 个组再次 finalize 成单组并翻转 isInvert_
    __aicore__ inline void FinalizeGroupsConsolidate(__ubuf__ float* dichotomyAddLocal,
                                                     LocalTensor<float>& groupMeanTensor,
                                                     LocalTensor<float>& groupVarTensor)
    {
        if ((isInvert_ && rCntGroupIdx_ <= 0) || (!isInvert_ && rCntGroupIdx_ >= WELFORD_GROUP_NUM)) {
            // RA finalize
            int64_t totalCnt = 0;
            __ubuf__ float* tmpCountLocal = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
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
        LocalTensor<T> outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(T)];
        __ubuf__ T* outMeanAddr = (__ubuf__ T*)outMeanTensor.GetPhyAddr();
        __ubuf__ T* outVarAddr = (__ubuf__ T*)outVarTensor.GetPhyAddr();

        __ubuf__ float* dichotomyAddLocal = (__ubuf__ float*)tDichAddTensor_.GetPhyAddr();

        float varScale = tiling_->varFactor;
        if (tiling_->correctionInvalid == 1) {
            varScale = *((float*)&FLOAT32_INF);
        }

        float meanScale = tiling_->meanFactor;
        if constexpr (SchLoopInfo.loopRCount > 0) {
            meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
        }

        // RA finalize
        int64_t totalRCnt = 0;
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
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
    __aicore__ inline void WelfordFinalize(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape, int64_t count,
                                           int64_t tailsNum, LocalTensor<float>& tMeanTensor,
                                           LocalTensor<float>& tVarTensor)
    {
        LocalTensor<T> outMeanTensor = outQueue_.AllocTensor<T>();
        LocalTensor<T> outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(T)];

        float varScale = tiling_->varFactor;
        if (tiling_->correctionInvalid == 1) {
            varScale = *((float*)&FLOAT32_INF);
        }

        bool enqueued = false;
        if constexpr (!InnerPattern::TailA) {
            enqueued = WelfordFinalizeNonTailA<T, isM2Out>(shape, count, tailsNum, tMeanTensor, tVarTensor,
                                                           outMeanTensor, outVarTensor, varScale);
        } else {
            enqueued = WelfordFinalizeTailA<T, isM2Out>(shape, count, tailsNum, tMeanTensor, tVarTensor, outMeanTensor,
                                                        outVarTensor, varScale);
        }
        if (!enqueued) {
            outQueue_.EnQue(outMeanTensor);
        }
    }

    // !TailA 分支：AR pattern finalize。返回 true 表示已 EnQue（isInvert 路径），false 表示由调用方 EnQue
    template <typename T, bool isM2Out = false>
    __aicore__ inline bool WelfordFinalizeNonTailA(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape,
                                                   int64_t count, int64_t& tailsNum, LocalTensor<float>& tMeanTensor,
                                                   LocalTensor<float>& tVarTensor, LocalTensor<T>& outMeanTensor,
                                                   LocalTensor<T>& outVarTensor, float varScale)
    {
        __ubuf__ float* dichotomyAddLocal = (__ubuf__ float*)tDichAddTensor_.GetPhyAddr();
        __ubuf__ float* meanBufAddr = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* varBufAddr = (__ubuf__ float*)tVarTensor.GetPhyAddr();
        __ubuf__ T* outMeanAddr = (__ubuf__ T*)outMeanTensor.GetPhyAddr();
        __ubuf__ T* outVarAddr = (__ubuf__ T*)outVarTensor.GetPhyAddr();

        if (tiling_->isInvert == 1) {
            __ubuf__ float* tmpCountLocal = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
            uint32_t RNum = static_cast<uint32_t>(shape.value[0]); // R 行数（主块几何）
            uint32_t ANum = static_cast<uint32_t>(shape.value[1]); // 行 stride（A 对齐长度）
            if (count == tailsNum) {
                tailsNum = 0;
                RNum = static_cast<uint32_t>(lastReduceTailR_);
            }
            int64_t tailRNum = (tailsNum == 0) ? 0 : lastReduceTailR_;
            int64_t addCnt = count - tailsNum;
            int64_t addTailCnt = count; // welford 累加次数, addTailCnt >= addCnt
            CaculateCountBuf(tmpCountLocal, RNum, tailRNum, addCnt, addTailCnt);

            float meanScale = tiling_->meanFactor;
            if constexpr (SchLoopInfo.loopRCount > 0) {
                meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
            }
            VFWelfordFinalizeRA<T, isStd, isM2Out>(RNum, ANum, tMeanTensor, tVarTensor, tmpCountLocal, outMeanTensor,
                                                   outVarTensor, dichotomyAddLocal, meanScale, varScale);
            outQueue_.EnQue(outMeanTensor);
            return true;
        }
        uint32_t aNum = static_cast<uint32_t>(shape.value[0]);
        uint32_t rNum = static_cast<uint32_t>(shape.value[1]);
        uint32_t rStride = static_cast<uint32_t>(shape.value[1]);
        if (count == tailsNum) {
            tailsNum = 0;
            rNum = static_cast<uint32_t>(lastReduceTailR_);
        }

        bool isLastAlign = ((Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) ||
                            (tiling_->useNddma == 1) ||
                            (tiling_->shape[Dim - 1] % (BLOCK_SIZE_BYTE / sizeof(DataType)) == 0));
        // NDDMA 场景: shape.value[1] 是尾 pad 后总长, rNum 需修正为真实数据数 (rStride 保持 shape.value[1])
        if (tiling_->useNddma == 1 && !(Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) &&
            count != tailsNum) {
            rNum = static_cast<uint32_t>(lastRAxisLen_ * loopLastRCnt_);
        }

        if (tiling_->isInvert == 1) {
            aNum = ubRealABundle_;
            rNum = ubRealRBundle_;
        }

        float meanScale = tiling_->meanFactor;
        if (tailsNum == 0) {
            meanScale = (float)count * tiling_->meanFactor; // 无尾块场景, meanscale需要乘以count
            if constexpr (SchLoopInfo.loopRCount > 0) {
                meanScale = float(count) /
                            static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
            }
            if (isLastAlign) {
                // 不带pad
                VFWelfordParallelFinalizeARAlign<T, isStd, isM2Out>(meanBufAddr, varBufAddr, dichotomyAddLocal,
                                                                    outMeanAddr, outVarAddr, aNum, rNum, rStride,
                                                                    varScale, meanScale, count);
            } else {
                // 带pad
                VFWelfordParallelFinalizeARAlignPad<T, isStd, isM2Out>(
                    meanBufAddr, varBufAddr, dichotomyAddLocal, outMeanAddr, outVarAddr, aNum, rNum, rStride, varScale,
                    meanScale, count, lastRAxisLen_, lastRAxisLenAlign_, loopLastRCnt_);
            }
        } else if (isLastAlign) {
            meanScale = tiling_->meanFactor;
            if constexpr (SchLoopInfo.loopRCount > 0) {
                meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
            }
            VFWelfordParallelFinalizeARNonAlign<T, isStd, isM2Out>(
                meanBufAddr, varBufAddr, dichotomyAddLocal, outMeanAddr, outVarAddr, aNum, rNum, rStride, varScale,
                meanScale, count - tailsNum, count, lastReduceTailR_);
        } else {
            meanScale = tiling_->meanFactor;
            if constexpr (SchLoopInfo.loopRCount > 0) {
                meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
            }
            VFWelfordParallelFinalizeARNonAlignPad<T, isStd, isM2Out>(
                tMeanTensor, tVarTensor, tDichAddTensor_, outMeanTensor, outVarTensor, aNum, rNum, rStride, varScale,
                meanScale, count - tailsNum, count, lastReduceTailR_, loopWelfTailRCnt_, lastRAxisLen_,
                lastRAxisLenAlign_, loopLastRCnt_);
        }
        return false;
    }

    // TailA 分支：RA pattern finalize。返回 true 表示已 EnQue（isInvert 路径），false 表示由调用方 EnQue
    template <typename T, bool isM2Out = false>
    __aicore__ inline bool WelfordFinalizeTailA(Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape, int64_t count,
                                                int64_t& tailsNum, LocalTensor<float>& tMeanTensor,
                                                LocalTensor<float>& tVarTensor, LocalTensor<T>& outMeanTensor,
                                                LocalTensor<T>& outVarTensor, float varScale)
    {
        __ubuf__ float* dichotomyAddLocal = (__ubuf__ float*)tDichAddTensor_.GetPhyAddr();
        __ubuf__ float* meanBufAddr = (__ubuf__ float*)tMeanTensor.GetPhyAddr();
        __ubuf__ float* varBufAddr = (__ubuf__ float*)tVarTensor.GetPhyAddr();
        __ubuf__ T* outMeanAddr = (__ubuf__ T*)outMeanTensor.GetPhyAddr();
        __ubuf__ T* outVarAddr = (__ubuf__ T*)outVarTensor.GetPhyAddr();

        if (tiling_->isInvert == 1) {
            uint32_t aNum = static_cast<uint32_t>(shape.value[0]);
            uint32_t rNum = 0;
            uint32_t rStride = static_cast<uint32_t>(shape.value[1]);
            if (count == tailsNum) {
                tailsNum = 0;
                rNum = static_cast<uint32_t>(lastReduceTailR_);
            } else {
                rNum = static_cast<uint32_t>(lastReduceMainR_);
            }
            float meanScale = tiling_->meanFactor;
            if (tailsNum == 0) {
                meanScale = (float)count * tiling_->meanFactor;
                if constexpr (SchLoopInfo.loopRCount > 0) {
                    meanScale = float(count) /
                                static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
                }
                VFWelfordParallelFinalizeARAlign<T, isStd, isM2Out>(meanBufAddr, varBufAddr, dichotomyAddLocal,
                                                                    outMeanAddr, outVarAddr, aNum, rNum, rStride,
                                                                    varScale, meanScale, count);
            } else {
                meanScale = tiling_->meanFactor;
                if constexpr (SchLoopInfo.loopRCount > 0) {
                    meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
                }
                VFWelfordParallelFinalizeARNonAlign<T, isStd, isM2Out>(
                    meanBufAddr, varBufAddr, dichotomyAddLocal, outMeanAddr, outVarAddr, aNum, rNum, rStride, varScale,
                    meanScale, count - tailsNum, count, lastReduceTailR_);
            }
            outQueue_.EnQue(outMeanTensor);
            return true;
        }
        // dichotomyAddLocal RA场景下空间分配
        __ubuf__ float* tmpCountLocal = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
        uint32_t RNum = shape.value[0];
        uint32_t ANum = shape.value[1];

        if (count == tailsNum) {
            tailsNum = 0;
        }
        int64_t tailRNum = (tailsNum == 0) ? 0 : lastReduceTailR_;
        int64_t addCnt = count - tailsNum;
        int64_t addTailCnt = count; // welford 累加次数, addTailCnt >= addCnt
        CaculateCountBuf(tmpCountLocal, RNum, tailRNum, addCnt, addTailCnt);

        float meanScale = tiling_->meanFactor;
        if constexpr (SchLoopInfo.loopRCount > 0) {
            meanScale = 1.0f / static_cast<float>(tiling_->reduceCntEachGroupR[blockIdx_ % tiling_->groupR]);
        }
        VFWelfordFinalizeRA<T, isStd, isM2Out>(RNum, ANum, tMeanTensor, tVarTensor, tmpCountLocal, outMeanTensor,
                                               outVarTensor, dichotomyAddLocal, meanScale, varScale);
        return false;
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
        int64_t realAnum = tiling_->outSize - static_cast<int64_t>(startWs);
        if (realAnum <= 0) {
            return;
        }
        if (likely(realAnum > ubFactorA_)) {
            realAnum = ubFactorA_;
        }

        __ubuf__ float* dichotomyAddAddr = (__ubuf__ float*)tDichAddTensor_.GetPhyAddr();
        LocalTensor<DataType> outMeanTensor = outQueue_.AllocTensor<DataType>();
        LocalTensor<DataType> outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(DataType)];

        uint64_t asize = Ops::Base::CeilAlign(tiling_->outSize, static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE));
        uint64_t varOffset = static_cast<uint64_t>(tiling_->workSpaceSize) / sizeof(PromoteDataType);
        float varScale = (tiling_->correctionInvalid == 1) ? (*((float*)&FLOAT32_INF)) : tiling_->varFactor;

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_S>(HardEvent::V_S);
        __ubuf__ float* groupCountBufAddr = (__ubuf__ float*)tCountTensor_.GetPhyAddr();
        for (int i = 0; i < tiling_->groupR; i++) {
            float reduceCnt = static_cast<float>(tiling_->reduceCntEachGroupR[i]);
            tCountTensor_.SetValue(i, reduceCnt);
        }

        DataCopyPadExtParams<PromoteDataType> padParams{true, 0, 0, static_cast<PromoteDataType>(0.0)};
        DataCopyExtParams copyInParams = {1, 1, 0, 0, 0};
        copyInParams.blockCount = tiling_->groupR;
        copyInParams.blockLen = ubFactorA_ * sizeof(PromoteDataType);
        copyInParams.srcStride = (asize - ubFactorA_) * sizeof(PromoteDataType);

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopyPad(tMeanTensor_, workspace_[startWs], copyInParams, padParams);
        DataCopyPad(tVarTensor_, workspace_[varOffset + startWs], copyInParams, padParams);

        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Ops::Base::ReduceOpTmpl::SetEvent<HardEvent::S_V>(HardEvent::S_V);

        VFWelfordFinalizeRA<DataType, isStd, false>(
            static_cast<uint32_t>(tiling_->groupR), static_cast<uint32_t>(ubFactorA_), tMeanTensor_, tVarTensor_,
            groupCountBufAddr, outMeanTensor, outVarTensor, dichotomyAddAddr, tiling_->meanFactor, varScale);

        outQueue_.EnQue(outMeanTensor);
        outMeanTensor = outQueue_.DeQue<DataType>();
        outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(DataType)];
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
        loopRStartIndex_ = blockId / tiling_->groupR * tiling_->factorRTotalCnt +
                           blockId % tiling_->groupR * tiling_->factorRCntPerCore;
        loopREndIndex_ = loopRStartIndex_ + tiling_->factorRCntPerCore;
        uint64_t maxRCnt = (blockId / tiling_->groupR + 1) * tiling_->factorRTotalCnt;
        uint64_t totalCnt = tiling_->factorATotalCnt * tiling_->factorRTotalCnt;
        maxRCnt = maxRCnt > totalCnt ? totalCnt : maxRCnt;
        if (unlikely(loopRStartIndex_ > maxRCnt)) {
            loopRStartIndex_ = maxRCnt;
        }
        if (unlikely(loopREndIndex_ > maxRCnt)) {
            loopREndIndex_ = maxRCnt;
        }

        constexpr int32_t rAxisIdx = SchLoopInfo.loopRCount - 1;
        constexpr int32_t rAxis = SchLoopInfo.loopRAxis[rAxisIdx];
        loopRAxisStep_ = Ops::Base::CeilDiv(tiling_->shape[rAxis], tiling_->ubFactorR); // 切分轴Rfactor的个数
        splitRAxisTail_ = tiling_->shape[rAxis] % tiling_->ubFactorR;

        if constexpr (SchLoopInfo.loopACount > 0) {
            constexpr int32_t aAxisIdx = SchLoopInfo.loopACount - 1;
            constexpr int32_t aAxis = SchLoopInfo.loopAAxis[aAxisIdx];
            loopAAxisStep_ = Ops::Base::CeilDiv(tiling_->shape[aAxis], tiling_->ubFactorA);
        }

        ubFactorA_ = tiling_->ubFactorA;
        ubFactorR_ = tiling_->ubFactorR;
    }

    __aicore__ inline void CopyInX(int64_t index,
                                   Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
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
                    iterAddr_[axis].stride = tiling_->shape[axis] - iterAddr_[axis].start;
                    if (likely(iterAddr_[axis].stride >= ubFactorR_)) {
                        iterAddr_[axis].stride = ubFactorR_;
                    }
                    temp = temp / loopRAxisStep_;
                } else {
                    auto axis = SchLoopInfo.loopRAxis[idx];
                    if (Ops::Base::ReduceOpTmpl::IsLoopSpliteAAxis<&SchLoopInfo>(axis)) {
                        auto cur = temp % loopAAxisStep_;
                        iterAddr_[axis].start = cur * ubFactorA_;
                        iterAddr_[axis].stride = tiling_->shape[axis] - iterAddr_[axis].start;
                        if (likely(iterAddr_[axis].stride >= ubFactorA_)) {
                            iterAddr_[axis].stride = ubFactorA_;
                        }
                        temp = temp / loopAAxisStep_;
                    } else {
                        iterAddr_[axis].start = temp % tiling_->shape[axis];
                        iterAddr_[axis].stride = 1;
                        temp = temp / tiling_->shape[axis];
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
                iterAddr_[axis].stride = tiling_->shape[axis] - iterAddr_[axis].start;
                if (likely(iterAddr_[axis].stride >= ubFactorR_)) {
                    iterAddr_[axis].stride = ubFactorR_;
                }
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / loopRAxisStep_);
            } else {
                iterAddr_[axis].start = basicBlockIdx % tiling_->shape[axis];
                iterAddr_[axis].stride = 1;
                CalcInnerIterR<LoopInnerRIdx - 1>(basicBlockIdx / tiling_->shape[axis]);
            }
        }
    }

    __aicore__ inline void CalcCopyInParam(Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view)
    {
        uint64_t addrOffset = 0;
        for (int32_t i = 0; i < Dim; i++) {
            addrOffset += iterAddr_[i].start * tiling_->stride[i];
        }

        constexpr static auto burstLenAxis = Dim - 1; // 获取搬运的最内轴的循环轴
        view.addr = addrOffset;                       // 搬运地址
        view.axis[0].repeat = Ops::Base::ReduceOpTmpl::GetBurstLen<&SchLoopInfo, burstLenAxis>(iterAddr_, tiling_);
        // burst 轴恒为 GM 最内轴：补全其轴类型（默认 0 会被误判为 R 轴，
        // TailA invert 时最内轴为 A，会导致 invDstStride/bundle 计算错误）
        view.axis[0].idx = burstLenAxis;
        view.axis[0].isAxisA = Ops::Base::ReduceOpTmpl::IsAxisA<Pattern::FirstA>(burstLenAxis);
        view.axisSize = 1; // 一次搬运时的循环轴个数

        if constexpr (burstLenAxis > 0) {
            int32_t axis = burstLenAxis;
            for (int32_t i = 1; i < Dim; i++) {
                view.axisSize = i + 1;
                view.axis[i].repeat = Ops::Base::ReduceOpTmpl::GetRepeatStride<&SchLoopInfo>(
                    axis - 1, iterAddr_, tiling_, view.axis[i].srcStride);
                view.axis[i].idx = axis - 1;
                view.axis[i].isAxisA = Ops::Base::ReduceOpTmpl::IsAxisA<Pattern::FirstA>(view.axis[i].idx);
                if (view.axis[i].idx <= 0) {
                    break;
                }
                axis = view.axis[i].idx;
            }
        }
    }

    __aicore__ inline void CalcInnerShapeLastR(
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
        Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        // NDDMA: innermost R no per-element CeilAlign, bundle then block-align at end
        int64_t value = tiling_->useNddma == 1 ?
                            view.axis[0].repeat :
                            Ops::Base::CeilAlign(view.axis[0].repeat, BLOCK_SIZE_BYTE / sizeof(DataType));
        lastReduceTailR_ = value;
        if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) {
            value = tiling_->useNddma == 1 ? ubFactorR_ :
                                             Ops::Base::CeilAlign(ubFactorR_, BLOCK_SIZE_BYTE / sizeof(DataType));
            lastReduceTailR_ = splitRAxisTail_;
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
        if (tiling_->useNddma == 1 && tiling_->isInvert == 0) {
            // NDDMA: CeilAlign after bundling all R axes as a single block
            // 仅非 invert 时需要补齐 inner dim；invert 路径 R 变为 outer dim，由 CopyIn 侧 otherAlign 保证 inner dim
            // 对齐
            value = Ops::Base::CeilAlign(static_cast<uint64_t>(value), BLOCK_SIZE_BYTE / sizeof(DataType));
        }
        shape.value[InnerPattern::Dim - 1] = value;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
            }
        }
        shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2] = value / shape.value[InnerPattern::Dim - 1];
        if (tiling_->isInvert == 1) {
            // AR→RA: swap inner/outer dims so VF sees transposed layout
            auto aBundled = shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2];
            shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2] = shape.value[InnerPattern::Dim -
                                                                                           1]; // R→outer
            // inner A 维须 VL(ELEMENT_ONE_REPEAT_COMPUTE=64)对齐: !TailA+isInvert 走 VFWelfordFinalizeRA,
            // 其内部按 stride=ANum 且每次 VL 元素访问, 非 VL 对齐会跨行读脏.
            shape.value[InnerPattern::Dim - 1] = Ops::Base::CeilAlign(
                static_cast<uint64_t>(aBundled), static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE)); // A_aligned→inner
            invOtherAlign_ = shape.value[InnerPattern::Dim - 1]; // 转置行 stride（主块几何）
            aOutBurstLen_ = aBundled;
            aOutNBurst_ = 1;
        }
    }

    __aicore__ inline void CalcInnerShapeLastA(
        Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
        Ops::Base::ReduceOpTmpl::Shape<InnerPattern::Dim>& shape)
    {
        int64_t value = tiling_->useNddma == 1 ?
                            view.axis[0].repeat :
                            Ops::Base::CeilAlign(view.axis[0].repeat, BLOCK_SIZE_BYTE / sizeof(DataType));
        aOutBurstLen_ = view.axis[0].repeat;
        if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(Dim - 1)) {
            value = tiling_->useNddma == 1 ? ubFactorA_ :
                                             Ops::Base::CeilAlign(ubFactorA_, BLOCK_SIZE_BYTE / sizeof(DataType));
            aOutBurstLen_ = ubFactorA_;
        }

        aOutNBurst_ = 1;
        lastReduceTailR_ = 1;
        lastReduceMainR_ = 1;
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
            if (tiling_->isInvert == 0) {
                // 仅非 invert 时补齐 inner dim；invert 路径该维度变为 outer，对齐由 CopyIn 侧 otherAlign 保证
                value = Ops::Base::CeilAlign(static_cast<uint64_t>(value), BLOCK_SIZE_BYTE / sizeof(DataType));
            }
        }
        shape.value[InnerPattern::Dim - 1] = value;
        for (uint64_t i = 1; i < view.axisSize; i++) {
            if (!view.axis[i].isAxisA) {
                view.axis[i].dstStride = value;
                value = value * view.axis[i].repeat;
                // 本函数仅在每次 Welford 调用的首个块（主块）执行，repeat 即主块长度
                lastReduceMainR_ = lastReduceMainR_ * view.axis[i].repeat;
                if (Ops::Base::ReduceOpTmpl::IsLoopSpliteRAxis<&SchLoopInfo>(view.axis[i].idx)) {
                    lastReduceTailR_ = lastReduceTailR_ * splitRAxisTail_;
                } else {
                    lastReduceTailR_ = lastReduceTailR_ * view.axis[i].repeat;
                }
            }
        }
        shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2] = value / shape.value[InnerPattern::Dim - 1];
        if (tiling_->isInvert == 1) {
            // RA→AR: swap inner/outer dims so VF sees transposed layout
            auto rBundled = shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2];
            shape.value[InnerPattern::Dim - Ops::Base::ReduceOpTmpl::CONST2] = shape.value[InnerPattern::Dim -
                                                                                           1]; // A→outer
            // 补齐inner dim使其与CopyIn dstStride一致，保证VF DataCopy访问每row起始地址32B对齐
            shape.value[InnerPattern::Dim - 1] = Ops::Base::CeilAlign(
                static_cast<uint64_t>(rBundled), BLOCK_SIZE_BYTE / sizeof(DataType)); // R_aligned→inner
            invOtherAlign_ = shape.value[InnerPattern::Dim - 1]; // 转置行 stride（主块几何）
            // aOutBurstLen_ already holds correct A count, keep it
            aOutNBurst_ = 1;
        }
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

    __aicore__ inline void CopyInWithNddma(
        const Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
        LocalTensor<DataType>& ubTensor)
    {
        // Step 1: Compute outer dimension product (axes beyond the first 2)
        uint64_t outer = 1;
        for (int32_t i = Ops::Base::ReduceOpTmpl::CONST2; i < view.axisSize; i++) {
            outer *= view.axis[i].repeat;
        }

        if (outer == 1) {
            // Step 2: Only 2 dims, fall back to DataCopyPad
            DataCopyPadExtParams<DataType> padParams{true, 0, 0, static_cast<DataType>(0.0)};
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = view.axis[1].repeat;
            copyInParams.blockLen = view.axis[0].repeat * sizeof(DataType);
            copyInParams.srcStride = (view.axis[1].srcStride - view.axis[0].repeat) * sizeof(DataType);
            copyInParams.dstStride = (view.axis[1].dstStride - view.axis[0].repeat) * sizeof(DataType) /
                                     BLOCK_SIZE_BYTE;
            DataCopyPad(ubTensor, inputGM_[view.addr], copyInParams, padParams);
            return;
        }

        // Step 3: Multi-dimensional NDDMA dispatch based on Dim
        static constexpr NdDmaConfig config = {false, 0, 0, false};

        if constexpr (Dim <= DIM4) {
            if constexpr (Dim == DIM3) {
                NdDmaLoopInfo<NDDMA_LOOP_DIM3> loopInfo = {
                    .loopSrcStride = {1, view.axis[1].srcStride, view.axis[2].srcStride},
                    .loopDstStride = {1, static_cast<uint32_t>(view.axis[1].dstStride),
                                      static_cast<uint32_t>(view.axis[2].dstStride)},
                    .loopSize = {static_cast<uint32_t>(view.axis[0].repeat), static_cast<uint32_t>(view.axis[1].repeat),
                                 static_cast<uint32_t>(view.axis[2].repeat)},
                    .loopLpSize = {0, 0, 0},
                    .loopRpSize = {0, 0, 0}};
                NdDmaParams<DataType, NDDMA_LOOP_DIM3> params = {loopInfo, 0};
                DataCopy<DataType, NDDMA_LOOP_DIM3, config>(ubTensor, inputGM_[view.addr], params);
            } else {
                // Dim == 4 (Dim == 1 or 2 already handled by outer == 1)
                NdDmaLoopInfo<NDDMA_LOOP_DIM4> loopInfo = {
                    .loopSrcStride = {1, view.axis[1].srcStride, view.axis[2].srcStride, view.axis[3].srcStride},
                    .loopDstStride = {1, static_cast<uint32_t>(view.axis[1].dstStride),
                                      static_cast<uint32_t>(view.axis[2].dstStride),
                                      static_cast<uint32_t>(view.axis[3].dstStride)},
                    .loopSize = {static_cast<uint32_t>(view.axis[0].repeat), static_cast<uint32_t>(view.axis[1].repeat),
                                 static_cast<uint32_t>(view.axis[2].repeat),
                                 static_cast<uint32_t>(view.axis[3].repeat)},
                    .loopLpSize = {0, 0, 0, 0},
                    .loopRpSize = {0, 0, 0, 0}};
                NdDmaParams<DataType, NDDMA_LOOP_DIM4> params = {loopInfo, 0};
                DataCopy<DataType, NDDMA_LOOP_DIM4, config>(ubTensor, inputGM_[view.addr], params);
            }
        } else {
            // Dim >= 5: inner 5 dims via NDDMA, outer dims via for loops
            NdDmaLoopInfo<NDDMA_LOOP_DIM5> loopInfo = {
                .loopSrcStride = {1, view.axis[1].srcStride, view.axis[2].srcStride, view.axis[3].srcStride,
                                  view.axis[AXIS_DIM4].srcStride},
                .loopDstStride = {1, static_cast<uint32_t>(view.axis[1].dstStride),
                                  static_cast<uint32_t>(view.axis[2].dstStride),
                                  static_cast<uint32_t>(view.axis[3].dstStride),
                                  static_cast<uint32_t>(view.axis[AXIS_DIM4].dstStride)},
                .loopSize = {static_cast<uint32_t>(view.axis[0].repeat), static_cast<uint32_t>(view.axis[1].repeat),
                             static_cast<uint32_t>(view.axis[2].repeat), static_cast<uint32_t>(view.axis[3].repeat),
                             static_cast<uint32_t>(view.axis[AXIS_DIM4].repeat)},
                .loopLpSize = {0, 0, 0, 0, 0},
                .loopRpSize = {0, 0, 0, 0, 0}};
            NdDmaParams<DataType, NDDMA_LOOP_DIM5> params = {loopInfo, 0};
            for (int32_t i = 0; i < view.axis[AXIS_DIM5].repeat; i++) {
                for (int32_t j = 0; j < view.axis[AXIS_DIM6].repeat; j++) {
                    for (int32_t k = 0; k < view.axis[AXIS_DIM7].repeat; k++) {
                        int64_t dstStride = i * view.axis[AXIS_DIM5].dstStride + j * view.axis[AXIS_DIM6].dstStride +
                                            k * view.axis[AXIS_DIM7].dstStride;
                        int64_t srcStride = i * view.axis[AXIS_DIM5].srcStride + j * view.axis[AXIS_DIM6].srcStride +
                                            k * view.axis[AXIS_DIM7].srcStride;
                        DataCopy<DataType, NDDMA_LOOP_DIM5, config>(ubTensor[dstStride],
                                                                    inputGM_[view.addr + srcStride], params);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyInWithNddmaInvert(
        const Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
        LocalTensor<DataType>& ubTensor)
    {
        static constexpr NdDmaConfig config = {false, 0, 0, false};

        // Step 1: bundle 所有 R / A 轴 repeat, 重算每根 axis 的 dstStride
        uint32_t tailBundle = 1;  // 所有 R 轴 repeat 乘积
        uint32_t otherBundle = 1; // 所有 A 轴 repeat 乘积
        for (int32_t i = 0; i < view.axisSize; i++) {
            if (view.axis[i].isAxisA) {
                otherBundle *= static_cast<uint32_t>(view.axis[i].repeat);
            } else {
                tailBundle *= static_cast<uint32_t>(view.axis[i].repeat);
            }
        }
        // 记录真实 A/R 数据数（不含 pad）供 WelfordUpdate / WelfordUpdateGroups / Finalize 使用
        // 多维分支语义 = 轴类型乘积：ubRealABundle_ = 所有 A 轴 repeat 乘积，ubRealRBundle_ = 所有 R 轴 repeat 乘积
        // （TailA 转置后布局为 [A 行 × R 列]，恰好对应 VFWelfordParallelUpdateARWithTail 的 (ANum, realRLen)）
        ubRealABundle_ = otherBundle;
        ubRealRBundle_ = tailBundle;
        uint32_t otherAlign = invOtherAlign_;

        uint32_t invDstStride[Ops::Base::ReduceOpTmpl::MAX_DIM] = {0};
        uint32_t aSeen = 1;
        uint32_t rSeen = 1;
        for (int32_t i = 0; i < view.axisSize; i++) {
            if constexpr (InnerPattern::TailA) {
                // RA→AR: R 轴落 inner（连续），A 轴落 outer（步长 otherAlign 按 A 轴累积）
                if (view.axis[i].isAxisA) {
                    invDstStride[i] = otherAlign * aSeen;
                    aSeen *= static_cast<uint32_t>(view.axis[i].repeat);
                } else {
                    invDstStride[i] = rSeen;
                    rSeen *= static_cast<uint32_t>(view.axis[i].repeat);
                }
            } else {
                // AR→RA: A 轴落 inner（连续），R 轴落 outer（步长 otherAlign 按 R 轴累积）
                if (view.axis[i].isAxisA) {
                    invDstStride[i] = aSeen;
                    aSeen *= static_cast<uint32_t>(view.axis[i].repeat);
                } else {
                    invDstStride[i] = otherAlign * rSeen;
                    rSeen *= static_cast<uint32_t>(view.axis[i].repeat);
                }
            }
        }

        uint32_t srcStrideL1 = static_cast<uint32_t>(view.axis[1].srcStride);

        // Step 2: 计算外层维度累积（axis[2] 之后所有轴的 repeat 乘积）
        uint64_t outer = 1;
        for (int32_t i = Ops::Base::ReduceOpTmpl::CONST2; i < view.axisSize; i++) {
            outer *= view.axis[i].repeat;
        }

        if (outer == 1) {
            // Step 2: 只有 2 根轴 (axis[0], axis[1])，直接 3-layer 转置搬运
            uint32_t tailBundle = static_cast<uint32_t>(view.axis[0].repeat);
            uint32_t otherBundle = static_cast<uint32_t>(view.axis[1].repeat);
            uint32_t srcStrideL1 = static_cast<uint32_t>(view.axis[1].srcStride);
            uint32_t otherAlign = invOtherAlign_;
            if (view.axis[0].isAxisA) {
                ubRealABundle_ = tailBundle;  // axis[0] = A
                ubRealRBundle_ = otherBundle; // axis[1] = R
            } else {
                ubRealABundle_ = otherBundle; // axis[1] = A
                ubRealRBundle_ = tailBundle;  // axis[0] = R
            }
            NdDmaLoopInfo<NDDMA_LOOP_DIM3> loopInfo = {.loopSrcStride = {1, srcStrideL1, 1},
                                                       .loopDstStride = {1, 1, otherAlign},
                                                       .loopSize = {1, otherBundle, tailBundle},
                                                       .loopLpSize = {0, 0, 0},
                                                       .loopRpSize = {0, 0, 0}};
            NdDmaParams<DataType, NDDMA_LOOP_DIM3> params = {loopInfo, 0};
            DataCopy<DataType, NDDMA_LOOP_DIM3, config>(ubTensor, inputGM_[view.addr], params);
            return;
        }

        // Step 3: 多维分发
        if constexpr (InnerPattern::TailA) {
            constexpr int32_t innerAxes = (Dim >= DIM5) ? DIM4 : Dim; // NDDMA 内层轴数（+1 退化层后 <=5 层）
            uint32_t lvlSize[5] = {1, 1, 1, 1, 1};
            uint32_t lvlSrc[5] = {1, 1, 1, 1, 1};
            uint32_t lvlDst[5] = {1, 1, 1, 1, 1};
            int32_t lvl = 1;
            for (int32_t i = 0; i < innerAxes; i++) { // R 轴 → 低层（dst 连续）
                if (!view.axis[i].isAxisA) {
                    lvlSize[lvl] = static_cast<uint32_t>(view.axis[i].repeat);
                    lvlSrc[lvl] = (i == 0) ? 1 : static_cast<uint32_t>(view.axis[i].srcStride);
                    lvlDst[lvl] = invDstStride[i];
                    lvl++;
                }
            }
            for (int32_t i = 0; i < innerAxes; i++) { // A 轴 → 高层（dst 散射 otherAlign*aSeen）
                if (view.axis[i].isAxisA) {
                    lvlSize[lvl] = static_cast<uint32_t>(view.axis[i].repeat);
                    lvlSrc[lvl] = (i == 0) ? 1 : static_cast<uint32_t>(view.axis[i].srcStride);
                    lvlDst[lvl] = invDstStride[i];
                    lvl++;
                }
            }
            if constexpr (Dim == DIM3) {
                constexpr int32_t peelIdx = Pattern::FirstA ? 2 : 1; // 第 2 根 A 轴在 view 中的下标
                if (view.axis[peelIdx].isAxisA) {
                    constexpr int32_t rIdx = (peelIdx == 2) ? 1 : 2; // R 轴在 view 中的下标
                    NdDmaLoopInfo<NDDMA_LOOP_DIM3> loopInfo = {
                        .loopSrcStride = {1, static_cast<uint32_t>(view.axis[rIdx].srcStride), 1},
                        .loopDstStride = {1, 1, otherAlign},
                        .loopSize = {1, static_cast<uint32_t>(view.axis[rIdx].repeat),
                                     static_cast<uint32_t>(view.axis[0].repeat)},
                        .loopLpSize = {0, 0, 0},
                        .loopRpSize = {0, 0, 0}};
                    NdDmaParams<DataType, NDDMA_LOOP_DIM3> params = {loopInfo, 0};
                    for (uint32_t k = 0; k < static_cast<uint32_t>(view.axis[peelIdx].repeat); k++) {
                        DataCopy<DataType, NDDMA_LOOP_DIM3, config>(
                            ubTensor[k * invDstStride[peelIdx]], inputGM_[view.addr + k * view.axis[peelIdx].srcStride],
                            params);
                    }
                } else {
                    // [R,R,A]：仅 1 根 A 散射层，维持 4 层单拷贝
                    NdDmaLoopInfo<NDDMA_LOOP_DIM4> loopInfo = {
                        .loopSrcStride = {lvlSrc[0], lvlSrc[1], lvlSrc[2], lvlSrc[3]},
                        .loopDstStride = {lvlDst[0], lvlDst[1], lvlDst[2], lvlDst[3]},
                        .loopSize = {lvlSize[0], lvlSize[1], lvlSize[2], lvlSize[3]},
                        .loopLpSize = {0, 0, 0, 0},
                        .loopRpSize = {0, 0, 0, 0}};
                    NdDmaParams<DataType, NDDMA_LOOP_DIM4> params = {loopInfo, 0};
                    DataCopy<DataType, NDDMA_LOOP_DIM4, config>(ubTensor, inputGM_[view.addr], params);
                }
            } else if constexpr (Dim == DIM4) {
                NdDmaLoopInfo<NDDMA_LOOP_DIM5> loopInfo = {
                    .loopSrcStride = {lvlSrc[0], lvlSrc[1], lvlSrc[2], lvlSrc[3], lvlSrc[4]},
                    .loopDstStride = {lvlDst[0], lvlDst[1], lvlDst[2], lvlDst[3], lvlDst[4]},
                    .loopSize = {lvlSize[0], lvlSize[1], lvlSize[2], lvlSize[3], lvlSize[4]},
                    .loopLpSize = {0, 0, 0, 0, 0},
                    .loopRpSize = {0, 0, 0, 0, 0}};
                NdDmaParams<DataType, NDDMA_LOOP_DIM5> params = {loopInfo, 0};
                DataCopy<DataType, NDDMA_LOOP_DIM5, config>(ubTensor, inputGM_[view.addr], params);
            } else {
                // Dim >= 5: 内 5 层 NDDMA（退化 L0 + axis[0..3]），剩余 axis[4..7] 外层 for 循环
                NdDmaLoopInfo<NDDMA_LOOP_DIM5> loopInfo = {
                    .loopSrcStride = {lvlSrc[0], lvlSrc[1], lvlSrc[2], lvlSrc[3], lvlSrc[4]},
                    .loopDstStride = {lvlDst[0], lvlDst[1], lvlDst[2], lvlDst[3], lvlDst[4]},
                    .loopSize = {lvlSize[0], lvlSize[1], lvlSize[2], lvlSize[3], lvlSize[4]},
                    .loopLpSize = {0, 0, 0, 0, 0},
                    .loopRpSize = {0, 0, 0, 0, 0}};
                NdDmaParams<DataType, NDDMA_LOOP_DIM5> params = {loopInfo, 0};
                for (int32_t i = 0; i < view.axis[AXIS_DIM4].repeat; i++) {
                    for (int32_t j = 0; j < view.axis[AXIS_DIM5].repeat; j++) {
                        for (int32_t k = 0; k < view.axis[AXIS_DIM6].repeat; k++) {
                            for (int32_t m = 0; m < view.axis[AXIS_DIM7].repeat; m++) {
                                int64_t srcOff = i * view.axis[AXIS_DIM4].srcStride +
                                                 j * view.axis[AXIS_DIM5].srcStride +
                                                 k * view.axis[AXIS_DIM6].srcStride +
                                                 m * view.axis[AXIS_DIM7].srcStride;
                                int64_t dstOff = i * invDstStride[AXIS_DIM4] + j * invDstStride[AXIS_DIM5] +
                                                 k * invDstStride[AXIS_DIM6] + m * invDstStride[AXIS_DIM7];
                                DataCopy<DataType, NDDMA_LOOP_DIM5, config>(ubTensor[dstOff],
                                                                            inputGM_[view.addr + srcOff], params);
                            }
                        }
                    }
                }
            }
            return;
        }

        // !TailA (AR→RA): 与历史实现逐位一致 —— axis[0](R) 落 L0 按 otherAlign 散射、axis[1](A) 连续
        if constexpr (Dim == DIM3) {
            NdDmaLoopInfo<NDDMA_LOOP_DIM3> loopInfo = {
                .loopSrcStride = {1, srcStrideL1, static_cast<uint32_t>(view.axis[2].srcStride)},
                .loopDstStride = {invDstStride[0], invDstStride[1], invDstStride[2]},
                .loopSize = {static_cast<uint32_t>(view.axis[0].repeat), static_cast<uint32_t>(view.axis[1].repeat),
                             static_cast<uint32_t>(view.axis[2].repeat)},
                .loopLpSize = {0, 0, 0},
                .loopRpSize = {0, 0, 0}};
            NdDmaParams<DataType, NDDMA_LOOP_DIM3> params = {loopInfo, 0};
            DataCopy<DataType, NDDMA_LOOP_DIM3, config>(ubTensor, inputGM_[view.addr], params);
        } else if constexpr (Dim == DIM4) {
            NdDmaLoopInfo<NDDMA_LOOP_DIM4> loopInfo = {
                .loopSrcStride = {1, srcStrideL1, static_cast<uint32_t>(view.axis[2].srcStride),
                                  static_cast<uint32_t>(view.axis[3].srcStride)},
                .loopDstStride = {invDstStride[0], invDstStride[1], invDstStride[2], invDstStride[3]},
                .loopSize = {static_cast<uint32_t>(view.axis[0].repeat), static_cast<uint32_t>(view.axis[1].repeat),
                             static_cast<uint32_t>(view.axis[2].repeat), static_cast<uint32_t>(view.axis[3].repeat)},
                .loopLpSize = {0, 0, 0, 0},
                .loopRpSize = {0, 0, 0, 0}};
            NdDmaParams<DataType, NDDMA_LOOP_DIM4> params = {loopInfo, 0};
            DataCopy<DataType, NDDMA_LOOP_DIM4, config>(ubTensor, inputGM_[view.addr], params);
        } else {
            // Dim >= 5: 内 5 层 NDDMA (axis[0..4]) + 外层 axis[5..7] for 循环
            NdDmaLoopInfo<NDDMA_LOOP_DIM5> loopInfo = {
                .loopSrcStride = {1, srcStrideL1, static_cast<uint32_t>(view.axis[2].srcStride),
                                  static_cast<uint32_t>(view.axis[3].srcStride),
                                  static_cast<uint32_t>(view.axis[AXIS_DIM4].srcStride)},
                .loopDstStride = {invDstStride[0], invDstStride[1], invDstStride[2], invDstStride[3],
                                  invDstStride[AXIS_DIM4]},
                .loopSize = {static_cast<uint32_t>(view.axis[0].repeat), static_cast<uint32_t>(view.axis[1].repeat),
                             static_cast<uint32_t>(view.axis[2].repeat), static_cast<uint32_t>(view.axis[3].repeat),
                             static_cast<uint32_t>(view.axis[AXIS_DIM4].repeat)},
                .loopLpSize = {0, 0, 0, 0, 0},
                .loopRpSize = {0, 0, 0, 0, 0}};
            NdDmaParams<DataType, NDDMA_LOOP_DIM5> params = {loopInfo, 0};
            for (int32_t i = 0; i < view.axis[AXIS_DIM5].repeat; i++) {
                for (int32_t j = 0; j < view.axis[AXIS_DIM6].repeat; j++) {
                    for (int32_t k = 0; k < view.axis[AXIS_DIM7].repeat; k++) {
                        int64_t srcOff = i * view.axis[AXIS_DIM5].srcStride + j * view.axis[AXIS_DIM6].srcStride +
                                         k * view.axis[AXIS_DIM7].srcStride;
                        int64_t dstOff = i * invDstStride[AXIS_DIM5] + j * invDstStride[AXIS_DIM6] +
                                         k * invDstStride[AXIS_DIM7];
                        DataCopy<DataType, NDDMA_LOOP_DIM5, config>(ubTensor[dstOff], inputGM_[view.addr + srcOff],
                                                                    params);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyIn(const Ops::Base::ReduceOpTmpl::SliceView<Ops::Base::ReduceOpTmpl::MAX_DIM>& view,
                                  LocalTensor<DataType>& ubTensor)
    {
        if (tiling_->useNddma == 1) {
            if (tiling_->isInvert == 1) {
                CopyInWithNddmaInvert(view, ubTensor);
            } else {
                CopyInWithNddma(view, ubTensor);
            }
            return;
        }

        DataCopyPadExtParams<DataType> padParams{true, 0, 0, static_cast<DataType>(0.0)};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = view.axis[1].repeat;
        copyInParams.blockLen = view.axis[0].repeat * sizeof(DataType);
        copyInParams.srcStride = (view.axis[1].srcStride - view.axis[0].repeat) * sizeof(DataType);
        copyInParams.dstStride = (view.axis[1].dstStride - view.axis[0].repeat) * sizeof(DataType) /
                                 BLOCK_SIZE_BYTE; // unit block(32byte) "gap"
        LoopModeParams loopParams;
        loopParams.loop1Size = view.axis[2].repeat;                            // 2: the second-to-last dim
        loopParams.loop1SrcStride = view.axis[2].srcStride * sizeof(DataType); // 2: the second-to-last dim
        loopParams.loop1DstStride = view.axis[2].dstStride * sizeof(DataType); // 2: the second-to-last dim
        loopParams.loop2Size = view.axis[3].repeat;                            // 3: the third-to-last dim
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
            addrOffset += iterAddr_[i].start * tiling_->dstStride[i];
        }

        LocalTensor<DataType> outMeanTensor = outQueue_.DeQue<DataType>();
        LocalTensor<DataType> outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(DataType)];

        DataCopyExtParams copyOutParams = {1, 1, 0, 0, 0};

        if constexpr (Pattern::TailA) {
            copyOutParams.blockCount = aOutNBurst_;
            copyOutParams.blockLen = aOutBurstLen_ * sizeof(DataType);
        } else {
            if (tiling_->isInvert == 1) {
                copyOutParams.blockCount = aOutNBurst_;
                copyOutParams.blockLen = aOutBurstLen_ * sizeof(DataType);
            } else {
                copyOutParams.blockCount = 1;
                copyOutParams.blockLen = shape.value[0] * sizeof(DataType);
            }
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
        LocalTensor<PromoteDataType> outVarTensor = outMeanTensor[tiling_->resultBlock / sizeof(PromoteDataType)];

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
            if (tiling_->isInvert == 1) {
                copyOutParams.blockLen = aOutBurstLen_ * sizeof(PromoteDataType);
                copyOutParams.blockCount = aOutNBurst_;
            } else {
                copyOutParams.blockLen = shape.value[0] * sizeof(PromoteDataType);
                copyOutParams.blockCount = 1;
            }
        }
        int32_t axis = Pattern::FirstA ? 0 : 1;
        if constexpr (SchLoopInfo.loopACount > 0) {
            axis = SchLoopInfo.loopAAxis[SchLoopInfo.loopACount - 1];
        }

        uint64_t addrOffset = 0;
        if constexpr (SchLoopInfo.loopInnerACount > 0) {
            for (int32_t i = axis; i < Dim; i += Ops::Base::ReduceOpTmpl::CONST2) {
                addrOffset += iterAddr_[i].start * tiling_->dstStride[i];
            }
        }

        uint64_t aSize = Ops::Base::CeilAlign(tiling_->outSize, static_cast<uint64_t>(ELEMENT_ONE_REPEAT_COMPUTE));
        uint64_t axisStep = SchLoopInfo.loopACount > 0 ? loopAAxisStep_ : 1;
        uint64_t addr = (blockId % tiling_->groupR) * aSize +                                      // group offset
                        (blockId / (tiling_->groupR * axisStep)) * tiling_->shape[axis] * innerA + // all A Axis offset
                        (blockId / tiling_->groupR % axisStep) * ubFactorA_ * innerA + // split A Axis offset
                        addrOffset;                                                    // innerA offset

        uint64_t varOffset = static_cast<uint64_t>(tiling_->workSpaceSize) / sizeof(PromoteDataType);
        DataCopyPad(workspace_[addr], outMeanTensor, copyOutParams);
        DataCopyPad(workspace_[addr + varOffset], outVarTensor, copyOutParams);

        outQueue_.FreeTensor(outMeanTensor);
    }
};

} // namespace ReduceOpTmpl
#endif // _REDUCE_VAR_SCH_H_
