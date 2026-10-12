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
 * \file flash_attention_score_grad_block_cube.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_H
#define FLASH_ATTENTION_SCORE_GRAD_BLOCK_CUBE_H
 
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"

namespace FagBaseApi {

constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
 
template<bool IS_ROPE = false>
struct PreloadArgs;
 
template<>
struct PreloadArgs<false> {
    int64_t nextQueryOffset;
    int64_t nextDyOffset;
    int32_t nextMOrN;
    bool copyCurrent = true;
    bool copyNext = false;
};
 
template<>
struct PreloadArgs<true> {
    int64_t nextQueryOffset;
    int64_t nextDyOffset;
    int64_t nextQueryRopeOffset;
    int64_t nextKeyRopeOffset;
    int32_t nextMOrN;
    bool copyCurrent = true;
    bool copyNext = false;
};
 
/* ============确定Query的L1类型============= */
template <uint8_t IS_L1_REUSE, uint8_t IS_L1_PRELOAD>
struct QL1BuffSelector {
    using TYPE = std::conditional_t<
        IS_L1_REUSE,
        std::conditional_t<IS_L1_PRELOAD, BuffersPolicy3buff<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>,
        BuffersPolicySingleBuffer<BufferType::L1>>;
};
 
/* ============确定Key的L1类型============= */
template <uint8_t IS_L1_REUSE, uint8_t IS_L1_PRELOAD>
struct KL1BuffSelector {
    using TYPE = std::conditional_t<
        IS_L1_REUSE,
        std::conditional_t<IS_L1_PRELOAD, BuffersPolicyDB<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>,
        BuffersPolicySingleBuffer<BufferType::L1>>;
};
 
/* ============确定Dy的L1类型============= */
template <uint8_t IS_L1_REUSE, uint8_t IS_L1_PRELOAD>
struct DyL1BuffSelector {
    using TYPE = std::conditional_t<
        IS_L1_REUSE,
        std::conditional_t<IS_L1_PRELOAD, BuffersPolicy3buff<BufferType::L1>, BuffersPolicyDB<BufferType::L1>>,
        BuffersPolicySingleBuffer<BufferType::L1>>;
};
 
/* ============确定mm1 mm2 mm3共用几块L0C buffer============= */
template <uint32_t HEAD_DIM_ALIGN>
struct mm1Mm2Mm3L0CBuffSelector {
    using TYPE = std::conditional_t<
        HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned128),
        std::conditional_t<HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned64), BuffersPolicy3buff<BufferType::L0C>,
        BuffersPolicyDB<BufferType::L0C>>, BuffersPolicySingleBuffer<BufferType::L0C>>;
};
 
/* ============DKV不驻留L0C场景下开几块L0C buffer============= */
template <uint32_t HEAD_DIM_ALIGN>
struct CommonL0CBufSelector {
    using TYPE = std::conditional_t<HEAD_DIM_ALIGN <= static_cast<uint16_t>(DTemplateType::Aligned256), BuffersPolicyDB<BufferType::L0C>,
                                    BuffersPolicySingleBuffer<BufferType::L0C>>;
};
 
template<typename T, const int64_t CUBE_BASEM, const int64_t HEAD_DIM_ALIGN>
static constexpr uint32_t GET_DQ_L0_SPLIT_K() {
    if constexpr (IsSameType<T, float>::value) {
        int64_t splitD = HEAD_DIM_ALIGN > 512 ? 512 : HEAD_DIM_ALIGN;
        return (std::min(L0_SINGLE_BUFFER_SIZE / (CUBE_BASEM * sizeof(float)),
            L0_SINGLE_BUFFER_SIZE / (splitD * sizeof(float))) / C0_SIZE) * C0_SIZE;
    } else {
        return static_cast<uint16_t>(S2TemplateType::Aligned128);
    }
}
 
template<typename T, const int64_t CUBE_BASEN, const int64_t HEAD_DIM_ALIGN>
static constexpr uint32_t GET_DKV_L0_SPLIT_K() {
    if constexpr (IsSameType<T, float>::value) {
        int64_t splitD = HEAD_DIM_ALIGN > 512 ? 512 : HEAD_DIM_ALIGN;
        return (std::min(L0_SINGLE_BUFFER_SIZE / (splitD * sizeof(float)),
            L0_SINGLE_BUFFER_SIZE / (CUBE_BASEN * sizeof(float))) / C0_SIZE) * C0_SIZE;
    } else {
        return static_cast<uint16_t>(S1TemplateType::Aligned128);
    }
}
 
TEMPLATES_DEF
class FAGBlockCube {
public:
    constexpr static bool IS_FP8_INPUT =
        IsSameType<INPUT_TYPE, fp8_e5m2_t>::value || IsSameType<INPUT_TYPE, fp8_e4m3fn_t>::value || IsSameType<INPUT_TYPE, hifloat8_t>::value;
    constexpr static bool IS_FP32_INPUT = IsSameType<INPUT_TYPE, float>::value;
    constexpr static uint32_t CUBE_BASEM = (uint32_t)s1TemplateType;
    constexpr static uint32_t CUBE_BASEN = (uint32_t)s2TemplateType;
    constexpr static uint32_t HEAD_DIM_ALIGN = (uint32_t)dTemplateType;
    constexpr static uint32_t C0_SIZE = 16;
    constexpr static uint32_t l1BaseD = 512;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    constexpr static bool IS_L1_REUSE =
        GET_IS_L1_REUSE<INPUT_TYPE>(HEAD_DIM_ALIGN, IS_DETER_OLD(DETER_SPARSE_TYPE), FP8_OPEN_TSCM);
    constexpr static bool IS_L1_PRELOAD = GET_IS_L1_PRELOAD<INPUT_TYPE>(
        HEAD_DIM_ALIGN, SPLIT_AXIS, IS_DETER_OLD(DETER_SPARSE_TYPE), IS_TND, FP8_OPEN_TSCM, IS_ROPE);
    constexpr static SyncType SYNC_TYPE = IS_L1_PRELOAD ? SyncType::NO_SYNC : SyncType::INNER_CORE_SYNC;
    constexpr static bool IS_DKV_RESIDENT_L0C = IS_DKV_RESIDENT_L0C(CUBE_BASEM, CUBE_BASEN, HEAD_DIM_ALIGN) && !IS_FP8_INPUT;
    constexpr static bool IS_FP32_D_EXCEED_256 = IS_FP32_INPUT && HEAD_DIM_ALIGN > 256;
 
    constexpr static uint32_t DQ_L0_SPLIT_K = GET_DQ_L0_SPLIT_K<INPUT_TYPE, CUBE_BASEM, HEAD_DIM_ALIGN>();
    constexpr static uint32_t DKV_L0_SPLIT_K = GET_DKV_L0_SPLIT_K<INPUT_TYPE, CUBE_BASEN, HEAD_DIM_ALIGN>();
 
    // input global mmemory
    GlobalTensor<INPUT_TYPE> queryGm, keyGm, valueGm, queryRopeGm, keyRopeGm;
    GlobalTensor<OUTDTYPE> dyGm;
 
    // output global mmemory
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm;
    // GlobalTensor<INPUT_TYPE> dqGm, dkGm, dvGm;
 
    TPipe *pipe;
    FagTilingType tilingData;
    // vector scaleDs通过ssbuf传递给cube
    FagCVSharedParams sharedParams;
    // l1 buffer manage
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    typename std::conditional<IS_L1_REUSE, typename DyL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD>::TYPE,
                              std::nullptr_t>::type dYL1Buf;
    typename std::conditional<IS_L1_REUSE, BuffersPolicySingleBuffer<BufferType::L1, SYNC_TYPE>, std::nullptr_t>::type vL1Buf;
    typename std::conditional<IS_L1_REUSE, typename QL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD>::TYPE,
                              std::nullptr_t>::type qL1Buf;
    typename std::conditional<IS_L1_REUSE, typename KL1BuffSelector<IS_L1_REUSE, IS_L1_PRELOAD>::TYPE,
                              std::nullptr_t>::type kL1Buf;
    typename std::conditional<!IS_L1_REUSE, BuffersPolicyDB<BufferType::L1>, std::nullptr_t>::type commonL1Buf;
    //fp32 d>256 l1buffer 
    typename std::conditional<IS_FP32_D_EXCEED_256, BuffersPolicySingleBuffer<BufferType::L1>, std::nullptr_t>::type fp32L1Buf1;
    typename std::conditional<IS_FP32_D_EXCEED_256, BuffersPolicySingleBuffer<BufferType::L1>, std::nullptr_t>::type fp32L1Buf2;
 
    // l0ab buffer manage, double buffer
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;
 
    // l0c buffer manage
    // l0c reuse, dk dv resident
    BufferManager<BufferType::L0C> l0cBufferManager;
    using L0CType = typename mm1Mm2Mm3L0CBuffSelector<HEAD_DIM_ALIGN>::TYPE;
    typename std::conditional<IS_DKV_RESIDENT_L0C, L0CType, std::nullptr_t>::type mm1Mm2Mm3L0CBuf;
    typename std::conditional<IS_DKV_RESIDENT_L0C, BuffersPolicySingleBuffer<BufferType::L0C>, std::nullptr_t>::type
        dkL0CBuf;
    typename std::conditional<IS_DKV_RESIDENT_L0C, BuffersPolicySingleBuffer<BufferType::L0C>, std::nullptr_t>::type
        dvL0CBuf;
    // no l0c reuse, all mm res
    typename std::conditional<!IS_DKV_RESIDENT_L0C, typename CommonL0CBufSelector<HEAD_DIM_ALIGN>::TYPE, std::nullptr_t>::type
        commonl0CBuf;
    // special for dSize=192, dSizeV=128
    BuffersPolicySingleBuffer<BufferType::L0C> mm1Mm2Mm3L0CSpecialBuf;
    BuffersPolicySingleBuffer<BufferType::L0C> dkL0CSpecialBuf;
    BuffersPolicySingleBuffer<BufferType::L0C> dvL0CSpecialBuf;
    
    bool isDkvL0CResidentForD192Dv128 = false;
    MutexID vL1BufMutexId;

    __aicore__ inline FAGBlockCube(){};
    __aicore__ inline ~FAGBlockCube();
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace);
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo);
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs); // mm1
    __aicore__ inline void IterateMmDyVQuant(LocalTensor<CALC_TYPE> &mm1ResTensor, Buffer<BufferType::L1, SyncType::NO_SYNC> &vL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs);
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs); // mm2
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm3 dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm4 dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo); // mm5 dv
private:
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsKNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDyNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                              BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                              FagConstInfo &constInfo, FagRunInfo &runInfo);
};
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline FAGBlockCube<TEMPLATE_ARGS>::~FAGBlockCube()
{
    if constexpr (IS_L1_PRELOAD) {
        ReleaseMutexID(vL1BufMutexId);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                                                       BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->pipe = pipe;
    this->tilingData = tilingData;
    this->l1BufferManagerPtr = l1BuffMgr;
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                                     GM_ADDR dy, GM_ADDR queryRope, GM_ADDR keyRope,
                                                                     GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                                                     GM_ADDR workspace)
{
    queryGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)query);
    keyGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)key);
    valueGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)value);
    dyGm.SetGlobalBuffer((__gm__ OUTDTYPE *)dy);
    queryRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)queryRope);
    keyRopeGm.SetGlobalBuffer((__gm__ INPUT_TYPE *)keyRope);
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::InitCubeBuffer(FagConstInfo &constInfo)
{
    isDkvL0CResidentForD192Dv128 =
        ((SPLIT_AXIS == BN2GS1S2 || SPLIT_AXIS == BN2S2) &&
         (HEAD_DIM_ALIGN == static_cast<uint32_t>(DTemplateType::Aligned192) &&
          constInfo.commonConstInfo.dSizeV <= static_cast<uint32_t>(DTemplateType::Aligned128))) ||
        IS_ROPE;
    // init l1 buffer
    if constexpr (IS_FP8_INPUT) {
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            // FP8 support D<=128, no rope
            dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(OUTDTYPE));
            qL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
            kL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        } else {
            commonL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(OUTDTYPE));
        }
    } else {
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            if constexpr (IS_L1_PRELOAD) {vL1BufMutexId = AllocMutexID();}
            if constexpr (IS_ROPE) {
                // rope场景mm1 headDim固定为128
                dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * ROPE_D_128 * sizeof(INPUT_TYPE));
                vL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * ROPE_D_128 * sizeof(INPUT_TYPE));
            } else {
                dYL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
                vL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
            }
            qL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
            kL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                fp32L1Buf1.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));
                fp32L1Buf2.Init(*l1BufferManagerPtr, CUBE_BASEN * l1BaseD * sizeof(INPUT_TYPE));
            } else {
                commonL1Buf.Init(*l1BufferManagerPtr, CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(INPUT_TYPE));            
            }
        }
    }
 
    // init l0a l0b buffer
    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);
 
    // init l0c buffer
    l0cBufferManager.Init(pipe, L0C_MAX_SIZE); // 256 * 1024
    if constexpr (IS_DKV_RESIDENT_L0C) {
        mm1Mm2Mm3L0CBuf.Init(l0cBufferManager, CUBE_BASEN > HEAD_DIM_ALIGN ?
                                                   CUBE_BASEM * CUBE_BASEN * sizeof(float) :
                                                   CUBE_BASEM * HEAD_DIM_ALIGN * sizeof(float));
        dkL0CBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
        dvL0CBuf.Init(l0cBufferManager, CUBE_BASEN * HEAD_DIM_ALIGN * sizeof(float));
    } else {
        if (isDkvL0CResidentForD192Dv128) {
            mm1Mm2Mm3L0CSpecialBuf.Init(l0cBufferManager, CUBE_BASEM * static_cast<uint16_t>(DTemplateType::Aligned192) * sizeof(float)); // 98304
            dkL0CSpecialBuf.Init(l0cBufferManager, CUBE_BASEN * static_cast<uint16_t>(DTemplateType::Aligned192) * sizeof(float)); // 98304
            dvL0CSpecialBuf.Init(l0cBufferManager, CUBE_BASEN * static_cast<uint16_t>(DTemplateType::Aligned128) * sizeof(float)); // 65536
        } else {
            if constexpr (HEAD_DIM_ALIGN <= 256) {
                commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE / NUM_TWO);
            } else {
                commonl0CBuf.Init(l0cBufferManager, L0C_MAX_SIZE);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                 PreloadArgs<IS_ROPE> &preloadArgs)
{
    Buffer<BufferType::L1> dyL1Buffer;
    Buffer<BufferType::L1> dyL1NextBuffer;
    Buffer<BufferType::L1, SYNC_TYPE> vL1Buffer;
    Nd2NzParams nd2NzParams;
 
    // load left matrix to L1
    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            dyL1Buffer = dYL1Buf.Get();
        } else {
            dyL1Buffer = dYL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            dyL1NextBuffer = dYL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        dyL1Buffer = dYL1Buf.Get();
    } else {
        if constexpr (IS_FP32_D_EXCEED_256) {
            dyL1Buffer = fp32L1Buf1.Get();
        } else {
            dyL1Buffer = commonL1Buf.Get();           
        }
    }
    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        dyL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1Buffer.GetTensor<INPUT_TYPE>();
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
        nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
        // L1->L0A，L1上的src stride为C0对齐后的SingleM
        if constexpr (IS_FP8_INPUT) {
            nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s1RealSize);
        } else {
            nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s1RealSize);
        }
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dyL1Tensor, this->dyGm[runInfo.dyOffset], nd2NzParams);
        // when IS_L1_PRELOAD=true, only first loop need set
        dyL1Buffer.Set<HardEvent::MTE2_MTE1>();
    }
    // wait pre
    dyL1Buffer.Wait<HardEvent::MTE2_MTE1>();

    constexpr uint32_t baseN = (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) ? CUBE_BASEN / 2 : CUBE_BASEN;
    uint32_t nLoops = (runInfo.commonRunInfo.s2RealSize + baseN - 1) / baseN; // 若为FP32,且Dtemplate>512，需要切分循环两次
    // uint32_t nLoops = (CUBE_BASEN + baseN - 1) / baseN;
    uint32_t realN = baseN;//这里切N，其实就是切S2的时候，单次循环的最大S2
    for (uint32_t n = 0; n < nLoops; ++n) {
        if(n == nLoops - 1){
            uint32_t tailSize = runInfo.commonRunInfo.s2RealSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        uint32_t gmNOffset = n * runInfo.vGmS2SplitOffset;
        uint32_t ubOffset = n * baseN;
        // load right matrix to L1
        bool isCopyRight = true;
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            isCopyRight = !runInfo.isS2IdxNoChange;
            vL1Buffer = vL1Buf.Get();
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                vL1Buffer = fp32L1Buf2.Get();
            } else {
                vL1Buffer = commonL1Buf.Get();           
            }
        }

        vL1Buffer.template Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        if constexpr (IS_L1_PRELOAD) {
            AscendC::Mutex::Lock<PIPE_MTE2>(vL1BufMutexId);
        }

        if (isCopyRight) {
            LocalTensor<INPUT_TYPE> vL1Tensor = vL1Buffer.template GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = realN;
            nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Kb;
            if constexpr (IS_FP8_INPUT) {
                nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s2RealSize);
            } else {
                nd2NzParams.dstNzC0Stride = AlignTo16(realN); //64
            }
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(vL1Tensor, this->valueGm[runInfo.commonRunInfo.valueOffset + gmNOffset], nd2NzParams);
            vL1Buffer.template Set<HardEvent::MTE2_MTE1>();
            vL1Buffer.template Wait<HardEvent::MTE2_MTE1>();
        }
        if constexpr (IS_L1_PRELOAD) {
            AscendC::Mutex::Unlock<PIPE_MTE2>(vL1BufMutexId);
        }

        // load next left matrix to L1
        if constexpr(IS_L1_PRELOAD) {
            if (preloadArgs.copyNext) {
                dyL1NextBuffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
                LocalTensor<INPUT_TYPE> dyL1Tensor = dyL1NextBuffer.GetTensor<INPUT_TYPE>();
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = preloadArgs.nextMOrN;
                nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
                if constexpr (IS_FP8_INPUT) {
                    nd2NzParams.dstNzC0Stride = AlignTo32(preloadArgs.nextMOrN);
                } else {
                    // L1->L0A，L1上的src stride为C0对齐后的SingleM
                    nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);
                }

                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(dyL1Tensor, this->dyGm[preloadArgs.nextDyOffset], nd2NzParams);
                // current loop no matched wait, will wait in next loop
                dyL1NextBuffer.Set<HardEvent::MTE2_MTE1>();
            }
        }

        Buffer<BufferType::L0C> mm1L0CBuffer;
        if constexpr (IS_DKV_RESIDENT_L0C) {
            mm1L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
        } else {
            if (isDkvL0CResidentForD192Dv128) {
                mm1L0CBuffer = mm1Mm2Mm3L0CSpecialBuf.Get();
            } else {
                mm1L0CBuffer = commonl0CBuf.Get();
            }
        }
        // load l1 to l0ab + mmad
        mm1L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
        MMParam param = {
            (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
            (uint32_t)realN, // singleN
            // (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleN
            (uint32_t)constInfo.commonConstInfo.dSizeV, // singleK
            false,                                      // isLeftTranspose
            true,                                       // isRightTranspose
            true,
            true,
            UNITFLAG_ENABLE
        };

        if constexpr (IS_L1_PRELOAD) {
            AscendC::Mutex::Lock<PIPE_MTE1>(vL1BufMutexId);
        }
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
                    L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
            dyL1Buffer.GetTensor<INPUT_TYPE>(), vL1Buffer.template GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);

        if (!IS_L1_REUSE && !IS_L1_PRELOAD && n == nLoops - 1) {
            dyL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        vL1Buffer.template Set<HardEvent::MTE1_MTE2>(); // 反向同步
        if constexpr (IS_L1_PRELOAD) {
            AscendC::Mutex::Unlock<PIPE_MTE1>(vL1BufMutexId);
        }

        mm1L0CBuffer.Set<HardEvent::M_FIX>();
        mm1L0CBuffer.Wait<HardEvent::M_FIX>();

        // fixp2ub
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = CUBE_BASEN;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor[ubOffset], mm1L0CBuffer.GetTensor<CALC_TYPE>(),
                                                            fixpipeParams); // 将matmul结果从L0C搬运到UB
        mm1L0CBuffer.Set<HardEvent::FIX_M>();                               // 反向同步
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDyVQuant(LocalTensor<CALC_TYPE> &mm1ResTensor,
                                                                 Buffer<BufferType::L1, SyncType::NO_SYNC> &vL1Buffer,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                 PreloadArgs<IS_ROPE> &preloadArgs)
{
    Buffer<BufferType::L1> dyL1Buffer;
    Buffer<BufferType::L1> dyL1NextBuffer;
    Nd2NzParams nd2NzParams;
 
    // load left matrix to L1
    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            dyL1Buffer = dYL1Buf.Get();
        } else {
            dyL1Buffer = dYL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            dyL1NextBuffer = dYL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        dyL1Buffer = dYL1Buf.Get();
    } else {
        if constexpr (IS_FP32_D_EXCEED_256) {
            dyL1Buffer = fp32L1Buf1.Get();
        } else {
            dyL1Buffer = commonL1Buf.Get();           
        }
    }
    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        dyL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        LocalTensor<OUTDTYPE> dyL1Tensor = dyL1Buffer.GetTensor<OUTDTYPE>();    // dy在L1上由INPUT_DTYPE修改为OUTDTYPE
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
        nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
        // L1->L0A，L1上的src stride为C0对齐后的SingleM
        nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s1RealSize);
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
        DataCopy(dyL1Tensor, this->dyGm[runInfo.dyOffset], nd2NzParams);
        // when IS_L1_PRELOAD=true, only first loop need set
        dyL1Buffer.Set<HardEvent::MTE2_MTE1>();
    }
    // wait pre
    dyL1Buffer.Wait<HardEvent::MTE2_MTE1>();

    constexpr uint32_t baseN = (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) ? CUBE_BASEN / 2 : CUBE_BASEN;
    uint32_t nLoops = (runInfo.commonRunInfo.s2RealSize + baseN - 1) / baseN; // 若为FP32,且Dtemplate>512，需要切分循环两次
    uint32_t realN = baseN;//这里切N，其实就是切S2的时候，单次循环的最大S2
    for (uint32_t n = 0; n < nLoops; ++n) {
        if(n == nLoops - 1){
            uint32_t tailSize = runInfo.commonRunInfo.s2RealSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        uint32_t gmNOffset = n * runInfo.vGmS2SplitOffset;
        uint32_t ubOffset = n * baseN ;
        // load right matrix to L1
        bool isCopyRight = true;
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            isCopyRight = !runInfo.isS2IdxNoChange;
        }

        // load next left matrix to L1
        if constexpr(IS_L1_PRELOAD) {
            if (preloadArgs.copyNext) {
                dyL1NextBuffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
                LocalTensor<OUTDTYPE> dyL1Tensor = dyL1NextBuffer.GetTensor<OUTDTYPE>();
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = preloadArgs.nextMOrN;
                nd2NzParams.dValue = constInfo.commonConstInfo.dSizeV;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
                nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);

                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(dyL1Tensor, this->dyGm[preloadArgs.nextDyOffset], nd2NzParams);
                // current loop no matched wait, will wait in next loop
                dyL1NextBuffer.Set<HardEvent::MTE2_MTE1>();
            }
        }

        Buffer<BufferType::L0C> mm1L0CBuffer;
        if constexpr (IS_DKV_RESIDENT_L0C) {
            mm1L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
        } else {
            if (isDkvL0CResidentForD192Dv128) {
                mm1L0CBuffer = mm1Mm2Mm3L0CSpecialBuf.Get();
            } else {
                mm1L0CBuffer = commonl0CBuf.Get();
            }
        }
        // load l1 to l0ab + mmad
        mm1L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
        MMParam param = {
            (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
            (uint32_t)realN, // singleN
            // (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleN
            (uint32_t)constInfo.commonConstInfo.dSizeV, // singleK
            false,                                      // isLeftTranspose
            true,                                       // isRightTranspose
            true,
            true,
            UNITFLAG_ENABLE
        };

        MatmulBase<OUTDTYPE, OUTDTYPE, CALC_TYPE, CUBE_BASEM, baseN,
                    L0_SINGLE_BUFFER_SIZE / baseN / sizeof(OUTDTYPE), ABLayout::MK, ABLayout::KN>(
            dyL1Buffer.GetTensor<OUTDTYPE>(), vL1Buffer.template GetTensor<OUTDTYPE>(), l0aBuf, l0bBuf,
            mm1L0CBuffer.GetTensor<CALC_TYPE>(), param);

        if (!IS_L1_REUSE && !IS_L1_PRELOAD && n == nLoops - 1) {
            dyL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        mm1L0CBuffer.Set<HardEvent::M_FIX>();
        mm1L0CBuffer.Wait<HardEvent::M_FIX>();

        // fixp2ub
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        fixpipeParams.nSize = realN;
        fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = CUBE_BASEN;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm1ResTensor[ubOffset], mm1L0CBuffer.GetTensor<CALC_TYPE>(),
                                                            fixpipeParams); // 将matmul结果从L0C搬运到UB
        mm1L0CBuffer.Set<HardEvent::FIX_M>();                              // 反向同步
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor,
                                                                FagConstInfo &constInfo, FagRunInfo &runInfo,
                                                                PreloadArgs<IS_ROPE> &preloadArgs)
{
    Buffer<BufferType::L1> qL1Buffer;
    Buffer<BufferType::L1> qL1NextBuffer;
    Buffer<BufferType::L1> kL1Buffer;
    Nd2NzParams nd2NzParams;
 
    // load left matrix to L1
    if constexpr (IS_L1_PRELOAD) {
        if (preloadArgs.copyCurrent) {
            qL1Buffer = qL1Buf.Get();
        } else {
            qL1Buffer = qL1Buf.GetPre();
        }
        if (preloadArgs.copyNext) {
            qL1NextBuffer = qL1Buf.Get();
        }
    } else if constexpr (IS_L1_REUSE) {
        qL1Buffer = qL1Buf.Get();
    } else {
        if constexpr (IS_FP32_D_EXCEED_256) {
            qL1Buffer = fp32L1Buf1.Get();
        } else {
            qL1Buffer = commonL1Buf.Get();           
        }
    }
 
    // copy current, when IS_L1_PRELOAD=true, only first loop copy current
    if (!IS_L1_PRELOAD || preloadArgs.copyCurrent) {
        qL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
        LocalTensor<INPUT_TYPE> qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
        if constexpr (IS_ROPE) {
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = ROPE_D_128;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            if constexpr (IS_FP8_INPUT) {
                nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s1RealSize);
            } else {
                // L1->L0A，L1上的src stride为C0对齐后的SingleM
                nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s1RealSize);
            }
 
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.queryOffsetWithRopeForMm12], nd2NzParams);
            nd2NzParams.dValue = ROPE_D_64;
            nd2NzParams.srcDValue = constInfo.mm2Ka / NUM_TWO;
            DataCopy(qL1Tensor[nd2NzParams.dstNzC0Stride * ROPE_D_128], this->queryRopeGm[runInfo.commonRunInfo.qRopeOffset], nd2NzParams);
        } else {
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            if constexpr (IS_FP8_INPUT) {
                nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s1RealSize); 
            } else {
                // L1->L0A，L1上的src stride为C0对齐后的SingleM
                nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s1RealSize);
            }
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset], nd2NzParams);
        }
        qL1Buffer.Set<HardEvent::MTE2_MTE1>();
    }
    
    qL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    constexpr uint32_t baseN = (IS_FP32_INPUT && HEAD_DIM_ALIGN > 512) ? CUBE_BASEN / 2 : CUBE_BASEN;
    uint32_t nLoops = (runInfo.commonRunInfo.s2RealSize + baseN - 1) / baseN; // 若为FP32，需要切分循环两次
    // uint32_t nLoops = (CUBE_BASEN + baseN - 1) / baseN; // 若为FP32，需要切分循环两次
    uint32_t realN = baseN;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if(n == nLoops - 1){
            uint32_t tailSize = runInfo.commonRunInfo.s2RealSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        uint32_t gmNOffset = n * runInfo.kGmS2SplitOffset;
        uint32_t ubOffset = n * baseN ;
        // load right matrix to L1
        bool isCopyRight = true;
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            isCopyRight = !runInfo.isS2IdxNoChange;
            if (isCopyRight) {
                kL1Buffer = kL1Buf.Get();
            } else {
                kL1Buffer = kL1Buf.GetPre();
            }
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                kL1Buffer = fp32L1Buf2.Get();
            } else {
                kL1Buffer = commonL1Buf.Get();           
            }
        }
    
        if (isCopyRight) {
            kL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            LocalTensor<INPUT_TYPE> kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>();
            if constexpr (IS_ROPE) {
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
                nd2NzParams.dValue = ROPE_D_128;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.mm2Kb;
                nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s2RealSize);
                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(kL1Tensor, this->keyGm[runInfo.keyOffsetWithRopeForMm12], nd2NzParams);
                nd2NzParams.dValue = ROPE_D_64;
                nd2NzParams.srcDValue = constInfo.mm2Kb / NUM_TWO;
                DataCopy(kL1Tensor[nd2NzParams.dstNzC0Stride * ROPE_D_128], this->keyRopeGm[runInfo.commonRunInfo.kRopeOffset], nd2NzParams);
            } else {
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue =  realN; 
                nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.mm2Kb;
                if constexpr (IS_FP8_INPUT) {
                    // nd2NzParams.dstNzC0Stride = (runInfo.commonRunInfo.s2RealSize + 32 - 1) >> 5 << 5;
                    nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s2RealSize);
                } else {
                    nd2NzParams.dstNzC0Stride = AlignTo16(realN);
                }
                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(kL1Tensor, this->keyGm[runInfo.commonRunInfo.keyOffset + gmNOffset], nd2NzParams);
            }
            kL1Buffer.Set<HardEvent::MTE2_MTE1>();
            kL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }
    
        // load next left matrix to L1
        if constexpr(IS_L1_PRELOAD) {
            if (preloadArgs.copyNext) {
                qL1NextBuffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
                LocalTensor<INPUT_TYPE> qL1Tensor = qL1NextBuffer.GetTensor<INPUT_TYPE>();
                nd2NzParams.ndNum = 1;
                nd2NzParams.nValue = preloadArgs.nextMOrN;
                if constexpr (IS_ROPE) {
                    nd2NzParams.dValue = ROPE_D_128;
                } else {
                    nd2NzParams.dValue = constInfo.commonConstInfo.dSize;
                }
                nd2NzParams.srcNdMatrixStride = 0;
                nd2NzParams.srcDValue = constInfo.mm2Ka;
                if constexpr (IS_FP8_INPUT) {
                    nd2NzParams.dstNzC0Stride = AlignTo32(preloadArgs.nextMOrN);
                } else {
                    // L1->L0A，L1上的src stride为C0对齐后的SingleM
                    nd2NzParams.dstNzC0Stride = AlignTo16(preloadArgs.nextMOrN);
                }
                nd2NzParams.dstNzNStride = 1;
                nd2NzParams.dstNzMatrixStride = 0;
                DataCopy(qL1Tensor, this->queryGm[preloadArgs.nextQueryOffset], nd2NzParams);
                if constexpr (IS_ROPE) {
                    nd2NzParams.dValue = ROPE_D_64;
                    nd2NzParams.srcDValue = constInfo.mm2Ka / NUM_TWO;
                    DataCopy(qL1Tensor[nd2NzParams.dstNzC0Stride * ROPE_D_128], this->queryRopeGm[preloadArgs.nextQueryRopeOffset], nd2NzParams);
                }
                // current loop no matched wait, will wait in next loop
                qL1NextBuffer.Set<HardEvent::MTE2_MTE1>();
            }
        }
    
        Buffer<BufferType::L0C> mm2L0CBuffer;
        if constexpr (IS_DKV_RESIDENT_L0C) {
            mm2L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
        } else {
            if (isDkvL0CResidentForD192Dv128) {
                mm2L0CBuffer = mm1Mm2Mm3L0CSpecialBuf.Get();
            } else {
                mm2L0CBuffer = commonl0CBuf.Get();
            }
        }
        // load l1 to l0ab + mmad
        mm2L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
        MMParam param = {
            (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
            (uint32_t)realN, // singleN
            (uint32_t)constInfo.commonConstInfo.dSize,  // singleK
            false,                                      // isLeftTranspose
            true,                                       // isRightTranspose
            true,
            true,
            UNITFLAG_ENABLE
        };
        MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, baseN,
                L0_SINGLE_BUFFER_SIZE / baseN / sizeof(INPUT_TYPE), ABLayout::MK, ABLayout::KN>(
            qL1Buffer.GetTensor<INPUT_TYPE>(), kL1Buffer.GetTensor<INPUT_TYPE>(), l0aBuf, l0bBuf,
            mm2L0CBuffer.GetTensor<CALC_TYPE>(), param);
    
        if  (!IS_L1_REUSE && !IS_L1_PRELOAD && n == nLoops - 1) {  // qL1不需要二次datacopy
            qL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        if  (!IS_L1_REUSE && !IS_L1_PRELOAD) {  // qL1不需要二次datacopy
            kL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
        mm2L0CBuffer.Set<HardEvent::M_FIX>();
        mm2L0CBuffer.Wait<HardEvent::M_FIX>();
    
        // fixp2ub
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
        fixpipeParams.nSize = realN;
        // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
        fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
        // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
        // 源Nz矩阵中相邻大Z排布的起始地址偏移
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        // mmResUb上两行之间的间隔，单位：element。
        fixpipeParams.dstStride = CUBE_BASEN;
        // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<CALC_TYPE, CALC_TYPE, PFA_CFG_ROW_MAJOR_UB>(mm2ResTensor[ubOffset], mm2L0CBuffer.GetTensor<CALC_TYPE>(),
                                                            fixpipeParams); // 将matmul结果从L0C搬运到UB
        mm2L0CBuffer.Set<HardEvent::FIX_M>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void
FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsKNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
 
    constexpr uint32_t baseN = l1BaseD;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSize + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    // 从ssbuf中读取数据
    if constexpr (IS_FP8_INPUT) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
    }
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        Buffer<BufferType::L1> kL1Buffer;
        LocalTensor<INPUT_TYPE> kL1Tensor;
        uint64_t gmNOffset = n * baseN;
        Nd2NzParams nd2NzParams;
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            kL1Buffer = kL1Buf.GetReused(runInfo.isNextS2IdxNoChange);
            kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>();
        } else {
            // load right matrix to L1
            if constexpr (IS_FP32_D_EXCEED_256) {
                kL1Buffer = fp32L1Buf2.Get();
            } else {
                kL1Buffer = commonL1Buf.Get();           
            }
            kL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            kL1Tensor = kL1Buffer.GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s2RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Kb;
            if constexpr (IS_FP8_INPUT) {
                nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s2RealSize);
            } else {
                nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s2RealSize);
            }
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(kL1Tensor, this->keyGm[runInfo.commonRunInfo.keyOffset + gmNOffset], nd2NzParams);
            kL1Buffer.Set<HardEvent::MTE2_MTE1>();
            kL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }
 
        Buffer<BufferType::L0C> mm3L0CBuffer;
        if constexpr (IS_DKV_RESIDENT_L0C) {
            mm3L0CBuffer = mm1Mm2Mm3L0CBuf.Get();
        } else {
            if (isDkvL0CResidentForD192Dv128) {
                mm3L0CBuffer = mm1Mm2Mm3L0CSpecialBuf.Get();
            } else {
                mm3L0CBuffer = commonl0CBuf.Get();
            }
        }
        // load l1 to l0ab + mmad
        mm3L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
        MMParam param = {
            (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleM
            realN,  // singleN
            (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleK
            false,                                      // isLeftTranspose
            false,                                      // isRightTranspose
            true,
            true,
            UNITFLAG_ENABLE
        };
        if constexpr (IS_FP8_INPUT) {
            MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, DQ_L0_SPLIT_K, CUBE_BASEN, ABLayout::MK, ABLayout::KN>(
                dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf, mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DQ_L0_SPLIT_K, ABLayout::MK, ABLayout::KN>(	
                dSL1Buffer.GetTensor<INPUT_TYPE>(), kL1Tensor, l0aBuf, l0bBuf, mm3L0CBuffer.GetTensor<CALC_TYPE>(), param);
        }
        bool isCopyRight = true;
        if constexpr (IS_L1_REUSE || IS_L1_PRELOAD) {
            isCopyRight = !runInfo.isNextS2IdxNoChange;
        }
        if (isCopyRight) {
            kL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步
        }
 
        mm3L0CBuffer.Set<HardEvent::M_FIX>();
        mm3L0CBuffer.Wait<HardEvent::M_FIX>();
 
        if constexpr (IS_WRITE_UB) {
            // fixp2ub
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
            // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
            fixpipeParams.mSize = (runInfo.commonRunInfo.s1RealSize + 1) >> 1 << 1;
            // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
            // 源Nz矩阵中相邻大Z排布的起始地址偏移
            fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
            // mmResUb上两行之间的间隔，单位：element。
            fixpipeParams.dstStride = AlignTo16(constInfo.commonConstInfo.dSize);
            // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
            fixpipeParams.dualDstCtl = 1;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
            Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[gmNOffset], mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
        } else {
            // fixp2GM
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = realN;
            // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
            fixpipeParams.mSize = runInfo.commonRunInfo.s1RealSize;
            // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T)
            // 源Nz矩阵中相邻大Z排布的起始地址偏移
            fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
            // mmResUb上两行之间的间隔，单位：element。
            fixpipeParams.dstStride = SPLIT_AXIS == BN2 ? AlignTo16(constInfo.commonConstInfo.dSize) : constInfo.mm3Ka;
            // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
            fixpipeParams.dualDstCtl = 0;
            if constexpr (IS_FP8_INPUT) {
                auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
                auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
                #pragma unroll
                for (int i = 0; i < sizeof(FagCVSharedParams) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
                    *tempTiling = *tempTilingSSbuf;
                }
                fixpipeParams.quantPre = QuantMode_t::QF322F32_PRE;
                float tmp = runInfo.quantScaleInfo.deqScaleKValue / sharedParams.qScaleDs;
                uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
                fixpipeParams.deqScalar = ans;
            }
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            constexpr static FixpipeConfig DQ_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
            bool needAtomic = (SPLIT_AXIS != BN2) || (IS_BN2_MULTIBLK && !runInfo.isFirstS1Outer);
            if (needAtomic) {
                SetAtomicAdd<CALC_TYPE>();
            }
            if constexpr (IS_ROPE) {
                Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[runInfo.queryOffsetWithRope],
                                                        mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            } else {
                int64_t offset = SPLIT_AXIS == BN2 ? GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN : runInfo.commonRunInfo.queryOffset;
                if constexpr (IS_BN2_MULTIBLK) {
                    offset = GetBlockIdx() * (AlignTo128(constInfo.commonConstInfo.s1Size) * HEAD_DIM_ALIGN);
                    offset += runInfo.commonRunInfo.s1oIdx * CUBE_BASEM * HEAD_DIM_ALIGN;
                }
                Fixpipe<T, CALC_TYPE, DQ_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset],
                                                        mm3L0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            }
            if (needAtomic) {
                SetAtomicNone();
            }
        }
        mm3L0CBuffer.Set<HardEvent::FIX_M>();
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void
FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> dSL1Buffer = dSL1Buf.Get();
    constexpr uint32_t baseN = l1BaseD;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSize + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSize % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        Buffer<BufferType::L1> qL1Buffer;
        LocalTensor<INPUT_TYPE> qL1Tensor;
        uint64_t gmNOffset = n * baseN;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            qL1Buffer = qL1Buf.GetReused();
            qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                qL1Buffer = fp32L1Buf1.Get();
            } else {
                qL1Buffer = commonL1Buf.Get();           
            }
            qL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            qL1Tensor = qL1Buffer.GetTensor<INPUT_TYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.mm2Ka;
            if constexpr (IS_FP8_INPUT) {
                nd2NzParams.dstNzC0Stride = AlignTo32(runInfo.commonRunInfo.s1RealSize);
            } else {
                nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s1RealSize);
            }
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(qL1Tensor, this->queryGm[runInfo.commonRunInfo.queryOffset + gmNOffset], nd2NzParams);
            qL1Buffer.Set<HardEvent::MTE2_MTE1>();
            qL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }
 
        Buffer<BufferType::L0C> dkL0CBuffer;
        if constexpr (IS_DKV_RESIDENT_L0C) {
            dkL0CBuffer = dkL0CBuf.Get();
        } else {
            if (isDkvL0CResidentForD192Dv128) {
                dkL0CBuffer = dkL0CSpecialBuf.Get();
            } else {
                dkL0CBuffer = commonl0CBuf.Get();
            }
        }
        // load l1 to l0ab + mmad
        dkL0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
        bool enPartialSum = false; // control l0c accumulate sum
        bool isFixpOut = true;
        if ((IS_DKV_RESIDENT_L0C || isDkvL0CResidentForD192Dv128) && !IS_FP8_INPUT && !(SPLIT_AXIS == BN2)) {
            enPartialSum = runInfo.isS2IdxNoChange;
            isFixpOut = !runInfo.isNextS2IdxNoChange;
        }
        if constexpr (IS_BN2_MULTIBLK && IS_DKV_RESIDENT_L0C) {
            enPartialSum = runInfo.isS2IdxNoChange;
            isFixpOut = !runInfo.isNextS2IdxNoChange;
        }
        MMParam param = {
            (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
            realN,  // singleN
            (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleK
            true,                                       // isLeftTranspose
            false,                                       // isRightTranspose
            true,
            !enPartialSum,
            !IS_DKV_RESIDENT_L0C && !isDkvL0CResidentForD192Dv128 && HEAD_DIM_ALIGN <= 512 ? UNITFLAG_ENABLE : UNITFLAG_DISABLE
        };
        if constexpr (IS_FP8_INPUT) {
            MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEN, DKV_L0_SPLIT_K, CUBE_BASEM, ABLayout::MK, ABLayout::KN>(
                dSL1Buffer.GetTensor<INPUT_TYPE>(), qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK, ABLayout::KN>(	
                dSL1Buffer.GetTensor<INPUT_TYPE>(), qL1Tensor, l0aBuf, l0bBuf, dkL0CBuffer.GetTensor<CALC_TYPE>(), param);
        }
        qL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步

        dkL0CBuffer.Set<HardEvent::M_FIX>();
        dkL0CBuffer.Wait<HardEvent::M_FIX>();
 
        if constexpr (IS_WRITE_UB) {
            // fixp2ub
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
            fixpipeParams.nSize = (realN + 7) >> 3 << 3;
            fixpipeParams.mSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1 << 1;
            fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
            fixpipeParams.dstStride = AlignTo16(constInfo.commonConstInfo.dSize);
            fixpipeParams.dualDstCtl = 1;
            fixpipeParams.params.ndNum = 1;
            fixpipeParams.params.srcNdStride = 0;
            fixpipeParams.params.dstNdStride = 0;
            constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
            if (isFixpOut) {
                Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[gmNOffset], dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            }
        } else {
            // fixp2gm
            if (isFixpOut) {
                bool needAtomic = SPLIT_AXIS == BN2GS1S2 ||
                                  (!IS_DKV_RESIDENT_L0C && !isDkvL0CResidentForD192Dv128 && runInfo.isS2IdxNoChange) ||
                                  (SPLIT_AXIS == BN2S2 && !runInfo.isFirstBlock && (runInfo.specialS2Index != -1));
                if constexpr (IS_BN2_MULTIBLK) {
                    needAtomic = (!IS_DKV_RESIDENT_L0C && runInfo.isS2IdxNoChange);
                }
                FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
                fixpipeParams.nSize = realN;
                fixpipeParams.mSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1 << 1;
                fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
                fixpipeParams.dstStride = SPLIT_AXIS != BN2GS1S2 ? AlignTo16(constInfo.commonConstInfo.dSize) : constInfo.mm4Kb;
                fixpipeParams.dualDstCtl = 0;
                if constexpr (IS_FP8_INPUT) {
                    fixpipeParams.quantPre = QuantMode_t::QF322F32_PRE;
                    float tmp = runInfo.quantScaleInfo.deqScaleQValue / sharedParams.qScaleDs;
                    uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
                    fixpipeParams.deqScalar = ans;
                }
                fixpipeParams.params.ndNum = 1;
                fixpipeParams.params.srcNdStride = 0;
                fixpipeParams.params.dstNdStride = 0;
                constexpr static FixpipeConfig DK_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
                if (needAtomic) {
                    SetAtomicAdd<CALC_TYPE>();
                }
                if constexpr (IS_ROPE) {
                    Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[runInfo.keyOffsetWithRope],
                                                            dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                } else {
                    int64_t offset = 0;
                    if constexpr (SPLIT_AXIS == BN2S2) {
                        if (runInfo.specialS2Index != -1) {
                            offset = runInfo.specialS2Index * CUBE_BASEM * HEAD_DIM_ALIGN * NUM_TWO;
                        } else {
                            offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                        }
                    } else if constexpr (SPLIT_AXIS == BN2) {
                        offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                    } else {
                        offset = runInfo.commonRunInfo.keyOffset;
                    }

                    Fixpipe<T, CALC_TYPE, DK_FIXPIPE_CONFIG>(outTensor[offset + gmNOffset],
                                                            dkL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                }
                if (needAtomic) {
                    SetAtomicNone();
                }
            }
        }
        dkL0CBuffer.Set<HardEvent::FIX_M>();
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void
FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDyNormal(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                                FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    Buffer<BufferType::L1, SyncType::NO_SYNC> pL1Buffer = pL1Buf.Get();
 
    constexpr uint32_t baseN = l1BaseD;
    uint32_t nLoops = ((uint32_t)constInfo.commonConstInfo.dSizeV + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.commonConstInfo.dSizeV % baseN;
            realN = tailSize ? tailSize : baseN;
        }    
        Buffer<BufferType::L1> dYL1Buffer;
        LocalTensor<OUTDTYPE> dYL1Tensor;
        uint64_t gmNOffset = n * baseN;
        Nd2NzParams nd2NzParams;
        // load right matrix to L1
        if constexpr (IS_L1_PRELOAD || IS_L1_REUSE) {
            dYL1Buffer = dYL1Buf.GetReused();
            dYL1Tensor = dYL1Buffer.GetTensor<OUTDTYPE>();
        } else {
            if constexpr (IS_FP32_D_EXCEED_256) {
                dYL1Buffer = fp32L1Buf2.Get();
            } else {
                dYL1Buffer = commonL1Buf.Get();           
            }
            dYL1Buffer.Wait<HardEvent::MTE1_MTE2>(); // 反向同步
            dYL1Tensor = dYL1Buffer.GetTensor<OUTDTYPE>();
            nd2NzParams.ndNum = 1;
            nd2NzParams.nValue = runInfo.commonRunInfo.s1RealSize;
            nd2NzParams.dValue = realN;
            nd2NzParams.srcNdMatrixStride = 0;
            nd2NzParams.srcDValue = constInfo.commonConstInfo.mm1Ka;
            nd2NzParams.dstNzC0Stride = AlignTo16(runInfo.commonRunInfo.s1RealSize);
            nd2NzParams.dstNzNStride = 1;
            nd2NzParams.dstNzMatrixStride = 0;
            DataCopy(dYL1Tensor, this->dyGm[runInfo.dyOffset + gmNOffset], nd2NzParams);
            dYL1Buffer.Set<HardEvent::MTE2_MTE1>();
            dYL1Buffer.Wait<HardEvent::MTE2_MTE1>();
        }
 
        Buffer<BufferType::L0C> dvL0CBuffer;
        if constexpr (IS_DKV_RESIDENT_L0C) {
            dvL0CBuffer = dvL0CBuf.Get();
        } else {
            if (isDkvL0CResidentForD192Dv128) {
                dvL0CBuffer = dvL0CSpecialBuf.Get();
            } else  {
                dvL0CBuffer = commonl0CBuf.Get();
            }
        }
        // load l1 to l0ab + mmad
        dvL0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
        bool enPartialSum = false; // control l0c accumulate sum
        bool isFixpOut = true;
        if ((IS_DKV_RESIDENT_L0C || isDkvL0CResidentForD192Dv128) && !IS_FP8_INPUT && !(SPLIT_AXIS == BN2)) {
            enPartialSum = runInfo.isS2IdxNoChange;
            isFixpOut = !runInfo.isNextS2IdxNoChange;
        }
        if constexpr (IS_BN2_MULTIBLK && IS_DKV_RESIDENT_L0C) {
            enPartialSum = runInfo.isS2IdxNoChange;
            isFixpOut = !runInfo.isNextS2IdxNoChange;
        }
        MMParam param = {
            (uint32_t)runInfo.commonRunInfo.s2RealSize, // singleM
            realN,                                      // singleN
            (uint32_t)runInfo.commonRunInfo.s1RealSize, // singleK
            true,                                       // isLeftTranspose
            false,                                      // isRightTranspose
            true,
            !enPartialSum,
            !IS_DKV_RESIDENT_L0C && !isDkvL0CResidentForD192Dv128 && HEAD_DIM_ALIGN <= 512 ? UNITFLAG_ENABLE : UNITFLAG_DISABLE
        };
        if constexpr (IS_FP8_INPUT) {
            MatmulBase<OUTDTYPE, OUTDTYPE, CALC_TYPE, CUBE_BASEN, DKV_L0_SPLIT_K, CUBE_BASEM, ABLayout::MK, ABLayout::KN>(
                pL1Buffer.GetTensor<OUTDTYPE>(), dYL1Tensor, l0aBuf, l0bBuf, dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        } else {
            MatmulBase<INPUT_TYPE, INPUT_TYPE, CALC_TYPE, CUBE_BASEM, CUBE_BASEN, DKV_L0_SPLIT_K, ABLayout::MK, ABLayout::KN>(	
                pL1Buffer.GetTensor<INPUT_TYPE>(), dYL1Tensor, l0aBuf, l0bBuf, dvL0CBuffer.GetTensor<CALC_TYPE>(), param);
        }
        dYL1Buffer.Set<HardEvent::MTE1_MTE2>(); // 反向同步

        dvL0CBuffer.Set<HardEvent::M_FIX>();
        dvL0CBuffer.Wait<HardEvent::M_FIX>();
 
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
        if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
            fixpipeParams.mSize = runInfo.commonRunInfo.s2RealSize;
            fixpipeParams.nSize = constInfo.commonConstInfo.dSizeV;
        } else {
            fixpipeParams.mSize = (runInfo.commonRunInfo.s2RealSize + 1) >> 1 << 1;
            fixpipeParams.nSize = realN;
        }    
        fixpipeParams.srcStride = AlignTo16(fixpipeParams.mSize);
        fixpipeParams.dstStride = (SPLIT_AXIS == BN2S2 || IS_BN2_MULTIBLK) ? AlignTo16(constInfo.commonConstInfo.dSizeV) : constInfo.commonConstInfo.mm1Kb;
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        constexpr static FixpipeConfig DV_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, IS_WRITE_UB};
        if constexpr (!IS_WRITE_UB) {
            // S1S2模板与BN2模板的dv均会输出到gm
            if constexpr (SPLIT_AXIS == BN2 && !IS_BN2_MULTIBLK) {
                if constexpr (IsSameType<T, half>::value) {
                    fixpipeParams.quantPre = QuantMode_t::F322F16;
                } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                    fixpipeParams.quantPre = QuantMode_t::F322BF16;
                }
                Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[runInfo.commonRunInfo.valueOffset + gmNOffset],
                                                        dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            } else if constexpr (IS_BN2_MULTIBLK) {
                if (isFixpOut) {
                    bool needAtomic = ((!IS_DKV_RESIDENT_L0C) && runInfo.isS2IdxNoChange && IS_BN2_MULTIBLK);
                    if (needAtomic) {
                        SetAtomicAdd<CALC_TYPE>();
                    }
                    int64_t offset = GetBlockIdx() * CUBE_BASEN * HEAD_DIM_ALIGN;
                    Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[offset],
                                                                dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                    if (needAtomic) {
                        SetAtomicNone();
                    }
                }
            } else if constexpr (SPLIT_AXIS == BN2GS1S2) {
                if (isFixpOut) {
                    SetAtomicAdd<CALC_TYPE>();
                    Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[runInfo.commonRunInfo.valueOffset + gmNOffset],
                                                            dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                    SetAtomicNone();
                }
            } else { // BNS2
                if (isFixpOut) {
                    bool needAtomic =
                        ((!IS_DKV_RESIDENT_L0C) && runInfo.isS2IdxNoChange && !isDkvL0CResidentForD192Dv128) ||
                        (!runInfo.isFirstBlock && (runInfo.specialS2Index != -1));
                    if (needAtomic) {
                        SetAtomicAdd<CALC_TYPE>();
                    }
                    int64_t offset = (runInfo.specialS2Index != -1) ? (runInfo.specialS2Index * CUBE_BASEN * HEAD_DIM_ALIGN * NUM_TWO + CUBE_BASEN * HEAD_DIM_ALIGN) :
                        GetBlockIdx() * CUBE_BASEM * HEAD_DIM_ALIGN;
                    Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[offset],
                                                            dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
                    if (needAtomic) {
                        SetAtomicNone();
                    }
                }
            }
        } else {
            if (isFixpOut) {
                fixpipeParams.nSize = (realN + 7) >> 3 << 3;
                Fixpipe<T, CALC_TYPE, DV_FIXPIPE_CONFIG>(outTensor[gmNOffset], dvL0CBuffer.GetTensor<CALC_TYPE>(), fixpipeParams);
            }
        }
        dvL0CBuffer.Set<HardEvent::FIX_M>();
    }
}

TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                                 BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
    } else {
        IterateMmDsKNormal<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                                 BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
    } else {
        IterateMmDsQNormal<T, IS_WRITE_UB>(outTensor, dSL1Buf, constInfo, runInfo);
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
template <typename T, bool IS_WRITE_UB>
__aicore__ inline void FAGBlockCube<TEMPLATE_ARGS>::IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                                                 BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                                                 FagConstInfo &constInfo, FagRunInfo &runInfo)
{
    if constexpr (IS_DETER_OLD(DETER_SPARSE_TYPE)) {
    } else {
        IterateMmPDyNormal<T, IS_WRITE_UB>(outTensor, pL1Buf, constInfo, runInfo);
    }
}
 
TEMPLATES_DEF
class FAGBlockCubeDummy {
public:
    __aicore__ inline FAGBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe *pipe, FagTilingType tilingData,
                                              BufferManager<BufferType::L1> *l1BuffMgr){};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR dy, GM_ADDR queryRope,
                                            GM_ADDR keyRope, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv, GM_ADDR workspace){};
    __aicore__ inline void InitCubeBuffer(FagConstInfo &constInfo){};
    __aicore__ inline void IterateMmDyV(LocalTensor<CALC_TYPE> &mm1ResTensor, FagConstInfo &constInfo,
                                        FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs){};
    __aicore__ inline void IterateMmDyVQuant(LocalTensor<CALC_TYPE> &mm1ResTensor, Buffer<BufferType::L1, SyncType::NO_SYNC> &vL1Buffer,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs){};
    __aicore__ inline void IterateMmQK(LocalTensor<CALC_TYPE> &mm2ResTensor, FagConstInfo &constInfo,
                                       FagRunInfo &runInfo, PreloadArgs<IS_ROPE> &preloadArgs){};
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsK(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){}; // dq
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmDsQ(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &dSL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){}; // dk
    template <typename T, bool IS_WRITE_UB>
    __aicore__ inline void IterateMmPDy(typename DqkvResPos<T, IS_WRITE_UB>::PosType outTensor,
                                        BuffersPolicySingleBuffer<BufferType::L1, SyncType::NO_SYNC> &pL1Buf,
                                        FagConstInfo &constInfo, FagRunInfo &runInfo){}; // dv
};
 
template <typename T>
struct CubeBlockTraits; // 声明
 
/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;
 
#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS)                                                                     \
    TEMPLATES_DEF_NO_DEFAULT                                                                                           \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> {                                                          \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE)                                                                  \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST)                                                                \
    };
 
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(FAGBlockCubeDummy);
 
// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS                                                                                                    \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)                                                                       \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
 
 
} // namespace FagBaseApi
#endif