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
 * \file arange.h
 * \brief
 */
#ifndef __ARANGE_H__
#define __ARANGE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arange_tiling_data.h"
#include "arange_tiling_key.h"

namespace NsArange {

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32; // 32B 对齐块大小
#define ALIGN_UP_32B_ELEMENTS(count, T) (((((count) * sizeof(T)) + 31) / 32) * (32 / sizeof(T)))

// 本核 former/tail 区间参数：由 ParseCoreParams 从 TilingData + GetBlockIdx() 解算。
struct ArangeCoreParams {
    uint32_t unitLoops; // 本核 UB 子循环次数
    uint32_t tailNum;   // 本核最后一个 UB 块的元素数
    int64_t coreOffset; // 本核全局元素起始偏移（GM 偏移链，放宽 int64 防 N 接近 2³² 时溢出）
    int64_t coreLen;    // 本核处理元素数（接 formerLength/tailLength）
};

// 用 GetBlockIdx() 解析本核 former/tail 区间，两个 Kernel 类共用。
__aicore__ inline ArangeCoreParams ParseCoreParams(const ArangeTilingData& tilingData)
{
    ArangeCoreParams p;
    uint32_t coreId = AscendC::GetBlockIdx();
    if (coreId < tilingData.formerNum) {
        p.coreLen = tilingData.formerLength;
        p.unitLoops = tilingData.formerUnitLoops;
        p.tailNum = tilingData.formerTailNum;
        p.coreOffset = tilingData.formerLength * coreId;
    } else {
        p.coreLen = tilingData.tailLength;
        p.unitLoops = tilingData.tailUnitLoops;
        p.tailNum = tilingData.tailTailNum;
        p.coreOffset = tilingData.formerLength * tilingData.formerNum +
                       tilingData.tailLength * (coreId - tilingData.formerNum);
    }
    return p;
}

// 末块 OOB 防护：former/tail 段长按 32B 对齐放大，末核名义 coreLen 可能超过真实剩余元素数，
// 直接 DataCopy 对齐元素数会越过 out[N] 末尾（窄整型 1B/2B 尤甚）。
//   realNum = min(num, totalNum - globalOffset)；
//   realNum 满 32B 对齐走 DataCopy 快路径，否则末块用 DataCopyPad 按真实字节精确写。
// 两个 Kernel 类共用。globalOffset 用于全局 N 兜底；localOffset 为本核内 outGm 写回偏移。
// globalOffset/totalNum 放宽 int64（全局元素计数链）；localOffset/num 保持 uint32（本核内偏移，可证 <2³²）。
template <typename TYPE_OUT>
__aicore__ inline void ArangeCopyOutImpl(const AscendC::GlobalTensor<TYPE_OUT>& outGm,
                                         const AscendC::LocalTensor<TYPE_OUT>& outLocal, int64_t globalOffset,
                                         uint32_t localOffset, uint32_t num, int64_t totalNum)
{
    // 全局 N 兜底：真实可写元素数，防 32B 对齐放大越过 out[N] 末尾（窄整型尤甚）
    uint32_t realNum = num;
    if (globalOffset < totalNum) {
        int64_t remain = totalNum - globalOffset; // 全局剩余元素数
        if (static_cast<int64_t>(realNum) > remain) {
            realNum = static_cast<uint32_t>(remain); // remain<num≤unitNum<2³²，收窄安全
        }
    } else {
        realNum = 0;
    }
    if (realNum != 0) {
        uint32_t aligned = ALIGN_UP_32B_ELEMENTS(realNum, TYPE_OUT);
        if (aligned == realNum) {
            // 真实元素数即 32B 对齐：快路径 DataCopy，不越界
            AscendC::DataCopy(outGm[localOffset], outLocal, realNum);
        } else {
            // 末块非 32B 对齐：DataCopyPad 按真实字节精确写，杜绝 1B/2B 尾轴越界
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(realNum *
                                                        sizeof(TYPE_OUT)); // byte 粒度，sizeof(TYPE_OUT) 整数倍
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(outGm[localOffset], outLocal, copyParams);
        }
    }
}

// 共用 Init 样板：从 TilingData + GetBlockIdx() 解析本核 former/tail 区间，写入六个引用出参。
__aicore__ inline void ArangeInitCoreParams(const ArangeTilingData& tilingData, uint32_t& unitNum, int64_t& totalNum,
                                            uint32_t& unitLoops, uint32_t& tailNum, int64_t& coreOffset,
                                            int64_t& coreLen)
{
    ASSERT(AscendC::GetBlockNum() != 0); // block dim 不可为 0
    unitNum = tilingData.unitNum;
    totalNum = tilingData.totalNum;
    ArangeCoreParams cp = ParseCoreParams(tilingData);
    unitLoops = cp.unitLoops;
    tailNum = cp.tailNum;
    coreOffset = cp.coreOffset;
    coreLen = cp.coreLen;
}

template <typename TYPE_START, typename TYPE_STEP, typename TYPE_OUT>
class KernelArange {
public:
    __aicore__ inline KernelArange(){};

    // 多核 former/tail 切分：每核用 GetBlockIdx() 定位本核区间 [coreOffset, coreOffset+coreLen)。
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                                const ArangeTilingData& tilingData)
    {
        ArangeInitCoreParams(tilingData, this->unitNum, this->totalNum, this->unitLoops, this->tailNum,
                             this->coreOffset, this->coreLen);

        startGm.SetGlobalBuffer((__gm__ float*)start);
        stepGm.SetGlobalBuffer((__gm__ float*)step);
        // outGm 基址 + 本核全局偏移，使 outGm[0] 即本核起点
        outGm.SetGlobalBuffer((__gm__ float*)out + this->coreOffset, this->coreLen);

        pipe.InitBuffer(outQueue, BUFFER_NUM, this->unitNum * sizeof(float));
        pipe.InitBuffer(temp1, this->unitNum * sizeof(float));
        pipe.InitBuffer(temp2, this->unitNum * sizeof(float));
        pipe.InitBuffer(temp3, this->unitNum * sizeof(float));
    }

    __aicore__ inline void work_init()
    {
        TYPE_START start = startGm.GetValue(0);
        TYPE_STEP step = stepGm.GetValue(0);

        this->calc_init = temp1.Get<float>();
        this->blockStep = temp2.Get<float>(); // 块间递推步进（unitNum*step，每块累加一次）
        this->calc_temp = temp3.Get<float>();
        // 本核首元素叠加 coreOffset*step：calc_init = start + coreOffset*step + step*iota
        // AICore 禁止 unsigned int -> float 直接转换，coreOffset 先转 int64 再转 float。
        float baseStart = static_cast<float>(start) +
                          static_cast<float>(static_cast<int64_t>(this->coreOffset)) * static_cast<float>(step);
        // setup 只造本核实际会用到的元素数，避免每核恒按满 UB 块（unitNum）做无用初始化。
        // setupNum = min(unitNum, 32B 对齐放大的 coreLen)，覆盖所有 Compute 用到的元素数且不超 buffer 容量。
        uint32_t setupNum = this->unitNum;
        if (this->coreLen < this->unitNum) {
            // coreLen 为 int64，本分支值很小，显式收窄消告警
            setupNum = static_cast<uint32_t>(ALIGN_UP_32B_ELEMENTS(this->coreLen, float));
            if (setupNum > this->unitNum) {
                setupNum = this->unitNum;
            }
        }
        // 用单条向量 ArithProgression 一次生成 baseStart + i*step，替代标量 SetValue 循环造 iota + Duplicate/Mul/Add。
        // 语义等价：ArithProgression(dst, firstValue, diffValue, count) → dst[i]=firstValue+i*diffValue。
        AscendC::ArithProgression<float>(calc_init, baseStart, static_cast<float>(step), setupNum);
        // 块间递推累加器：calc_temp 起始 0（本块 outLocal=calc_init+calc_temp）；blockStep=unitNum*step（每块步进）。
        AscendC::Duplicate(calc_temp, static_cast<float>(0.0), setupNum);
        this->offset_step_base = this->unitNum * step;
        AscendC::Duplicate(blockStep, this->offset_step_base, setupNum);
    }

    __aicore__ inline void Process()
    {
        if (this->coreLen == 0) {
            return; // 空核（小 shape 退化 / tail 段为 0）直接返回
        }
        /*初始化第一个UNIT序列值*/
        work_init();

        for (int32_t i = 0; i < this->unitLoops; i++) {
            if (i == this->unitLoops - 1 && this->tailNum > 0) {
                Compute(i, this->tailNum);
                CopyOut(i, this->tailNum);
            } else {
                Compute(i, this->unitNum);
                CopyOut(i, this->unitNum);
            }
        }
    }

private:
    __aicore__ inline void Compute(int32_t iter, int32_t num)
    {
        uint32_t calc_num = ALIGN_UP_32B_ELEMENTS(num, float);
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        AscendC::Add(outLocal, this->calc_init, this->calc_temp, calc_num);
        AscendC::Add(this->calc_temp, this->calc_temp, this->blockStep, calc_num);
        outQueue.EnQue<float>(outLocal);
    }

    __aicore__ inline void CopyOut(int32_t iter, int32_t num)
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        // globalOffset 用 int64 计算，防大 N 下 uint32 溢出
        int64_t globalOffset = this->coreOffset + static_cast<int64_t>(iter) * this->unitNum;
        // localOffset 保持 uint32：本核内偏移可证 <2³²，直接作 GlobalTensor 下标
        uint32_t localOffset = static_cast<uint32_t>(iter) * this->unitNum;
        ArangeCopyOutImpl<float>(outGm, outLocal, globalOffset, localOffset, static_cast<uint32_t>(num),
                                 this->totalNum);
        outQueue.FreeTensor(outLocal);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> temp1, temp2, temp3;
    AscendC::GlobalTensor<float> startGm;
    AscendC::GlobalTensor<float> stepGm;
    AscendC::GlobalTensor<float> outGm;
    AscendC::LocalTensor<float> calc_init, blockStep, calc_temp;

    int64_t totalNum; // 元素总数（放宽 int64 防大 N 溢出）
    uint32_t unitNum;
    uint32_t unitLoops;
    uint32_t tailNum;
    int64_t coreOffset; // 本核全局元素起始偏移（GM 偏移链，放宽 int64）
    int64_t coreLen;    // 本核处理元素数
    /*UNIT之间元素值差间隔*/
    float offset_step_base;
};

/*INT64/BF16/FP16均转成FP32运算*/
template <typename TYPE_START, typename TYPE_STEP, typename TYPE_OUT>
class KernelArange_Cast {
public:
    __aicore__ inline KernelArange_Cast() {}
    // 多核 former/tail 切分：每核用 GetBlockIdx() 定位本核区间 [coreOffset, coreOffset+coreLen)。
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR step, GM_ADDR out,
                                const ArangeTilingData& tilingData)
    {
        ArangeInitCoreParams(tilingData, this->unitNum, this->totalNum, this->unitLoops, this->tailNum,
                             this->coreOffset, this->coreLen);

        startGm.SetGlobalBuffer((__gm__ TYPE_START*)start);
        stepGm.SetGlobalBuffer((__gm__ TYPE_STEP*)step);
        // outGm 基址 + 本核全局偏移，使 outGm[0] 即本核起点
        outGm.SetGlobalBuffer((__gm__ TYPE_OUT*)out + this->coreOffset, this->coreLen);

        // inQueue 块大小取 start/step dtype 的较大者，且至少容纳 1 个 32B block（窄整型标量搬入）。
        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_SIZE);
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->unitNum * sizeof(TYPE_OUT));
        pipe.InitBuffer(temp1, this->unitNum * sizeof(float));
        pipe.InitBuffer(temp2, this->unitNum * sizeof(float));
        pipe.InitBuffer(temp3, this->unitNum * sizeof(float));
        pipe.InitBuffer(temp4, this->unitNum * sizeof(float));
        // 入口 half/bf16 整块 widening Cast(→float) 的中转缓冲：至少 16 float（64B）。
        //   仅 half/bf16 入口用到；整型入口走标量域直读，但统一申请避免分支。
        pipe.InitBuffer(tempCastIn, BLOCK_SIZE * 2); // 64B = 16 float
        // 出口 1B 两段式（float→half→int8/uint8）的 half 中转缓冲。
        // 硬件不支持 float→int8/uint8 直转，必须经 half 中转。
        if constexpr (std::is_same_v<TYPE_OUT, int8_t> || std::is_same_v<TYPE_OUT, uint8_t>) {
            pipe.InitBuffer(tempOutHalf, this->unitNum * sizeof(half));
        }
    }

    // 出口 Cast：FP32 → TYPE_OUT（count 元素）。
    // 支持矩阵：float→{half,int16,int32,int64,bf16} 直转；float→int8/uint8 不支持，
    //   必须两段式 float→half(CAST_ROUND)→int8/uint8(CAST_ROUND)。half→int8/uint8 硬件默认饱和。
    __aicore__ inline void CastFloatToOut(const AscendC::LocalTensor<TYPE_OUT>& outLocal,
                                          const AscendC::LocalTensor<float>& srcFp32, uint32_t count)
    {
        if constexpr (std::is_same_v<TYPE_OUT, int8_t> || std::is_same_v<TYPE_OUT, uint8_t>) {
            AscendC::LocalTensor<half> halfLocal = tempOutHalf.Get<half>();
            AscendC::Cast(halfLocal, srcFp32, AscendC::RoundMode::CAST_ROUND, count);  // float→half 就近取整
            AscendC::Cast(outLocal, halfLocal, AscendC::RoundMode::CAST_ROUND, count); // half→int8/uint8 取整+饱和
        } else {
            AscendC::Cast(outLocal, srcFp32, AscendC::RoundMode::CAST_ROUND,
                          count); // float→int16/int32/int64/fp16/bf16
        }
    }

    // 标量入口读取：把 1 个 TYPE_IN 标量（src 的 [0] 元素）取到 float。
    // 硬件坑：2 字节源（int16/half/bf16）的单元素向量 Cast(count=1) 产出 0；
    //   int8/uint8（经 half 两段式）、int32/int64（4/8B 源直转）不受影响。
    //   half/bf16→float 的标量域转换本 toolchain 不支持，必须走向量 Cast，故改用整块 count=16 规避 count=1 缺陷。
    //   整型走标量域 GetValue(0) + C++ 转换，绕开向量 Cast；整数值 ≤2^24 在 float 精确。
    template <typename TYPE_IN>
    __aicore__ inline float ReadScalarAsFloat(const AscendC::LocalTensor<TYPE_IN>& src)
    {
        if constexpr (std::is_same_v<TYPE_IN, half> || std::is_same_v<TYPE_IN, bfloat16_t>) {
            // 半精度：整块 Cast(half/bf16 → float) 写入 tempCastIn，再标量读 [0]。整块 count=16 规避 count=1 缺陷。
            constexpr uint32_t BLOCK_2B_ELEMS = BLOCK_SIZE / sizeof(TYPE_IN); // 16
            AscendC::LocalTensor<float> tmp = tempCastIn.Get<float>();        // ≥16 float
            AscendC::Cast(tmp, src, AscendC::RoundMode::CAST_NONE, BLOCK_2B_ELEMS);
            AscendC::PipeBarrier<PIPE_ALL>(); // 向量 Cast 写后，标量读前同步
            return tmp.GetValue(0);
        } else {
            // 整型：标量域直读 + C++ 转换，先转 int64 再转 float，规避无符号/位宽问题。
            return static_cast<float>(static_cast<int64_t>(src.GetValue(0)));
        }
    }

    // GM 标量入口读取为 float。
    //   - 窄整型 + int32（≤4B）：GM 标量域 GetValue(0) 直读，绕过 inQueue 的两次 GM↔UB 搬运。
    //   - int64（8B）：8B GM 标量域直读在 DAV_2201 触发 507035 向量核异常，须走 inQueue 经 DataCopy 搬到 UB 后再读。
    //   - half/bf16：须走 inQueue + 整块向量 Cast（2B 单元素向量 Cast bug 规避）。
    template <typename TYPE_IN>
    __aicore__ inline float ReadGmScalarAsFloat(const AscendC::GlobalTensor<TYPE_IN>& srcGm)
    {
        if constexpr (std::is_same_v<TYPE_IN, int8_t> || std::is_same_v<TYPE_IN, uint8_t> ||
                      std::is_same_v<TYPE_IN, int16_t> || std::is_same_v<TYPE_IN, int32_t>) {
            // 窄整型 + int32（≤4B）：GM 标量域直读，绕过 inQueue 两次搬运往返。
            return static_cast<float>(static_cast<int64_t>(srcGm.GetValue(0)));
        } else {
            // int64（8B 直读 507035）/ half / bf16：走 inQueue 搬到 UB 后再读。
            AscendC::LocalTensor<TYPE_IN> local_in = inQueue.AllocTensor<TYPE_IN>();
            AscendC::DataCopy(local_in, srcGm, ALIGN_UP_32B_ELEMENTS(1, TYPE_IN));
            inQueue.EnQue<TYPE_IN>(local_in);
            AscendC::LocalTensor<TYPE_IN> local_out = inQueue.DeQue<TYPE_IN>();
            float v = ReadScalarAsFloat<TYPE_IN>(local_out);
            inQueue.FreeTensor(local_in);
            return v;
        }
    }

    // int32 原生整数域读 GM 标量。
    //   int32 start/step 是 4B GM 标量，GM 标量域 GetValue(0) 直读，不转 float，
    //   直接返回 int32 原值供整数域 ArithProgression<int32_t> 使用，避免 FP32 2^24 精度天花板。
    //   注：8B int64 GM 标量域直读会触发 507035（见 ReadGmScalarAsFloat），本函数仅用于 int32（4B）。
    __aicore__ inline int32_t ReadGmScalarAsInt32(const AscendC::GlobalTensor<int32_t>& srcGm)
    {
        return srcGm.GetValue(0);
    }

    // int32 原生整数域 setup（仅 TYPE_OUT==int32_t 走此分支；其它 dtype 走 FP32 setup）。
    //   全程 int32 整数运算（ArithProgression<int32_t> + 整型 Add 块递推），无 FP32 中转、无出口 Cast，
    //   精确到 2^31。复用同一组 TBuf（sizeof(int32_t)==sizeof(float)==4，容量一致）。
    __aicore__ inline void work_init_int32()
    {
        int32_t int_start = ReadGmScalarAsInt32(startGm);
        int32_t int_step = ReadGmScalarAsInt32(stepGm);

        this->calc_init_i = temp1.Get<int32_t>();
        this->blockStep_i = temp2.Get<int32_t>();
        this->calc_temp_i = temp3.Get<int32_t>();
        // 本核首元素 = start + coreOffset*step。中间用 int64 防 coreOffset*step 溢出，
        //   最终 baseStart 落在 int32 值域（int32 输出语义保证 |start+i*step| ≤ 2^31-1）。
        int64_t baseStart64 = static_cast<int64_t>(int_start) +
                              static_cast<int64_t>(this->coreOffset) * static_cast<int64_t>(int_step);
        int32_t baseStart = static_cast<int32_t>(baseStart64);
        // setup 只造本核实际会用到的元素数（sizeof(int32_t)==sizeof(float)，对齐元素数一致）。
        uint32_t setupNum = this->unitNum;
        if (this->coreLen < this->unitNum) {
            // coreLen 为 int64，本分支值很小，显式收窄消告警
            setupNum = static_cast<uint32_t>(ALIGN_UP_32B_ELEMENTS(this->coreLen, int32_t));
            if (setupNum > this->unitNum) {
                setupNum = this->unitNum;
            }
        }
        // 整数域单条向量 ArithProgression：calc_init_i[i] = baseStart + i*int_step（全 int32，精确到 2^31）。
        AscendC::ArithProgression<int32_t>(calc_init_i, baseStart, int_step, setupNum);
        // 块间递推累加器（整数域）：calc_temp_i 起始 0；blockStep_i = unitNum*int_step（每块步进，int32 域）。
        AscendC::Duplicate(calc_temp_i, static_cast<int32_t>(0), setupNum);
        int32_t blockStepVal = static_cast<int32_t>(static_cast<int64_t>(this->unitNum) *
                                                    static_cast<int64_t>(int_step));
        AscendC::Duplicate(blockStep_i, blockStepVal, setupNum);
    }

    __aicore__ inline void work_init()
    {
        // int32 走原生整数域 setup（精确到 2^31），其它 dtype 走 FP32 路径。
        if constexpr (std::is_same_v<TYPE_OUT, int32_t>) {
            work_init_int32();
            return;
        }
        // —— start / step 标量入口：整型走 GM 标量域直读（免搬运）；half/bf16 走 inQueue 向量 Cast ——
        float float_start = ReadGmScalarAsFloat<TYPE_START>(startGm);
        float float_step = ReadGmScalarAsFloat<TYPE_STEP>(stepGm);

        this->calc_init = temp1.Get<float>();
        this->blockStep = temp2.Get<float>(); // 块间递推步进（unitNum*step）
        this->calc_temp = temp3.Get<float>();
        this->calc_out = temp4.Get<float>();
        // 本核首元素叠加 coreOffset*float_step（calc_init = float_start + coreOffset*float_step + float_step*iota）
        // AICore 禁止 unsigned int -> float 直接转换，coreOffset 先转 int64 再转 float。
        float baseStart = float_start + static_cast<float>(static_cast<int64_t>(this->coreOffset)) * float_step;
        // setup 只造本核实际会用到的元素数，避免恒按满 UB 块初始化。
        uint32_t setupNum = this->unitNum;
        if (this->coreLen < this->unitNum) {
            // coreLen 为 int64，本分支值很小，显式收窄消告警
            setupNum = static_cast<uint32_t>(ALIGN_UP_32B_ELEMENTS(this->coreLen, float));
            if (setupNum > this->unitNum) {
                setupNum = this->unitNum;
            }
        }
        // 用单条向量 ArithProgression 一次生成 baseStart + i*float_step，替代标量 SetValue 循环造 iota +
        // Duplicate/Mul/Add。 语义：dst[i]=baseStart+i*float_step。
        AscendC::ArithProgression<float>(calc_init, baseStart, float_step, setupNum);
        // 块间递推累加器：calc_temp 起始 0（本块
        // calc_out=calc_init+calc_temp）；blockStep=unitNum*float_step（每块步进）。
        AscendC::Duplicate(calc_temp, static_cast<float>(0.0), setupNum);
        this->offset_step_base = this->unitNum * float_step;
        AscendC::Duplicate(blockStep, this->offset_step_base, setupNum);
    }

    __aicore__ inline void Process()
    {
        if (this->coreLen == 0) {
            return; // 空核（小 shape 退化 / tail 段为 0）直接返回
        }
        /*初始化第一个UNIT序列值*/
        work_init();

        for (int32_t i = 0; i < this->unitLoops; i++) {
            if (i == this->unitLoops - 1 && this->tailNum > 0) {
                Compute(i, this->tailNum);
                CopyOut(i, this->tailNum);
            } else {
                Compute(i, this->unitNum);
                CopyOut(i, this->unitNum);
            }
        }
    }

private:
    __aicore__ inline void Compute(int32_t iter, int32_t num)
    {
        uint32_t calc_num = ALIGN_UP_32B_ELEMENTS(num, float); // sizeof(int32_t)==sizeof(float)==4，对齐元素数一致

        AscendC::LocalTensor<TYPE_OUT> outLocal = outQueue.AllocTensor<TYPE_OUT>();
        if constexpr (std::is_same_v<TYPE_OUT, int32_t>) {
            // int32 全程整数域：outLocal(int32) = calc_init_i + calc_temp_i；
            //   块递推 calc_temp_i += blockStep_i。无 FP32 中转、无出口 Cast。
            AscendC::Add(outLocal, this->calc_init_i, this->calc_temp_i, calc_num);
            AscendC::Add(this->calc_temp_i, this->calc_temp_i, this->blockStep_i, calc_num);
        } else {
            AscendC::Add(this->calc_out, this->calc_init, this->calc_temp, calc_num);
            AscendC::Add(this->calc_temp, this->calc_temp, this->blockStep, calc_num);
            CastFloatToOut(outLocal, this->calc_out, calc_num); // 出口 Cast（int8/uint8 两段式经 half）
        }
        outQueue.EnQue<TYPE_OUT>(outLocal);
    }

    __aicore__ inline void CopyOut(int32_t iter, int32_t num)
    {
        AscendC::LocalTensor<TYPE_OUT> outLocal = outQueue.DeQue<TYPE_OUT>();
        // globalOffset 用 int64 计算，防大 N 下 uint32 溢出
        int64_t globalOffset = this->coreOffset + static_cast<int64_t>(iter) * this->unitNum;
        // localOffset 保持 uint32：本核内偏移可证 <2³²，直接作 GlobalTensor 下标
        uint32_t localOffset = static_cast<uint32_t>(iter) * this->unitNum;
        ArangeCopyOutImpl<TYPE_OUT>(outGm, outLocal, globalOffset, localOffset, static_cast<uint32_t>(num),
                                    this->totalNum);
        outQueue.FreeTensor(outLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC>
        tempCastIn; // 入口 half/bf16 整块 widening Cast(→float) 的中转（≥16 float=64B）
    AscendC::TBuf<AscendC::QuePosition::VECCALC>
        tempOutHalf; // 出口 1B 两段式（float→half→int8/uint8）的 half 中转（n 元素）
    AscendC::TBuf<AscendC::QuePosition::VECCALC> temp1, temp2, temp3, temp4;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<TYPE_START> startGm;
    AscendC::GlobalTensor<TYPE_STEP> stepGm;
    AscendC::GlobalTensor<TYPE_OUT> outGm;
    AscendC::LocalTensor<float> calc_init, blockStep, calc_temp;
    AscendC::LocalTensor<float> calc_out;
    // int32 原生整数域工作张量（与 calc_init/blockStep/calc_temp 复用同一组 TBuf，
    //   sizeof(int32_t)==sizeof(float)，仅 int32 路径用到）。
    AscendC::LocalTensor<int32_t> calc_init_i, blockStep_i, calc_temp_i;

    int64_t totalNum; // 元素总数（放宽 int64 防大 N 溢出）
    uint32_t unitNum;
    uint32_t unitLoops;
    uint32_t tailNum;
    int64_t coreOffset; // 本核全局元素起始偏移（GM 偏移链，放宽 int64）
    int64_t coreLen;    // 本核处理元素数
    /*UNIT之间元素值差间隔*/
    float offset_step_base;
};

} // namespace NsArange
#endif // ARANGE_H
