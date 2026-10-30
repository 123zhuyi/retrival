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
 * \file power_dag.h
 * \brief Power 算子的 DAG 拓扑集合。
 *
 * 完整计算公式：y = exp(power * log(x * scale + shift))。
 * 本文件定义了 Tiling 层 7 个 culType 分支各自对应的 DAG 实现，外加一个用于
 * 通用幂运算的自定义计算节点 PowerGenericCompute<zeroIsPos>。
 *
 * DAG 一览（与 optiling::CulTypeEnum / POWER_TPL_CUL_* 一一对应）：
 *   PowerAllZerosDag       (ALL_ZEROS)        y = 0
 *   PowerBcastScalarDag    (BROADCAST_SCALAR) y = bcastVal（host 端预计算）
 *   PowerLinearDag         (LINEAR)           y = x*scale + shift
 *   PowerSquareDag         (SQUARE)           y = (x*scale + shift)^2
 *   PowerCubeDag           (CUBE)             y = (x*scale + shift)^3
 *   PowerGenericDag<.,1>   (GENERIC_POW_POS)  通用幂运算，power > 0
 *   PowerGenericDag<.,0>   (GENERIC_POW_NEG)  通用幂运算，power < 0
 */

#ifndef OPS_MATH_POWER_DAG_H
#define OPS_MATH_POWER_DAG_H

#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
#include "atvoss/util/elems.h"

#ifdef __CCE_AICORE__
namespace PowerOp {
// 输入侧 Cast trait：fp16/bf16 升 fp32，无饱和、零填充未触及的 lane。
constexpr static AscendC::Reg::CastTrait POWER_CAST_TRAIT_IN = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
// 输出侧 Cast trait：fp32 降回 fp16/bf16，启用饱和与就近偶舍入。
constexpr static AscendC::Reg::CastTrait POWER_CAST_TRAIT_OUT = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
} // namespace PowerOp
#endif

namespace PowerOp {
using namespace Ops::Base;

// Vec::Cast 的舍入模式枚举：NONE 直接转换，RINT 启用就近偶舍入。
constexpr int POWER_CAST_MODE_NONE = 0;
constexpr int POWER_CAST_MODE_RINT = 1;
// Placeholder::Var 索引：与 tiling 写入 scalarData 的顺序保持一致。
//   IDX_0 = scale / bcastVal
//   IDX_1 = shift
//   IDX_2 = power
//   IDX_3 = negScalar（GENERIC 分支专用）
constexpr int POWER_VAR_IDX_0 = 0;
constexpr int POWER_VAR_IDX_1 = 1;
constexpr int POWER_VAR_IDX_2 = 2;
constexpr int POWER_VAR_IDX_3 = 3;

// ----------------------------------------------------------------------------
// MulAddDstScalar:  dst = src * scaleScalar + shiftScalar
//
// 用 AscendC 的 Axpy 标量变体实现 fused 乘加，硬件单指令完成 MulAdd：
//     AscendC::Axpy(dst, src, scalar, count)  →  dst[i] = src[i] * scalar + dst[i]
// AscendC::MulAddDst / FusedMulAdd 的接口要求 src0/src1 均为 LocalTensor，
// 而本算子的 scale/shift 都是 host 端注入的标量，因此选 Axpy 这个 scalar
// 变体（kernel_operator_vec_ternary_scalar_intf.h）。
//
// 拆分两步：
//   1) Duplicate(dst, shiftScalar, count) —— 把累加项 shift 广播到 dst，
//      为 Axpy 提供累加初值；
//   2) Axpy(dst, src, scaleScalar, count) —— 单条 fused MulAdd 完成
//      src * scale 与 shift 的累加。
//
// 与原 Muls + Adds 实现相比：
//   - 不增加 UB 占用（仅依赖既有的 dst / src）；
//   - 乘加在同一向量指令内完成，省去 Muls→Adds 之间对 dst 的额外读回，
//     vec 流水更紧凑，关键路径少一条独立 vec 指令的发射延迟。
// ----------------------------------------------------------------------------
template <class T>
struct MulAddDstScalar : public Vec::ElemwiseTernaryOP<T, T, T, T> {
    __aicore__ inline MulAddDstScalar(LocalTensor<T>& dst, LocalTensor<T>& src, T scaleScalar, T shiftScalar,
                                      uint32_t count)
    {
#ifdef __CCE_AICORE__
        // 第 1 步：dst = shift  （广播 shift 标量到所有 lanes，作为 Axpy 的累加初值）
        AscendC::Duplicate(dst, shiftScalar, static_cast<int32_t>(count));
        // 第 2 步：dst = src * scaleScalar + dst  （fused MulAdd，单条向量指令）
        AscendC::Axpy(dst, src, scaleScalar, static_cast<int32_t>(count));
#endif
    }
};

// ----------------------------------------------------------------------------
// PowerGenericCompute<zeroIsPos>
//   输入  ：base   (LocalTensor<float>)
//   标量  ：power, negScalar  （构造函数参数）
//   模板量：zeroIsPos
//             1 -> power > 0，base == 0 时输出 0
//             0 -> power < 0，base == 0 时输出 +inf（异常值）
//
// 等价计算（逐元素）：
//     absBase  = |base|
//     logAbs   = log(absBase)
//     rawExp   = exp(power * logAbs)
//     posVal   = rawExp                  // base > 0
//     negVal   = rawExp * negScalar      // base < 0
//                                        //   - 整数 power：negScalar = ±1 → (-1)^power * rawExp
//                                        //   - 非整数 power：negScalar = NaN → 返回 NaN（实数域未定义）
//     zeroVal  = 0 (zeroIsPos==1) 或 +inf (zeroIsPos==0)   // base == 0 时的 IEEE 语义
//     mask_pos  = base > 0
//     tmp       = select(mask_pos, posVal, negVal)
//     mask_zero = base == 0
//     y         = select(mask_zero, zeroVal, tmp)
//
// 实现要点：
//   - 在一个 __VEC_SCOPE__ 内串联 Abs/Log/Mul/Exp/Compare/Select 等 Reg，
//     全程使用 RegTensor 寄存器中间量，避免落 UB；
//   - 严格遵循需求中的“先按 base>0 分支选 posVal/negVal，再按 base==0 覆盖为 zeroVal”
//     两次 Compare+Select 的合并逻辑。
// ----------------------------------------------------------------------------
template <int zeroIsPos>
struct PowerGenericCompute : public Vec::ElemwiseTernaryOP<float, float, float, float> {
    __aicore__ inline PowerGenericCompute(LocalTensor<float>& dst, LocalTensor<float>& base, float power,
                                          float negScalar, uint32_t count)
    {
#ifdef __CCE_AICORE__
        // vl = 单条向量寄存器一次能处理的 fp32 元素数；loopNum = 向上取整的 batch 数。
        constexpr uint32_t dtypeSize = sizeof(float);
        constexpr uint32_t vl = AscendC::VECTOR_REG_WIDTH / dtypeSize;
        const uint16_t loopNum = (count + vl - 1) / vl;

        __ubuf__ float* baseAddr = (__ubuf__ float*)base.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();

        // 常量寄存器源：0 和 +inf。
        constexpr float ZERO_F = 0.0f;
        constexpr float INF_F = __builtin_inff(); // base == 0 且 power < 0 的取值（IEEE）

        // 寄存器拓扑：只需 3 个 RegTensor 即可承接完整数据流（依赖 Reg
        // Abs/Log/Exp/Muls/Select 全部支持 dst 与 src 同寄存器的 in-place 写）：
        //   regBase   : 输入 base，必须存活到 base>0、base==0 两次 Compares
        //   regMain   : 主路径 |base|→log→×power→exp 全部在该寄存器上 in-place 推进，
        //               得到 rawExp；Select#1 之后 rawExp 死亡，复用为 zeroBcast
        //               (0 / +inf) 作为 Select#2 的 src0
        //   regBranch : 先承载 rawExp*negScalar (负底数分支)，Select#1 in-place
        //               写回得到 tmp，Select#2 in-place 写回得到最终结果，
        //               最后 DataCopy 出 UB
        AscendC::Reg::RegTensor<float> regBase;
        AscendC::Reg::RegTensor<float> regMain;
        AscendC::Reg::RegTensor<float> regBranch;
        AscendC::Reg::MaskReg mask;     // 当前 batch 的尾元素 mask
        AscendC::Reg::MaskReg maskPos;  // base > 0
        AscendC::Reg::MaskReg maskZero; // base == 0

        __VEC_SCOPE__
        {
            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                // 更新尾元素 mask 并把当前 batch 的 base 搬入寄存器。
                mask = AscendC::Reg::UpdateMask<float, AscendC::Reg::RegTraitNumOne>(count);
                AscendC::Reg::DataCopy(regBase, (__ubuf__ float*)(baseAddr + loopIdx * vl));

                // ----- 主路径：rawExp = exp(power * log(|base|))，全程 in-place 复用 regMain -----
                AscendC::Reg::Abs(regMain, regBase, mask);         // regMain = |base|
                AscendC::Reg::Log(regMain, regMain, mask);         // regMain = log(|base|)
                AscendC::Reg::Muls(regMain, regMain, power, mask); // regMain = power * log(|base|)
                AscendC::Reg::Exp(regMain, regMain, mask);         // regMain = rawExp

                // ----- 负底数分支：regBranch = rawExp * negScalar -----
                // 整数幂 power：negScalar 为 ±1，结果即 (-1)^power * rawExp；
                // 非整数 power：negScalar 为 NaN（host 端预置），结果为 NaN（实数域未定义）。
                AscendC::Reg::Muls(regBranch, regMain, negScalar, mask);

                // ----- 第 1 次合并：tmp = (base > 0 ? rawExp : negVal)，结果就地写回 regBranch -----
                AscendC::Reg::Compares<float, AscendC::CMPMODE::GT>(maskPos, regBase, ZERO_F, mask);
                AscendC::Reg::Select(regBranch, regMain, regBranch, maskPos);

                // 至此 regMain 中的 rawExp 已用完，复用为 zeroBcast（base==0 时的取值）。
                // zeroIsPos 在编译期已知，编译器会消去未走的 if 分支。
                if constexpr (zeroIsPos == 1) {
                    AscendC::Reg::Duplicate(regMain, ZERO_F, mask);
                } else {
                    AscendC::Reg::Duplicate(regMain, INF_F, mask);
                }

                // ----- 第 2 次合并：y = (base == 0 ? zeroBcast : tmp)，结果就地写回 regBranch -----
                AscendC::Reg::Compares<float, AscendC::CMPMODE::EQ>(maskZero, regBase, ZERO_F, mask);
                AscendC::Reg::Select(regBranch, regMain, regBranch, maskZero);

                // 把当前 batch 的最终结果写回 UB。
                AscendC::Reg::DataCopy((__ubuf__ float*)(dstAddr + loopIdx * vl), regBranch, mask);
            }
        }
#endif
    }
};

// ----------------------------------------------------------------------------
// DAG 0: ALL_ZEROS  —— 输出全 0，不读取输入 x。
//   对应 culType: ALL_ZEROS
//   触发条件: power == 0 或 (shift == 0 且 power > 0 且 scale*power == 0)
//
// 拓扑：Duplicate(0) → CopyOut
// ----------------------------------------------------------------------------
template <typename U>
struct PowerAllZerosDag {
    // 常量定义：创建值为 0 的常量张量
    using ConstZero = MAKE_CONST(U, 0);
    //  Duplicate算子：将常量 0 复制扩展成与输出相同形状的张量
    using OpDuplicate = Bind<Vec::Duplicate<U>, ConstZero>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpDuplicate>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ----------------------------------------------------------------------------
// DAG 1: BROADCAST_SCALAR —— 输出每个元素都等于 host 预计算好的 bcastVal。
//   对应 culType: BROADCAST_SCALAR
//   触发条件: scale*power == 0 且 power != 0（结果整体退化为 pow(shift, power)、
//             +inf [0^negative]、NaN [shift<0 且 power 非整数]，或 power=0 时的 1.0）
//
// bcastVal 由 tiling 以 fp32 写入 Var[0]，kernel 端 Duplicate 后再 Cast 回输出 dtype。
// ----------------------------------------------------------------------------
template <typename U>
struct PowerBcastScalarDag {
    // 从变量槽位0读取预计算的标量值（float类型） 并广播
    using OpDup = Bind<Vec::Duplicate<float>, Placeholder::Var<float, POWER_VAR_IDX_0>>;
    // 将广播后的 float 张量转换为目标数据类型 U
    using OpCastOut = Bind<Vec::Cast<U, float, POWER_CAST_MODE_RINT>, OpDup>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ----------------------------------------------------------------------------
// DAG 2: LINEAR (power == 1) —— y = x*scale + shift
//   对应 culType: LINEAR
//
// 拓扑：CopyIn → Cast(U→fp32) → MulAddDstScalar(scale, shift) → Cast(fp32→U) → CopyOut
// scale 取 Var[0]，shift 取 Var[1]。
// ----------------------------------------------------------------------------
template <typename U, typename T = float>
struct PowerLinearDag {
    using OpCopyIn = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCastIn = Bind<Vec::Cast<T, U, POWER_CAST_MODE_NONE>, OpCopyIn>;
    using OpBase = Bind<MulAddDstScalar<T>, OpCastIn, Placeholder::Var<T, POWER_VAR_IDX_0>, // scale
                        Placeholder::Var<T, POWER_VAR_IDX_1>>;                              // shift
    using OpCastOut = Bind<Vec::Cast<U, T, POWER_CAST_MODE_RINT>, OpBase>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ----------------------------------------------------------------------------
// DAG 3: SQUARE (power == 2) —— y = base * base，其中 base = x*scale + shift
//   对应 culType: SQUARE
//
// 复用 OpBase 节点两次作为 Vec::Mul 的两个输入，避免重复计算 base。
// ----------------------------------------------------------------------------
template <typename U, typename T = float>
struct PowerSquareDag {
    using OpCopyIn = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCastIn = Bind<Vec::Cast<T, U, POWER_CAST_MODE_NONE>, OpCopyIn>;
    using OpBase = Bind<MulAddDstScalar<T>, OpCastIn, Placeholder::Var<T, POWER_VAR_IDX_0>,
                        Placeholder::Var<T, POWER_VAR_IDX_1>>;
    using OpSquare = Bind<Vec::Mul<T>, OpBase, OpBase>; // base * base
    using OpCastOut = Bind<Vec::Cast<U, T, POWER_CAST_MODE_RINT>, OpSquare>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ----------------------------------------------------------------------------
// DAG 4: CUBE (power == 3) —— y = base * base * base
//   对应 culType: CUBE
//
// 先用 base*base 得到 base^2，再与 base 相乘得到 base^3，
// 与需求中“mul(base, base), mul(base, base)”一致（实际是先平方再立方）。
// ----------------------------------------------------------------------------
template <typename U, typename T = float>
struct PowerCubeDag {
    using OpCopyIn = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCastIn = Bind<Vec::Cast<T, U, POWER_CAST_MODE_NONE>, OpCopyIn>;
    using OpBase = Bind<MulAddDstScalar<T>, OpCastIn, Placeholder::Var<T, POWER_VAR_IDX_0>,
                        Placeholder::Var<T, POWER_VAR_IDX_1>>;
    using OpSquare = Bind<Vec::Mul<T>, OpBase, OpBase>; // base^2
    using OpCube = Bind<Vec::Mul<T>, OpSquare, OpBase>; // base^3
    using OpCastOut = Bind<Vec::Cast<U, T, POWER_CAST_MODE_RINT>, OpCube>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

// ----------------------------------------------------------------------------
// DAG 5 / 6: GENERIC (power 不在 {0, 1, 2, 3}) —— 通用幂运算
//   对应 culType: GENERIC_POW_POS (zeroIsPos=1) / GENERIC_POW_NEG (zeroIsPos=0)
//
// 拓扑：CopyIn → Cast(U→fp32) → MulAddDstScalar(scale, shift) [得到 base]
//      → PowerGenericCompute(base, power, negScalar)
//      → Cast(fp32→U) → CopyOut
//
// 4 个 Var 槽位：
//   Var[0] = scale
//   Var[1] = shift
//   Var[2] = power
//   Var[3] = negScalar   （base<0 时的修正系数）
// ----------------------------------------------------------------------------
template <typename U, int zeroIsPos, typename T = float>
struct PowerGenericDag {
    using OpCopyIn = Bind<Vec::CopyIn<U>, Placeholder::In0<U>>;
    using OpCastIn = Bind<Vec::Cast<T, U, POWER_CAST_MODE_NONE>, OpCopyIn>;
    using OpBase = Bind<MulAddDstScalar<T>, OpCastIn, Placeholder::Var<T, POWER_VAR_IDX_0>,              // scale
                        Placeholder::Var<T, POWER_VAR_IDX_1>>;                                           // shift
    using OpGeneric = Bind<PowerGenericCompute<zeroIsPos>, OpBase, Placeholder::Var<T, POWER_VAR_IDX_2>, // power
                           Placeholder::Var<T, POWER_VAR_IDX_3>>;                                        // negScalar
    using OpCastOut = Bind<Vec::Cast<U, T, POWER_CAST_MODE_RINT>, OpGeneric>;
    using OpCopyOut = Bind<Vec::CopyOut<U>, Placeholder::Out0<U>, OpCastOut>;

    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};

} // namespace PowerOp

#endif // OPS_MATH_POWER_DAG_H
