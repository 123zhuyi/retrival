/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file is_close_dag.h
 * atan2:
 * y = atan2(x1, x2) = atan(x1/x2)  (x2>0)
 *                   = atan(x1/x2) + pi (x2<0 and x1>=0)
 *                   = atan(x1/x2) - pi (x2<0 and x1<0)
 *                   = pi/2 (x2=0 and x1>0)
 *                   = -pi/2 (x2=0 and x1<0)
 *                   = nan (x2=0 and x1=0)
 * atan(x) = x - x^3/3 + x^5/5 - x^7/7 + x^9/9 - x^11/11 + x^13/13... (|x|<=1)
 *         = pi/4 + atan((x-1)/(x+1)) (x>1)
 *         = pi/8 + atan((x-tan(pi/8))/(1+tan(pi/8)*x)) (tan(pi/8)=0.4142135623730950) (x > tan(pi/8) and x <
 * tan(pi/4)))
 */

#ifndef ATAN2_DAG_H
#define ATAN2_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace Atan2Op {
using namespace Ops::Base;
using namespace AscendC;
template <typename T>
__aicore__ constexpr float GetMaxFloat()
{
    if constexpr (std::is_same<T, float>::value) {
        return 10000000.0f;
    }
    return 65504.0f;
}
constexpr float PI = 3.14159265358979323846f;
constexpr float PI_BY_2 = 1.5707963267948966192313216916398f;
constexpr float PI_BY_8 = 0.392699081698724155f;
constexpr float TAN_PI_BY_8 = 0.4142135623730950f;
constexpr float PI_BY_4 = 0.78539816339744831f;
constexpr float PI_3_4 = 2.3561944901923448f;
constexpr uint16_t TAYLOR_SERIES_ORDER = 13;
constexpr uint16_t TAYLOR_SERIES_TERMS = (TAYLOR_SERIES_ORDER + 1) / 2;
constexpr float TAYLOR_SERIES_COEFF[TAYLOR_SERIES_TERMS] = {1.0f,        -1.0f / 3.0f,  1.0f / 5.0f, -1.0f / 7.0f,
                                                            1.0f / 9.0f, -1.0f / 11.0f, 1.0f / 13.0f};
constexpr uint32_t FLOAT_SIGN_MASK = 0x80000000;
constexpr uint32_t FLOAT_ONE_MASK = 0x3F800000;
constexpr float CONST_INF = INFINITY;
constexpr float CONST_NEG_INF = -INFINITY;

template <typename T>
struct Atan2Compute : public Vec::ElemwiseBinaryOP<float, float, float> {
    __aicore__ inline Atan2Compute(LocalTensor<float>& dst, LocalTensor<float>& src0, LocalTensor<float>& src1,
                                   uint32_t count)
    {
#ifdef __CCE_AICORE__
        uint32_t dtypeSize = sizeof(float);
        constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint32_t loopNum = (count + vl - 1) / vl;
        uint32_t vlSize = vl;

        __ubuf__ float* src0Addr = (__ubuf__ float*)src0.GetPhyAddr();
        __ubuf__ float* src1Addr = (__ubuf__ float*)src1.GetPhyAddr();
        __ubuf__ float* dstAddr = (__ubuf__ float*)dst.GetPhyAddr();

        __VEC_SCOPE__
        {
            Reg::MaskReg mask0 = Reg::CreateMask<uint32_t>();
            Reg::RegTensor<uint32_t> floatSignMaskReg;
            Reg::RegTensor<uint32_t> floatOneMaskReg;
            Reg::RegTensor<float> floatZeroReg;
            Reg::RegTensor<float> floatNegZeroReg;
            Reg::RegTensor<float> floatMaxReg;
            Reg::RegTensor<float> piReg;
            Reg::RegTensor<float> negPiReg;
            Reg::RegTensor<float> piBy2Reg;
            Reg::RegTensor<float> negPiBy2Reg;
            Reg::RegTensor<float> piBy4Reg;
            Reg::RegTensor<float> negPiBy4Reg;
            Reg::RegTensor<float> pi3By4Reg;
            Reg::RegTensor<float> negPi3By4Reg;
            Reg::RegTensor<float> adjustReg;
            Reg::Duplicate(floatSignMaskReg, FLOAT_SIGN_MASK, mask0);
            Reg::Duplicate(floatOneMaskReg, FLOAT_ONE_MASK, mask0);
            Reg::Duplicate(floatZeroReg, 0.0f, mask0);
            Reg::Duplicate(floatMaxReg, GetMaxFloat<float>(), mask0);
            Reg::Duplicate(piReg, PI, mask0);
            Reg::Duplicate(negPiReg, -PI, mask0);
            Reg::Duplicate(piBy2Reg, PI_BY_2, mask0);
            Reg::Duplicate(negPiBy2Reg, -PI_BY_2, mask0);
            Reg::Duplicate(piBy4Reg, PI_BY_4, mask0);
            Reg::Duplicate(negPiBy4Reg, -PI_BY_4, mask0);
            Reg::Duplicate(pi3By4Reg, PI_3_4, mask0);
            Reg::Duplicate(negPi3By4Reg, -PI_3_4, mask0);

            Reg::RegTensor<float> addReg1; // 1.0f
            Reg::RegTensor<float> addReg3; // -1.0f / 3.0f
            Reg::RegTensor<float> addReg5; // 1.0f / 5.0f
            Reg::RegTensor<float> addReg7; // -1.0f / 7.0f
            Reg::RegTensor<float> addReg9; // 1.0f / 9.0f
            Reg::Duplicate(addReg1, 1.0f, mask0);
            Reg::Duplicate(addReg3, -1.0f / 3.0f, mask0);
            Reg::Duplicate(addReg5, 1.0f / 5.0f, mask0);
            Reg::Duplicate(addReg7, -1.0f / 7.0f, mask0);
            Reg::Duplicate(addReg9, 1.0f / 9.0f, mask0);

            Reg::RegTensor<float> x1Reg;
            Reg::RegTensor<float> x2Reg;
            Reg::RegTensor<float> yReg;
            Reg::RegTensor<float> xReg;
            Reg::MaskReg x1Mask;
            Reg::MaskReg x2Mask;
            Reg::MaskReg yMask;
            Reg::MaskReg mask;

            for (uint16_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
                Reg::Duplicate(adjustReg, 0.0f, mask0);
                mask = Reg::UpdateMask<float, Reg::RegTraitNumOne>(count);
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(x1Reg, src0Addr, vlSize);
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(x2Reg, src1Addr, vlSize);
                Reg::Div(xReg, x1Reg, x2Reg, mask); // x1/x2
                Atan(yReg, xReg, floatSignMaskReg, floatOneMaskReg, floatMaxReg, addReg1, addReg3, addReg5, addReg7,
                     addReg9, mask);
                // add pi where x2<0 and x1>=0, -pi where x2<0 and x1<0
                Reg::CompareScalar<float, CMPMODE::LT>(x2Mask, x2Reg, 0.0f, mask); // x2 < 0
                Reg::CompareScalar<float, CMPMODE::GE>(x1Mask, x1Reg, 0.0f, mask); // x1 >= 0
                Reg::MaskAnd(yMask, x1Mask, x2Mask, mask);                         // x1 >=0 && x2 < 0
                Reg::Select(adjustReg, piReg, adjustReg, yMask);
                Reg::MaskNot(x1Mask, x1Mask, mask);        // x1 < 0
                Reg::MaskAnd(yMask, x1Mask, x2Mask, mask); // x1 < 0 && x2 < 0
                Reg::Select(adjustReg, negPiReg, adjustReg, yMask);
                Reg::Add(yReg, yReg, adjustReg, mask);
                PostProcess(yReg, x1Reg, x2Reg, yMask, mask, floatSignMaskReg, floatOneMaskReg, floatZeroReg, piReg,
                            negPiReg, piBy2Reg, negPiBy2Reg, piBy4Reg, negPiBy4Reg, pi3By4Reg, negPi3By4Reg);
                Reg::DataCopy<float, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, yReg, vlSize, mask);
            }
        }
    }

    __aicore__ inline void PostProcess(Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& x1Reg,
                                       Reg::RegTensor<float>& x2Reg, Reg::MaskReg& yMask, Reg::MaskReg& mask,
                                       Reg::RegTensor<uint32_t>& floatSignMaskReg,
                                       Reg::RegTensor<uint32_t>& floatOneMaskReg, Reg::RegTensor<float>& floatZeroReg,
                                       Reg::RegTensor<float>& piReg, Reg::RegTensor<float>& negPiReg,
                                       Reg::RegTensor<float>& piBy2Reg, Reg::RegTensor<float>& negPiBy2Reg,
                                       Reg::RegTensor<float>& piBy4Reg, Reg::RegTensor<float>& negPiBy4Reg,
                                       Reg::RegTensor<float>& pi3By4Reg, Reg::RegTensor<float>& negPi3By4Reg)
    {
        Reg::RegTensor<float> xSignReg;
        Reg::MaskReg x1ZeroMask;
        Reg::MaskReg x1NegMask;
        Reg::MaskReg x1PosZeroMask;
        Reg::MaskReg x1NegZeroMask;
        Reg::MaskReg x1PosInfMask;
        Reg::MaskReg x1NegInfMask;

        Reg::MaskReg x2ZeroMask;
        Reg::MaskReg x2NegMask;
        Reg::MaskReg x2PosMask;
        Reg::MaskReg x2NegZeroMask;
        Reg::MaskReg x2PosInfMask;
        Reg::MaskReg x2NegInfMask;

        Reg::CompareScalar<float, CMPMODE::EQ>(x1ZeroMask, x1Reg, 0.0f, mask); // x1 = ±0.0
        Reg::And((Reg::RegTensor<uint32_t>&)xSignReg, (Reg::RegTensor<uint32_t>&)x1Reg, floatSignMaskReg,
                 mask); // 取x1符号位
        Reg::Or((Reg::RegTensor<uint32_t>&)xSignReg, (Reg::RegTensor<uint32_t>&)xSignReg, floatOneMaskReg,
                mask);                                                            // +/- 1
        Reg::CompareScalar<float, CMPMODE::EQ>(x1NegMask, xSignReg, -1.0f, mask); // x1 is neg
        Reg::MaskAnd(x1NegZeroMask, x1ZeroMask, x1NegMask, mask);                 // x1 = -0.0
        Reg::MaskXor(x1PosZeroMask, x1ZeroMask, x1NegZeroMask, mask);             // x1 = +0.0

        Reg::CompareScalar<float, CMPMODE::EQ>(x2ZeroMask, x2Reg, 0.0f, mask); // x2 = ±0.0
        Reg::And((Reg::RegTensor<uint32_t>&)xSignReg, (Reg::RegTensor<uint32_t>&)x2Reg, floatSignMaskReg,
                 mask); // 取x2符号位
        Reg::Or((Reg::RegTensor<uint32_t>&)xSignReg, (Reg::RegTensor<uint32_t>&)xSignReg, floatOneMaskReg,
                mask);                                                            // +/- 1
        Reg::CompareScalar<float, CMPMODE::EQ>(x2NegMask, xSignReg, -1.0f, mask); // x2 is neg
        Reg::MaskNot(x2PosMask, x2NegMask, mask);                                 // x2 is pos
        Reg::MaskAnd(x2NegZeroMask, x2ZeroMask, x2NegMask, mask);                 // x2 = -0.0

        // (x1 < 0 || x1 = -0.0) && x2 = -0.0 --> -pi/2; x1 = -0.0后续修正
        Reg::MaskAnd(yMask, x1NegMask, x2NegZeroMask, mask);
        Reg::Select(yReg, negPiBy2Reg, yReg, yMask);

        // (x1 > 0 || x1 = +0.0) && x2 = -0.0 --> pi/2; x1 = 0.0后续修正
        Reg::MaskNot(yMask, x1NegMask, mask); // x1 is pos
        Reg::MaskAnd(yMask, yMask, x2NegZeroMask, mask);
        Reg::Select(yReg, piBy2Reg, yReg, yMask);

        // x1 = -0.0 && (x2 < 0.0 || x2 = -0.0) --> -pi
        Reg::MaskAnd(yMask, x1NegZeroMask, x2NegMask, mask);
        Reg::Select(yReg, negPiReg, yReg, yMask);

        // x1 = -0.0 && (x2 > 0.0 || x2 = +0.0) --> -0.0
        Reg::MaskAnd(yMask, x1NegZeroMask, x2PosMask, mask);
        Reg::Select(yReg, (Reg::RegTensor<float>&)floatSignMaskReg, yReg, yMask);

        // x1 = 0.0 && (x2 < 0.0 || x2 = -0.0) --> pi
        Reg::MaskAnd(yMask, x1PosZeroMask, x2NegMask, mask);
        Reg::Select(yReg, piReg, yReg, yMask);

        // x1 = 0.0 && (x2 > 0.0 || x2 = +0.0) --> 0.0
        Reg::MaskAnd(yMask, x1PosZeroMask, x2PosMask, mask);
        Reg::Select(yReg, floatZeroReg, yReg, yMask);

        Reg::CompareScalar<float, CMPMODE::EQ>(x1PosInfMask, x1Reg, CONST_INF, mask);     // x1 = +inf
        Reg::CompareScalar<float, CMPMODE::EQ>(x1NegInfMask, x1Reg, CONST_NEG_INF, mask); // x1 = -inf
        Reg::CompareScalar<float, CMPMODE::EQ>(x2PosInfMask, x2Reg, CONST_INF, mask);     // x2 = +inf
        Reg::CompareScalar<float, CMPMODE::EQ>(x2NegInfMask, x2Reg, CONST_NEG_INF, mask); // x2 = -inf

        // x1 < 0 && x2 = inf --> -0.0 --> -0.0; x1 = -inf后续修正
        Reg::MaskAnd(yMask, x1NegMask, x2PosInfMask, mask);
        Reg::Select(yReg, (Reg::RegTensor<float>&)floatSignMaskReg, yReg, yMask);

        // x1 = inf && x2 = inf --> pi/4
        Reg::MaskAnd(yMask, x1PosInfMask, x2PosInfMask, mask);
        Reg::Select(yReg, piBy4Reg, yReg, yMask);

        // x1 = -inf && x2 = inf --> -pi/4
        Reg::MaskAnd(yMask, x1NegInfMask, x2PosInfMask, mask);
        Reg::Select(yReg, negPiBy4Reg, yReg, yMask);

        // x1 = inf && x2 = -inf --> 3pi/4
        Reg::MaskAnd(yMask, x1PosInfMask, x2NegInfMask, mask);
        Reg::Select(yReg, pi3By4Reg, yReg, yMask);

        // x1 = -inf && x2 = -inf --> -3pi/4
        Reg::MaskAnd(yMask, x1NegInfMask, x2NegInfMask, mask);
        Reg::Select(yReg, negPi3By4Reg, yReg, yMask);
    }

    __aicore__ inline void TaylorSeriesExpansion(Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& xReg,
                                                 Reg::RegTensor<float>& addReg1, Reg::RegTensor<float>& addReg3,
                                                 Reg::RegTensor<float>& addReg5, Reg::RegTensor<float>& addReg7,
                                                 Reg::RegTensor<float>& addReg9, Reg::MaskReg& mask)
    {
        // x - 1/3*x^3 + 1/5*x^5 - 1/7*x^7 + 1/9*x^9 - 1/11*x^11 + 1/13*x^13
        // x(1 - x^2(1/3 - x^2(1/5 - x^2(1/7 - x^2(1/9 - x^2(1/11 - x^2/13))))))
        Reg::RegTensor<float> xSquarReg;
        Reg::Mul(xSquarReg, xReg, xReg, mask);

        Reg::Duplicate(yReg, -1.0f / 11.0f, mask);
        Reg::Axpy(yReg, xSquarReg, 1.0f / 13.0f, mask);
        Reg::MulDstAdd(yReg, xSquarReg, addReg9, mask);
        Reg::MulDstAdd(yReg, xSquarReg, addReg7, mask);
        Reg::MulDstAdd(yReg, xSquarReg, addReg5, mask);
        Reg::MulDstAdd(yReg, xSquarReg, addReg3, mask);
        Reg::MulDstAdd(yReg, xSquarReg, addReg1, mask);
        Reg::Mul(yReg, yReg, xReg, mask);
    }

    __aicore__ inline void AtanLT1(Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& xReg,
                                   Reg::RegTensor<float>& addReg1, Reg::RegTensor<float>& addReg3,
                                   Reg::RegTensor<float>& addReg5, Reg::RegTensor<float>& addReg7,
                                   Reg::RegTensor<float>& addReg9, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> tmpReg0;
        Reg::RegTensor<float> tmpReg1;
        Reg::Muls(tmpReg0, xReg, TAN_PI_BY_8, mask);
        Reg::Adds(tmpReg0, tmpReg0, 1.0f, mask);
        Reg::Adds(tmpReg1, xReg, -TAN_PI_BY_8, mask);
        Reg::Div(tmpReg0, tmpReg1, tmpReg0, mask);
        Reg::Abs(tmpReg1, tmpReg0, mask);
        TaylorSeriesExpansion(tmpReg0, tmpReg1, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        Reg::Adds(tmpReg0, tmpReg0, PI_BY_8, mask);
        TaylorSeriesExpansion(tmpReg1, xReg, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        Reg::Min(yReg, tmpReg0, tmpReg1, mask);
    }

    __aicore__ inline void Atan(Reg::RegTensor<float>& yReg, Reg::RegTensor<float>& xReg,
                                Reg::RegTensor<uint32_t>& floatSignMaskReg, Reg::RegTensor<uint32_t>& floatOneMaskReg,
                                Reg::RegTensor<float>& thresholdReg, Reg::RegTensor<float>& addReg1,
                                Reg::RegTensor<float>& addReg3, Reg::RegTensor<float>& addReg5,
                                Reg::RegTensor<float>& addReg7, Reg::RegTensor<float>& addReg9, Reg::MaskReg& mask)
    {
        Reg::RegTensor<float> absReg;
        Reg::RegTensor<float> tmpReg0;
        Reg::RegTensor<float> tmpReg1;
        Reg::RegTensor<float> signReg;
        Reg::And((Reg::RegTensor<uint32_t>&)signReg, (Reg::RegTensor<uint32_t>&)xReg, floatSignMaskReg,
                 mask); // 取x2符号位
        Reg::Or((Reg::RegTensor<uint32_t>&)signReg, (Reg::RegTensor<uint32_t>&)signReg, floatOneMaskReg,
                mask); // +/- 1
        Reg::Abs(absReg, xReg, mask);
        Reg::Min(absReg, absReg, thresholdReg, mask); // avoid overflow

        Reg::Adds(tmpReg0, absReg, -1.0f, mask);
        Reg::Adds(tmpReg1, absReg, 1.0f, mask);
        Reg::Div(tmpReg0, tmpReg0, tmpReg1, mask);
        Reg::Abs(tmpReg1, tmpReg0, mask);
        AtanLT1(tmpReg0, tmpReg1, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        Reg::Adds(tmpReg0, tmpReg0, PI_BY_4, mask);
        AtanLT1(tmpReg1, absReg, addReg1, addReg3, addReg5, addReg7, addReg9, mask);
        Reg::Min(yReg, tmpReg0, tmpReg1, mask);
        // recover sign
        Reg::Mul(yReg, yReg, signReg, mask);
#endif
    }
};

template <typename T>
struct Atan2Dag {
    using InputX1T = Bind<Vec::CopyInBrc<T>, Placeholder::In0<T>>;
    using InputX2T = Bind<Vec::CopyInBrc<T>, Placeholder::In1<T>>;

    // cast
    using InputX1 = Bind<Vec::Cast<float, T, 0>, InputX1T>;
    using InputX2 = Bind<Vec::Cast<float, T, 0>, InputX2T>;

    using Atan2Res = Bind<Atan2Compute<T>, InputX1, InputX2>;
    using OpCastRes = Bind<Vec::Cast<T, float, 1>, Atan2Res>;
    using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpCastRes>;

    // 指定输出节点
    using Outputs = Elems<OpCopyOut>;
    using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
    using OpDag = DAGSch<Outputs, void, MemCfg>;
};
} // namespace Atan2Op

#endif // ATAN2_DAG_H
