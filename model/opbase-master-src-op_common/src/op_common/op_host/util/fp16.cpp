/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 * The code snippet comes from Huawei's open-source Mindspore project.
 * Copyright 2019-2020 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*!
 * \file fp16.cpp
 * \brief Half precision float
 */
#include "op_common/op_host/util/fp16.h"

namespace Ops {
namespace Base {

constexpr uint16_t FP16_BIT_SHIFT_LEN_2 = 2;
constexpr uint16_t FP16_BIT_SHIFT_LEN_3 = 3;
constexpr uint16_t FP16_BIT_SHIFT_LEN_7 = 7;
constexpr uint16_t FP16_BIT_SHIFT_LEN_10 = 10;
constexpr uint16_t FP16_BIT_SHIFT_LEN_11 = 11;
constexpr uint16_t FP16_BIT_SHIFT_LEN_15 = 15;
constexpr uint16_t FP16_BIT_SHIFT_LEN_31 = 31;
constexpr uint16_t FP16_BIT_SHIFT_LEN_32 = 32;

/**
 * @ingroup fp16 basic parameter
 * @brief   fp16 exponent bias
 */
constexpr uint16_t FP16_EXP_BIAS = 15;

/**
 * @ingroup fp16 basic parameter
 * @brief   the mantissa bit length of fp16 is 10
 */
constexpr uint16_t FP16_MAN_LEN = 10;
/**
 * @ingroup fp16 basic parameter
 * @brief   bit index of sign in fp16
 */
constexpr uint16_t FP16_SIGN_INDEX = 15;
/**
 * @ingroup fp16 basic parameter
 * @brief   sign mask of fp16         (1 00000 00000 00000)
 */
constexpr uint16_t FP16_SIGN_MASK = 0x8000;
/**
 * @ingroup fp16 basic parameter
 * @brief   exponent mask of fp16     (  11111 00000 00000)
 */
constexpr uint16_t FP16_EXP_MASK = 0x7C00;
/**
 * @ingroup fp16 basic parameter
 * @brief   mantissa mask of fp16     (        11111 11111)
 */
constexpr uint16_t FP16_MAN_MASK = 0x03FF;
/**
 * @ingroup fp16 basic parameter
 * @brief   conceal bit of mantissa of fp16(   1 00000 00000)
 */
constexpr uint16_t FP16_MAN_HIDE_BIT = 0x0400;
/**
 * @ingroup fp16 basic parameter
 * @brief   maximum value            (0111 1011 1111 1111)
 */
constexpr uint16_t FP16_MAX = 0x7BFF;
/**
 * @ingroup fp16 basic parameter
 * @brief   minimum value            (1111 1011 1111 1111)
 */
constexpr uint16_t FP16_MIN = 0xFBFF;
/**
 * @ingroup fp16 basic parameter
 * @brief   absolute maximum value   (0111 1111 1111 1111)
 */
constexpr uint16_t FP16_ABS_MAX = 0x7FFF;
/**
 * @ingroup fp16 basic parameter
 * @brief   maximum exponent value of fp16 is 15(11111)
 */
constexpr uint16_t FP16_MAX_EXP = 0x001F;

/**
 * @ingroup fp16 basic parameter
 * @brief   maximum mantissa value of fp16(11111 11111)
 */
constexpr uint16_t FP16_MAX_MAN = 0x03FF;

/**
 * @ingroup fp16 basic operator
 * @brief   get sign of fp16
 */
static inline uint16_t Fp16ExtractSign(uint16_t x) { return (x >> FP16_EXP_BIAS) & 1; }
/**
 * @ingroup fp16 basic operator
 * @brief   get exponent of fp16
 */
static inline uint16_t Fp16ExtractExp(uint16_t x) { return (x >> FP16_MAN_LEN) & FP16_MAX_EXP; }
/**
 * @ingroup fp16 basic operator
 * @brief   get mantissa of fp16
 */
static inline uint16_t Fp16ExtractMan(uint16_t x)
{
    return static_cast<uint16_t>(((x >> 0) & 0x3FF) | ((((x >> FP16_MAN_LEN) & 0x1F) > 0 ? 1 : 0) * 0x400));
}
/**
 * @ingroup fp16 basic operator
 * @brief   constructor of fp16 from sign exponent and mantissa
 */
static inline uint16_t Fp16Constructor(uint16_t s, uint16_t e, uint16_t m)
{
    return static_cast<uint16_t>((s << FP16_SIGN_INDEX) | (e << FP16_MAN_LEN) | (m & FP16_MAX_MAN));
}

/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is zero
 */
static inline bool Fp16IsZero(uint16_t x) { return (x & FP16_ABS_MAX) == 0; }
/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is a denormalized value
 */
static inline bool Fp16IsDenorm(uint16_t x) { return (x & FP16_EXP_MASK) == 0; }

/**
 * @ingroup fp16 special value judgment
 * @brief   whether a fp16 is invalid
 */
static inline bool Fp16IsInvalid(uint16_t x) { return (x & FP16_EXP_MASK) == FP16_EXP_MASK; }

/**
 * @ingroup fp32 basic parameter
 * @brief   fp32 exponent bias
 */
constexpr uint32_t FP32_EXP_BIAS = 127;

/**
 * @ingroup fp32 basic parameter
 * @brief   the mantissa bit length of float/fp32 is 23
 */
constexpr uint32_t FP32_MAN_LEN = 23;
/**
 * @ingroup fp32 basic parameter
 * @brief   bit index of sign in float/fp32
 */
constexpr uint32_t FP32_SIGN_INDEX = 31;
/**
 * @ingroup fp32 basic parameter
 * @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
 */
constexpr uint32_t FP32_SIGN_MASK = 0x80000000u;
/**
 * @ingroup fp32 basic parameter
 * @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
 */
constexpr uint32_t FP32_EXP_MASK = 0x7F800000u;
/**
 * @ingroup fp32 basic parameter
 * @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
 */
constexpr uint32_t FP32_MAN_MASK = 0x007FFFFFu;
/**
 * @ingroup fp32 basic parameter
 * @brief   conceal bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
 */
constexpr uint32_t FP32_MAN_HIDE_BIT = 0x00800000u;
/**
 * @ingroup fp32 basic parameter
 * @brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)
 */
constexpr uint32_t FP32_ABS_MAX = 0x7FFFFFFFu;

/**
 * @ingroup fp32 basic parameter
 * @brief   maximum mantissa value of fp32    (1111 1111 1111 1111 1111 111)
 */
constexpr uint32_t FP32_MAX_MAN = 0x7FFFFF;

static inline uint32_t Fp32Constructor(uint32_t s, uint32_t e, uint32_t m)
{
    return static_cast<uint32_t>(((s) << FP32_SIGN_INDEX) | ((e) << FP32_MAN_LEN) | ((m)&FP32_MAX_MAN));
}

/**
 * @ingroup fp64 basic parameter
 * @brief   fp64 exponent bias
 */
constexpr uint64_t FP64_EXP_BIAS = 1023;

/**
 * @ingroup fp64 basic parameter
 * @brief   the mantissa bit length of double/fp64 is 52
 */
constexpr uint64_t FP64_MAN_LEN = 52;
/**
 * @ingroup fp64 basic parameter
 * @brief   bit index of sign in double/fp64 is 63
 */
constexpr uint64_t FP64_SIGN_INDEX = 63;
/**
 * @ingroup fp64 basic parameter
 * @brief   sign mask of fp64                 (1 000                   (total 63bits 0))
 */
constexpr uint64_t FP64_SIGN_MASK = 0x8000000000000000LLu;
/**
 * @ingroup fp64 basic parameter
 * @brief   exponent mask of fp64            (0 1 11111 11111  0000?-?-(total 52bits 0))
 */
constexpr uint64_t FP64_EXP_MASK = 0x7FF0000000000000LLu;
/**
 * @ingroup fp64 basic parameter
 * @brief   mantissa mask of fp64            (                 1111?-?-(total 52bits 1))
 */
constexpr uint64_t FP64_MAN_MASK = 0x000FFFFFFFFFFFFFLLu;
/**
 * @ingroup fp64 basic parameter
 * @brief   conceal bit of mantissa of fp64     (               1 0000?-?-(total 52bits 0))
 */
constexpr uint64_t FP64_MAN_HIDE_BIT = 0x0010000000000000LLu;

/**
 * @ingroup integer special value judgment
 * @brief   maximum positive value of int8_t            (0111 1111)
 */
constexpr int8_t INT8_T_MAX = 0x7F;
/**
 * @ingroup integer special value judgment
 * @brief   maximum value of a data with 8 bits length  (1111 111)
 */
constexpr uint8_t BIT_LEN8_MAX = 0xFF;
/**
 * @ingroup integer special value judgment
 * @brief   maximum positive value of int16_t           (0111 1111 1111 1111)
 */
constexpr int16_t INT16_T_MAX = 0x7FFF;
/**
 * @ingroup integer special value judgment
 * @brief   maximum value of a data with 16 bits length (1111 1111 1111 1111)
 */
constexpr uint16_t BIT_LEN16_MAX = 0xFFFF;
/**
 * @ingroup integer special value judgment
 * @brief   maximum positive value of int32_t           (0111 1111 1111 1111 1111 1111 1111 1111)
 */
constexpr int32_t INT32_T_MAX = 0x7FFFFFFFu;
/**
 * @ingroup integer special value judgment
 * @brief   maximum value of a data with 32 bits length (1111 1111 1111 1111 1111 1111 1111 1111)
 */
constexpr uint32_t BIT_LEN32_MAX = 0xFFFFFFFFu;
/**
 * @ingroup print switch
 * @brief   print an error if input fp16 is overflow
 */

/**
 * @ingroup fp16_t enum
 * @brief   round mode of last valid digital
 */
enum class Fp16RoundMode : int {
    ROUND_TO_NEAREST = 0,   /**< round to nearest even */
    ROUND_BY_TRUNCATED = 1, /**< round by truncated    */
    ROUND_MODE_RESERVED = 2,
};

/**
 * @ingroup fp16_t global filed
 * @brief   round mode of last valid digital
 */
constexpr Fp16RoundMode g_RoundMode = Fp16RoundMode::ROUND_TO_NEAREST;

/**
 * @ingroup fp16_t public method
 * @param [in]     negative sign is negative
 * @param [in|out] man      mantissa to be reverse
 * @brief   Calculate a mantissa's complement (add ont to it's radix-minus-one complement)
 * @return  Return complement of man
 */
template <typename T>
static void ReverseMan(bool negative, T* man)
{
    if (negative) {
        *man = (~(*man)) + 1;
    }
}

/**
 * @ingroup fp16_t public method
 * @param [in] man   mantissa to be operate
 * @param [in] shift right shift bits
 * @brief   right shift a mantissa
 * @return  Return right-shift mantissa
 */
template <typename T>
static auto RightShift(T man, int16_t shift) -> T
{
    unsigned int bits = static_cast<unsigned int>(sizeof(T) * 8U);
    T mask = (static_cast<T>(1U)) << (static_cast<T>(bits - 1U));
    for (int i = 0; i < shift; i++) {
        man = ((man & mask) | (man >> 1));
    }
    return man;
}

/**
 * @ingroup fp16_t public method
 * @param [in] ea exponent of one temp fp16_t number
 * @param [in] ma mantissa of one temp fp16_t number
 * @param [in] eb exponent of another temp fp16_t number
 * @param [in] mb mantissa of another temp fp16_t number
 * @brief   Get mantissa sum of two temp fp16_t numbers, T support types: uint16_t/uint32_t/uint64_t
 * @return  Return mantissa sum
 */
template <typename T>
static auto GetManSum(int16_t ea, const T& ma, int16_t eb, const T& mb) -> T
{
    T sum = 0;
    if (ea != eb) {
        T m_tmp = 0;
        int16_t e_tmp = std::abs(ea - eb);
        if (ea > eb) {
            m_tmp = mb;
            m_tmp = RightShift(m_tmp, e_tmp);
            sum = ma + m_tmp;
        } else {
            m_tmp = ma;
            m_tmp = RightShift(m_tmp, e_tmp);
            sum = m_tmp + mb;
        }
    } else {
        sum = ma + mb;
    }
    return sum;
}

/**
 * @ingroup fp16_t public method
 * @param [in] bit0    whether the last preserved bit is 1 before round
 * @param [in] bit1    whether the abbreviation's highest bit is 1
 * @param [in] bitLeft whether the abbreviation's bits which not contain highest bit grater than 0
 * @param [in] man     mantissa of a fp16_t or float number, support types: uint16_t/uint32_t/uint64_t
 * @param [in] shift   abbreviation bits
 * @brief    Round fp16_t or float mantissa to nearest value
 * @return   Returns true if round 1,otherwise false;
 */
template <typename T>
static auto ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, T man, uint16_t shift = 0) -> T
{
    man = (man >> shift) + ((bit1 && (bitLeft || bit0)) ? 1 : 0);
    return man;
}

/**
 * @ingroup fp16_t public method
 * @param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
 * @brief   Get bit length of a uint32_t number
 * @return  Return bit length of man
 */
template <typename T>
static int16_t GetManBitLength(T man)
{
    int16_t len = 0;
    while (man) {
        man >>= 1;
        len++;
    }
    return len;
}

/**
 * @ingroup fp16_t public method
 * @param [in]     val signature is negative
 * @param [in|out] s   sign of fp16_t object
 * @param [in|out] e   exponent of fp16_t object
 * @param [in|out] m   mantissa of fp16_t object
 * @brief   Extract the sign, exponent and mantissa of a fp16_t object, if num is denormal, convert to normal
 */
static void ExtractFP16Norm(const uint16_t& val, uint16_t* s, int16_t* e, uint16_t* m)
{
    // 1.Extract
    *s = Fp16ExtractSign(val);
    *e = Fp16ExtractExp(val);
    *m = Fp16ExtractMan(val);

    // Denormal
    if (0 == (*e)) {
        *e = 1;
    }
}

/**
 * @ingroup fp16_t public method
 * @param [in]     val signature is negative
 * @param [in|out] s   sign of fp16_t object
 * @param [in|out] e   exponent of fp16_t object
 * @param [in|out] m   mantissa of fp16_t object
 * @brief   Extract the sign, exponent and mantissa of a fp16_t object, if num is denormal, convert to normal
 */
static void ExtractFP16(const uint16_t& val, uint8_t* s, uint16_t* e, uint16_t* m)
{
    *s = Fp16ExtractSign(val);
    *e = Fp16ExtractExp(val);
    *m = Fp16ExtractMan(val);
}
/**
 * @ingroup fp16_t static method
 * @param [in] man       truncated mantissa
 * @param [in] shiftOut left shift bits based on ten bits
 * @brief   judge whether to add one to the result while converting fp16_t to other datatype
 * @return  Return true if add one, otherwise false
 */
static bool IsRoundOne(uint64_t man, uint16_t truncLen)
{
    uint16_t shiftOut = truncLen - FP16_BIT_SHIFT_LEN_2;
    uint64_t mask0 = 0x4;
    mask0 = mask0 << shiftOut;
    uint64_t mask1 = 0x2;
    mask1 = mask1 << shiftOut;
    uint64_t mask2 = mask1 - 1;

    bool lastBit = ((man & mask0) > 0);
    bool truncHigh = false;
    bool truncLeft = false;
    if (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) {
        truncHigh = ((man & mask1) > 0);
        truncLeft = ((man & mask2) > 0);
    }
    return (truncHigh && (truncLeft || lastBit));
}

/**
 * @ingroup fp16_t public method
 * @param [in] exp       exponent of fp16_t value
 * @param [in] man       exponent of fp16_t value
 * @brief   normalize fp16_t value
 * @return
 */
static void Fp16Normalize(int16_t& exp, uint16_t& man)
{
    if (exp >= FP16_MAX_EXP) {
        exp = FP16_MAX_EXP - 1;
        man = FP16_MAX_MAN;
    } else if (exp == 0 && man == FP16_MAN_HIDE_BIT) {
        exp++;
        man = 0;
    }
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to float/fp32
 * @return  Return float/fp32 value of fpVal which is the value of fp16_t object
 */
static float fp16ToFloat(const uint16_t& fpVal)
{
    float ret;

    uint16_t hfSign, hfMan;
    int16_t hfExp;
    ExtractFP16Norm(fpVal, &hfSign, &hfExp, &hfMan);

    while (hfMan != 0 && (hfMan & FP16_MAN_HIDE_BIT) == 0) {
        hfMan <<= 1;
        hfExp--;
    }

    uint32_t sRet, eRet, mRet, fVal;

    sRet = hfSign;
    if (hfMan == 0) {
        eRet = 0;
        mRet = 0;
    } else {
        eRet = static_cast<uint32_t>(hfExp - FP16_EXP_BIAS + FP32_EXP_BIAS);
        mRet = hfMan & FP16_MAN_MASK;
        mRet = mRet << (FP32_MAN_LEN - FP16_MAN_LEN);
    }
    fVal = Fp32Constructor(sRet, eRet, mRet);
    uint32_t* ptrFVal = &fVal;
    float* ptrRet = reinterpret_cast<float*>(ptrFVal);
    ret = *ptrRet;

    return ret;
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to double/fp64
 * @return  Return double/fp64 value of fpVal which is the value of fp16_t object
 */
static double fp16ToDouble(const uint16_t& fpVal)
{
    double ret;

    uint16_t hfSign, hfMan;
    int16_t hfExp;
    ExtractFP16Norm(fpVal, &hfSign, &hfExp, &hfMan);

    while (hfMan != 0 && (hfMan & FP16_MAN_HIDE_BIT) == 0) {
        hfMan <<= 1;
        hfExp--;
    }

    uint64_t sRet, eRet, mRet, fVal;

    sRet = hfSign;
    if (hfMan == 0) {
        eRet = 0;
        mRet = 0;
    } else {
        eRet = static_cast<uint64_t>(hfExp - FP16_EXP_BIAS + FP64_EXP_BIAS);
        mRet = hfMan & FP16_MAN_MASK;
        mRet = mRet << (FP64_MAN_LEN - FP16_MAN_LEN);
    }
    fVal = (sRet << FP64_SIGN_INDEX) | (eRet << FP64_MAN_LEN) | (mRet);
    uint64_t* ptrFVal = &fVal;
    double* ptrRet = reinterpret_cast<double*>(ptrFVal);
    ret = *ptrRet;

    return ret;
}

static inline void Fp16AlignExp4Int8(uint8_t sRet, uint16_t hfE, uint64_t& longIntM, uint8_t& overflowFlag,
                                     uint16_t& shiftOut)
{
    while (hfE != FP16_EXP_BIAS) {
        if (hfE > FP16_EXP_BIAS) {
            hfE--;
            longIntM = longIntM << 1;
            if (sRet == 1 && longIntM >= 0x20000u) { // sign=1,negative number(<0)
                longIntM = 0x20000u;                 // 10 0000 0000 0000 0000  10(fp16_t-man)+7(int8)=17bit
                overflowFlag = 1;
                break;
            } else if (sRet != 1 && longIntM >= 0x1FFFFu) { // sign=0,positive number(>0)
                longIntM = 0x1FFFFu;                        // 01 1111 1111 1111 1111  10(fp16_t-man)+7(int8)
                overflowFlag = 1;
                break;
            }
        } else {
            hfE++;
            shiftOut++;
        }
    }
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int8_t
 * @return  Return int8_t value of fpVal which is the value of fp16_t object
 */
static int8_t fp16ToInt8(const uint16_t& fpVal)
{
    int8_t ret;
    uint8_t retV;
    uint8_t sRet;
    uint8_t mRet = 0;
    uint16_t hfE, hfM;

    ExtractFP16(fpVal, &sRet, &hfE, &hfM);

    if (Fp16IsDenorm(fpVal)) { // Denormalized number
        return 0;
    }

    uint16_t shiftOut = 0;
    uint8_t overflowFlag = 0;
    uint64_t longIntM = hfM;

    if (Fp16IsInvalid(fpVal)) { // Inf or NaN
        overflowFlag = 1;
    } else {
        Fp16AlignExp4Int8(sRet, hfE, longIntM, overflowFlag, shiftOut);
    }
    if (overflowFlag != 0) {
        retV = INT8_T_MAX + sRet;
    } else {
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint8_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN8_MAX);
        needRound = needRound && ((sRet == 0 && mRet < INT8_T_MAX) || (sRet == 1 && mRet <= INT8_T_MAX));
        if (needRound) {
            mRet++;
        }
        if (sRet != 0) {
            mRet = (~mRet) + 1;
        }
        if (mRet == 0) {
            sRet = 0;
        }
        // Generate final result
        retV = (sRet << FP16_BIT_SHIFT_LEN_7) | (mRet);
    }

    ret = *reinterpret_cast<int8_t*>(&retV);
    return ret;
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint8_t
 * @return  Return uint8_t value of fpVal which is the value of fp16_t object
 */
static uint8_t fp16ToUInt8(const uint16_t& fpVal)
{
    uint8_t sRet;
    uint8_t mRet = 0;
    uint16_t hfE, hfM;

    if (Fp16IsDenorm(fpVal)) { // Denormalized number
        return 0;
    }

    if (Fp16IsInvalid(fpVal)) { // Inf or NaN
        return ~0;
    }

    ExtractFP16(fpVal, &sRet, &hfE, &hfM);
    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;
    uint8_t overflowFlag = 0;

    while (hfE != FP16_EXP_BIAS) {
        if (hfE > FP16_EXP_BIAS) {
            hfE--;
            longIntM = longIntM << 1;
            if (longIntM >= 0x40000Lu) { // overflow 0100 0000 0000 0000 0000
                longIntM = 0x3FFFFLu;    // 11 1111 1111 1111 1111   10(fp16_t-man)+8(uint8)=18bit
                overflowFlag = 1;
                mRet = ~0;
                break;
            }
        } else {
            hfE++;
            shiftOut++;
        }
    }
    if (overflowFlag == 0) {
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint8_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN8_MAX);
        if (needRound && mRet != BIT_LEN8_MAX) {
            mRet++;
        }
    }

    if (sRet == 1) { // Negative number
        mRet = 0;
    }

    return mRet;
}

static inline void Fp16AlignExp4Int16(uint8_t sRet, uint16_t hfE, uint64_t& longIntM, uint8_t& overflowFlag,
                                      uint16_t& shiftOut)
{
    while (hfE != FP16_EXP_BIAS) {
        if (hfE > FP16_EXP_BIAS) {
            hfE--;
            longIntM = longIntM << 1;
            if (sRet == 1 && longIntM > 0x2000000Lu) { // sign=1,negative number(<0)
                longIntM = 0x2000000Lu;                // 10(fp16_t-man)+15(int16)=25bit
                overflowFlag = 1;
                break;
            } else if (sRet != 1 && longIntM >= 0x1FFFFFFLu) { // sign=0,positive number(>0) Overflow
                longIntM = 0x1FFFFFFLu;                        // 10(fp16_t-man)+15(int16)=25bit
                overflowFlag = 1;
                break;
            }
        } else {
            hfE++;
            shiftOut++;
        }
    }
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int16_t
 * @return  Return int16_t value of fpVal which is the value of fp16_t object
 */
static int16_t fp16ToInt16(const uint16_t& fpVal)
{
    int16_t ret;
    uint16_t retV;
    uint8_t sRet;
    uint16_t mRet = 0;
    uint16_t hfE, hfM;

    ExtractFP16(fpVal, &sRet, &hfE, &hfM);

    if (Fp16IsDenorm(fpVal)) { // Denormalized number
        return 0;
    }

    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;
    uint8_t overflowFlag = 0;

    if (Fp16IsInvalid(fpVal)) { // Inf or NaN
        overflowFlag = 1;
    } else {
        Fp16AlignExp4Int16(sRet, hfE, longIntM, overflowFlag, shiftOut);
    }
    if (overflowFlag == 1) {
        retV = INT16_T_MAX + sRet;
    } else {
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint16_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
        if (needRound && mRet < INT16_T_MAX) {
            mRet++;
        }
        if (sRet != 0) {
            mRet = (~mRet) + 1;
        }
        if (mRet == 0) {
            sRet = 0;
        }
        // Generate final result
        retV = (sRet << FP16_BIT_SHIFT_LEN_15) | (mRet);
    }

    ret = *reinterpret_cast<int16_t*>(&retV);
    return ret;
}

static inline void Fp16AlignExp(uint16_t hfE, uint16_t& shiftOut, uint64_t& longIntM)
{
    while (hfE != FP16_EXP_BIAS) {
        if (hfE > FP16_EXP_BIAS) {
            hfE--;
            longIntM = longIntM << 1;
        } else {
            hfE++;
            shiftOut++;
        }
    }
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint16_t
 * @return  Return uint16_t value of fpVal which is the value of fp16_t object
 */
static uint16_t fp16ToUInt16(const uint16_t& fpVal)
{
    uint8_t sRet = 0;
    uint16_t mRet = 0;
    uint16_t hfE, hfM;

    if (Fp16IsDenorm(fpVal)) { // Denormalized number
        return 0;
    }
    if (Fp16IsInvalid(fpVal)) { // Inf or NaN
        mRet = ~0;
        return mRet;
    }
    ExtractFP16(fpVal, &sRet, &hfE, &hfM);
    uint16_t shiftOut = 0;
    uint64_t longIntM = hfM;
    Fp16AlignExp(hfE, shiftOut, longIntM);
    bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
    mRet = static_cast<uint16_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN16_MAX);
    if (needRound && mRet != BIT_LEN16_MAX) {
        mRet++;
    }

    if (sRet == 1) { // Negative number
        mRet = 0;
    }

    return mRet;
}

/**
 * @ingroup fp16_t math convertion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to int32_t
 * @return  Return int32_t value of fpVal which is the value of fp16_t object
 */
static int32_t fp16ToInt32(const uint16_t& fpVal)
{
    int32_t ret;
    uint32_t retV;
    uint8_t sRet = 0;
    uint32_t mRet;
    uint16_t hfE, hfM;

    ExtractFP16(fpVal, &sRet, &hfE, &hfM);
    if (Fp16IsInvalid(fpVal)) { // Inf or NaN
        retV = INT32_T_MAX + sRet;
    } else {
        uint16_t shiftOut = 0;
        uint64_t longIntM = hfM;
        Fp16AlignExp(hfE, shiftOut, longIntM);
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint32_t>((longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN32_MAX);
        if (needRound && mRet < INT32_T_MAX) {
            mRet++;
        }

        if (sRet == 1) {
            mRet = (~mRet) + 1;
        }
        if (mRet == 0) {
            sRet = 0;
        }
        // Generate final result
        retV = (sRet << FP16_BIT_SHIFT_LEN_31) | (mRet);
    }

    ret = *reinterpret_cast<int32_t*>(&retV);
    return ret;
}

/**
 * @ingroup fp16_t math conversion static method
 * @param [in] fpVal uint16_t value of fp16_t object
 * @brief   Convert fp16_t to uint32_t
 * @return  Return uint32_t value of fpVal which is the value of fp16_t object
 */
static uint32_t fp16ToUInt32(const uint16_t& fpVal)
{
    uint32_t ret;
    uint8_t sRet;
    uint32_t mRet;
    uint16_t hfE, hfM;

    ExtractFP16(fpVal, &sRet, &hfE, &hfM);
    if (Fp16IsDenorm(fpVal)) { // Denormalized number
        return 0u;
    }
    if (Fp16IsInvalid(fpVal)) { // Inf or NaN
        mRet = ~0u;
    } else {
        uint16_t shiftOut = 0;
        uint64_t longIntM = hfM;
        Fp16AlignExp(hfE, shiftOut, longIntM);
        bool needRound = IsRoundOne(longIntM, shiftOut + FP16_MAN_LEN);
        mRet = static_cast<uint32_t>(longIntM >> (FP16_MAN_LEN + shiftOut)) & BIT_LEN32_MAX;
        if (needRound && mRet != BIT_LEN32_MAX) {
            mRet++;
        }
    }

    if (sRet == 1) { // Negative number
        mRet = 0;
    }
    // Generate final result
    ret = mRet;

    return ret;
}

static void NormalizeMantissa(uint16_t& mRet, int16_t& eRet, uint32_t& mTrunc, uint16_t shiftOut)
{
    uint16_t mMin = static_cast<uint16_t>(FP16_MAN_HIDE_BIT << shiftOut);
    uint16_t mMax = mMin << 1;
    // Denormal
    while (mRet < mMin && eRet > 0) { // the value of mRet should not be smaller than 2^23
        mRet = mRet << 1;
        mRet += (FP32_SIGN_MASK & mTrunc) >> FP32_SIGN_INDEX;
        mTrunc = mTrunc << 1;
        eRet = eRet - 1;
    }
    while (mRet >= mMax) { // the value of mRet should be smaller than 2^24
        mTrunc = mTrunc >> 1;
        mTrunc = mTrunc | (FP32_SIGN_MASK * (mRet & 1));
        mRet = mRet >> 1;
        eRet = eRet + 1;
    }

    bool bLastBit = ((mRet & 1) > 0);
    bool bTruncHigh = 0;
    bool bTruncLeft = 0;
    bTruncHigh = (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_SIGN_MASK) > 0);
    bTruncLeft = (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_ABS_MAX) > 0);
    mRet = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mRet, shiftOut);
    while (mRet >= mMax) {
        mRet = mRet >> 1;
        eRet = eRet + 1;
    }

    if (eRet == 0 && mRet <= mMax) {
        mRet = mRet >> 1;
    }
    Fp16Normalize(eRet, mRet);
}

/**
 * @ingroup fp16_t math operator
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t addition
 * @return  Return fp16_t result of adding this and fp
 */
static uint16_t fp16Add(uint16_t v1, uint16_t v2)
{
    uint16_t ret;
    uint16_t sA, sB;
    int16_t eA, eB;
    uint32_t mA, mB;
    uint16_t sRet, mRet;
    int16_t eRet;
    uint16_t maTmp, mbTmp;

    uint16_t shiftOut = 0;
    // 1.Extract
    ExtractFP16Norm(v1, &sA, &eA, &maTmp);
    ExtractFP16Norm(v2, &sB, &eB, &mbTmp);
    mA = maTmp;
    mB = mbTmp;

    uint16_t sum;
    if (sA != sB) {
        ReverseMan(sA > 0, &mA);
        ReverseMan(sB > 0, &mB);
        sum = static_cast<uint16_t>(GetManSum(eA, mA, eB, mB));
        sRet = (sum & FP16_SIGN_MASK) >> FP16_SIGN_INDEX;
        ReverseMan(sRet > 0, &mA);
        ReverseMan(sRet > 0, &mB);
    } else {
        sum = static_cast<uint16_t>(GetManSum(eA, mA, eB, mB));
        sRet = sA;
    }

    if (sum == 0) {
        shiftOut = FP16_BIT_SHIFT_LEN_3;
        mA = mA << shiftOut;
        mB = mB << shiftOut;
    }

    int16_t eTmp = 0;
    uint32_t mTrunc = 0;

    eRet = std::max(eA, eB);
    eTmp = std::abs(eA - eB);
    if (eA > eB) {
        mTrunc = (mB << (FP16_BIT_SHIFT_LEN_32 - static_cast<uint16_t>(eTmp)));
        mB = RightShift(mB, eTmp);
    } else if (eA < eB) {
        mTrunc = (mA << (FP16_BIT_SHIFT_LEN_32 - static_cast<uint16_t>(eTmp)));
        mA = RightShift(mA, eTmp);
    }
    // calculate mantissa
    mRet = static_cast<uint16_t>(mA + mB);

    NormalizeMantissa(mRet, eRet, mTrunc, shiftOut);

    ret = Fp16Constructor(sRet, static_cast<uint16_t>(eRet), mRet);
    return ret;
}

/**
 * @ingroup fp16_t math operator
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t subtraction
 * @return  Return fp16_t result of subtraction fp from this
 */
static uint16_t fp16Sub(uint16_t v1, uint16_t v2)
{
    uint16_t ret;
    uint16_t tmp;

    // Reverse
    tmp = ((~(v2)) & FP16_SIGN_MASK) | (v2 & FP16_ABS_MAX);

    ret = fp16Add(v1, tmp);

    return ret;
}

/**
 * @ingroup fp16_t math operator
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t multiplication
 * @return  Return fp16_t result of multiplying this and fp
 */
static uint16_t fp16Mul(uint16_t v1, uint16_t v2)
{
    uint16_t sA, sB;
    int16_t eA, eB;
    uint32_t mA, mB;
    uint16_t sRet, mRet;
    int16_t eRet;
    uint32_t mulM;
    uint16_t maTmp, mbTmp;
    // 1.Extract
    ExtractFP16Norm(v1, &sA, &eA, &maTmp);
    ExtractFP16Norm(v2, &sB, &eB, &mbTmp);
    mA = maTmp;
    mB = mbTmp;

    eRet = eA + eB - FP16_EXP_BIAS - FP16_BIT_SHIFT_LEN_10;
    mulM = mA * mB;
    sRet = sA ^ sB;

    uint32_t mMin = FP16_MAN_HIDE_BIT;
    uint32_t mMax = mMin << 1;
    uint32_t mTrunc = 0;
    // the value of mRet should not be smaller than 2^23
    while (mulM < mMin && eRet > 1) {
        mulM = mulM << 1;
        eRet = eRet - 1;
    }
    while (mulM >= mMax || eRet < 1) {
        mTrunc = mTrunc >> 1;
        mTrunc = mTrunc | (FP32_SIGN_MASK * (mulM & 1));
        mulM = mulM >> 1;
        eRet = eRet + 1;
    }
    bool bLastBit = ((mulM & 1) > 0);
    bool bTruncHigh = (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_SIGN_MASK) > 0);
    bool bTruncLeft = (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) && ((mTrunc & FP32_ABS_MAX) > 0);
    mulM = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mulM);

    while (mulM >= mMax || eRet < 0) {
        mulM = mulM >> 1;
        eRet = eRet + 1;
    }

    if (eRet == 1 && mulM < FP16_MAN_HIDE_BIT) {
        eRet = 0;
    }
    mRet = static_cast<uint16_t>(mulM);

    Fp16Normalize(eRet, mRet);

    uint16_t ret = Fp16Constructor(sRet, static_cast<uint16_t>(eRet), mRet);
    return ret;
}

/**
 * @ingroup fp16_t math operator divided
 * @param [in] v1 left operator value of fp16_t object
 * @param [in] v2 right operator value of fp16_t object
 * @brief   Performing fp16_t division
 * @return  Return fp16_t result of division this by fp
 */
static uint16_t fp16Div(uint16_t v1, uint16_t v2)
{
    uint16_t ret;
    if (Fp16IsZero(v2)) { // result is inf
        // throw "fp16_t division by zero.";
        uint16_t sA, sB;
        uint16_t sRet;
        sA = Fp16ExtractSign(v1);
        sB = Fp16ExtractSign(v2);
        sRet = sA ^ sB;
        ret = Fp16Constructor(sRet, FP16_MAX_EXP - 1, FP16_MAX_MAN);
    } else if (Fp16IsZero(v1)) {
        ret = 0u;
    } else {
        uint16_t sA, sB;
        int16_t eA, eB;
        uint64_t mA, mB;
        float mDiv;
        uint16_t maTmp, mbTmp;
        // 1.Extract
        ExtractFP16Norm(v1, &sA, &eA, &maTmp);
        ExtractFP16Norm(v2, &sB, &eB, &mbTmp);
        mA = maTmp;
        mB = mbTmp;

        uint64_t mTmp;
        if (eA > eB) {
            mTmp = mA;
            uint16_t tmp = static_cast<uint16_t>(eA - eB);
            for (int i = 0; i < tmp; i++) {
                mTmp = mTmp << 1;
            }
            mA = mTmp;
        } else if (eA < eB) {
            mTmp = mB;
            uint16_t tmp = static_cast<uint16_t>(eB - eA);
            for (int i = 0; i < tmp; i++) {
                mTmp = mTmp << 1;
            }
            mB = mTmp;
        }
        mDiv = static_cast<float>(mA * 1.0f / mB);
        fp16_t fpDiv = mDiv;
        ret = fpDiv.val;
        if (sA != sB) {
            ret |= FP16_SIGN_MASK;
        }
    }
    return ret;
}

// operate
fp16_t fp16_t::operator+(const fp16_t fp) const
{
    uint16_t retVal = fp16Add(val, fp.val);
    fp16_t ret(retVal);
    return ret;
}
fp16_t fp16_t::operator-(const fp16_t fp) const
{
    uint16_t retVal = fp16Sub(val, fp.val);
    fp16_t ret(retVal);
    return ret;
}
fp16_t fp16_t::operator*(const fp16_t fp) const
{
    uint16_t retVal = fp16Mul(val, fp.val);
    fp16_t ret(retVal);
    return ret;
}
fp16_t fp16_t::operator/(const fp16_t fp) const
{
    uint16_t retVal = fp16Div(val, fp.val);
    fp16_t ret(retVal);
    return ret;
}

fp16_t fp16_t::operator+=(const fp16_t fp)
{
    val = fp16Add(val, fp.val);
    return *this;
}
fp16_t fp16_t::operator-=(const fp16_t fp)
{
    val = fp16Sub(val, fp.val);
    return *this;
}
fp16_t fp16_t::operator*=(const fp16_t fp)
{
    val = fp16Mul(val, fp.val);
    return *this;
}
fp16_t fp16_t::operator/=(const fp16_t fp)
{
    val = fp16Div(val, fp.val);
    return *this;
}

// compare
bool fp16_t::operator==(const fp16_t& fp) const
{
    bool result = true;
    if (Fp16IsZero(val) && Fp16IsZero(fp.val)) {
        result = true;
    } else {
        result = ((val & BIT_LEN16_MAX) == (fp.val & BIT_LEN16_MAX)); // bit compare
    }
    return result;
}
bool fp16_t::operator!=(const fp16_t& fp) const
{
    bool result = true;
    if (Fp16IsZero(val) && Fp16IsZero(fp.val)) {
        result = false;
    } else {
        result = ((val & BIT_LEN16_MAX) != (fp.val & BIT_LEN16_MAX)); // bit compare
    }
    return result;
}
bool fp16_t::operator>(const fp16_t& fp) const
{
    uint16_t sA, sB;
    uint16_t eA, eB;
    uint16_t mA, mB;
    bool result = true;

    // 1.Extract
    sA = Fp16ExtractSign(val);
    sB = Fp16ExtractSign(fp.val);
    eA = Fp16ExtractExp(val);
    eB = Fp16ExtractExp(fp.val);
    mA = Fp16ExtractMan(val);
    mB = Fp16ExtractMan(fp.val);

    // Compare
    if ((sA == 0) && (sB > 0)) { // +  -
        // -0=0
        result = !(Fp16IsZero(val) && Fp16IsZero(fp.val));
    } else if ((sA == 0) && (sB == 0)) { // + +
        if (eA > eB) {                   // eA - eB >= 1; Va always larger than Vb
            result = true;
        } else if (eA == eB) {
            result = mA > mB;
        } else {
            result = false;
        }
    } else if ((sA > 0) && (sB > 0)) { // - -    opposite to  + +
        if (eA < eB) {
            result = true;
        } else if (eA == eB) {
            result = mA < mB;
        } else {
            result = false;
        }
    } else { // -  +
        result = false;
    }

    return result;
}
bool fp16_t::operator>=(const fp16_t& fp) const
{
    bool result = true;
    if ((*this) > fp) {
        result = true;
    } else if ((*this) == fp) {
        result = true;
    } else {
        result = false;
    }

    return result;
}
bool fp16_t::operator<(const fp16_t& fp) const
{
    bool result = true;
    if ((*this) >= fp) {
        result = false;
    } else {
        result = true;
    }

    return result;
}
bool fp16_t::operator<=(const fp16_t& fp) const
{
    bool result = true;
    if ((*this) > fp) {
        result = false;
    } else {
        result = true;
    }

    return result;
}

// evaluation
fp16_t& fp16_t::operator=(const fp16_t& fp)
{
    if (this == &fp) {
        return *this;
    }
    val = fp.val;
    return *this;
}

fp16_t& fp16_t::operator=(const float& fVal)
{
    uint16_t sRet, manRet;
    int16_t eRet;
    uint32_t eF, mF;
    uint32_t ui32V = *reinterpret_cast<const uint32_t*>(&fVal); // 1:8:23bit sign:exp:man
    uint32_t mLenDelta;

    sRet = static_cast<uint16_t>((ui32V & FP32_SIGN_MASK) >> FP32_SIGN_INDEX); // 4Byte->2Byte
    eF = (ui32V & FP32_EXP_MASK) >> FP32_MAN_LEN;                              // 8 bit exponent
    mF = (ui32V & FP32_MAN_MASK); // 23 bit mantissa dont't need to care about denormal
    mLenDelta = FP32_MAN_LEN - FP16_MAN_LEN;

    bool needRound = false;
    // Exponent overflow/NaN converts to signed inf/NaN
    if (eF > 0x8Fu) { // 0x8Fu:142=127+15
        eRet = FP16_MAX_EXP - 1;
        manRet = FP16_MAX_MAN;
    } else if (eF <= 0x70u) { // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
        eRet = 0;
        if (eF >= 0x67) { // 0x67:103=127-24 Denormal
            mF = (mF | FP32_MAN_HIDE_BIT);
            uint16_t shiftOut = FP32_MAN_LEN;
            uint64_t mTmp = static_cast<uint64_t>(mF) << (eF - 0x67);

            needRound = IsRoundOne(mTmp, shiftOut);
            manRet = static_cast<uint16_t>(mTmp >> shiftOut);
            if (needRound) {
                manRet++;
            }
        } else if (eF == 0x66 && mF > 0) { // 0x66:102 Denormal 0<f_v<min(Denormal)
            manRet = 1;
        } else {
            manRet = 0;
        }
    } else { // Regular case with no overflow or underflow
        eRet = static_cast<int16_t>(eF - 0x70u);

        needRound = IsRoundOne(mF, mLenDelta);
        manRet = static_cast<uint16_t>(mF >> mLenDelta);
        if (needRound) {
            manRet++;
        }
        if ((manRet & FP16_MAN_HIDE_BIT) != 0) {
            eRet++;
        }
    }

    Fp16Normalize(eRet, manRet);
    val = Fp16Constructor(sRet, static_cast<uint16_t>(eRet), manRet);
    return *this;
}

static inline uint16_t EncodeFp16FromUint8(uint16_t sign, uint16_t mantissa)
{
    uint16_t exp = 0;
    if (mantissa != 0) {
        exp = FP16_MAN_LEN;
        while ((mantissa & FP16_MAN_HIDE_BIT) == 0) {
            mantissa <<= 1;
            exp -= 1;
        }
        exp += FP16_EXP_BIAS;
    }
    return Fp16Constructor(sign, exp, mantissa);
}

fp16_t& fp16_t::operator=(const int8_t& iVal)
{
    uint16_t sRet = static_cast<uint16_t>((static_cast<uint8_t>(iVal) & 0x80) >> FP16_BIT_SHIFT_LEN_7);
    uint16_t mRet = static_cast<uint16_t>((static_cast<uint8_t>(iVal) & INT8_T_MAX));

    if (sRet != 0) {                                  // negative number(<0)
        mRet = static_cast<uint16_t>(std::abs(iVal)); // complement
    }

    val = EncodeFp16FromUint8(sRet, mRet);
    return *this;
}

fp16_t& fp16_t::operator=(const uint8_t& uiVal)
{
    uint16_t sRet = 0;
    uint16_t mRet = uiVal;

    val = EncodeFp16FromUint8(sRet, mRet);
    return *this;
}
static inline uint16_t EncodeFp16FromUint16(uint16_t sign, uint16_t mRet)
{
    if (mRet == 0) {
        return 0;
    }
    int16_t eRet;
    uint16_t mMin = FP16_MAN_HIDE_BIT;
    uint16_t mMax = mMin << 1;
    uint16_t len = static_cast<uint16_t>(GetManBitLength(mRet));
    if (len > FP16_BIT_SHIFT_LEN_11) {
        eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
        uint32_t mTrunc;
        uint32_t truncMask = 1;
        uint16_t eTmp = len - FP16_BIT_SHIFT_LEN_11;
        for (int i = 1; i < eTmp; i++) {
            truncMask = (truncMask << 1) + 1;
        }
        mTrunc = (mRet & truncMask) << (FP16_BIT_SHIFT_LEN_32 - eTmp);
        for (int i = 0; i < eTmp; i++) {
            mRet = (mRet >> 1);
            eRet = eRet + 1;
        }
        bool bLastBit = ((mRet & 1) > 0);
        bool bTruncHigh = false;
        bool bTruncLeft = false;
        if (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) { // trunc
            bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
            bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
        }
        mRet = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mRet);
        while (mRet >= mMax || eRet < 0) {
            mRet = mRet >> 1;
            eRet = eRet + 1;
        }
    } else {
        eRet = FP16_EXP_BIAS;
        mRet = mRet << (FP16_BIT_SHIFT_LEN_11 - len);
        eRet = eRet + (len - 1);
    }
    return Fp16Constructor(sign, static_cast<uint16_t>(eRet), mRet);
}

fp16_t& fp16_t::operator=(const int16_t& iVal)
{
    if (iVal == 0) {
        val = 0;
        return *this;
    }
    uint16_t sRet;
    uint16_t uiVal = *reinterpret_cast<const uint16_t*>(&iVal);
    sRet = static_cast<uint16_t>(uiVal >> FP16_BIT_SHIFT_LEN_15);
    if (sRet != 0) {
        int16_t iValM = -iVal;
        uiVal = *reinterpret_cast<uint16_t*>(&iValM);
    }
    uint32_t mTmp = (uiVal & FP32_ABS_MAX);
    val = EncodeFp16FromUint16(sRet, mTmp);

    return *this;
}

fp16_t& fp16_t::operator=(const uint16_t& uiVal)
{
    val = EncodeFp16FromUint16(0U, uiVal);
    return *this;
}

static inline uint16_t EncodeFp16FromUint32(uint16_t sign, uint32_t mTmp)
{
    if (mTmp == 0) {
        return 0;
    }
    int16_t eRet;
    uint32_t mMin = FP16_MAN_HIDE_BIT;
    uint32_t mMax = mMin << 1;
    uint16_t len = static_cast<uint16_t>(GetManBitLength(mTmp));
    if (len > FP16_BIT_SHIFT_LEN_11) {
        eRet = FP16_EXP_BIAS + FP16_MAN_LEN;
        uint32_t mTrunc = 0;
        uint32_t truncMask = 1;
        uint16_t eTmp = len - FP16_BIT_SHIFT_LEN_11;
        for (int i = 1; i < eTmp; i++) {
            truncMask = (truncMask << 1) + 1;
        }
        mTrunc = (mTmp & truncMask) << (FP16_BIT_SHIFT_LEN_32 - eTmp);
        for (int i = 0; i < eTmp; i++) {
            mTmp = (mTmp >> 1);
            eRet = eRet + 1;
        }
        bool bLastBit = ((mTmp & 1) > 0);
        bool bTruncHigh = false;
        bool bTruncLeft = false;
        if (Fp16RoundMode::ROUND_TO_NEAREST == g_RoundMode) { // trunc
            bTruncHigh = ((mTrunc & FP32_SIGN_MASK) > 0);
            bTruncLeft = ((mTrunc & FP32_ABS_MAX) > 0);
        }
        mTmp = ManRoundToNearest(bLastBit, bTruncHigh, bTruncLeft, mTmp);
        while (mTmp >= mMax || eRet < 0) {
            mTmp = mTmp >> 1;
            eRet = eRet + 1;
        }
        if (eRet >= FP16_MAX_EXP) {
            eRet = FP16_MAX_EXP - 1;
            mTmp = FP16_MAX_MAN;
        }
    } else {
        eRet = FP16_EXP_BIAS;
        mTmp = mTmp << (FP16_BIT_SHIFT_LEN_11 - len);
        eRet = eRet + (len - 1);
    }
    uint16_t mRet = static_cast<uint16_t>(mTmp);
    return Fp16Constructor(sign, static_cast<uint16_t>(eRet), mRet);
}

fp16_t& fp16_t::operator=(const int32_t& iVal)
{
    if (iVal == 0) {
        val = 0;
        return *this;
    }
    uint32_t uiVal = *reinterpret_cast<const uint32_t*>(&iVal);
    uint16_t sRet = static_cast<uint16_t>(uiVal >> FP16_BIT_SHIFT_LEN_31);
    if (sRet != 0) {
        int32_t iValM = -iVal;
        uiVal = *reinterpret_cast<uint32_t*>(&iValM);
    }

    uint32_t mTmp = (uiVal & FP32_ABS_MAX);
    val = EncodeFp16FromUint32(sRet, mTmp);

    return *this;
}

fp16_t& fp16_t::operator=(const uint32_t& uiVal)
{
    val = EncodeFp16FromUint32(0U, uiVal);
    return *this;
}

fp16_t& fp16_t::operator=(const double& dVal)
{
    uint16_t sRet, mRet;
    int16_t eRet;
    uint64_t eD, mD;
    uint64_t ui64V = *reinterpret_cast<const uint64_t*>(&dVal); // 1:11:52bit sign:exp:man
    uint32_t mLenDelta;

    sRet = static_cast<uint16_t>((ui64V & FP64_SIGN_MASK) >> FP64_SIGN_INDEX); // 4Byte
    eD = (ui64V & FP64_EXP_MASK) >> FP64_MAN_LEN;                              // 10 bit exponent
    mD = (ui64V & FP64_MAN_MASK);                                              // 52 bit mantissa
    mLenDelta = FP64_MAN_LEN - FP16_MAN_LEN;

    bool needRound = false;
    // Exponent overflow/NaN converts to signed inf/NaN
    if (eD >= 0x410u) { // 0x410:1040=1023+16
        eRet = FP16_MAX_EXP - 1;
        mRet = FP16_MAX_MAN;
        val = Fp16Constructor(sRet, static_cast<uint16_t>(eRet), mRet);
    } else if (eD <= 0x3F0u) { // Exponent underflow converts to denormalized half or signed zero
        // 0x3F0:1008=1023-15
        /**
         * Signed zeros, denormalized floats, and floats with small
         * exponents all convert to signed zero half precision.
         */
        eRet = 0;
        if (eD >= 0x3E7u) { // 0x3E7u:999=1023-24 Denormal
            // Underflows to a denormalized value
            mD = (FP64_MAN_HIDE_BIT | mD);
            uint16_t shiftOut = FP64_MAN_LEN;
            uint64_t mTmp = (static_cast<uint64_t>(mD)) << (eD - 0x3E7u);

            needRound = IsRoundOne(mTmp, shiftOut);
            mRet = static_cast<uint16_t>(mTmp >> shiftOut);
            if (needRound) {
                mRet++;
            }
        } else if (eD == 0x3E6u && mD > 0) {
            mRet = 1;
        } else {
            mRet = 0;
        }
    } else { // Regular case with no overflow or underflow
        eRet = static_cast<int16_t>(eD - 0x3F0u);

        needRound = IsRoundOne(mD, mLenDelta);
        mRet = static_cast<uint16_t>(mD >> mLenDelta);
        if (needRound) {
            mRet++;
        }
        if ((mRet & FP16_MAN_HIDE_BIT) != 0) {
            eRet++;
        }
    }

    Fp16Normalize(eRet, mRet);
    val = Fp16Constructor(sRet, static_cast<uint16_t>(eRet), mRet);
    return *this;
}

// convert
fp16_t::operator float() const { return fp16ToFloat(val); }
fp16_t::operator double() const { return fp16ToDouble(val); }
fp16_t::operator int8_t() const { return fp16ToInt8(val); }
fp16_t::operator uint8_t() const { return fp16ToUInt8(val); }
fp16_t::operator int16_t() const { return fp16ToInt16(val); }
fp16_t::operator uint16_t() const { return fp16ToUInt16(val); }
fp16_t::operator int32_t() const { return fp16ToInt32(val); }
fp16_t::operator uint32_t() const { return fp16ToUInt32(val); }
int fp16_t::IsInf() const
{
    if ((val & FP16_ABS_MAX) == FP16_EXP_MASK) {
        if ((val & FP16_SIGN_MASK) != 0) {
            return -1;
        } else {
            return 1;
        }
    } else {
        return 0;
    }
}

float fp16_t::toFloat() { return fp16ToFloat(val); }
double fp16_t::toDouble() { return fp16ToDouble(val); }
int8_t fp16_t::toInt8() { return fp16ToInt8(val); }
uint8_t fp16_t::toUInt8() { return fp16ToUInt8(val); }
int16_t fp16_t::toInt16() { return fp16ToInt16(val); }
uint16_t fp16_t::toUInt16() { return fp16ToUInt16(val); }
int32_t fp16_t::toInt32() { return fp16ToInt32(val); }
uint32_t fp16_t::toUInt32() { return fp16ToUInt32(val); }
} // namespace Base
} // namespace Ops
