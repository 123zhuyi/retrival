/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETAIL_GCD_SCALAR_UTILS_H_
#define DETAIL_GCD_SCALAR_UTILS_H_

namespace GcdOp {

__aicore__ inline uint64_t GcdUnsigned64(uint64_t a, uint64_t b)
{
    if (a == 0) {
        return b;
    }
    if (b == 0) {
        return a;
    }
    if (a == b) {
        return a;
    }
    if (a == 1 || b == 1) {
        return 1;
    }
    if (a < b) {
        uint64_t tmp = a;
        a = b;
        b = tmp;
    }
    while (b != 0) {
        uint64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

template <typename T>
__aicore__ inline uint64_t AbsToU64(T value)
{
    int64_t signedValue = static_cast<int64_t>(value);
    if (signedValue >= 0) {
        return static_cast<uint64_t>(signedValue);
    }
    return static_cast<uint64_t>(-(signedValue + 1)) + 1;
}

template <>
__aicore__ inline uint64_t AbsToU64<uint8_t>(uint8_t value)
{
    return static_cast<uint64_t>(value);
}

template <>
__aicore__ inline uint64_t AbsToU64<float>(float value)
{
    union FloatBits {
        float value;
        uint32_t bits;
    } converter = {value};
    uint32_t magnitudeBits = converter.bits & 0x7fffffffU;
    uint32_t exponentBits = (magnitudeBits >> 23) & 0xffU;
    if (exponentBits == 0 || exponentBits == 0xffU) {
        return 0;
    }

    int64_t exponent = static_cast<int64_t>(exponentBits) - 127;
    if (exponent < 0) {
        return 0;
    }
    if (exponent >= 64) {
        return static_cast<uint64_t>(-1);
    }

    uint64_t significand = static_cast<uint64_t>(0x800000U | (magnitudeBits & 0x7fffffU));
    if (exponent >= 23) {
        return significand << (exponent - 23);
    }
    return significand >> (23 - exponent);
}

__aicore__ inline uint64_t AbsBf16BitsToU64(uint16_t bits)
{
    int64_t exponent = static_cast<int64_t>((bits >> 7) & 0xff) - 127;
    uint64_t significand = static_cast<uint64_t>(0x80 | (bits & 0x7f));
    uint64_t magnitude = 0;
    if (exponent >= 7) {
        magnitude = significand << (exponent - 7);
    } else if (exponent >= 0) {
        magnitude = significand >> (7 - exponent);
    }
    return magnitude;
}

__aicore__ inline uint64_t AbsFp16BitsToU64(uint16_t bits)
{
    int64_t exponent = static_cast<int64_t>((bits >> 10) & 0x1f) - 15;
    uint64_t significand = static_cast<uint64_t>(0x400 | (bits & 0x3ff));
    uint64_t magnitude = 0;
    if (exponent >= 10) {
        magnitude = significand << (exponent - 10);
    } else if (exponent >= 0) {
        magnitude = significand >> (10 - exponent);
    }
    return magnitude;
}

__aicore__ inline uint64_t AbsI8BitsToU64(uint8_t bits)
{
    return (bits & 0x80U) != 0 ? static_cast<uint64_t>(0x100U - static_cast<uint32_t>(bits)) :
                                 static_cast<uint64_t>(bits);
}

__aicore__ inline uint64_t AbsI16BitsToU64(uint16_t bits)
{
    return (bits & 0x8000U) != 0 ? static_cast<uint64_t>(0x10000U - static_cast<uint32_t>(bits)) :
                                   static_cast<uint64_t>(bits);
}

__aicore__ inline uint16_t U64ToFloatBits(uint64_t value, int64_t mantissaBits, int64_t exponentBias,
                                          uint64_t overflowSignificand)
{
    if (value == 0) {
        return 0;
    }

    int64_t msb = 0;
    uint64_t tmp = value;
    while ((tmp >> 1) != 0) {
        tmp >>= 1;
        ++msb;
    }

    uint64_t significand = 0;
    int64_t exponent = msb + exponentBias;
    if (msb <= mantissaBits) {
        significand = value << (mantissaBits - msb);
    } else {
        int64_t shift = msb - mantissaBits;
        significand = value >> shift;
        uint64_t remainderMask = (static_cast<uint64_t>(1) << shift) - 1;
        uint64_t remainder = value & remainderMask;
        uint64_t halfway = static_cast<uint64_t>(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (significand & 1) != 0)) {
            ++significand;
        }
        if (significand == overflowSignificand) {
            significand >>= 1;
            ++exponent;
        }
    }
    uint64_t mantissaMask = (static_cast<uint64_t>(1) << mantissaBits) - 1;
    return static_cast<uint16_t>((static_cast<uint64_t>(exponent) << mantissaBits) | (significand & mantissaMask));
}

__aicore__ inline uint16_t U64ToBf16Bits(uint64_t value) { return U64ToFloatBits(value, 7, 127, 0x100); }

__aicore__ inline uint16_t U64ToFp16Bits(uint64_t value) { return U64ToFloatBits(value, 10, 15, 0x800); }

__aicore__ inline float U64ToFp32(uint64_t value)
{
    if (value == 0) {
        return 0.0f;
    }

    int64_t msb = 0;
    uint64_t tmp = value;
    while ((tmp >> 1) != 0) {
        tmp >>= 1;
        ++msb;
    }

    uint64_t significand = 0;
    int64_t exponent = msb + 127;
    if (msb <= 23) {
        significand = value << (23 - msb);
    } else {
        int64_t shift = msb - 23;
        significand = value >> shift;
        uint64_t remainderMask = (static_cast<uint64_t>(1) << shift) - 1;
        uint64_t remainder = value & remainderMask;
        uint64_t halfway = static_cast<uint64_t>(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (significand & 1) != 0)) {
            ++significand;
        }
        if (significand == 0x1000000ULL) {
            significand >>= 1;
            ++exponent;
        }
    }

    union FloatBits {
        uint32_t bits;
        float value;
    } converter = {static_cast<uint32_t>((static_cast<uint64_t>(exponent) << 23) | (significand & 0x7fffffULL))};
    return converter.value;
}

__aicore__ inline uint8_t GcdUint8Bits(uint8_t lhs, uint8_t rhs)
{
    return static_cast<uint8_t>(GcdUnsigned64(AbsToU64<uint8_t>(lhs), AbsToU64<uint8_t>(rhs)));
}

__aicore__ inline uint8_t GcdInt8RawBits(uint8_t lhsBits, uint8_t rhsBits)
{
    uint64_t result = GcdUnsigned64(AbsI8BitsToU64(lhsBits), AbsI8BitsToU64(rhsBits));
    return static_cast<uint8_t>(result);
}

__aicore__ inline uint16_t GcdInt16RawBits(uint16_t lhsBits, uint16_t rhsBits)
{
    uint64_t result = GcdUnsigned64(AbsI16BitsToU64(lhsBits), AbsI16BitsToU64(rhsBits));
    return static_cast<uint16_t>(result);
}

template <typename T>
__aicore__ inline T GcdScalar(T lhs, T rhs)
{
    int64_t result = static_cast<int64_t>(GcdUnsigned64(AbsToU64<T>(lhs), AbsToU64<T>(rhs)));
    return static_cast<T>(result);
}

template <>
__aicore__ inline float GcdScalar<float>(float lhs, float rhs)
{
    uint64_t result = GcdUnsigned64(AbsToU64<float>(lhs), AbsToU64<float>(rhs));
    return U64ToFp32(result);
}

} // namespace GcdOp

#endif // DETAIL_GCD_SCALAR_UTILS_H_
