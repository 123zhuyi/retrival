/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sign_bits_pack_tiling_compute.h"
#include "util/math_util.h"

namespace optiling {

SignBitsPackTilingStatus ComputeTilingSignBitsPack(const SignBitsPackTilingInputs& in, SignBitsPackTilingData& out)
{
    if (in.n <= 0) {
        return SignBitsPackTilingStatus::kEmptyShortCircuit;
    }

    if (in.dtype != ge::DT_FLOAT16 && in.dtype != ge::DT_FLOAT) {
        return SignBitsPackTilingStatus::kDtypeNotSupported;
    }

    if (in.rank != 1) {
        return SignBitsPackTilingStatus::kShapeMismatch;
    }

    if (in.sizeAttr < 1) {
        return SignBitsPackTilingStatus::kAttrOutOfRange;
    }
    const int64_t packedLen = Ops::Base::CeilDiv(in.n, kPackRate);
    if (packedLen % in.sizeAttr != 0) {
        return SignBitsPackTilingStatus::kAttrOutOfRange;
    }

    const uint32_t dtypeBytes = (in.dtype == ge::DT_FLOAT16) ? 2u : 4u;
    const uint64_t coreNum = (in.coreNum == 0) ? 1u : in.coreNum;
    const uint64_t totalCount = static_cast<uint64_t>(Ops::Base::CeilDiv(in.n, kAlignUnit));
    const uint64_t perCoreCount = (totalCount + coreNum - 1) / coreNum;
    const uint64_t realCoreNum = (perCoreCount == 0) ? 0 : (totalCount + perCoreCount - 1) / perCoreCount;
    const uint64_t tailElemCount = static_cast<uint64_t>(in.n % kAlignUnit);
    const uint64_t tailByteCount = static_cast<uint64_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(tailElemCount), kPackRate));
    const uint32_t mask = 256u / dtypeBytes;
    const uint32_t block = mask / 8u;

    out.rank = static_cast<uint8_t>(1);
    out.inShape[0] = in.n;
    out.outShape[0] = in.sizeAttr;
    out.outShape[1] = packedLen / in.sizeAttr;
    out.totalCount = totalCount;
    out.perCoreCount = perCoreCount;
    out.ubAxis = static_cast<uint8_t>(0);
    out.ubFactor = static_cast<uint32_t>(kAlignUnit);
    out.bufferSize = static_cast<uint32_t>(kAlignUnit) * dtypeBytes;

    out.n = in.n;
    out.sizeAttr = in.sizeAttr;
    out.packedLen = packedLen;
    out.padCount = packedLen * kPackRate - in.n;
    out.mask = mask;
    out.block = block;
    out.tailElemCount = tailElemCount;
    out.tailByteCount = tailByteCount;
    out.realCoreNum = static_cast<uint32_t>(realCoreNum);

    return SignBitsPackTilingStatus::kSuccess;
}

int32_t GetTilingKeyForSignBitsPack(SignBitsPackTilingStatus status)
{
    return (status == SignBitsPackTilingStatus::kSuccess) ? 1 : -1;
}

SignBitsPackTilingStatus ComputeBranch1TilingSignBitsPack(const SignBitsPackTilingInputs& in,
                                                          SignBitsPackTilingData& out)
{
    const uint32_t dtypeBytes = (in.dtype == ge::DT_FLOAT16) ? 2u : 4u;
    const int64_t packedLen = Ops::Base::CeilDiv(in.n, kPackRate);
    const int64_t padCount = packedLen * kPackRate - in.n;
    const uint32_t mask = 256u / dtypeBytes;
    const uint32_t block = mask / 8u;

    const uint64_t coreNum = (in.coreNum == 0) ? 1u : in.coreNum;
    const uint64_t totalCount = static_cast<uint64_t>(Ops::Base::CeilDiv(in.n, kAlignUnit));
    const uint64_t perCoreCount = (totalCount + coreNum - 1) / coreNum;
    const uint64_t realCoreNum = (perCoreCount == 0) ? 0 : (totalCount + perCoreCount - 1) / perCoreCount;

    const uint32_t ubFactor = static_cast<uint32_t>(kAlignUnit);
    const uint32_t bufferSize = static_cast<uint32_t>(kAlignUnit) * dtypeBytes;

    const uint64_t tailElemCount = static_cast<uint64_t>(in.n % kAlignUnit);
    const uint64_t tailByteCount = static_cast<uint64_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(tailElemCount), kPackRate));

    out.rank = static_cast<uint8_t>(1);
    out.inShape[0] = in.n;
    out.outShape[0] = in.sizeAttr;
    out.outShape[1] = packedLen / in.sizeAttr;
    out.totalCount = totalCount;
    out.perCoreCount = perCoreCount;
    out.ubAxis = static_cast<uint8_t>(0);
    out.ubFactor = ubFactor;
    out.bufferSize = bufferSize;

    out.n = in.n;
    out.sizeAttr = in.sizeAttr;
    out.packedLen = packedLen;
    out.padCount = padCount;
    out.mask = mask;
    out.block = block;
    out.tailElemCount = tailElemCount;
    out.tailByteCount = tailByteCount;
    out.realCoreNum = static_cast<uint32_t>(realCoreNum);

    return SignBitsPackTilingStatus::kSuccess;
}

} // namespace optiling
