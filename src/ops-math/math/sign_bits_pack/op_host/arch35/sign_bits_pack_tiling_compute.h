/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIGN_BITS_PACK_TILING_COMPUTE_H_
#define SIGN_BITS_PACK_TILING_COMPUTE_H_

#include <cstdint>
#include "graph/types.h"
#include "../../op_kernel/arch35/sign_bits_pack_tiling_data.h"

namespace optiling {

struct SignBitsPackTilingInputs {
    int64_t n;
    int64_t sizeAttr;
    int64_t rank;
    ge::DataType dtype;
    uint64_t coreNum;
};

enum class SignBitsPackTilingStatus : int {
    kSuccess = 0,
    kEmptyShortCircuit = 1,
    kDtypeNotSupported = 2,
    kShapeMismatch = 3,
    kAttrOutOfRange = 4
};

SignBitsPackTilingStatus ComputeTilingSignBitsPack(const SignBitsPackTilingInputs& in, SignBitsPackTilingData& out);

int32_t GetTilingKeyForSignBitsPack(SignBitsPackTilingStatus status);

SignBitsPackTilingStatus ComputeBranch1TilingSignBitsPack(const SignBitsPackTilingInputs& in,
                                                          SignBitsPackTilingData& out);

} // namespace optiling

#endif // SIGN_BITS_PACK_TILING_COMPUTE_H_
