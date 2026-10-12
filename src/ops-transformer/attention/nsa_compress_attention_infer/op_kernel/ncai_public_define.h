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
 * \file ncai_public_define.h
 * \brief
 */

#ifndef NCAI_PUBLIC_DEFINE_H
#define NCAI_PUBLIC_DEFINE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace NSA_COMPRESS_ATTENTION_INFER {

constexpr uint32_t BASE_L1_BLOCK_ELEM_NUM_QKV = 128 * 192;
constexpr uint32_t BASE_L0C_BLOCK_ELEM_NUM = 128 * 128;
constexpr uint32_t BASE_L0AB_BLOCK_ELEM_NUM = 128 * 128;
constexpr uint32_t BASE_L1_BUF_ADDR_OFFSET = 128 * 192 * 2 * 2;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t SOFTMAX_BLOCK_SIZE = 32;
constexpr int32_t MM1_READY = 0;
constexpr int32_t SOFTMAX_READY = 1;
constexpr int32_t MM2_READY = 2;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint64_t SYNC_MODE2 = 2;
constexpr uint32_t BASE_TOPK_ELEM_NUM_OFFSET = 8192;

enum class LAYOUT : uint8_t {
    TND = 0,
    BSND
};

template <typename Q_T, typename KV_T, typename OUT_T, LAYOUT LAYOUT_T = LAYOUT::TND, typename... Args>
struct NCAIType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr LAYOUT layout = LAYOUT_T;
};

} // namespace NSA_COMPRESS_ATTENTION_INFER
#endif // NCAI_PUBLIC_DEFINE_H
