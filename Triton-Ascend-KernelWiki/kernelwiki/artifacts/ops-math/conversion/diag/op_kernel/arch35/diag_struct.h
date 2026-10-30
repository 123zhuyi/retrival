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
 * \file diag_struct.h
 * \brief Tiling data structure and key definitions for Diag operator arch35
 */

#ifndef DIAG_STRUCT_H
#define DIAG_STRUCT_H

#include <cstdint>

constexpr uint64_t TILING_SIMT = 102UL;

#define TILING_KEY_SIMT 102

struct DiagSimtTilingData {
    int64_t nSize{0};
    int64_t ubSize{0};
    int64_t realCoreNum{0};
    int64_t mainBlockCount{0};
    int64_t mainBlockFactor{0};
    int64_t tailBlockFactor{0};
};

#endif // DIAG_STRUCT_H
