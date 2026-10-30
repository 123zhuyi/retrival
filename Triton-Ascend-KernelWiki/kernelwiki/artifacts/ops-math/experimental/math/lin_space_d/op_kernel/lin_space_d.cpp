/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
/*!
 * \file lin_space_d.cpp
 * \brief
 */
#include "lin_space_d.h"

using namespace NsLinSpaceD;
enum class LinSpaceDTilingKey : uint32_t {
    TILING_KEY_TPL_BF16 = 27, // bfloat16
    TILING_KEY_TPL_UINT8 = 4, // uint8
    TILING_KEY_TPL_FP16 = 1,  // float16
    TILING_KEY_TPL_FP32 = 0,  // float32
    TILING_KEY_TPL_INT8 = 2,  // int8
    TILING_KEY_TPL_INT16 = 6, // int16
    TILING_KEY_TPL_INT32 = 3, // int32
};

using BF16 = bfloat16_t;
using UINT8 = uint8_t;
using FP16 = half;
using FP32 = float;
using INT8 = int8_t;
using INT16 = int16_t;
using INT32 = int32_t;

// kernel function
template <uint32_t TYPE_START, uint32_t TYPE_END>
__global__ __aicore__ void lin_space_d(
    GM_ADDR start, GM_ADDR end, GM_ADDR num, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(LinSpaceDTilingData);
    GET_TILING_DATA_WITH_STRUCT(LinSpaceDTilingData, tilingData, tiling);

    // BF16作为起始类型的7种组合
    if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<BF16, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<BF16, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<BF16, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<BF16, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<BF16, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<BF16, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<BF16, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
    // UINT8作为起始类型的7种组合
    else if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<UINT8, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<UINT8, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<UINT8, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<UINT8, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<UINT8, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<UINT8, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<UINT8, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
    // FP16作为起始类型的7种组合
    else if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<FP16, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<FP16, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<FP16, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<FP16, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<FP16, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<FP16, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<FP16, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
    // FP32作为起始类型的7种组合
    else if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<FP32, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<FP32, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<FP32, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<FP32, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<FP32, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<FP32, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<FP32, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
    // INT8作为起始类型的7种组合
    else if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<INT8, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<INT8, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<INT8, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<INT8, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<INT8, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<INT8, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<INT8, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
    // INT16作为起始类型的7种组合
    else if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<INT16, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<INT16, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<INT16, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<INT16, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<INT16, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<INT16, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<INT16, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
    // INT32作为起始类型的7种组合
    else if constexpr (TYPE_START == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
        if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_BF16)) {
            LinSpaceD<INT32, BF16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_UINT8)) {
            LinSpaceD<INT32, UINT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP16)) {
            LinSpaceD<INT32, FP16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_FP32)) {
            LinSpaceD<INT32, FP32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT8)) {
            LinSpaceD<INT32, INT8> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT16)) {
            LinSpaceD<INT32, INT16> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        } else if constexpr (TYPE_END == static_cast<uint32_t>(LinSpaceDTilingKey::TILING_KEY_TPL_INT32)) {
            LinSpaceD<INT32, INT32> op;
            op.Init(start, end, z, &tilingData);
            op.Process();
        }
    }
}
