/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cast.cpp
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/cast_struct.h"
#include "arch35/cast_impl.h"

namespace AscendcCast {
template <int t>
__aicore__ constexpr inline AscendC::Reg::LoadDist ToLoadDist()
{
    if constexpr (t == CAST_MODE_REG_COPYIN_NORM) {
        return AscendC::Reg::LoadDist::DIST_NORM;
    } else if constexpr (t == CAST_MODE_REG_COPYIN_DS_B8) {
        return AscendC::Reg::LoadDist::DIST_DS_B8;
    } else if constexpr (t == CAST_MODE_REG_COPYIN_DS_B16) {
        return AscendC::Reg::LoadDist::DIST_DS_B16;
    } else if constexpr (t == CAST_MODE_REG_COPYIN_UNPACK_B8) {
        return AscendC::Reg::LoadDist::DIST_UNPACK_B8;
    } else if constexpr (t == CAST_MODE_REG_COPYIN_UNPACK_B16) {
        return AscendC::Reg::LoadDist::DIST_UNPACK_B16;
    } else if constexpr (t == CAST_MODE_REG_COPYIN_UNPACK_B32) {
        return AscendC::Reg::LoadDist::DIST_UNPACK_B32;
    } else if constexpr (t == CAST_MODE_REG_COPYIN_UNPACK4_B8) {
        return AscendC::Reg::LoadDist::DIST_UNPACK4_B8;
    }

    return AscendC::Reg::LoadDist::DIST_NORM;
}

template <int t>
__aicore__ constexpr inline AscendC::Reg::StoreDist ToStoreDist()
{
    if constexpr (t == CAST_MODE_REG_COPYOUT_NORM) {
        return AscendC::Reg::StoreDist::DIST_NORM;
    } else if constexpr (t == CAST_MODE_REG_COPYOUT_PACK_B16) {
        return AscendC::Reg::StoreDist::DIST_PACK_B16;
    } else if constexpr (t == CAST_MODE_REG_COPYOUT_PACK_B32) {
        return AscendC::Reg::StoreDist::DIST_PACK_B32;
    } else if constexpr (t == CAST_MODE_REG_COPYOUT_PACK_B64) {
        return AscendC::Reg::StoreDist::DIST_PACK_B64;
    } else if constexpr (t == CAST_MODE_REG_COPYOUT_PACK4_B32) {
        return AscendC::Reg::StoreDist::DIST_PACK4_B32;
    }
    return AscendC::Reg::StoreDist::DIST_NORM;
}

template <int t>
__aicore__ constexpr inline AscendC::RoundMode ToRoundMode()
{
    if constexpr (t == CAST_ROUND_MODE_UNKNOWN) {
        return AscendC::RoundMode::UNKNOWN;
    } else if constexpr (t == CAST_ROUND_MODE_NONE) {
        return AscendC::RoundMode::CAST_NONE;
    } else if constexpr (t == CAST_ROUND_MODE_RINT) {
        return AscendC::RoundMode::CAST_RINT;
    } else if constexpr (t == CAST_ROUND_MODE_FLOOR) {
        return AscendC::RoundMode::CAST_FLOOR;
    } else if constexpr (t == CAST_ROUND_MODE_CEIL) {
        return AscendC::RoundMode::CAST_CEIL;
    } else if constexpr (t == CAST_ROUND_MODE_ROUND) {
        return AscendC::RoundMode::CAST_ROUND;
    } else if constexpr (t == CAST_ROUND_MODE_TRUNC) {
        return AscendC::RoundMode::CAST_TRUNC;
    } else if constexpr (t == CAST_ROUND_MODE_ODD) {
        return AscendC::RoundMode::CAST_ODD;
    } else if constexpr (t == CAST_ROUND_MODE_HYBRID) {
        return AscendC::RoundMode::CAST_HYBRID;
    }
    return AscendC::RoundMode::CAST_NONE;
}
} // namespace AscendcCast

template <int id>
__global__ __aicore__ void cast(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    if constexpr (!(GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::isValid_)) {
        return;
    }
    constexpr int templateId = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::id_;
    constexpr int mapDtypeX = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::srcMapType_;
    constexpr int mapDtypeMid = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::midType_;
    constexpr int mapDtypeY = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::dstMapType_;
    constexpr int castMode1 = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::castMode1_;
    constexpr int castMode2 = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::castMode2_;
    constexpr int regCopyInMode = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::regCopyInMode_;
    constexpr int regCopyOutMode = GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::regCopyOutMode_;
    using dTypeX = typename GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::srcType;
    using dTypeY = typename GetCastPolicy<ORIG_DTYPE_X, ORIG_DTYPE_Y>::dstType;

    using mapInType = typename AscendcCast::TypeGetTool<mapDtypeX>::type;
    using mapMidType = typename AscendcCast::TypeGetTool<mapDtypeMid>::type;
    using mapOutType = typename AscendcCast::TypeGetTool<mapDtypeY>::type;
    if constexpr (templateId == CAST_TEMPLATE_DIRECT_CAST) {
        AscendcCast::CastDirect<mapInType, mapOutType> op;
        constexpr AscendC::RoundMode rMode1 = AscendcCast::ToRoundMode<castMode1>();
        op.Init(x, y, rMode1, &tilingData, &pipe);
        op.Process();
    } else if constexpr (templateId == CAST_TEMPLATE_DST_BOOL) {
#if ORIG_DTYPE_Y == DT_BOOL
        AscendcCast::CastDstBool<dTypeX> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
#endif
    } else if constexpr (templateId == CAST_TEMPLATE_THROUGH) {
        AscendcCast::CastThrough<dTypeY> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (templateId == CAST_TEMPLATE_SRC_UINT1) {
        AscendcCast::CastUint1<dTypeY> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (templateId == CAST_TEMPLATE_TWO_CAST) {
        AscendcCast::CastTwo<mapInType, mapMidType, mapOutType> op;
        constexpr AscendC::RoundMode rMode1 = AscendcCast::ToRoundMode<castMode1>();
        constexpr AscendC::RoundMode rMode2 = AscendcCast::ToRoundMode<castMode2>();
        op.Init(x, y, rMode1, rMode2, &tilingData, &pipe);
        op.Process();
    } else if constexpr (templateId == CAST_TEMPLATE_MIRCRO_INOUT || templateId == CAST_TEMPLATE_MIRCRO_CAST ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_INTER ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_DEINTER ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_CAST ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_DEINTER_CAST ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_CAST_DEINTER_CAST ||
                         templateId == CAST_TEMPLATE_MIRCRO_CAST_INTER_CAST_CAST ||
                         templateId == CAST_TEMPLATE_MIRCRO_DEINTER_SHIFT) {
        constexpr AscendC::Reg::LoadDist ldDist = AscendcCast::ToLoadDist<regCopyInMode>();
        constexpr AscendC::Reg::StoreDist stDist = AscendcCast::ToStoreDist<regCopyOutMode>();
        constexpr AscendC::RoundMode rMode1 = AscendcCast::ToRoundMode<castMode1>();
        constexpr AscendC::RoundMode rMode2 = AscendcCast::ToRoundMode<castMode2>();
        AscendcCast::CastMicro<templateId, dTypeX, dTypeY, mapInType, mapMidType, mapOutType, ldDist, stDist, rMode1,
                               rMode2>
            op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
    return;
}
