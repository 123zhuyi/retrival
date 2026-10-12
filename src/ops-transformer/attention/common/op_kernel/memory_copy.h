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
 * \file memory_copy.h
   GM->L1
   PA
   PARope
 * \brief
 */
#ifndef MEMMORY_COPY_H
#define MEMMORY_COPY_H
#include "fia_public_define.h"
#include "memcopy/gm_layout.h"
#include "memcopy/parser.h"
#include "memcopy/offset_calculator_v2.h"
#include "memcopy/fa_gm_tensor.h"
#include "memcopy/fa_l1_tensor.h"
#include "memcopy/fa_ub_tensor.h"
#include "memcopy/gm_coord.h"
#include "memcopy/copy_gm_to_l1.h"
#include "memcopy/copy_gm_to_ub.h"
#include "memcopy/copy_ub_to_gm.h"

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode() {
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}
template <FIA_LAYOUT LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode() {
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr GmFormat GetQueryGmFormat() {
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) ||
                  (LAYOUT_T == FIA_LAYOUT::BNSD) ||
                  (LAYOUT_T == FIA_LAYOUT::TND) ||
                  (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::NTD) {
        return GmFormat::NGTD;
    }
}

template <FIA_LAYOUT KV_LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr GmFormat GetKVFormat() {
    if constexpr (PAGE_ATTENTION) {
        static_assert((KV_LAYOUT_T == FIA_LAYOUT::BSH) ||
                      (KV_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                      (KV_LAYOUT_T == FIA_LAYOUT::NZ),
                      "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when PageAttention");
        if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::NZ) {
            return GmFormat::PA_NZ;
        }
    } else {
        static_assert((KV_LAYOUT_T == FIA_LAYOUT::BSH) ||
                      (KV_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                      (KV_LAYOUT_T == FIA_LAYOUT::TND) ||
                      (KV_LAYOUT_T == FIA_LAYOUT::NTD),
                      "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when KV Continuous or TensorList");
        if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
            return GmFormat::BSND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::TND) {
            return GmFormat::TND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::NTD) {
            return GmFormat::NTD;
        }
    }
}

template <FIA_LAYOUT OUT_LAYOUT_T>
__aicore__ inline constexpr GmFormat GetOutGmFormat() {
    static_assert((OUT_LAYOUT_T == FIA_LAYOUT::BSH) ||
                  (OUT_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                  (OUT_LAYOUT_T == FIA_LAYOUT::TND) ||
                  (OUT_LAYOUT_T == FIA_LAYOUT::NTD) ||
                  (OUT_LAYOUT_T == FIA_LAYOUT::NBSD),
                  "Get OutAttention GmFormat fail, OUT_LAYOUT_T is incorrect");
    if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::NTD) {
        return GmFormat::NGTD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::NBSD) {
        return GmFormat::NGBSD;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbFormat GetOutUbFormat() {
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) ||
                  (LAYOUT_T == FIA_LAYOUT::BNSD) ||
                  (LAYOUT_T == FIA_LAYOUT::TND) ||
                  (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr bool IsSupportPse() {
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::BSH) {
        return true;
    } else {
        return false;
    }
}

template <FIA_LAYOUT LAYOUT_T>
__aicore__ inline constexpr UbFormat GetPseUbFormat() {
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) ||
                  (LAYOUT_T == FIA_LAYOUT::BNSD) ||
                  (LAYOUT_T == FIA_LAYOUT::TND) ||
                  (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get PSE UbFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    } else {
        return UbFormat::S1G;
    }
}

// ---------------------------------------Set attention Gm To Zero--------------------------------
template <GmFormat FORMAT, typename OUT_T>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, OffsetCalculator<FORMAT> &offsetCalculator,
                                           GlobalTensor<OUT_T>& attentionOutGm)
{  
    if constexpr (FORMAT == GmFormat::TNGD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(bIdx);
        for (int s1Idx = 0; s1Idx < s1Count; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    }  else if constexpr (FORMAT == GmFormat::NGTD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(bIdx);
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * offsetCalculator.GetDimD(), 0);
        }
    }  else if constexpr (FORMAT == GmFormat::BNGSD) {
        uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, 0, 0); 
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
    }  else if constexpr (FORMAT == GmFormat::BSNGD) {
        uint32_t s1Size = offsetCalculator.GetDimS1();
        for (int s1Idx = 0; s1Idx < s1Size; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);  
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    }  else if constexpr (FORMAT == GmFormat::NGBSD) {
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);  
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideB(), 0);
        }
    } 
}
#endif
