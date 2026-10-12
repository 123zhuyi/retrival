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
 * \file cube1.h
 * \brief Formula: s = q * k^T
 * l0_a： dimG * dimDqk * sizof(T1) (0-6k)
 * l0_b： selectedBlockSize * dimDqk * sizof(T1) (0-24k / 24k-48k)
 * l0_c： dimG * selectedBlockSize * sizof(T1) (0-4k / 64k-68k)
 */

template <typename T1>
__aicore__ inline __attribute__((always_inline)) void
CubeOp<T1>::cube1Process(const int64_t queryGmOffset, const int64_t queryRopeGmOffset, const int64_t keyGmOffset, const int64_t indicesGmOffset,
                         const int64_t outGmOffset, const int32_t blkCntOffset, const int32_t mmPingPongIdx, const RunInfo &runInfo)
{
    uint32_t dLoopTimes = (dimDTotal + 127) / K_SPLIT_SIZE;
    uint32_t perLoopDSize = K_SPLIT_SIZE;
    uint32_t tailLoopDSize = dimDTotal - (dLoopTimes - 1) * perLoopDSize;
    uint32_t blockOffset = N_SPLIT_SIZE / selectedBlockSize;

    MMParam mmParam;
    mmParam.singleM = dimG;
    mmParam.dstStride = PER_LOOP_BLOCK_SIZE;

    uint32_t totalSel = selectedCntOffset * selectedBlockSize;
    if (runInfo.isLastBasicBlock) {
        totalSel = totalSel - selectedBlockSize + runInfo.lastBlockSize;
    }

    const bool isSmallS2 = runInfo.isSmallS2;
    const int64_t sparseKeyGmOffset = keyGmOffset;
    const int64_t denseKeyGmOffset = runInfo.keyGmOffset;
    const int64_t denseKeyRopeGmOffset = runInfo.keyRopeGmOffset;
    const int64_t denseOutGmOffset = runInfo.mm12GmOffset;

    for (int32_t nIdx = blkCntOffset; nIdx < blkCntOffset + selectedCntOffset; nIdx+=blockOffset) {
        LocalTensor<float> l0cTensor = cL0TensorPingPong[ping_pong_flag_l0c_ & 1];
        mmParam.isFixOut = false;
        mmParam.singleK = perLoopDSize;

        mmParam.singleN = min(selectedBlockSize * blockOffset, totalSel - (nIdx - blkCntOffset) * selectedBlockSize);

        int64_t mm1WorkspaceGmOffset;
        if (isSmallS2) {
            mm1WorkspaceGmOffset = denseOutGmOffset + (nIdx - blkCntOffset) * selectedBlockSize;
        } else {
            mm1WorkspaceGmOffset = outGmOffset + (nIdx - blkCntOffset) * selectedBlockSize;
        }

        LocalTensor<T1> current_l1_query_tensor, l1_key_tensor;
        int64_t currentQueryOffset, currentKeyOffset;

        // query node @ key node
        for (int32_t dIdx = 0; dIdx < dLoopTimes - 1; dIdx++) {
            mmParam.isOutKFisrt = dIdx == 0;
            // read query from l1
            WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            current_l1_query_tensor = l1_query_tensor[dIdx * dimGAlign * perLoopDSize];
            l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];

            if (isSmallS2) {
                currentKeyOffset = denseKeyGmOffset + (blkCntOffset * dimN2 + nIdx - blkCntOffset) * selectedBlockSizeDqk + dIdx * K_SPLIT_SIZE;
                CopyGmToL1(l1_key_tensor, keyGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDqk);
            } else {
                currentKeyOffset = sparseKeyGmOffset + (nIdx - blkCntOffset) * selectedBlockSizeDtotal + dIdx * K_SPLIT_SIZE;
                CopyGmToL1(l1_key_tensor, selectedKWorkspaceGm[currentKeyOffset], mmParam.singleN, K_SPLIT_SIZE, dimDTotal);
            }

            MmadInnerWithSync<T1>(l0cTensor, current_l1_query_tensor, l1_key_tensor,
                        aL0TensorPingPong, bL0TensorPingPong,
                        mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm1WorkspaceGm[mm1WorkspaceGmOffset]);

            SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
            UpdatePingPongFlag(ping_pong_flag_l1_common_);
        }

        // query rope @ key rope
        mmParam.isFixOut = true;
        mmParam.isOutKFisrt = dLoopTimes == 1;
        mmParam.singleK = tailLoopDSize;
        WaitFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);
        current_l1_query_tensor = l1_query_tensor[(dLoopTimes - 1) * dimGAlign * perLoopDSize];
        l1_key_tensor = l1_common_tensors[ping_pong_flag_l1_common_];

        if (isSmallS2) {
            currentKeyOffset = HAS_ROPE ? denseKeyRopeGmOffset + blkCntOffset * dimN2 * selectedBlockSizeDrope +
                                              (nIdx - blkCntOffset) * selectedBlockSize * dimRope :
                                          denseKeyGmOffset + (blkCntOffset * dimN2 + nIdx - blkCntOffset) * selectedBlockSizeDqk +
                                              (dLoopTimes - 1) * K_SPLIT_SIZE;
            GlobalTensor<T1> kSrcGm = HAS_ROPE ? keyRopeGm[currentKeyOffset] : keyGm[currentKeyOffset];
            int64_t kSrcDstride = HAS_ROPE ? tailLoopDSize : dimDTotal;
            CopyGmToL1(l1_key_tensor, kSrcGm, mmParam.singleN, tailLoopDSize, kSrcDstride);
        } else {
            currentKeyOffset = sparseKeyGmOffset + (nIdx - blkCntOffset) * selectedBlockSizeDtotal + (dLoopTimes - 1) * K_SPLIT_SIZE;
            int64_t srcDstride = HAS_ROPE ? tailLoopDSize : dimDTotal;
            CopyGmToL1(l1_key_tensor, selectedKWorkspaceGm[currentKeyOffset], mmParam.singleN, tailLoopDSize, dimDTotal);
        }

        MmadInnerWithSync<T1>(l0cTensor, current_l1_query_tensor, l1_key_tensor,
                aL0TensorPingPong, bL0TensorPingPong,
                mmParam, ping_pong_flag_l0a_, ping_pong_flag_l0b_, ping_pong_flag_l0c_, true, mm1WorkspaceGm[mm1WorkspaceGmOffset]);
        SetFlag<HardEvent::MTE1_MTE2>(MM_L1_COMMON_EVENTS[ping_pong_flag_l1_common_]);

        UpdatePingPongFlag(ping_pong_flag_l1_common_);
        UpdatePingPongFlag(ping_pong_flag_l0c_);
    }
}