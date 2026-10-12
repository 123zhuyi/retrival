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
 * \file pfa_policy_data.h
 * \brief
 */
#ifndef PFA_POLICY_DATA_H
#define PFA_POLICY_DATA_H

struct PAFlagData {
    int32_t bIdx : 32;
    uint32_t nIdx : 32;
    uint32_t s2SingleOffset : 32;
    uint64_t tensorBAddr : 64;
    uint64_t blockTableAddr : 64;
    int32_t blockTableDim2 : 32;
    int32_t blockSize : 32;
    uint32_t isLayoutBSH : 32;
    uint32_t kvHeadNum : 32;
    uint32_t kvD : 32;
    int32_t paBlockNumSum : 32;
    uint32_t isbmm1 : 1;
    uint32_t reuseLeft : 1; // 是否复用左矩阵
    uint32_t leftBufIdx : 3; // 左矩阵的buf index
    // m方向是否有additional数据，sameAB模式下TailM只能为偶数
    // 所以当实际M为奇数的时候（TailM=实际M+1），我们要通过这个字段来还原实际的M的值
    uint32_t mOrNAdditionalSize : 1;    
    uint32_t reverse : 26;
};

struct FaPaPolicyData {
    int32_t bIdx;
    uint32_t nIdx;
    uint32_t s2SingleOffset;
    uint64_t tensorBAddr;
    uint64_t blockTableAddr;
    int32_t blockTableDim2;
    int32_t blockSize;
    uint32_t isLayoutBSH;
    uint32_t kvHeadNum;
    uint32_t kvD;
    int32_t paBlockNumSum;
    uint32_t splitD : 1;        // 是否切分D轴
    uint32_t reuseLeft : 1;     // 是否复用左矩阵
    uint32_t leftBufIdx : 3;    // 左矩阵的buf index
    uint32_t rightBufIdx : 3;   // 右矩阵的buf index
    // m方向是否有additional数据，sameAB模式下TailM只能为偶数
    // 所以当实际M为奇数的时候（TailM=实际M+1），我们要通过这个字段来还原实际的M的值
    uint32_t mOrNAdditionalSize : 1;    
    uint32_t reverse : 23;
};

struct PFAMatmulPolicyData {
    uint64_t reuseLeft : 1;  // 是否复用左矩阵
    uint64_t leftBufIdx : 3;  // 左矩阵的buf index
    uint64_t rightBufIdx : 3;  // 右矩阵的buf index
    // m方向是否有additional数据。sameAB模式下TailM只能为偶数，
    // 所以当实际M为奇数的时候(TailM=实际M + 1)，我们要通过这个字段来还原实际M的值
    uint64_t mOrNAdditionalSize : 5;
};

struct IFAMLAMatmulPolicyData {
    uint64_t reuseLeft : 1;  // 是否复用左矩阵
    uint64_t leftBufIdx : 3;  // 左矩阵的buf index
    uint64_t rightBufIdx : 3;  // 右矩阵的buf index
    // m方向是否有additional数据。sameAB模式下TailM只能为偶数，
    // 所以当实际M为奇数的时候(TailM=实际M + 1)，我们要通过这个字段来还原实际M的值
    uint64_t mOrNAdditionalSize : 1;
    uint64_t rRightStride;
    uint64_t rLeftStride;
    uint64_t qRopeAddr;
    uint64_t kRopeAddr;
};

struct IFAMLAPaMatmulPolicyData {
    int32_t bIdx;
    uint32_t nIdx;
    uint32_t s2SingleOffset;
    int32_t blockTableDim2;
    uint32_t rRightStride;
    uint32_t rLeftStride;
    int32_t blockSize;
    uint32_t kvHeadNum;
    uint32_t kvD;
    int32_t paBlockNumSum;
    uint64_t qRopeAddr;
    uint64_t kRopeAddr;
    uint64_t tensorBAddr;
    uint64_t blockTableAddr;
    uint32_t isLayoutBSH;
    uint32_t reuseLeft : 1; // 是否复用左矩阵
    uint32_t leftBufIdx : 3; // 左矩阵的buf index
    // m方向是否有additional数据，sameAB模式下TailM只能为偶数
    // 所以当实际M为奇数的时候（TailM=实际M+1），我们要通过这个字段来还原实际的M的值
    uint32_t mOrNAdditionalSize : 1; 
    uint32_t reverse : 27; 
};

namespace AscendC {
namespace Impl {
namespace Detail {
static constexpr AscendC::TQueConfig TSCM_CONFIG_PFA = {.nd2nz = false,
    .nz2nd = false,
    .scmBlockGroup = true,
    .bufferLen = 0,
    .bufferNumber = 1,
    .consumerSize = 0,
    .consumer = {},
    .enableStaticEvtId = true};

// L1 global variable
struct PFAGlobalTscmArray {
    __aicore__ inline PFAGlobalTscmArray () {};
#ifndef __CCE_KT_TEST__
    AscendC::TSCM<AscendC::TPosition::GM, 1, &TSCM_CONFIG_PFA> localScm[6]; // q k v各两块共6块常驻l1buf
#else
    AscendC::TSCM<TPosition::GM, 1, 0x4> localScm[6]; // q k v各两块共6块常驻l1buf
#endif
};
__BLOCK_LOCAL__ __inline__ PFAGlobalTscmArray* tscmGlobalPFA;
}
}
}
#endif // PFA_POLICY_DATA_H