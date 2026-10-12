/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file slogdet_tiling_data.h
 * \brief Slogdet TilingData 结构体定义（现代 registry-invoke 规范：普通 C++ struct，无废弃宏）。
 *        host 侧 context->GetTilingData<SlogdetTilingData>() 写入，
 *        kernel 侧 REGISTER_TILING_DEFAULT(SlogdetTilingData) + GET_TILING_DATA_WITH_STRUCT 读取。
 */

#ifndef SLOGDET_TILING_DATA_H
#define SLOGDET_TILING_DATA_H

#include <cstdint>

struct SlogdetTilingData {
    uint32_t matSizeN = 0;          // 方阵维度 n（kernel 消费）
    uint64_t matrixNumCount = 0;    // batch 内方阵总数（∏ self.shape[:-2]，kernel 消费）
    // blockSize/blockNum 为 kernel 当前未消费的预留/占位字段。
    //   BLOCKED 实际用编译期常量 SLOGDET_ROW_BLOCK=16 做行块化消元（见 op_kernel/slogdet.h），
    //   不读 tiling 下发的 blockSize(=COL_BLOCK 64)/blockNum(=1)。保留为结构体末尾 ABI 占位 +
    //   tiling 侧记账（host 写入、UT 断言），便于后续 BLOCKED 列块调参时复用。
    uint32_t blockSize = 0;         // [预留/kernel 未消费] BLOCKED 列块宽度（host=COL_BLOCK）；FULL=n
    uint32_t blockNum = 0;          // [预留/kernel 未消费] 列分块数（恒 1）
    float epsSingular = 0.f;        // 奇异判定阈值绝对下限 floor（kernel 消费，相对阈值 maxPiv 在 kernel 算）
};

#endif // SLOGDET_TILING_DATA_H
