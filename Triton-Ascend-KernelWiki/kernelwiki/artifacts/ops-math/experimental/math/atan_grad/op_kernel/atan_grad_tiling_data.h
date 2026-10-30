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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file atan_grad_tiling_data.h
 * \brief AtanGrad TilingData 结构体定义
 */

#ifndef _ATAN_GRAD_TILING_DATA_H_
#define _ATAN_GRAD_TILING_DATA_H_

struct AtanGradTilingData {
    int64_t totalNum = 0;    // 总元素数量
    int64_t blockFactor = 0; // 每个 AI Core 负责的元素数量（多核切分粒度）
    int64_t ubFactor = 0;    // 每次 UB 循环处理的元素数量（UB 切分粒度）
};

#endif // _ATAN_GRAD_TILING_DATA_H_
