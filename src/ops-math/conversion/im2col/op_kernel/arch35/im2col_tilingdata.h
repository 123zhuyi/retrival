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
 * \file im2col_tilingdata.h
 * \brief
 */

#ifndef _IM2COL_TILINGDATA_
#define _IM2COL_TILINGDATA_

#include <cstdint>

struct Im2ColCompileInfo {};

struct Im2ColInputInfo {
    int64_t N;
    int64_t C;
    int64_t H;
    int64_t W;
    int64_t hKernelSize;
    int64_t wKernelSize;
    int64_t hStride;
    int64_t wStride;
    int64_t hDilation;
    int64_t wDilation;
    int64_t hPaddingBefore;
    int64_t hPaddingAfter;
    int64_t wPaddingBefore;
    int64_t wPaddingAfter;
};

struct Im2ColNCHWTilingData {
    Im2ColInputInfo input;
    int32_t ubFactorH;             // ub内输出H
    int32_t ubFactorW;             // ub内输出W
    int32_t ubFactorNC;            // ub内NC大小
    int32_t w4ubFactorW;           // ubFactorW 对应的搬入长度burstLen
    int32_t lines4ubFactorW;       // ubFactorW对应的输入H行数
    int32_t lines4ubFactorH;       // ubFactorH对应的输入H行数
    int64_t convKernelNumInWidth;  // 每个W方向卷积核
    int64_t convKernelNumInHeight; // 每个H方向卷积核
    int64_t totalRectAngles;       // 总的矩阵个数
    int32_t rectAnglesPerCore;     // 每个核处理的矩阵个数
    int32_t outHWrectAngles;       // 一个输出HW有多少个UB块
    int32_t inputBufferSize;       // 输入buffer大小
    int32_t outputBufferSize;      // 输出buffer大小
};

struct Im2ColNHWCTilingData {
    Im2ColInputInfo input;
    // min(240KB / 4 , 64KB)
    int32_t ubFactorC;
    int32_t ubFactorW;
    int32_t ubFactorH;
    int32_t ubFactorN;
    int64_t convKernelNumInWidth;  // 每个W方向卷积核
    int64_t convKernelNumInHeight; // 每个H方向卷积核
    int64_t totalLines;            // 总的矩阵个数
    int32_t linesPerCore;          // 每个核处理的矩阵个数
    int32_t outputBufferSize;      // 输出buffer大小
};

struct Im2ColSIMTTilingData {
    Im2ColInputInfo input;
    uint32_t convKernelNumInHeight{0};
    uint32_t convKernelNumInWidth{0};
    uint32_t realCoreNum{0};
    uint32_t blockFactor{0};
    uint32_t blockTailFactor{0};
    uint32_t mainCoreNum{0};
    uint32_t threadNum{0};
};

#endif // _IM2COL_TILINGDATA_
