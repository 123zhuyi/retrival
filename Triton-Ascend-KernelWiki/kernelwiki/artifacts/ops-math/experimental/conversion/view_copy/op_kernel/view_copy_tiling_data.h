/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file view_copy_tiling_data.h
 * \brief ViewCopy tiling data.
 */

#ifndef VIEWCOPY_TILING_DATA_H_
#define VIEWCOPY_TILING_DATA_H_

#include <cstdint>

constexpr int64_t VIEWCOPY_MAX_DIMS = 8;

struct ViewCopyTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int64_t storageNum = 0;
    int64_t srcStorageNum = 0;
    int64_t ndim = 0;
    int64_t metaTypeBytes = 0;
    int64_t viewNum = 0;
    int64_t metadataReady = 0;
    int64_t dstOverlap = 0;
    int64_t dstSpan = 0;
    int64_t srcOffset = 0;
    int64_t dstOffset = 0;
    int64_t sizes[VIEWCOPY_MAX_DIMS] = {0};
    int64_t srcStrides[VIEWCOPY_MAX_DIMS] = {0};
    int64_t dstStrides[VIEWCOPY_MAX_DIMS] = {0};
};

#endif // VIEWCOPY_TILING_DATA_H_
