/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef EXPERIMENTAL_IM2COL_TILING_DATA_H_
#define EXPERIMENTAL_IM2COL_TILING_DATA_H_

#include <cstddef>
#include <cstdint>

constexpr uint32_t IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS = 5120;
constexpr uint32_t IM2COL_CHANNEL_INDEX_TEMPLATE_NONE = 0U;
constexpr uint32_t IM2COL_CHANNEL_INDEX_TEMPLATE_UINT32 = 1U;
constexpr uint32_t IM2COL_CHANNEL_INDEX_TEMPLATE_INT16 = 2U;
constexpr uint32_t IM2COL_CHANNEL_INDEX_TEMPLATE_UINT8 = 3U;
constexpr uint32_t IM2COL_TILING_FLAG_DISABLED = 0U;
constexpr uint32_t IM2COL_TILING_FLAG_ENABLED = 1U;

struct Im2colTilingHeader {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t kernelH;
    int64_t kernelW;
    int64_t strideH;
    int64_t strideW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t padTop;
    int64_t padBottom;
    int64_t padLeft;
    int64_t padRight;
    int64_t outH;
    int64_t outW;
    int64_t totalRows;
    int64_t totalGroups;
    int64_t totalInputElements;
    int64_t totalOutputElements;
    int64_t baseRowsPerCore;
    int64_t extraRows;
    int64_t baseGroupsPerCore;
    int64_t extraGroups;
    int64_t totalChannels;
    int64_t baseChannelsPerCore;
    int64_t extraChannels;
    int64_t tileElements;
    int64_t rawElements;
    int64_t batchRows;
    int64_t outRowStrideElements;
    int64_t rawRowStrideElements;
    int64_t groupBatch;
    int64_t channelBatch;
    int64_t rawChannelStrideElements;
    int64_t outputChannelElements;
    int64_t outputGroupStrideElements;
    int64_t outputChannelStrideElements;
    int64_t rawInputBaseElements;
    uint32_t outBufferBytes;
    uint32_t rawBufferBytes;
    uint32_t indexBufferBytes;
    uint32_t outWideBufferBytes;
    uint32_t rawWideBufferBytes;
    uint32_t fastGroup;
    uint32_t fastChannel;
    uint32_t channelIdentity;
    uint32_t channelFlatGather;
    uint32_t channelContiguousRaw;
    uint32_t channelIndexTemplateValid;
    uint32_t channelIndexTemplateElements;
};

struct Im2colTilingData {
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;
    int64_t kernelH;
    int64_t kernelW;
    int64_t strideH;
    int64_t strideW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t padTop;
    int64_t padBottom;
    int64_t padLeft;
    int64_t padRight;
    int64_t outH;
    int64_t outW;
    int64_t totalRows;
    int64_t totalGroups;
    int64_t totalInputElements;
    int64_t totalOutputElements;
    int64_t baseRowsPerCore;
    int64_t extraRows;
    int64_t baseGroupsPerCore;
    int64_t extraGroups;
    int64_t totalChannels;
    int64_t baseChannelsPerCore;
    int64_t extraChannels;
    int64_t tileElements;
    int64_t rawElements;
    int64_t batchRows;
    int64_t outRowStrideElements;
    int64_t rawRowStrideElements;
    int64_t groupBatch;
    int64_t channelBatch;
    int64_t rawChannelStrideElements;
    int64_t outputChannelElements;
    int64_t outputGroupStrideElements;
    int64_t outputChannelStrideElements;
    int64_t rawInputBaseElements;
    uint32_t outBufferBytes;
    uint32_t rawBufferBytes;
    uint32_t indexBufferBytes;
    uint32_t outWideBufferBytes;
    uint32_t rawWideBufferBytes;
    uint32_t fastGroup;
    uint32_t fastChannel;
    uint32_t channelIdentity;
    uint32_t channelFlatGather;
    uint32_t channelContiguousRaw;
    uint32_t channelIndexTemplateValid;
    uint32_t channelIndexTemplateElements;
    uint32_t channelIndexTemplate[IM2COL_CHANNEL_INDEX_TEMPLATE_ELEMENTS];
};

static_assert(sizeof(Im2colTilingHeader) == offsetof(Im2colTilingData, channelIndexTemplate));

#endif // EXPERIMENTAL_IM2COL_TILING_DATA_H_
