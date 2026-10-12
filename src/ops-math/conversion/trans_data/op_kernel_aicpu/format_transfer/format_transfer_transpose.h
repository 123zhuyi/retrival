/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_TRANSFER_TRANSPOSE_H
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_TRANSFER_TRANSPOSE_H

#include <vector>

#include "register_format_transfer.h"

namespace aicpu {
namespace formats {
uint32_t Transpose(const uint8_t* src, uint8_t* output, const std::vector<int64_t>& src_shape, DataType src_data_type,
                   const std::vector<int64_t>& perm_arg);

uint32_t GetPermByForamt(Format src_format, Format dst_format, std::vector<int64_t>& perm);
class FormatTransferTranspose : public FormatTransfer {
public:
    uint32_t TransFormat(const TransArgs& args) override;
    uint32_t TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse) override;
};
} // namespace formats
} // namespace aicpu

#endif // AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_TRANSFER_H_
