/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_FORMAT_TRANSFER_FORMAT_TRANSFER_C1HWC0_H
#define AICPU_KERNELS_FORMAT_TRANSFER_FORMAT_TRANSFER_C1HWC0_H

#include "register_format_transfer.h"

namespace aicpu {
namespace formats {
class FormatTransferC1HWC0 : public FormatTransfer {
public:
    uint32_t TransFormat(const TransArgs& args) override;
    uint32_t TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse = false) override;
};
} // namespace formats
} // namespace aicpu

#endif
