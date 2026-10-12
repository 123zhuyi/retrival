/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_REGISTER_FORMAT_TRANSFER_H
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_REGISTER_FORMAT_TRANSFER_H

#include <functional>
#include <memory>
#include <vector>

#include "cpu_types.h"
#include "cpu_context.h"

namespace aicpu {
namespace formats {
struct TransArgs {
    const uint8_t* data;
    uint8_t* output;
    // format from GetFormat
    const int32_t input_format;
    const int32_t output_format;
    // primary format
    Format src_format;
    Format dst_format;
    // For scenes that need to supplement the shape, for example, 5D to 4D
    // It is not possible to convert the format normally if you only get the
    // src_shape, and must get the shape before you mend the shape. So the
    // parameters here need to be passed in both src_shape and dst_shape
    std::vector<int64_t> src_shape;
    std::vector<int64_t> dst_shape;
    DataType src_data_type;
    int64_t groups;
    const CpuKernelContext* ctx;
};

class FormatTransfer {
public:
    virtual ~FormatTransfer() = default;
    virtual uint32_t TransFormat(const TransArgs& args) = 0;
    virtual uint32_t TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse) = 0;
};

using FormatTransferBuilder = std::function<std::shared_ptr<FormatTransfer>()>;

class FormatTransferRegister {
public:
    FormatTransferRegister(FormatTransferBuilder builder, Format src, Format dst);
    ~FormatTransferRegister() = default;
};

#define REGISTER_FORMAT_TRANSFER(TransferClass, format1, format2)                      \
    namespace {                                                                        \
    FormatTransferRegister format_transfer_register_##TransferClass##format1##format2( \
        []() { return std::make_shared<TransferClass>(); }, format1, format2);         \
    }

/**
 * Build a FormatTransfer according to 'args'
 * @param args
 * @param result
 * @return
 */
std::shared_ptr<FormatTransfer> BuildFormatTransfer(const TransArgs& args);

bool FormatTransferExists(const TransArgs& args);
} // namespace formats
} // namespace aicpu
#endif
