/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_c1hwc0.h"

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace formats {
namespace {

struct C1hwc0SrcShape {
    int64_t n_dim;
    int64_t c_dim;
    int64_t h_dim;
    int64_t w_dim;
};

uint32_t TransShapeToC1hwc0(int64_t c0, const C1hwc0SrcShape& shape, std::vector<int64_t>& dst_shape)
{
    int64_t n = shape.n_dim;
    int64_t c = shape.c_dim;
    int64_t h = shape.h_dim;
    int64_t w = shape.w_dim;
    // nd to c1hwc0 zhe c dim must 1
    if (c != 1) {
        KERNEL_LOG_ERROR("Check shape failed, c is not one [%ld]", c);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    int64_t c1 = Ceil(n, c0);

    dst_shape.clear();
    dst_shape.push_back(c1);
    dst_shape.push_back(h);
    dst_shape.push_back(w);
    dst_shape.push_back(c0);
    if (!IsShapeValid(dst_shape)) {
        KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t TransShapeHwcnToC1hwc0(const std::vector<int64_t>& src_shape, DataType data_type,
                                std::vector<int64_t>& dst_shape, const int32_t format)
{
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto h = src_shape.at(kHwcnH);
    auto w = src_shape.at(kHwcnW);
    auto c = src_shape.at(kHwcnC);
    auto n = src_shape.at(kHwcnN);
    const C1hwc0SrcShape hwcn{n, c, h, w};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToC1hwc0(c0, hwcn, dst_shape);
}

uint32_t TransShapeNchwToC1hwc0(const std::vector<int64_t>& src_shape, DataType data_type,
                                std::vector<int64_t>& dst_shape, const int32_t format)
{
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto n = src_shape.at(kNchwN);
    auto c = src_shape.at(kNchwC);
    auto h = src_shape.at(kNchwH);
    auto w = src_shape.at(kNchwW);
    const C1hwc0SrcShape nchw{n, c, h, w};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToC1hwc0(c0, nchw, dst_shape);
}

struct C1hwc0StrideParams {
    int64_t c0;
    int64_t c1;
    int64_t n_dim;
    int64_t h;
    int64_t w;
    int64_t w_dim;
    int64_t hW;
    int64_t wC0;
    int64_t hWC0;
    int64_t data_size;
};

inline void CopyHwcnColumn(const TransArgs& args, const C1hwc0StrideParams& p, int64_t c1i, int64_t hi, int64_t wi)
{
    for (int64_t c0i = 0; c0i < p.c0; c0i++) {
        if (((c1i * p.c0) + c0i) >= p.n_dim) {
            break;
        }
        const int64_t idx_c1hwc0 = (c1i * p.hWC0) + (hi * p.wC0) + (wi * p.c0) + c0i;
        const int64_t idx_4d = hi * p.w_dim * p.n_dim + wi * p.n_dim + (c1i * p.c0) + c0i;
        copy_data(args.data, args.output, idx_4d, idx_c1hwc0, p.data_size);
    }
}

inline void CopyNchwColumn(const TransArgs& args, const C1hwc0StrideParams& p, int64_t c1i, int64_t hi, int64_t wi)
{
    for (int64_t c0i = 0; c0i < p.c0; c0i++) {
        if (((c1i * p.c0) + c0i) >= p.n_dim) {
            break;
        }
        const int64_t idx_c1hwc0 = (c1i * p.hWC0) + (hi * p.wC0) + (wi * p.c0) + c0i;
        const int64_t idx_4d = ((c1i * p.c0) + c0i) * p.hW + hi * p.w_dim + wi;
        copy_data(args.data, args.output, idx_4d, idx_c1hwc0, p.data_size);
    }
}

template <typename ColumnFn>
void IterateC1hwc0(const TransArgs& args, const C1hwc0StrideParams& p, ColumnFn column_fn)
{
    for (int64_t c1i = 0; c1i < p.c1; c1i++) {
        for (int64_t hi = 0; hi < p.h; hi++) {
            for (int64_t wi = 0; wi < p.w; wi++) {
                column_fn(args, p, c1i, hi, wi);
            }
        }
    }
}

uint32_t TransFormatToC1hwc0(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args)
{
    Format4dBasics b;
    if (Prepare4dFormatBasics(format_4d, shape_4d, args, b) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (b.dst_size == 0) {
        return KERNEL_STATUS_OK;
    }
    BiggerMemSet(args.output, static_cast<size_t>(b.dst_size), 0, static_cast<size_t>(b.dst_size));

    const C1hwc0StrideParams p{
        b.c0,           Ceil(b.n_dim, b.c0),      b.n_dim,    b.h_dim, b.w_dim, b.w_dim, b.h_dim * b.w_dim,
        b.w_dim * b.c0, b.h_dim * b.w_dim * b.c0, b.data_size};
    if (format_4d == FORMAT_HWCN) {
        IterateC1hwc0(args, p, CopyHwcnColumn);
    } else if (format_4d == FORMAT_NCHW) {
        IterateC1hwc0(args, p, CopyNchwColumn);
    } else {
        KERNEL_LOG_ERROR("format should be NCHW or HWCN, but got [%s]", FormatToSerialString(format_4d).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}
} // namespace

uint32_t FormatTransferC1HWC0::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    (void)reverse;
    const DataType data_type = args.src_data_type;
    Format src_format = args.src_format;
    std::vector<int64_t> src_shape = args.src_shape;
    // 这里不能使用dst format，因为dst format已经经过了GetPrimaryFormat
    int32_t format = args.output_format;
    if (src_format == FORMAT_HWCN) {
        return TransShapeHwcnToC1hwc0(src_shape, data_type, dst_shape, format);
    } else {
        return TransShapeNchwToC1hwc0(src_shape, data_type, dst_shape, format);
    }
}

uint32_t FormatTransferC1HWC0::TransFormat(const TransArgs& args)
{
    KERNEL_LOG_DEBUG("Begin to trans format from [%s] to [%s], src shape [%s], data type "
                     "[%s], dst shape [%s]",
                     FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                     VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                     VectorToString(args.dst_shape).c_str());

    std::vector<int64_t> expect_shape;
    auto ret = TransShape(args, expect_shape);
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Fail trans format from [%s] to [%s], src shape [%s], data type "
                         "[%s], dst shape [%s]",
                         FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                         VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                         VectorToString(args.dst_shape).c_str());
        return ret;
    }
    if (!IsTransShapeDstCorrect(args, expect_shape)) {
        KERNEL_LOG_ERROR("Expect shape not correct trans format from [%s] to [%s], src shape [%s], data type "
                         "[%s], dst shape [%s]",
                         FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                         VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                         VectorToString(args.dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return TransFormatToC1hwc0(args.src_format, args.src_shape, args);
}

REGISTER_FORMAT_TRANSFER(FormatTransferC1HWC0, FORMAT_NCHW, FORMAT_C1HWC0)
REGISTER_FORMAT_TRANSFER(FormatTransferC1HWC0, FORMAT_HWCN, FORMAT_C1HWC0)
} // namespace formats
} // namespace  aicpu
