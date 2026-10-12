/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_fractal_z_wino.h"

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace formats {
namespace {
int64_t kCubeN = 16;
int64_t N0_2 = 2;
int64_t N0_8 = 8;

struct FractalZWin0Shape {
    int64_t n_dim;
    int64_t c_dim;
    int64_t h_dim;
    int64_t w_dim;
};

uint32_t TransShapeToFzWino(int64_t c0, const FractalZWin0Shape& shape, std::vector<int64_t>& dst_shape)
{
    int64_t n = shape.n_dim;
    int64_t c = shape.c_dim;
    int64_t h = shape.h_dim;
    int64_t w = shape.w_dim;

    int64_t c1 = Ceil(c, c0);
    int64_t n1 = Ceil(n, kCubeN);

    dst_shape.clear();
    dst_shape.push_back(c1);
    dst_shape.push_back(n1);
    dst_shape.push_back(N0_2);
    dst_shape.push_back(h * w);
    dst_shape.push_back(N0_8);
    dst_shape.push_back(c0);
    if (!IsShapeValid(dst_shape)) {
        KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t TransShapeHwcnToFzWino(const std::vector<int64_t>& src_shape, DataType data_type,
                                std::vector<int64_t>& dst_shape, const int32_t format)
{
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto h = src_shape.at(kHwcnH);
    auto w = src_shape.at(kHwcnW);
    auto c = src_shape.at(kHwcnC);
    auto n = src_shape.at(kHwcnN);
    const FractalZWin0Shape hwcn{n, c, h, w};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToFzWino(c0, hwcn, dst_shape);
}

uint32_t TransShapeNchwToFzWino(const std::vector<int64_t>& src_shape, DataType data_type,
                                std::vector<int64_t>& dst_shape, const int32_t format)
{
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto n = src_shape.at(kNchwN);
    auto c = src_shape.at(kNchwC);
    auto h = src_shape.at(kNchwH);
    auto w = src_shape.at(kNchwW);
    const FractalZWin0Shape nchw{n, c, h, w};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToFzWino(c0, nchw, dst_shape);
}

uint32_t TransShapeFzWinoToFz(const std::vector<int64_t>& src_shape, std::vector<int64_t>& dst_shape)
{
    if (!CheckShapeValid(src_shape, kFracZWinoDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto c1 = src_shape.at(kFracZWinoC1);
    auto n1 = src_shape.at(kFracZWinoN1);
    auto hw = src_shape.at(kFracZWinoHW);
    auto c0 = src_shape.at(kFracZWinoC0);

    dst_shape.clear();
    dst_shape.push_back(hw * c1);
    dst_shape.push_back(n1);
    dst_shape.push_back(kNiSize);
    dst_shape.push_back(c0);
    if (!IsShapeValid(dst_shape)) {
        KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

struct WinoStrides {
    int64_t c1;
    int64_t n1;
    int64_t c_dim;
    int64_t n_dim;
    int64_t c0;
    int64_t h_dim;
    int64_t w_dim;
    int64_t hW;
    int64_t n8C0;
    int64_t hWN8C0;
    int64_t n2HWN8C0;
    int64_t n1N2HWN8C0;
    int64_t data_size;
    Format format_4d;
};

inline void CopyWinoInnerBlock(const TransArgs& args, const WinoStrides& p, int64_t c1i, int64_t n1i, int64_t n2i,
                               int64_t hwi)
{
    for (int64_t n8i = 0; n8i < N0_8; n8i++) {
        for (int64_t c0i = 0; c0i < p.c0; c0i++) {
            if ((((c1i * p.c0) + c0i) >= p.c_dim) || ((n1i * N0_2 * N0_8 + n2i * N0_8 + n8i) >= p.n_dim)) {
                continue;
            }
            const int64_t inx_fz = (c1i * p.n1N2HWN8C0) + (n1i * p.n2HWN8C0) + (n2i * p.hWN8C0) + (hwi * p.n8C0) +
                                   (n8i * p.c0) + c0i;
            int64_t inx_4d = 0;
            if (p.format_4d == FORMAT_HWCN) {
                inx_4d = (hwi * p.c_dim * p.n_dim) + ((c1i * p.c0 + c0i) * p.n_dim) +
                         (n1i * N0_2 * N0_8 + n2i * N0_8 + n8i);
            } else if (p.format_4d == FORMAT_NCHW) {
                inx_4d = (((n1i * N0_2 * N0_8 + n2i * N0_8 + n8i)) * p.c_dim * p.hW) + ((c1i * p.c0 + c0i) * p.hW) +
                         hwi;
            } else {
                KERNEL_LOG_ERROR("format should be NCHW or HWCN, but got [%s]",
                                 FormatToSerialString(p.format_4d).c_str());
                return;
            }
            copy_data(args.data, args.output, inx_4d, inx_fz, p.data_size);
        }
    }
}

uint32_t TransFormatToWINO(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args)
{
    if (format_4d != FORMAT_HWCN && format_4d != FORMAT_NCHW) {
        KERNEL_LOG_ERROR("format should be NCHW or HWCN, but got [%s]", FormatToSerialString(format_4d).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    Format4dBasics b;
    if (Prepare4dFormatBasics(format_4d, shape_4d, args, b) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t c1 = Ceil(b.c_dim, b.c0);
    const int64_t n1 = Ceil(b.n_dim, kCubeN);
    if (b.dst_size == 0) {
        return KERNEL_STATUS_OK;
    }
    if (!BiggerMemSet(args.output, static_cast<size_t>(b.dst_size), 0, static_cast<size_t>(b.dst_size))) {
        KERNEL_LOG_ERROR("BiggerMemSet failed, size [%ld].", b.dst_size);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    const WinoStrides p{c1,
                        n1,
                        b.c_dim,
                        b.n_dim,
                        b.c0,
                        b.h_dim,
                        b.w_dim,
                        b.h_dim * b.w_dim,
                        N0_8 * b.c0,
                        b.h_dim * b.w_dim * N0_8 * b.c0,
                        N0_2 * b.h_dim * b.w_dim * N0_8 * b.c0,
                        n1 * N0_2 * b.h_dim * b.w_dim * N0_8 * b.c0,
                        b.data_size,
                        format_4d};
    for (int64_t c1i = 0; c1i < c1; c1i++) {
        for (int64_t n1i = 0; n1i < n1; n1i++) {
            for (int64_t n2i = 0; n2i < N0_2; n2i++) {
                for (int64_t hwi = 0; hwi < p.hW; hwi++) {
                    CopyWinoInnerBlock(args, p, c1i, n1i, n2i, hwi);
                }
            }
        }
    }
    return KERNEL_STATUS_OK;
}

inline void CopyFzToWinoInnerBlock(const TransArgs& args, int64_t c1i, int64_t n1i, int64_t hwi, int64_t hw, int64_t c0,
                                   int64_t n1N0C0, int64_t n0C0, int64_t n8C0, int64_t hWN8C0, int64_t n2HWN8C0,
                                   int64_t n1N2HWN8C0, int64_t data_size)
{
    for (int64_t n2i = 0; n2i < N0_2; n2i++) {
        for (int64_t n8i = 0; n8i < N0_8; n8i++) {
            for (int64_t c0i = 0; c0i < c0; c0i++) {
                const int64_t inx_fz_wino = (c1i * n1N2HWN8C0) + (n1i * n2HWN8C0) + (n2i * hWN8C0) + (hwi * n8C0) +
                                            (n8i * c0) + c0i;
                const int64_t inx_fz = (((c1i * hw) + hwi) * n1N0C0) + (n1i * n0C0) + ((n2i * N0_8) + n8i) * c0 + c0i;
                copy_data(args.data, args.output, inx_fz, inx_fz_wino, data_size);
            }
        }
    }
}

uint32_t TransFormatFzToWINO(const std::vector<int64_t>& shape_fz_wino, const TransArgs& args)
{
    auto c1 = shape_fz_wino.at(kFracZWinoC1);
    auto n1 = shape_fz_wino.at(kFracZWinoN1);
    auto hw = shape_fz_wino.at(kFracZWinoHW);
    auto c0 = shape_fz_wino.at(kFracZWinoC0);

    const DataType data_type = args.src_data_type;
    int64_t data_size = GetSizeByDataType(data_type);
    int64_t dst_size = GetItemNumByShape(args.dst_shape) * data_size;
    if (dst_size == 0) {
        KERNEL_LOG_INFO("input tensor is empty tensor");
        return KERNEL_STATUS_OK;
    }

    auto ret = BiggerMemSet(args.output, static_cast<size_t>(dst_size), 0, static_cast<size_t>(dst_size));
    if (!ret) {
        KERNEL_LOG_ERROR("memset failed");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t n8C0 = N0_8 * c0;
    const int64_t hWN8C0 = hw * n8C0;
    const int64_t n2HWN8C0 = N0_2 * hWN8C0;
    const int64_t n1N2HWN8C0 = n1 * n2HWN8C0;
    const int64_t n0C0 = N0_2 * n8C0;
    const int64_t n1N0C0 = n1 * n0C0;

    for (int64_t c1i = 0; c1i < c1; c1i++) {
        for (int64_t n1i = 0; n1i < n1; n1i++) {
            for (int64_t hwi = 0; hwi < hw; hwi++) {
                CopyFzToWinoInnerBlock(args, c1i, n1i, hwi, hw, c0, n1N0C0, n0C0, n8C0, hWN8C0, n2HWN8C0, n1N2HWN8C0,
                                       data_size);
            }
        }
    }
    return KERNEL_STATUS_OK;
}
} // namespace

uint32_t FormatTransferFractalZWINO::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    (void)reverse;
    const DataType data_type = args.src_data_type;
    Format src_format = args.src_format;
    std::vector<int64_t> src_shape = args.src_shape;
    // 这里不能使用dst format，因为dst format已经经过了GetPrimaryFormat
    int32_t format = args.output_format;
    if (src_format == FORMAT_HWCN) {
        return TransShapeHwcnToFzWino(src_shape, data_type, dst_shape, format);
    } else if (src_format == FORMAT_NCHW) {
        return TransShapeNchwToFzWino(src_shape, data_type, dst_shape, format);
    } else if (src_format == FORMAT_FRACTAL_Z) {
        src_shape = args.dst_shape;
        return TransShapeFzWinoToFz(src_shape, dst_shape);
    } else {
        KERNEL_LOG_ERROR("src_format[%u] not support", static_cast<uint32_t>(src_format));
        return KERNEL_STATUS_PARAM_INVALID;
    }
}

uint32_t FormatTransferFractalZWINO::TransFormat(const TransArgs& args)
{
    KERNEL_LOG_DEBUG("Begin to trans format from [%s] to [%s], src shape [%s], data type "
                     "[%s], dst "
                     "shape [%s]",
                     FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                     VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                     VectorToString(args.dst_shape).c_str());

    std::vector<int64_t> expect_shape;
    auto ret = TransShape(args, expect_shape);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    if (args.src_format == FORMAT_FRACTAL_Z) {
        if (!IsTransShapeSrcCorrect(args, expect_shape)) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return TransFormatFzToWINO(args.dst_shape, args);
    } else {
        if (!IsTransShapeDstCorrect(args, expect_shape)) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return TransFormatToWINO(args.src_format, args.src_shape, args);
    }
}

REGISTER_FORMAT_TRANSFER(FormatTransferFractalZWINO, FORMAT_NCHW, FORMAT_FRACTAL_Z_WINO)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZWINO, FORMAT_HWCN, FORMAT_FRACTAL_Z_WINO)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZWINO, FORMAT_FRACTAL_Z, FORMAT_FRACTAL_Z_WINO)
} // namespace formats
} // namespace  aicpu
