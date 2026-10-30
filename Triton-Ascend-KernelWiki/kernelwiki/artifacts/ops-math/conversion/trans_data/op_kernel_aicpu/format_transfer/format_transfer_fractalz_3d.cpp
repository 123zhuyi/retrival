/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_fractalz_3d.h"

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"

namespace aicpu {
namespace formats {
namespace {
KernelStatus CheckDataTypeSupport(DataType data_type)
{
    return GetSizeByDataType(data_type) > 0 ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is
 * small n and large Z. If 4D(eg.NCHW) is used to represent convolution kernel,
 * N is width, HWC is height.
 *
 * frac_z_3d axises: (C1 * H* W * D, N1, Ni, C0), which Ni = 16, C0 = 16 / 32, No =
 * Ceil(N / Ni), C1 = Ceil(C / C0)
 * @return
 */

uint32_t TransShapeToFz3DWithGroups(int64_t cube_k, const FractalZ3DShape& shape, DataType data_type,
                                    std::vector<int64_t>& dst_shape, int64_t groups)
{
    (void)data_type;
    return BuildFzWithGroupsShape(shape.n_dim, shape.c_dim, shape.d_dim * shape.h_dim * shape.w_dim, cube_k, groups,
                                  dst_shape);
}

uint32_t TransShapeNcdhwToFzWithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                       std::vector<int64_t>& dst_shape, int64_t groups, const int32_t format)
{
    if (!CheckShapeValid(src_shape, static_cast<int64_t>(kNcdhwDimsNum))) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto n = src_shape.at(kNcdhwN);
    auto c = src_shape.at(kNcdhwC);
    auto d = src_shape.at(kNcdhwD);
    auto h = src_shape.at(kNcdhwH);
    auto w = src_shape.at(kNcdhwW);
    const FractalZ3DShape ncdhw{n, c, h, w, d};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToFz3DWithGroups(c0, ncdhw, data_type, dst_shape, groups);
}

uint32_t TransShapeDhwcnToFzWithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                       std::vector<int64_t>& dst_shape, int64_t groups, const int32_t format)
{
    if (!CheckShapeValid(src_shape, static_cast<int64_t>(kDhwcnDimsNum))) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    auto d = src_shape.at(kDhwcnD);
    auto h = src_shape.at(kDhwcnH);
    auto w = src_shape.at(kDhwcnW);
    auto c = src_shape.at(kDhwcnC);
    auto n = src_shape.at(kDhwcnN);
    const FractalZ3DShape dhwcn{n, c, h, w, d};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToFz3DWithGroups(c0, dhwcn, data_type, dst_shape, groups);
}

uint32_t TransShapeNdhwcToFzWithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                       std::vector<int64_t>& dst_shape, int64_t groups, const int32_t format)
{
    if (!CheckShapeValid(src_shape, kNdhwcDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto n = src_shape.at(kNdhwcN);
    auto d = src_shape.at(kNdhwcD);
    auto h = src_shape.at(kNdhwcH);
    auto w = src_shape.at(kNdhwcW);
    auto c = src_shape.at(kNdhwcC);
    const FractalZ3DShape ndhwc{n, c, h, w, d};

    int64_t c0 = GetC0ValueForTransShape(data_type, format);
    KERNEL_CHECK_FALSE((c0 > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].", c0);

    return TransShapeToFz3DWithGroups(c0, ndhwc, data_type, dst_shape, groups);
}

// Supporting NCDHW, DHWCN, NDHWC converte to FORMAT_FRACTAL_Z_3D (GDC1HWN1N0C0),
// the final effect achieved is for the data to be distributed diagonally.
// For example: When the input filter format is NCDHW, calculated the Correspondence of
// index between NCDHW and FORMAT_FRACTAL_Z_3D , then Convert the old filter to the new
// filter, and finally added 0 to the position where there is no data.
struct Fractalz3DCtx {
    int64_t h_dim;
    int64_t w_dim;
    int64_t c_dim;
    int64_t n_dim;
    int64_t d_dim;
    int64_t cin_ori;
    int64_t cout_ori;
    int64_t cube_k;
    int64_t e_mult;
    int64_t cout_opt;
    int64_t c1_dim;
    int64_t data_size;
    int64_t dst_size;
};

uint32_t BuildFractalz3DCtx(const Format& format_5d, const std::vector<int64_t>& shape_5d, const TransArgs& args,
                            Fractalz3DCtx& ctx)
{
    ctx.h_dim = 0;
    ctx.w_dim = 0;
    ctx.c_dim = 0;
    ctx.n_dim = 0;
    ctx.d_dim = 0;
    if (GetFormatDim(ctx.d_dim, ctx.h_dim, ctx.w_dim, ctx.c_dim, ctx.n_dim, format_5d, shape_5d) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ctx.cin_ori = ctx.c_dim;
    // For this place , groups is not equal to 0, which had been checked in [Transdata] entrance.
    ctx.cout_ori = ctx.n_dim / args.groups;
    if (CheckDimOri(ctx.cin_ori, ctx.cout_ori) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const DataType data_type = args.src_data_type;
    ctx.cube_k = GetC0ValueForTransFormat(data_type, args.input_format, args.output_format);
    if (ctx.cube_k == 0) {
        KERNEL_LOG_ERROR("Cube_k must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ctx.e_mult = std::min(Lcm(Lcm(ctx.cin_ori, ctx.cube_k) / (ctx.cin_ori),
                              Lcm(ctx.cout_ori, static_cast<int64_t>(kCubeSize)) / (ctx.cout_ori)),
                          args.groups);
    if (ctx.e_mult == 0) {
        KERNEL_LOG_ERROR("E_mult must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t cin_opt = Ceil(ctx.e_mult * ctx.cin_ori, ctx.cube_k) * ctx.cube_k;
    ctx.cout_opt = Ceil(ctx.e_mult * ctx.cout_ori, static_cast<int64_t>(kCubeSize)) * static_cast<int64_t>(kCubeSize);
    ctx.c1_dim = cin_opt / ctx.cube_k;
    ctx.data_size = GetSizeByDataType(data_type);
    ctx.dst_size = GetItemNumByShape(args.dst_shape) * ctx.data_size;
    return KERNEL_STATUS_OK;
}

inline int64_t Compute5dIndexForFractalz3D(const Format& format_5d, const Fractalz3DCtx& ctx, int64_t d, int64_t h,
                                           int64_t w, int64_t c, int64_t src_co)
{
    if (format_5d == FORMAT_DHWCN) {
        return d * ctx.h_dim * ctx.w_dim * ctx.c_dim * ctx.n_dim + h * ctx.w_dim * ctx.c_dim * ctx.n_dim +
               w * ctx.c_dim * ctx.n_dim + c * ctx.n_dim + src_co;
    }
    if (format_5d == FORMAT_NCDHW) {
        return src_co * ctx.c_dim * ctx.d_dim * ctx.h_dim * ctx.w_dim + c * ctx.d_dim * ctx.h_dim * ctx.w_dim +
               d * ctx.h_dim * ctx.w_dim + h * ctx.w_dim + w;
    }
    if (format_5d == FORMAT_NDHWC) {
        return src_co * ctx.d_dim * ctx.h_dim * ctx.w_dim * ctx.c_dim + d * ctx.h_dim * ctx.w_dim * ctx.c_dim +
               h * ctx.w_dim * ctx.c_dim + w * ctx.c_dim + c;
    }
    return 0;
}

inline void CopyFractalz3DCoutColumn(const TransArgs& args, const Format& format_5d, const Fractalz3DCtx& ctx,
                                     int64_t g, int64_t d, int64_t c, int64_t h, int64_t w, bool reverse)
{
    for (int64_t n = 0; n < ctx.cout_ori; n++) {
        const int64_t e_val = g % ctx.e_mult;
        const int64_t dst_ci = e_val * ctx.cin_ori + c;
        const int64_t dst_co = e_val * ctx.cout_ori + n;
        const int64_t src_co = g * ctx.cout_ori + n;
        const int64_t tempory = dst_ci % ctx.cube_k;
        const int64_t index_fz = (g / ctx.e_mult) * ctx.d_dim * ctx.c1_dim * ctx.h_dim * ctx.w_dim * ctx.cout_opt *
                                     ctx.cube_k +
                                 d * ctx.c1_dim * ctx.h_dim * ctx.w_dim * ctx.cout_opt * ctx.cube_k +
                                 (dst_ci / ctx.cube_k) * ctx.h_dim * ctx.w_dim * ctx.cout_opt * ctx.cube_k +
                                 h * ctx.w_dim * ctx.cout_opt * ctx.cube_k + w * ctx.cout_opt * ctx.cube_k +
                                 dst_co * ctx.cube_k + tempory;
        const int64_t index_5d = Compute5dIndexForFractalz3D(format_5d, ctx, d, h, w, c, src_co);
        if (!reverse) {
            copy_data(args.data, args.output, index_5d, index_fz, ctx.data_size);
        } else {
            copy_data(args.data, args.output, index_fz, index_5d, ctx.data_size);
        }
    }
}

inline void CopyFractalz3DHWSlice(const TransArgs& args, const Format& format_5d, const Fractalz3DCtx& ctx, int64_t g,
                                  int64_t d, int64_t c, bool reverse)
{
    for (int64_t h = 0; h < ctx.h_dim; h++) {
        for (int64_t w = 0; w < ctx.w_dim; w++) {
            CopyFractalz3DCoutColumn(args, format_5d, ctx, g, d, c, h, w, reverse);
        }
    }
}

uint32_t TransFormatWithGroups(const Format& format_5d, const std::vector<int64_t>& shape_5d, const TransArgs& args,
                               bool reverse)
{
    Fractalz3DCtx ctx;
    const uint32_t ret = BuildFractalz3DCtx(format_5d, shape_5d, args, ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return RunGroupedFormatTransfer(args, ctx, [&](int64_t g, int64_t d, int64_t c) {
        CopyFractalz3DHWSlice(args, format_5d, ctx, g, d, c, reverse);
    });
}

} // namespace

uint32_t FormatTransferFractalz3D::TransFormat(const TransArgs& args)
{
    KERNEL_LOG_DEBUG("Begin to trans format from [%s] to [%s], src shape [%s], data type "
                     "[%s], dst "
                     "shape [%s]",
                     FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                     VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                     VectorToString(args.dst_shape).c_str());

    if ((args.groups) == 0) {
        KERNEL_LOG_ERROR("Attr[groups] must not be equal 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (((args.src_format == FORMAT_NDHWC) || (args.src_format == FORMAT_DHWCN) || (args.src_format == FORMAT_NCDHW)) &&
        args.dst_format == FORMAT_FRACTAL_Z_3D) {
        std::vector<int64_t> expect_shape;
        auto ret = TransShape(args, expect_shape, false);
        if (ret != KERNEL_STATUS_OK) {
            return ret;
        }
        if (!IsTransShapeDstCorrect(args, expect_shape)) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return TransFormatWithGroups(args.src_format, args.src_shape, args, false);
    } else if (((args.dst_format == FORMAT_NDHWC) || (args.dst_format == FORMAT_DHWCN) ||
                (args.dst_format == FORMAT_NCDHW)) &&
               args.src_format == FORMAT_FRACTAL_Z_3D) {
        std::vector<int64_t> expect_input_shape;
        auto ret = TransShape(args, expect_input_shape, true);
        if (ret != KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("Check dst shape failed, dst shape [%s]", VectorToString(args.dst_shape).c_str());
            return ret;
        }

        if ((!args.src_shape.empty()) && (args.src_shape != expect_input_shape)) {
            KERNEL_LOG_ERROR("Check dst shape failed, dst shape [%s]", VectorToString(args.dst_shape).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }

        return TransFormatWithGroups(args.dst_format, args.dst_shape, args, true);
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t FormatTransferFractalz3D::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    const DataType data_type = args.src_data_type;
    const int64_t groups = args.groups;
    if (CheckDataTypeSupport(data_type) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    Format src_format = args.src_format;
    Format dst_format = args.dst_format;
    std::vector<int64_t> src_shape = args.src_shape;
    int32_t format = args.output_format;
    if (reverse) {
        src_format = args.dst_format;
        dst_format = args.src_format;
        src_shape = args.dst_shape;
        format = args.input_format;
    }

    if (src_format == FORMAT_NDHWC &&
        GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z_3D)) {
        return TransShapeNdhwcToFzWithGroups(src_shape, data_type, dst_shape, groups, format);
    }
    if ((src_format == FORMAT_DHWCN) &&
        GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z_3D)) {
        return TransShapeDhwcnToFzWithGroups(src_shape, data_type, dst_shape, groups, format);
    }
    if (src_format == FORMAT_NCDHW &&
        GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z_3D)) {
        return TransShapeNcdhwToFzWithGroups(src_shape, data_type, dst_shape, groups, format);
    }

    return KERNEL_STATUS_PARAM_INVALID;
}
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_NCDHW, FORMAT_FRACTAL_Z_3D)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_DHWCN, FORMAT_FRACTAL_Z_3D)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_NDHWC, FORMAT_FRACTAL_Z_3D)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_FRACTAL_Z_3D, FORMAT_NCDHW)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_FRACTAL_Z_3D, FORMAT_DHWCN)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalz3D, FORMAT_FRACTAL_Z_3D, FORMAT_NDHWC)
} // namespace formats
} // namespace  aicpu
