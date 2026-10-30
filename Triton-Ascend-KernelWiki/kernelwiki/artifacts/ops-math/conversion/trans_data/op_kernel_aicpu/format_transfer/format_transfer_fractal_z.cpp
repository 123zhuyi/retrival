/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "format_transfer_fractal_z.h"

#include "format_transfer_utils.h"
#include "formats_definitions.h"
#include "kernel_util.h"
#include "log.h"
#include "securec.h"
#include "status.h"
#include "graph/types.h"

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
 * frac_z axises: (C1*H*W, No, Ni, C0), which Ni = 16, C0 = cube_k,
 * No = Ceil(N/Ni), C1 = Ceil(C/C0)
 * @return
 */

uint32_t TransShapeToFzWithGroups(int64_t cube_k, const FractalZShape& shape, DataType data_type,
                                  std::vector<int64_t>& dst_shape, int64_t groups)
{
    (void)data_type;
    return BuildFzWithGroupsShape(shape.n_dim, shape.c_dim, shape.h_dim * shape.w_dim, cube_k, groups, dst_shape);
}

/*
 * FzC04 axises: (Ceil(kC0*h*w*C1/16), N1, N0, 16), which N1 = 16, C0 = 4, N1 =
 * Ceil(N/N0), C1 = Ceil(C/C0)
 */
uint32_t TransShapeToFzC04(const std::vector<int64_t>& src_shape, DataType data_type, int64_t c,
                           std::vector<int64_t>& dst_shape, const int64_t cube_k)
{
    (void)data_type;
    if (cube_k == 0) {
        KERNEL_LOG_ERROR("Cube_k must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    int64_t h = src_shape.at(kHwcnH);
    int64_t w = src_shape.at(kHwcnW);
    int64_t n = src_shape.at(kHwcnN);

    int64_t cin_ori = c;
    // For this place , groups is not equal to 0, which had been checked in
    // [Transdata] entrance.
    int64_t cout_ori = n;
    if (cin_ori == 0) {
        KERNEL_LOG_ERROR("Cin_ori must not be equal 0, and current cin_ori is [%ld]", cin_ori);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t cin_opt = Ceil(cin_ori, cube_k) * cube_k;
    int64_t c1 = cin_opt / cube_k;
    int64_t n1 = Ceil(cout_ori, static_cast<int64_t>(kCubeSize));

    dst_shape.clear();
    dst_shape.push_back(Ceil((kC0 * h * w * c1), cube_k));
    dst_shape.push_back(n1);
    dst_shape.push_back(kNiSize);
    dst_shape.push_back(cube_k);
    if (!IsShapeValid(dst_shape)) {
        KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t TransShapeNchwToFzWithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                      std::vector<int64_t>& dst_shape, int64_t groups, const int64_t c0)
{
    if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto n = src_shape.at(kNchwN);
    auto c = src_shape.at(kNchwC);
    auto h = src_shape.at(kNchwH);
    auto w = src_shape.at(kNchwW);
    const FractalZShape nchw{n, c, h, w};

    return TransShapeToFzWithGroups(c0, nchw, data_type, dst_shape, groups);
}

uint32_t TransShapeHwcnToFzWithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                      std::vector<int64_t>& dst_shape, int64_t groups, const int64_t c0)
{
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto h = src_shape.at(kHwcnH);
    auto w = src_shape.at(kHwcnW);
    auto c = src_shape.at(kHwcnC);
    auto n = src_shape.at(kHwcnN);
    const FractalZShape hwcn{n, c, h, w};

    return TransShapeToFzWithGroups(c0, hwcn, data_type, dst_shape, groups);
}

uint32_t TransShapeHwcnToFzC04WithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                         std::vector<int64_t>& dst_shape, int64_t groups, const int64_t c0)
{
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (groups != 1) {
        KERNEL_LOG_ERROR("The attr of groups must be equal to 1.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t c = src_shape.at(kHwcnC);
    if (c > kC0) {
        KERNEL_LOG_ERROR("The dim of C must be less than or equal to 4.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return TransShapeToFzC04(src_shape, data_type, c, dst_shape, c0);
}

uint32_t TransShapeNhwcToFzWithGroups(const std::vector<int64_t>& src_shape, DataType data_type,
                                      std::vector<int64_t>& dst_shape, int64_t groups, const int64_t c0)
{
    if (!CheckShapeValid(src_shape, kNhwcDimsNum)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto n = src_shape.at(kNhwcN);
    auto h = src_shape.at(kNhwcH);
    auto w = src_shape.at(kNhwcW);
    auto c = src_shape.at(kNhwcC);
    const FractalZShape nhwc{n, c, h, w};

    return TransShapeToFzWithGroups(c0, nhwc, data_type, dst_shape, groups);
}

int64_t GetCubeSizeByArgs(const TransArgs& args)
{
    if (((GetPrimaryFormat(static_cast<int32_t>(args.src_format)) == FORMAT_FRACTAL_Z) ||
         (GetPrimaryFormat(static_cast<int32_t>(args.src_format)) == FORMAT_FRACTAL_Z_C04)) &&
        (args.src_shape.size() > 0)) {
        return args.src_shape[args.src_shape.size() - 1];
    } else if (((GetPrimaryFormat(static_cast<int32_t>(args.dst_format)) == FORMAT_FRACTAL_Z) ||
                (GetPrimaryFormat(static_cast<int32_t>(args.dst_format)) == FORMAT_FRACTAL_Z_C04)) &&
               (args.dst_shape.size() > 0)) {
        return args.dst_shape[args.dst_shape.size() - 1];
    }
    return -1;
}

int64_t GetC0ValueByArgs(const TransArgs& args)
{
    if (ge::HasC0Format(args.input_format)) {
        return ge::GetC0Value(args.input_format);
    } else if (ge::HasC0Format(args.output_format)) {
        return ge::GetC0Value(args.output_format);
    } else {
        return GetCubeSizeByArgs(args);
    }
}
// Supporting NHWC/NCHW/HWCN <=> FORMAT_FRACTAL_Z (GC1HWN1N0C0),
// the final effect achieved is for the data to be distributed diagonally.
// For example: When the input filter format is NCHW, calculated the
// Correspondence of index between NCHW and FORMAT_FRACTAL_Z , then Convert the
// old filter to the new filter, and finally added 0 to the position where there
// is no data.
struct FractalZTransCtx {
    int64_t h_dim;
    int64_t w_dim;
    int64_t c_dim;
    int64_t n_dim;
    int64_t d_dim;
    int64_t cin_ori;
    int64_t cout_ori;
    int64_t cube_k;
    int64_t e_mult;
    int64_t cin_opt;
    int64_t cout_opt;
    int64_t c1_dim;
    int64_t data_size;
    int64_t dst_size;
};

uint32_t BuildFractalZCtx(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args,
                          int64_t cout_ori_divider, FractalZTransCtx& ctx)
{
    int64_t d_dim = 1;
    ctx.h_dim = 0;
    ctx.w_dim = 0;
    ctx.c_dim = 0;
    ctx.n_dim = 0;
    if (GetFormatDim(d_dim, ctx.h_dim, ctx.w_dim, ctx.c_dim, ctx.n_dim, format_4d, shape_4d) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ctx.d_dim = d_dim;
    ctx.cin_ori = ctx.c_dim;
    if (cout_ori_divider == 0) {
        KERNEL_LOG_ERROR("cout_ori divider must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ctx.cout_ori = ctx.n_dim / cout_ori_divider;
    if (CheckDimOri(ctx.cin_ori, ctx.cout_ori) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const DataType data_type = args.src_data_type;
    ctx.cube_k = GetC0ValueByArgs(args);
    KERNEL_LOG_INFO("C0 is [%ld]", ctx.cube_k);
    KERNEL_CHECK_FALSE((ctx.cube_k > 0), KERNEL_STATUS_PARAM_INVALID, "C0 must greater than 0, now is [%ld].",
                       ctx.cube_k);
    ctx.e_mult = std::min(Lcm(Lcm(ctx.cin_ori, ctx.cube_k) / (ctx.cin_ori),
                              Lcm(ctx.cout_ori, static_cast<int64_t>(kCubeSize)) / (ctx.cout_ori)),
                          args.groups);
    if (ctx.e_mult == 0) {
        KERNEL_LOG_ERROR("E_mult must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    ctx.cin_opt = Ceil(ctx.e_mult * ctx.cin_ori, ctx.cube_k) * ctx.cube_k;
    ctx.cout_opt = Ceil(ctx.e_mult * ctx.cout_ori, static_cast<int64_t>(kCubeSize)) * static_cast<int64_t>(kCubeSize);
    ctx.c1_dim = ctx.cin_opt / ctx.cube_k;
    ctx.data_size = GetSizeByDataType(data_type);
    ctx.dst_size = GetItemNumByShape(args.dst_shape) * ctx.data_size;
    return KERNEL_STATUS_OK;
}

inline int64_t Compute4dIndexForFractalZ(const Format& format_4d, const FractalZTransCtx& ctx, int64_t d, int64_t h,
                                         int64_t w, int64_t c, int64_t src_co)
{
    if (format_4d == FORMAT_HWCN) {
        return d * ctx.h_dim * ctx.w_dim * ctx.c_dim * ctx.n_dim + h * ctx.w_dim * ctx.c_dim * ctx.n_dim +
               w * ctx.c_dim * ctx.n_dim + c * ctx.n_dim + src_co;
    }
    if (format_4d == FORMAT_NCHW) {
        return src_co * ctx.c_dim * ctx.d_dim * ctx.h_dim * ctx.w_dim + c * ctx.d_dim * ctx.h_dim * ctx.w_dim +
               d * ctx.h_dim * ctx.w_dim + h * ctx.w_dim + w;
    }
    if (format_4d == FORMAT_NHWC) {
        return src_co * ctx.d_dim * ctx.h_dim * ctx.w_dim * ctx.c_dim + d * ctx.h_dim * ctx.w_dim * ctx.c_dim +
               h * ctx.w_dim * ctx.c_dim + w * ctx.c_dim + c;
    }
    return 0;
}

inline void CopyFractalZCoutColumn(const TransArgs& args, const Format& format_4d, const FractalZTransCtx& ctx,
                                   int64_t g, int64_t d, int64_t c, int64_t h, int64_t w, bool reverse)
{
    for (int64_t n = 0; n < ctx.cout_ori; n++) {
        int64_t e_val = g % ctx.e_mult;
        int64_t dst_ci = e_val * ctx.cin_ori + c;
        int64_t dst_co = e_val * ctx.cout_ori + n;
        int64_t src_co = g * ctx.cout_ori + n;
        int64_t tempory = dst_ci % ctx.cube_k;
        int64_t inx_fz = (g / ctx.e_mult) * ctx.d_dim * ctx.c1_dim * ctx.h_dim * ctx.w_dim * ctx.cout_opt * ctx.cube_k +
                         d * ctx.c1_dim * ctx.h_dim * ctx.w_dim * ctx.cout_opt * ctx.cube_k +
                         (dst_ci / ctx.cube_k) * ctx.h_dim * ctx.w_dim * ctx.cout_opt * ctx.cube_k +
                         h * ctx.w_dim * ctx.cout_opt * ctx.cube_k + w * ctx.cout_opt * ctx.cube_k +
                         dst_co * ctx.cube_k + tempory;
        int64_t inx_4d = Compute4dIndexForFractalZ(format_4d, ctx, d, h, w, c, src_co);
        if (!reverse) {
            copy_data(args.data, args.output, inx_4d, inx_fz, ctx.data_size);
        } else {
            copy_data(args.data, args.output, inx_fz, inx_4d, ctx.data_size);
        }
    }
}

inline void CopyFractalZHWSlice(const TransArgs& args, const Format& format_4d, const FractalZTransCtx& ctx, int64_t g,
                                int64_t d, int64_t c, bool reverse)
{
    for (int64_t h = 0; h < ctx.h_dim; h++) {
        for (int64_t w = 0; w < ctx.w_dim; w++) {
            CopyFractalZCoutColumn(args, format_4d, ctx, g, d, c, h, w, reverse);
        }
    }
}

uint32_t TransFormatWithGroups(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args,
                               bool reverse)
{
    FractalZTransCtx ctx;
    const uint32_t ret = BuildFractalZCtx(format_4d, shape_4d, args, args.groups, ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    return RunGroupedFormatTransfer(args, ctx, [&](int64_t g, int64_t d, int64_t c) {
        CopyFractalZHWSlice(args, format_4d, ctx, g, d, c, reverse);
    });
}

inline void CopyFzC04Nout(const TransArgs& args, const FractalZTransCtx& ctx, int64_t c, int64_t h, int64_t w,
                          int64_t c1, int64_t n1, bool reverse)
{
    for (int64_t n = 0; n < ctx.n_dim; n++) {
        int64_t common_factor = h * ctx.w_dim * c1 * kC0 + w * kC0 + c % kC0;
        int64_t inx_fz = common_factor / ctx.cube_k * (ctx.cube_k * kNiSize * n1) +
                         (n / kNiSize * (ctx.cube_k * kNiSize)) + (n % kNiSize * ctx.cube_k) +
                         common_factor % ctx.cube_k;
        int64_t inx_4d = h * ctx.w_dim * ctx.c_dim * ctx.n_dim + w * ctx.c_dim * ctx.n_dim + c * ctx.n_dim + n;
        if (!reverse) {
            copy_data(args.data, args.output, inx_4d, inx_fz, ctx.data_size);
        } else {
            copy_data(args.data, args.output, inx_fz, inx_4d, ctx.data_size);
        }
    }
}

uint32_t TransFormatForFzC04(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args,
                             bool reverse)
{
    FractalZTransCtx ctx;
    // For fz_c04 cout_ori uses n_dim directly (divider = 1).
    uint32_t ret = BuildFractalZCtx(format_4d, shape_4d, args, 1, ctx);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }
    int64_t c1 = ctx.cin_opt / ctx.cube_k;
    int64_t n1 = Ceil(ctx.e_mult * ctx.cout_ori, static_cast<int64_t>(kCubeSize));
    if (ctx.dst_size == 0) {
        return KERNEL_STATUS_OK;
    }
    const errno_t mret = memset_s(args.output, static_cast<size_t>(ctx.dst_size), 0, static_cast<size_t>(ctx.dst_size));
    if (mret != EOK) {
        KERNEL_LOG_ERROR("memset_s failed, size [%ld], errno [%d].", ctx.dst_size, mret);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    for (int64_t c = 0; c < ctx.c_dim; c++) {
        for (int64_t h = 0; h < ctx.h_dim; h++) {
            for (int64_t w = 0; w < ctx.w_dim; w++) {
                CopyFzC04Nout(args, ctx, c, h, w, c1, n1, reverse);
            }
        }
    }
    return KERNEL_STATUS_OK;
}
uint32_t DispatchToFractalZ(FormatTransferFractalZ& transfer, const TransArgs& args, bool reverse, bool is_fz_c04)
{
    std::vector<int64_t> expect_shape;
    auto ret = transfer.TransShape(args, expect_shape, reverse);
    if (ret != KERNEL_STATUS_OK) {
        if (reverse) {
            KERNEL_LOG_ERROR("Check dst shape failed, dst shape [%s]", VectorToString(args.dst_shape).c_str());
        }
        return ret;
    }
    if (!reverse) {
        if (!IsTransShapeDstCorrect(args, expect_shape)) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        return TransFormatWithGroups(args.src_format, args.src_shape, args, false);
    }
    if ((!args.src_shape.empty()) && (args.src_shape != expect_shape)) {
        KERNEL_LOG_ERROR("Check dst shape failed, dst shape [%s]", VectorToString(args.dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (is_fz_c04) {
        return TransFormatForFzC04(args.dst_format, args.dst_shape, args, true);
    }
    return TransFormatWithGroups(args.dst_format, args.dst_shape, args, true);
}
} // namespace

uint32_t FormatTransferFractalZ::TransFormat(const TransArgs& args)
{
    if (args.groups == 0) {
        KERNEL_LOG_ERROR("Attr[groups] must not be equal to 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    KERNEL_LOG_DEBUG("Begin to trans format from [%s] to [%s], src shape [%s], data type "
                     "[%s], dst "
                     "shape [%s], groups [%ld]",
                     FormatToSerialString(args.src_format).c_str(), FormatToSerialString(args.dst_format).c_str(),
                     VectorToString(args.src_shape).c_str(), DTypeStr(args.src_data_type).c_str(),
                     VectorToString(args.dst_shape).c_str(), args.groups);

    const bool dst_format_is_fz = ((args.src_format == FORMAT_NHWC) || (args.src_format == FORMAT_HWCN) ||
                                   (args.src_format == FORMAT_NCHW)) &&
                                  args.dst_format == FORMAT_FRACTAL_Z;
    const bool src_format_is_fz = ((args.dst_format == FORMAT_NHWC) || (args.dst_format == FORMAT_HWCN) ||
                                   (args.dst_format == FORMAT_NCHW)) &&
                                  args.src_format == FORMAT_FRACTAL_Z;
    const bool src_format_is_fzc04 = (args.src_format == FORMAT_FRACTAL_Z_C04) && (args.dst_format == FORMAT_HWCN);
    if (dst_format_is_fz) {
        return DispatchToFractalZ(*this, args, false, false);
    }
    if (src_format_is_fz) {
        return DispatchToFractalZ(*this, args, true, false);
    }
    if (src_format_is_fzc04) {
        return DispatchToFractalZ(*this, args, true, true);
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

uint32_t FormatTransferFractalZ::TransShape(const TransArgs& args, std::vector<int64_t>& dst_shape, bool reverse)
{
    const DataType data_type = args.src_data_type;
    const int64_t groups = args.groups;
    if (CheckDataTypeSupport(data_type) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    Format src_format = args.src_format;
    Format dst_format = args.dst_format;
    std::vector<int64_t> src_shape = args.src_shape;
    if (reverse) {
        src_format = args.dst_format;
        dst_format = args.src_format;
        src_shape = args.dst_shape;
    }
    int64_t c0 = GetC0ValueByArgs(args);
    if (c0 <= 0) {
        std::string error = "trans format from" + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                            FmtToStr(FormatToSerialString(args.dst_format)) +
                            ", invalid relationship between src shape " + FmtToStr(VectorToString(args.src_shape)) +
                            " and dst " + FmtToStr(VectorToString(args.dst_shape));
        KERNEL_LOG_ERROR("C0 must greater than 0, now is [%ld], %s", c0, error.c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (src_format == FORMAT_NHWC && GetPrimaryFormat(static_cast<int32_t>(dst_format)) == FORMAT_FRACTAL_Z) {
        return TransShapeNhwcToFzWithGroups(src_shape, data_type, dst_shape, groups, c0);
    }
    if ((src_format == FORMAT_HWCN) &&
        (GetPrimaryFormat(static_cast<int32_t>(dst_format)) == static_cast<int32_t>(FORMAT_FRACTAL_Z))) {
        return TransShapeHwcnToFzWithGroups(src_shape, data_type, dst_shape, groups, c0);
    }
    if (src_format == FORMAT_NCHW && GetPrimaryFormat(static_cast<int32_t>(dst_format)) == FORMAT_FRACTAL_Z) {
        return TransShapeNchwToFzWithGroups(src_shape, data_type, dst_shape, groups, c0);
    }
    if ((src_format == FORMAT_HWCN) && (GetPrimaryFormat(static_cast<int32_t>(dst_format)) == FORMAT_FRACTAL_Z_C04)) {
        return TransShapeHwcnToFzC04WithGroups(src_shape, data_type, dst_shape, groups, c0);
    }
    return KERNEL_STATUS_PARAM_INVALID;
}
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NCHW, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_HWCN, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NHWC, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_FRACTAL_Z, FORMAT_NCHW)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_FRACTAL_Z, FORMAT_HWCN)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_FRACTAL_Z, FORMAT_NHWC)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_FRACTAL_Z_C04, FORMAT_HWCN)
} // namespace formats
} // namespace  aicpu
