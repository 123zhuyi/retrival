/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_transfer_utils.h"

#include "formats_definitions.h"
#include "kernel_util.h"
#include "securec.h"
#include "log.h"
#include "graph/types.h"

using namespace std;
using namespace ge;

namespace aicpu {
namespace formats {
bool IsShapeValid(const vector<int64_t>& shape)
{
    if (shape.empty()) {
        return false;
    }
    int64_t num = 1;
    for (auto dim : shape) {
        if (dim < 0) {
            string error = "Invalid negative dims in the shape " + FmtToStr(VectorToString(shape));
            KERNEL_LOG_ERROR("%s", error.c_str());
            return false;
        }
        if (dim != 0 && kShapeItemNumMAX / dim < num) {
            string error = "Shape overflow, the total count should be less than " + FmtToStr(kShapeItemNumMAX);
            KERNEL_LOG_ERROR("%s", error.c_str());
            return false;
        }
        num *= dim;
    }
    return true;
}

bool CheckShapeValid(const vector<int64_t>& shape, const int64_t expect_dims)
{
    if (expect_dims <= 0 || shape.size() != static_cast<size_t>(expect_dims)) {
        string error = "Invalid shape, dims num " + FmtToStr(shape.size()) + ", expect " + FmtToStr(expect_dims);
        KERNEL_LOG_ERROR("%s", error.c_str());
        return false;
    }
    return IsShapeValid(shape);
}

int64_t GetCubeSizeByDataType(DataType data_type)
{
    // Current cube does not support 4 bytes and longer data
    auto size = GetSizeByDataType(data_type);
    if (size <= 0) {
        std::string error = "Failed to get cube size, the data type " + FmtToStr(DTypeStr(data_type)) + " is invalid";
        KERNEL_LOG_ERROR("%s", error.c_str());
        return -1;
    } else if (size == 1) {
        return kCubeSize * 2; // 32 bytes cube size
    } else {
        return kCubeSize;
    }
}

int64_t GetC0ValueForTransShape(DataType data_type, const int32_t format)
{
    if (ge::HasC0Format(format)) {
        return ge::GetC0Value(format);
    } else {
        return GetCubeSizeByDataType(data_type);
    }
}

int64_t GetC0ValueForTransFormat(DataType data_type, const int32_t input_format, const int32_t output_format)
{
    if (ge::HasC0Format(input_format)) {
        return ge::GetC0Value(input_format);
    } else if (ge::HasC0Format(output_format)) {
        return ge::GetC0Value(output_format);
    } else {
        return GetCubeSizeByDataType(data_type);
    }
}

bool IsTransShapeSrcCorrect(const TransArgs& args, std::vector<int64_t>& expect_shape)
{
    if (args.src_shape != expect_shape) {
        string error = "Failed to trans format from" + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                       FmtToStr(FormatToSerialString(args.dst_format)) + ", invalid relationship between src shape " +
                       FmtToStr(VectorToString(args.src_shape)) + " and dst " +
                       FmtToStr(VectorToString(args.dst_shape));
        KERNEL_LOG_ERROR("%s", error.c_str());
        return false;
    }
    return true;
}

bool IsTransShapeDstCorrect(const TransArgs& args, vector<int64_t>& expect_shape)
{
    if (!args.dst_shape.empty() && args.dst_shape != expect_shape) {
        string error = "Failed to trans format from " + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                       FmtToStr(FormatToSerialString(args.dst_format)) + ", the dst shape" +
                       FmtToStr(VectorToString(args.dst_shape)) + " is invalid, expect" +
                       FmtToStr(VectorToString(expect_shape));
        KERNEL_LOG_ERROR("%s", error.c_str());
        return false;
    }
    return true;
}

int64_t GetItemNumByShape(const vector<int64_t>& shape)
{
    // shape will not be greater than INT_MAX
    int64_t num = 1;
    for (auto dim : shape) {
        num *= dim;
    }
    return num;
}

uint32_t TransFormat(const TransArgs& args)
{
    auto transfer = BuildFormatTransfer(args);
    if (transfer == nullptr) {
        string error = "Failed to trans data from format " + FmtToStr(FormatToSerialString(args.src_format)) + " to " +
                       FmtToStr(FormatToSerialString(args.dst_format));
        KERNEL_LOG_WARN("%s", error.c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto src_shape_size = GetItemNumByShape(args.src_shape);
    if (args.data == nullptr && src_shape_size != 0) {
        KERNEL_LOG_WARN("Invalid input null data");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return transfer->TransFormat(args);
}

int64_t Measure(int64_t x, int64_t y)
{
    if (y == 0) {
        return 1;
    }
    int64_t z = y;
    while (x % y != 0) {
        z = x % y;
        x = y;
        y = z;
    }
    return z;
}
// least common multiple
int64_t Lcm(int64_t a, int64_t b)
{
    if (b == 0) {
        return -1;
    }
    int64_t temp = (a * b) / (Measure(a, b));
    return temp;
}

void copy_data(const uint8_t* input_data, uint8_t* dst, int64_t src_index, int64_t dst_index, int64_t data_size)
{
    auto ret = memcpy_s(dst + dst_index * data_size, static_cast<size_t>(data_size), input_data + src_index * data_size,
                        static_cast<size_t>(data_size));
    if (ret != EOK) {
        KERNEL_LOG_ERROR("memcpy_s failed, ret [%d].", ret);
    }
}

KernelStatus CheckDimOri(int64_t cin_ori, int64_t cout_ori)
{
    if (cin_ori == 0 || cout_ori == 0) {
        KERNEL_LOG_ERROR("Cin_ori, cout_ori must not be equal 0, and current cin_ori is [%ld], "
                         "cout_ori is [%ld]",
                         cin_ori, cout_ori);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t Prepare4dFormatBasics(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args,
                               Format4dBasics& out)
{
    out.d_dim = 1;
    out.h_dim = 0;
    out.w_dim = 0;
    out.c_dim = 0;
    out.n_dim = 0;
    if (GetFormatDim(out.d_dim, out.h_dim, out.w_dim, out.c_dim, out.n_dim, format_4d, shape_4d) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    out.data_type = args.src_data_type;
    out.c0 = GetC0ValueForTransFormat(out.data_type, args.input_format, args.output_format);
    out.data_size = GetSizeByDataType(out.data_type);
    out.dst_size = GetItemNumByShape(args.dst_shape) * out.data_size;
    return KERNEL_STATUS_OK;
}

uint32_t ComputeCinCoutOri(int64_t n, int64_t c, int64_t groups, int64_t cube_k, int64_t& cin_ori, int64_t& cout_ori)
{
    if (groups == 0 || cube_k == 0) {
        KERNEL_LOG_ERROR("Groups and cube_k must not be equal to 0, now [%ld] [%ld]", groups, cube_k);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    cin_ori = c;
    // groups is not equal to 0, which had been checked above.
    cout_ori = n / groups;
    if (cin_ori == 0 || cout_ori == 0) {
        KERNEL_LOG_ERROR("Cin_ori, cout_ori must not be equal 0, "
                         "and current cin_ori, cout_ori, groups are [%ld] [%ld] [%ld]",
                         cin_ori, cout_ori, groups);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t BuildFzWithGroupsShape(int64_t n, int64_t c, int64_t spatial_dim, int64_t cube_k, int64_t groups,
                                std::vector<int64_t>& dst_shape)
{
    int64_t cin_ori = 0;
    int64_t cout_ori = 0;
    if (ComputeCinCoutOri(n, c, groups, cube_k, cin_ori, cout_ori) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    // ComputeCinCoutOri guarantees cube_k > 0 and cin_ori/cout_ori > 0 on success;
    // reassert locally to satisfy static analyzers on the divisions below.
    if (cube_k == 0 || cin_ori == 0 || cout_ori == 0) {
        KERNEL_LOG_ERROR("Invalid cube_k/cin_ori/cout_ori [%ld] [%ld] [%ld].", cube_k, cin_ori, cout_ori);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (cube_k == 0) {
        KERNEL_LOG_ERROR("cube_k must not be 0.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (cin_ori == 0 || cout_ori == 0) {
        KERNEL_LOG_ERROR("cin_ori/cout_ori must not be 0.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t e_mult = std::min(
        Lcm(Lcm(cin_ori, cube_k) / cin_ori, Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / cout_ori), groups);
    if (e_mult == 0) {
        KERNEL_LOG_ERROR("e_mult must not be 0.");
        return KERNEL_STATUS_PARAM_INVALID;
    }
    const int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
    const int64_t c1_dim = cin_opt / cube_k;
    const int64_t g_dim = Ceil(groups, e_mult);
    const int64_t n1 = Ceil(cout_ori * e_mult, static_cast<int64_t>(kCubeSize));
    dst_shape.clear();
    dst_shape.push_back(g_dim * c1_dim * spatial_dim);
    dst_shape.push_back(n1);
    dst_shape.push_back(kNiSize);
    dst_shape.push_back(cube_k);
    if (!IsShapeValid(dst_shape)) {
        KERNEL_LOG_ERROR("Check shape failed, dst shape [%s]", VectorToString(dst_shape).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

KernelStatus GetFormatDim(int64_t& d_dim, int64_t& h_dim, int64_t& w_dim, int64_t& c_dim, int64_t& n_dim,
                          const Format& input_format, const std::vector<int64_t>& dims)
{
    if (input_format == FORMAT_NCDHW) {
        n_dim = dims[kNcdhwN];
        c_dim = dims[kNcdhwC];
        d_dim = dims[kNcdhwD];
        h_dim = dims[kNcdhwH];
        w_dim = dims[kNcdhwW];
    } else if (input_format == FORMAT_DHWCN) {
        d_dim = dims[kDhwcnD];
        h_dim = dims[kDhwcnH];
        w_dim = dims[kDhwcnW];
        c_dim = dims[kDhwcnC];
        n_dim = dims[kDhwcnN];
    } else if (input_format == FORMAT_NDHWC) {
        n_dim = dims[kNdhwcN];
        d_dim = dims[kNdhwcD];
        h_dim = dims[kNdhwcH];
        w_dim = dims[kNdhwcW];
        c_dim = dims[kNdhwcC];
    } else if (input_format == FORMAT_NHWC) {
        n_dim = dims[kNhwcN];
        h_dim = dims[kNhwcH];
        d_dim = 1;
        w_dim = dims[kNhwcW];
        c_dim = dims[kNhwcC];
    } else if (input_format == FORMAT_NCHW) {
        n_dim = dims[kNchwN];
        c_dim = dims[kNchwC];
        h_dim = dims[kNchwH];
        w_dim = dims[kNchwW];
        d_dim = 1;
    } else if (input_format == FORMAT_HWCN) {
        h_dim = dims[kHwcnH];
        w_dim = dims[kHwcnW];
        c_dim = dims[kHwcnC];
        n_dim = dims[kHwcnN];
        d_dim = 1;
    } else {
        KERNEL_LOG_WARN("Format is not FORMAT_DHWCN or FORMAT_NDHWC or FORMAT_NCDHW or "
                        "FORMAT_NHWC or FORMAT_NCHW or FORMAT_HWCN, current input "
                        "format is [%d]",
                        static_cast<int32_t>(input_format));
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}
} // namespace formats
} // namespace aicpu
