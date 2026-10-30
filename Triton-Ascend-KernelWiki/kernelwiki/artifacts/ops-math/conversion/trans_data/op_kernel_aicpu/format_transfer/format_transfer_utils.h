/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_UTILS_H_
#define AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_UTILS_H_

#include <string>
#include <vector>
#include "kernel_util.h"
#include "status.h"
#include "register_format_transfer.h"

namespace aicpu {
namespace formats {
static const int kCubeSize = 16;
static const int kNiSize = 16;
static const int kC0 = 4;
static const int64_t kShapeItemNumMAX = 1024UL * 1024UL * 1024UL * 1024UL;
int64_t Measure(int64_t x, int64_t y);
int64_t Lcm(int64_t a, int64_t b);
bool IsShapeValid(const std::vector<int64_t>& shape);

bool CheckShapeValid(const std::vector<int64_t>& shape, const int64_t expect_dims);

int64_t GetC0ValueForTransShape(DataType data_type, const int32_t format);

int64_t GetC0ValueForTransFormat(DataType data_type, const int32_t input_format, const int32_t output_format);

int64_t GetCubeSizeByDataType(DataType data_type);

bool IsTransShapeSrcCorrect(const TransArgs& args, std::vector<int64_t>& expect_shape);

bool IsTransShapeDstCorrect(const TransArgs& args, std::vector<int64_t>& expect_shape);

int64_t GetItemNumByShape(const std::vector<int64_t>& shape);

void copy_data(const uint8_t* input_data, uint8_t* dst, int64_t src_index, int64_t dst_index, int64_t data_size);

KernelStatus GetFormatDim(int64_t& d_dim, int64_t& h_dim, int64_t& w_dim, int64_t& c_dim, int64_t& n_dim,
                          const Format& input_format, const std::vector<int64_t>& dims);
KernelStatus CheckDimOri(int64_t cin_ori, int64_t cout_ori);

// Shared 4D-format preamble used by TransFormatToC1hwc0 / TransFormatToWINO.
struct Format4dBasics {
    int64_t d_dim;
    int64_t h_dim;
    int64_t w_dim;
    int64_t c_dim;
    int64_t n_dim;
    DataType data_type;
    int64_t c0;
    int64_t data_size;
    int64_t dst_size;
};
uint32_t Prepare4dFormatBasics(const Format& format_4d, const std::vector<int64_t>& shape_4d, const TransArgs& args,
                               Format4dBasics& out);

// Shared groups/cube_k/cin/cout validation used by TransShapeToFz*WithGroups.
uint32_t ComputeCinCoutOri(int64_t n, int64_t c, int64_t groups, int64_t cube_k, int64_t& cin_ori, int64_t& cout_ori);

// Shared shape builder for both TransShapeToFzWithGroups (fractal_z, spatial_dim = h*w)
// and TransShapeToFz3DWithGroups (fractalz_3d, spatial_dim = d*h*w).
uint32_t BuildFzWithGroupsShape(int64_t n, int64_t c, int64_t spatial_dim, int64_t cube_k, int64_t groups,
                                std::vector<int64_t>& dst_shape);

// Shared grouped-format loop skeleton used by TransFormatWithGroups in fractal_z / fractalz_3d.
template <typename Ctx, typename CopyHwSliceFn>
uint32_t RunGroupedFormatTransfer(const TransArgs& args, const Ctx& ctx, CopyHwSliceFn copy_hw_slice)
{
    if (ctx.dst_size == 0) {
        return KERNEL_STATUS_OK;
    }
    if (!BiggerMemSet(args.output, static_cast<size_t>(ctx.dst_size), 0, static_cast<size_t>(ctx.dst_size))) {
        KERNEL_LOG_ERROR("BiggerMemSet failed, size [%ld].", ctx.dst_size);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    for (int64_t g = 0; g < args.groups; g++) {
        for (int64_t d = 0; d < ctx.d_dim; d++) {
            for (int64_t c = 0; c < ctx.c_dim; c++) {
                copy_hw_slice(g, d, c);
            }
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
T Ceil(T n1, T n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (n1 - 1) / n2 + 1 : 0;
}

/**
 * Convert the data format, and put the converted format and length in the
 * result
 * @param args
 * @param result
 * @return
 */
uint32_t TransFormat(const TransArgs& args);
} // namespace formats
} // namespace aicpu
#endif // AICPU_KERNELS_HOST_FORMAT_TRANSFER_FORMAT_TRANSFER_UTILS_H_
