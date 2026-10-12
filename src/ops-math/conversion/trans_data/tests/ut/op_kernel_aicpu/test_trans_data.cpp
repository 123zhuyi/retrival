/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace aicpu;

class TEST_TRANS_DATA_UT : public testing::Test {};

namespace {
std::shared_ptr<NodeDef> BuildTransDataNode(DataType dtype, Format src_fmt, Format dst_fmt,
                                            const std::vector<int64_t>& src_shape,
                                            const std::vector<int64_t>& dst_shape, void* src_data, void* dst_data,
                                            const std::string& src_format_str, const std::string& dst_format_str,
                                            int64_t groups = 1)
{
    auto node_def = CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "TransData", "TransData")
        .Input({"src", dtype, src_shape, src_data, src_fmt})
        .Output({"dst", dtype, dst_shape, dst_data, dst_fmt})
        .Attr("src_format", src_format_str)
        .Attr("dst_format", dst_format_str)
        .Attr("groups", groups);
    return node_def;
}

void VerifyDataInOutput(const std::vector<float>& src, const std::vector<float>& dst)
{
    std::set<float> src_set(src.begin(), src.end());
    std::set<float> dst_set(dst.begin(), dst.end());
    for (auto v : src_set) {
        EXPECT_GT(dst_set.count(v), 0) << "input value " << v << " missing from output";
    }
}
} // namespace

// ---------- Compute 分支5: 不支持的格式转换 → PARAM_INVALID ----------
TEST_F(TEST_TRANS_DATA_UT, unsupported_output_format_returns_param_invalid)
{
    std::vector<int64_t> src_shape = {1, 2, 2, 4};
    std::vector<int64_t> dst_shape = {1, 2, 2, 4};
    std::vector<float> src(16, 1.0f);
    std::vector<float> dst(16, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NHWC, FORMAT_NHWC, src_shape, dst_shape, src.data(), dst.data(),
                                       "NHWC", "NHWC");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---------- Compute 分支6: dims < 4 → PARAM_INVALID ----------
TEST_F(TEST_TRANS_DATA_UT, insufficient_dims_returns_param_invalid)
{
    std::vector<int64_t> src_shape = {2, 4};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(8, 1.0f);
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---------- Compute 分支7: groups == 0 → PARAM_INVALID ----------
TEST_F(TEST_TRANS_DATA_UT, groups_zero_returns_param_invalid)
{
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(16, 1.0f);
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z", 0);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ---------- NewCompute: HWCN→FRACTAL_Z，校验输出数据正确性 ----------
TEST_F(TEST_TRANS_DATA_UT, hwcn_to_fractal_z_float_success)
{
    // HWCN [1,1,1,16]: h=1, w=1, c=1, n=16 → FRACTAL_Z [1,1,16,16], dst_size=256
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NCHW→FRACTAL_Z ----------
TEST_F(TEST_TRANS_DATA_UT, nchw_to_fractal_z_float_success)
{
    // NCHW [1,1,1,16]: n=1, c=1, h=1, w=16 → FRACTAL_Z [16,1,16,16], dst_size=4096
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {16, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(4096, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NCHW, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "NCHW", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: ND→FRACTAL_NZ，校验输出数据正确性 ----------
TEST_F(TEST_TRANS_DATA_UT, nd_to_fractal_nz_float_success)
{
    std::vector<int64_t> src_shape = {16, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(256);
    for (int i = 0; i < 256; ++i) {
        src[i] = static_cast<float>(i);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_ND, FORMAT_FRACTAL_NZ, src_shape, dst_shape, src.data(),
                                       dst.data(), "ND", "FRACTAL_NZ");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // ND [16,16] → FRACTAL_NZ [1,1,16,16]: single block, no padding
    for (int i = 0; i < 256; ++i) {
        EXPECT_FLOAT_EQ(dst[i], static_cast<float>(i)) << "mismatch at dst[" << i << "]";
    }
}

// ---------- NewCompute: FRACTAL_NZ→ND (反向) ----------
TEST_F(TEST_TRANS_DATA_UT, fractal_nz_to_nd_float_success)
{
    std::vector<int64_t> src_shape = {1, 1, 16, 16};
    std::vector<int64_t> dst_shape = {16, 16};
    std::vector<float> src(256);
    for (int i = 0; i < 256; ++i) {
        src[i] = static_cast<float>(i);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_FRACTAL_NZ, FORMAT_ND, src_shape, dst_shape, src.data(),
                                       dst.data(), "FRACTAL_NZ", "ND");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    for (int i = 0; i < 256; ++i) {
        EXPECT_FLOAT_EQ(dst[i], static_cast<float>(i)) << "mismatch at dst[" << i << "]";
    }
}

// ---------- NewCompute: NCHW→FRACTAL_NZ ----------
TEST_F(TEST_TRANS_DATA_UT, nchw_to_fractal_nz_float_success)
{
    // NCHW [1,1,16,16] → FRACTAL_NZ [1,1,1,1,16,16], dst_size=256
    std::vector<int64_t> src_shape = {1, 1, 16, 16};
    std::vector<int64_t> dst_shape = {1, 1, 1, 1, 16, 16};
    std::vector<float> src(256);
    for (int i = 0; i < 256; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NCHW, FORMAT_FRACTAL_NZ, src_shape, dst_shape, src.data(),
                                       dst.data(), "NCHW", "FRACTAL_NZ");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NHWC→FRACTAL_NZ ----------
TEST_F(TEST_TRANS_DATA_UT, nhwc_to_fractal_nz_float_success)
{
    // NHWC [1,16,16,1] → FRACTAL_NZ [1,16,1,1,16,16], dst_size=4096
    std::vector<int64_t> src_shape = {1, 16, 16, 1};
    std::vector<int64_t> dst_shape = {1, 16, 1, 1, 16, 16};
    std::vector<float> src(256);
    for (int i = 0; i < 256; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(4096, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NHWC, FORMAT_FRACTAL_NZ, src_shape, dst_shape, src.data(),
                                       dst.data(), "NHWC", "FRACTAL_NZ");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NCHW→C1HWC0 ----------
TEST_F(TEST_TRANS_DATA_UT, nchw_to_c1hwc0_float_success)
{
    // NCHW [1,1,4,4] → C1HWC0 [1,4,4,16], c must be 1
    std::vector<int64_t> src_shape = {1, 1, 4, 4};
    std::vector<int64_t> dst_shape = {1, 4, 4, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NCHW, FORMAT_C1HWC0, src_shape, dst_shape, src.data(),
                                       dst.data(), "NCHW", "C1HWC0");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NCDHW→FRACTAL_Z_3D ----------
TEST_F(TEST_TRANS_DATA_UT, ncdhw_to_fractal_z_3d_float_success)
{
    // NCDHW [1,1,1,1,16] → FRACTAL_Z_3D: dst_shape=[16,1,16,16], dst_size=4096
    std::vector<int64_t> src_shape = {1, 1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {16, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(4096, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NCDHW, FORMAT_FRACTAL_Z_3D, src_shape, dst_shape, src.data(),
                                       dst.data(), "NCDHW", "FRACTAL_Z_3D");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- Compute 分支2: HWCN→FRACTAL_Z_C04 → HandleHwcnToFzC04 ----------
TEST_F(TEST_TRANS_DATA_UT, hwcn_to_fractal_z_c04_float_success)
{
    // HWCN [2,2,2,2] → FRACTAL_Z_C04
    std::vector<int64_t> src_shape = {2, 2, 2, 2};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(8);
    for (int i = 0; i < 8; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_FRACTAL_Z_C04, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z_C04");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- fp16 数据类型: HWCN→FRACTAL_Z ----------
TEST_F(TEST_TRANS_DATA_UT, hwcn_to_fractal_z_half_success)
{
    // HWCN [1,1,1,16] → FRACTAL_Z [1,1,16,16], dst_size=256
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<Eigen::half> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<Eigen::half>(static_cast<float>(i));
    }
    std::vector<Eigen::half> dst(256, Eigen::half(0.0f));

    auto node_def = BuildTransDataNode(DT_FLOAT16, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::set<float> src_vals;
    for (int i = 0; i < 16; ++i) {
        src_vals.insert(static_cast<float>(i));
    }
    std::set<float> dst_vals;
    for (int i = 0; i < 256; ++i) {
        dst_vals.insert(static_cast<float>(dst[i]));
    }
    for (auto v : src_vals) {
        EXPECT_GT(dst_vals.count(v), 0) << "input value " << v << " missing from output";
    }
}

// ---------- NewCompute: HWCN→C1HWC0 ----------
TEST_F(TEST_TRANS_DATA_UT, hwcn_to_c1hwc0_float_success)
{
    // HWCN [1,1,1,4] → C1HWC0 [1,1,1,16], c must be 1
    std::vector<int64_t> src_shape = {1, 1, 1, 4};
    std::vector<int64_t> dst_shape = {1, 1, 1, 16};
    std::vector<float> src(4);
    for (int i = 0; i < 4; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(16, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_C1HWC0, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "C1HWC0");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NHWC→FRACTAL_Z ----------
TEST_F(TEST_TRANS_DATA_UT, nhwc_to_fractal_z_float_success)
{
    // NHWC [1,1,1,16]: n=1,h=1,w=1,c=16 → FRACTAL_Z [1,1,16,16]
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NHWC, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "NHWC", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: DHWCN→FRACTAL_Z_3D ----------
TEST_F(TEST_TRANS_DATA_UT, dhwcn_to_fractal_z_3d_float_success)
{
    // DHWCN [1,1,1,1,16] → FRACTAL_Z_3D [1,1,16,16]
    std::vector<int64_t> src_shape = {1, 1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_DHWCN, FORMAT_FRACTAL_Z_3D, src_shape, dst_shape, src.data(),
                                       dst.data(), "DHWCN", "FRACTAL_Z_3D");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NDHWC→FRACTAL_Z_3D ----------
TEST_F(TEST_TRANS_DATA_UT, ndhwc_to_fractal_z_3d_float_success)
{
    // NDHWC [1,1,1,1,16] → FRACTAL_Z_3D [1,1,16,16]
    std::vector<int64_t> src_shape = {1, 1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NDHWC, FORMAT_FRACTAL_Z_3D, src_shape, dst_shape, src.data(),
                                       dst.data(), "NDHWC", "FRACTAL_Z_3D");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: FRACTAL_NZ→NCHW (反向) ----------
TEST_F(TEST_TRANS_DATA_UT, fractal_nz_to_nchw_float_success)
{
    // FRACTAL_NZ [1,1,1,1,16,16] → NCHW [1,1,16,16]
    std::vector<int64_t> src_shape = {1, 1, 1, 1, 16, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 16};
    std::vector<float> src(256);
    for (int i = 0; i < 256; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_FRACTAL_NZ, FORMAT_NCHW, src_shape, dst_shape, src.data(),
                                       dst.data(), "FRACTAL_NZ", "NCHW");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: HWCN→NCHW (格式互换) ----------
TEST_F(TEST_TRANS_DATA_UT, hwcn_to_nchw_float_success)
{
    // HWCN [1,1,1,4] → NCHW [4,1,1,1]
    std::vector<int64_t> src_shape = {1, 1, 1, 4};
    std::vector<int64_t> dst_shape = {4, 1, 1, 1};
    std::vector<float> src(4);
    for (int i = 0; i < 4; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(4, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_HWCN, FORMAT_NCHW, src_shape, dst_shape, src.data(), dst.data(),
                                       "HWCN", "NCHW");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NCHW→HWCN (格式互换反向) ----------
TEST_F(TEST_TRANS_DATA_UT, nchw_to_hwcn_float_success)
{
    // NCHW [4,1,1,1] → HWCN [1,1,1,4]
    std::vector<int64_t> src_shape = {4, 1, 1, 1};
    std::vector<int64_t> dst_shape = {1, 1, 1, 4};
    std::vector<float> src(4);
    for (int i = 0; i < 4; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(4, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NCHW, FORMAT_HWCN, src_shape, dst_shape, src.data(), dst.data(),
                                       "NCHW", "HWCN");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NCDHW→NDC1HWC0 ----------
TEST_F(TEST_TRANS_DATA_UT, ncdhw_to_ndc1hwc0_float_success)
{
    // NCDHW [1,1,1,1,16] → NDC1HWC0 [1,1,1,1,16,16]
    std::vector<int64_t> src_shape = {1, 1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 1, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NCDHW, FORMAT_NDC1HWC0, src_shape, dst_shape, src.data(),
                                       dst.data(), "NCDHW", "NDC1HWC0");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- NewCompute: NDHWC→NDC1HWC0 ----------
TEST_F(TEST_TRANS_DATA_UT, ndhwc_to_ndc1hwc0_float_success)
{
    // NDHWC [1,1,1,16,1]: n=1,d=1,h=1,w=16,c=1 → NDC1HWC0 [1,1,1,1,16,16]
    std::vector<int64_t> src_shape = {1, 1, 1, 16, 1};
    std::vector<int64_t> dst_shape = {1, 1, 1, 1, 16, 16};
    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dst(256, 0.0f);

    auto node_def = BuildTransDataNode(DT_FLOAT, FORMAT_NDHWC, FORMAT_NDC1HWC0, src_shape, dst_shape, src.data(),
                                       dst.data(), "NDHWC", "NDC1HWC0");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    VerifyDataInOutput(src, dst);
}

// ---------- int8 数据类型: HWCN→FRACTAL_Z ----------
TEST_F(TEST_TRANS_DATA_UT, hwcn_to_fractal_z_int8_success)
{
    // HWCN [1,1,1,16] → FRACTAL_Z [1,1,16,16], cube_k=32 for int8
    std::vector<int64_t> src_shape = {1, 1, 1, 16};
    std::vector<int64_t> dst_shape = {1, 1, 16, 32};
    std::vector<int8_t> src(16);
    for (int i = 0; i < 16; ++i) {
        src[i] = static_cast<int8_t>(i + 1);
    }
    std::vector<int8_t> dst(512, 0);

    auto node_def = BuildTransDataNode(DT_INT8, FORMAT_HWCN, FORMAT_FRACTAL_Z, src_shape, dst_shape, src.data(),
                                       dst.data(), "HWCN", "FRACTAL_Z");
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    std::set<int> src_vals;
    for (auto v : src)
        src_vals.insert(static_cast<int>(v));
    std::set<int> dst_vals;
    for (auto v : dst)
        dst_vals.insert(static_cast<int>(v));
    for (auto v : src_vals) {
        EXPECT_GT(dst_vals.count(v), 0) << "input value " << v << " missing from output";
    }
}
