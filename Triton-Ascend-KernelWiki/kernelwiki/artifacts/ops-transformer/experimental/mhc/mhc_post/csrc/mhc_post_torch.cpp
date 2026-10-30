/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

extern "C" void mhc_post_do_fp32(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_post_do_fp16(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

extern "C" void mhc_post_do_bf16(
    uint32_t blockDim, void* stream,
    uint8_t* input, uint8_t* h_post_fp32, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

torch::Tensor mhc_post_forward(
    torch::Tensor x,
    torch::Tensor h_post,
    int64_t block_dim = 0
) {
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(h_post.is_contiguous(), "Weight must be contiguous");

    int64_t batch = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t dim = x.size(2);
    int64_t num_streams = h_post.size(0);
    TORCH_CHECK(num_streams > 0, "num_streams must be > 0");

    auto output = torch::empty({batch * num_streams, seq_len, dim}, x.options());
    void* stream = c10_npu::getCurrentNPUStream().stream();
    uint32_t blk = static_cast<uint32_t>(block_dim);

    if (x.scalar_type() == torch::kFloat32) {
        mhc_post_do_fp32(blk, stream,
            reinterpret_cast<uint8_t*>(x.data_ptr<float>()),
            reinterpret_cast<uint8_t*>(h_post.data_ptr<float>()),
            reinterpret_cast<uint8_t*>(output.data_ptr<float>()),
            batch, seq_len, dim, num_streams);
    } else if (x.scalar_type() == torch::kFloat16) {
        mhc_post_do_fp16(blk, stream,
            reinterpret_cast<uint8_t*>(x.data_ptr()),
            reinterpret_cast<uint8_t*>(h_post.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            batch, seq_len, dim, num_streams);
    } else if (x.scalar_type() == torch::kBFloat16) {
        auto h_fp32 = h_post.to(torch::kFloat32).contiguous();
        mhc_post_do_bf16(blk, stream,
            reinterpret_cast<uint8_t*>(x.data_ptr()),
            reinterpret_cast<uint8_t*>(h_fp32.data_ptr<float>()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            batch, seq_len, dim, num_streams);
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", x.scalar_type());
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mhc_post_forward, "mhc_post forward (NPU, adaptive strategy)",
          py::arg("x"), py::arg("h_post"), py::arg("block_dim") = 0);
}
