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
/**
 * mhc_post 算子测试用例
 * 
 * mHC论文 (DeepSeek 2024.12.31): https://arxiv.org/abs/2512.24880
 * mhc_post操作: 将branch输出乘以H_post权重分发到多个streams
 * 
 * 输入:
 *   - branch_output: [batch, seq_len, dim] 分支模块输出
 *   - h_post: [num_streams] 分发权重(经softmax归一化)
 * 
 * 输出:
 *   - output: [batch * num_streams, seq_len, dim]
 *   - output[b*num_streams + s, :, :] = branch_output[b, :, :] * h_post[s]
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"

extern "C" void mhc_post_do(
    uint32_t blockDim,
    void* stream,
    uint8_t* branch_output,
    uint8_t* h_post,
    uint8_t* output,
    int64_t batch,
    int64_t seq_len,
    int64_t dim,
    int64_t num_streams
);

void mhc_post_cpu(
    const float* branch_output,
    const float* h_post,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t dim,
    int64_t num_streams
) {
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < num_streams; ++s) {
            float weight = h_post[s];
            int64_t out_batch_idx = b * num_streams + s;
            for (int64_t seq = 0; seq < seq_len; ++seq) {
                for (int64_t d = 0; d < dim; ++d) {
                    int64_t in_idx = b * seq_len * dim + seq * dim + d;
                    int64_t out_idx = out_batch_idx * seq_len * dim + seq * dim + d;
                    output[out_idx] = branch_output[in_idx] * weight;
                }
            }
        }
    }
}

void softmax(float* data, int64_t n) {
    float max_val = data[0];
    for (int64_t i = 1; i < n; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    for (int64_t i = 0; i < n; ++i) {
        data[i] /= sum;
    }
}

bool compare_results(const float* expected, const float* actual, int64_t size, float tolerance = 1e-4f) {
    for (int64_t i = 0; i < size; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        if (diff > tolerance) {
            std::cout << "Mismatch at index " << i << ": expected=" << expected[i] 
                      << ", actual=" << actual[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int64_t batch = 2;
    const int64_t seq_len = 4;
    const int64_t dim = 64;
    const int64_t num_streams = 4;
    
    const int64_t input_size = batch * seq_len * dim;
    const int64_t output_size = batch * num_streams * seq_len * dim;
    
    std::cout << "=== mhc_post Operator Test ===" << std::endl;
    std::cout << "batch=" << batch << ", seq_len=" << seq_len 
              << ", dim=" << dim << ", num_streams=" << num_streams << std::endl;
    
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclInit failed: " << ret << std::endl;
        return 1;
    }
    
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed: " << ret << std::endl;
        aclFinalize();
        return 1;
    }
    
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtCreateStream failed: " << ret << std::endl;
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
    
    std::vector<float> h_branch_output(input_size);
    std::vector<float> h_h_post(num_streams);
    std::vector<float> h_output(output_size);
    std::vector<float> h_output_cpu(output_size);
    
    for (int64_t i = 0; i < input_size; ++i) {
        h_branch_output[i] = static_cast<float>(i % 100) / 100.0f;
    }
    
    for (int64_t i = 0; i < num_streams; ++i) {
        h_h_post[i] = static_cast<float>(i + 1);
    }
    softmax(h_h_post.data(), num_streams);
    
    std::cout << "h_post weights: ";
    for (int64_t i = 0; i < num_streams; ++i) {
        std::cout << h_h_post[i] << " ";
    }
    std::cout << std::endl;
    
    mhc_post_cpu(h_branch_output.data(), h_h_post.data(), h_output_cpu.data(),
                 batch, seq_len, dim, num_streams);
    
    void *d_branch_output, *d_h_post, *d_output;
    aclrtMalloc(&d_branch_output, input_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_h_post, num_streams * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_output, output_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtMemcpy(d_branch_output, input_size * sizeof(float), 
                h_branch_output.data(), input_size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_h_post, num_streams * sizeof(float),
                h_h_post.data(), num_streams * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    
    uint32_t blockDim = batch * num_streams;
    mhc_post_do(blockDim, stream,
                reinterpret_cast<uint8_t*>(d_branch_output),
                reinterpret_cast<uint8_t*>(d_h_post),
                reinterpret_cast<uint8_t*>(d_output),
                batch, seq_len, dim, num_streams);
    
    aclrtSynchronizeStream(stream);
    
    aclrtMemcpy(h_output.data(), output_size * sizeof(float),
                d_output, output_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    bool passed = compare_results(h_output_cpu.data(), h_output.data(), output_size);
    
    std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;
    
    aclrtFree(d_branch_output);
    aclrtFree(d_h_post);
    aclrtFree(d_output);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return passed ? 0 : 1;
}
