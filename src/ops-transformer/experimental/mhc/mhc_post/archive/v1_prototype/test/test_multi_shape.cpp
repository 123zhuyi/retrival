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
 * mhc_post 多种shape综合测试
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include "acl/acl.h"

extern "C" void mhc_post_do(
    uint32_t blockDim, void* stream,
    uint8_t* branch_output, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

void softmax(float* data, int64_t n) {
    float max_val = data[0];
    for (int64_t i = 1; i < n; ++i) if (data[i] > max_val) max_val = data[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) { data[i] = std::exp(data[i] - max_val); sum += data[i]; }
    for (int64_t i = 0; i < n; ++i) data[i] /= sum;
}

void mhc_post_cpu(const float* input, const float* h_post, float* output,
                  int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < num_streams; ++s) {
            float w = h_post[s];
            int64_t out_b = b * num_streams + s;
            for (int64_t i = 0; i < seq_len * dim; ++i) {
                output[out_b * seq_len * dim + i] = input[b * seq_len * dim + i] * w;
            }
        }
    }
}

bool test_case(aclrtStream stream, const char* name,
               int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams) {
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "测试: " << name << std::endl;
    std::cout << "  batch=" << batch << ", seq_len=" << seq_len 
              << ", dim=" << dim << ", num_streams=" << num_streams << std::endl;
    
    int64_t in_size = batch * seq_len * dim;
    int64_t out_size = batch * num_streams * seq_len * dim;
    
    std::vector<float> h_input(in_size), h_hpost(num_streams);
    std::vector<float> h_output_cpu(out_size), h_output_npu(out_size);
    
    // 随机初始化
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_input) v = dist(gen);
    for (auto& v : h_hpost) v = dist(gen);
    softmax(h_hpost.data(), num_streams);
    
    // CPU参考
    mhc_post_cpu(h_input.data(), h_hpost.data(), h_output_cpu.data(),
                 batch, seq_len, dim, num_streams);
    
    // NPU计算
    void *d_in, *d_hp, *d_out;
    aclrtMalloc(&d_in, in_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_hp, num_streams * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d_out, out_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    
    aclrtMemcpy(d_in, in_size*4, h_input.data(), in_size*4, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_hp, num_streams*4, h_hpost.data(), num_streams*4, ACL_MEMCPY_HOST_TO_DEVICE);
    
    mhc_post_do(batch * num_streams, stream,
                (uint8_t*)d_in, (uint8_t*)d_hp, (uint8_t*)d_out,
                batch, seq_len, dim, num_streams);
    aclrtSynchronizeStream(stream);
    
    aclrtMemcpy(h_output_npu.data(), out_size*4, d_out, out_size*4, ACL_MEMCPY_DEVICE_TO_HOST);
    
    aclrtFree(d_in); aclrtFree(d_hp); aclrtFree(d_out);
    
    // 精度检查
    float max_abs = 0, max_rel = 0, sum_err = 0;
    for (int64_t i = 0; i < out_size; ++i) {
        float diff = std::abs(h_output_cpu[i] - h_output_npu[i]);
        float rel = diff / (std::abs(h_output_cpu[i]) + 1e-8f);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, rel);
        sum_err += diff;
    }
    
    std::cout << std::scientific << std::setprecision(2);
    std::cout << "  最大绝对误差: " << max_abs << std::endl;
    std::cout << "  最大相对误差: " << max_rel << std::endl;
    std::cout << "  平均绝对误差: " << sum_err / out_size << std::endl;
    
    bool passed = (max_abs < 1e-4) && (max_rel < 1e-3);
    std::cout << "  结果: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    return passed;
}

int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "mhc_post 算子多种shape综合测试" << std::endl;
    std::cout << "论文公式: output[b*s+i] = branch_output[b] * h_post[i]" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    int passed = 0, total = 0;
    
    // 测试用例: name, batch, seq_len, dim, num_streams
    struct TestCase { const char* name; int64_t b, s, d, n; };
    TestCase cases[] = {
        {"基础测试", 2, 4, 64, 4},
        {"单batch", 1, 8, 128, 4},
        {"大batch", 8, 4, 64, 4},
        {"长序列", 2, 32, 64, 4},
        {"大dim", 2, 4, 256, 4},
        {"多streams", 2, 4, 64, 8},
        {"典型配置", 4, 16, 128, 4},
        {"边界: dim=32", 1, 2, 32, 2},
        {"边界: streams=2", 2, 4, 64, 2},
        {"大规模", 4, 64, 256, 4},
    };
    
    for (const auto& tc : cases) {
        if (test_case(stream, tc.name, tc.b, tc.s, tc.d, tc.n)) passed++;
        total++;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "测试汇总: " << passed << "/" << total << " 通过" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return (passed == total) ? 0 : 1;
}
