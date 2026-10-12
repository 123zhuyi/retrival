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
 * mhc_post 精度详细测试
 * 数据类型: float32 (FP32)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include "acl/acl.h"

extern "C" void mhc_post_do(
    uint32_t blockDim, void* stream,
    uint8_t* branch_output, uint8_t* h_post, uint8_t* output,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t num_streams
);

void softmax(float* data, int64_t n) {
    float max_val = *std::max_element(data, data + n);
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

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "mhc_post 算子精度详细分析" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "\n[数据类型]" << std::endl;
    std::cout << "  输入 branch_output: float32 (FP32)" << std::endl;
    std::cout << "  输入 h_post:        float32 (FP32)" << std::endl;
    std::cout << "  输出 output:        float32 (FP32)" << std::endl;
    std::cout << "  计算精度:           float32 (FP32)" << std::endl;
    
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    
    // 测试配置
    struct TestConfig {
        const char* name;
        int64_t batch, seq_len, dim, num_streams;
        int seed;
        float input_scale;  // 输入数据范围
    };
    
    TestConfig configs[] = {
        {"小数值输入", 2, 8, 128, 4, 42, 0.01f},
        {"标准正态分布", 2, 8, 128, 4, 123, 1.0f},
        {"大数值输入", 2, 8, 128, 4, 456, 100.0f},
        {"混合正负值", 4, 16, 256, 4, 789, 10.0f},
        {"大规模测试", 8, 32, 256, 8, 999, 1.0f},
    };
    
    std::cout << "\n[精度测试结果]" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(20) << "测试名称"
              << std::right << std::setw(12) << "最大绝对误差"
              << std::setw(12) << "最大相对误差"
              << std::setw(12) << "平均误差"
              << std::setw(12) << "状态" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::vector<float> all_abs_errors, all_rel_errors;
    
    for (const auto& cfg : configs) {
        int64_t in_size = cfg.batch * cfg.seq_len * cfg.dim;
        int64_t out_size = cfg.batch * cfg.num_streams * cfg.seq_len * cfg.dim;
        
        std::vector<float> h_input(in_size), h_hpost(cfg.num_streams);
        std::vector<float> h_output_cpu(out_size), h_output_npu(out_size);
        
        // 随机初始化
        std::mt19937 gen(cfg.seed);
        std::normal_distribution<float> dist(0.0f, cfg.input_scale);
        for (auto& v : h_input) v = dist(gen);
        for (auto& v : h_hpost) v = dist(gen);
        softmax(h_hpost.data(), cfg.num_streams);
        
        // CPU参考
        mhc_post_cpu(h_input.data(), h_hpost.data(), h_output_cpu.data(),
                     cfg.batch, cfg.seq_len, cfg.dim, cfg.num_streams);
        
        // NPU计算
        void *d_in, *d_hp, *d_out;
        aclrtMalloc(&d_in, in_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&d_hp, cfg.num_streams * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&d_out, out_size * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclrtMemcpy(d_in, in_size*4, h_input.data(), in_size*4, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(d_hp, cfg.num_streams*4, h_hpost.data(), cfg.num_streams*4, ACL_MEMCPY_HOST_TO_DEVICE);
        
        mhc_post_do(cfg.batch * cfg.num_streams, stream,
                    (uint8_t*)d_in, (uint8_t*)d_hp, (uint8_t*)d_out,
                    cfg.batch, cfg.seq_len, cfg.dim, cfg.num_streams);
        aclrtSynchronizeStream(stream);
        
        aclrtMemcpy(h_output_npu.data(), out_size*4, d_out, out_size*4, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtFree(d_in); aclrtFree(d_hp); aclrtFree(d_out);
        
        // 精度统计
        float max_abs = 0, max_rel = 0, sum_abs = 0;
        int64_t err_count = 0;
        for (int64_t i = 0; i < out_size; ++i) {
            float diff = std::abs(h_output_cpu[i] - h_output_npu[i]);
            float rel = diff / (std::abs(h_output_cpu[i]) + 1e-10f);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, rel);
            sum_abs += diff;
            if (diff > 0) err_count++;
            all_abs_errors.push_back(diff);
            all_rel_errors.push_back(rel);
        }
        
        std::cout << std::left << std::setw(20) << cfg.name
                  << std::right << std::scientific << std::setprecision(2)
                  << std::setw(12) << max_abs
                  << std::setw(12) << max_rel
                  << std::setw(12) << sum_abs / out_size
                  << std::setw(12) << (max_abs < 1e-5 ? "✓ PASS" : "✗ FAIL")
                  << std::endl;
    }
    
    // 总体统计
    std::cout << std::string(70, '=') << std::endl;
    
    float global_max_abs = *std::max_element(all_abs_errors.begin(), all_abs_errors.end());
    float global_max_rel = *std::max_element(all_rel_errors.begin(), all_rel_errors.end());
    float global_avg = std::accumulate(all_abs_errors.begin(), all_abs_errors.end(), 0.0f) / all_abs_errors.size();
    
    int64_t zero_err_count = std::count_if(all_abs_errors.begin(), all_abs_errors.end(), 
                                           [](float x) { return x == 0.0f; });
    
    std::cout << "\n[总体精度统计]" << std::endl;
    std::cout << "  测试数据点总数: " << all_abs_errors.size() << std::endl;
    std::cout << "  完全一致的点数: " << zero_err_count 
              << " (" << std::fixed << std::setprecision(2) 
              << 100.0 * zero_err_count / all_abs_errors.size() << "%)" << std::endl;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  全局最大绝对误差: " << global_max_abs << std::endl;
    std::cout << "  全局最大相对误差: " << global_max_rel << std::endl;
    std::cout << "  全局平均绝对误差: " << global_avg << std::endl;
    
    std::cout << "\n[精度分析]" << std::endl;
    if (global_max_abs == 0.0f) {
        std::cout << "  NPU计算结果与CPU参考实现 **完全一致** (bit-exact)" << std::endl;
        std::cout << "  说明: AscendC的Muls指令在float32精度下与CPU结果完全相同" << std::endl;
    } else {
        std::cout << "  存在微小精度差异，但在FP32误差允许范围内" << std::endl;
    }
    
    std::cout << "\n[结论]" << std::endl;
    bool passed = global_max_abs < 1e-5;
    std::cout << "  精度要求: max_abs_error < 1e-5" << std::endl;
    std::cout << "  测试结果: " << (passed ? "✓ PASSED" : "✗ FAILED") << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return passed ? 0 : 1;
}
