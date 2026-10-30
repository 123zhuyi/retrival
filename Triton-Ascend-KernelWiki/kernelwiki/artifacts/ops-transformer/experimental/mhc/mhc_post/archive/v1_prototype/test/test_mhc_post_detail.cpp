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
 * mhc_post 算子详细验证测试
 * 
 * 论文公式验证:
 *   output[b*num_streams + s, seq, d] = branch_output[b, seq, d] * h_post[s]
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
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

bool load_binary(const char* filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return true;
}

void print_compare(const char* name, const float* expected, const float* actual, int64_t count, int64_t show = 8) {
    std::cout << "\n" << name << " 对比 (前" << show << "个值):" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "  期望:  ";
    for (int64_t i = 0; i < std::min(show, count); ++i) std::cout << expected[i] << " ";
    std::cout << std::endl;
    std::cout << "  实际:  ";
    for (int64_t i = 0; i < std::min(show, count); ++i) std::cout << actual[i] << " ";
    std::cout << std::endl;
    
    float max_diff = 0.0f, max_rel_diff = 0.0f;
    int64_t max_idx = 0;
    for (int64_t i = 0; i < count; ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        float rel_diff = (std::abs(expected[i]) > 1e-8) ? diff / std::abs(expected[i]) : diff;
        if (diff > max_diff) {
            max_diff = diff;
            max_rel_diff = rel_diff;
            max_idx = i;
        }
    }
    std::cout << "  最大绝对误差: " << std::scientific << max_diff 
              << " @ idx=" << max_idx << std::endl;
    std::cout << "  最大相对误差: " << max_rel_diff << std::endl;
}

int main() {
    const int64_t batch = 2;
    const int64_t seq_len = 4;
    const int64_t dim = 64;
    const int64_t num_streams = 4;
    
    const int64_t input_size = batch * seq_len * dim;
    const int64_t output_size = batch * num_streams * seq_len * dim;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "mhc_post AscendC 算子详细验证" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\n论文公式: output[b*s+i] = branch_output[b] * h_post[i]" << std::endl;
    std::cout << "\u53c2数: batch=" << batch << ", seq_len=" << seq_len 
              << ", dim=" << dim << ", num_streams=" << num_streams << std::endl;
    
    // 读取Python生成的测试向量
    std::vector<float> py_branch_output, py_h_post, py_expected;
    bool loaded = load_binary("../branch_output.bin", py_branch_output) &&
                  load_binary("../h_post.bin", py_h_post) &&
                  load_binary("../expected_output.bin", py_expected);
    
    if (!loaded) {
        std::cerr << "无法加载Python测试向量，使用C++生成的数据" << std::endl;
    }
    
    // ACL初始化
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) { std::cerr << "aclInit failed" << std::endl; return 1; }
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) { aclFinalize(); return 1; }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    
    // 准备数据
    std::vector<float> h_branch_output(input_size);
    std::vector<float> h_h_post(num_streams);
    std::vector<float> h_output_npu(output_size);
    std::vector<float> h_output_cpu(output_size);
    
    // 使用与Python相同的初始化
    for (int64_t i = 0; i < input_size; ++i) {
        h_branch_output[i] = static_cast<float>(i % 100) / 100.0f;
    }
    // softmax
    float sum = 0.0f;
    for (int64_t i = 0; i < num_streams; ++i) {
        h_h_post[i] = std::exp(static_cast<float>(i + 1));
        sum += h_h_post[i];
    }
    for (int64_t i = 0; i < num_streams; ++i) {
        h_h_post[i] /= sum;
    }
    
    std::cout << "\nh_post权重 (softmax归一化): ";
    for (int64_t i = 0; i < num_streams; ++i) {
        std::cout << h_h_post[i] << " ";
    }
    std::cout << std::endl;
    
    // CPU参考计算
    mhc_post_cpu(h_branch_output.data(), h_h_post.data(), h_output_cpu.data(),
                 batch, seq_len, dim, num_streams);
    
    // NPU计算
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
    
    aclrtMemcpy(h_output_npu.data(), output_size * sizeof(float),
                d_output, output_size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    
    // 验证结果
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "结果验证" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    // 1. NPU vs CPU参考
    print_compare("NPU vs CPU参考", h_output_cpu.data(), h_output_npu.data(), output_size);
    
    // 2. 如果有Python数据，NPU vs Python
    if (loaded && py_expected.size() == output_size) {
        print_compare("NPU vs Python期望", py_expected.data(), h_output_npu.data(), output_size);
    }
    
    // 3. 手工验证公式
    std::cout << "\n" << std::string(60, '-') << std::endl;
    std::cout << "手工公式验证 (output[b*s+i] = branch_output[b] * h_post[i])" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    bool all_pass = true;
    for (int64_t b = 0; b < batch && all_pass; ++b) {
        for (int64_t s = 0; s < num_streams && all_pass; ++s) {
            int64_t out_idx = (b * num_streams + s) * seq_len * dim;
            int64_t in_idx = b * seq_len * dim;
            
            float expected = h_branch_output[in_idx] * h_h_post[s];
            float actual = h_output_npu[out_idx];
            float diff = std::abs(expected - actual);
            
            const char* status = (diff < 1e-5) ? "✓" : "✗";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  output[" << b*num_streams+s << ",0,0] = "
                      << "input[" << b << ",0,0] * h_post[" << s << "] = "
                      << h_branch_output[in_idx] << " * " << h_h_post[s] 
                      << " = " << expected << " | NPU: " << actual 
                      << " " << status << std::endl;
            
            if (diff >= 1e-5) all_pass = false;
        }
    }
    
    // 统计精度
    float max_abs_err = 0.0f, max_rel_err = 0.0f, avg_err = 0.0f;
    for (int64_t i = 0; i < output_size; ++i) {
        float diff = std::abs(h_output_cpu[i] - h_output_npu[i]);
        float rel = (std::abs(h_output_cpu[i]) > 1e-8) ? diff / std::abs(h_output_cpu[i]) : diff;
        max_abs_err = std::max(max_abs_err, diff);
        max_rel_err = std::max(max_rel_err, rel);
        avg_err += diff;
    }
    avg_err /= output_size;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "精度统计" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::scientific;
    std::cout << "  最大绝对误差: " << max_abs_err << std::endl;
    std::cout << "  最大相对误差: " << max_rel_err << std::endl;
    std::cout << "  平均绝对误差: " << avg_err << std::endl;
    
    bool precision_pass = (max_abs_err < 1e-5) && (max_rel_err < 1e-4);
    std::cout << "\n精度要求: max_abs_err < 1e-5, max_rel_err < 1e-4" << std::endl;
    std::cout << "测试结果: " << (precision_pass ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    // 清理
    aclrtFree(d_branch_output);
    aclrtFree(d_h_post);
    aclrtFree(d_output);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    
    return precision_pass ? 0 : 1;
}
