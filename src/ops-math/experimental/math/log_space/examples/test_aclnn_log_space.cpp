/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_log_space.cpp
 * @brief LogSpace 算子 aclnn 两段式调用示例
 *
 * 计算公式:
 *   out[i] = base ^ (start + i * (end - start) / (steps - 1)),  i in [0, steps)
 *   steps == 1: out[0] = base^start
 *   steps == 0: 空 Tensor，不下发 Kernel
 *
 * 本示例覆盖 FLOAT / FLOAT16 / BFLOAT16 / int8 / int16 / int32 / uint8 七种输出 dtype：
 *   Case1: dtype=FLOAT    start=0.0 end=2.0  steps=5  base=10.0  -> [1, 10, 100, 1000, 10000]
 *   Case2: dtype=FLOAT16  start=-1.0 end=1.0 steps=5  base=2.0   -> [0.5, ~0.707, 1.0, ~1.414, 2.0]
 *   Case3: dtype=BFLOAT16 start=0.0 end=3.0  steps=4  base=10.0  -> [1, 10, 100, 1000]
 *   Case4: dtype=INT32    start=0.0 end=5.0  steps=6  base=2.0   -> [1, 2, 4, 8, 16, 32]
 *   Case5: dtype=INT16    start=0.0 end=4.0  steps=5  base=3.0   -> [1, 3, 9, 27, 81]
 *   Case6: dtype=INT8     start=0.0 end=6.0  steps=7  base=2.0   -> [1, 2, 4, 8, 16, 32, 64]
 *   Case7: dtype=UINT8    start=0.0 end=6.0  steps=7  base=2.0   -> [1, 2, 4, 8, 16, 32, 64]
 *   (整型 = trunc(base^x)，向零取整；取值域均落在各 dtype 范围内避免饱和)
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include "acl/acl.h"
#include "aclnn_log_space.h"

#define CHECK_RET(cond, msg, ...)                                \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "[ERROR] " msg "\n", ##__VA_ARGS__); \
            return 1;                                            \
        }                                                        \
    } while (0)

// ---------------- fp16 / bf16 转换（仅用于打印结果） ----------------
static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            int e = -1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                e -= 1;
            }
            mant &= 0x3FF;
            uint32_t fexp = static_cast<uint32_t>(127 + e);
            f = (sign << 31) | (fexp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        uint32_t fexp = exp + 127 - 15;
        f = (sign << 31) | (fexp << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static float Bf16ToFloat(uint16_t b)
{
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// ---------------- 单次调用封装 ----------------
struct ExampleCase {
    std::string name;
    aclDataType dtype;
    double start;
    double end;
    int64_t steps;
    double base;
};

static int RunOne(const ExampleCase& c, aclrtStream stream)
{
    printf("\n========================================\n");
    printf("[%s] dtype=%d start=%g end=%g steps=%lld base=%g\n", c.name.c_str(), (int)c.dtype, c.start, c.end,
           (long long)c.steps, c.base);
    printf("========================================\n");

    // 1) 构造 start / end aclScalar (ACL_FLOAT)
    float start_f = static_cast<float>(c.start);
    float end_f = static_cast<float>(c.end);
    aclScalar* startScalar = aclCreateScalar(&start_f, ACL_FLOAT);
    aclScalar* endScalar = aclCreateScalar(&end_f, ACL_FLOAT);
    CHECK_RET(startScalar && endScalar, "aclCreateScalar failed");

    // 2) 分配 result Device 内存 + 创建 aclTensor
    size_t elem_size = sizeof(float);
    switch (c.dtype) {
        case ACL_FLOAT:
            elem_size = sizeof(float);
            break;
        case ACL_FLOAT16:
        case ACL_BF16:
            elem_size = sizeof(uint16_t);
            break;
        case ACL_INT32:
            elem_size = sizeof(int32_t);
            break;
        case ACL_INT16:
            elem_size = sizeof(int16_t);
            break;
        case ACL_INT8:
            elem_size = sizeof(int8_t);
            break;
        case ACL_UINT8:
            elem_size = sizeof(uint8_t);
            break;
        default:
            elem_size = sizeof(float);
            break;
    }
    int64_t shape[1] = {c.steps};
    int64_t strides[1] = {1};
    size_t buf_size = static_cast<size_t>(c.steps) * elem_size;
    if (buf_size == 0)
        buf_size = elem_size;

    void* result_dev = nullptr;
    aclError ret = aclrtMalloc(&result_dev, buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc result failed: %d", ret);
    aclrtMemset(result_dev, buf_size, 0, buf_size);

    aclTensor* resultTensor = aclCreateTensor(shape, 1, c.dtype, strides, 0, aclFormat::ACL_FORMAT_ND, shape, 1,
                                              result_dev);
    CHECK_RET(resultTensor != nullptr, "aclCreateTensor failed");

    // 3) 调用第一段接口 GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnLogSpaceGetWorkspaceSize(startScalar, endScalar, c.steps, c.base, resultTensor, &workspaceSize,
                                        &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnLogSpaceGetWorkspaceSize failed: %d", ret);
    printf("  workspaceSize = %llu\n", (unsigned long long)workspaceSize);

    // 4) 分配 workspace (按需)
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "aclrtMalloc workspace failed: %d", ret);
    }

    // 5) 调用第二段接口 aclnnLogSpace
    ret = aclnnLogSpace(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnLogSpace failed: %d", ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed: %d", ret);

    // 6) D2H 拷贝 + 打印
    std::vector<uint8_t> host_buf(buf_size);
    ret = aclrtMemcpy(host_buf.data(), buf_size, result_dev, buf_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtMemcpy D2H failed: %d", ret);

    printf("  result = [");
    for (int64_t i = 0; i < c.steps; ++i) {
        const char* sep = (i == 0 ? "" : ", ");
        if (c.dtype == ACL_FLOAT) {
            printf("%s%g", sep, reinterpret_cast<const float*>(host_buf.data())[i]);
        } else if (c.dtype == ACL_FLOAT16) {
            printf("%s%g", sep, Fp16ToFloat(reinterpret_cast<const uint16_t*>(host_buf.data())[i]));
        } else if (c.dtype == ACL_BF16) {
            printf("%s%g", sep, Bf16ToFloat(reinterpret_cast<const uint16_t*>(host_buf.data())[i]));
        } else if (c.dtype == ACL_INT32) {
            printf("%s%d", sep, reinterpret_cast<const int32_t*>(host_buf.data())[i]);
        } else if (c.dtype == ACL_INT16) {
            printf("%s%d", sep, static_cast<int>(reinterpret_cast<const int16_t*>(host_buf.data())[i]));
        } else if (c.dtype == ACL_INT8) {
            printf("%s%d", sep, static_cast<int>(reinterpret_cast<const int8_t*>(host_buf.data())[i]));
        } else if (c.dtype == ACL_UINT8) {
            printf("%s%u", sep, static_cast<unsigned>(reinterpret_cast<const uint8_t*>(host_buf.data())[i]));
        }
    }
    printf("]\n");

    // 7) 资源释放
    if (workspace)
        aclrtFree(workspace);
    aclDestroyTensor(resultTensor);
    aclrtFree(result_dev);
    aclDestroyScalar(startScalar);
    aclDestroyScalar(endScalar);
    return 0;
}

int main(int /*argc*/, char** /*argv*/)
{
    printf("LogSpace aclnn invoke example\n");

    int32_t deviceId = 0;
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed: %d", ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed: %d", ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed: %d", ret);

    std::vector<ExampleCase> cases = {
        {"Case1_FLOAT", ACL_FLOAT, 0.0, 2.0, 5, 10.0},
        {"Case2_FLOAT16", ACL_FLOAT16, -1.0, 1.0, 5, 2.0},
        {"Case3_BFLOAT16", ACL_BF16, 0.0, 3.0, 4, 10.0},
        {"Case4_INT32", ACL_INT32, 0.0, 5.0, 6, 2.0},
        {"Case5_INT16", ACL_INT16, 0.0, 4.0, 5, 3.0},
        {"Case6_INT8", ACL_INT8, 0.0, 6.0, 7, 2.0},
        {"Case7_UINT8", ACL_UINT8, 0.0, 6.0, 7, 2.0},
        // 分数指数（测向零取整 CAST_TRUNC，对齐 torch .to(int)）
        {"Case8_INT32_frac", ACL_INT32, 0.0, 3.0, 7,
         2.0}, // 2^[0,.5,1,1.5,2,2.5,3]=[1,1.41,2,2.83,4,5.66,8] -> [1,1,2,2,4,5,8]
        {"Case9_INT16_neg", ACL_INT16, -2.0, 2.0, 5, 10.0}, // 10^[-2,-1,0,1,2]=[.01,.1,1,10,100] -> [0,0,1,10,100]
        {"Case10_INT8_frac", ACL_INT8, 0.0, 3.0, 7, 2.0},   // -> [1,1,2,2,4,5,8]
        {"Case11_UINT8_frac", ACL_UINT8, 0.0, 4.0, 9, 2.0}, // 2^[0,.5,..,4] -> [1,1,2,2,4,5,8,11,16]
        // 边界：steps==1 (SINGLE 模式，整型也走 half 中转) / steps==0 (空 tensor)
        {"Case12_INT8_s1", ACL_INT8, 2.5, 2.5, 1, 2.0},   // 2^2.5=5.66 -> [5]
        {"Case13_FP32_s1", ACL_FLOAT, 2.0, 2.0, 1, 10.0}, // 10^2 -> [100]
        {"Case14_INT32_s0", ACL_INT32, 0.0, 1.0, 0, 2.0}, // 空 tensor -> []
        // 注：torch.logspace(dtype=uint8) 在 start<0 时报溢出（torch 在 uint8 里算指数）；
        //     本算子以 fp32 算指数，能正确处理负 start（如 uint8/start=-1 得 [0,1,2,4,8]），但无 torch golden 可比。
    };

    int failed = 0;
    for (const auto& c : cases) {
        if (RunOne(c, stream) != 0)
            failed++;
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    printf("\n========================================\n");
    printf("All cases: %zu  failed: %d\n", cases.size(), failed);
    printf("========================================\n");
    return failed == 0 ? 0 : 1;
}
