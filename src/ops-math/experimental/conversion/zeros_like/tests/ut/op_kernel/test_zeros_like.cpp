/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_zeros_like.cpp
 * \brief experimental 自包含 ascend910b (DAV_2201) 标准 AscendC kernel 单元测试（tikicpulib CPU 模拟执行）。
 *        被测对象：experimental/conversion/zeros_like/op_kernel/zeros_like.cpp
 *                  （template<int BYTE_KEY> + if constexpr，Duplicate+DataCopy/DataCopyPad 写全 0）。
 *
 *        执行后端：**tikicpulib CPU 模拟**（ICPU_RUN_KF，链接 tikicpulib::ascend910b）。
 *
 *        驱动方式：用 ExecuteTiling 跑真实 Tiling4ZerosLike（得到正确 tilingKey 下标 + TilingData），
 *                  再以该 TilingData 直接驱动 kernel；golden=全 0，逐字节断言二进制完全一致(绝对/相对误差均为 0)。
 *                  输入 x 预填非零(0xA5)，验证 kernel「不读输入、只写 0」。
 *
 *        覆盖：4 字节宽度桶（1/2/4/8B）代表 dtype × 代表 shape：
 *          - int8/uint8/bool (1B)：33x17(对齐主体+非对齐尾) / 单元素
 *          - fp16/bf16 (2B)：33x17 / 单元素
 *          - fp32/int32 (4B)：33x17 / 单核小 shape
 *          - int64 (8B)：33x17 / 单元素 / 大 shape 多核
 *          - 多核满载（大 shape）/ 单核（小 shape）/ 非对齐尾块 / 单元素 / 空 tensor
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/zeros_like_tiling.h"
#include "../../../op_kernel/zeros_like_tiling_data.h"

using namespace std;
using ZerosLikeNs::ZerosLikeTilingData;

// 被测 kernel 的 4 个字节宽度桶非模板 extern "C" 包装（定义在 zeros_like_kernel_inst.cpp，
// 该 TU 单独 #include kernel 源码并实例化模板，隔离 kernel_operator.h 与 gtest/<iostream> 的符号冲突）。
extern "C" __global__ __aicore__ void zeros_like_1b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" __global__ __aicore__ void zeros_like_2b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" __global__ __aicore__ void zeros_like_4b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
extern "C" __global__ __aicore__ void zeros_like_8b(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint64_t ZL_UT_BLOCK_BYTES = 32;

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

optiling::ZerosLikeCompileInfo MakeCompileInfo(int32_t coreNum, int64_t ubSize)
{
    optiling::ZerosLikeCompileInfo info;
    info.totalCoreNum = coreNum;
    info.ubSize = ubSize;
    return info;
}

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    for (auto d : dims) {
        ss.MutableOriginShape().AppendDim(d);
        ss.MutableStorageShape().AppendDim(d);
    }
    return ss;
}

uint64_t ElemNum(const std::vector<int64_t>& shape)
{
    uint64_t n = 1;
    bool empty = false;
    for (auto d : shape) {
        if (d == 0) {
            empty = true;
        }
        n *= static_cast<uint64_t>(d);
    }
    return empty ? 0 : n;
}
} // namespace

class test_zeros_like : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "zeros_like_kernel_test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "zeros_like_kernel_test TearDown" << std::endl;
    }

    // 通用流程：真实 tiling → CPU 模拟 kernel → 逐字节断言全 0（bitwise）。
    // 输入 x 预填 0xA5（非零）以验证 kernel 不读输入仅写出全 0。
    void RunKernelAndCheckZero(
        const std::vector<int64_t>& shape, ge::DataType dtype, uint32_t bytesPerElem, uint32_t coreNum, uint64_t ubSize)
    {
        // 1) 跑真实 host tiling（Tiling4ZerosLike），拿到 tilingKey 下标 + TilingData。
        optiling::ZerosLikeCompileInfo compileInfo =
            MakeCompileInfo(static_cast<int32_t>(coreNum), static_cast<int64_t>(ubSize));
        gert::StorageShape ss = MakeShape(shape);
        gert::TilingContextPara para(
            "ZerosLike", {{ss, dtype, ge::FORMAT_ND}}, {{ss, dtype, ge::FORMAT_ND}}, &compileInfo, coreNum, ubSize);
        // ::TilingInfo 显式取全局（tiling_case_executor.h）；kernel cpp 的 using namespace AscendC
        // 引入了 AscendC::TilingInfo 造成歧义，故全限定。
        ::TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling should return GRAPH_SUCCESS";
        ASSERT_GE(info.tilingDataSize, sizeof(ZerosLikeTilingData));

        uint64_t elemNum = ElemNum(shape);
        size_t outputByteSize = static_cast<size_t>(elemNum) * bytesPerElem;

        // 2) 分配 GM：输入 x 预填非零；输出 y 预填非零，验证确被写 0。
        size_t allocOut = outputByteSize == 0 ? ZL_UT_BLOCK_BYTES : CeilAlign(outputByteSize, ZL_UT_BLOCK_BYTES);
        size_t allocIn = allocOut; // x 与 y 同 shape 同 dtype
        uint8_t* x = (uint8_t*)AscendC::GmAlloc(allocIn);
        uint8_t* y = (uint8_t*)AscendC::GmAlloc(allocOut);
        std::memset(x, 0xA5, allocIn);
        std::memset(y, 0xA5, allocOut);

        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(
            info.workspaceSizes.empty() ? ZL_UT_BLOCK_BYTES :
                                          (info.workspaceSizes[0] == 0 ? ZL_UT_BLOCK_BYTES : info.workspaceSizes[0]));
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(info.tilingDataSize);
        std::memcpy(tiling, info.tilingData.get(), info.tilingDataSize);

        // 3) CPU 模拟执行 kernel。blockNum 来自真实 tiling 的 SetBlockDim。
        //    kernel 为 template<int BYTE_KEY>，按字节宽度桶选对应实例（与 if constexpr 分支一一对应）。
        ICPU_SET_TILING_KEY(static_cast<uint64_t>(info.tilingKey));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ASSERT_GT(info.blockNum, 0u);
        switch (bytesPerElem) {
            case 1:
                ICPU_RUN_KF(zeros_like_1b, info.blockNum, x, y, workspace, tiling);
                break;
            case 2:
                ICPU_RUN_KF(zeros_like_2b, info.blockNum, x, y, workspace, tiling);
                break;
            case 4:
                ICPU_RUN_KF(zeros_like_4b, info.blockNum, x, y, workspace, tiling);
                break;
            case 8:
                ICPU_RUN_KF(zeros_like_8b, info.blockNum, x, y, workspace, tiling);
                break;
            default:
                FAIL() << "unexpected bytesPerElem=" << bytesPerElem;
        }

        // 4) golden：输出全 0 字节，逐字节断言（bitwise，绝对/相对误差均为 0）。
        bool allZero = true;
        int firstBad = -1;
        for (size_t i = 0; i < outputByteSize; i++) {
            if (y[i] != 0) {
                allZero = false;
                firstBad = static_cast<int>(i);
                break;
            }
        }
        EXPECT_TRUE(allZero) << "output not all-zero, first non-zero byte at index " << firstBad
                             << " value=" << (firstBad >= 0 ? (int)y[firstBad] : -1)
                             << " (bytesPerElem=" << bytesPerElem << ", elemNum=" << elemNum << ")";

        AscendC::GmFree((void*)x);
        AscendC::GmFree((void*)y);
        AscendC::GmFree((void*)workspace);
        AscendC::GmFree((void*)tiling);
    }

    // 输入 x 预填给定 32-bit 位模式（小端循环填充，覆盖 fp16/fp32 的
    // NaN/Inf 精确位），驱动 kernel 后断言输出 y 仍 bitwise 全 0。
    // 与基线 0xA5 填充不同：本路径写入「精确的 NaN/Inf 浮点位模式」，证明 kernel
    // 物理上不 CopyIn x（不读输入），NaN/Inf 不被读取、不传播到输出。
    void RunKernelWithInputPatternAndCheckZero(
        const std::vector<int64_t>& shape, ge::DataType dtype, uint32_t bytesPerElem, uint32_t coreNum, uint64_t ubSize,
        uint32_t fillPattern32)
    {
        optiling::ZerosLikeCompileInfo compileInfo =
            MakeCompileInfo(static_cast<int32_t>(coreNum), static_cast<int64_t>(ubSize));
        gert::StorageShape ss = MakeShape(shape);
        gert::TilingContextPara para(
            "ZerosLike", {{ss, dtype, ge::FORMAT_ND}}, {{ss, dtype, ge::FORMAT_ND}}, &compileInfo, coreNum, ubSize);
        ::TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling should return GRAPH_SUCCESS";
        ASSERT_GE(info.tilingDataSize, sizeof(ZerosLikeTilingData));

        uint64_t elemNum = ElemNum(shape);
        size_t outputByteSize = static_cast<size_t>(elemNum) * bytesPerElem;
        size_t allocOut = outputByteSize == 0 ? ZL_UT_BLOCK_BYTES : CeilAlign(outputByteSize, ZL_UT_BLOCK_BYTES);
        size_t allocIn = allocOut;

        uint8_t* x = (uint8_t*)AscendC::GmAlloc(allocIn);
        uint8_t* y = (uint8_t*)AscendC::GmAlloc(allocOut);
        // 输入按 32-bit 位模式循环填充（小端），输出预填非零验证确被写 0。
        for (size_t i = 0; i < allocIn; i++) {
            x[i] = static_cast<uint8_t>((fillPattern32 >> ((i % 4) * 8)) & 0xFF);
        }
        std::memset(y, 0xA5, allocOut);

        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(
            info.workspaceSizes.empty() ? ZL_UT_BLOCK_BYTES :
                                          (info.workspaceSizes[0] == 0 ? ZL_UT_BLOCK_BYTES : info.workspaceSizes[0]));
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(info.tilingDataSize);
        std::memcpy(tiling, info.tilingData.get(), info.tilingDataSize);

        ICPU_SET_TILING_KEY(static_cast<uint64_t>(info.tilingKey));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ASSERT_GT(info.blockNum, 0u);
        switch (bytesPerElem) {
            case 1:
                ICPU_RUN_KF(zeros_like_1b, info.blockNum, x, y, workspace, tiling);
                break;
            case 2:
                ICPU_RUN_KF(zeros_like_2b, info.blockNum, x, y, workspace, tiling);
                break;
            case 4:
                ICPU_RUN_KF(zeros_like_4b, info.blockNum, x, y, workspace, tiling);
                break;
            case 8:
                ICPU_RUN_KF(zeros_like_8b, info.blockNum, x, y, workspace, tiling);
                break;
            default:
                FAIL() << "unexpected bytesPerElem=" << bytesPerElem;
        }

        bool allZero = true;
        int firstBad = -1;
        for (size_t i = 0; i < outputByteSize; i++) {
            if (y[i] != 0) {
                allZero = false;
                firstBad = static_cast<int>(i);
                break;
            }
        }
        EXPECT_TRUE(allZero) << "output not all-zero with input pattern 0x" << std::hex << fillPattern32 << std::dec
                             << ", first non-zero byte at index " << firstBad
                             << " value=" << (firstBad >= 0 ? (int)y[firstBad] : -1);

        AscendC::GmFree((void*)x);
        AscendC::GmFree((void*)y);
        AscendC::GmFree((void*)workspace);
        AscendC::GmFree((void*)tiling);
    }

    // 同一 shape/dtype/tiling 连续执行 kernel 2 次，断言两次输出
    // bitwise 完全一致（且均全 0）。验证 spec determinism.bitwise_reproducible:true。
    void RunKernelTwiceAndCheckDeterministic(
        const std::vector<int64_t>& shape, ge::DataType dtype, uint32_t bytesPerElem, uint32_t coreNum, uint64_t ubSize)
    {
        optiling::ZerosLikeCompileInfo compileInfo =
            MakeCompileInfo(static_cast<int32_t>(coreNum), static_cast<int64_t>(ubSize));
        gert::StorageShape ss = MakeShape(shape);
        gert::TilingContextPara para(
            "ZerosLike", {{ss, dtype, ge::FORMAT_ND}}, {{ss, dtype, ge::FORMAT_ND}}, &compileInfo, coreNum, ubSize);
        ::TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << "tiling should return GRAPH_SUCCESS";
        ASSERT_GE(info.tilingDataSize, sizeof(ZerosLikeTilingData));

        uint64_t elemNum = ElemNum(shape);
        size_t outputByteSize = static_cast<size_t>(elemNum) * bytesPerElem;
        size_t allocOut = outputByteSize == 0 ? ZL_UT_BLOCK_BYTES : CeilAlign(outputByteSize, ZL_UT_BLOCK_BYTES);

        uint8_t* x = (uint8_t*)AscendC::GmAlloc(allocOut);
        uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(allocOut);
        uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(allocOut);
        std::memset(x, 0xA5, allocOut);
        // 两次输出 buffer 预填不同非零字节，确保「相同」结论来自 kernel 写出而非残留。
        std::memset(y1, 0x5A, allocOut);
        std::memset(y2, 0x3C, allocOut);

        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(
            info.workspaceSizes.empty() ? ZL_UT_BLOCK_BYTES :
                                          (info.workspaceSizes[0] == 0 ? ZL_UT_BLOCK_BYTES : info.workspaceSizes[0]));
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(info.tilingDataSize);
        std::memcpy(tiling, info.tilingData.get(), info.tilingDataSize);

        ICPU_SET_TILING_KEY(static_cast<uint64_t>(info.tilingKey));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ASSERT_GT(info.blockNum, 0u);

        auto runOnce = [&](uint8_t* yOut) {
            switch (bytesPerElem) {
                case 1:
                    ICPU_RUN_KF(zeros_like_1b, info.blockNum, x, yOut, workspace, tiling);
                    break;
                case 2:
                    ICPU_RUN_KF(zeros_like_2b, info.blockNum, x, yOut, workspace, tiling);
                    break;
                case 4:
                    ICPU_RUN_KF(zeros_like_4b, info.blockNum, x, yOut, workspace, tiling);
                    break;
                case 8:
                    ICPU_RUN_KF(zeros_like_8b, info.blockNum, x, yOut, workspace, tiling);
                    break;
                default:
                    FAIL() << "unexpected bytesPerElem=" << bytesPerElem;
            }
        };
        runOnce(y1);
        runOnce(y2);

        // 两次输出 bitwise 完全一致 + 均全 0。
        EXPECT_EQ(std::memcmp(y1, y2, outputByteSize), 0) << "two runs not bitwise identical (non-deterministic)";
        bool allZero1 = true, allZero2 = true;
        for (size_t i = 0; i < outputByteSize; i++) {
            if (y1[i] != 0) {
                allZero1 = false;
            }
            if (y2[i] != 0) {
                allZero2 = false;
            }
        }
        EXPECT_TRUE(allZero1) << "run#1 output not all-zero";
        EXPECT_TRUE(allZero2) << "run#2 output not all-zero";

        AscendC::GmFree((void*)x);
        AscendC::GmFree((void*)y1);
        AscendC::GmFree((void*)y2);
        AscendC::GmFree((void*)workspace);
        AscendC::GmFree((void*)tiling);
    }
};

// ===================== 1B 桶（int8/uint8/bool）：UB uint16 视图，按字节长度写出 =====================
// 33x17=561 字节，含 32B 对齐主体(17×32=544) + 17 字节非对齐尾块（DataCopyPad）。
TEST_F(test_zeros_like, kernel_int8_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_INT8, 1, 40, 262144);
}
TEST_F(test_zeros_like, kernel_uint8_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_UINT8, 1, 40, 262144);
}
TEST_F(test_zeros_like, kernel_bool_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_BOOL, 1, 40, 262144);
}
// 单元素：纯非对齐尾块路径（1 字节）。
TEST_F(test_zeros_like, kernel_int8_single_element_tail_only)
{
    RunKernelAndCheckZero({1}, ge::DT_INT8, 1, 40, 262144);
}

// ===================== 2B 桶（fp16/bf16）：UB uint16 视图 =====================
// 33x17=561 元素 ×2B=1122 字节，含对齐主体 + 2 字节尾块。
TEST_F(test_zeros_like, kernel_fp16_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_FLOAT16, 2, 40, 262144);
}
TEST_F(test_zeros_like, kernel_bf16_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_BF16, 2, 40, 262144);
}
// fp16 单元素（2 字节，纯尾块）。
TEST_F(test_zeros_like, kernel_fp16_single_element_tail_only)
{
    RunKernelAndCheckZero({1}, ge::DT_FLOAT16, 2, 40, 262144);
}

// ===================== 4B 桶（fp32/int32）：UB uint32 视图 =====================
// 33x17=561 元素 ×4B=2244 字节，含对齐主体 + 4 字节尾块。
TEST_F(test_zeros_like, kernel_fp32_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_FLOAT, 4, 40, 262144);
}
TEST_F(test_zeros_like, kernel_int32_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_INT32, 4, 40, 262144);
}
// 4B 单核小 shape（8 元素=32 字节，恰 1 个 32B 块，单核纯对齐）。
TEST_F(test_zeros_like, kernel_fp32_single_core_aligned_8elem)
{
    RunKernelAndCheckZero({8}, ge::DT_FLOAT, 4, 40, 262144);
}

// ===================== 8B 桶（int64）：UB uint32 视图（2×elem） =====================
// 33x17=561 元素 ×8B=4488 字节，含对齐主体 + 8 字节尾块。
TEST_F(test_zeros_like, kernel_int64_33x17_aligned_and_tail)
{
    RunKernelAndCheckZero({33, 17}, ge::DT_INT64, 8, 40, 262144);
}
// int64 单元素（8 字节，纯尾块）。
TEST_F(test_zeros_like, kernel_int64_single_element_tail_only)
{
    RunKernelAndCheckZero({1}, ge::DT_INT64, 8, 40, 262144);
}

// ===================== 多核满载 / 单核 边界 =====================
// 大 shape 多核满载：4096×4096 fp16（32MB），blockDim 应取满核数，逐核字节切分正确。
TEST_F(test_zeros_like, kernel_fp16_large_shape_multicore)
{
    RunKernelAndCheckZero({4096, 4096}, ge::DT_FLOAT16, 2, 40, 262144);
}
// 大 shape int64 多核（含每核内 UB 多次循环写出）。
TEST_F(test_zeros_like, kernel_int64_large_shape_multicore)
{
    RunKernelAndCheckZero({1024, 1024}, ge::DT_INT64, 8, 40, 262144);
}
// 小 ubSize 触发单核内 UB 多 chunk 循环（tileBytes 受限）。
TEST_F(test_zeros_like, kernel_fp32_small_ub_multi_chunk)
{
    RunKernelAndCheckZero({4096}, ge::DT_FLOAT, 4, 1, 49152);
}

// ===================== rank=0 标量 [] =====================
// Shape::GetShapeSize() 对 dim_num_==0 返回 1 → 单元素退化（blockDim=1，纯尾块写出）。
// fp32(4B)：1 元素 = 4 字节，纯非对齐尾块 DataCopyPad；断言输出 bitwise 全 0。
TEST_F(test_zeros_like, kernel_rank0_scalar_fp32)
{
    RunKernelAndCheckZero({}, ge::DT_FLOAT, 4, 40, 262144);
}

// ===================== 8 维 rank 上限 [2,1,2,1,2,1,2,3] =====================
// 48 元素 × 8B(int64) = 384 字节 = 12 个 32B 块；coreNum=5 → usedCore=5,
// tailCoreNum=12%5=2(>0)，验证 8 维 shape 下多核切分 + 尾块写出路径，输出全 0。
TEST_F(test_zeros_like, kernel_8dim_int64_multicore_tail)
{
    RunKernelAndCheckZero({2, 1, 2, 1, 2, 1, 2, 3}, ge::DT_INT64, 8, 5, 262144);
}

// ===================== NaN/Inf 精确位输入 → 输出仍 bitwise 全 0 =====================
// 区别于基线 0xA5 填充：输入写入「精确的 NaN/Inf 浮点位模式」，证明 kernel 不读 x、
// NaN/Inf 不传播。fp16 位模式按 16-bit 在 32-bit 内重复（高低半字相同），小端循环填充。
// fp16: qNaN=0x7E00, +Inf=0x7C00；fp32: qNaN=0x7FC00000, +Inf=0x7F800000。
TEST_F(test_zeros_like, kernel_fp16_input_nan_output_zero)
{
    // 0x7E007E00：低/高 16-bit 均为 fp16 qNaN，按 uint16 视图每元素都是 NaN 位模式。
    RunKernelWithInputPatternAndCheckZero({8}, ge::DT_FLOAT16, 2, 40, 262144, 0x7E007E00u);
}
TEST_F(test_zeros_like, kernel_fp16_input_posinf_output_zero)
{
    // 0x7C007C00：fp16 +Inf 位模式。
    RunKernelWithInputPatternAndCheckZero({8}, ge::DT_FLOAT16, 2, 40, 262144, 0x7C007C00u);
}
TEST_F(test_zeros_like, kernel_fp32_input_nan_output_zero)
{
    // 0x7FC00000：fp32 qNaN 位模式（按 uint32 视图每元素都是 NaN）。
    RunKernelWithInputPatternAndCheckZero({8}, ge::DT_FLOAT, 4, 40, 262144, 0x7FC00000u);
}
TEST_F(test_zeros_like, kernel_fp32_input_posinf_output_zero)
{
    // 0x7F800000：fp32 +Inf 位模式。
    RunKernelWithInputPatternAndCheckZero({8}, ge::DT_FLOAT, 4, 40, 262144, 0x7F800000u);
}
// 非对齐尾块 + NaN 输入组合（33 元素，含 DataCopyPad 尾），覆盖尾块路径下不读输入。
TEST_F(test_zeros_like, kernel_fp32_input_nan_tail_output_zero)
{
    RunKernelWithInputPatternAndCheckZero({33}, ge::DT_FLOAT, 4, 40, 262144, 0x7FC00000u);
}

// ===================== determinism（同输入连跑 2 次 bitwise 一致） =====================
// spec determinism.bitwise_reproducible:true / accumulation_order:none。
// 多核大 shape（[1024,1023] fp32，多核满载 + 非对齐尾）连续执行 2 次，两次输出
// bitwise 完全一致（且均全 0），验证与核数/切分无关的确定性。
TEST_F(test_zeros_like, kernel_determinism_fp32_large_multicore)
{
    RunKernelTwiceAndCheckDeterministic({1024, 1023}, ge::DT_FLOAT, 4, 40, 262144);
}
// 8B 桶 + 不均衡多核的确定性补充（int64，33x17 非对齐尾）。
TEST_F(test_zeros_like, kernel_determinism_int64_tail)
{
    RunKernelTwiceAndCheckDeterministic({33, 17}, ge::DT_INT64, 8, 40, 262144);
}
