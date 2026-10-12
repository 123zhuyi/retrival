/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_kernel/log_space_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

namespace {
constexpr uint64_t DEFAULT_CORE_NUM = 64;
constexpr uint64_t DEFAULT_UB_SIZE = 262144;
// 与 op_host/log_space_tiling.cpp 对齐：每核最少 64 元素；常规搬运粒度 2048，df-V 路径放大到 4096
constexpr int64_t MIN_PER_CORE = 64;
constexpr uint32_t UB_CHUNK_NORMAL = 2048;
constexpr uint32_t UB_CHUNK_DFV = 4096;

struct LogSpaceCompileInfo {};
LogSpaceCompileInfo g_compileInfo{};

gert::StorageShape Shape(std::initializer_list<int64_t> dims) { return {dims, dims}; }

// ★下方排列顺序不可调换：faker 的 Attr(name, value) 会丢弃 name 形参、只按调用顺序 AppendAttr，
//   属性最终是按「下标」寻址的。故此处四项必须与 op_host/log_space_tiling.cpp 的
//   ATTR_IDX_START/END/STEPS/BASE(0/1/2/3) 及 op_host/log_space_def.cpp 的声明顺序逐一对齐，
//   否则 tiling 会静默读错属性（如把 start 当成 steps）。类型也须对应：start/end/base 为
//   Float()->GetAttrPointer<float>，steps 为 Int()->GetAttrPointer<int64_t>。
std::vector<gert::TilingContextPara::OpAttr> Attrs(float start, float end, int64_t steps, float base)
{
    return {
        {"start", Ops::Math::AnyValue::CreateFrom<float>(start)},
        {"end", Ops::Math::AnyValue::CreateFrom<float>(end)},
        {"steps", Ops::Math::AnyValue::CreateFrom<int64_t>(steps)},
        {"base", Ops::Math::AnyValue::CreateFrom<float>(base)},
    };
}

// LogSpace 是纯生成类算子：无输入 tensor，仅 1 个输出 + 4 个属性（见 op_host/log_space_def.cpp）。
// 但 GE 的 UT 上下文构建器不接受零输入节点——传 input_num=0 时 CreateComputeNodeInfo 会判
// "Invalid params" 失败，ContextHolder 构造出空 impl，DO_TILING 随后解引用空指针崩溃。故这里补一个
// 占位输入把节点凑合法；同为零输入算子的 Eye 其 tiling UT 也是这么做的
// （math/eye/tests/ut/op_host/arch35/test_eye_tiling_arch35.cpp，Eye 同样只有 Output + 属性）。
// 占位输入不改变被测行为：LogSpaceTilingFunc 只读 GetAttrs() 与 GetOutputDesc(0)，从不访问输入，
// 且属性下标由 Attrs() 的排列顺序决定（见上），与输入个数无关。
gert::TilingContextPara BuildTilingContext(std::initializer_list<int64_t> outShape, ge::DataType outDtype, float start,
                                           float end, int64_t steps, float base)
{
    return gert::TilingContextPara("LogSpace", {{Shape({1}), ge::DT_FLOAT, ge::FORMAT_ND}},
                                   {{Shape(outShape), outDtype, ge::FORMAT_ND}}, Attrs(start, end, steps, base),
                                   &g_compileInfo, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, sizeof(LogSpaceTilingData));
}

LogSpaceTilingData GetTilingData(const TilingInfo& tilingInfo)
{
    LogSpaceTilingData tilingData{};
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(LogSpaceTilingData));
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(LogSpaceTilingData));
    return tilingData;
}

// 复算 tiling 的分核公式，避免用例里手抄一堆魔数
void ExpectCoreSplit(const LogSpaceTilingData& tilingData, int64_t steps)
{
    const int64_t maxCores = (steps + MIN_PER_CORE - 1) / MIN_PER_CORE;
    const int64_t useCores = (maxCores < static_cast<int64_t>(DEFAULT_CORE_NUM)) ?
                                 maxCores :
                                 static_cast<int64_t>(DEFAULT_CORE_NUM);
    const int64_t tileLen = steps / useCores;
    const int64_t tailLen = steps - tileLen * (useCores - 1);

    EXPECT_EQ(tilingData.coreNum, static_cast<uint32_t>(useCores));
    EXPECT_EQ(tilingData.tileLen, static_cast<uint32_t>(tileLen));
    EXPECT_EQ(tilingData.tailTileLen, static_cast<uint32_t>(tailLen));
    EXPECT_EQ(tilingData.tailCoreIdx, static_cast<uint32_t>(useCores - 1));
    // 分核必须无缝覆盖 [0, steps)
    EXPECT_EQ(tileLen * (useCores - 1) + tailLen, steps);
}

void ExpectNoWorkspace(const TilingInfo& tilingInfo)
{
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0);
}
} // namespace

class LogSpaceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "LogSpaceTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "LogSpaceTiling TearDown" << endl; }
};

// ---------------- NORMAL 模式 + 多核分片 ----------------
TEST_F(LogSpaceTiling, Float32NormalMultiCore)
{
    // steps=1024 → maxCores=16，恰好整除：每核 64 元素
    auto para = BuildTilingContext({1024}, ge::DT_FLOAT, 0.0f, 2.0f, 1024, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    ExpectNoWorkspace(tilingInfo);
    EXPECT_EQ(tilingInfo.blockNum, 16U);

    auto tilingData = GetTilingData(tilingInfo);
    ExpectCoreSplit(tilingData, 1024);
    EXPECT_EQ(tilingData.coreNum, 16U);
    EXPECT_EQ(tilingData.tileLen, 64U);
    EXPECT_EQ(tilingData.totalLen, 1024U);
    EXPECT_EQ(tilingData.ubChunk, UB_CHUNK_NORMAL);
    EXPECT_FLOAT_EQ(tilingData.startF, 0.0f);
    EXPECT_FLOAT_EQ(tilingData.stepF, 2.0f / 1023.0f);
    EXPECT_FLOAT_EQ(tilingData.logBase, std::log(10.0f));
    // 端点由 host 用 double pow 精确下发
    EXPECT_FLOAT_EQ(tilingData.startValF, 1.0f);
    EXPECT_FLOAT_EQ(tilingData.endValF, 100.0f);
    // 浮点输出不需要整数幂覆写表，也不走 df-V
    EXPECT_EQ(tilingData.nCount, 0);
    EXPECT_EQ(tilingData.useDfV, 0);
}

TEST_F(LogSpaceTiling, Float32NormalUnevenTail)
{
    // steps=1000 不能被 16 整除 → 常规核 62、尾核 70，验证尾块吸收余量
    auto para = BuildTilingContext({1000}, ge::DT_FLOAT, 0.0f, 2.0f, 1000, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    EXPECT_EQ(tilingInfo.blockNum, 16U);
    auto tilingData = GetTilingData(tilingInfo);
    ExpectCoreSplit(tilingData, 1000);
    EXPECT_EQ(tilingData.coreNum, 16U);
    EXPECT_EQ(tilingData.tileLen, 62U);
    EXPECT_EQ(tilingData.tailTileLen, 70U);
    EXPECT_EQ(tilingData.tailCoreIdx, 15U);
}

TEST_F(LogSpaceTiling, Float32SmallShapeSingleCore)
{
    // steps=64 恰为每核下限 → 只用 1 个核（不为凑核数把每核切到 64 以下）
    auto para = BuildTilingContext({64}, ge::DT_FLOAT, 0.0f, 2.0f, 64, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    EXPECT_EQ(tilingInfo.blockNum, 1U);
    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.coreNum, 1U);
    EXPECT_EQ(tilingData.tileLen, 64U);
    EXPECT_EQ(tilingData.tailTileLen, 64U);
    EXPECT_EQ(tilingData.tailCoreIdx, 0U);
}

TEST_F(LogSpaceTiling, Float32MaxCoreClampedByPlatform)
{
    // steps 足够大时核数被平台核数上限截断（而非 steps/64）
    const int64_t steps = 1024 * 1024;
    auto para = BuildTilingContext({steps}, ge::DT_FLOAT, 0.0f, 5.0f, steps, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    EXPECT_EQ(tilingInfo.blockNum, DEFAULT_CORE_NUM);
    auto tilingData = GetTilingData(tilingInfo);
    ExpectCoreSplit(tilingData, steps);
    EXPECT_EQ(tilingData.coreNum, static_cast<uint32_t>(DEFAULT_CORE_NUM));
}

// ---------------- SINGLE 模式（steps 0/1 边界） ----------------
TEST_F(LogSpaceTiling, SingleModeStepsOne)
{
    auto para = BuildTilingContext({1}, ge::DT_FLOAT, 0.0f, 2.0f, 1, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    ExpectNoWorkspace(tilingInfo);
    EXPECT_EQ(tilingInfo.blockNum, 1U);

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalLen, 1U);
    EXPECT_EQ(tilingData.coreNum, 1U);
    EXPECT_EQ(tilingData.tileLen, 1U);
    EXPECT_EQ(tilingData.tailTileLen, 1U);
    EXPECT_EQ(tilingData.tailCoreIdx, 0U);
    // steps==1 无步长，out[0]=base^start
    EXPECT_FLOAT_EQ(tilingData.stepF, 0.0f);
    EXPECT_FLOAT_EQ(tilingData.startValF, 1.0f);
}

TEST_F(LogSpaceTiling, SingleModeStepsZero)
{
    // steps==0：L2 已短路不下发 Kernel，tiling 侧仍需给出安全的单核空跑参数
    auto para = BuildTilingContext({0}, ge::DT_FLOAT, 0.0f, 2.0f, 0, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    EXPECT_EQ(tilingInfo.blockNum, 1U);
    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.totalLen, 0U);
    EXPECT_EQ(tilingData.coreNum, 1U);
    EXPECT_EQ(tilingData.tileLen, 0U);
    EXPECT_EQ(tilingData.tailTileLen, 0U);
    EXPECT_FLOAT_EQ(tilingData.stepF, 0.0f);
}

// ---------------- 整型路径：整数幂精确覆写表 ----------------
TEST_F(LogSpaceTiling, Int32IntegerPowerTable)
{
    // base=2, x=0..5 全落在整数指数网格点上 → 幂表 [1,2,4,8,16,32]
    auto para = BuildTilingContext({6}, ge::DT_INT32, 0.0f, 5.0f, 6, 2.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.nmin, 0);
    EXPECT_EQ(tilingData.nCount, 6);
    const float expectBaseN[6] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f};
    for (int32_t k = 0; k < tilingData.nCount; ++k) {
        EXPECT_FLOAT_EQ(tilingData.baseN[k], expectBaseN[k]);
    }
    // arg 空间的 double-float 系数：argStart=start*ln(base)=0，stepLn=1.0*ln(2)
    EXPECT_FLOAT_EQ(tilingData.argStartHi, 0.0f);
    EXPECT_FLOAT_EQ(tilingData.stepLnHi, static_cast<float>(std::log(2.0)));
    // int32 不走 df-V（仅 int8/uint8 大值才需要）
    EXPECT_EQ(tilingData.useDfV, 0);
    EXPECT_EQ(tilingData.ubChunk, UB_CHUNK_NORMAL);
}

TEST_F(LogSpaceTiling, Int16NegativeExponentRange)
{
    // 起点为负指数：幂表下界应为 floor(min(start,end))
    auto para = BuildTilingContext({8}, ge::DT_INT16, -3.0f, 4.0f, 8, 2.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.nmin, -3);
    EXPECT_EQ(tilingData.nCount, 8); // -3..4
    EXPECT_FLOAT_EQ(tilingData.baseN[0], 0.125f);
    EXPECT_EQ(tilingData.useDfV, 0);
}

// ---------------- df-V 路径（int8/uint8 大值） ----------------
TEST_F(LogSpaceTiling, Int8DfVEnabledOnLargeValue)
{
    // 10^8 = 1e8 > 2^24 → 标量 fp32 不足以让 mod256 准确，启用向量 df-exp
    auto para = BuildTilingContext({1024}, ge::DT_INT8, 0.0f, 8.0f, 1024, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.useDfV, 1);
    EXPECT_EQ(tilingData.ubChunk, UB_CHUNK_DFV); // df-exp 是每 chunk 固定开销 → 放大 chunk
    EXPECT_EQ(tilingData.nCount, 9);             // 0..8
    ExpectCoreSplit(tilingData, 1024);
    // 递推等比常数 const = base^(ubChunk*step)，必须是有限正数
    EXPECT_GT(tilingData.constHi, 0.0f);
    EXPECT_TRUE(std::isfinite(tilingData.constHi));
    // 泰勒 exp 的 1/k! 系数表：rfHi[0]=1/0!=1，rfHi[1]=1/1!=1，rfHi[2]=1/2!=0.5
    EXPECT_FLOAT_EQ(tilingData.rfHi[0], 1.0f);
    EXPECT_FLOAT_EQ(tilingData.rfHi[1], 1.0f);
    EXPECT_FLOAT_EQ(tilingData.rfHi[2], 0.5f);
}

TEST_F(LogSpaceTiling, Uint8DfVEnabledOnLargeValue)
{
    auto para = BuildTilingContext({1024}, ge::DT_UINT8, 0.0f, 8.0f, 1024, 10.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.useDfV, 1);
    EXPECT_EQ(tilingData.ubChunk, UB_CHUNK_DFV);
}

TEST_F(LogSpaceTiling, Int8DfVDisabledOnSmallValue)
{
    // 2^6 = 64 远小于 2^24 → 常规 fp32 路径足够，不启用 df-V
    auto para = BuildTilingContext({7}, ge::DT_INT8, 0.0f, 6.0f, 7, 2.0f);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto tilingData = GetTilingData(tilingInfo);
    EXPECT_EQ(tilingData.useDfV, 0);
    EXPECT_EQ(tilingData.ubChunk, UB_CHUNK_NORMAL);
    EXPECT_EQ(tilingData.nCount, 7); // 0..6
}

// ---------------- TilingKey 分流 ----------------
TEST_F(LogSpaceTiling, TilingKeyDiffersByModeAndDtype)
{
    // TilingKey 由 (D_T_Y, MODE) 二元组下发，具体编码由 ASCENDC_TPL 框架决定，
    // 故此处只校验不同 dtype / 不同 MODE 必须落到不同的 key，避免误合流到同一模板实例。
    TilingInfo fp32Normal;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext({1024}, ge::DT_FLOAT, 0.0f, 2.0f, 1024, 10.0f), fp32Normal));
    TilingInfo fp32Single;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext({1}, ge::DT_FLOAT, 0.0f, 2.0f, 1, 10.0f), fp32Single));
    TilingInfo int32Normal;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext({1024}, ge::DT_INT32, 0.0f, 2.0f, 1024, 10.0f), int32Normal));
    TilingInfo fp16Normal;
    ASSERT_TRUE(ExecuteTiling(BuildTilingContext({1024}, ge::DT_FLOAT16, 0.0f, 2.0f, 1024, 10.0f), fp16Normal));

    EXPECT_NE(fp32Normal.tilingKey, fp32Single.tilingKey);  // MODE 维度
    EXPECT_NE(fp32Normal.tilingKey, int32Normal.tilingKey); // dtype 维度
    EXPECT_NE(fp32Normal.tilingKey, fp16Normal.tilingKey);
    EXPECT_NE(int32Normal.tilingKey, fp16Normal.tilingKey);
}

// ---------------- 异常参数 ----------------
TEST_F(LogSpaceTiling, InvalidNegativeSteps)
{
    ExecuteTestCase(BuildTilingContext({5}, ge::DT_FLOAT, 0.0f, 2.0f, -1, 10.0f), ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(LogSpaceTiling, InvalidBaseZero)
{
    ExecuteTestCase(BuildTilingContext({5}, ge::DT_FLOAT, 0.0f, 2.0f, 5, 0.0f), ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(LogSpaceTiling, InvalidBaseNegative)
{
    ExecuteTestCase(BuildTilingContext({5}, ge::DT_FLOAT, 0.0f, 2.0f, 5, -2.0f), ge::GRAPH_FAILED, 0, "", {});
}

TEST_F(LogSpaceTiling, InvalidOutputDtype)
{
    ExecuteTestCase(BuildTilingContext({5}, ge::DT_INT64, 0.0f, 2.0f, 5, 10.0f), ge::GRAPH_FAILED, 0, "", {});
}
