/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_reduce_scatter_tiling.cpp
 * \brief host侧tiling ut
 */
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace QuantReduceScatterUT {

const std::string OP_NAME = "QuantReduceScatter";

struct QuantReduceScatterTestParam {
    std::string caseName;
    // x
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    // scales
    std::initializer_list<int64_t> scalesShape;
    ge::DataType scalesDtype;
    ge::Format scalesFormat;
    // output
    std::initializer_list<int64_t> outputShape;
    ge::DataType outputDtype;
    ge::Format outputFormat;
    // attrs
    std::string groupAttr;
    std::string reduceOpAttr;
    int64_t outputDtypeAttr;
    // rank size
    uint64_t rankNum;
    // soc version
    std::string socVersion;
    // expert result
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t mc2TilingDataReservedLen;
};

inline std::ostream& operator<<(std::ostream& os, const QuantReduceScatterTestParam& param)
{
    return os << param.caseName;
}

// 构造ut用例：这里可以按照正常用例/异常用例分开声明
static QuantReduceScatterTestParam g_testCases[] = {
    {"quant_reduce_scatter_critical_case_1",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_SUCCESS, 0UL, "1024 5120 80 64 209715200 ", {16777216}, MC2_TILING_DATA_RESERVED_LEN},
    {"quant_reduce_scatter_critical_case_2",
    {2048, 5120}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    {2048, 40}, ge::DT_FLOAT, ge::FORMAT_ND,
    {256, 5120}, ge::DT_FLOAT, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT, 8, "3510",
    ge::GRAPH_SUCCESS, 0UL, "2048 5120 40 64 209715200 ", {16777216}, MC2_TILING_DATA_RESERVED_LEN},
    {"quant_reduce_scatter_critical_case_3",
    {1024, 7168}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {1024, 56}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 7168}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_SUCCESS, 0UL, "1024 7168 56 64 209715200 ", {16777216}, MC2_TILING_DATA_RESERVED_LEN},
    {"quant_reduce_scatter_critical_case_1_x3d",
    {8, 128, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {8, 128, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 4096}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_SUCCESS, 0UL, "1024 4096 64 64 209715200 ", {16777216}, MC2_TILING_DATA_RESERVED_LEN},
    {"quant_reduce_scatter_critical_case_2_x3d",
    {16, 128, 4096}, ge::DT_HIFLOAT8, ge::FORMAT_ND,
    {16, 128, 32}, ge::DT_FLOAT, ge::FORMAT_ND,
    {256, 4096}, ge::DT_FLOAT, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT, 8, "3510",
    ge::GRAPH_SUCCESS, 0UL, "2048 4096 32 64 209715200 ", {16777216}, MC2_TILING_DATA_RESERVED_LEN},
    {"quant_reduce_scatter_critical_case_3_x3d",
    {8, 128, 8192}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {8, 128, 64}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 8192}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_SUCCESS, 0UL, "1024 8192 64 64 209715200 ", {16777216}, MC2_TILING_DATA_RESERVED_LEN},
    {"quant_reduce_scatter_abuse_case_1_x3d",
    {8, 128, 8192}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {8, 128, 64}, ge::DT_FLOAT, ge::FORMAT_ND,
    {1, 128, 8192}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_2_x3d",
    {16, 128, 8192}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND,
    {16, 128, 64}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 8192}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_1",
    {0, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {0, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_2",
    {1024, 0}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 0, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_3",
    {1024, 4096}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 64, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_4",
    {2, 1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {2, 1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_5",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_6",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_7",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 160}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_8",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_9",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1023, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_10",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 79, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_11",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 3}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_12",
    {}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_13",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_14",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "add", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_15",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {1024, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_mx_16",
    {1024, 5120}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {1024, 80, 2}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_1",
    {2, 1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1024, 40}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_2",
    {1024, 5120}, ge::DT_FLOAT, ge::FORMAT_ND,
    {1024, 40}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_3",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_FRACTAL_NZ,
    {1024, 40}, ge::DT_FLOAT, ge::FORMAT_FRACTAL_NZ,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_FRACTAL_NZ,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_4",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1024, 40, 2}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_5",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1024, 40}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_6",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1024, 40}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_7",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1023, 40}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_8",
    {1024, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1024, 35}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
    {"quant_reduce_scatter_abuse_case_kg_9",
    {1023, 5120}, ge::DT_INT8, ge::FORMAT_ND,
    {1023, 40}, ge::DT_FLOAT, ge::FORMAT_ND,
    {128, 5120}, ge::DT_FLOAT16, ge::FORMAT_ND,
    "group", "sum", ge::DT_FLOAT16, 8, "3510",
    ge::GRAPH_FAILED, 0UL, "", {}, 0},
};

class QuantReduceScatterArch35TilingTest : public testing::TestWithParam<QuantReduceScatterTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantReduceScatterArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantReduceScatterArch35TilingTest TearDown." << std::endl;
    }
};

static struct QuantReduceScatterCompileInfo {} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const QuantReduceScatterTestParam &param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    // 参数封装
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape scalesShape = {param.scalesShape, param.scalesShape};
    gert::StorageShape outputShape = {param.outputShape, param.outputShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_({
        {xShape, param.xDtype, param.xFormat},
        {scalesShape, param.scalesDtype, param.scalesFormat}
    });
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({
        {outputShape, param.outputDtype, param.outputFormat}
    });
    std::vector<gert::TilingContextPara::OpAttr> attrs_({
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.groupAttr)},
        {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduceOpAttr)},
        {"output_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(param.outputDtypeAttr))}
    });
    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

static void ThreadFunction(const QuantReduceScatterTestParam *testCases, size_t caseNum, size_t threadIdx,
                           size_t threadNum)
{
    for (size_t idx = threadIdx; idx < caseNum; idx += threadNum) {
        auto param = testCases[idx];
        auto tilingContextPara = BuildTilingContextPara(param);
        ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                        param.expectWorkspaces, param.mc2TilingDataReservedLen);
    }
}

static void TestExecMultiThread(const QuantReduceScatterTestParam *testCases, size_t testCaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunction, testCases, testCaseNum, idx, threadNum);
    }
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(QuantReduceScatterArch35TilingTest, GeneralCasesTest)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.status, param.expectTilingKey,
                       param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
}

TEST_F(QuantReduceScatterArch35TilingTest, GeneralCasesMultiThreadTest)
{
    TestExecMultiThread(g_testCases, sizeof(g_testCases) / sizeof(QuantReduceScatterTestParam), 3);
}

INSTANTIATE_TEST_CASE_P(QuantReduceScatterTilingUT, QuantReduceScatterArch35TilingTest, testing::ValuesIn(g_testCases));

} // namespace
