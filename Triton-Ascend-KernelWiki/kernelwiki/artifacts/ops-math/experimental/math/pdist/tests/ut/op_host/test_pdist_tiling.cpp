#include <iostream>
#include <cmath>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "any_value.h"
#include "../../../op_kernel/pdist_tiling_data.h"

using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "Pdist";

struct PdistCompileInfo {} compileInfo;

class PdistTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "PdistTilingTest SetUp." << endl; }
    static void TearDownTestCase() { cout << "PdistTilingTest TearDown." << endl; }
};

static bool RunTilingTest(
    std::initializer_list<int64_t> xShape,
    ge::DataType dtype,
    std::initializer_list<int64_t> yShape,
    float pValue)
{
    gert::StorageShape xSS = {xShape, xShape};
    gert::StorageShape ySS = {yShape, yShape};

    std::vector<gert::TilingContextPara::TensorDescription> inputs({{xSS, dtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::TensorDescription> outputs({{ySS, dtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.push_back({"p", Ops::Math::AnyValue::CreateFrom<float>(pValue)});

    gert::TilingContextPara para(OP_NAME, inputs, outputs, attrs, &compileInfo, 24, 196608, 4096);
    TilingInfo tilingInfo;
    return ExecuteTiling(para, tilingInfo);
}

TEST_F(PdistTilingTest, fp32_p2_basic)
{
    EXPECT_TRUE(RunTilingTest({4, 8}, ge::DT_FLOAT, {6}, 2.0f));
}

TEST_F(PdistTilingTest, fp32_p0_hamming)
{
    EXPECT_TRUE(RunTilingTest({5, 16}, ge::DT_FLOAT, {10}, 0.0f));
}

TEST_F(PdistTilingTest, fp32_pinf_chebyshev)
{
    EXPECT_TRUE(RunTilingTest({3, 32}, ge::DT_FLOAT, {3},
                              std::numeric_limits<float>::infinity()));
}

TEST_F(PdistTilingTest, fp16_p1_manhattan)
{
    EXPECT_TRUE(RunTilingTest({6, 4}, ge::DT_FLOAT16, {15}, 1.0f));
}

TEST_F(PdistTilingTest, fp32_large_N)
{
    EXPECT_TRUE(RunTilingTest({100, 32}, ge::DT_FLOAT, {4950}, 2.0f));
}

TEST_F(PdistTilingTest, fp32_N2_minimum)
{
    EXPECT_TRUE(RunTilingTest({2, 1}, ge::DT_FLOAT, {1}, 2.0f));
}

TEST_F(PdistTilingTest, fp16_p2_basic)
{
    EXPECT_TRUE(RunTilingTest({4, 8}, ge::DT_FLOAT16, {6}, 2.0f));
}

TEST_F(PdistTilingTest, fp32_p05_fractional)
{
    EXPECT_TRUE(RunTilingTest({5, 16}, ge::DT_FLOAT, {10}, 0.5f));
}
