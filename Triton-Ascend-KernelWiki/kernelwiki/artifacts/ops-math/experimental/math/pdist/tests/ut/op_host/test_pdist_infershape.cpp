#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class PdistInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PdistInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "PdistInfershape TearDown" << std::endl; }
};

TEST_F(PdistInfershape, pdist_infershape_basic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pdist",
        { {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} });
    std::vector<std::vector<int64_t>> expectOutputShape = {{6}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PdistInfershape, pdist_infershape_n2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pdist",
        { {{{2, 1}, {2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PdistInfershape, pdist_infershape_large_n)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pdist",
        { {{{100, 32}, {100, 32}}, ge::DT_FLOAT, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4950}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PdistInfershape, pdist_infershape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pdist",
        { {{{5, 64}, {5, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND} });
    std::vector<std::vector<int64_t>> expectOutputShape = {{10}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(PdistInfershape, pdist_infershape_fail_1d_input)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pdist",
        { {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(PdistInfershape, pdist_infershape_fail_n_less_than_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Pdist",
        { {{{1, 8}, {1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND} },
        { {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND} });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
