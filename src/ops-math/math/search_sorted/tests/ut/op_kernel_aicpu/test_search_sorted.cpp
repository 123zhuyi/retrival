/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

const float input1_float_data[] = {
    10.274563f, 22.392572f, 23.028294f, 35.85503f,  42.911854f, 49.58565f,  61.913395f, 67.3148f,   81.3377f,
    83.20328f,  46.36467f,  54.379158f, 67.43358f,  68.719315f, 70.72506f,  92.54831f,  92.57594f,  98.53133f,
    98.67426f,  99.74494f,  6.6154857f, 9.609693f,  14.914657f, 36.245827f, 46.443245f, 46.88357f,  52.865215f,
    53.333603f, 78.41768f,  94.559235f, 12.250699f, 14.527651f, 22.289686f, 45.292423f, 46.305958f, 78.947334f,
    79.180954f, 86.726875f, 88.99657f,  93.17653f,  9.802182f,  21.195236f, 41.55525f,  50.20249f,  50.41457f,
    64.81418f,  68.828186f, 71.1445f,   71.145706f, 74.940315f, 8.694717f,  30.563177f, 48.90927f,  49.865623f,
    50.600056f, 60.201866f, 63.17426f,  66.939255f, 84.841286f, 91.31864f,  18.77629f,  24.146885f, 44.992783f,
    59.760105f, 61.67494f,  67.962296f, 77.14767f,  77.225136f, 83.45356f,  83.77025f,  2.9066865f, 35.196434f,
    38.42185f,  39.551373f, 47.142277f, 77.85041f,  77.85396f,  84.79707f,  85.456184f, 85.95588f,  6.575237f,
    6.762218f,  12.445413f, 30.020798f, 34.228344f, 36.543858f, 55.66742f,  74.97183f,  93.96571f,  98.10762f,
    8.557885f,  22.670553f, 33.95384f,  36.75872f,  39.134098f, 48.29664f,  57.24083f,  69.48666f,  70.10226f,
    98.04606f,  15.739977f, 31.473179f, 36.398487f, 38.73297f,  41.14757f,  50.229572f, 53.02835f,  53.478603f,
    55.033455f, 91.815475f, 2.875517f,  4.0039406f, 10.197294f, 19.335278f, 21.843681f, 24.972792f, 28.422f,
    53.072044f, 53.82339f,  60.27561f};

const float input2_float_data[] = {
    35.274387f, 85.11903f,  20.373728f, 54.49726f,   99.10766f, 26.271276f, 58.605366f, 99.50345f,  58.61402f,
    30.568392f, 20.60171f,  6.3036118f, 98.20946f,   36.22734f, 89.435616f, 30.290298f, 15.582192f, 48.159847f,
    23.817549f, 9.765527f,  10.564091f, 47.475735f,  89.66093f, 12.688463f, 83.80613f,  11.345059f, 44.72875f,
    71.586975f, 16.365799f, 80.78974f,  0.83431894f, 73.07463f, 38.269215f, 95.04992f,  41.122395f, 66.72289f};

const double input1_double_data[] = {
    2.9066866249524814, 6.575236891077774,  6.615485646323549,  6.762218120223141,  8.55788531460594,
    8.69471785185414,   9.609692666640212,  9.80218184088859,   10.274562444770318, 12.250699013371458,
    12.44541270691586,  14.527650583138186, 14.91465665615147,  18.77629005510545,  21.19523600363391,
    22.289686944543075, 22.392572550653888, 22.67055327038485,  23.028294195310018, 24.146884094554967,
    30.020797765877973, 30.56317685979947,  33.95383861791378,  34.228342882789335, 35.19643589736153,
    35.85503065425404,  36.24582814981078,  36.543858739776894, 36.75872130483092,  38.421847806462026,
    39.13409992613162,  39.55137103338957,  41.555249984360955, 42.911854234567194, 44.99278290456618,
    45.292424052561685, 46.30595923913967,  46.36467053338188,  46.4432454223301,   46.88357289934575,
    47.14227661700847,  48.296640377350464, 48.909270483417735, 49.58565010357161,  49.86562175990436,
    50.202491092845115, 50.41456923450788,  50.600057233609796, 52.865215859166824, 53.333603196063564,
    54.379159147180026, 55.66741834589588,  57.24082796669529,  59.76010406364701,  60.20186444750551,
    61.67493907135619,  61.91339515152623,  63.174260758095855, 64.81418076566547,  66.93925385692513,
    67.31479813235946,  67.43358079206307,  67.96229288032863,  68.7193118091968,   68.82818715434436,
    69.48665804171556,  70.10225349640203,  70.72505650375786,  71.14450096577197,  71.14570489462028,
    74.94031570579459,  74.9718326977705,   77.14766577647929,  77.22513569504933,  77.8504075471102,
    77.85395502244717,  78.41768205745,     78.94733566830399,  79.18095602003774,  81.33769774525612,
    83.2032805951059,   83.45355924448855,  83.77025222261061,  84.79707070726536,  84.8412866104459,
    85.45618111820872,  85.95587831453774,  86.72687862033645,  88.99656919674734,  91.31864341379739,
    92.54831269884872,  92.57593997798537,  93.17652966749317,  93.96571410821234,  94.55923389329655,
    98.04605994716053,  98.10762238208213,  98.53132645586987,  98.67426487295742,  99.74493833062787};

const double input2_double_data[] = {
    106.05670472528598, 62.94635887081106,  100.45914480060911, 72.79697569101396,  183.63094362540284,
    31.479954711532,    77.46594543559684,  82.2951409662946,   106.95720683841516, 110.06691216954101,
    49.94558228359773,  106.14408514708606, 8.007880850843986,  38.670554378099986, 5.751033676899908,
    120.55122435710608, 20.394589177939594, 107.6467817880827,  56.84400114846475,  43.687361309592234,
    70.54877751604724,  170.2380718582778,  40.74745517976404,  108.99452285377048, 198.2153222600747,
    52.54255159261121,  117.21073494512595, 199.00689498469592, 117.22804635804988, 61.1367845159789,
    41.203419131731735, 12.60722313657503,  196.41891529628853, 72.45468478536776,  178.87122900776833,
    60.58059875323873,  31.164385145701743, 96.31969234540638,  47.63509777121169,  19.531053419551125,
    21.12818116915205,  94.95147149409506,  179.32186105502436, 25.376925811659177, 167.61226053897528,
    22.690117954584288, 89.4574936313971,   143.17394595864107, 32.73159872417031,  161.57948154572793,
    1.6686379203650237, 146.1492620857177,  76.53842763352696,  190.0998443409722,  82.24478594546252,
    133.44579200631554, 168.08385971342773, 136.61980126170369, 9.38483497767244,   165.34620008620632,
    123.52109347160996, 14.994886517795369, 108.6588932987375,  178.9043183672402,  186.82069203568622,
    156.84958470331,    26.603922610489562, 42.684999104594645, 194.37905592242137, 83.50884813624948,
    9.978876604366494,  10.61574914340111,  37.076851583108784, 92.90029304998349,  61.381434787846835,
    92.8010025922446,   190.91860178217595, 68.90739950663341,  95.2184190499248,   161.1917787111029,
    123.18758309936051, 189.69593101580904, 16.93411075602782,  125.6342524930409,  25.411654272327787,
    150.15068429743908, 66.72642325549336,  159.06462866980962, 147.69583726266362, 60.40123073576724,
    160.39934784899793, 38.475980019913614, 63.06605442251241,  168.52641767863895, 66.82971773437933,
    44.615557917397865, 194.52142064540757, 145.1389467618058,  12.735905868840568, 47.53263649673545,
    128.23036243984467, 158.1251431536277,  143.0548421095135,  49.062174756220415, 35.17959062155336,
    148.0060147325134,  137.47210644064737, 179.64012545174916, 62.42763831724567,  185.72182584083475,
    85.02569021040843,  169.7590403533167,  16.00237136536109,  103.83972829571289, 152.52918212458232,
    192.54891261326725, 171.24060225375183, 106.836180650938,   158.3310915813356,  2.7818762028496202,
    65.16155612376313,  109.0952822755606,  22.927602867018138, 143.64761593850463, 183.85059556696086,
    21.828439273311506, 175.50568216042785, 193.23251586376279, 91.20481109695749,  8.939832798907243,
    93.06446234382398,  12.885608561256646, 87.62624029008317,  84.2841579425695,   160.45958275806322,
    154.19794513296097, 175.32932059212706, 19.201237440286654, 7.716461916307149,  37.347257749759066,
    7.03000802511673,   102.1687865019409,  94.85563930365848,  194.96940277065738, 96.04606388685386,
    14.839354940983917, 124.63429958025922, 56.35669801563332,  80.1401697960592,   75.46936624714962,
    122.54394941690467, 189.08977183833852, 123.512990793851,   115.09483139537315, 18.628908667711343,
    14.755254532746044, 120.25353671855919, 48.96207475530821,  85.89938208911252,  83.00754084140381,
    153.32415545472165, 142.0121580372877,  19.41727914873168,  134.31824473968044, 103.36964063942484,
    52.169845443111186, 55.58771678915719,  86.33034894615297,  78.04632900828918,  85.78140081123627,
    139.04562539698583, 98.67858213844012,  97.23600470268244,  72.9379959040135,   79.70829330373381,
    12.946402823235982, 3.777313346963007,  126.5487062667863,  74.4731428986279,   87.74564037555363,
    39.81086399954552,  96.14489379484957,  66.22523226855907,  140.23248754516257, 64.07331549446367,
    104.96357917141397, 35.505544015015666, 153.06369784514112, 117.62935331117856, 19.789907030018618,
    88.2267100027615,   89.30092349932302,  116.02006795481734, 97.9115806212808,   78.47497382607939,
    4.547772218217427,  125.96223893026391, 178.87297102487025, 133.97113048572638, 100.71226778195803};

const int32_t output_exp_int32_t_data[] = {
    100, 57,  100, 70,  100, 22,  74,  80,  100, 100, 45,  100, 4,   30,  1,   100, 14,  100, 52,  34,  67,  100, 32,
    100, 100, 48,  100, 100, 100, 55,  32,  11,  100, 70,  100, 55,  22,  95,  41,  14,  14,  95,  100, 20,  100, 18,
    89,  100, 22,  100, 0,   100, 72,  100, 80,  100, 100, 100, 6,   100, 100, 13,  100, 100, 100, 100, 20,  33,  100,
    82,  8,   9,   29,  92,  55,  92,  100, 65,  95,  100, 100, 100, 13,  100, 20,  100, 59,  100, 100, 55,  100, 30,
    57,  100, 59,  34,  100, 100, 11,  41,  100, 100, 100, 43,  24,  100, 100, 100, 57,  100, 85,  100, 13,  100, 100,
    100, 100, 100, 100, 0,   59,  100, 18,  100, 100, 15,  100, 100, 89,  6,   92,  11,  88,  83,  100, 100, 100, 14,
    4,   29,  4,   100, 95,  100, 95,  12,  100, 52,  79,  72,  100, 100, 100, 100, 13,  12,  100, 43,  86,  80,  100,
    100, 14,  100, 100, 48,  51,  87,  76,  86,  100, 99,  95,  70,  79,  11,  1,   100, 70,  88,  32,  95,  59,  100,
    58,  100, 25,  100, 100, 14,  88,  89,  100, 95,  77,  1,   100, 100, 100, 100};

class TEST_SearchSorted_UTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "search_sorted_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "search_sorted_test TearDown\n" << endl;
    }
};

#define ADD_CASE(aicpu_type, base_type, out_type, out_dtype)                        \
    TEST_F(TEST_SearchSorted_UTest, SearchSorted_Success_##aicpu_type##_##out_type) \
    {                                                                               \
        aicpu_type sorted_sequence[2 * 5] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};        \
        aicpu_type values[2 * 3] = {3, 6, 9, 3, 6, 9};                              \
        out_type output[2 * 3] = {0};                                               \
        out_type expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};                       \
        auto nodeDef = CpuKernelUtils::CreateNodeDef();                             \
        nodeDef->SetOpType("SearchSorted");                                         \
        auto right = CpuKernelUtils::CreateAttrValue();                             \
        right->SetBool(false);                                                      \
        nodeDef->AddAttrs("right", right.get());                                    \
        auto side = CpuKernelUtils::CreateAttrValue();                              \
        side->SetString("");                                                        \
        nodeDef->AddAttrs("side", side.get());                                      \
        auto inputTensor0 = nodeDef->AddInputs();                                   \
        EXPECT_NE(inputTensor0, nullptr);                                           \
        auto aicpuShape0 = inputTensor0->GetTensorShape();                          \
        std::vector<int64_t> shapes0 = {2, 5};                                      \
        aicpuShape0->SetDimSizes(shapes0);                                          \
        inputTensor0->SetDataType(base_type);                                       \
        inputTensor0->SetData(sorted_sequence);                                     \
        inputTensor0->SetDataSize(2 * 5 * sizeof(aicpu_type));                      \
        auto inputTensor1 = nodeDef->AddInputs();                                   \
        EXPECT_NE(inputTensor1, nullptr);                                           \
        auto aicpuShape1 = inputTensor1->GetTensorShape();                          \
        std::vector<int64_t> shapes1 = {2, 3};                                      \
        aicpuShape1->SetDimSizes(shapes1);                                          \
        inputTensor1->SetDataType(base_type);                                       \
        inputTensor1->SetData(values);                                              \
        inputTensor1->SetDataSize(2 * 3 * sizeof(aicpu_type));                      \
        auto outputTensor1 = nodeDef->AddOutputs();                                 \
        EXPECT_NE(outputTensor1, nullptr);                                          \
        auto aicpuShape3 = outputTensor1->GetTensorShape();                         \
        std::vector<int64_t> shapes2 = {2, 3};                                      \
        aicpuShape3->SetDimSizes(shapes2);                                          \
        outputTensor1->SetDataType(out_dtype);                                      \
        outputTensor1->SetData(output);                                             \
        outputTensor1->SetDataSize(2 * 3 * sizeof(out_type));                       \
        CpuKernelContext ctx(DEVICE);                                               \
        EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);                       \
        uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);             \
        EXPECT_EQ(ret, KERNEL_STATUS_OK);                                           \
        EXPECT_EQ(0, std::memcmp(output, expected_output, sizeof(output)));         \
    }

TEST_F(TEST_SearchSorted_UTest, SearchSorted_Input_Type_Error)
{
    uint32_t sorted_sequence[2 * 5] = {1, 3, 5, 7, 9, 2, 4, 6, 8, 10};
    uint32_t values[2 * 3] = {3, 6, 9, 3, 6, 9};
    int32_t output[2 * 3] = {0};
    int32_t expected_output[2 * 3] = {1, 3, 4, 1, 2, 4};
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SearchSorted");
    auto right = CpuKernelUtils::CreateAttrValue();
    right->SetBool(false);
    nodeDef->AddAttrs("right", right.get());
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {2, 5};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_UINT32);
    inputTensor0->SetData(sorted_sequence);
    inputTensor0->SetDataSize(2 * 5 * sizeof(uint32_t));
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {2, 3};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_UINT32);
    inputTensor1->SetData(values);
    inputTensor1->SetDataSize(2 * 3 * sizeof(uint32_t));
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    auto aicpuShape3 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes2 = {2, 3};
    aicpuShape3->SetDimSizes(shapes2);
    outputTensor1->SetDataType(DT_INT32);
    outputTensor1->SetData(output);
    outputTensor1->SetDataSize(2 * 3 * sizeof(int32_t));
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SearchSorted_Input_Unsorted_Error)
{
    float sorted_sequence[2 * 5] = {9, 3, 5, 7, 9, 2, 4, 6, 8, 1};
    float values[2 * 3] = {3, 6, 9, 3, 6, 9};
    int32_t output[2 * 3] = {0};
    int32_t expected_output[2 * 3] = {0, 3, 4, 1, 2, 5};
    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("SearchSorted");
    auto right = CpuKernelUtils::CreateAttrValue();
    right->SetBool(false);
    nodeDef->AddAttrs("right", right.get());
    auto side = CpuKernelUtils::CreateAttrValue();
    side->SetString("");
    nodeDef->AddAttrs("side", side.get());
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {2, 5};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(sorted_sequence);
    inputTensor0->SetDataSize(2 * 5 * sizeof(float));
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {2, 3};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(values);
    inputTensor1->SetDataSize(2 * 3 * sizeof(float));
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    auto aicpuShape3 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes2 = {2, 3};
    aicpuShape3->SetDimSizes(shapes2);
    outputTensor1->SetDataType(DT_INT32);
    outputTensor1->SetData(output);
    outputTensor1->SetDataSize(2 * 3 * sizeof(int32_t));
    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    EXPECT_EQ(0, std::memcmp(output, expected_output, sizeof(output)));
}

ADD_CASE(float, DT_FLOAT, int, DT_INT32)

ADD_CASE(double, DT_DOUBLE, int, DT_INT32)

ADD_CASE(int32_t, DT_INT32, int, DT_INT32)

ADD_CASE(int64_t, DT_INT64, int, DT_INT32)

ADD_CASE(float, DT_FLOAT, int64_t, DT_INT64)

ADD_CASE(double, DT_DOUBLE, int64_t, DT_INT64)

ADD_CASE(int32_t, DT_INT32, int64_t, DT_INT64)

ADD_CASE(int64_t, DT_INT64, int64_t, DT_INT64)

template <typename T1, typename T2>
void RunSearchSortedKernel(
    vector<DataType> data_type, vector<vector<int64_t>>& shapes, const bool right, const T1* input1_data,
    const T1* input2_data, const T2* output_exp_data)
{
    uint64_t input1_size = CalTotalElements(shapes, 0);
    T1* input1_tensor_buffer = new T1[input1_size];
    for (uint64_t i = 0; i < input1_size; ++i) {
        input1_tensor_buffer[i] = input1_data[i];
    }

    uint64_t input2_size = CalTotalElements(shapes, 1);
    T1* input2_tensor_buffer = new T1[input2_size];
    for (uint64_t i = 0; i < input2_size; ++i) {
        input2_tensor_buffer[i] = input2_data[i];
    }

    uint64_t output_size = CalTotalElements(shapes, 2);
    T2* output_tensor_buffer = new T2[output_size];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", right);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T2* output_exp = new T2[output_size];
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    bool compare = CompareResult(output_tensor_buffer, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input1_tensor_buffer;
    delete[] input2_tensor_buffer;
    delete[] output_tensor_buffer;
    delete[] output_exp;
}

TEST_F(TEST_SearchSorted_UTest, FLOAT_INT64_FALSE_SUCC)
{
    bool right = false;
    vector<DataType> data_type = {DT_FLOAT, DT_INT64};
    vector<vector<int64_t>> shapes = {{4, 3, 10}, {4, 3, 3}, {4, 3, 3}};
    const int64_t output_exp_data[] = {3, 10, 1, 2, 9,  0, 8, 10, 8, 3, 2, 0, 10, 2, 10, 1,  1, 2,
                                       1, 0,  0, 5, 10, 1, 8, 2,  6, 9, 1, 9, 0,  9, 3,  10, 7, 10};

    RunSearchSortedKernel<float, int64_t>(
        data_type, shapes, right, input1_float_data, input2_float_data, output_exp_data);
}

TEST_F(TEST_SearchSorted_UTest, DOUBLE_INT32_TRUE_SUCC)
{
    bool right = false;
    vector<DataType> data_type = {DT_DOUBLE, DT_INT32};
    vector<vector<int64_t>> shapes = {{100}, {200}, {200}};

    RunSearchSortedKernel<double, int32_t>(
        data_type, shapes, right, input1_double_data, input2_double_data, output_exp_int32_t_data);
}

TEST_F(TEST_SearchSorted_UTest, SORTER_DTYPE_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {3}, {3}};
    int32_t input1_tensor_buffer[5] = {1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[3] = {3, 6, 9};
    int32_t input3_tensor_buffer[5] = {0, 1, 2, 4, 3};
    int32_t output_tensor_buffer[3];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT32, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SORTER_NUMBER_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {3}, {3}};
    int32_t input1_tensor_buffer[5] = {1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[3] = {3, 6, 9};
    int64_t input3_tensor_buffer[3] = {0, 1, 2};
    int32_t output_tensor_buffer[3];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[2], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SORTER_OUT_OF_RANGE_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 5}, {2, 3}, {2, 3}};
    int32_t input1_tensor_buffer[2 * 5] = {1, 3, 5, 9, 7, 1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[2 * 3] = {3, 6, 9, 3, 6, 9};
    int64_t input3_tensor_buffer[2 * 5] = {-1, 1, 2, 4, 3, 0, 1, 2, 4, 10};
    int32_t output_tensor_buffer[2 * 3];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, SEQUENCE_VALUE_DIMS_ERROR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 5}, {6}, {6}};
    int32_t input1_tensor_buffer[2 * 5] = {1, 3, 5, 9, 7, 1, 3, 5, 9, 7};
    int32_t input2_tensor_buffer[6] = {3, 6, 9, 3, 6, 9};
    int64_t input3_tensor_buffer[2 * 5] = {0, 1, 2, 4, 3, 0, 1, 2, 4, 3};
    int32_t output_tensor_buffer[6];
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Input({"sorter", DT_INT64, shapes[0], (void*)input3_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SearchSorted_UTest, EMPTY_SEQUENCE_TENSOR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2, 0}, {2, 3}, {2, 3}};
    int32_t input1_tensor_buffer[] = {};
    int32_t input2_tensor_buffer[2 * 3] = {3, 6, 9, 3, 6, 9};
    int32_t output_tensor_buffer[2 * 3] = {0, 0, 0, 0, 0, 0};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int32_t output_exp[2 * 3] = {0, 0, 0, 0, 0, 0};
    bool compare = CompareResult(output_tensor_buffer, output_exp, 2 * 3);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SearchSorted_UTest, EMPTY_VALUE_TENSOR)
{
    vector<DataType> data_type = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{5}, {0}, {0}};
    int32_t input1_tensor_buffer[5] = {1, 3, 5, 7, 9};
    ;
    int32_t input2_tensor_buffer[] = {};
    int32_t output_tensor_buffer[] = {};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SearchSorted", "SearchSorted")
        .Input({"sorted_sequence", data_type[0], shapes[0], (void*)input1_tensor_buffer})
        .Input({"values", data_type[0], shapes[1], (void*)input2_tensor_buffer})
        .Output({"out", data_type[1], shapes[2], (void*)output_tensor_buffer})
        .Attr("dtype", data_type[1])
        .Attr("right", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
}