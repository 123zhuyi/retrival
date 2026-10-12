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
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_CUMSUM_UT : public testing::Test {};

static const Eigen::half kCumsumInputF16[60] = {
    static_cast<Eigen::half>(10.78f),  static_cast<Eigen::half>(-41.88f),  static_cast<Eigen::half>(-50.72f),
    static_cast<Eigen::half>(10.805f), static_cast<Eigen::half>(98.25f),   static_cast<Eigen::half>(-78.6f),
    static_cast<Eigen::half>(-14.02f), static_cast<Eigen::half>(48.8f),    static_cast<Eigen::half>(-83.94f),
    static_cast<Eigen::half>(-55.3f),  static_cast<Eigen::half>(68.25f),   static_cast<Eigen::half>(-28.86f),
    static_cast<Eigen::half>(-1.745f), static_cast<Eigen::half>(-97.0f),   static_cast<Eigen::half>(15.95f),
    static_cast<Eigen::half>(80.4f),   static_cast<Eigen::half>(7.977f),   static_cast<Eigen::half>(3.44f),
    static_cast<Eigen::half>(42.03f),  static_cast<Eigen::half>(59.22f),   static_cast<Eigen::half>(-36.72f),
    static_cast<Eigen::half>(-46.47f), static_cast<Eigen::half>(-88.5f),   static_cast<Eigen::half>(-27.56f),
    static_cast<Eigen::half>(86.0f),   static_cast<Eigen::half>(29.12f),   static_cast<Eigen::half>(90.44f),
    static_cast<Eigen::half>(-95.5f),  static_cast<Eigen::half>(40.56f),   static_cast<Eigen::half>(-49.78f),
    static_cast<Eigen::half>(-93.3f),  static_cast<Eigen::half>(16.88f),   static_cast<Eigen::half>(75.4f),
    static_cast<Eigen::half>(-62.34f), static_cast<Eigen::half>(-66.2f),   static_cast<Eigen::half>(83.44f),
    static_cast<Eigen::half>(29.16f),  static_cast<Eigen::half>(-98.7f),   static_cast<Eigen::half>(-75.56f),
    static_cast<Eigen::half>(-76.25f), static_cast<Eigen::half>(49.66f),   static_cast<Eigen::half>(-32.28f),
    static_cast<Eigen::half>(-81.0f),  static_cast<Eigen::half>(42.72f),   static_cast<Eigen::half>(-23.92f),
    static_cast<Eigen::half>(71.06f),  static_cast<Eigen::half>(-0.7515f), static_cast<Eigen::half>(67.25f),
    static_cast<Eigen::half>(-90.7f),  static_cast<Eigen::half>(61.47f),   static_cast<Eigen::half>(-30.89f),
    static_cast<Eigen::half>(-37.72f), static_cast<Eigen::half>(-90.25f),  static_cast<Eigen::half>(-16.73f),
    static_cast<Eigen::half>(96.1f),   static_cast<Eigen::half>(-64.4f),   static_cast<Eigen::half>(15.96f),
    static_cast<Eigen::half>(-87.75f), static_cast<Eigen::half>(94.1f),    static_cast<Eigen::half>(7.164f)};

static const int32_t kCumsumAxisF16[1] = {-1};

static const Eigen::half kCumsumOutputF16_ET_RT[60] = {
    static_cast<Eigen::half>(16.47f),   static_cast<Eigen::half>(58.34f),  static_cast<Eigen::half>(109.06f),
    static_cast<Eigen::half>(98.25f),   static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(-104.44f),
    static_cast<Eigen::half>(-90.44f),  static_cast<Eigen::half>(-139.2f), static_cast<Eigen::half>(-55.3f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(-111.7f), static_cast<Eigen::half>(-82.8f),
    static_cast<Eigen::half>(-81.06f),  static_cast<Eigen::half>(15.95f),  static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(112.7f),   static_cast<Eigen::half>(104.7f),  static_cast<Eigen::half>(101.25f),
    static_cast<Eigen::half>(59.22f),   static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(-76.5f),
    static_cast<Eigen::half>(-30.06f),  static_cast<Eigen::half>(58.44f),  static_cast<Eigen::half>(86.0f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(-14.31f), static_cast<Eigen::half>(-104.75f),
    static_cast<Eigen::half>(-9.22f),   static_cast<Eigen::half>(-49.78f), static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(-36.25f),  static_cast<Eigen::half>(-53.12f), static_cast<Eigen::half>(-128.5f),
    static_cast<Eigen::half>(-66.2f),   static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(-221.4f),
    static_cast<Eigen::half>(-250.5f),  static_cast<Eigen::half>(-151.8f), static_cast<Eigen::half>(-76.25f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(-94.5f),  static_cast<Eigen::half>(-62.2f),
    static_cast<Eigen::half>(18.8f),    static_cast<Eigen::half>(-23.92f), static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(37.28f),   static_cast<Eigen::half>(38.03f),  static_cast<Eigen::half>(-29.22f),
    static_cast<Eigen::half>(61.47f),   static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(-48.6f),
    static_cast<Eigen::half>(-10.875f), static_cast<Eigen::half>(79.4f),   static_cast<Eigen::half>(96.1f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(29.53f),  static_cast<Eigen::half>(13.56f),
    static_cast<Eigen::half>(101.3f),   static_cast<Eigen::half>(7.164f),  static_cast<Eigen::half>(0.0f),
};

static const Eigen::half kCumsumOutputF16_EF_RF[60] = {
    static_cast<Eigen::half>(10.78f),   static_cast<Eigen::half>(-31.1f),  static_cast<Eigen::half>(-81.8f),
    static_cast<Eigen::half>(-71.0f),   static_cast<Eigen::half>(27.25f),  static_cast<Eigen::half>(-78.6f),
    static_cast<Eigen::half>(-92.6f),   static_cast<Eigen::half>(-43.8f),  static_cast<Eigen::half>(-127.75f),
    static_cast<Eigen::half>(-183.0f),  static_cast<Eigen::half>(68.25f),  static_cast<Eigen::half>(39.38f),
    static_cast<Eigen::half>(37.62f),   static_cast<Eigen::half>(-59.38f), static_cast<Eigen::half>(-43.44f),
    static_cast<Eigen::half>(80.4f),    static_cast<Eigen::half>(88.4f),   static_cast<Eigen::half>(91.8f),
    static_cast<Eigen::half>(133.9f),   static_cast<Eigen::half>(193.1f),  static_cast<Eigen::half>(-36.72f),
    static_cast<Eigen::half>(-83.2f),   static_cast<Eigen::half>(-171.8f), static_cast<Eigen::half>(-199.2f),
    static_cast<Eigen::half>(-113.25f), static_cast<Eigen::half>(29.12f),  static_cast<Eigen::half>(119.56f),
    static_cast<Eigen::half>(24.06f),   static_cast<Eigen::half>(64.6f),   static_cast<Eigen::half>(14.84f),
    static_cast<Eigen::half>(-93.3f),   static_cast<Eigen::half>(-76.44f), static_cast<Eigen::half>(-1.0625f),
    static_cast<Eigen::half>(-63.4f),   static_cast<Eigen::half>(-129.6f), static_cast<Eigen::half>(83.44f),
    static_cast<Eigen::half>(112.6f),   static_cast<Eigen::half>(13.94f),  static_cast<Eigen::half>(-61.62f),
    static_cast<Eigen::half>(-137.9f),  static_cast<Eigen::half>(49.66f),  static_cast<Eigen::half>(17.38f),
    static_cast<Eigen::half>(-63.62f),  static_cast<Eigen::half>(-20.9f),  static_cast<Eigen::half>(-44.8f),
    static_cast<Eigen::half>(71.06f),   static_cast<Eigen::half>(70.3f),   static_cast<Eigen::half>(137.5f),
    static_cast<Eigen::half>(46.8f),    static_cast<Eigen::half>(108.25f), static_cast<Eigen::half>(-30.89f),
    static_cast<Eigen::half>(-68.6f),   static_cast<Eigen::half>(-158.9f), static_cast<Eigen::half>(-175.6f),
    static_cast<Eigen::half>(-79.5f),   static_cast<Eigen::half>(-64.4f),  static_cast<Eigen::half>(-48.4f),
    static_cast<Eigen::half>(-136.1f),  static_cast<Eigen::half>(-42.0f),  static_cast<Eigen::half>(-34.84f)};

static const Eigen::half kCumsumOutputF16_ET_RF[60] = {
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(10.78f),  static_cast<Eigen::half>(-31.1f),
    static_cast<Eigen::half>(-81.8f),   static_cast<Eigen::half>(-71.0f),  static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(-78.6f),   static_cast<Eigen::half>(-92.6f),  static_cast<Eigen::half>(-43.8f),
    static_cast<Eigen::half>(-127.75f), static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(68.25f),
    static_cast<Eigen::half>(39.38f),   static_cast<Eigen::half>(37.62f),  static_cast<Eigen::half>(-59.38f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(80.4f),   static_cast<Eigen::half>(88.4f),
    static_cast<Eigen::half>(91.8f),    static_cast<Eigen::half>(133.9f),  static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(-36.72f),  static_cast<Eigen::half>(-83.2f),  static_cast<Eigen::half>(-171.8f),
    static_cast<Eigen::half>(-199.2f),  static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(29.12f),
    static_cast<Eigen::half>(119.56f),  static_cast<Eigen::half>(24.06f),  static_cast<Eigen::half>(64.6f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(-93.3f),  static_cast<Eigen::half>(-76.44f),
    static_cast<Eigen::half>(-1.0625f), static_cast<Eigen::half>(-63.4f),  static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(83.44f),   static_cast<Eigen::half>(112.6f),  static_cast<Eigen::half>(13.94f),
    static_cast<Eigen::half>(-61.62f),  static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(49.66f),
    static_cast<Eigen::half>(17.38f),   static_cast<Eigen::half>(-63.62f), static_cast<Eigen::half>(-20.9f),
    static_cast<Eigen::half>(0.0f),     static_cast<Eigen::half>(71.06f),  static_cast<Eigen::half>(70.3f),
    static_cast<Eigen::half>(137.5f),   static_cast<Eigen::half>(46.8f),   static_cast<Eigen::half>(0.0f),
    static_cast<Eigen::half>(-30.89f),  static_cast<Eigen::half>(-68.6f),  static_cast<Eigen::half>(-158.9f),
    static_cast<Eigen::half>(-175.6f),  static_cast<Eigen::half>(0.0f),    static_cast<Eigen::half>(-64.4f),
    static_cast<Eigen::half>(-48.4f),   static_cast<Eigen::half>(-136.1f), static_cast<Eigen::half>(-42.0f)};

static const Eigen::half kCumsumOutputF16_EF_RT[60] = {
    static_cast<Eigen::half>(27.25f),   static_cast<Eigen::half>(16.47f),   static_cast<Eigen::half>(58.34f),
    static_cast<Eigen::half>(109.06f),  static_cast<Eigen::half>(98.25f),   static_cast<Eigen::half>(-183.0f),
    static_cast<Eigen::half>(-104.44f), static_cast<Eigen::half>(-90.44f),  static_cast<Eigen::half>(-139.2f),
    static_cast<Eigen::half>(-55.3f),   static_cast<Eigen::half>(-43.44f),  static_cast<Eigen::half>(-111.7f),
    static_cast<Eigen::half>(-82.8f),   static_cast<Eigen::half>(-81.06f),  static_cast<Eigen::half>(15.95f),
    static_cast<Eigen::half>(193.0f),   static_cast<Eigen::half>(112.7f),   static_cast<Eigen::half>(104.7f),
    static_cast<Eigen::half>(101.25f),  static_cast<Eigen::half>(59.22f),   static_cast<Eigen::half>(-113.25f),
    static_cast<Eigen::half>(-76.5f),   static_cast<Eigen::half>(-30.06f),  static_cast<Eigen::half>(58.44f),
    static_cast<Eigen::half>(86.0f),    static_cast<Eigen::half>(14.81f),   static_cast<Eigen::half>(-14.31f),
    static_cast<Eigen::half>(-104.75f), static_cast<Eigen::half>(-9.22f),   static_cast<Eigen::half>(-49.78f),
    static_cast<Eigen::half>(-129.5f),  static_cast<Eigen::half>(-36.25f),  static_cast<Eigen::half>(-53.12f),
    static_cast<Eigen::half>(-128.5f),  static_cast<Eigen::half>(-66.2f),   static_cast<Eigen::half>(-138.0f),
    static_cast<Eigen::half>(-221.4f),  static_cast<Eigen::half>(-250.5f),  static_cast<Eigen::half>(-151.8f),
    static_cast<Eigen::half>(-76.25f),  static_cast<Eigen::half>(-44.84f),  static_cast<Eigen::half>(-94.5f),
    static_cast<Eigen::half>(-62.2f),   static_cast<Eigen::half>(18.8f),    static_cast<Eigen::half>(-23.92f),
    static_cast<Eigen::half>(108.4f),   static_cast<Eigen::half>(37.28f),   static_cast<Eigen::half>(38.03f),
    static_cast<Eigen::half>(-29.22f),  static_cast<Eigen::half>(61.47f),   static_cast<Eigen::half>(-79.5f),
    static_cast<Eigen::half>(-48.6f),   static_cast<Eigen::half>(-10.875f), static_cast<Eigen::half>(79.4f),
    static_cast<Eigen::half>(96.1f),    static_cast<Eigen::half>(-34.84f),  static_cast<Eigen::half>(29.53f),
    static_cast<Eigen::half>(13.56f),   static_cast<Eigen::half>(101.3f),   static_cast<Eigen::half>(7.164f),
};

const complex<float> kCumsumInputComplex64[] = {
    {5.53906f, 7.556307f},     {2.906982f, 6.0400014f},   {2.4645746f, 6.724752f},   {5.5401525f, 5.6381016f},
    {9.912748f, 1.1879822f},   {1.0681784f, 7.0246205f},  {4.2988396f, 7.786666f},   {7.441078f, 4.7914596f},
    {0.8021204f, 8.454078f},   {2.2337294f, 9.955402f},   {8.411811f, 3.460961f},    {3.5571952f, 6.6695905f},
    {4.9127684f, 8.680267f},   {0.1488914f, 4.06674f},    {5.797472f, 8.660624f},    {9.018072f, 9.831862f},
    {5.39886f, 0.5507977f},    {5.1720185f, 2.909996f},   {7.1013856f, 3.729373f},   {7.9603214f, 2.1822631f},
    {3.164598f, 0.43376127f},  {2.6760952f, 0.65103865f}, {0.57463145f, 2.3281841f}, {3.6221468f, 5.1316557f},
    {9.299807f, 5.491883f},    {6.4558673f, 5.028384f},   {9.523138f, 6.7180853f},   {0.22626387f, 6.111597f},
    {7.028024f, 1.8402656f},   {2.5114818f, 9.507145f},   {0.33529425f, 9.139737f},  {5.8435025f, 8.615855f},
    {8.769209f, 0.30162644f},  {1.8834907f, 0.5703028f},  {1.6920036f, 1.6485615f},  {9.171918f, 0.6310439f},
    {6.4580445f, 0.28866547f}, {0.06413611f, 0.3973825f}, {1.2230002f, 1.7179115f},  {1.1888297f, 6.013071f},
    {7.483581f, 5.5245748f},   {3.3859046f, 6.892145f},   {0.95132154f, 7.661414f},  {7.1361494f, 3.971761f},
    {3.8037605f, 7.3133984f},  {8.553995f, 8.106712f},    {4.9624243f, 2.768255f},   {8.362767f, 3.0574098f},
    {0.46602836f, 9.053704f},  {8.072726f, 7.056997f},    {3.4554381f, 2.6155517f},  {3.1141925f, 5.4117885f},
    {0.48653412f, 1.0547354f}, {4.1634207f, 9.086554f},   {9.806167f, 9.703933f},    {1.7821459f, 3.8735166f},
    {5.798192f, 2.252236f},    {0.612733f, 8.46156f},     {9.707471f, 2.4276192f},   {5.3582354f, 2.6543605f}};

const complex<float> kCumsumExpOutputComplex64EFRF[] = {
    {5.53906f, 7.556307f},    {2.906982f, 6.0400014f},  {2.4645746f, 6.724752f},  {5.5401525f, 5.6381016f},
    {9.912748f, 1.1879822f},  {1.0681784f, 7.0246205f}, {4.2988396f, 7.786666f},  {7.441078f, 4.7914596f},
    {0.8021204f, 8.454078f},  {2.2337294f, 9.955402f},  {8.411811f, 3.460961f},   {3.5571952f, 6.6695905f},
    {4.9127684f, 8.680267f},  {0.1488914f, 4.06674f},   {5.797472f, 8.660624f},   {9.018072f, 9.831862f},
    {5.39886f, 0.5507977f},   {5.1720185f, 2.909996f},  {7.1013856f, 3.729373f},  {7.9603214f, 2.1822631f},
    {8.703658f, 7.990068f},   {5.5830774f, 6.69104f},   {3.039206f, 9.052937f},   {9.162299f, 10.769757f},
    {19.212555f, 6.679865f},  {7.524046f, 12.053005f},  {13.821978f, 14.504751f}, {7.667342f, 10.903057f},
    {7.8301444f, 10.294343f}, {4.745211f, 19.462547f},  {8.747105f, 12.600698f},  {9.400698f, 15.285446f},
    {13.681977f, 8.981894f},  {2.032382f, 4.637043f},   {7.4894757f, 10.309185f}, {18.189991f, 10.462906f},
    {11.856905f, 0.8394632f}, {5.2361546f, 3.3073785f}, {8.324386f, 5.4472847f},  {9.149151f, 8.195334f},
    {16.187239f, 13.514643f}, {8.968982f, 13.583185f},  {3.9905276f, 16.714352f}, {16.298449f, 14.741518f},
    {23.016315f, 13.993263f}, {16.078041f, 20.159718f}, {18.7844f, 17.273006f},   {16.03011f, 13.960466f},
    {8.296173f, 19.348047f},  {12.817938f, 26.519545f}, {12.202543f, 15.21625f},  {12.514891f, 20.697235f},
    {14.168511f, 10.036629f}, {6.1958027f, 13.723597f}, {17.295643f, 20.013119f}, {19.972137f, 14.336422f},
    {17.655098f, 3.0916991f}, {5.8488874f, 11.768939f}, {18.031857f, 7.8749037f}, {14.507386f, 10.849695f}};

template <typename T>
void CalcExpectWithType(const NodeDef& node_def, bool exclusive, bool reverse, T expect_out[])
{
    auto input_data = reinterpret_cast<T*>(node_def.MutableInputs(0)->GetData());
    auto axis_data = reinterpret_cast<int32_t*>(node_def.MutableInputs(1)->GetData());
    int32_t axis = *axis_data;
    auto shape = node_def.MutableInputs(0)->GetTensorShape();
    const int64_t rank = shape->GetDims();
    if (axis < 0)
        axis += shape->GetDims();
    size_t inner = 1;
    size_t outer = 1;
    size_t depth = 1;
    for (int32_t i = 0; i < rank; ++i) {
        if (i < axis)
            inner *= shape->GetDimSize(i);
        else if (i > axis)
            outer *= shape->GetDimSize(i);
        else
            depth = shape->GetDimSize(i);
    }
    for (size_t outer_index = 0; outer_index < outer; ++outer_index) {
        size_t outer_index_adj;
        if (reverse)
            outer_index_adj = (outer - 1) - outer_index;
        else
            outer_index_adj = outer_index;
        for (size_t inner_index = 0; inner_index < inner; inner_index++) {
            // T accumulator = 0;
            auto accumulator = static_cast<T>(0);
            size_t inner_index_adj;
            if (reverse)
                inner_index_adj = (inner - 1) - inner_index;
            else
                inner_index_adj = inner_index;
            for (size_t depth_index = 0; depth_index < depth; depth_index++) {
                size_t depth_index_adj;
                if (reverse)
                    depth_index_adj = (depth - 1) - depth_index;
                else
                    depth_index_adj = depth_index;
                size_t index = outer_index_adj;
                index += inner_index_adj * depth * outer;
                index += depth_index_adj * outer;
                if (exclusive) {
                    expect_out[index] = accumulator;
                    accumulator += input_data[index];
                } else {
                    accumulator += input_data[index];
                    expect_out[index] = accumulator;
                }
            }
        }
    }
}
#define CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse) \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();  \
    NodeDefBuilder(node_def.get(), "Cumsum", "Cumsum")                \
        .Input({"x", data_types[0], shapes[0], datas[0]})             \
        .Input({"axis", data_types[1], shapes[1], datas[1]})          \
        .Output({"y", data_types[2], shapes[2], datas[2]})            \
        .Attr("exclusive", exclusive)                                 \
        .Attr("reverse", reverse)

template <typename T1, typename T2, typename T3>
void RunCumsumKernel(
    const vector<DataType>& data_types, vector<vector<int64_t>>& shapes, bool exclusive, bool reverse,
    const T1* input_data, const T2* axis_data, const T3* output_exp_data)
{
    uint64_t input_size = CalTotalElements(shapes, 0);
    uint64_t axis_size = CalTotalElements(shapes, 1);
    uint64_t output_size = CalTotalElements(shapes, 2);

    T1* input = new T1[input_size];
    T2* axis = new T2[axis_size];
    T3* output = new T3[output_size];
    T3* output_exp = new T3[output_size];

    for (uint64_t i = 0; i < input_size; ++i) {
        input[i] = input_data[i];
    }
    for (uint64_t i = 0; i < axis_size; ++i) {
        axis[i] = axis_data[i];
    }
    for (uint64_t i = 0; i < output_size; ++i) {
        output_exp[i] = output_exp_data[i];
    }

    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);

    delete[] input;
    delete[] axis;
    delete[] output;
    delete[] output_exp;
}

// only generate input data by SetRandomValue,
// and calculate output by youself function
template <typename T1, typename T2, typename T3>
void RunCumsumKernel2(vector<DataType> data_types, vector<vector<int64_t>>& shapes, bool exclusive, bool reverse)
{
    // gen data use SetRandomValue for input
    uint64_t input_size = CalTotalElements(shapes, 0);
    T1* input = new T1[input_size];
    SetRandomValue<T1>(input, input_size);

    // gen data use SetRandomValue for axis
    uint64_t axis_size = CalTotalElements(shapes, 1);
    T2* axis = new T2[axis_size];
    SetRandomValue<T2>(axis, axis_size, -3.0, 3.0);

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // calculate output_exp
    T3* output_exp = new T3[output_size];
    CalcExpectWithType<T1>(*node_def.get(), exclusive, reverse, output_exp);

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input;
    delete[] axis;
    delete[] output;
    delete[] output_exp;
}

// input0 is scalar not complex
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_EF_RF_SCALAR)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{}, {1}, {}};

    int32_t* input = new int32_t[1];
    input[0] = 2;
    int32_t* axis = new int32_t[1];
    axis[0] = 0;
    int32_t* output = new int32_t[1];

    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, false, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // calculate output_exp
    int32_t* output_exp = new int32_t[1];
    output_exp[0] = 2;

    bool compare = CompareResult(output, output_exp, 1);
    EXPECT_EQ(compare, true);
    delete[] input;
    delete[] axis;
    delete[] output;
    delete[] output_exp;
}

// input0 is scalar datatype is complex
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX128_EF_RF_SCALAR)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{}, {1}, {}};

    complex<double>* input = new complex<double>[1];
    input->real(2);
    input->imag(5);
    int32_t* axis = new int32_t[1];
    axis[0] = 0;
    complex<double>* output = new complex<double>[1];

    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, false, false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // calculate output_exp
    complex<double>* output_exp = new complex<double>[1];
    output_exp->real(2);
    output_exp->imag(5);

    bool compare = CompareResult(output, output_exp, 1);
    EXPECT_EQ(compare, true);
    delete[] input;
    delete[] axis;
    delete[] output;
    delete[] output_exp;
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_EF_RF)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool exclusive = false;
    bool reverse = false;

    RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(
        data_types, shapes, exclusive, reverse, kCumsumInputF16, kCumsumAxisF16, kCumsumOutputF16_EF_RF);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_ET_RF)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};

    bool exclusive = true;
    bool reverse = false;
    RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(
        data_types, shapes, exclusive, reverse, kCumsumInputF16, kCumsumAxisF16, kCumsumOutputF16_ET_RF);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_EF_RT)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};

    bool exclusive = false;
    bool reverse = true;
    RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(
        data_types, shapes, exclusive, reverse, kCumsumInputF16, kCumsumAxisF16, kCumsumOutputF16_EF_RT);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT16_ET_RT)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};

    bool exclusive = true;
    bool reverse = true;
    RunCumsumKernel<Eigen::half, int32_t, Eigen::half>(
        data_types, shapes, exclusive, reverse, kCumsumInputF16, kCumsumAxisF16, kCumsumOutputF16_ET_RT);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_EF_RF)
{
    vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool exclusive = false;
    bool reverse = false;
    RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive, reverse);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_ET_RF)
{
    vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool exclusive = true;
    bool reverse = false;
    RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_EF_RT)
{
    vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool exclusive = false;
    bool reverse = true;
    RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive, reverse);
}
TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT32_ET_RT)
{
    vector<DataType> data_types = {DT_UINT32, DT_INT32, DT_UINT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool exclusive = true;
    bool reverse = true;
    RunCumsumKernel2<uint32_t, int32_t, uint32_t>(data_types, shapes, exclusive, reverse);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT64_EF_RF)
{
    vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_UINT64};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool exclusive = false;
    bool reverse = false;
    RunCumsumKernel2<uint64_t, int32_t, uint64_t>(data_types, shapes, exclusive, reverse);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX64_EF_RF)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};

    bool exclusive = false;
    bool reverse = false;
    int32_t kCumsumAxisComplex64[] = {0};
    RunCumsumKernel<complex<float>, int32_t, complex<float>>(
        data_types, shapes, exclusive, reverse, kCumsumInputComplex64, kCumsumAxisComplex64,
        kCumsumExpOutputComplex64EFRF);
}

// exception instance
TEST_F(TEST_CUMSUM_UT, AXIS_SHAPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {2, 2}, {3, 4, 5}};
    int32_t input[60] = {(int32_t)1};
    int32_t axis[4] = {(int32_t)0};
    int32_t output[60] = {(bool)0};
    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};
    bool exclusive = false;
    bool reverse = false;
    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, AXIS_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_FLOAT, DT_INT32};
    vector<vector<int64_t>> shapes = {{4, 5, 6}, {1}, {4, 5, 6}};
    int32_t input[120] = {(int32_t)1};
    int32_t axis[1] = {1};
    int32_t output[120] = {(int32_t)0};
    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};
    bool exclusive = false;
    bool reverse = false;
    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_DTYPE_EXCEPTION)
{
    vector<DataType> data_types = {DT_STRING, DT_INT32, DT_STRING};
    vector<vector<int64_t>> shapes = {{4, 5, 6}, {1}, {4, 5, 6}};
    int32_t input[120] = {'1'};
    int32_t axis[1] = {1};
    int32_t output[120] = {'0'};
    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};
    bool exclusive = false;
    bool reverse = false;
    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_NULL_EXCEPTION)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    int32_t axis[1] = {(int32_t)0};
    int32_t output[60] = {(int32_t)0};
    vector<void*> datas = {(void*)nullptr, (void*)axis, (void*)output};
    bool exclusive = false;
    bool reverse = false;
    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CUMSUM_UT, INPUT_BOOL_UNSUPPORT)
{
    vector<DataType> data_types = {DT_BOOL, DT_INT32, DT_BOOL};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    bool input[60] = {(bool)1};
    int32_t axis[4] = {(int32_t)0};
    bool output[60] = {(bool)0};
    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};
    bool exclusive = false;
    bool reverse = false;
    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// Helper: run cumsum with a specific axis value (not random)
template <typename T1, typename T3>
void RunCumsumKernelWithAxis(
    vector<DataType> data_types, vector<vector<int64_t>>& shapes, bool exclusive, bool reverse, int32_t axis_val)
{
    uint64_t input_size = CalTotalElements(shapes, 0);
    T1* input = new T1[input_size];
    SetRandomValue<T1>(input, input_size);

    uint64_t axis_size = CalTotalElements(shapes, 1);
    int32_t* axis = new int32_t[axis_size];
    axis[0] = axis_val;

    uint64_t output_size = CalTotalElements(shapes, 2);
    T3* output = new T3[output_size];
    vector<void*> datas = {(void*)input, (void*)axis, (void*)output};

    CREATE_NODEDEF(shapes, data_types, datas, exclusive, reverse);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    T3* output_exp = new T3[output_size];
    CalcExpectWithType<T1>(*node_def.get(), exclusive, reverse, output_exp);

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
    delete[] input;
    delete[] axis;
    delete[] output;
    delete[] output_exp;
}

// ==================== Data type coverage ====================

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_FLOAT_EF_RF)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    RunCumsumKernel2<float, int32_t, float>(data_types, shapes, false, false);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_DOUBLE_ET_RT)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    RunCumsumKernel2<double, int32_t, double>(data_types, shapes, true, true);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT8_EF_RF)
{
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_INT8};
    vector<vector<int64_t>> shapes = {{4, 5, 3}, {1}, {4, 5, 3}};
    RunCumsumKernel2<int8_t, int32_t, int8_t>(data_types, shapes, false, false);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT16_EF_RT)
{
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_INT16};
    vector<vector<int64_t>> shapes = {{4, 5, 3}, {1}, {4, 5, 3}};
    RunCumsumKernel2<int16_t, int32_t, int16_t>(data_types, shapes, false, true);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT32_EF_RF)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    RunCumsumKernel2<int32_t, int32_t, int32_t>(data_types, shapes, false, false);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_INT64_ET_RF)
{
    vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    RunCumsumKernel2<int64_t, int32_t, int64_t>(data_types, shapes, true, false);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT8_EF_RF)
{
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_UINT8};
    vector<vector<int64_t>> shapes = {{4, 5, 3}, {1}, {4, 5, 3}};
    RunCumsumKernel2<uint8_t, int32_t, uint8_t>(data_types, shapes, false, false);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_UINT16_ET_RT)
{
    vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_UINT16};
    vector<vector<int64_t>> shapes = {{4, 5, 3}, {1}, {4, 5, 3}};
    RunCumsumKernel2<uint16_t, int32_t, uint16_t>(data_types, shapes, true, true);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX64_ET_RF)
{
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    RunCumsumKernel2<complex<float>, int32_t, complex<float>>(data_types, shapes, true, false);
}

TEST_F(TEST_CUMSUM_UT, DATA_TYPE_COMPLEX128_EF_RT)
{
    vector<DataType> data_types = {DT_COMPLEX128, DT_INT32, DT_COMPLEX128};
    vector<vector<int64_t>> shapes = {{3, 4, 5}, {1}, {3, 4, 5}};
    RunCumsumKernel2<complex<double>, int32_t, complex<double>>(data_types, shapes, false, true);
}

// ==================== Edge cases: 1D / 4D / specific axis ====================

TEST_F(TEST_CUMSUM_UT, SHAPE_1D_AXIS0)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{20}, {1}, {20}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, false, false, 0);
}

TEST_F(TEST_CUMSUM_UT, SHAPE_4D_AXIS2_ET_RT)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 3, 4, 5}, {1}, {2, 3, 4, 5}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, true, true, 2);
}

TEST_F(TEST_CUMSUM_UT, SHAPE_4D_AXIS_NEG1)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{2, 3, 4, 5}, {1}, {2, 3, 4, 5}};
    RunCumsumKernelWithAxis<double, double>(data_types, shapes, false, true, -1);
}

// ==================== Large tensor: ParallelFor path ====================

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_AXIS0_PARALLEL)
{
    // shape [256, 1024], axis=0 => inner=1, depth=256, outer=1024
    // data_size = 256*1024*4 = 1MB > 512KB threshold
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{256, 1024}, {1}, {256, 1024}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, false, false, 0);
}

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_AXIS1_PARALLEL)
{
    // shape [256, 1024], axis=1 => inner=256, depth=1024, outer=1
    // Old code: outer=1 => single core! New code: total_sequences=256 => parallel
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{256, 1024}, {1}, {256, 1024}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, false, false, 1);
}

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_AXIS_NEG1_ET_RT_PARALLEL)
{
    // shape [128, 1024], axis=-1 => inner=128, depth=1024, outer=1
    // Tests reverse + exclusive on large tensor with parallel path
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{128, 1024}, {1}, {128, 1024}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, true, true, -1);
}

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_COMPLEX64_PARALLEL)
{
    // shape [64, 1024], axis=1, complex64 => 64*1024*8 = 512KB => just at threshold
    // Use slightly larger to ensure parallel path
    vector<DataType> data_types = {DT_COMPLEX64, DT_INT32, DT_COMPLEX64};
    vector<vector<int64_t>> shapes = {{128, 1024}, {1}, {128, 1024}};
    RunCumsumKernelWithAxis<complex<float>, complex<float>>(data_types, shapes, false, false, 1);
}

// ==================== Inner >> outer optimization verification ====================

TEST_F(TEST_CUMSUM_UT, INNER_GT_OUTER_AXIS1)
{
    // shape [64, 8, 2], axis=1 => inner=64, depth=8, outer=2
    // total_sequences = 128 vs old parallel on outer=2
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{64, 8, 2}, {1}, {64, 8, 2}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, false, false, 1);
}

TEST_F(TEST_CUMSUM_UT, INNER_GT_OUTER_AXIS1_ET_RF)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{32, 16, 4}, {1}, {32, 16, 4}};
    RunCumsumKernelWithAxis<int32_t, int32_t>(data_types, shapes, true, false, 1);
}

TEST_F(TEST_CUMSUM_UT, INNER_GT_OUTER_AXIS1_EF_RT)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{32, 16, 4}, {1}, {32, 16, 4}};
    RunCumsumKernelWithAxis<double, double>(data_types, shapes, false, true, 1);
}

TEST_F(TEST_CUMSUM_UT, INNER_GT_OUTER_AXIS1_ET_RT)
{
    vector<DataType> data_types = {DT_INT64, DT_INT32, DT_INT64};
    vector<vector<int64_t>> shapes = {{32, 16, 4}, {1}, {32, 16, 4}};
    RunCumsumKernelWithAxis<int64_t, int64_t>(data_types, shapes, true, true, 1);
}

// ==================== Batched parallel path coverage ====================

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_AXIS1_BATCHED_PARALLEL)
{
    // shape [32, 32, 512], axis=1 => inner=32, depth=32, outer=512
    // data_size = 32*32*512*4 = 2MB > 512KB threshold
    // inner=32 >= avail_cores => Path 2 (batched, parallel over inner, full outer range)
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{32, 32, 512}, {1}, {32, 32, 512}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, false, false, 1);
}

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_AXIS0_OUTER_CHUNKING)
{
    // shape [256, 1024], axis=0 => inner=1, depth=256, outer=1024
    // data_size = 256*1024*4 = 1MB > 512KB threshold
    // inner=1 < avail_cores => Path 3 (batched with outer chunking)
    // This was the original bug: inner=1 caused single-core execution for 1MB data
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{256, 1024}, {1}, {256, 1024}};
    RunCumsumKernelWithAxis<float, float>(data_types, shapes, false, false, 0);
}

TEST_F(TEST_CUMSUM_UT, LARGE_TENSOR_AXIS0_OUTER_CHUNKING_ET_RT)
{
    // Same shape as above but with exclusive=true, reverse=true to cover all code paths
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{128, 1024}, {1}, {128, 1024}};
    RunCumsumKernelWithAxis<double, double>(data_types, shapes, true, true, 0);
}