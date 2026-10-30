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

class TEST_UNPACK_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, axis, num)            \
  auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();      \
  NodeDefBuilder node(node_def.get(), "Unpack", "Unpack");              \
  node.Input({"x", data_types[0], shapes[0], datas[0]})                 \
      .Attr("num", num)                                                 \
      .Attr("axis", axis);                                              \
  for (int i = 0; i < num; i++) {                                       \
    node.Output({"y", data_types[i + 1], shapes[i + 1], datas[i + 1]}); \
  }

#define ADD_CASE(case_name, aicpu_type, base_type, axis, num)                  \
  TEST_F(TEST_UNPACK_UT, TestUnpack_##case_name##_##aicpu_type) {              \
    if (num == 1) {                                                            \
      vector<DataType> data_types = {aicpu_type, aicpu_type};                  \
      vector<vector<int64_t>> shapes = {{3, 1}, {3}};                          \
      base_type input[3] = {(base_type)1, (base_type)2, (base_type)3};         \
      base_type output[3] = {(base_type)0};                                    \
      vector<void *> datas = {(void *)input, (void *)output};                  \
      CREATE_NODEDEF(shapes, data_types, datas, 1, num);                       \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
      base_type execpt_output[3] = {(base_type)1, (base_type)2, (base_type)3}; \
      EXPECT_EQ(CompareResult<base_type>(output, execpt_output, 3), true);     \
    } else if (axis == 0) {                                                    \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type,       \
                                     aicpu_type};                              \
      vector<vector<int64_t>> shapes = {{3, 3}, {3}, {3}, {3}};                \
      base_type input[9] = {(base_type)1, (base_type)2, (base_type)3,          \
                            (base_type)4, (base_type)5, (base_type)6,          \
                            (base_type)7, (base_type)8, (base_type)9};         \
      base_type output1[3] = {(base_type)0};                                   \
      base_type output2[3] = {(base_type)0};                                   \
      base_type output3[3] = {(base_type)0};                                   \
      vector<void *> datas = {(void *)input, (void *)output1, (void *)output2, \
                              (void *)output3};                                \
      CREATE_NODEDEF(shapes, data_types, datas, axis, 3);                      \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
      base_type execpt_output1[3] = {(base_type)1, (base_type)2,               \
                                     (base_type)3};                            \
      base_type execpt_output2[3] = {(base_type)4, (base_type)5,               \
                                     (base_type)6};                            \
      base_type execpt_output3[3] = {(base_type)7, (base_type)8,               \
                                     (base_type)9};                            \
      EXPECT_EQ(CompareResult<base_type>(output1, execpt_output1, 3), true);   \
      EXPECT_EQ(CompareResult<base_type>(output2, execpt_output2, 3), true);   \
      EXPECT_EQ(CompareResult<base_type>(output3, execpt_output3, 3), true);   \
    } else if (axis == 1) {                                                    \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type,       \
                                     aicpu_type};                              \
      vector<vector<int64_t>> shapes = {{3, 3}, {3}, {3}, {3}};                \
      base_type input[9] = {(base_type)1, (base_type)2, (base_type)3,          \
                            (base_type)4, (base_type)5, (base_type)6,          \
                            (base_type)7, (base_type)8, (base_type)9};         \
      base_type output1[3] = {(base_type)0};                                   \
      base_type output2[3] = {(base_type)0};                                   \
      base_type output3[3] = {(base_type)0};                                   \
      vector<void *> datas = {(void *)input, (void *)output1, (void *)output2, \
                              (void *)output3};                                \
      CREATE_NODEDEF(shapes, data_types, datas, axis, 3);                      \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
      base_type execpt_output1[3] = {(base_type)1, (base_type)4,               \
                                     (base_type)7};                            \
      base_type execpt_output2[3] = {(base_type)2, (base_type)5,               \
                                     (base_type)8};                            \
      base_type execpt_output3[3] = {(base_type)3, (base_type)6,               \
                                     (base_type)9};                            \
      EXPECT_EQ(CompareResult<base_type>(output1, execpt_output1, 3), true);   \
      EXPECT_EQ(CompareResult<base_type>(output2, execpt_output2, 3), true);   \
      EXPECT_EQ(CompareResult<base_type>(output3, execpt_output3, 3), true);   \
    } else if (axis == 2) {                                                    \
      vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type};      \
      vector<vector<int64_t>> shapes = {                                       \
          {1, 1, 2, 1, 3}, {1, 1, 1, 3}, {1, 1, 1, 3}};                        \
      base_type input[6] = {(base_type)1, (base_type)2, (base_type)3,          \
                            (base_type)4, (base_type)5, (base_type)6};         \
      base_type output1[3] = {(base_type)0};                                   \
      base_type output2[3] = {(base_type)0};                                   \
      vector<void *> datas = {(void *)input, (void *)output1,                  \
                              (void *)output2};                                \
      CREATE_NODEDEF(shapes, data_types, datas, axis, num);                    \
      RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);                            \
      base_type execpt_output1[3] = {(base_type)1, (base_type)2,               \
                                     (base_type)3};                            \
      base_type execpt_output2[3] = {(base_type)4, (base_type)5,               \
                                     (base_type)6};                            \
      EXPECT_EQ(CompareResult<base_type>(output1, execpt_output1, 3), true);   \
      EXPECT_EQ(CompareResult<base_type>(output2, execpt_output2, 3), true);   \
    }                                                                          \
  }

#define ADD_CASE_FAILED(case_name, aicpu_type, base_type, axis, num)         \
  TEST_F(TEST_UNPACK_UT, TestUnpack_##case_name##_##aicpu_type) {            \
    vector<DataType> data_types = {aicpu_type, aicpu_type, aicpu_type,       \
                                   aicpu_type};                              \
    vector<vector<int64_t>> shapes = {{3, 3}, {3}, {3}, {3}};                \
    base_type input[9] = {(base_type)1, (base_type)2, (base_type)3,          \
                          (base_type)4, (base_type)5, (base_type)6,          \
                          (base_type)7, (base_type)8, (base_type)9};         \
    base_type output1[3] = {(base_type)0};                                   \
    base_type output2[3] = {(base_type)0};                                   \
    base_type output3[3] = {(base_type)0};                                   \
    vector<void *> datas = {(void *)input, (void *)output1, (void *)output2, \
                            (void *)output3};                                \
    CREATE_NODEDEF(shapes, data_types, datas, axis, num);                    \
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);                 \
  }

//********************params: num*******************
int64_t unpack_num0 = 0;
int64_t unpack_num1 = 1;
int64_t unpack_num2 = 2;
int64_t unpack_num3 = 3;
int64_t unpack_num4 = 4;

//********************params: axis*******************
int64_t unpack_axis0 = 0;
int64_t unpack_axis1 = 1;
int64_t unpack_axis2 = 2;

//********************unpack to one tensor*******************
ADD_CASE(unpack_to_one_tensor, DT_FLOAT, float, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_BOOL, bool, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_DOUBLE, double, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_FLOAT16, Eigen::half, unpack_axis1,
         unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_INT8, int8_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_INT16, int16_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_INT32, int32_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_INT64, int64_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_UINT8, uint8_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_UINT16, uint16_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_UINT32, uint32_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_UINT64, uint64_t, unpack_axis1, unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_COMPLEX64, std::complex<float>, unpack_axis1,
         unpack_num1)

ADD_CASE(unpack_to_one_tensor, DT_COMPLEX128, std::complex<double>, unpack_axis1,
         unpack_num1)

//********************unpack through axis 0*******************
ADD_CASE(unpack_to_three_tensors_through_axis0, DT_FLOAT, float, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_BOOL, bool, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_DOUBLE, double, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_FLOAT16, Eigen::half,
         unpack_axis0, unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_UINT8, uint8_t, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_UINT16, uint16_t,
         unpack_axis0, unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_UINT32, uint32_t,
         unpack_axis0, unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_UINT64, uint64_t,
         unpack_axis0, unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_INT32, int32_t, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_INT16, int16_t, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_INT64, int64_t, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_INT8, int8_t, unpack_axis0,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis0, DT_COMPLEX64, std::complex<float>,
         unpack_axis0, unpack_num3)
         
ADD_CASE(unpack_to_three_tensors_through_axis0, DT_COMPLEX128, std::complex<double>,
         unpack_axis0, unpack_num3)

//********************unpack to serval tensors not through axis0
ADD_CASE(unpack_to_three_tensors_through_axis1, DT_INT8, int8_t, unpack_axis1,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_INT16, int16_t, unpack_axis1,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_INT32, int32_t, unpack_axis1,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_INT64, int64_t, unpack_axis1,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_UINT8, uint8_t, unpack_axis1,
         unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_UINT16, uint16_t,
         unpack_axis1, unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_UINT32, uint32_t,
         unpack_axis1, unpack_num3)

ADD_CASE(unpack_to_three_tensors_through_axis1, DT_UINT64, uint64_t,
         unpack_axis1, unpack_num3)

ADD_CASE(unpack_to_two_tensors_through_axis2, DT_FLOAT16, Eigen::half,
         unpack_axis2, unpack_num2)

ADD_CASE(unpack_to_two_tensors_through_axis2, DT_BOOL, bool, unpack_axis2,
         unpack_num2)

ADD_CASE(unpack_to_two_tensors_through_axis2, DT_DOUBLE, double, unpack_axis2,
         unpack_num2)

ADD_CASE(unpack_to_two_tensors_through_axis2, DT_FLOAT, float, unpack_axis2,
         unpack_num2)

ADD_CASE(unpack_to_two_tensors_through_axis2, DT_COMPLEX64, std::complex<float>,
         unpack_axis2, unpack_num2)

ADD_CASE(unpack_to_two_tensors_through_axis2, DT_COMPLEX128, std::complex<double>,
         unpack_axis2, unpack_num2)

//********************unpack failed*******************
ADD_CASE_FAILED(unpack_num_not_equal_specified_axis_size, DT_INT64, int64_t,
                unpack_axis1, unpack_num1)

ADD_CASE_FAILED(unpack_axis_illegal, DT_INT32, int32_t, unpack_axis2,
                unpack_num3)

ADD_CASE_FAILED(unpack_axis_illegal, DT_INT16, int16_t, unpack_axis2,
                unpack_num3)

ADD_CASE_FAILED(unpack_type_illegal, DT_STRING, int16_t, unpack_axis1,
                unpack_num3)