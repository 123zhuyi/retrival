/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/equal_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(idx, name, dtype, shape)  \
  do {                                                                                    \
    std::string name##idx = "placeholder" + std::to_string(idx);                          \
    auto placeholder##idx = op::Data(name##idx.c_str()).set_attr_index(0);                \
    TensorDesc placeholder##idx##_desc(ge::Shape(shape), FORMAT_ND, dtype);               \
    placeholder##idx##_desc.SetPlacement(ge::kPlacementHost);                             \
    Tensor tensor_placeholder##idx;                                                       \
    ret = GenOnesDataFloat32(shape, tensor_placeholder##idx, placeholder##idx##_desc, 2.3f); \
    if (ret != SUCCESS) {                                                                 \
        printf("%s - ERROR - generate input failed\n", GetTime().c_str());                \
        return FAILED;                                                                    \
    }                                                                                     \
    placeholder##idx.update_input_desc_x(placeholder##idx##_desc);                        \
    graph.AddOp(placeholder##idx);                                                        \
    input.push_back(tensor_placeholder##idx);                                             \
    equal1.set_input_##name(placeholder##idx);                                            \
    inputs.push_back(placeholder##idx);                                                   \
  } while(0)

#define ADD_OUTPUT(idx, name, dtype, shape)                                              \
  do {                                                                                 \
    TensorDesc name##idx##_desc(ge::Shape(shape), FORMAT_ND, dtype);                     \
    equal1.update_output_desc_##name(name##idx##_desc);                                  \
  } while(0)

string GetTime() {
  time_t t;
  time(&t);
  char buf[64];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&t));
  return buf;
}

uint32_t GetDataTypeSize(DataType dt) {
  if (dt == DT_FLOAT) return 4;
  if (dt == DT_FLOAT16) return 2;
  if (dt == DT_BF16) return 2;
  if (dt == DT_BOOL) return 1;
  return 4;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor &tensor,
                           TensorDesc &desc, float value) {
  desc.SetRealDimCnt(shapes.size());

  size_t size = 1;
  for (auto s: shapes) size *= s;

  uint32_t data_len = size * 4;
  float *p = new float[size];

  for (size_t i = 0; i < size; ++i)
    p[i] = value + (i % 3) * 0.4f;

  tensor = Tensor(desc, (uint8_t *) p, data_len);
  return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData) {
  FILE *fp = fopen(bin_file.c_str(), "wb");
  if (fp == nullptr) {
    return FAILED;
  }
  size_t written = fwrite(inputData, 1, data_size, fp);
  fclose(fp);
  if (written != data_size) {
    return FAILED;
  }
  return SUCCESS;
}

int CreateOppInGraph(
    DataType inDtype,
    vector<Tensor> &input, vector<Operator> &inputs,
    vector<Operator> &outputs, Graph &graph) {
  Status ret = SUCCESS;
  auto equal1 = op::Equal("equal1");
  vector<int64_t> xShape = {4, 3};
  ADD_INPUT(1, x1, inDtype, xShape);
  ADD_INPUT(2, x2, inDtype, xShape);
  {
    Tensor &tx2 = input.back();
    float *p = reinterpret_cast<float *>(tx2.GetData());
    int64_t total = xShape[0] * xShape[1];
    for (int i = 0; i < 3 && i < total; i++) {
      p[i] += 1.0f;
    }
  }
  ADD_OUTPUT(1, y, DT_BOOL, xShape);
  outputs.push_back(equal1);
  return SUCCESS;
}

int main(int argc, char *argv[]) {
  const char *graph_name = "tc_ge_irrun_test";
  Graph graph(graph_name);
  std::vector<ge::Tensor> input;

  printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
  std::map<AscendString, AscendString> global_options = {
      {"ge.exec.deviceId", "0"},
      {"ge.graphRunMode",  "1"}
  };
  Status ret = ge::GEInitialize(global_options);
  if (ret != SUCCESS) {
    printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
    return FAILED;
  }
  printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

  std::vector<Operator> inputs{};
  std::vector<Operator> outputs{};

  if (argc > 1) {
    std::cout << argv[1] << std::endl;
  }

  DataType inDtype = DT_FLOAT;
  std::cout << inDtype << std::endl;

  ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
  if (ret != SUCCESS) {
    printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
    return FAILED;
  }

  if (!inputs.empty() && !outputs.empty()) {
    graph.SetInputs(inputs).SetOutputs(outputs);
  }

  std::map<AscendString, AscendString> build_options = {};
  printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
  ge::Session *session = new Session(build_options);
  if (session == nullptr) {
    printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
    return FAILED;
  }
  printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
  printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

  std::map<AscendString, AscendString> graph_options = {};

  uint32_t graph_id = 0;
  ret = session->AddGraph(graph_id, graph, graph_options);

  printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
  printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());

  std::string file_path = "./dump";
  aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

  printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
  std::vector<ge::Tensor> output;
  ret = session->RunGraph(graph_id, input, output);
  if (ret != SUCCESS) {
    printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
    delete session;
    GEFinalize();
    return FAILED;
  }
  printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

  int input_num = input.size();
  for (int i = 0; i < input_num; i++) {
    std::cout << "input " << i << " dtype :  "
              << input[i].GetTensorDesc().GetDataType() << std::endl;

    string input_file = "./tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin";
    uint8_t *input_data_i = input[i].GetData();
    int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();

    std::cout << "this is " << i << "th input, input shape size = " << input_shape << std::endl;

    uint32_t data_size =
        input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());

    WriteDataToFile(input_file.c_str(), data_size, input_data_i);
  }

  int output_num = output.size();
  for (int i = 0; i < output_num; i++) {
    std::cout << "output " << i << " dtype :  "
              << output[i].GetTensorDesc().GetDataType() << std::endl;

    string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";

    uint8_t *output_data_i = output[i].GetData();
    int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();

    std::cout << "this is " << i << "th output, output shape size = " << output_shape << std::endl;

    uint32_t data_size =
        output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());

    WriteDataToFile(output_file.c_str(), data_size, output_data_i);

    for (int64_t j = 0; j < output_shape; j++) {
      printf("result[%ld] = %d\n", j, output_data_i[j]);
    }
  }

  ge::AscendString error_msg = ge::GEGetErrorMsgV2();
  std::cout << "Error message: " << error_msg.GetString() << std::endl;

  ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
  std::cout << "Warning message: " << warning_msg.GetString() << std::endl;

  printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
  ret = ge::GEFinalize();
  if (ret != SUCCESS) {
    printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
    return FAILED;
  }
  printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());

  return SUCCESS;
}