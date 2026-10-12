//===-- TestDialect.td - BiShengIR Test dialect definition -*- tablegen -*-===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TESTDIALECT
#define TEST_TESTDIALECT

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "bishengir_test";
  let cppNamespace = "::bishengir_test";
}

#endif // TEST_TESTDIALECT
