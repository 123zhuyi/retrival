//===- RegisterEverything.cpp - Register all BiShengIR entities -----------===//
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

#include "bishengir-c/RegisterEverything.h"

#include "bishengir/InitAllDialects.h"
#include "bishengir/InitAllExtensions.h"
#include "bishengir/InitAllPasses.h"
#include "mlir/CAPI/IR.h"

void bishengirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  bishengir::registerAllDialects(registry);
  bishengir::registerAllExtensions(registry);
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void bishengirRegisterAllPasses() {
  bishengir::registerAllPasses();
}
