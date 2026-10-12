//===- AnnotationPasses.cpp - C API for Annotation Dialect Passes -*- C -*-===//
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

#include "bishengir/Dialect/Annotation/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "bishengir/Dialect/Annotation/Transforms/Passes.capi.h.inc"

using namespace mlir;
using namespace mlir::annotation;

#ifdef __cplusplus
extern "C" {
#endif

#include "bishengir/Dialect/Annotation/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
