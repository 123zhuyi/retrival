//===- PopulatePatterns.h -- Populate Torch to HFusion patterns -*- C++ -*-===//
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
#ifndef BISHENGIR_CONVERSION_TORCHTOHFUSION_POPULATEPATTERNS_H
#define BISHENGIR_CONVERSION_TORCHTOHFUSION_POPULATEPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
struct ConvertTorchToHFusionOptions;

// -----------------------------------------------------------------------------
// TorchToNamedOp Conversion Patterns
// -----------------------------------------------------------------------------
void populateElementWisePatternsAndLegality(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            ConversionTarget &target);

void populateReductionPatternsAndLegality(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target);

void populateDataMovementPatternsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, const ConvertTorchToHFusionOptions &options);

void populateUncategorizedPatternsAndLegality(TypeConverter &typeConverter,
                                              RewritePatternSet &patterns,
                                              ConversionTarget &target);

void populateTensorConstructorsPatternsAndLegality(TypeConverter &typeConverter,
                                                   RewritePatternSet &patterns,
                                                   ConversionTarget &target);
} // namespace mlir

#endif // BISHENGIR_CONVERSION_TORCHTOHFUSION_POPULATEPATTERNS_H
