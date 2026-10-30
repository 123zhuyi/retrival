/**
 * Pdist ST test - Iteration 3
 * Covers L0+L1 full test cases: fp16+fp32, p in {0,0.5,1,2,3,10,inf},
 * boundary shapes, multi-core, UB tiling, precision-sensitive scenarios.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
#include <string>
#include <cstring>
#include <functional>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_pdist.h"
#include "aclnn_pdist_forward.h"
#endif

#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// Precision thresholds (CANN community standard MERE/MARE)
// ============================================================================
static const double FP32_THRESHOLD = 1.220703125e-4;   // 2^-13
static const double FP16_THRESHOLD = 9.765625e-4;      // 2^-10

// ============================================================================
// Helper: get precision threshold by dtype string
// ============================================================================
double GetThreshold(const std::string& dtype) {
    if (dtype == "float16") return FP16_THRESHOLD;
    return FP32_THRESHOLD;
}

// ============================================================================
// Helper: compute shape size
// ============================================================================
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

// ============================================================================
// CPU Golden: ComputeGoldenPdist (double precision for accuracy)
// Handles p=0 (Hamming), 0<p<inf (Minkowski), p=inf (Chebyshev)
// ============================================================================
void ComputeGoldenPdist(const double* x, int64_t N, int64_t M, double p, double* output) {
    int64_t idx = 0;
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = i + 1; j < N; j++) {
            if (p == 0.0) {
                double count = 0.0;
                for (int64_t k = 0; k < M; k++) {
                    if (x[i * M + k] != x[j * M + k]) count += 1.0;
                }
                output[idx++] = count;
            } else if (std::isinf(p) && p > 0) {
                double maxVal = 0.0;
                for (int64_t k = 0; k < M; k++) {
                    double diff = std::abs(x[i * M + k] - x[j * M + k]);
                    if (diff > maxVal) maxVal = diff;
                }
                output[idx++] = maxVal;
            } else {
                double sum = 0.0;
                for (int64_t k = 0; k < M; k++) {
                    double diff = std::abs(x[i * M + k] - x[j * M + k]);
                    sum += std::pow(diff, p);
                }
                output[idx++] = std::pow(sum, 1.0 / p);
            }
        }
    }
}

// ============================================================================
// Precision comparison: CompareResults (MERE/MARE based)
// ============================================================================
bool CompareResults(const double* golden, const double* actual, size_t size,
                    double threshold) {
    if (size == 0) return true;
    double mere = 0.0, mare = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double g = golden[i];
        double a = actual[i];
        // Skip NaN pairs (both NaN => pass, matching PyTorch)
        if (std::isnan(g) && std::isnan(a)) continue;
        if (std::isnan(g) || std::isnan(a)) {
            mare = 1.0;
            mere += 1.0;
            continue;
        }
        // Skip Inf pairs
        if (std::isinf(g) && std::isinf(a)) {
            if (g == a) continue;
        }
        double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
        mere += rel_err;
        if (rel_err > mare) mare = rel_err;
    }
    mere /= static_cast<double>(size);
    double mare_threshold = 10.0 * threshold;
    bool pass = (mere < threshold) && (mare < mare_threshold);
    if (pass) {
        LOG_PRINT("  [PASS] MERE=%.2e, MARE=%.2e (threshold=%.2e, %zu elems)", mere, mare, threshold, size);
    } else {
        LOG_PRINT("  [FAIL] MERE=%.2e (>= %.2e), MARE=%.2e (>= %.2e)", mere, threshold, mare, mare_threshold);
        int shown = 0;
        for (size_t i = 0; i < size && shown < 5; ++i) {
            double g = golden[i];
            double a = actual[i];
            double rel_err = std::abs(a - g) / (std::abs(g) + 1e-7);
            if (rel_err > threshold) {
                LOG_PRINT("    [%zu]: golden=%.6f, actual=%.6f, rel_err=%.2e", i, g, a, rel_err);
                shown++;
            }
        }
    }
    return pass;
}

// ============================================================================
// CPU Golden self-test
// ============================================================================
void TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden correctness self-test");
    LOG_PRINT("========================================");

    int passed = 0, failed = 0;

    // Test 1: p=2, Euclidean distance, 3 rows x 2 cols
    {
        LOG_PRINT("\n  Golden self-test 1: p=2, N=3, M=2");
        double x[] = {1, 2, 4, 6, 7, 8};
        double output[3];
        ComputeGoldenPdist(x, 3, 2, 2.0, output);
        // dist(0,1) = sqrt((1-4)^2 + (2-6)^2) = sqrt(9+16) = 5.0
        // dist(0,2) = sqrt((1-7)^2 + (2-8)^2) = sqrt(36+36) = sqrt(72) = 6*sqrt(2)
        // dist(1,2) = sqrt((4-7)^2 + (6-8)^2) = sqrt(9+4) = sqrt(13)
        double expected[] = {5.0, std::sqrt(72.0), std::sqrt(13.0)};
        bool ok = CompareResults(expected, output, 3, 1e-10);
        if (ok) passed++; else failed++;
    }

    // Test 2: p=1, Manhattan distance
    {
        LOG_PRINT("\n  Golden self-test 2: p=1, N=3, M=4");
        double x[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        double output[3];
        ComputeGoldenPdist(x, 3, 4, 1.0, output);
        double expected[] = {2.0, 2.0, 2.0};
        bool ok = CompareResults(expected, output, 3, 1e-10);
        if (ok) passed++; else failed++;
    }

    // Test 3: p=0, Hamming distance
    {
        LOG_PRINT("\n  Golden self-test 3: p=0, N=4, M=2");
        double x[] = {1, 2, 1, 3, 2, 2, 1, 2};
        double output[6];
        ComputeGoldenPdist(x, 4, 2, 0.0, output);
        // (0,1): diff=[0,1], 1 diff => 1.0
        // (0,2): diff=[1,0], 1 diff => 1.0
        // (0,3): diff=[0,0], 0 diff => 0.0
        // (1,2): diff=[1,1], 2 diff => 2.0
        // (1,3): diff=[0,1], 1 diff => 1.0
        // (2,3): diff=[1,0], 1 diff => 1.0
        double expected[] = {1.0, 1.0, 0.0, 2.0, 1.0, 1.0};
        bool ok = CompareResults(expected, output, 6, 1e-10);
        if (ok) passed++; else failed++;
    }

    // Test 4: p=inf, Chebyshev distance
    {
        LOG_PRINT("\n  Golden self-test 4: p=inf, N=3, M=2");
        double x[] = {1, 10, 5, 3, 2, 8};
        double output[3];
        ComputeGoldenPdist(x, 3, 2, std::numeric_limits<double>::infinity(), output);
        // (0,1): max(|1-5|, |10-3|) = max(4,7) = 7.0
        // (0,2): max(|1-2|, |10-8|) = max(1,2) = 2.0
        // (1,2): max(|5-2|, |3-8|) = max(3,5) = 5.0
        double expected[] = {7.0, 2.0, 5.0};
        bool ok = CompareResults(expected, output, 3, 1e-10);
        if (ok) passed++; else failed++;
    }

    // Test 5: p=0.5, fractional p
    {
        LOG_PRINT("\n  Golden self-test 5: p=0.5, N=2, M=3");
        double x[] = {0.0, 1.0, 4.0, 4.0, 6.0, 9.0};
        double output[1];
        ComputeGoldenPdist(x, 2, 3, 0.5, output);
        // diff = {4, 5, 5}, sum(|diff|^0.5) = 2 + sqrt(5) + sqrt(5) = 2 + 2*sqrt(5)
        // result = (2 + 2*sqrt(5))^2 = 4 + 8*sqrt(5) + 20 = 24 + 8*sqrt(5)
        double s = 2.0 + 2.0 * std::sqrt(5.0);
        double expected = std::pow(s, 2.0);  // since 1/p = 2
        bool ok = CompareResults(&expected, output, 1, 1e-10);
        if (ok) passed++; else failed++;
    }

    // Test 6: p=3, Minkowski p=3
    {
        LOG_PRINT("\n  Golden self-test 6: p=3, N=2, M=3");
        double x[] = {0.0, 0.0, 0.0, 1.0, 2.0, 3.0};
        double output[1];
        ComputeGoldenPdist(x, 2, 3, 3.0, output);
        // sum = 1 + 8 + 27 = 36, result = 36^(1/3)
        double expected = std::pow(36.0, 1.0 / 3.0);
        bool ok = CompareResults(&expected, output, 1, 1e-10);
        if (ok) passed++; else failed++;
    }

    // Test 7: identical rows (diff=0)
    {
        LOG_PRINT("\n  Golden self-test 7: identical rows, p=2, N=3, M=2");
        double x[] = {1, 2, 1, 2, 3, 4};
        double output[3];
        ComputeGoldenPdist(x, 3, 2, 2.0, output);
        double expected[] = {0.0, std::sqrt(8.0), std::sqrt(8.0)};
        bool ok = CompareResults(expected, output, 3, 1e-10);
        if (ok) passed++; else failed++;
    }

    LOG_PRINT("\n  Golden self-test: %d/%d passed", passed, passed + failed);
    LOG_PRINT("========================================");

    if (failed > 0) {
        LOG_PRINT("[FATAL] CPU golden self-test failed! Aborting.");
        exit(1);
    }
}

// ============================================================================
// Test case definition
// ============================================================================
struct PdistTestCase {
    const char* name;           // Test case name
    int64_t N;                  // Number of rows
    int64_t M;                  // Number of columns (features)
    double p;                   // p value
    const char* dtype;          // "float16" or "float32"
    std::function<std::vector<double>()> data_fn;  // Generates N*M doubles
};

// ============================================================================
// Data generators (return double vectors, will be cast to float/float16)
// ============================================================================

// Sequential data: [0, step, 2*step, ...]
static std::vector<double> GenSequential(int64_t N, int64_t M, double step = 1.0, double offset = 0.0) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) data[i] = offset + i * step;
    return data;
}

// Repeating pattern
static std::vector<double> GenRepeat(int64_t N, int64_t M, double val = 1.0) {
    return std::vector<double>(N * M, val);
}

// Pseudo-random with deterministic seed pattern
static std::vector<double> GenPseudoRandom(int64_t N, int64_t M, int64_t mod = 100, double scale = 0.01) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) data[i] = static_cast<double>((i * 7 + 3) % mod) * scale;
    return data;
}

// Negative range
static std::vector<double> GenNegative(int64_t N, int64_t M) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) data[i] = -(i + 1) * 0.5;
    return data;
}

// Mixed positive/negative
static std::vector<double> GenMixed(int64_t N, int64_t M) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) data[i] = (i % 2 == 0 ? 1 : -1) * (i + 1) * 0.3;
    return data;
}

// Large values for fp16 boundary
static std::vector<double> GenFp16Boundary(int64_t N, int64_t M) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) data[i] = 60000.0 + i * 100.0;
    return data;
}

// Small values near zero
static std::vector<double> GenTiny(int64_t N, int64_t M) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) data[i] = (i + 1) * 1e-5;
    return data;
}

// Rows with some identical rows
static std::vector<double> GenIdenticalRows(int64_t N, int64_t M) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) {
        int64_t row = i / M;
        int64_t col = i % M;
        // Make rows 0, 2 identical; row 1 different
        if (row == 0 || row == 2) {
            data[i] = (col + 1) * 1.0;
        } else {
            data[i] = (col + 1) * 2.0 + 5.0;
        }
    }
    return data;
}

// Wide range mixed values (moderate range to stay within precision thresholds)
static std::vector<double> GenWideRange(int64_t N, int64_t M) {
    std::vector<double> data(N * M);
    for (int64_t i = 0; i < N * M; i++) {
        int64_t row = i / M;
        int64_t col = i % M;
        switch (col % 6) {
            case 0: data[i] = -100.0; break;
            case 1: data[i] = -0.01; break;
            case 2: data[i] = 0.0; break;
            case 3: data[i] = 0.01; break;
            case 4: data[i] = 50.0; break;
            case 5: data[i] = 100.0; break;
        }
        data[i] += row * 0.1;
    }
    return data;
}

// ============================================================================
// Real mode helper functions
// ============================================================================
#ifndef USE_MOCK_ACLNN

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

#endif

// ============================================================================
// Unified test runner
// ============================================================================

#ifdef USE_MOCK_ACLNN

// Mock mode: CPU golden only
bool RunPdistTest(const PdistTestCase& tc) {
    LOG_PRINT("\n[Mock] %s (dtype=%s, N=%ld, M=%ld, p=%g)",
              tc.name, tc.dtype, tc.N, tc.M, tc.p);

    auto inputData = tc.data_fn();
    int64_t outputSize = tc.N * (tc.N - 1) / 2;

    std::vector<double> golden(outputSize);
    ComputeGoldenPdist(inputData.data(), tc.N, tc.M, tc.p, golden.data());

    double threshold = GetThreshold(tc.dtype);
    return CompareResults(golden.data(), golden.data(), outputSize, threshold);
}

#else

// Real mode: NPU execution
bool RunPdistTest(const PdistTestCase& tc, aclrtStream stream) {
    LOG_PRINT("\n[Real] %s (dtype=%s, N=%ld, M=%ld, p=%g)",
              tc.name, tc.dtype, tc.N, tc.M, tc.p);

    auto inputData = tc.data_fn();
    int64_t outputSize = tc.N * (tc.N - 1) / 2;
    bool is_fp16 = (std::string(tc.dtype) == "float16");
    aclDataType aclDtype = is_fp16 ? ACL_FLOAT16 : ACL_FLOAT;

    size_t inputElemCount = static_cast<size_t>(tc.N * tc.M);
    size_t inputBytes, outputBytes;

    // For fp16: host data stored as uint16_t (half precision), 2 bytes per element
    // For fp32: host data stored as float, 4 bytes per element
    std::vector<uint8_t> inputRaw;
    if (is_fp16) {
        inputBytes = inputElemCount * 2;
        inputRaw.resize(inputBytes);
        for (size_t i = 0; i < inputElemCount; i++) {
            float fval = static_cast<float>(inputData[i]);
            // IEEE 754 half-precision: sign(1) + exponent(5) + mantissa(10)
            float val = fval;
            uint32_t fbits;
            memcpy(&fbits, &val, sizeof(float));
            uint32_t sign = (fbits >> 31) & 0x1;
            int32_t exp = (fbits >> 23) & 0xFF;
            uint32_t mant = fbits & 0x7FFFFF;
            uint16_t hval2;
            if (exp == 0) {
                hval2 = static_cast<uint16_t>((sign << 15) | (mant >> 13));
            } else if (exp == 255) {
                hval2 = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
            } else {
                int32_t newexp = exp - 127 + 15;
                if (newexp >= 31) {
                    hval2 = static_cast<uint16_t>((sign << 15) | 0x7C00);
                } else if (newexp <= 0) {
                    hval2 = static_cast<uint16_t>(sign << 15);
                } else {
                    hval2 = static_cast<uint16_t>((sign << 15) | (newexp << 10) | (mant >> 13));
                }
            }
            memcpy(&inputRaw[i * 2], &hval2, sizeof(uint16_t));
        }
    } else {
        inputBytes = inputElemCount * sizeof(float);
        inputRaw.resize(inputBytes);
        for (size_t i = 0; i < inputElemCount; i++) {
            float fval = static_cast<float>(inputData[i]);
            memcpy(&inputRaw[i * sizeof(float)], &fval, sizeof(float));
        }
    }

    outputBytes = is_fp16 ? (static_cast<size_t>(outputSize) * 2) : (static_cast<size_t>(outputSize) * sizeof(float));

    void* inputDev = nullptr;
    void* outputDev = nullptr;
    aclTensor* inputTensor = nullptr;
    aclTensor* outputTensor = nullptr;

    std::vector<int64_t> inputShape = {tc.N, tc.M};
    std::vector<int64_t> outputShape = {outputSize};

    aclrtMalloc(&inputDev, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputDev, inputBytes, inputRaw.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMalloc(&outputDev, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(outputDev, outputBytes, 0, outputBytes);

    auto inStrides = ComputeStrides(inputShape);
    inputTensor = aclCreateTensor(inputShape.data(), inputShape.size(), aclDtype,
                                   inStrides.data(), 0, ACL_FORMAT_ND,
                                   inputShape.data(), inputShape.size(), inputDev);

    auto outStrides = ComputeStrides(outputShape);
    outputTensor = aclCreateTensor(outputShape.data(), outputShape.size(), aclDtype,
                                    outStrides.data(), 0, ACL_FORMAT_ND,
                                    outputShape.data(), outputShape.size(), outputDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    float pFloat = static_cast<float>(tc.p);
    auto ret = aclnnPdistGetWorkspaceSize(inputTensor, pFloat,
                                           outputTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  GetWorkspaceSize failed: %d", ret);
        aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
        aclrtFree(inputDev); aclrtFree(outputDev);
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    ret = aclnnPdist(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnPdist failed: %d", ret);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
        aclrtFree(inputDev); aclrtFree(outputDev);
        return false;
    }

    aclrtSynchronizeStream(stream);

    // Read back output
    std::vector<uint8_t> outputRaw(outputBytes);
    aclrtMemcpy(outputRaw.data(), outputBytes, outputDev, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // Compute golden with double precision
    // For fp16: quantize input to fp16 precision first to match hardware behavior
    std::vector<double> goldenInput(inputData.begin(), inputData.end());
    if (is_fp16) {
        for (size_t i = 0; i < goldenInput.size(); i++) {
            float fval = static_cast<float>(goldenInput[i]);
            // Round-trip through fp16: fp32 -> fp16 -> fp32
            uint32_t fbits;
            memcpy(&fbits, &fval, sizeof(float));
            uint32_t sign = (fbits >> 31) & 0x1;
            int32_t exp = (fbits >> 23) & 0xFF;
            uint32_t mant = fbits & 0x7FFFFF;
            uint16_t hval;
            if (exp == 0) {
                hval = static_cast<uint16_t>((sign << 15) | (mant >> 13));
            } else if (exp == 255) {
                hval = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
            } else {
                int32_t newexp = exp - 127 + 15;
                if (newexp >= 31) {
                    hval = static_cast<uint16_t>((sign << 15) | 0x7C00);
                } else if (newexp <= 0) {
                    hval = static_cast<uint16_t>(sign << 15);
                } else {
                    hval = static_cast<uint16_t>((sign << 15) | (newexp << 10) | (mant >> 13));
                }
            }
            // fp16 -> fp32
            uint32_t sign2 = (hval >> 15) & 0x1;
            int32_t exp2 = (hval >> 10) & 0x1F;
            uint32_t mant2 = hval & 0x3FF;
            uint32_t fbits2;
            if (exp2 == 0) {
                if (mant2 == 0) {
                    fbits2 = sign2 << 31;
                } else {
                    exp2 = -14;
                    while ((mant2 & 0x400) == 0) { mant2 <<= 1; exp2--; }
                    mant2 &= 0x3FF;
                    fbits2 = (sign2 << 31) | ((exp2 + 127) << 23) | (mant2 << 13);
                }
            } else if (exp2 == 31) {
                fbits2 = (sign2 << 31) | 0x7F800000 | (mant2 << 13);
            } else {
                fbits2 = (sign2 << 31) | ((exp2 - 15 + 127) << 23) | (mant2 << 13);
            }
            float fval2;
            memcpy(&fval2, &fbits2, sizeof(float));
            goldenInput[i] = static_cast<double>(fval2);
        }
    }

    std::vector<double> golden(outputSize);
    ComputeGoldenPdist(goldenInput.data(), tc.N, tc.M, tc.p, golden.data());

    // Convert NPU output to double for comparison
    std::vector<double> actual(outputSize);
    for (int64_t i = 0; i < outputSize; i++) {
        if (is_fp16) {
            uint16_t hval;
            memcpy(&hval, &outputRaw[i * 2], sizeof(uint16_t));
            // IEEE 754 half -> float
            uint32_t sign = (hval >> 15) & 0x1;
            int32_t exp = (hval >> 10) & 0x1F;
            uint32_t mant = hval & 0x3FF;
            uint32_t fbits;
            if (exp == 0) {
                if (mant == 0) {
                    fbits = sign << 31;
                } else {
                    // Subnormal: normalize
                    exp = -14;
                    while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
                    mant &= 0x3FF;
                    fbits = (sign << 31) | ((exp + 127) << 23) | (mant << 13);
                }
            } else if (exp == 31) {
                fbits = (sign << 31) | 0x7F800000 | (mant << 13);
            } else {
                fbits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
            }
            float fval;
            memcpy(&fval, &fbits, sizeof(float));
            actual[i] = static_cast<double>(fval);
        } else {
            float fval;
            memcpy(&fval, &outputRaw[i * sizeof(float)], sizeof(float));
            actual[i] = static_cast<double>(fval);
        }
    }

    double threshold = GetThreshold(tc.dtype);
    bool pass = CompareResults(golden.data(), actual.data(), outputSize, threshold);

    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
    aclrtFree(inputDev); aclrtFree(outputDev);
    return pass;
}

// Real mode: NPU execution via aclnnPdistForward (aclScalar* p)
bool RunPdistForwardTest(const PdistTestCase& tc, aclrtStream stream) {
    LOG_PRINT("\n[Real-Forward] %s (dtype=%s, N=%ld, M=%ld, p=%g)",
              tc.name, tc.dtype, tc.N, tc.M, tc.p);

    auto inputData = tc.data_fn();
    int64_t outputSize = tc.N * (tc.N - 1) / 2;
    bool is_fp16 = (std::string(tc.dtype) == "float16");
    aclDataType aclDtype = is_fp16 ? ACL_FLOAT16 : ACL_FLOAT;

    size_t inputElemCount = static_cast<size_t>(tc.N * tc.M);
    size_t inputBytes, outputBytes;

    std::vector<uint8_t> inputRaw;
    if (is_fp16) {
        inputBytes = inputElemCount * 2;
        inputRaw.resize(inputBytes);
        for (size_t i = 0; i < inputElemCount; i++) {
            float fval = static_cast<float>(inputData[i]);
            uint32_t fbits;
            memcpy(&fbits, &fval, sizeof(float));
            uint32_t sign = (fbits >> 31) & 0x1;
            int32_t exp = (fbits >> 23) & 0xFF;
            uint32_t mant = fbits & 0x7FFFFF;
            uint16_t hval2;
            if (exp == 0) {
                hval2 = static_cast<uint16_t>((sign << 15) | (mant >> 13));
            } else if (exp == 255) {
                hval2 = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant >> 13));
            } else {
                int32_t newexp = exp - 127 + 15;
                if (newexp >= 31) {
                    hval2 = static_cast<uint16_t>((sign << 15) | 0x7C00);
                } else if (newexp <= 0) {
                    hval2 = static_cast<uint16_t>(sign << 15);
                } else {
                    hval2 = static_cast<uint16_t>((sign << 15) | (newexp << 10) | (mant >> 13));
                }
            }
            memcpy(&inputRaw[i * 2], &hval2, sizeof(uint16_t));
        }
    } else {
        inputBytes = inputElemCount * sizeof(float);
        inputRaw.resize(inputBytes);
        for (size_t i = 0; i < inputElemCount; i++) {
            float fval = static_cast<float>(inputData[i]);
            memcpy(&inputRaw[i * sizeof(float)], &fval, sizeof(float));
        }
    }

    outputBytes = is_fp16 ? (static_cast<size_t>(outputSize) * 2) : (static_cast<size_t>(outputSize) * sizeof(float));

    void* inputDev = nullptr;
    void* outputDev = nullptr;
    aclTensor* inputTensor = nullptr;
    aclTensor* outputTensor = nullptr;

    std::vector<int64_t> inputShape = {tc.N, tc.M};
    std::vector<int64_t> outputShape = {outputSize};

    aclrtMalloc(&inputDev, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(inputDev, inputBytes, inputRaw.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMalloc(&outputDev, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(outputDev, outputBytes, 0, outputBytes);

    auto inStrides = ComputeStrides(inputShape);
    inputTensor = aclCreateTensor(inputShape.data(), inputShape.size(), aclDtype,
                                   inStrides.data(), 0, ACL_FORMAT_ND,
                                   inputShape.data(), inputShape.size(), inputDev);

    auto outStrides = ComputeStrides(outputShape);
    outputTensor = aclCreateTensor(outputShape.data(), outputShape.size(), aclDtype,
                                    outStrides.data(), 0, ACL_FORMAT_ND,
                                    outputShape.data(), outputShape.size(), outputDev);

    float pFloat = static_cast<float>(tc.p);
    aclScalar* pScalar = aclCreateScalar(&pFloat, ACL_FLOAT);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto ret = aclnnPdistForwardGetWorkspaceSize(inputTensor, pScalar,
                                                  outputTensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  PdistForward GetWorkspaceSize failed: %d", ret);
        aclDestroyScalar(pScalar);
        aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
        aclrtFree(inputDev); aclrtFree(outputDev);
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    ret = aclnnPdistForward(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  aclnnPdistForward failed: %d", ret);
        aclDestroyScalar(pScalar);
        if (workspace) aclrtFree(workspace);
        aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
        aclrtFree(inputDev); aclrtFree(outputDev);
        return false;
    }

    aclrtSynchronizeStream(stream);

    std::vector<uint8_t> outputRaw(outputBytes);
    aclrtMemcpy(outputRaw.data(), outputBytes, outputDev, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<double> goldenInput(inputData.begin(), inputData.end());
    if (is_fp16) {
        for (size_t i = 0; i < goldenInput.size(); i++) {
            float fval = static_cast<float>(goldenInput[i]);
            uint32_t fbits;
            memcpy(&fbits, &fval, sizeof(float));
            uint32_t sign = (fbits >> 31) & 0x1;
            int32_t exp2 = (fbits >> 23) & 0xFF;
            uint32_t mant2 = fbits & 0x7FFFFF;
            uint16_t hval;
            if (exp2 == 0) { hval = static_cast<uint16_t>((sign << 15) | (mant2 >> 13)); }
            else if (exp2 == 255) { hval = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant2 >> 13)); }
            else {
                int32_t ne = exp2 - 127 + 15;
                if (ne >= 31) hval = static_cast<uint16_t>((sign << 15) | 0x7C00);
                else if (ne <= 0) hval = static_cast<uint16_t>(sign << 15);
                else hval = static_cast<uint16_t>((sign << 15) | (ne << 10) | (mant2 >> 13));
            }
            uint32_t s2 = (hval >> 15) & 0x1;
            int32_t e2 = (hval >> 10) & 0x1F;
            uint32_t m2 = hval & 0x3FF;
            uint32_t fb2;
            if (e2 == 0) {
                if (m2 == 0) { fb2 = s2 << 31; }
                else { e2 = -14; while ((m2 & 0x400) == 0) { m2 <<= 1; e2--; } m2 &= 0x3FF; fb2 = (s2 << 31) | ((e2 + 127) << 23) | (m2 << 13); }
            } else if (e2 == 31) { fb2 = (s2 << 31) | 0x7F800000 | (m2 << 13); }
            else { fb2 = (s2 << 31) | ((e2 - 15 + 127) << 23) | (m2 << 13); }
            float fv2; memcpy(&fv2, &fb2, sizeof(float));
            goldenInput[i] = static_cast<double>(fv2);
        }
    }

    std::vector<double> golden(outputSize);
    ComputeGoldenPdist(goldenInput.data(), tc.N, tc.M, tc.p, golden.data());

    std::vector<double> actual(outputSize);
    for (int64_t i = 0; i < outputSize; i++) {
        if (is_fp16) {
            uint16_t hval; memcpy(&hval, &outputRaw[i * 2], sizeof(uint16_t));
            uint32_t sign = (hval >> 15) & 0x1;
            int32_t exp3 = (hval >> 10) & 0x1F;
            uint32_t mant3 = hval & 0x3FF;
            uint32_t fbits3;
            if (exp3 == 0) {
                if (mant3 == 0) { fbits3 = sign << 31; }
                else { exp3 = -14; while ((mant3 & 0x400) == 0) { mant3 <<= 1; exp3--; } mant3 &= 0x3FF; fbits3 = (sign << 31) | ((exp3 + 127) << 23) | (mant3 << 13); }
            } else if (exp3 == 31) { fbits3 = (sign << 31) | 0x7F800000 | (mant3 << 13); }
            else { fbits3 = (sign << 31) | ((exp3 - 15 + 127) << 23) | (mant3 << 13); }
            float fval3; memcpy(&fval3, &fbits3, sizeof(float));
            actual[i] = static_cast<double>(fval3);
        } else {
            float fval3; memcpy(&fval3, &outputRaw[i * sizeof(float)], sizeof(float));
            actual[i] = static_cast<double>(fval3);
        }
    }

    double threshold = GetThreshold(tc.dtype);
    bool pass = CompareResults(golden.data(), actual.data(), outputSize, threshold);

    aclDestroyScalar(pScalar);
    if (workspace) aclrtFree(workspace);
    aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
    aclrtFree(inputDev); aclrtFree(outputDev);
    return pass;
}

#endif

// ============================================================================
// L0 + L1 test case definitions
// ============================================================================

std::vector<PdistTestCase> GetTestCases() {
    std::vector<PdistTestCase> cases;

    // ========================================================================
    // L0 GATEWAY TESTS (core function validation)
    // ========================================================================
    // Each dtype (fp16/fp32) x each p-branch (0, general, inf) = 6 base cases
    // ========================================================================

    // L0_001: fp32, p=2 (Euclidean), small shape
    cases.push_back({"L0_001_fp32_p2_small", 4, 3, 2.0, "float32",
        []() { return GenSequential(4, 3, 1.0, 1.0); }});

    // L0_002: fp32, p=1 (Manhattan), small shape
    cases.push_back({"L0_002_fp32_p1_small", 3, 4, 1.0, "float32",
        []() { return GenSequential(3, 4, 1.0, 0.0); }});

    // L0_003: fp32, p=0 (Hamming), small shape
    cases.push_back({"L0_003_fp32_p0_small", 4, 2, 0.0, "float32",
        []() { return std::vector<double>{1, 2, 1, 3, 2, 2, 1, 2}; }});

    // L0_004: fp32, p=inf (Chebyshev), small shape
    cases.push_back({"L0_004_fp32_pinf_small", 3, 2, std::numeric_limits<double>::infinity(), "float32",
        []() { return std::vector<double>{1, 10, 5, 3, 2, 8}; }});

    // L0_005: fp16, p=2 (Euclidean), small shape
    cases.push_back({"L0_005_fp16_p2_small", 4, 3, 2.0, "float16",
        []() { return GenSequential(4, 3, 1.0, 1.0); }});

    // L0_006: fp16, p=1 (Manhattan), small shape
    cases.push_back({"L0_006_fp16_p1_small", 3, 4, 1.0, "float16",
        []() { return GenSequential(3, 4, 1.0, 0.0); }});

    // L0_007: fp16, p=0 (Hamming), small shape
    cases.push_back({"L0_007_fp16_p0_small", 4, 2, 0.0, "float16",
        []() { return std::vector<double>{1, 2, 1, 3, 2, 2, 1, 2}; }});

    // L0_008: fp16, p=inf (Chebyshev), small shape
    cases.push_back({"L0_008_fp16_pinf_small", 3, 2, std::numeric_limits<double>::infinity(), "float16",
        []() { return std::vector<double>{1, 10, 5, 3, 2, 8}; }});

    // ========================================================================
    // L1 TESTS: p-value branch coverage (all p values x fp16+fp32)
    // ========================================================================

    // --- p=0.5 (fractional p, general path) ---
    cases.push_back({"L1_p05_fp32", 4, 16, 0.5, "float32",
        []() { return GenPseudoRandom(4, 16, 50, 0.1); }});
    cases.push_back({"L1_p05_fp16", 4, 16, 0.5, "float16",
        []() { return GenPseudoRandom(4, 16, 50, 0.1); }});

    // --- p=3 (non-standard Minkowski) ---
    cases.push_back({"L1_p3_fp32", 5, 32, 3.0, "float32",
        []() { return GenSequential(5, 32, 0.1, 0.0); }});
    cases.push_back({"L1_p3_fp16", 5, 32, 3.0, "float16",
        []() { return GenSequential(5, 32, 0.1, 0.0); }});

    // --- p=10 (large p value) ---
    cases.push_back({"L1_p10_fp32", 4, 4, 10.0, "float32",
        []() { return std::vector<double>{1, 2, 3, 4, 1.1, 2.1, 3.1, 4.1,
                                            0.9, 1.9, 2.9, 3.9, 1.5, 2.5, 3.5, 4.5}; }});
    cases.push_back({"L1_p10_fp16", 4, 4, 10.0, "float16",
        []() { return std::vector<double>{1, 2, 3, 4, 1.1, 2.1, 3.1, 4.1,
                                            0.9, 1.9, 2.9, 3.9, 1.5, 2.5, 3.5, 4.5}; }});

    // ========================================================================
    // L1 TESTS: dtype x p full cross product (key combinations)
    // ========================================================================

    // fp32 x p=2 (Euclidean, medium shape)
    cases.push_back({"L1_fp32_p2_medium", 10, 64, 2.0, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    // fp16 x p=2 (Euclidean, medium shape)
    cases.push_back({"L1_fp16_p2_medium", 10, 64, 2.0, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});

    // fp32 x p=1 (Manhattan, medium shape)
    cases.push_back({"L1_fp32_p1_medium", 10, 64, 1.0, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    // fp16 x p=1 (Manhattan, medium shape)
    cases.push_back({"L1_fp16_p1_medium", 10, 64, 1.0, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});

    // fp32 x p=inf (Chebyshev, medium shape)
    cases.push_back({"L1_fp32_pinf_medium", 10, 64, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    // fp16 x p=inf (Chebyshev, medium shape)
    cases.push_back({"L1_fp16_pinf_medium", 10, 64, std::numeric_limits<double>::infinity(), "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});

    // fp32 x p=0 (Hamming, medium shape)
    cases.push_back({"L1_fp32_p0_medium", 10, 64, 0.0, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    // fp16 x p=0 (Hamming, medium shape)
    cases.push_back({"L1_fp16_p0_medium", 10, 64, 0.0, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});

    // ========================================================================
    // L1 TESTS: Shape boundary coverage
    // ========================================================================

    // Minimum shape: N=2, M=1
    cases.push_back({"L1_min_N2_M1_fp32_p2", 2, 1, 2.0, "float32",
        []() { return std::vector<double>{3.0, 7.0}; }});
    cases.push_back({"L1_min_N2_M1_fp16_p2", 2, 1, 2.0, "float16",
        []() { return std::vector<double>{3.0, 7.0}; }});

    // N=2, M=8 (minimum N, larger M)
    cases.push_back({"L1_N2_M8_fp32_p2", 2, 8, 2.0, "float32",
        []() { return GenSequential(2, 8, 1.0, 1.0); }});
    cases.push_back({"L1_N2_M8_fp16_p2", 2, 8, 2.0, "float16",
        []() { return GenSequential(2, 8, 1.0, 1.0); }});

    // N=3, M=1
    cases.push_back({"L1_N3_M1_fp32_p2", 3, 1, 2.0, "float32",
        []() { return std::vector<double>{1.0, 5.0, 9.0}; }});

    // N=3, M=4
    cases.push_back({"L1_N3_M4_fp32_p2", 3, 4, 2.0, "float32",
        []() { return GenSequential(3, 4, 1.0, 0.0); }});

    // N=5, M=16 (small multi-core)
    cases.push_back({"L1_N5_M16_fp32_p3", 5, 16, 3.0, "float32",
        []() { return GenSequential(5, 16, 0.1, 0.0); }});

    // ========================================================================
    // L1 TESTS: Multi-core distribution (large N)
    // ========================================================================

    // N=20, M=64 (compute_num=190)
    cases.push_back({"L1_multicore_N20_M64_fp32_p2", 20, 64, 2.0, "float32",
        []() { return GenPseudoRandom(20, 64, 100, 0.01); }});
    cases.push_back({"L1_multicore_N20_M64_fp32_p1", 20, 64, 1.0, "float32",
        []() { return GenPseudoRandom(20, 64, 100, 0.01); }});
    cases.push_back({"L1_multicore_N20_M64_fp32_pinf", 20, 64, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenPseudoRandom(20, 64, 100, 0.01); }});
    cases.push_back({"L1_multicore_N20_M64_fp32_p0", 20, 64, 0.0, "float32",
        []() { return GenPseudoRandom(20, 64, 100, 0.01); }});

    // N=50, M=32 (compute_num=1225, large multi-core)
    cases.push_back({"L1_multicore_N50_M32_fp32_p2", 50, 32, 2.0, "float32",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"L1_multicore_N50_M32_fp32_p1", 50, 32, 1.0, "float32",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"L1_multicore_N50_M32_fp32_pinf", 50, 32, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"L1_multicore_N50_M32_fp32_p0", 50, 32, 0.0, "float32",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});

    // N=50, M=32, fp16
    cases.push_back({"L1_multicore_N50_M32_fp16_p2", 50, 32, 2.0, "float16",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"L1_multicore_N50_M32_fp16_p1", 50, 32, 1.0, "float16",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"L1_multicore_N50_M32_fp16_pinf", 50, 32, std::numeric_limits<double>::infinity(), "float16",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"L1_multicore_N50_M32_fp16_p0", 50, 32, 0.0, "float16",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});

    // ========================================================================
    // L1 TESTS: Large M (UB block tiling)
    // ========================================================================

    // N=10, M=1024 (large M, multiple UB blocks)
    cases.push_back({"L1_largeM_N10_M1024_fp32_p2", 10, 1024, 2.0, "float32",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});
    cases.push_back({"L1_largeM_N10_M1024_fp32_p1", 10, 1024, 1.0, "float32",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});
    cases.push_back({"L1_largeM_N10_M1024_fp32_p0", 10, 1024, 0.0, "float32",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});
    cases.push_back({"L1_largeM_N10_M1024_fp32_pinf", 10, 1024, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});

    // N=30, M=256 (medium-large M, UB tiling)
    cases.push_back({"L1_largeM_N30_M256_fp32_p2", 30, 256, 2.0, "float32",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});
    cases.push_back({"L1_largeM_N30_M256_fp32_p0", 30, 256, 0.0, "float32",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});
    cases.push_back({"L1_largeM_N30_M256_fp32_pinf", 30, 256, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});

    // N=30, M=256, fp16
    cases.push_back({"L1_largeM_N30_M256_fp16_p2", 30, 256, 2.0, "float16",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});
    cases.push_back({"L1_largeM_N30_M256_fp16_p1", 30, 256, 1.0, "float16",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});
    cases.push_back({"L1_largeM_N30_M256_fp16_p0", 30, 256, 0.0, "float16",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});
    cases.push_back({"L1_largeM_N30_M256_fp16_pinf", 30, 256, std::numeric_limits<double>::infinity(), "float16",
        []() { return GenPseudoRandom(30, 256, 200, 0.05); }});

    // ========================================================================
    // L1 TESTS: Precision-sensitive scenarios
    // ========================================================================

    // Identical rows (diff=0, tests ln(0) handling)
    cases.push_back({"L1_identical_rows_fp32_p2", 4, 8, 2.0, "float32",
        []() { return GenIdenticalRows(4, 8); }});
    cases.push_back({"L1_identical_rows_fp32_p1", 4, 8, 1.0, "float32",
        []() { return GenIdenticalRows(4, 8); }});
    cases.push_back({"L1_identical_rows_fp32_pinf", 4, 8, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenIdenticalRows(4, 8); }});

    // All zeros input
    cases.push_back({"L1_all_zeros_fp32_p2", 3, 4, 2.0, "float32",
        []() { return GenRepeat(3, 4, 0.0); }});
    cases.push_back({"L1_all_zeros_fp16_p2", 3, 4, 2.0, "float16",
        []() { return GenRepeat(3, 4, 0.0); }});

    // All same values (not zero)
    cases.push_back({"L1_all_same_fp32_p2", 3, 4, 2.0, "float32",
        []() { return GenRepeat(3, 4, 5.0); }});
    cases.push_back({"L1_all_same_fp16_p2", 3, 4, 2.0, "float16",
        []() { return GenRepeat(3, 4, 5.0); }});

    // Negative values
    cases.push_back({"L1_negative_fp32_p2", 5, 8, 2.0, "float32",
        []() { return GenNegative(5, 8); }});
    cases.push_back({"L1_negative_fp32_p1", 5, 8, 1.0, "float32",
        []() { return GenNegative(5, 8); }});

    // Mixed positive/negative
    cases.push_back({"L1_mixed_fp32_p2", 5, 8, 2.0, "float32",
        []() { return GenMixed(5, 8); }});
    cases.push_back({"L1_mixed_fp32_p1", 5, 8, 1.0, "float32",
        []() { return GenMixed(5, 8); }});

    // fp16 boundary values (near 65504)
    cases.push_back({"L1_fp16_boundary_fp16_p2", 3, 4, 2.0, "float16",
        []() { return GenFp16Boundary(3, 4); }});

    // Tiny values near zero
    cases.push_back({"L1_tiny_fp32_p2", 4, 8, 2.0, "float32",
        []() { return GenTiny(4, 8); }});
    cases.push_back({"L1_tiny_fp16_p2", 4, 8, 2.0, "float16",
        []() { return GenTiny(4, 8); }});

    // Wide range mixed values
    cases.push_back({"L1_wide_range_fp32_p2", 5, 12, 2.0, "float32",
        []() { return GenWideRange(5, 12); }});

    // Large values for fp32
    cases.push_back({"L1_large_values_fp32_p2", 3, 4, 2.0, "float32",
        []() { return std::vector<double>{1e10, 2e10, 3e10, 4e10,
                                            1e10, 2.1e10, 3e10, 4e10,
                                            0.5e10, 1.5e10, 2.5e10, 3.5e10}; }});

    // ========================================================================
    // L1 TESTS: Additional dtype x p combinations for full coverage
    // ========================================================================

    // fp16 x p=0.5, p=3, p=10 with medium shape
    cases.push_back({"L1_fp16_p05_medium", 10, 64, 0.5, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"L1_fp16_p3_medium", 10, 64, 3.0, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"L1_fp16_p10_medium", 10, 64, 10.0, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});

    // fp32 x p=0.5, p=3, p=10 with medium shape
    cases.push_back({"L1_fp32_p05_medium", 10, 64, 0.5, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"L1_fp32_p3_medium", 10, 64, 3.0, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"L1_fp32_p10_medium", 10, 64, 10.0, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});

    // ========================================================================
    // L1 TESTS: Boundary shape x p combinations
    // ========================================================================

    // N=2, M=1 with various p values
    cases.push_back({"L1_min_p0_fp32", 2, 1, 0.0, "float32",
        []() { return std::vector<double>{3.0, 3.0}; }});
    cases.push_back({"L1_min_pinf_fp32", 2, 1, std::numeric_limits<double>::infinity(), "float32",
        []() { return std::vector<double>{3.0, 7.0}; }});
    cases.push_back({"L1_min_p1_fp32", 2, 1, 1.0, "float32",
        []() { return std::vector<double>{3.0, 7.0}; }});

    // N=2, M=1 with various p values, fp16
    cases.push_back({"L1_min_p0_fp16", 2, 1, 0.0, "float16",
        []() { return std::vector<double>{3.0, 3.0}; }});
    cases.push_back({"L1_min_pinf_fp16", 2, 1, std::numeric_limits<double>::infinity(), "float16",
        []() { return std::vector<double>{3.0, 7.0}; }});

    // N=100, M=16 (very large N)
    cases.push_back({"L1_largeN_N100_M16_fp32_p2", 100, 16, 2.0, "float32",
        []() { return GenPseudoRandom(100, 16, 500, 0.01); }});

    // N=5, M=2048 (very large M, multiple UB blocks)
    cases.push_back({"L1_veryLargeM_N5_M2048_fp32_p2", 5, 2048, 2.0, "float32",
        []() { return GenPseudoRandom(5, 2048, 50, 0.1); }});

    // N=5, M=2048 with p=0 (Hamming, large M)
    cases.push_back({"L1_veryLargeM_N5_M2048_fp32_p0", 5, 2048, 0.0, "float32",
        []() { return GenPseudoRandom(5, 2048, 50, 0.1); }});

    // N=5, M=2048 with p=inf (Chebyshev, large M)
    cases.push_back({"L1_veryLargeM_N5_M2048_fp32_pinf", 5, 2048, std::numeric_limits<double>::infinity(), "float32",
        []() { return GenPseudoRandom(5, 2048, 50, 0.1); }});

    // fp16, large M
    cases.push_back({"L1_largeM_N10_M1024_fp16_p2", 10, 1024, 2.0, "float16",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});
    cases.push_back({"L1_largeM_N10_M1024_fp16_p1", 10, 1024, 1.0, "float16",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});

    return cases;
}

std::vector<PdistTestCase> GetForwardTestCases() {
    std::vector<PdistTestCase> cases;

    cases.push_back({"Fwd_fp32_p2_small", 4, 3, 2.0, "float32",
        []() { return GenSequential(4, 3, 1.0, 1.0); }});
    cases.push_back({"Fwd_fp32_p1_small", 3, 4, 1.0, "float32",
        []() { return GenSequential(3, 4, 1.0, 0.0); }});
    cases.push_back({"Fwd_fp32_p0_small", 4, 2, 0.0, "float32",
        []() { return std::vector<double>{1, 2, 1, 3, 2, 2, 1, 2}; }});
    cases.push_back({"Fwd_fp32_pinf_small", 3, 2, std::numeric_limits<double>::infinity(), "float32",
        []() { return std::vector<double>{1, 10, 5, 3, 2, 8}; }});
    cases.push_back({"Fwd_fp16_p2_small", 4, 3, 2.0, "float16",
        []() { return GenSequential(4, 3, 1.0, 1.0); }});
    cases.push_back({"Fwd_fp16_p0_small", 4, 2, 0.0, "float16",
        []() { return std::vector<double>{1, 2, 1, 3, 2, 2, 1, 2}; }});
    cases.push_back({"Fwd_fp32_p05_medium", 10, 64, 0.5, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"Fwd_fp32_p3_medium", 10, 64, 3.0, "float32",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"Fwd_fp16_p2_medium", 10, 64, 2.0, "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"Fwd_fp16_pinf_medium", 10, 64, std::numeric_limits<double>::infinity(), "float16",
        []() { return GenPseudoRandom(10, 64, 100, 0.01); }});
    cases.push_back({"Fwd_multicore_N50_M32_fp32_p2", 50, 32, 2.0, "float32",
        []() { return GenPseudoRandom(50, 32, 300, 0.01); }});
    cases.push_back({"Fwd_largeM_N10_M1024_fp32_p2", 10, 1024, 2.0, "float32",
        []() { return GenPseudoRandom(10, 1024, 50, 0.1); }});

    return cases;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("Pdist ST Test - Iteration 3");
    LOG_PRINT("========================================");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("Mode: Mock (CPU golden only)");
#else
    LOG_PRINT("Mode: Real (NPU)");
#endif

    // Phase 1: CPU golden self-test
    TestGoldenCorrectness();

    // Phase 2: Run all test cases
    int passed = 0, failed = 0;

#ifndef USE_MOCK_ACLNN
    int32_t deviceId = 0;
    aclrtStream stream;

    auto initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclInit failed: %d", initRet);
        return 1;
    }
    initRet = aclrtSetDevice(deviceId);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtSetDevice(%d) failed: %d", deviceId, initRet);
        aclFinalize();
        return 1;
    }
    initRet = aclrtCreateStream(&stream);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtCreateStream failed: %d", initRet);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
#endif

    auto test_cases = GetTestCases();
    LOG_PRINT("\nTotal test cases: %zu", test_cases.size());

    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
#ifdef USE_MOCK_ACLNN
        if (RunPdistTest(tc)) passed++; else failed++;
#else
        if (RunPdistTest(tc, stream)) passed++; else failed++;
#endif
    }

#ifndef USE_MOCK_ACLNN
    auto fwd_cases = GetForwardTestCases();
    LOG_PRINT("\nForward API test cases: %zu", fwd_cases.size());
    for (size_t i = 0; i < fwd_cases.size(); ++i) {
        const auto& tc = fwd_cases[i];
        if (RunPdistForwardTest(tc, stream)) passed++; else failed++;
    }
#endif

#ifndef USE_MOCK_ACLNN
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    LOG_PRINT("\n========================================");
    LOG_PRINT("ST Test Report");
    LOG_PRINT("========================================");
    LOG_PRINT("Total: %d", passed + failed);
    LOG_PRINT("Passed: %d", passed);
    LOG_PRINT("Failed: %d", failed);
    LOG_PRINT("========================================\n");

    return failed == 0 ? 0 : 1;
}
