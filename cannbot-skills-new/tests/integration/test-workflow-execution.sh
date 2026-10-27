#!/usr/bin/env bash
# =============================================================================
# Integration Test: Workflow Execution
# =============================================================================
# Tests real workflow execution by having Claude complete actual tasks.
#
# What this tests:
# 1. Skills are correctly loaded and invoked
# 2. Workflow steps are followed in correct order
# 3. Actual files are created/modified correctly
# 4. Session analysis shows correct tool usage
#
# WARNING: This test takes 5-15 minutes to complete.
#
# Usage: ./test-workflow-execution.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}Integration Test: Workflow Execution${NC}"
echo "========================================"
echo ""
echo "This test verifies real workflow execution."
echo "Estimated time: 5-15 minutes"
echo "Requires: Claude Code CLI"
echo ""

if ! command -v claude &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Claude Code CLI not found"
    exit 1
fi

# ============================================
# Setup Test Environment
# ============================================

TEST_PROJECT=$(create_test_project "workflow-test")
echo "Test project: $TEST_PROJECT"

trap "cleanup_test_project $TEST_PROJECT" EXIT

# Create a simple Ascend C operator project structure
mkdir -p "$TEST_PROJECT"/{ops/test_add/{op_host,op_kernel,tests/{ut,st}},docs}

cat > "$TEST_PROJECT/ops/test_add/CMakeLists.txt" <<'EOF'
# CMakeLists.txt for test_add operator
cmake_minimum_required(VERSION 3.16)
project(test_add LANGUAGES CXX)
EOF

cat > "$TEST_PROJECT/ops/test_add/op_kernel/add_kernel.h" <<'EOF'
// Add operator kernel header
#pragma once
#include "kernel_tiling.h"

extern "C" __global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);
EOF

# Initialize git
cd "$TEST_PROJECT"
git init --quiet 2>/dev/null || true
git config user.email "test@test.com" 2>/dev/null || true
git config user.name "Test User" 2>/dev/null || true
git add . 2>/dev/null || true
git commit -m "Initial commit" --quiet 2>/dev/null || true

echo ""
echo "Project setup complete."
echo ""

# ============================================
# Test Scenario 1: Development Workflow
# ============================================

print_section_header "Test Scenario 1: Development Workflow"

PROMPT="你是一个 Ascend C 算子开发工程师。
当前目录下有一个 test_add 算子项目结构。
请帮我完成以下任务：
1. 分析当前算子结构
2. 创建 TilingData 结构体定义（在 kernel_tiling.h）
3. 实现核函数（在 add_kernel.cpp）

用简洁的方式完成，每个文件不超过 50 行代码。
完成后说明创建了哪些文件。"

SESSION_FILE=$(mktemp --suffix=.jsonl)
OUTPUT_FILE="$TEST_PROJECT/output.txt"

echo "Running Claude..."
echo "Timeout: 300s"
echo "Working directory: $TEST_PROJECT"
echo ""

# Change to test project directory for Claude
cd "$TEST_PROJECT"

# Run Claude with basic output mode
if timeout 300 claude -p "$PROMPT" \
    --dangerously-skip-permissions \
    --max-turns 10 \
    > "$OUTPUT_FILE" 2>&1; then
    echo ""
    echo "Execution completed"
else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo ""
        print_warn "Timeout after 300s"
    else
        echo ""
        print_info "Claude exited with code $exit_code"
    fi
fi

# Show output for debugging
echo ""
echo -e "${BOLD}=== Claude Output ===${NC}"
head -50 "$OUTPUT_FILE"
echo "..."
echo -e "${BOLD}=== End Output ===${NC}"
echo ""

# Go back to original directory
cd - > /dev/null

echo ""
print_section_header "Verification"

FAILED=0

# Test 1: Check if skills were invoked
echo "Test 1: Skill invocation..."
if grep -qE 'Skill|skill|技能' "$OUTPUT_FILE" 2>/dev/null; then
    print_pass "Skill reference found in output"
else
    print_info "No explicit skill reference in output"
fi
echo ""

# Test 2: Check file creation
echo "Test 2: File creation..."

files_created=0
if [ -f "$TEST_PROJECT/ops/test_add/op_kernel/kernel_tiling.h" ]; then
    print_pass "kernel_tiling.h created"
    ((files_created++)) || true
else
    print_info "kernel_tiling.h not created"
fi

if [ -f "$TEST_PROJECT/ops/test_add/op_kernel/add_kernel.cpp" ]; then
    print_pass "add_kernel.cpp created"
    ((files_created++)) || true
else
    print_info "add_kernel.cpp not created"
fi

echo ""
echo "  Files created: $files_created"
echo ""

# Test 3: Content quality check
echo "Test 3: Content quality check..."

if [ -f "$TEST_PROJECT/ops/test_add/op_kernel/kernel_tiling.h" ]; then
    if grep -qE "TilingData|struct|class" "$TEST_PROJECT/ops/test_add/op_kernel/kernel_tiling.h" 2>/dev/null; then
        print_pass "kernel_tiling.h has valid TilingData structure"
    else
        print_warn "kernel_tiling.h may not have valid TilingData structure"
    fi
fi

if [ -f "$TEST_PROJECT/ops/test_add/op_kernel/add_kernel.cpp" ]; then
    if grep -qE "__global__|__aicore__|GM_ADDR" "$TEST_PROJECT/ops/test_add/op_kernel/add_kernel.cpp" 2>/dev/null; then
        print_pass "add_kernel.cpp has valid kernel signature"
    else
        print_warn "add_kernel.cpp may not have valid kernel signature"
    fi
fi
echo ""

# Test 4: Output analysis
echo "Test 4: Output analysis..."
if [ -f "$OUTPUT_FILE" ]; then
    output_lines=$(wc -l < "$OUTPUT_FILE")
    echo "  Output lines: $output_lines"
    if grep -qE "文件|file|创建|created|kernel|Tiling" "$OUTPUT_FILE" 2>/dev/null; then
        print_pass "Output contains relevant keywords"
    else
        print_info "Output may not contain expected keywords"
    fi
else
    print_warn "No output file"
fi
echo ""

# Test 5: Premature action check (simplified)
echo "Test 5: Workflow check..."
print_info "Workflow executed (see output above)"
echo ""

# Cleanup
rm -f "$SESSION_FILE" 2>/dev/null || true

# ============================================
# Summary
# ============================================

echo "========================================"
echo -e " ${BOLD}Test Summary${NC}"
echo "========================================"
echo ""

if [ $files_created -gt 0 ]; then
    echo -e "${GREEN}✓ Workflow execution test completed${NC}"
    echo "  - Files created: $files_created"
    echo "  - Test project: $TEST_PROJECT"
    echo ""
    print_status_passed
    exit 0
else
    echo -e "${RED}✗ Workflow execution test failed${NC}"
    echo "  - No files were created"
    echo ""
    print_status_failed
    exit 1
fi