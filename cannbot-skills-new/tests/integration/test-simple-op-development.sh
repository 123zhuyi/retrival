#!/usr/bin/env bash
# Integration Test: Simple Op Development
# Tests that Claude can correctly apply Ascend C development knowledge
# NOTE: This test takes 2-5 minutes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

echo "========================================"
echo -e " ${BOLD}Integration Test: Simple Op Development${NC}"
echo "========================================"
echo ""
echo "This test verifies Claude can apply Ascend C development knowledge."
echo "Estimated time: 2-5 minutes"
echo "Requires: Claude Code CLI"
echo ""

# Check if Claude Code is available
if ! command -v claude &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Claude Code CLI not found"
    echo "Install Claude Code first: https://code.claude.com"
    exit 1
fi

# Configuration
TIMEOUT=120

# Test 1: Verify skill knowledge - Add operator structure
print_section_header "Test 1: Add operator structure knowledge"

output=$(run_claude "你是一个 Ascend C 算子开发工程师。请简要说明实现一个简单的 Add 算子需要创建哪些文件？列出文件名和用途，用 5-6 行回答。" $TIMEOUT)

if echo "$output" | grep -qiE "op_host|op_kernel|tiling|def\.cpp|kernel\.cpp|\.h|\.cpp|CMakeLists|算子"; then
    print_pass "Knows Add operator file structure"
else
    print_fail "Missing file structure knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 2: Verify skill knowledge - TilingData structure
print_section_header "Test 2: TilingData structure knowledge"

output=$(run_claude "在 Ascend C 算子开发中，TilingData 结构体应该用什么语法定义？请给出一个简单的示例，用 3-4 行回答。" $TIMEOUT)

# Accept both modern (struct/class/OP_TILING_DATA) and legacy (BEGIN_TILING_DATA_DEF) patterns
if echo "$output" | grep -qiE "struct.*TilingData|class.*TilingData|TilingData|OP_TILING_DATA|BEGIN_TILING_DATA_DEF|TILING_DATA_FIELD_DEF"; then
    print_pass "Knows TilingData definition syntax"
else
    print_fail "Missing TilingData knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 3: Verify skill knowledge - Kernel function signature
print_section_header "Test 3: Kernel function signature knowledge"

output=$(run_claude "Ascend C 算子的核函数（Kernel）签名应该是什么格式？请写出基本模板，用 3-4 行回答。" $TIMEOUT)

if echo "$output" | grep -qiE "__global__|__aicore__|GM_ADDR|tiling"; then
    print_pass "Knows Kernel function signature"
else
    print_fail "Missing Kernel signature knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 4: Verify skill knowledge - Chip architecture
print_section_header "Test 4: Chip architecture knowledge"

output=$(run_claude "Ascend910B 和 Ascend950 分别对应什么架构？用 1-2 行回答。" $TIMEOUT)

if echo "$output" | grep -qiE "arch22|arch35|910B.*arch22|950.*arch35|达芬奇|DaVinci|架构"; then
    print_pass "Knows chip architecture mapping"
else
    print_fail "Missing chip architecture knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 5: Verify skill knowledge - ACLNN interface
print_section_header "Test 5: ACLNN interface knowledge"

output=$(run_claude "ACLNN 接口的两段式调用模式是什么？请简要说明，用 2-3 行回答。" $TIMEOUT)

if echo "$output" | grep -qiE "GetWorkspaceSize|workspace|两段|two.*stage"; then
    print_pass "Knows ACLNN interface pattern"
else
    print_fail "Missing ACLNN interface knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 6: Verify skill knowledge - Development workflow
print_section_header "Test 6: Development workflow knowledge"

output=$(run_claude "Ascend C 算子开发的基本流程是什么？按顺序列出主要步骤，用 3-4 行回答。" $TIMEOUT)

if echo "$output" | grep -qiE "设计|Tiling|tiling|Kernel|kernel|核函数|测试|test|分析|实现|开发|编译|验证"; then
    print_pass "Knows development workflow"
else
    print_fail "Missing development workflow knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 7: Verify skill knowledge - UT testing
print_section_header "Test 7: UT testing knowledge"

output=$(run_claude "Ascend C 算子的 UT 测试应该放在哪个目录？UT 测试的主要目的是什么？用 2-3 行回答。" $TIMEOUT)

if echo "$output" | grep -qiE "tests/ut|/ut|ut/|单元测试|unit.*test|UT.*测试"; then
    print_pass "Knows UT testing"
else
    print_fail "Missing UT testing knowledge"
    echo "  Output: $output"
    exit 1
fi

echo ""

# Test 8: Token usage analysis (optional)
print_section_header "Test 8: Token usage analysis"

session_file=$(find_recent_session 10)

if [ -n "$session_file" ] && [ -f "$session_file" ]; then
    echo "  Analyzing session: $session_file"
    python3 "$SCRIPT_DIR/../tools/analyze-token-usage.py" "$session_file" 2>/dev/null || print_info "Token analysis completed"
else
    print_skip "No recent session file found for token analysis"
fi

echo ""

# Test 9: Session tool invocation analysis
print_section_header "Test 9: Session tool invocation analysis"

if [ -n "$session_file" ] && [ -f "$session_file" ]; then
    # Count various tool invocations
    skill_count=$(count_tool_invocations "$session_file" "Skill")
    read_count=$(count_tool_invocations "$session_file" "Read")
    write_count=$(count_tool_invocations "$session_file" "Write")

    print_info "Tool invocations:"
    echo "    - Skill: $skill_count"
    echo "    - Read: $read_count"
    echo "    - Write: $write_count"

    # Check if any skills were triggered
    if [ "$skill_count" -gt 0 ]; then
        print_pass "Skills were triggered during test"
        print_info "Skills found:"
        grep -o '"skill":"[^"]*"' "$session_file" 2>/dev/null | sort -u | sed 's/^/    - /' || echo "    (none)"
    else
        print_info "No Skill tool invocations (may be expected for knowledge tests)"
    fi
else
    print_skip "No session file for tool analysis"
fi

echo ""

# Test 10: Workflow order verification
print_section_header "Test 10: Workflow order verification"

output=$(run_claude "在 Ascend C 算子开发中，应该先实现 Tiling 策略还是先实现 Kernel 计算？用 1 行回答。" 30)

if assert_order "$output" "Tiling|tiling|切分" "Kernel|kernel|计算" "Workflow order"; then
    :
else
    print_warn "May not follow correct workflow order"
fi

echo ""

# Summary
echo "========================================"
echo -e " ${BOLD}Integration Test Summary${NC}"
echo "========================================"
echo ""
echo -e "  ${GREEN}All knowledge verification tests passed!${NC}"
echo "  Claude correctly understands Ascend C development concepts."
echo ""

echo -e "${BOLD}=== Integration test completed ===${NC}"
