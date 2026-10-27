#!/usr/bin/env bash
# =============================================================================
# Workflow Analysis Tool
# =============================================================================
# Analyze the execution workflow from a session file
#
# Checks:
#   - Skill invocation order
#   - Tool call sequence
#   - Premature actions
#   - Review loops
#   - Subagent dispatching
#
# Usage:
#   ./analyze-workflow.sh <session-file>
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <session-file.jsonl>"
    echo ""
    echo "Analyzes the execution workflow from a session file."
    exit 1
fi

SESSION_FILE="$1"

if [ ! -f "$SESSION_FILE" ]; then
    echo "[ERROR] Session file not found: $SESSION_FILE"
    exit 1
fi

echo "=== Workflow Analysis ==="
echo ""
echo "Session: $SESSION_FILE"
echo ""

# ============================================
# 1. Skill Triggering Analysis
# ============================================
echo "--- Skill Triggering ---"

first_skill_line=$(grep -n '"name":"Skill"' "$SESSION_FILE" 2>/dev/null | head -1 | cut -d: -f1 || true)

if [ -n "$first_skill_line" ]; then
    skill_name=$(sed -n "${first_skill_line}p" "$SESSION_FILE" 2>/dev/null | grep -o '"skill":"[^"]*"' | sed 's/"skill":"//;s/"$//' | head -1)
    echo "First skill invoked at line $first_skill_line: $skill_name"
else
    echo "No skill invocations found"
fi

# All triggered skills
echo ""
echo "All triggered skills:"
get_triggered_skills "$SESSION_FILE" | while read -r skill; do
    echo "  - $skill"
done

echo ""

# ============================================
# 2. Tool Call Sequence
# ============================================
echo "--- Tool Call Sequence ---"
analyze_tool_chain "$SESSION_FILE"

# ============================================
# 3. Premature Action Check
# ============================================
echo ""
echo "--- Premature Action Check ---"
check_premature_action "$SESSION_FILE" "any"

# ============================================
# 4. Subagent Analysis
# ============================================
echo ""
echo "--- Subagent Analysis ---"

subagent_count=$(grep -c '"name":"Task"' "$SESSION_FILE" 2>/dev/null || echo "0")
echo "Total subagents dispatched: $subagent_count"

if [ "$subagent_count" -gt 0 ]; then
    echo ""
    echo "Subagent details:"
    
    if command -v jq &> /dev/null; then
        jq -s '[.[] | select(.type == "user") | .toolUseResult? | select(. != null) | select(.agentId != null)] | 
            .[] | {agentId, prompt: .prompt[0:60]}' \
            "$SESSION_FILE" 2>/dev/null | \
            jq -r '"  \(.agentId): \(.prompt)..."' || true
    else
        grep -o '"agentId":"[^"]*"' "$SESSION_FILE" 2>/dev/null | sort -u | while read -r match; do
            agent_id=$(echo "$match" | sed 's/"agentId":"//;s/"$//')
            echo "  - $agent_id"
        done
    fi
fi

# ============================================
# 5. Todo/Progress Tracking
# ============================================
echo ""
echo "--- Progress Tracking ---"

todo_count=$(grep -c '"name":"TodoWrite"' "$SESSION_FILE" 2>/dev/null || echo "0")
echo "TodoWrite calls: $todo_count"

if [ "$todo_count" -gt 0 ]; then
    echo "Progress tracking was used during execution"
else
    echo "No progress tracking detected"
fi

# ============================================
# 6. Summary
# ============================================
echo ""
echo "=== Summary ==="
echo ""

# Calculate quality score
score=100
deductions=""

# Deduct for no skill invocation
if [ -z "$first_skill_line" ]; then
    score=$((score - 30))
    deductions="$deductions\n  - No skill invocation (-30)"
fi

# Deduct for premature actions
if grep -q '"name":"Write"\|"name":"Edit"' "$SESSION_FILE" 2>/dev/null; then
    first_write=$(grep -n '"name":"Write"\|"name":"Edit"' "$SESSION_FILE" 2>/dev/null | head -1 | cut -d: -f1 || true)
    if [ -n "$first_skill_line" ] && [ -n "$first_write" ] && [ "$first_write" -lt "$first_skill_line" ]; then
        score=$((score - 20))
        deductions="$deductions\n  - Premature Write/Edit (-20)"
    fi
fi

# Bonus for progress tracking
if [ "$todo_count" -gt 0 ]; then
    score=$((score + 5))
fi

echo "Workflow Quality Score: $score/100"

if [ -n "$deductions" ]; then
    echo ""
    echo "Deductions:"
    echo -e "$deductions"
fi

echo ""
echo "Analysis complete."