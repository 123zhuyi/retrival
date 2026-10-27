#!/usr/bin/env bash
# =============================================================================
# Token Usage Analysis Tool
# =============================================================================
# Analyze token usage from session files with detailed breakdown
#
# Usage:
#   ./analyze-tokens.sh <session-file>
#   ./analyze-tokens.sh <session-file> --json
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <session-file.jsonl> [--json]"
    echo ""
    echo "Analyzes token usage from Claude Code session transcripts."
    echo "Breaks down usage by main session and individual subagents."
    exit 1
fi

SESSION_FILE="$1"
OUTPUT_MODE="${2:---text}"

if [ ! -f "$SESSION_FILE" ]; then
    echo "[ERROR] Session file not found: $SESSION_FILE"
    exit 1
fi

# Use Python for detailed analysis if available
if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/analyze-token-usage.py" "$SESSION_FILE"
else
    # Fallback to basic bash analysis
    echo "=== Token Usage Analysis (basic) ==="
    echo ""
    
    # Count total lines
    total_lines=$(wc -l < "$SESSION_FILE" 2>/dev/null || echo "0")
    echo "Session lines: $total_lines"
    
    # Count tool invocations
    skill_count=$(grep -c '"name":"Skill"' "$SESSION_FILE" 2>/dev/null || echo "0")
    task_count=$(grep -c '"name":"Task"' "$SESSION_FILE" 2>/dev/null || echo "0")
    
    echo "Skill invocations: $skill_count"
    echo "Task (subagent) invocations: $task_count"
    echo ""
    
    # Extract basic usage if jq available
    if command -v jq &> /dev/null; then
        echo "Token usage (from last result):"
        jq -s '[.[] | select(.type == "result")] | last | .usage // {}' "$SESSION_FILE" 2>/dev/null || echo "  (not available)"
    fi
fi