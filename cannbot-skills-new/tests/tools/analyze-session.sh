#!/usr/bin/env bash
# =============================================================================
# Session Analysis Tool
# =============================================================================
# Comprehensive analysis of Claude/OpenCode session files (.jsonl)
#
# Usage:
#   ./analyze-session.sh <session-file> [--full|--brief|--json]
#
# Output includes:
#   - Triggered skills and agents
#   - Tool call sequence
#   - Token usage breakdown
#   - Cost estimation
#   - Workflow analysis
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/test-helpers.sh"

MODE="${2:---brief}"

show_usage() {
    cat <<EOF
Session Analysis Tool

Usage: $0 <session-file> [options]

Options:
  --brief    Show summary only (default)
  --full     Show detailed analysis
  --json     Output as JSON
  --tools    Show tool call sequence
  --cost     Show cost breakdown only

Examples:
  $0 ~/.claude/projects/-path/to/project/session.jsonl
  $0 session.jsonl --full
  $0 session.jsonl --json | jq .
EOF
    exit 0
}

# Check arguments
if [ -z "${1:-}" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
fi

SESSION_FILE="$1"

if [ ! -f "$SESSION_FILE" ]; then
    echo "[ERROR] Session file not found: $SESSION_FILE"
    exit 1
fi

# ============================================
# Analysis Functions
# ============================================

analyze_session() {
    local file="$1"
    local mode="$2"
    
    # Basic stats
    local total_lines=$(wc -l < "$file" 2>/dev/null || echo "0")
    local file_size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "unknown")
    
    # Skill invocations
    local skill_count=$(count_tool_invocations "$file" "Skill")
    local skills=$(get_triggered_skills "$file")
    
    # Other tool counts
    local read_count=$(count_tool_invocations "$file" "Read")
    local write_count=$(count_tool_invocations "$file" "Write")
    local edit_count=$(count_tool_invocations "$file" "Edit")
    local bash_count=$(count_tool_invocations "$file" "Bash")
    local task_count=$(count_tool_invocations "$file" "Task")
    local todo_count=$(count_tool_invocations "$file" "TodoWrite")
    
    # Token usage
    local usage=$(extract_token_usage "$file")
    
    case "$mode" in
        --brief)
            echo "=== Session Analysis ==="
            echo ""
            echo "File: $file"
            echo "Size: $file_size"
            echo "Lines: $total_lines"
            echo ""
            echo "--- Skills Triggered ---"
            if [ -n "$skills" ]; then
                echo "$skills" | while read -r skill; do
                    echo "  - $skill"
                done
            else
                echo "  (none)"
            fi
            echo ""
            echo "--- Tool Invocations ---"
            echo "  Skill: $skill_count"
            echo "  Read: $read_count"
            echo "  Write: $write_count"
            echo "  Edit: $edit_count"
            echo "  Bash: $bash_count"
            echo "  Task: $task_count"
            echo "  TodoWrite: $todo_count"
            echo ""
            ;;
            
        --full)
            echo "=== Full Session Analysis ==="
            echo ""
            echo "File: $file"
            echo "Size: $file_size"
            echo "Lines: $total_lines"
            echo ""
            
            echo "=== Skills Triggered ==="
            if [ -n "$skills" ]; then
                echo "$skills" | while read -r skill; do
                    echo "  - $skill"
                done
            else
                echo "  (none)"
            fi
            echo ""
            
            analyze_tool_chain "$file"
            echo ""
            
            analyze_cost_breakdown "$file"
            echo ""
            
            # Premature action check
            check_premature_action "$file" "any"
            echo ""
            ;;
            
        --tools)
            analyze_tool_chain "$file"
            ;;
            
        --cost)
            analyze_cost_breakdown "$file"
            ;;
            
        --json)
            # Output as JSON
            local skills_json="[]"
            if [ -n "$skills" ]; then
                skills_json=$(echo "$skills" | awk '{s=s "\"" $0 "\","} END {print "[" substr(s,1,length(s)-1) "]"}')
            fi
            
            cat <<EOF
{
  "file": "$file",
  "size": "$file_size",
  "lines": $total_lines,
  "skills": $skills_json,
  "tools": {
    "Skill": $skill_count,
    "Read": $read_count,
    "Write": $write_count,
    "Edit": $edit_count,
    "Bash": $bash_count,
    "Task": $task_count,
    "TodoWrite": $todo_count
  },
  "usage": $usage
}
EOF
            ;;
    esac
}

# Run analysis
analyze_session "$SESSION_FILE" "$MODE"