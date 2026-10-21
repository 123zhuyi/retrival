#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

if [ -t 1 ]; then
  GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${DIM}${GREEN}✓${NC}${DIM} $*${NC}"; }
warn() { echo -e "  ${YELLOW}⚠${NC}${DIM} $*${NC}"; }
err()  { echo -e "  ${RED}✗${NC}${DIM} $*${NC}"; }
info() { echo -e "  ${DIM}${CYAN}→${NC}${DIM} $*${NC}"; }
step() { echo -e "${DIM}$*${NC}"; }

PLUGIN_ID="triton-optimizer-multi-agent"
VERSION="0.1.0"
INCLUDED_SKILLS="triton-task-extractor triton-op-designer triton-op-coding triton-op-verifier triton-latency-optimizer triton-knowledge-retrieval triton-simulator-optimizer npu-arch triton-precision-debug"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_ROOT="$SCRIPT_DIR"
LOCAL_HOOK_ROOT="$PLUGIN_ROOT/hooks"

detect_trae_variant() {
    if [ -d "$HOME/.trae-cn" ]; then
        TRAE_VARIANT="ide"
    elif [ -d "$HOME/.marscode" ]; then
        TRAE_VARIANT="plugin"
    elif [ -d "$HOME/.traecli" ]; then
        TRAE_VARIANT="cli"
    else
        TRAE_VARIANT="unknown"
    fi
}

# Plugin-local control skills source
LOCAL_CTRL_SKILLS_DIR="$PLUGIN_ROOT/skills"

find_ops_root() {
    local candidate
    # 1) Shared ops directory (sibling in source tree)
    #    Operator skills (triton-op-coding, etc.) live in repo root ops/
    for candidate in "$PLUGIN_ROOT/../../ops" "$PLUGIN_ROOT/../../../ops"; do
        if [ -d "$candidate" ]; then
            cd "$candidate" && pwd
            return 0
        fi
    done

    local dependency_root="$PLUGIN_ROOT/../triton-op-generator-skills"
    if [ -d "$dependency_root" ]; then
        candidate="$(find "$dependency_root" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
        if [ -n "$candidate" ]; then
            cd "$candidate" && pwd
            return 0
        fi
    fi

    local search_dir="$PLUGIN_ROOT"
    for _ in 1 2 3 4 5; do
        search_dir="$(dirname "$search_dir")"
        if [ -d "$search_dir/ops" ]; then
            cd "$search_dir/ops" && pwd
            return 0
        fi
    done
    return 1
}

show_banner() {
  echo ""
  echo -e "${CYAN}"
  cat << 'BANNER'
   ____    _    _   _ _   _ ____        _
  / ___|  / \  | \ | | \ | | __ )  ___ | |_
 | |     / _ \ |  \| |  \| |  _ \ / _ \| __|
 | |___ / ___ \| |\  | |\  | |_) | (_) | |_
  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|
BANNER
  echo -e "${NC}"
  echo -e "  ${BOLD}Triton Optimizer — Multi-Agent Orchestrator${NC}"
  echo ""
}

show_help() {
    cat << EOF
CANNBot - Triton Optimizer Multi-Agent Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - Installation level: "project" (default) or "global"
  tool         - Target tool: "opencode" (default), "claude", "trae", "cursor", or "copilot"
  install_path - Project-level installation directory (default: current working directory)

Examples:
  init.sh project opencode
  init.sh project claude
  init.sh global claude
  init.sh project cursor /path/to/project

Installed content:
  Skills:     triton-task-extractor, triton-op-designer, triton-op-coding, triton-op-verifier, triton-latency-optimizer, triton-knowledge-retrieval, triton-simulator-optimizer
  Control:    triton-agent-loop (Multi-Agent round dispatch and orchestration)
  Template:   Index computation, math compute, normalization, object detection, quantization, tensor transform, transformer inference (copied per project)
  Hooks:      SessionStart/SessionEnd/PreToolUse
  Config:     AGENTS.md (Orchestrator-only round scheduling; per-round Phase 0-8 runs in isolated sub CLIs)
EOF
}

LEVEL="project"
TOOL="opencode"
INSTALL_PATH=""

for arg in "$@"; do
    case "$arg" in
        --help) show_help; exit 0 ;;
        global|project) LEVEL="$arg" ;;
        opencode|claude|trae|cursor|copilot) TOOL="$arg" ;;
        *)
            if [ -n "$INSTALL_PATH" ]; then
                err "Unexpected argument: $arg"
                show_help
                exit 1
            fi
            INSTALL_PATH="$arg"
            ;;
    esac
done

OPS_SKILL_ROOT="$(find_ops_root || true)"
if [ -z "$OPS_SKILL_ROOT" ] || [ ! -d "$OPS_SKILL_ROOT" ]; then
    err "Cannot find shared ops/ directory. Please run init.sh from the cannbot-skills source tree."
    exit 1
fi

if [ "$LEVEL" = "project" ]; then
    if [ -n "$INSTALL_PATH" ]; then
        if [ ! -d "$INSTALL_PATH" ]; then
            err "install_path is not an existing directory: $INSTALL_PATH"
            exit 1
        fi
        INSTALL_BASE="$(cd "$INSTALL_PATH" && pwd)"
    else
        INSTALL_BASE="$PWD"
    fi
else
    INSTALL_BASE="$PWD"
fi

if [ "$LEVEL" = "global" ]; then
    case "$TOOL" in
        opencode) CONFIG_ROOT="$HOME/.config/opencode" ;;
        claude) CONFIG_ROOT="$HOME/.claude" ;;
        trae)
            detect_trae_variant
            case "$TRAE_VARIANT" in
                plugin) CONFIG_ROOT="$HOME/.marscode" ;;
                cli) CONFIG_ROOT="$HOME/.traecli" ;;
                *) CONFIG_ROOT="$HOME/.trae-cn" ;;
            esac
            ;;
        cursor) CONFIG_ROOT="$HOME/.cursor" ;;
        copilot) CONFIG_ROOT="$HOME/.copilot" ;;
    esac
else
    case "$TOOL" in
        opencode) CONFIG_ROOT="$INSTALL_BASE/.opencode" ;;
        claude) CONFIG_ROOT="$INSTALL_BASE/.claude" ;;
        trae)
            detect_trae_variant
            case "$TRAE_VARIANT" in
                plugin) CONFIG_ROOT="$INSTALL_BASE/.marscode" ;;
                cli) CONFIG_ROOT="$INSTALL_BASE/.traecli" ;;
                *) CONFIG_ROOT="$INSTALL_BASE/.trae" ;;
            esac
            ;;
        cursor) CONFIG_ROOT="$INSTALL_BASE/.cursor" ;;
        copilot) CONFIG_ROOT="$INSTALL_BASE/.github" ;;
    esac
fi

CANNBOT_DIR="$CONFIG_ROOT"

if [ "$TOOL" = "claude" ]; then
    CONFIG_FILE_NAME="CLAUDE.md"
else
    CONFIG_FILE_NAME="AGENTS.md"
fi

install_skill_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    local count=0
    local skill src target
    for skill in $INCLUDED_SKILLS; do
        src="$OPS_SKILL_ROOT/$skill"
        target="$target_root/$skill"
        if [ ! -d "$src" ]; then
            warn "Skill not found: $skill"
            continue
        fi
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
        fi
        ln -sfn "$(realpath "$src")" "$target"
        count=$((count + 1))
    done
    ok "Skills: $count linked"
}

install_control_skill_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    if [ ! -d "$LOCAL_CTRL_SKILLS_DIR" ]; then
        info "No control skills to install (skills/ not found)"
        return 0
    fi
    local count=0
    local entry name target
    for entry in "$LOCAL_CTRL_SKILLS_DIR"/*; do
        [ -d "$entry" ] || continue
        name="$(basename "$entry")"
        target="$target_root/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
        fi
        ln -sfn "$(realpath "$entry")" "$target"
        count=$((count + 1))
    done
    ok "Control skills: $count linked"
}

install_template_links() {
    local target_root="$1"
    if [ -d "$PLUGIN_ROOT/template" ]; then
        # Avoid copying the template directory onto itself when installing into
        # the plugin root (self-install). Use realpath -s so symlinks are NOT
        # followed; otherwise an old symlink install would be mistaken for a copy.
        local src_real target_real
        src_real="$(realpath -s "$PLUGIN_ROOT/template" 2>/dev/null || echo "")"
        target_real="$(realpath -s "$target_root/template" 2>/dev/null || echo "")"
        if [ -n "$src_real" ] && [ -n "$target_real" ] && [ "$src_real" = "$target_real" ]; then
            tpl_count=$(find "$PLUGIN_ROOT/template" -maxdepth 1 -name '*.md' | wc -l)
            ok "Template: $tpl_count files in place"
            return 0
        fi

        # Old installs used a symlink. Remove it first so cp -R creates an
        # independent copy instead of following the link.
        if [ -L "$target_root/template" ]; then
            rm -f "$target_root/template"
        fi

        # Backup any per-project template already accumulated, rather than
        # silently deleting it with rm -rf.
        if [ -d "$target_root/template" ]; then
            local backup_name="template.backup.$(date +%Y%m%d_%H%M%S)"
            mv "$target_root/template" "$target_root/$backup_name"
            info "Existing template backed up to $target_root/$backup_name"
        fi

        if [ -e "$target_root/template" ] || [ -L "$target_root/template" ]; then
            rm -rf "$target_root/template"
        fi
        cp -R "$PLUGIN_ROOT/template" "$target_root/template"
        tpl_count=$(find "$PLUGIN_ROOT/template" -maxdepth 1 -name '*.md' | wc -l)
        ok "Template: $tpl_count files copied"
    else
        warn "Template dir not found, skip"
    fi
}

install_references_links() {
    local target_root="$1"
    local ref_src="$PLUGIN_ROOT/references"
    if [ -d "$ref_src" ]; then
        # If installing into the plugin root itself, references are already at the
        # target location (realpath resolves the symlink back to the source).
        # Do NOT symlink them onto themselves, which would delete the source directory.
        local src_real target_real
        src_real="$(realpath "$ref_src" 2>/dev/null || echo "")"
        target_real="$(realpath "$target_root/references" 2>/dev/null || echo "")"
        if [ -n "$src_real" ] && [ -n "$target_real" ] && [ "$src_real" = "$target_real" ]; then
            ref_count=$(find "$ref_src" -type f -name '*.md' 2>/dev/null | wc -l)
            ok "References: $ref_count files/dirs in place"
            return 0
        fi
        if [ -e "$target_root/references" ] || [ -L "$target_root/references" ]; then
            rm -rf "$target_root/references"
        fi
        ln -sfn "$ref_src" "$target_root/references"
        ref_count=$(find "$ref_src" -type f -name '*.md' 2>/dev/null | wc -l)
        ok "References: $ref_count files/dirs linked"
    else
        warn "References dir not found, skip"
    fi
}

install_hook_links() {
    local target_root="$1"
    mkdir -p "$target_root"
    local count=0
    local hook_entry name target
    for hook_entry in "$LOCAL_HOOK_ROOT"/*; do
        [ -e "$hook_entry" ] || continue
        name="$(basename "$hook_entry")"
        # Skip Python cache directories and other non-hook entries
        if [ "$name" = "__pycache__" ] || [ "$name" = ".DS_Store" ]; then
            continue
        fi
        target="$target_root/$name"
        if [ -e "$target" ] || [ -L "$target" ]; then
            rm -rf "$target"
        fi
        ln -sfn "$(realpath "$hook_entry")" "$target"
        count=$((count + 1))
    done
    ok "Hooks: $count linked"
}

install_config() {
    mkdir -p "$CONFIG_ROOT"
    local config_src="$PLUGIN_ROOT/AGENTS.md"
    local config_target
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/$CONFIG_FILE_NAME"
    else
        config_target="$CONFIG_ROOT/$CONFIG_FILE_NAME"
    fi

    if [ "$LEVEL" = "project" ] && [ "$PLUGIN_ROOT" = "$INSTALL_BASE" ] && [ "$CONFIG_FILE_NAME" = "AGENTS.md" ]; then
        ok "$CONFIG_FILE_NAME already in current directory"
    else
        ln -sf "$config_src" "$config_target"
        ok "$CONFIG_FILE_NAME"
    fi

    local config_root_target="$CONFIG_ROOT/$CONFIG_FILE_NAME"
    if [ "$config_root_target" != "$config_target" ]; then
        ln -sf "$config_src" "$config_root_target"
        ok "$CONFIG_FILE_NAME -> $(basename "$CONFIG_ROOT")/"
    fi
}

activate_hooks_settings() {
    if [ "$TOOL" != "claude" ] && [ "$TOOL" != "cursor" ]; then
        info "Hooks linked but settings activation is only configured for Claude/Cursor"
        return 0
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        warn "python3 not found; hooks linked but settings.json was not generated"
        return 0
    fi

    local settings_target="$CONFIG_ROOT/settings.json"
    local plugin_root_abs
    plugin_root_abs="$(realpath "$PLUGIN_ROOT")"
    python3 - "$LOCAL_HOOK_ROOT/hooks.json" "$settings_target" "$plugin_root_abs" <<'PY'
import json
import sys
from pathlib import Path

hooks_path = Path(sys.argv[1])
settings_path = Path(sys.argv[2])
plugin_root = sys.argv[3]

new_hooks = json.loads(
    hooks_path.read_text(encoding="utf-8").replace("${CLAUDE_PLUGIN_ROOT}", plugin_root)
)

if settings_path.exists():
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            settings = {}
    except json.JSONDecodeError:
        settings = {}
else:
    settings = {}

settings_hooks = settings.setdefault("hooks", {})
for event, entries in new_hooks.get("hooks", {}).items():
    dest = settings_hooks.setdefault(event, [])
    existing = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in dest}
    for entry in entries:
        key = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if key not in existing:
            dest.append(entry)
            existing.add(key)

settings_path.parent.mkdir(parents=True, exist_ok=True)
settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
    ok "settings.json hooks activated"
}

write_manifest() {
    local manifest="$CONFIG_ROOT/cannbot-manifest.json"
    local skills_json agents_json hooks_json

    # Combine operator skills from INCLUDED_SKILLS with plugin-local control skills
    local combined_skills=""
    for s in $INCLUDED_SKILLS; do combined_skills="$combined_skills $s"; done
    if [ -d "$LOCAL_CTRL_SKILLS_DIR" ]; then
        for entry in "$LOCAL_CTRL_SKILLS_DIR"/*; do
            [ -d "$entry" ] && combined_skills="$combined_skills $(basename "$entry")"
        done
    fi
    skills_json=$(printf '%s\n' $combined_skills | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")

    agents_json="[]"
    hooks_json=$(find "$LOCAL_HOOK_ROOT" -mindepth 1 -maxdepth 1 -exec basename {} \; | sort | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
    cat > "$manifest" << EOF
{
  "brand": "CANNBot",
  "version": "$VERSION",
  "team": "$PLUGIN_ID",
  "level": "$LEVEL",
  "tool": "$TOOL",
  "installed_skills": $skills_json,
  "installed_agents": $agents_json,
  "installed_hooks": $hooks_json,
  "brand_dir": "$CONFIG_ROOT",
  "install_time": "$(date -u +%Y%m%dT%H%M%SZ)"
}
EOF
    ok "Manifest: $manifest"
}

show_banner
echo "  Tool:      $TOOL"
echo "  Level:     $LEVEL"
echo "  Path:      $CONFIG_ROOT"
echo "  Skills:    $OPS_SKILL_ROOT"
echo ""

if [ "$TOOL" = "trae" ]; then
    detect_trae_variant
    info "TRAE variant: $TRAE_VARIANT"
    echo ""
fi

step "[1/5] Installing operator skills..."
mkdir -p "$CANNBOT_DIR"
install_skill_links "$CANNBOT_DIR/skills"
echo ""

step "[2/5] Installing control skills, template and references..."
install_control_skill_links "$CANNBOT_DIR/skills"
install_template_links "$CANNBOT_DIR"
install_references_links "$CANNBOT_DIR"
echo ""

step "[3/5] Installing hooks..."
install_hook_links "$CANNBOT_DIR/hooks"
activate_hooks_settings

# 生成插件自身 .claude/settings.json（从 hooks/hooks.json 模板替换 ${CLAUDE_PLUGIN_ROOT} 为真实路径）
if [ -f "$LOCAL_HOOK_ROOT/hooks.json" ]; then
    plugin_root_abs="$(realpath "$PLUGIN_ROOT")"
    mkdir -p "$PLUGIN_ROOT/.claude"
    sed "s|\${CLAUDE_PLUGIN_ROOT}|${plugin_root_abs}|g" \
        "$LOCAL_HOOK_ROOT/hooks.json" > "$PLUGIN_ROOT/.claude/settings.json"
    ok "Plugin .claude/settings.json generated from hooks.json"
fi

# 保护 hooks/ 目录下的脚本，防止被 Agent 删除或篡改。
# 注意：开发调试期间如需修改这些文件，必须先执行
#   chattr -i -R <path>
# 解除不可变属性，否则编辑/保存会报 EPERM。
if command -v chattr >/dev/null 2>&1; then
    for _hf in "$LOCAL_HOOK_ROOT"/*.py "$LOCAL_HOOK_ROOT"/*.json; do
        [ -f "$_hf" ] && chattr +i "$_hf" 2>/dev/null || true
    done
    for _hrf in "$LOCAL_HOOK_ROOT"/hook_runtime/*.py; do
        [ -f "$_hrf" ] && chattr +i "$_hrf" 2>/dev/null || true
    done
    ok "Hooks: immutable flag set ($(ls -1 "$LOCAL_HOOK_ROOT"/*.py "$LOCAL_HOOK_ROOT"/*.json "$LOCAL_HOOK_ROOT"/hook_runtime/*.py 2>/dev/null | wc -l) files)"
else
    warn "chattr not available, skip setting immutable flag on hooks (less secure)"
fi

# 保护 skills/triton-agent-loop/ 控制技能，防止被 Agent 删除。
# 注意：开发调试期间如需修改这些文件，必须先执行
#   chattr -i -R skills/triton-agent-loop
# 解除不可变属性，否则编辑/保存会报 EPERM。
if command -v chattr >/dev/null 2>&1; then
    if [ -d "$PLUGIN_ROOT/skills/triton-agent-loop" ]; then
        find "$PLUGIN_ROOT/skills/triton-agent-loop" -type f -exec chattr +i {} \; 2>/dev/null || true
        find "$PLUGIN_ROOT/skills/triton-agent-loop" -type d -exec chattr +i {} \; 2>/dev/null || true
        ok "Control skills: immutable flag set"
    fi
else
    warn "chattr not available, skip setting immutable flag on control skills (less secure)"
fi
echo ""

step "[4/5] Installing configuration..."
install_config
echo ""

step "[5/5] Writing manifest..."
write_manifest
echo ""

step "Health check..."
health_ok=true
[ -d "$CANNBOT_DIR/skills" ] || { err "skills/ missing"; health_ok=false; }
[ -d "$CANNBOT_DIR/hooks" ] || { err "hooks/ missing"; health_ok=false; }
[ -d "$CANNBOT_DIR/template" ] || warn "template/ missing (optional, skip)"
[ -f "$CONFIG_ROOT/cannbot-manifest.json" ] || { err "manifest missing"; health_ok=false; }
if [ "$LEVEL" = "project" ]; then
    [ -f "$INSTALL_BASE/$CONFIG_FILE_NAME" ] || { err "$CONFIG_FILE_NAME missing in project directory"; health_ok=false; }
else
    [ -f "$CONFIG_ROOT/$CONFIG_FILE_NAME" ] || { err "$CONFIG_FILE_NAME missing"; health_ok=false; }
fi

if [ "$health_ok" = true ]; then
    ok "All checks passed"
else
    exit 1
fi

echo ""
echo -e "  ${GREEN}${BOLD}✓ triton-optimizer-multi-agent installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo -e "  ${CYAN}1.${NC} 在目标 Triton Ascend NPU 算子项目中启动工具"
echo -e "  ${CYAN}2.${NC} 优化输入：${GREEN}${BOLD}请优化当前目录的 Triton Ascend 算子，从 baseline 开始按多轮迭代优化${NC}"
echo -e "  ${CYAN}3.${NC} 主 agent 负责多轮调度，每轮启动独立子 CLI 执行 Phase 0-8"
echo -e "  ${CYAN}4.${NC} 查看流程文档：${BOLD}cat .claude/references/orchestrator-loop.md${NC} 与 ${BOLD}cat .claude/references/algorithms/index.md${NC}（以及 algorithm 对应的 round.md）"
echo ""
