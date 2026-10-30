#!/usr/bin/env python3
"""第三步：对导出的训练数据做结构规则校验 + 二元分流。

针对 export_training.py 产出的 *.raw.json 做检查并按结果分流到 pass/ 和 fail/。

设计原则：脚本只做能 100% 判定的结构规则；不能 100% 判定的语义信号
只扫描并回显到报告里（标 needs_manual_review），让用户自行决定是否手动剔除。

硬规则（违反即 FAIL，100% 可判定）：
    R1  messages 至少 3 条
    R2  第一条 role == "system"，且 content 非空
    R3  第二条 role == "user"
    R4  user 之后必须紧跟 assistant（不允许连续 user、user→tool、user 结尾）
    R5  不允许连续两条 assistant
    R6  最后一条 role == "assistant"
    R7  第一条 system 的 content 不能是 compaction marker
        （[--- context compaction boundary (Claude Code /compact) ---]）
    R8  仅主 agent，且仅在【无 --eval-report】时启用（文本回退判定）：
        最后一条 assistant 必须明确报告"编译通过 + 全部用例通过"
        （命中 编译...通过 + N/N 全 pass 或 "全量/全部...pass/通过" 关键词，
         且不含 "正确性未通过" 等强失败信号）
        子 agent（文件名含 __sub）不查 R8——它是局部任务，不产出最终汇报。
        主 agent 仅因 R8 失败时回退看配对 sub：sub 含成功信号则救回 PASS。
    R9  同算子分组级别（同一子文件夹内）：每个算子必须有且只有 1 个 subagent 文件
        （文件名 <base>__sub[N]_<type>.raw.json）。多了（多次委派，可能首次失败
        重试）或少了（没委派或失败没产出）都判失败，整组（主+子）→ fail/。
        分组键带父目录（父目录::base）——不同子文件夹的同名算子互不连带
        （不同文件夹是不同人/不同批次收集的，跨文件夹不应互相污染 R9 计数）。
    R10 整文件级：任意位置（含 trace 中段）出现 /compact 压缩边界标记
        → 整条 trace 视为无效。R7 只查首条 system，本规则补全文扫描。
    R11 整文件级：任意位置出现「Subagent 运行失败」即判 FAIL。

注：用例是否实际通过优先由 R-eval（batch_evaluate.py 实际跑
evaluate_ascendc.sh 重编译 + verification）唯一决定。eval_report.json 里 status
非 PASS 都强制判 FAIL（含 NO_TEST/未记录）；batch_evaluate 报告里的
impl_regression.regressed=true（实现退化为 PyTorch 原生，validate_ascendc_impl.py
的 Type 1-4）同样强制判 FAIL——纯 torch 兜底实现能数值通过真机评测，但对
SFT 是毒样本。R8 文本判定只在没给 --eval-report 时作为回退启用。

软规则（命中给 WARN，标 needs_manual_review，样本仍走 pass/）：
    这些规则的关键词可能出现在合法上下文里（如"修复了 error"），所以
    不能 100% 判定失败。命中后只在报告里回显。
    S1  最后一条 assistant 命中失败关键词（traceback/exception/exit code 等）
    S2  任意 tool result 命中 token / context 超限信号（max_tokens/context length 等）
    S3  任意 Agent/Task tool_call 对应的 tool result 命中 subagent 失败信号
    S-cross 跨子目录同算子重复（源自 stat_operators 的跨来源检测）：同一算子
        语义名（去编号前缀）出现在 >1 个子文件夹。只报告不 FAIL——
        分组键带父目录后跨目录被正确隔离，但训练集会过采样该算子，
        需人工决定去重。明细在报告 summary.cross_dir_duplicates。

分流（--split <dir>）：
    PASS（硬规则全过）→ mv 到 <dir>/pass/
    FAIL（任意硬规则违反）→ mv 到 <dir>/fail/

分流（--copy-passed <dir>，与 --split 互斥）：
    只把 PASS 文件 copy 到目标目录（保留相对 --copy-base 的子目录结构），
    绝不移动/删除原文件；FAIL 文件不复制，仅在报告里列路径+数量。

用法：
    # 只检查
    python3 scripts/check_dataset.py work/cleaned/ -r
    # 检查 + 分流（推荐）
    python3 scripts/check_dataset.py work/cleaned/ -r --split work/cleaned -o work/check_report.json
    # 关闭软规则（只跑硬规则）
    python3 scripts/check_dataset.py work/cleaned/ -r --no-heuristic --split work/cleaned
    # 试运行（不实际 move 文件）
    python3 scripts/check_dataset.py work/cleaned/ -r --split work/cleaned --dry-run
    # 检查 + 只复制合格文件（原数据不动）
    python3 scripts/check_dataset.py work/cleaned/ -r --copy-passed work/passed --copy-base work/cleaned

退出码：所有文件全 PASS = 0；任意文件 FAIL = 1。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

COMPACTION_MARKER = "[--- context compaction boundary (Claude Code /compact) ---]"

# S1: 最后一条 assistant 的失败信号（强信号，但仍可能误伤——"修复了 traceback"）
FAILURE_HINTS = re.compile(
    r"(?:traceback|exception\s+occurred|fatal\s+error|"
    r"<error>|<tool_use_error>|exit code\s*[1-9]\d*|"
    r"errno\s*[1-9]\d*|command not found)",
    re.I,
)

# S2: token / context 超限信号
TOKEN_LIMIT_HINTS = re.compile(
    r"(?:max_tokens|maximum\s+(?:context|tokens?)|context\s+length|"
    r"prompt\s+too\s+long|输入过长|超过\s*(?:token|上下文)|"
    r"context\s+window|exceeds?\s+(?:the\s+)?(?:token|context))",
    re.I,
)

# S3: subagent 调用 result 的失败信号
SUBAGENT_FAIL_HINTS = re.compile(
    r"(?:agent\s+(?:failed|error)|task\s+failed|"
    r"sub-?agent\s+(?:returned?\s+no|did\s+not\s+(?:return|complete))|"
    r"<tool_use_error>|no\s+result\s+(?:from|returned))",
    re.I,
)

SUBAGENT_TOOL_NAMES = {"Agent", "Task"}

# ---- R8（仅主 agent，且仅在无 --eval-report 时启用）：文本成功信号判定 ----
# 编译通过：编译 15 字符内出现 通过/✓/✅/成功/完成/pass（用负向先行断言排除"编译未通过/失败/出错"）
COMPILE_OK_RE = re.compile(
    r"编译(?!未|失败|出错|错误)[^\n]{0,15}?(?:通过|✓|✅|成功|完成|pass\b)|"
    r"compile(?!.*fail)[^\n]{0,20}(?:pass|complete)",
    re.I,
)
# N/N case pass 模式——必须带「用例语境」锚点，避免误吃路径段（如 ..._compact_75/109_...）
# 或 shape 数字（如 625/14）。比率两侧 30 字符内须出现 case/用例/通过/pass/匹配 等词。
# 比率本身要求两数之间是纯 " / "（允许空格），且不紧贴路径分隔或被字母/下划线包裹。
_RATIO = r"(?<![\w/])(\d+)\s*/\s*(\d+)(?![\w])"
CASE_CTX = r"(?:case|cases|用例|通过|pass|匹配|验证)"
CASE_RATIO_RE = re.compile(
    rf"{CASE_CTX}[^\n]{{0,15}}?{_RATIO}|{_RATIO}\s*(?:个)?\s*{CASE_CTX}",
    re.I,
)
# 全量/全部/所有 + 通过/pass 关键词（如 "全量 PASS"、"全部用例通过"）
ALL_PASS_KW_RE = re.compile(
    r"(?:全[部量]|所有|全部用例)[^\n]{0,40}(?:pass|通过|PASS|✓|✅)",
    re.I,
)
# 精度（验证）... ✅/PASS/通过（如 "精度验证（10 case） | ✅ PASS"）
ACCURACY_OK_RE = re.compile(
    r"精度(?:验证)?[^\n]{0,30}?(?:✅|✓|pass\b|通过)",
    re.I,
)
# 全量验证 ... PASS/通过（表格式表述，如 "全量验证 | 12/12 PASS"、"全量验证 | **PASS**"）
FULL_VERIFY_OK_RE = re.compile(
    r"全量验证[^\n]{0,40}?(?:✅|✓|pass\b|\*\*pass\*\*|通过)",
    re.I,
)
# 最终结论式成功信号（强结论词）：
#   "最终状态 **PASS**" / "最终状态：PASS" / "算子开发完成…PASS" / "finally PASS" / "最终判定：成功"
FINAL_STATE_OK_RE = re.compile(
    r"最终状态[:：]?\s*\**\s*pass\b|"
    r"最终判定[:：]?\s*\**\s*(?:成功|pass\b)|"
    r"算子开发完成[^\n]{0,20}\**\s*pass\b|"
    r"finally\s+pass\b",
    re.I,
)
# 明显失败信号（出现即 FAIL，即使有部分 pass 信号也不算成功）
# 只放无歧义的强失败短语 + 环境错误；不放 "验证失败 / 精度失败" 这种宽泛词
# （会被"验证失败后修复收敛"这种描述历史的话误伤）
HARD_FAIL_RE = re.compile(
    r"正确性未通过|编译未通过|编译失败|"
    r"全部失败|完全失败|"
    r"全量验证受限|"
    r"验证未通过|测试未通过|无法通过|"
    r"环境错误|环境问题.*失败",
    re.I,
)
# R11 整文件级强失败信号：trace 中任意位置出现「Subagent 运行失败」即判 FAIL
SUBAGENT_RUN_FAIL_RE = re.compile(r"Subagent\s*运行失败", re.I)


def _flatten_text(value: Any) -> str:
    """content / tool_calls 里取可用于关键词扫描的纯文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for blk in value:
            if isinstance(blk, str):
                parts.append(blk)
            elif isinstance(blk, dict):
                t = blk.get("text") or blk.get("thinking") or ""
                if isinstance(t, str):
                    parts.append(t)
                args = (blk.get("function") or {}).get("arguments")
                if isinstance(args, str):
                    parts.append(args)
        return "\n".join(parts)
    return str(value)


def _scan_soft_signals(messages: List[Dict[str, Any]]) -> List[str]:
    """扫 S1-S3 软规则，返回 warnings 列表（每条带 S 编号前缀）。"""
    warns: List[str] = []
    if not messages:
        return warns

    # S1: 最后一条 assistant 命中失败关键词
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    if last.get("role") == "assistant":
        text = _flatten_text(last.get("content"))
        for tc in last.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                text += "\n" + _flatten_text(fn.get("arguments"))
        m = FAILURE_HINTS.search(text)
        if m:
            warns.append(f"S1 最后一条 assistant 命中失败关键词 {m.group(0)!r}")

    # S2 / S3: 扫所有 tool result
    # 先把 subagent 的 tool_call_id 收集起来，之后看到对应 role=tool 消息时
    # 额外用 SUBAGENT_FAIL_HINTS 扫一遍。
    pending_subagent_call_ids: Set[str] = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                if fn.get("name") in SUBAGENT_TOOL_NAMES:
                    cid = tc.get("id")
                    if cid:
                        pending_subagent_call_ids.add(cid)
        elif role == "tool":
            content_text = _flatten_text(msg.get("content"))
            # S2: 任何 tool result 命中 token 超限信号
            mt = TOKEN_LIMIT_HINTS.search(content_text)
            if mt:
                warns.append(f"S2 tool result 命中 token 超限信号 {mt.group(0)!r}（位置 {i}）")
            # S3: subagent 调用的 result 命中失败信号
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id and tool_call_id in pending_subagent_call_ids:
                ms = SUBAGENT_FAIL_HINTS.search(content_text)
                if ms:
                    warns.append(
                        f"S3 subagent 调用 result 命中失败信号 {ms.group(0)!r}（位置 {i}）"
                    )
                pending_subagent_call_ids.discard(tool_call_id)

    return warns



def _has_success_signal(text: str) -> bool:
    """text 中是否出现「全部用例通过」类成功信号。

    成功信号任一即可：
      - N==N≥5 全过比率（用例语境下，如 16/16 case 通过）
      - 「全量/全部...pass/通过」关键词
      - 「精度（验证）... ✅/PASS/通过」
      - 「全量验证 ... PASS/通过」
      - 编译通过信号（补充）
    """
    for m in CASE_RATIO_RE.finditer(text):
        nums = m.group(1, 2) if m.group(1) is not None else m.group(3, 4)
        a, b = int(nums[0]), int(nums[1])
        if a >= 5 and b >= 5 and a == b:
            return True
    return bool(ALL_PASS_KW_RE.search(text) or ACCURACY_OK_RE.search(text)
                or FULL_VERIFY_OK_RE.search(text) or FINAL_STATE_OK_RE.search(text)
                or COMPILE_OK_RE.search(text))


def _r8_verdict(last_text: str, full_text_main: str) -> Optional[str]:
    """R8 主 agent 成功判定（最后一条 → 主 agent 全文 两级）。

    返回 FAIL 原因字符串；成功返回 None。
    - 任意范围出现硬失败信号（且最终无全过信号覆盖）→ FAIL
    - 成功信号：先看最后一条，没有再看主 agent 全文（debug trace 的 PASS
      常在中间消息，如 "全量验证 | **PASS**（16/16）"）。
    """
    # 用例比率部分通过：只在【没有任何全过信号】时才判失败（历史比率不否决最终全过）
    partial_fail: Optional[str] = None
    has_full_pass = False
    for m in CASE_RATIO_RE.finditer(full_text_main):
        nums = m.group(1, 2) if m.group(1) is not None else m.group(3, 4)
        a, b = int(nums[0]), int(nums[1])
        if a < 5 or b < 5:
            continue
        if a == b:
            has_full_pass = True
        elif partial_fail is None:
            partial_fail = f"{a}/{b}"

    success = _has_success_signal(last_text) or _has_success_signal(full_text_main)

    # 硬失败信号：出现且无全过信号覆盖 → FAIL
    if HARD_FAIL_RE.search(full_text_main) and not success:
        return "R8 主 agent 含明显失败信号（正确性未通过 / 编译失败 / 全量验证受限 等），且无全过信号"
    if partial_fail and not success:
        return f"R8 主 agent 报告用例部分通过（{partial_fail}），且无全过信号"
    if success:
        return None
    return ("R8 主 agent 未报告任何成功信号"
            "（缺 N/N 全 pass、'全量通过'、'精度...✅'、'全量验证 PASS' 或 '编译通过' 类表述）")


def check_sample(sample: Dict[str, Any], use_heuristic: bool,
                 is_main: bool = True,
                 use_r8: bool = False) -> Tuple[List[str], List[str]]:
    """对一条样本返回 (errors, warnings)。

    errors   = 硬规则违反（R1-R7 结构 + R8/R10/R11），命中任意一条 → FAIL
    warnings = 软规则命中（S1-S3），样本仍 PASS 但标 needs_manual_review

    is_main：文件名含 __sub 的子 agent 传 False，不查 R8（它是局部任务，
    不产出最终汇报）。
    use_r8：R8 文本成功判定的总开关——仅无 --eval-report（无真机评测）时
    由 main() 打开；有真机评测报告则成败完全由 R-eval 决定，不跑 R8。
    """
    errors: List[str] = []
    warnings: List[str] = []

    messages = sample.get("messages")
    if not isinstance(messages, list):
        return [f"messages 字段缺失或不是 list（实际类型 {type(messages).__name__}）"], []

    n = len(messages)
    if n < 3:
        return [f"R1 消息数 {n} < 3，无法构成最小训练样本（system+user+assistant）"], []

    # 整文件文本（content + tool_calls arguments 全拼接），供 R10/R11/R8 全文扫描
    whole_text_parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        whole_text_parts.append(_flatten_text(m.get("content")))
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                whole_text_parts.append(_flatten_text(fn.get("arguments")))
    whole_text = "\n".join(whole_text_parts)

    # R10 整文件级：任意位置出现 /compact 压缩边界标记 → 整条 trace 无效
    # （compact 可能发生在 trace 中段，只查首条 system 的 R7 检不出来）
    if COMPACTION_MARKER in whole_text:
        errors.append("R10 文件中出现 /compact 压缩边界标记，整条 trace 视为无效")

    # R11 整文件级：任意位置出现「Subagent 运行失败」即判 FAIL
    if SUBAGENT_RUN_FAIL_RE.search(whole_text):
        errors.append("R11 文件中出现「Subagent 运行失败」，判为失败")

    # R2 第一条必须是 system 且非空 / R7 不能是 compaction marker
    first = messages[0] if isinstance(messages[0], dict) else {}
    if first.get("role") != "system":
        errors.append(f"R2 第一条 role={first.get('role')!r}，应为 'system'")
    else:
        content = _flatten_text(first.get("content")).strip()
        if not content:
            errors.append("R2 第一条 system 的 content 为空")
        elif COMPACTION_MARKER in content:
            errors.append("R7 第一条 system 的 content 是 compaction marker（/compact 残留）")

    # R3 第二条必须是 user
    second = messages[1] if isinstance(messages[1], dict) else {}
    if second.get("role") != "user":
        errors.append(f"R3 第二条 role={second.get('role')!r}，应为 'user'")

    # R4 user 之后必须紧跟 assistant / R5 不允许连续 assistant
    for i in range(n - 1):
        cur = messages[i] if isinstance(messages[i], dict) else {}
        nxt = messages[i + 1] if isinstance(messages[i + 1], dict) else {}
        cur_role = cur.get("role")
        nxt_role = nxt.get("role")
        if cur_role == "user" and nxt_role != "assistant":
            errors.append(
                f"R4 位置 {i} 是 user，但 {i + 1} 是 {nxt_role!r}（应为 assistant）"
            )
        if cur_role == "assistant" and nxt_role == "assistant":
            errors.append(f"R5 位置 {i}/{i + 1} 出现连续两条 assistant")

    # R6 最后一条必须是 assistant
    last = messages[-1] if isinstance(messages[-1], dict) else {}
    if last.get("role") != "assistant":
        errors.append(f"R6 最后一条 role={last.get('role')!r}，应为 'assistant'")

    # R8（仅主 agent，且仅在无真机评测报告时）：必须报告"全部用例通过"成功信号
    # —— 扫描两级：最后一条 assistant → 主 agent 全文（debug trace 的 PASS 常在中间消息）。
    # 若主 agent 仍无成功信号，由 main() 做第三级回退：看配对 sub agent。
    if use_r8 and is_main and last.get("role") == "assistant":
        last_content = _flatten_text(last.get("content"))
        last_text = last_content
        for tc in last.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                last_text += "\n" + _flatten_text(fn.get("arguments"))
        r8_err = _r8_verdict(last_text, whole_text)
        if r8_err:
            # 最后一条 content 为空时，在原因里额外点明（仍允许 main 做 sub 回退）
            if not last_content.strip():
                r8_err = "R8 最后一条 assistant 的 content 为空；" + r8_err
            errors.append(r8_err)

    # 软规则
    if use_heuristic:
        warnings.extend(_scan_soft_signals(messages))

    return errors, warnings


def _extract_base(filename: str) -> str:
    """从样本文件名抽算子 base name。

    export_training.py 的实际命名（见其 1404 行）：
      单 sub:  <base>__sub_<type>.raw.json      （suffix='sub'，无数字）
      多 sub:  <base>__subN_<type>.raw.json     （N=1,2,...）

    外部收集的数据还有纯 .json 命名（不带 .raw），__sub 剥离不能依赖后缀：
      0_add__sub_explore.json             → '0_add'
    否则主/子文件抽出的 base 不同，分不到同组，R9 会把明明有 sub 的组
    误报成 subagent 数量 0。

    所以：
      0_add.raw.json                       → '0_add'
      0_add__sub_explore.raw.json          → '0_add'   ← 单子，无数字
      0_add__sub1_explore.raw.json         → '0_add'   ← 多子第 1 个
      0_add__sub10_plan.raw.json           → '0_add'
      0_add__sub_explore.json              → '0_add'   ← 纯 .json 同样剥
    其他形态 → 文件 stem。
    """
    # 先剥 .raw.json / .clean.json 后缀；纯 .json 则取 stem
    m = re.match(r'^(.+?)\.(?:raw|clean)\.json$', filename)
    stem = m.group(1) if m else Path(filename).stem
    # 切掉 __sub<可选数字>_ 后缀
    sep = re.search(r'__sub\d*_', stem)
    return stem[:sep.start()] if sep else stem



def _group_key(entry: Dict[str, Any]) -> str:
    """同组判定的键：父目录 + 算子 base name。

    必须带父目录，使【同名算子但在不同子文件夹】不被当作同组——
    不同文件夹是不同人/不同批次收集的，跨文件夹不应互相连带判失败
    （否则两个子文件夹各有 1 个 sub 的同名算子会被 R9 误算成 2 个而整组 FAIL）。
    """
    path = Path(entry.get("file") or entry.get("source") or "")
    base = _extract_base(path.name)
    return f"{path.parent}::{base}"


def _detect_cross_dir_duplicates(report_passed: List[Dict[str, Any]],
                                 report_failed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """检测跨子目录的同算子重复（S-cross 规则，只报告不 FAIL）。

    源自 tools/scripts/stat_operators.py 的跨来源重复检测：同一算子
    （去编号前缀后的语义名）出现在 >1 个不同子文件夹 → 记录到报告，
    提醒训练集会过采样该算子。分组的父目录::base 键恰好隔离了跨目录
    情形，这里显式回收这个信号——只标记，不影响 pass/fail 判定。
    """
    from collections import defaultdict

    # semantic name: 去掉 "9_TopK" 的 "9_" 前缀（跨来源归并口径与 stat_operators 一致）
    def _sem(base: str) -> str:
        m = re.match(r"\d+_(.+)", base)
        return m.group(1) if m else base

    # 语义名 -> {父目录: [组键]}
    sem_to_dirs: Dict[str, Dict[str, list]] = defaultdict(dict)
    for e in report_passed + report_failed:
        gkey = _group_key(e)
        parent, base = gkey.split("::", 1)
        sem = _sem(base)
        sem_to_dirs[sem].setdefault(parent, []).append(gkey)

    dups = []
    for sem, dirs in sorted(sem_to_dirs.items()):
        if len(dirs) < 2:
            continue
        dups.append({
            "op": sem,
            "n_dirs": len(dirs),
            "dirs": sorted(dirs),
            "groups": sorted(g for gs in dirs.values() for g in gs),
        })
    return dups


def _apply_eval_override(
    report_passed: List[Dict[str, Any]],
    report_failed: List[Dict[str, Any]],
    eval_report: Dict[str, Any],
) -> Tuple[int, int]:
    """根据 batch_evaluate.py 产出的报告判定（用例是否通过由评测脚本唯一决定）。

    用户规则：pass/ 算子必须是脚本验证通过的。
      - eval 里 status=PASS → 不影响
      - eval 里 status=FAIL/ERROR/TIMEOUT → 强制整组 FAIL
      - eval 里 status=NO_TEST 或该算子没记录 → 强制整组 FAIL（没评测就是没验证）
      - eval 里 impl_regression.regressed=true（实现退化为 PyTorch 原生，
        见 validate_ascendc_impl.py）→ 强制整组 FAIL——纯 torch 兜底实现
        也能数值通过真机评测，但对 SFT 是毒样本，必须拦下。

    返回：(overridden, blocked) 两个计数。
      overridden：原本 PASS 被推翻为 FAIL 的样本数
      blocked：没评测记录（NO_TEST/缺失）被强制 FAIL 的样本数
    """
    if not eval_report or "ops" not in eval_report:
        return 0, 0
    overridden = 0
    blocked = 0
    # 收集所有需要判定的组键（父目录::base，即所有出现过的算子组）
    all_keys = set()
    for e in report_passed + report_failed:
        all_keys.add(_group_key(e))

    for gkey in all_keys:
        base = gkey.split("::", 1)[-1]
        op_eval = eval_report["ops"].get(base)
        if op_eval is None:
            # 该算子没评测记录 → 强制 FAIL
            verdict = "NO_TEST"
            detail = "（无评测记录）"
        else:
            # 实现退化检测优先拦：即使真机 status=PASS，退化为 PyTorch 原生
            # 实现的样本也判失败（真机数值对≠真的是 AscendC kernel 在算）
            impl = op_eval.get("impl_regression") or {}
            if impl.get("regressed"):
                rtype = impl.get("regression_type")
                sug = (impl.get("suggestion") or "")[:120].replace("\n", " ")
                verdict = "REGRESSED"
                detail = f"（实现退化 Type {rtype}：{sug}）"
            else:
                verdict = op_eval.get("status", "NO_TEST")
                if verdict == "PASS":
                    continue  # PASS 不影响
                if verdict == "FAIL":
                    p = op_eval.get("passed", 0)
                    f = op_eval.get("failed", 0)
                    detail = f"（实际跑测试 {p}/{p + f} passed）"
                elif verdict == "ERROR":
                    snippet = (op_eval.get("error_snippet", "") or "")[:80].replace("\n", " ")
                    detail = f"（实际跑测试崩溃: {snippet}）"
                elif verdict == "TIMEOUT":
                    detail = f"（实际跑测试超时 {op_eval.get('duration_s', 0)}s）"
                else:  # NO_TEST
                    detail = f"（{op_eval.get('error_snippet', '无 test 脚本')[:60]}）"

        # 该组（父目录::base）下所有 PASS 标 FAIL
        for e in list(report_passed):
            if _group_key(e) != gkey:
                continue
            e["status"] = "FAIL"
            e["errors"].append(
                f"R-eval 实际评测 {verdict}{detail}，强制判失败"
                f"（pass/ 必须是脚本验证通过的）"
            )
            e["needs_manual_review"] = False
            report_passed.remove(e)
            report_failed.append(e)
            if verdict == "NO_TEST":
                blocked += 1
            else:
                overridden += 1
    return overridden, blocked


def _apply_grouping(
    report_passed: List[Dict[str, Any]],
    report_failed: List[Dict[str, Any]],
    eval_report: Optional[Dict[str, Any]] = None,
) -> Tuple[int, int, int, int]:
    """同 base 的样本作为一组，原地修改 entries：
       - R9: subagent 数量必须正好是 1（多了少了都 FAIL）
       - R-eval: 实际评测非 PASS（FAIL/ERROR/TIMEOUT/NO_TEST）→ 整组 FAIL
       - 任意成员 FAIL → 整组 FAIL（含从 passed 移到 failed）
       - 任意成员 needs_manual_review → 整组 needs_manual_review

    返回：(r9_demoted, eval_overridden, eval_blocked, group_demoted) 四种计数。
    """
    r9_demoted = 0
    eval_overridden = 0
    eval_blocked = 0
    group_demoted = 0

    all_entries = report_passed + report_failed
    group_to_entries: Dict[str, List[Dict[str, Any]]] = {}
    for e in all_entries:
        group_to_entries.setdefault(_group_key(e), []).append(e)

    # 1. R9: subagent 数量必须正好是 1（用户规则：有且只有 1 个 subagent）
    for gkey, entries in group_to_entries.items():
        disp = gkey.split("::", 1)[-1]
        sub_count = sum(1 for e in entries
                        if '__sub' in Path(e["source"]).stem)
        if sub_count == 1:
            continue
        for e in entries:
            if e["status"] != "PASS":
                continue
            e["status"] = "FAIL"
            e["errors"].append(
                f"R9 算子组 {disp!r}（同子文件夹）的 subagent 数量为 {sub_count}，应为 1"
            )
            e["needs_manual_review"] = False
            report_passed.remove(e)
            report_failed.append(e)
            r9_demoted += 1

    # 1.5 R-eval: 实际评测非 PASS（FAIL/ERROR/TIMEOUT/NO_TEST/缺失）→ 整组 FAIL
    if eval_report:
        eval_overridden, eval_blocked = _apply_eval_override(
            report_passed, report_failed, eval_report
        )

    # 3. R-group: 任意成员 FAIL → 整组 FAIL（含从 passed 移到 failed）
    for gkey, entries in group_to_entries.items():
        disp = gkey.split("::", 1)[-1]
        fail_names = [Path(m["source"]).name for m in entries
                      if m["status"] == "FAIL"]
        if not fail_names:
            continue
        for e in entries:
            if e["status"] != "PASS":
                continue
            e["status"] = "FAIL"
            e["errors"].append(
                f"R-group 同算子组 {disp!r}（同子文件夹）内有成员 FAIL"
                f"（{', '.join(fail_names)}），整组判失败"
            )
            e["needs_manual_review"] = False
            report_passed.remove(e)
            report_failed.append(e)
            group_demoted += 1

    # 4. needs_manual_review 传播（只在剩下的 PASS 成员间）
    group_to_pass: Dict[str, List[Dict[str, Any]]] = {}
    for e in report_passed:
        group_to_pass.setdefault(_group_key(e), []).append(e)

    for gkey, entries in group_to_pass.items():
        disp = gkey.split("::", 1)[-1]
        review_members = [e for e in entries if e.get("needs_manual_review")]
        clean_members = [e for e in entries if not e.get("needs_manual_review")]
        if not review_members or not clean_members:
            continue
        review_summary = []
        for m in review_members:
            fname = Path(m["source"]).name
            for w in m["warnings"]:
                review_summary.append(f"{fname}: {w}")
        reason = (f"S-group 同算子组 {disp!r}（同子文件夹）内有可疑成员"
                  f"（{'；'.join(review_summary)}）")
        for e in clean_members:
            e["warnings"].append(reason)
            e["needs_manual_review"] = True

    return r9_demoted, eval_overridden, eval_blocked, group_demoted


def load_samples(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """加载一个文件返回 [(来源标识, sample), ...]。

    支持两种形态：
      - 单个 dict（带 messages 字段）
      - list[dict] / JSONL：每个元素是一条样本
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{ln} JSON 解析失败：{e}")
                samples.append((f"{path.name}#L{ln}", obj))
        return samples

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} JSON 解析失败：{e}")

    if isinstance(data, dict):
        return [(path.name, data)]
    if isinstance(data, list):
        return [(f"{path.name}[{i}]", s) for i, s in enumerate(data)]
    raise ValueError(f"{path} 顶层既不是 dict 也不是 list")


def _is_under(path: Path, ancestor: Path) -> bool:
    """path 是否位于 ancestor 目录下（含多层子目录）。"""
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except ValueError:
        return False


def collect_files(targets: List[str], recursive: bool,
                  exclude_dirs: List[Path] = None,
                  exclude_suffixes: List[str] = None) -> List[Path]:
    """收集所有待检查的文件。

    exclude_dirs       : 这些目录（含子树）下的文件会被跳过。
    exclude_suffixes   : 路径中任一层目录名「小写后以该后缀结尾」则整树跳过
                         （如 'failed' 排除 level_1_failed/）。大小写不敏感。
    """
    excludes = exclude_dirs or []
    suffixes = [s.lower() for s in (exclude_suffixes or [])]

    def _suffix_excluded(c: Path) -> bool:
        if not suffixes:
            return False
        for part in c.parts:
            pl = part.lower()
            if any(pl.endswith(suf) for suf in suffixes):
                return True
        return False

    files: List[Path] = []
    for t in targets:
        p = Path(t)
        if not p.exists():
            print(f"[warn] 跳过不存在的路径：{p}", file=sys.stderr)
            continue
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            candidates = sorted(p.rglob("*.json")) if recursive else sorted(p.glob("*.json"))
            candidates += sorted(p.rglob("*.jsonl")) if recursive else sorted(p.glob("*.jsonl"))
            for c in candidates:
                if any(_is_under(c, ex) for ex in excludes):
                    continue
                if _suffix_excluded(c):
                    continue
                files.append(c)
        else:
            print(f"[warn] 跳过非文件非目录的路径：{p}", file=sys.stderr)
    return sorted(set(files))


def _split_sample(src_file: Path, split_dir: Path, status: str, dry_run: bool) -> Path:
    """把样本文件 move 到 split_dir/pass/ 或 split_dir/fail/。返回目标路径。"""
    subdir = split_dir / ("pass" if status == "PASS" else "fail")
    subdir.mkdir(parents=True, exist_ok=True)
    dst = subdir / src_file.name
    if dry_run:
        print(f"[dry] mv {src_file} -> {dst}")
    else:
        if dst.exists():
            print(f"[warn] 目标已存在，覆盖：{dst}", file=sys.stderr)
            dst.unlink()
        shutil.move(str(src_file), str(dst))
    return dst


def _copy_passed(src_file: Path, base_root: Path, dst_root: Path,
                 dry_run: bool) -> Path:
    """把合格文件 copy 到 dst_root，保留相对 base_root 的子目录结构。

    只复制，绝不移动/删除原文件。
    """
    try:
        rel = src_file.resolve().relative_to(base_root.resolve())
    except ValueError:
        rel = Path(src_file.name)  # 不在 base_root 下则只保留文件名
    dst = dst_root / rel
    if dry_run:
        print(f"[dry] copy {src_file} -> {dst}")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_file), str(dst))
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="待检查的 .json / .jsonl 文件或目录")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="对目录递归扫描 *.json / *.jsonl")
    ap.add_argument("--no-heuristic", dest="heuristic", action="store_false",
                    help="关闭软规则（S 系列扫描），只跑硬规则")
    ap.add_argument("--no-group-by-base", dest="group_by_base",
                    action="store_false",
                    help="关闭同算子分组判定（默认开）。默认主 agent 和它的 "
                         "sub agent（<base>__sub<i>_<type>.raw.json）作为一组："
                         "组内任意 FAIL 整组 FAIL、任意 needs_manual_review "
                         "整组标。加这个参数后每个文件独立判定。")
    ap.add_argument("--split", metavar="DIR",
                    help="检测完后把样本文件 mv 到 DIR/pass/（PASS）或 DIR/fail/（FAIL）")
    ap.add_argument("--copy-passed", metavar="DIR",
                    help="只把合格(PASS)文件 copy 到 DIR（保留相对子目录结构），"
                         "绝不移动/删除原文件；FAIL 文件不复制，仅在报告里列路径+数量。"
                         "与 --split 互斥（--split 会移动原文件）。")
    ap.add_argument("--copy-base", metavar="DIR",
                    help="计算相对子目录结构时的基准根目录（默认取第一个输入目录）。"
                         "仅配合 --copy-passed 使用。")
    ap.add_argument("--exclude-suffix", action="append", default=[],
                    metavar="SUFFIX",
                    help="排除名字（小写后）以该后缀结尾的目录，整树跳过。可重复。"
                         "如 --exclude-suffix failed 排除 level_1_failed/。")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印会做什么，不实际 move/copy（与 --split/--copy-passed 配合）")
    ap.add_argument("--eval-report", metavar="JSON",
                    help="batch_evaluate.py 产出的 JSON 报告路径。"
                         "强制规则：pass/ 必须是脚本验证通过的。eval 里 status=PASS 才放行，"
                         "FAIL/ERROR/TIMEOUT/NO_TEST/未记录都强制判 FAIL。")
    ap.add_argument("-o", "--output", help="把详细报告写入该 JSON 文件")
    ap.add_argument("--fail-fast", action="store_true",
                    help="遇到第一条 ERROR 立即终止（默认会扫完所有样本）")
    ap.set_defaults(heuristic=True, group_by_base=True)
    args = ap.parse_args()

    if args.split and args.copy_passed:
        print("[check] --split 与 --copy-passed 互斥（前者移动原文件，后者只复制）",
              file=sys.stderr)
        return 2

    split_dir = Path(args.split) if args.split else None
    copy_dir = Path(args.copy_passed) if args.copy_passed else None
    # copy-passed 的相对结构基准：优先 --copy-base，否则取第一个输入目录
    copy_base = None
    if copy_dir:
        if args.copy_base:
            copy_base = Path(args.copy_base)
        else:
            first = Path(args.inputs[0])
            copy_base = first if first.is_dir() else first.parent
    # 输入和 split 同目录时，跳过前次分流产生的 pass/ fail/ 子目录
    exclude_dirs: List[Path] = []
    if split_dir:
        exclude_dirs = [split_dir / "pass", split_dir / "fail"]

    files = collect_files(args.inputs, args.recursive, exclude_dirs,
                          exclude_suffixes=args.exclude_suffix)
    if not files:
        print("[check] 没有可检查的文件", file=sys.stderr)
        return 2

    # R8 文本成功判定：仅作无真机评测时的回退。给了 --eval-report（无论文件
    # 是否存在/可解析）成败都交给 R-eval，不跑 R8——真机评测优先于文本启发式。
    use_r8 = not args.eval_report
    if use_r8:
        print("[check] 未提供 --eval-report：启用 R8 文本成功判定（回退模式）")
    else:
        print("[check] 已提供 --eval-report：成败由真机评测（R-eval）决定，R8 不生效")

    report_passed: List[Dict[str, Any]] = []
    report_failed: List[Dict[str, Any]] = []
    n_pass = n_fail = n_warn = 0
    needs_review_entries: List[Dict[str, Any]] = []
    file_fulltext: Dict[str, str] = {}  # 文件路径 -> 整文件文本（用于 R8 sub 回退）

    for f in files:
        try:
            samples = load_samples(f)
        except ValueError as e:
            print(f"[FAIL] {f}: {e}", file=sys.stderr)
            report_failed.append({"file": str(f), "source": str(f),
                                  "status": "FAIL",
                                  "errors": [str(e)], "warnings": [],
                                  "needs_manual_review": False})
            n_fail += 1
            if args.fail_fast:
                break
            continue

        # 文件名含 __sub 的是子 agent，不查 R8
        is_main_for_file = "__sub" not in f.stem
        for src, sample in samples:
            # 缓存整文件文本（供 sub 回退用）
            msgs = sample.get("messages") if isinstance(sample, dict) else None
            if isinstance(msgs, list):
                parts = []
                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    parts.append(_flatten_text(m.get("content")))
                    for tc in m.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                            parts.append(_flatten_text(fn.get("arguments")))
                file_fulltext[str(f)] = "\n".join(parts)

            errors, warnings = check_sample(sample, args.heuristic,
                                            is_main=is_main_for_file,
                                            use_r8=use_r8)
            needs_review = bool(warnings)
            if errors:
                status = "FAIL"
                n_fail += 1
            else:
                status = "PASS"
                n_pass += 1
                if warnings:
                    n_warn += 1
            entry = {
                "file": str(f),
                "source": src,
                "status": status,
                "errors": errors,
                "warnings": warnings,
                "is_main": is_main_for_file,
                "needs_manual_review": needs_review,
            }
            if errors:
                report_failed.append(entry)
                print(f"[FAIL] {src}")
                for e in errors:
                    print(f"        - {e}")
            else:
                report_passed.append(entry)
                if warnings:
                    print(f"[WARN] {src}")
                    for w in warnings:
                        print(f"        - {w}")
                    needs_review_entries.append(entry)
                else:
                    print(f"[ OK ] {src}")
            if errors and args.fail_fast:
                break

    # 同算子分组判定：默认开，把同 base 的主+子 agent 作为一组
    # 读 eval 报告（可选）
    eval_report_data: Optional[Dict[str, Any]] = None
    if args.eval_report:
        eval_path = Path(args.eval_report)
        if not eval_path.is_file():
            print(f"[warn] eval 报告不存在: {eval_path}（跳过 R-eval）", file=sys.stderr)
        else:
            try:
                eval_report_data = json.loads(eval_path.read_text(encoding="utf-8"))
                print(f"[check] 读入 eval 报告: {eval_path}")
            except json.JSONDecodeError as e:
                print(f"[warn] eval 报告 JSON 解析失败: {e}（跳过 R-eval）", file=sys.stderr)

    # R8 第三级回退：主 agent 仅因 R8（无成功信号）失败时，回退看【同子文件夹的配对 sub】。
    # 若配对 sub 的整文件含成功信号且无硬失败，则认定主 agent 也成功，撤销该 FAIL。
    def _is_r8_only_fail(entry):
        errs = entry.get("errors") or []
        return bool(errs) and all(e.startswith("R8") for e in errs)

    # 建索引：同 base（父目录::base）下的 sub 文件全文
    sub_text_by_key: Dict[str, List[str]] = {}
    for path, txt in file_fulltext.items():
        p = Path(path)
        if "__sub" in p.stem:
            key = f"{p.parent}::{_extract_base(p.name)}"
            sub_text_by_key.setdefault(key, []).append(txt)

    rescued = 0
    for e in list(report_failed):
        if not e.get("is_main") or not _is_r8_only_fail(e):
            continue
        key = f"{Path(e['file']).parent}::{_extract_base(Path(e['file']).name)}"
        for sub_txt in sub_text_by_key.get(key, []):
            if _has_success_signal(sub_txt) and not (
                    HARD_FAIL_RE.search(sub_txt) or SUBAGENT_RUN_FAIL_RE.search(sub_txt)):
                e["status"] = "PASS"
                e["errors"] = []
                e["warnings"].append(
                    "R8-sub-rescue 主 agent 无成功信号，但配对 sub agent 含全过信号，判为成功")
                e["needs_manual_review"] = True
                report_failed.remove(e)
                report_passed.append(e)
                rescued += 1
                break
    if rescued:
        print(f"[check] R8 sub 回退：{rescued} 个主 agent 因配对 sub 含成功信号被救回为 PASS")

    if args.group_by_base:
        r9_demoted, eval_overridden, eval_blocked, group_demoted = _apply_grouping(
            report_passed, report_failed, eval_report_data
        )
        # 重算计数（_apply_grouping 可能改变了 report_passed/failed 的成员）
        n_pass = len(report_passed)
        n_fail = len(report_failed)
        n_warn = sum(1 for e in report_passed if e.get("needs_manual_review"))
        needs_review_entries = [e for e in report_passed if e.get("needs_manual_review")]
        if r9_demoted > 0:
            print(f"[check] R9 subagent 数量异常：{r9_demoted} 个样本被降级为 FAIL")
        if eval_overridden > 0:
            print(f"[check] R-eval 实际评测失败：{eval_overridden} 个样本推翻 R8 判定为 FAIL")
        if eval_blocked > 0:
            print(f"[check] R-eval 未评测算子：{eval_blocked} 个样本强制判 FAIL（pass/ 必须是脚本验证过的）")
        if group_demoted > 0:
            print(f"[check] 同算子分组：{group_demoted} 个 PASS 因组内有 FAIL 被降级为 FAIL")

    # 分流：用 set 去重文件路径（一文件多样本时只 move 一次）
    if split_dir:
        files_to_pass: Set[Path] = {Path(e["file"]) for e in report_passed}
        files_to_fail: Set[Path] = {Path(e["file"]) for e in report_failed}
        for f in files_to_pass:
            _split_sample(f, split_dir, "PASS", args.dry_run)
        for f in files_to_fail:
            _split_sample(f, split_dir, "FAIL", args.dry_run)
        # 更新报告里的 file 字段指向新路径
        pass_dir = split_dir / "pass"
        fail_dir = split_dir / "fail"
        for e in report_passed:
            e["file"] = str(pass_dir / Path(e["file"]).name)
        for e in report_failed:
            e["file"] = str(fail_dir / Path(e["file"]).name)
        for e in needs_review_entries:
            e["file"] = str(pass_dir / Path(e["file"]).name)

    # 只复制合格文件（不动原文件）。FAIL 文件不复制，只在下方报告里列出。
    if copy_dir:
        files_to_pass: Set[Path] = {Path(e["file"]) for e in report_passed}
        n_copied = 0
        for f in sorted(files_to_pass):
            _copy_passed(f, copy_base, copy_dir, args.dry_run)
            n_copied += 1
        tag = "（dry-run，未实际 copy）" if args.dry_run else ""
        print(f"[check] 已复制合格文件 {n_copied} 个 -> {copy_dir}{tag}")

    print()
    print(f"[check] 总结：PASS={n_pass}  FAIL={n_fail}（其中带 WARN 的 PASS={n_warn}）")
    if split_dir:
        tag = "（dry-run，未实际 move）" if args.dry_run else ""
        print(f"[check] 已分流：pass/ {n_pass} 个，fail/ {n_fail} 个{tag}")

    # 不合格文件清单（路径 + 数量）—— 供用户判断是否需要处理
    fail_files: List[str] = sorted({e["file"] for e in report_failed})
    if fail_files:
        print(f"[check] ✗ 不合格文件（{len(fail_files)} 个）:")
        for ff in fail_files:
            print(f"  - {ff}")
    else:
        print("[check] ✓ 无不合格文件")
    if needs_review_entries:
        print(f"[check] ⚠ 需人工复查的 PASS 样本（{len(needs_review_entries)} 个）:")
        for e in needs_review_entries:
            print(f"  - {e['file']}  ← {'; '.join(e['warnings'])}")

    # S-cross：跨子目录同算子重复（只报告，不 FAIL）——分组键带父目录后
    # 跨目录情形被隔离，这里显式回收该信号，防止训练集无声过采样
    cross_dups = _detect_cross_dir_duplicates(report_passed, report_failed)
    if cross_dups:
        print(f"[check] ⚠ 跨子目录重复算子（{len(cross_dups)} 个，训练集会过采样，建议人工去重）:")
        for d in cross_dups:
            print(f"  - {d['op']}  ×{d['n_dirs']} 个目录: "
                  f"{', '.join(p.split('/')[-1] for p in d['dirs'])}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # 报告里没有内部字段需要清理（_last_text 已废弃）
        def _strip(entry):
            return entry
        out_path.write_text(
            json.dumps(
                {"summary": {"pass": n_pass, "fail": n_fail, "warn_pass": n_warn,
                             "needs_manual_review": len(needs_review_entries),
                             "fail_files": fail_files,
                             "fail_file_count": len(fail_files),
                             "cross_dir_duplicates": cross_dups},
                 "passed": [_strip(e) for e in report_passed],
                 "failed": [_strip(e) for e in report_failed]},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[check] 详细报告 -> {out_path}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
