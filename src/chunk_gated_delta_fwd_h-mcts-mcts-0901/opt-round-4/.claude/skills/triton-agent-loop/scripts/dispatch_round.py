#!/usr/bin/env python3
"""
dispatch_round.py — 启动一个独立的子 Claude Code CLI 执行单个 round。

用法:
    python3 dispatch_round.py \
        --manifest triton_ascend_output/{op_name}-{algorithm}-{run_tag}/.task_manifest_{N}.json \
        [--timeout 7200] \
        [--command-template 'claude -p --plugin-dir {plugin_dir} --system-prompt-file {sub_reference} --allowed-tools Bash,Read,Write,Edit,Glob,Grep "Execute round worker. Manifest: {manifest_path}"']

环境变量:
    TRITON_WORKER_COMMAND_TEMPLATE: 覆盖默认命令模板
    TRITON_WORKER_TIMEOUT_SECONDS:  覆盖默认超时
    TRITON_NPU_DEVICE_ID:           显式指定要使用的 NPU 设备 ID（需空闲）
    TRITON_SKIP_NPU_IDLE_CHECK:     设置为 1/true/yes 时跳过空闲卡检测
    TRITON_NPU_IDLE_MAX_AICORE:     AICore 空闲阈值（默认 1%）
    TRITON_NPU_IDLE_MAX_AIVECTOR:   AIVector 空闲阈值（默认 1%）
    TRITON_NPU_IDLE_MAX_NPU_UTIL:   NPU 利用率空闲阈值（默认 1%）
    TRITON_NPU_IDLE_MAX_HBM:        HBM 使用空闲阈值（默认 10%）

输出 (JSON):
    {
        "status": "success" | "timeout" | "error" | "no_result",
        "round_result": { ... },
        "stdout": "...",
        "stderr": "...",
        "returncode": 0,
        "pid": 12345
    }
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from log_parser import StreamJsonReadableParser
import check_npu_idle


# Sub-directories of .claude/ that the worker needs locally so it can run
# with cwd=work_dir without requiring write access to the plugin root.
CLAUDE_RESOURCE_DIRS = ("skills", "template", "references")


DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("TRITON_WORKER_TIMEOUT_SECONDS", "7200"))

DEFAULT_COMMAND_TEMPLATE = (
    'claude -p '
    '--permission-mode auto '
    '--output-format stream-json '
    '--include-partial-messages '
    '--include-hook-events '
    '--verbose '
    '--system-prompt-file {sub_reference} '
    '"Execute triton-optimizer-multi-agent round worker. Read manifest: {manifest_path}" '
    '--allowed-tools Bash Read Write Edit Glob Grep'
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}")


def resolve_plugin_dir() -> Path:
    """Resolve plugin root relative to this script."""
    script_path = Path(__file__).resolve()
    # skills/triton-agent-loop/scripts/dispatch_round.py -> plugin root
    return script_path.parents[3]


def resolve_knowledge_base_root(
    plugin_dir: Path,
    claude_root: Path | None = None,
) -> Path | None:
    """Return the real directory of the shared triton-knowledge-retrieval skill.

    The skill may be installed as a symlink under ``claude_root/skills/`` or
    ``plugin_dir/.claude/skills/``, or it may live in the repository-level
    ``ops/`` directory.  We follow symlinks to find the real path so the worker
    can share the same knowledge base across tasks.
    """
    candidates: list[Path] = []
    if claude_root is not None:
        candidates.append(claude_root / "skills" / "triton-knowledge-retrieval")
    candidates.append(plugin_dir / ".claude" / "skills" / "triton-knowledge-retrieval")

    for candidate in candidates:
        if candidate.is_symlink():
            real = candidate.resolve()
            if real.is_dir():
                return real
        if candidate.is_dir():
            return candidate

    # Fallback: look in the repository-level ops/ directory
    repo_root = plugin_dir.parents[1]
    ops_candidate = repo_root / "ops" / "triton-knowledge-retrieval"
    if ops_candidate.is_dir():
        return ops_candidate

    # Search upward for ops/
    search_dir = plugin_dir
    for _ in range(5):
        search_dir = search_dir.parent
        ops_candidate = search_dir / "ops" / "triton-knowledge-retrieval"
        if ops_candidate.is_dir():
            return ops_candidate

    return None


def resolve_claude_root(plugin_dir: Path, work_dir: Path) -> Path:
    """Find the installed ``.claude`` resource root used by this project.

    ``init.sh`` may install resources into the plugin root (self-install) or
    into the target project directory.  The worker needs that resource tree, so
    we look for a complete installation near ``work_dir`` first, then fall back
    to the plugin root.
    """
    candidates: list[Path] = []

    # 1) Walk up from work_dir (covers both project-directory and self-install).
    resolved_work = work_dir.resolve()
    for parent in resolved_work.parents:
        candidates.append(parent / ".claude")

    # 2) Current working directory (e.g. when work_dir is a temporary path).
    candidates.append(Path.cwd() / ".claude")

    # 3) Plugin root as final fallback.
    candidates.append(plugin_dir / ".claude")

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "references").exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find installed .claude resources. "
        f"Tried candidates: {candidates}. Please run init.sh in the project directory first."
    )


def prepare_isolated_claude(work_dir: Path, plugin_dir: Path) -> Path:
    """Copy installed ``.claude`` resources into ``work_dir/.claude``.

    The worker runs with ``cwd=work_dir``.  Giving it a local copy of the
    ``.claude/`` subtree keeps it isolated: it does not need write access to
    the plugin root and any accidental modifications only affect the copy.

    One exception: ``triton-knowledge-retrieval`` is symlinked instead of
    copied because its ``knowledge/cache/``, ``log.md`` and
    ``knowledge/raw/experiences/`` are designed to accumulate across tasks.
    """
    src_claude = resolve_claude_root(plugin_dir, work_dir)
    dst_claude = work_dir / ".claude"
    dst_claude.mkdir(parents=True, exist_ok=True)

    for name in CLAUDE_RESOURCE_DIRS:
        src = src_claude / name
        if not src.exists():
            continue
        dst = dst_claude / name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=False)

    # Knowledge base is the one exception to isolation: its cache, log.md,
    # and raw experiences must accumulate across tasks.  Replace the copied
    # directory with a symlink to the real shared knowledge base.
    kkb_dst = dst_claude / "skills" / "triton-knowledge-retrieval"
    kkb_src = resolve_knowledge_base_root(plugin_dir, claude_root=src_claude)
    if kkb_src is not None:
        if kkb_dst.exists() or kkb_dst.is_symlink():
            shutil.rmtree(kkb_dst, ignore_errors=True)
        kkb_dst.parent.mkdir(parents=True, exist_ok=True)
        kkb_dst.symlink_to(kkb_src.resolve(), target_is_directory=True)
    else:
        # Warn but do not fail: the worker can still run without KB sharing.
        print(
            "WARN: triton-knowledge-retrieval not found; "
            "knowledge base will not be shared across tasks",
            file=sys.stderr,
        )

    settings_src = src_claude / "settings.json"
    if settings_src.is_file():
        shutil.copy2(settings_src, dst_claude / "settings.json")

    return dst_claude


def build_command(
    manifest_path: Path,
    work_dir: Path,
    command_template: str | None = None,
) -> str:
    plugin_dir = resolve_plugin_dir()

    # Read the manifest to find the algorithm-specific reference document.
    manifest = load_json(manifest_path)
    reference_path = manifest.get("algorithm_reference") or manifest.get("worker_reference")
    if not reference_path:
        reference_path = ".claude/references/algorithms/naive/round.md"

    # Prefer the isolated copy so the worker's reference files live inside
    # work_dir.  Fall back to the installed .claude root before
    # prepare_isolated_claude has run (e.g. custom command templates).
    sub_reference = work_dir / reference_path
    if not sub_reference.is_file():
        try:
            claude_root = resolve_claude_root(plugin_dir, work_dir)
            # reference_path starts with ".claude/"; claude_root is already the
            # .claude directory, so strip the leading ".claude/" segment.
            relative_reference = Path(reference_path).relative_to(".claude")
            sub_reference = claude_root / relative_reference
        except (FileNotFoundError, ValueError):
            sub_reference = plugin_dir / reference_path

    if not sub_reference.is_file():
        raise FileNotFoundError(
            f"Algorithm reference not found: {reference_path} "
            f"(tried work_dir={work_dir / reference_path}, "
            f"claude_root={claude_root if 'claude_root' in locals() else 'n/a'}, "
            f"plugin_dir={plugin_dir / reference_path})"
        )

    template = command_template or os.environ.get(
        "TRITON_WORKER_COMMAND_TEMPLATE", DEFAULT_COMMAND_TEMPLATE
    )

    # Support shell-style quoting in template by using shell=True
    cmd = template.format(
        manifest_path=str(manifest_path),
        work_dir=str(work_dir),
        plugin_dir=str(plugin_dir),
        reference_dir=str(work_dir / ".claude" / "references"),
        sub_reference=str(sub_reference),
    )
    return cmd


def _stream_subprocess(
    proc: subprocess.Popen[str],
    work_dir: Path,
    cmd: str,
    timeout: int | None,
) -> tuple[int, str, str, bool]:
    """Stream sub CLI stdout/stderr to disk while the round is running.

    Writes the raw stream to ``work_dir/sub_agent.log`` and a human-readable
    summary to ``work_dir/sub_agent_readable.log``.  This lets operators tail
    the round progress without waiting for the sub CLI to finish.
    """
    log_path = work_dir / "sub_agent.log"
    readable_log_path = work_dir / "sub_agent_readable.log"
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    timed_out = False

    def reader(stream, prefix: str, chunks: list[str], log_f, parser) -> None:
        for line in iter(stream.readline, ""):
            chunks.append(line)
            formatted = f"{prefix}{line}"
            log_f.write(formatted)
            log_f.flush()
            if parser is not None:
                parser.feed_line(formatted)

    with open(log_path, "w", encoding="utf-8") as log_f, \
         open(readable_log_path, "w", encoding="utf-8") as readable_f:
        log_f.write(f"# Command: {cmd}\n")
        log_f.write(f"# CWD: {work_dir}\n")
        log_f.write(f"# Timeout: {timeout}s\n")
        log_f.write("-" * 80 + "\n")
        log_f.flush()

        parser = StreamJsonReadableParser(readable_f)
        t_out = threading.Thread(
            target=reader,
            args=(proc.stdout, "[sub-agent stdout] ", stdout_chunks, log_f, parser),
            daemon=True,
        )
        t_err = threading.Thread(
            target=reader,
            args=(proc.stderr, "[sub-agent stderr] ", stderr_chunks, log_f, parser),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            rc = proc.wait()

        t_out.join()
        t_err.join()
        parser.flush()

        log_f.write("\n" + "-" * 80 + "\n")
        log_f.write(f"# Exit code: {rc}  timed_out={timed_out}\n")
        log_f.flush()

    return rc, "".join(stdout_chunks), "".join(stderr_chunks), timed_out


@contextmanager
def _child_process_scope(proc: subprocess.Popen[str]):
    """Forward SIGINT/SIGTERM to the child and ensure it is reaped.

    This reduces the chance that a sub-agent keeps running as an orphan after
    the main orchestrator is stopped.
    """
    old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    old_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def _forward(signum: int, frame: Any) -> None:
        try:
            if proc.poll() is None:
                proc.send_signal(signum)
        except ProcessLookupError:
            pass

    def _cleanup() -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass

    try:
        signal.signal(signal.SIGINT, _forward)
        signal.signal(signal.SIGTERM, _forward)
        yield
    finally:
        _cleanup()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def wait_for_round_result(work_dir: Path, timeout: float, interval: float = 2.0) -> Path | None:
    """Poll for round_result.json until it appears or timeout."""
    result_path = work_dir / "round_result.json"
    elapsed = 0.0
    while elapsed < timeout:
        if result_path.is_file():
            return result_path
        time.sleep(interval)
        elapsed += interval
    return None


def dispatch_round(
    manifest_path: Path,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    command_template: str | None = None,
) -> dict[str, Any]:
    """Spawn a sub CLI to execute the round and return the parsed result.

    Before launching, installed ``.claude`` resources (skills, template,
    references, settings.json) are copied into ``work_dir/.claude``.  The sub
    CLI then runs with ``cwd=work_dir`` so the worker resolves all
    ``.claude/...`` paths locally and does not need write access to the plugin
    root.
    """
    manifest = load_json(manifest_path)
    plugin_dir = resolve_plugin_dir()
    work_dir = Path(manifest.get("work_dir", manifest_path.parent)).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    prepare_isolated_claude(work_dir, plugin_dir)

    result: dict[str, Any] = {
        "status": "error",
        "round_result": None,
        "stdout": "",
        "stderr": "",
        "returncode": None,
        "pid": None,
    }

    cmd = build_command(manifest_path, work_dir, command_template)

    env = os.environ.copy()
    env["TRITON_OPTIMIZER_MODE"] = "worker"

    # ── Idle NPU selection (must run before launching the worker) ──
    if not check_npu_idle.should_skip_idle_check():
        try:
            device_info = check_npu_idle.select_idle_device(
                preferred_id=check_npu_idle.preferred_device_id()
            )
        except check_npu_idle.NpuIdleCheckError as exc:
            result["status"] = "error"
            result["stderr"] = (
                f"NPU idle check failed: {exc}\n"
                f"details: {json.dumps(exc.details, indent=2, ensure_ascii=False)}"
            )
            return result

        device_id = device_info["device_id"]
        env["ASCEND_VISIBLE_DEVICES"] = str(device_id)
        env["NPU_CALCULATE_DEVICE"] = str(device_id)
        # Some CANN / torch_npu versions honour ASCEND_RT_VISIBLE_DEVICES.
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)

        try:
            (work_dir / "npu_device.json").write_text(
                json.dumps(device_info, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    except OSError as exc:
        result["stderr"] = f"failed to start sub CLI: {exc}"
        return result

    result["pid"] = proc.pid

    with _child_process_scope(proc):
        returncode, stdout, stderr, timed_out = _stream_subprocess(proc, work_dir, cmd, timeout)

    result["stdout"] = stdout
    result["stderr"] = stderr
    result["returncode"] = returncode

    if timed_out:
        result["status"] = "timeout"
        return result

    # Wait a bit for round_result.json flush
    result_path = wait_for_round_result(work_dir, timeout=30.0)
    if result_path is None:
        # Fallback: search for round_result.json in work_dir
        candidates = list(work_dir.glob("round_result.json"))
        if candidates:
            result_path = candidates[0]

    if result_path is None or not result_path.is_file():
        result["status"] = "no_result"
        return result

    try:
        round_result = load_json(result_path)
        result["round_result"] = round_result
        result["status"] = "success"
    except ValueError as exc:
        result["status"] = "error"
        result["stderr"] += f"\nfailed to parse round_result.json: {exc}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch a per-round worker sub CLI")
    parser.add_argument("--manifest", required=True, help="Path to task_manifest.json")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Sub CLI timeout in seconds (default: 7200)")
    parser.add_argument("--command-template", default=None, help="Shell command template for launching sub CLI")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_file():
        print(json.dumps({
            "status": "error",
            "stderr": f"manifest not found: {manifest_path}",
        }))
        return 1

    result = dispatch_round(manifest_path, args.timeout, args.command_template)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
