#!/usr/bin/env python3
"""
check_npu_idle.py — Detect idle Ascend NPU devices before dispatching a worker.

Usage:
    python3 check_npu_idle.py [--json] [--preferred-id 0]

Primary configuration is in the plugin-root `config.json`:

    {
      "npu_idle": {
        "aicore": 1.0,
        "aivector": 1.0,
        "npu_util": 1.0,
        "hbm": 10.0
      },
      "npu_device_id": null
    }

Environment variables (optional overrides):
    TRITON_SKIP_NPU_IDLE_CHECK=1   Bypass the idle check entirely.
    TRITON_NPU_DEVICE_ID=<id>      Prefer a specific NPU device ID.
    TRITON_NPU_IDLE_MAX_AICORE     AICore usage threshold.
    TRITON_NPU_IDLE_MAX_AIVECTOR   AIVector usage threshold.
    TRITON_NPU_IDLE_MAX_NPU_UTIL   NPU utilization threshold.
    TRITON_NPU_IDLE_MAX_HBM        HBM usage threshold.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


def _plugin_config() -> dict[str, Any]:
    """Load plugin-root config.json if it exists; otherwise return empty dict."""
    try:
        plugin_root = Path(__file__).resolve().parents[3]
        config_path = plugin_root / "config.json"
        if config_path.is_file():
            return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_default_thresholds() -> dict[str, float]:
    """Load thresholds: defaults < config.json < environment variables."""
    cfg = _plugin_config()
    idle_cfg = cfg.get("npu_idle", {})

    thresholds = {
        "aicore": float(idle_cfg.get("aicore", 1.0)),
        "aivector": float(idle_cfg.get("aivector", 1.0)),
        "npu_util": float(idle_cfg.get("npu_util", 1.0)),
        "hbm": float(idle_cfg.get("hbm", 10.0)),
    }

    env_map = {
        "aicore": "TRITON_NPU_IDLE_MAX_AICORE",
        "aivector": "TRITON_NPU_IDLE_MAX_AIVECTOR",
        "npu_util": "TRITON_NPU_IDLE_MAX_NPU_UTIL",
        "hbm": "TRITON_NPU_IDLE_MAX_HBM",
    }
    for key, env_var in env_map.items():
        raw = os.environ.get(env_var)
        if raw:
            try:
                thresholds[key] = float(raw)
            except ValueError:
                pass

    return thresholds


DEFAULT_THRESHOLDS = _load_default_thresholds()


class NpuIdleCheckError(Exception):
    """Raised when the idle check cannot be satisfied."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass
class NpuDevice:
    npu_id: int
    chip_id: int
    chip_name: str = "unknown"
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def global_id(self) -> int:
        """Global chip identifier used by some runtimes."""
        return self.npu_id * 2 + self.chip_id

    def is_idle(self, thresholds: Optional[dict[str, float]] = None) -> tuple[bool, str]:
        """Return (idle, reason) for this device."""
        th = thresholds or DEFAULT_THRESHOLDS
        m = self.metrics

        # If we cannot read usage at all, treat as not idle to stay safe.
        if not m:
            return False, "usage metrics unavailable"

        aicore = m.get("aicore")
        if aicore is not None and aicore > th["aicore"]:
            return False, f"AICore usage {aicore:.1f}% > {th['aicore']}%"

        aivector = m.get("aivector")
        if aivector is not None and aivector > th["aivector"]:
            return False, f"AIVector usage {aivector:.1f}% > {th['aivector']}%"

        npu_util = m.get("npu_util")
        if npu_util is not None and npu_util > th["npu_util"]:
            return False, f"NPU utilization {npu_util:.1f}% > {th['npu_util']}%"

        hbm = m.get("hbm")
        if hbm is not None and hbm > th["hbm"]:
            return False, f"HBM usage {hbm:.1f}% > {th['hbm']}%"

        return True, ""


def _run_cmd(cmd: list[str], timeout: int = 10) -> tuple[str, int, str]:
    """Run a command and return (stdout, returncode, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.returncode, result.stderr
    except FileNotFoundError:
        return "", -1, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired as exc:
        return "", -1, f"Command timed out: {exc}"


def _parse_kv(output: str, key_pattern: str) -> Optional[str]:
    """Extract the first value whose key starts with key_pattern."""
    for line in output.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key_part, val_part = line.split(":", 1)
        if key_part.strip().startswith(key_pattern):
            return val_part.strip()
    return None


def _parse_float_or_none(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    cleaned = raw.replace("%", "").replace(",", "").strip()
    if cleaned in ("-", "N/A", ""):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def npu_smi_available() -> bool:
    """Check whether npu-smi is installed."""
    stdout, rc, _ = _run_cmd(["npu-smi", "-v"])
    return rc == 0 and bool(stdout.strip())


def discover_npu_devices() -> list[NpuDevice]:
    """Enumerate compute-capable NPU chips from npu-smi info -m."""
    stdout, rc, stderr = _run_cmd(["npu-smi", "info", "-m"])
    if rc != 0:
        raise NpuIdleCheckError(
            f"npu-smi info -m failed: {stderr or stdout}",
            {"rc": rc},
        )

    lines = [line.rstrip() for line in stdout.splitlines() if line.strip()]
    if not lines or "NPU ID" not in lines[0]:
        raise NpuIdleCheckError(
            "unexpected npu-smi info -m output",
            {"raw": stdout[:500]},
        )

    # Parse the fixed-width header by splitting on two or more spaces so that
    # multi-word column names (e.g. "Chip Logic ID", "Chip Name") stay intact.
    header = [col.strip() for col in re.split(r"\s{2,}", lines[0].strip())]
    try:
        idx_npu = header.index("NPU ID")
        idx_chip = header.index("Chip ID")
        idx_logic = header.index("Chip Logic ID")
        idx_name = header.index("Chip Name")
    except ValueError:
        # Fallback for older formats with exactly 4 columns.
        idx_npu, idx_chip, idx_logic, idx_name = 0, 1, 2, 3

    devices: list[NpuDevice] = []
    for line in lines[1:]:
        parts = [col.strip() for col in re.split(r"\s{2,}", line.strip())]
        if len(parts) <= max(idx_npu, idx_chip, idx_logic, idx_name):
            continue
        try:
            npu_id = int(parts[idx_npu])
            chip_id = int(parts[idx_chip])
        except ValueError:
            continue
        logic_id = parts[idx_logic]
        # Skip MCU chips: logic ID is "-".
        if logic_id == "-":
            continue
        chip_name = parts[idx_name]
        devices.append(NpuDevice(npu_id=npu_id, chip_id=chip_id, chip_name=chip_name))

    if not devices:
        raise NpuIdleCheckError(
            "no compute-capable NPU devices found",
            {"raw": stdout[:500]},
        )

    return devices


def fetch_device_usage(device: NpuDevice) -> None:
    """Populate device.metrics from npu-smi info -t usages -i <npu_id>."""
    stdout, rc, stderr = _run_cmd(
        ["npu-smi", "info", "-t", "usages", "-i", str(device.npu_id)]
    )
    if rc != 0:
        device.warnings.append(
            f"usages query failed for NPU {device.npu_id}: {stderr or stdout}"
        )
        return

    device.metrics["aicore"] = _parse_float_or_none(
        _parse_kv(stdout, "Aicore Usage Rate")
    )
    device.metrics["aivector"] = _parse_float_or_none(
        _parse_kv(stdout, "Aivector Usage Rate")
    )
    device.metrics["npu_util"] = _parse_float_or_none(
        _parse_kv(stdout, "NPU Utilization")
    )
    device.metrics["hbm"] = _parse_float_or_none(
        _parse_kv(stdout, "HBM Usage Rate")
    )

    # Drop None entries so callers see only successfully read metrics.
    device.metrics = {k: v for k, v in device.metrics.items() if v is not None}

    if not device.metrics:
        device.warnings.append(
            f"no usage metrics parsed for NPU {device.npu_id}"
        )


def select_idle_device(
    preferred_id: Optional[int] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Select an idle NPU device.

    If preferred_id is provided and that device is idle, it is returned.
    Otherwise the first idle device from npu-smi enumeration is returned.
    Raises NpuIdleCheckError if no idle device is available.
    """
    devices = discover_npu_devices()
    for device in devices:
        fetch_device_usage(device)

    idle_devices: list[dict[str, Any]] = []
    busy_devices: list[dict[str, Any]] = []
    for device in devices:
        idle, reason = device.is_idle(thresholds)
        info = {
            "npu_id": device.npu_id,
            "chip_id": device.chip_id,
            "global_id": device.global_id,
            "chip_name": device.chip_name,
            "metrics": device.metrics,
            "warnings": device.warnings,
        }
        if idle:
            idle_devices.append(info)
        else:
            info["busy_reason"] = reason
            busy_devices.append(info)

    selected: Optional[dict[str, Any]] = None
    selected_from = "auto"

    if preferred_id is not None:
        for info in idle_devices:
            if info["npu_id"] == preferred_id:
                selected = info
                selected_from = "preferred"
                break
        if selected is None:
            raise NpuIdleCheckError(
                f"preferred NPU device {preferred_id} is not idle or not available",
                {
                    "preferred_id": preferred_id,
                    "idle_devices": idle_devices,
                    "busy_devices": busy_devices,
                },
            )

    if selected is None and idle_devices:
        selected = idle_devices[0]

    if selected is None:
        raise NpuIdleCheckError(
            "no idle NPU device available",
            {
                "idle_devices": idle_devices,
                "busy_devices": busy_devices,
                "thresholds": thresholds or DEFAULT_THRESHOLDS,
            },
        )

    return {
        "device_id": selected["npu_id"],
        "chip_id": selected["chip_id"],
        "global_id": selected["global_id"],
        "chip_name": selected["chip_name"],
        "metrics": selected["metrics"],
        "selected_from": selected_from,
        "idle_devices": idle_devices,
        "busy_devices": busy_devices,
    }


def should_skip_idle_check() -> bool:
    """Return True if the user explicitly disabled the idle check."""
    return os.environ.get("TRITON_SKIP_NPU_IDLE_CHECK", "").lower() in (
        "1",
        "true",
        "yes",
    )


def preferred_device_id() -> Optional[int]:
    """Read preferred device ID: environment variable overrides config.json."""
    raw = os.environ.get("TRITON_NPU_DEVICE_ID", "")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass

    cfg = _plugin_config()
    device_id = cfg.get("npu_device_id")
    if device_id is not None:
        try:
            return int(device_id)
        except ValueError:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check for an idle Ascend NPU")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument(
        "--preferred-id", type=int, default=None, help="Preferred NPU device ID"
    )
    args = parser.parse_args()

    if should_skip_idle_check():
        result = {
            "skipped": True,
            "reason": "TRITON_SKIP_NPU_IDLE_CHECK is set",
        }
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("NPU idle check skipped (TRITON_SKIP_NPU_IDLE_CHECK is set)")
        return 0

    if not npu_smi_available():
        message = "npu-smi not available; cannot verify NPU idle status"
        if args.json:
            print(json.dumps({"error": message}, indent=2, ensure_ascii=False))
        else:
            print(message, file=sys.stderr)
        return 1

    try:
        result = select_idle_device(
            preferred_id=args.preferred_id,
            thresholds=DEFAULT_THRESHOLDS,
        )
    except NpuIdleCheckError as exc:
        if args.json:
            print(
                json.dumps(
                    {"error": str(exc), "details": exc.details},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
            for key, value in exc.details.items():
                print(f"  {key}: {value}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(
            f"Idle NPU selected: NPU {result['device_id']} "
            f"({result['chip_name']}, metrics={result['metrics']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
