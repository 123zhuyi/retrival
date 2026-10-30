#!/usr/bin/env python3
from _common import validate


if __name__ == "__main__":
    errors = validate()
    for error in errors:
        print(f"ERROR: {error}")
    if not errors:
        print("OK: KernelWiki validation passed")
    raise SystemExit(1 if errors else 0)
