#!/usr/bin/env python3
"""Helper to migrate old config.json to new structure (same shape)."""
import json
import sys
from pathlib import Path


def main(old_path: str, new_path: str) -> None:
    old = Path(old_path)
    new = Path(new_path)
    if not old.exists():
        print(f"No old config at {old_path}")
        return
    data = json.loads(old.read_text())
    new.parent.mkdir(parents=True, exist_ok=True)
    new.write_text(json.dumps(data, indent=4))
    print(f"Migrated {old_path} -> {new_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
