from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_local_packages() -> None:
    project_root = Path(__file__).resolve().parents[1]
    mpl_config_dir = project_root / ".cache" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return

    packages_dir = project_root / ".packages"
    if packages_dir.exists():
        packages_path = str(packages_dir)
        if packages_path not in sys.path:
            sys.path.insert(0, packages_path)
