"""Validate the bundled IDL with the installed Cyclone DDS ``idlc`` compiler."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from . import IDL_PATH


def main() -> None:
    compiler = shutil.which("idlc")
    if compiler is None:
        raise SystemExit("idlc was not found; install Cyclone DDS with its development tools")

    # Ubuntu's apt idlc exposes the C backend and writes output in cwd. Keep
    # validation artifacts out of the source tree. Python IdlStruct bindings
    # live in types.py because wheel backends are ABI/version-specific.
    with tempfile.TemporaryDirectory(prefix="rll-idlc-") as output_dir:
        result = subprocess.run(
            [compiler, str(IDL_PATH)], cwd=output_dir, check=False
        )
        if result.returncode:
            raise SystemExit(f"Cyclone DDS idlc failed with status {result.returncode}")
        if not (Path(output_dir) / f"{IDL_PATH.stem}.c").is_file():
            raise SystemExit("Cyclone DDS idlc completed without producing its C validation output")


if __name__ == "__main__":
    main()
