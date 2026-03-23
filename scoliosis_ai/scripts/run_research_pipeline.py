"""
Run a lightweight research pipeline: reproducibility + analysis + benchmark.
"""

import subprocess
import sys


def run(cmd):
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main():
    cmds = [
        [sys.executable, "scripts/reproducibility_pack.py"],
        [sys.executable, "run_data_science.py"],
        [sys.executable, "scripts/benchmark_suite.py"],
    ]

    for cmd in cmds:
        code = run(cmd)
        if code != 0:
            return code

    print("Research pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
