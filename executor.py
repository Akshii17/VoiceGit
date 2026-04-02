from __future__ import annotations

import shlex
import subprocess
from typing import List


def execute_commands(commands: List[str]) -> None:
    """
    Execute commands sequentially.

    - Prints each command before running it
    - Streams result per command (stdout/stderr)
    - Stops at the first failure (non-zero exit code)
    """
    for cmd in commands:
        print(f"\n$ {cmd}")

        try:
            proc = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            print(f"Error: command not found: {e}")
            return
        except Exception as e:
            print(f"Error: failed to execute command: {e}")
            return

        stdout = (proc.stdout or "").rstrip()
        stderr = (proc.stderr or "").rstrip()

        if stdout:
            print(stdout)
        if stderr:
            print(stderr)

        if proc.returncode != 0:
            print(f"Stopped: command failed (exit code {proc.returncode}).")
            return

