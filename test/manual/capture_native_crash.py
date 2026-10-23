#!/usr/bin/env python3
"""Run the original server command under GDB, following scheduler children.

Example (keep every original serve argument/environment variable):
  python test/manual/capture_native_crash.py --output /path/to/new-report -- \
    /path/to/env/bin/python -m sglang.launch_server <original arguments>

For a shell launcher, explicitly supply /bin/bash before its script path.
This is a diagnostic launch, not a production or performance run.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New report directory; existing paths are never overwritten")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide the original executable and arguments after --")
    gdb = shutil.which("gdb")
    if not gdb:
        parser.error("gdb is required on the server; no server process was started")
    script = Path(__file__).parent / "native_crash/collect.gdb"
    if not script.is_file():
        parser.error(f"missing companion script: {script}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2) + "\n")
    env = os.environ.copy()
    env["LK_NATIVE_TRACE"] = "1"
    # No CUDA_LAUNCH_BLOCKING change: first capture preserves existing timing.
    argv = [gdb, "-nx", "--batch", "-x", str(script.resolve()), "--args", *command]
    print(f"Native report: {args.output.resolve() / 'native-crash.log'}", flush=True)
    with (args.output / "native-crash.log").open("wb") as log:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        try:
            while data := proc.stdout.read1(65536):
                log.write(data)
                log.flush()
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            return proc.wait()
        except KeyboardInterrupt:
            # GDB receives terminal SIGINT too. Let its batch shutdown release
            # the traced children; do not detach a half-initialized server.
            proc.terminate()
            return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
