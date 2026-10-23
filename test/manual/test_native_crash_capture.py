"""Native smoke test: a forked child's worker faults while its main thread waits.

Needs GDB and a C compiler; no torch, CUDA, LK, or model is used.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).with_name("capture_native_crash.py")


def test_missing_command_starts_nothing(tmp_path):
    report = tmp_path / "report"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--output", str(report)],
                          capture_output=True, text=True)
    assert proc.returncode == 2
    assert "provide the original executable" in proc.stderr
    assert not report.exists()


@pytest.mark.skipif(not shutil.which("gdb") or not shutil.which("cc"),
                    reason="requires native GDB and C compiler")
def test_forked_native_worker_is_identified(tmp_path):
    source = tmp_path / "fault.c"
    source.write_text('''
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
__attribute__((noinline)) void *fault_in_worker(void *unused) {
    volatile int *ptr = (int *)0;
    *ptr = 42;
    return 0;
}
int main(void) {
    if (fork() == 0) {
        pthread_t worker;
        pthread_create(&worker, 0, fault_in_worker, 0);
        pthread_join(worker, 0);
        _exit(0);
    }
    wait(0);
    return 0;
}
''')
    binary = tmp_path / "fault"
    subprocess.run(["cc", "-O0", "-g", "-pthread", str(source), "-o", str(binary)], check=True)
    report = tmp_path / "report"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--output", str(report),
                           "--", str(binary)], capture_output=True, text=True, timeout=30)
    log = (report / "native-crash.log").read_text()
    assert proc.returncode == 139, log
    assert "fault_in_worker" in log and "=== END-NATIVE-FAULT ===" in log
    match = re.search(r"NATIVE-FAULT pid=(\d+) ptid=\((\d+), (\d+),", log)
    assert match, log
    pid, ptid_pid, lwp = map(int, match.groups())
    assert pid == ptid_pid and lwp != pid  # worker, not the Python/main thread
