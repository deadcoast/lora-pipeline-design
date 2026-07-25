from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.process_control import (
    DESKTOP_OWNER_MUTEX_ENV,
    DesktopOwnerLifetime,
    run_managed_process,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_managed_process_returns_child_exit_code(tmp_path: Path) -> None:
    return_code = run_managed_process(
        [sys.executable, "-c", "raise SystemExit(7)"],
        cwd=tmp_path,
    )

    assert return_code == 7


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object behavior")
def test_windows_job_kills_trainer_when_lease_owner_is_hard_killed(
    tmp_path: Path,
) -> None:
    import ctypes
    from ctypes import wintypes

    pid_path = tmp_path / "trainer.pid"
    child_code = (
        "import os,sys,time;"
        "open(sys.argv[1],'w',encoding='utf-8').write(str(os.getpid()));"
        "time.sleep(120)"
    )
    helper_code = (
        "import sys;"
        "from pathlib import Path;"
        "from src.process_control import run_managed_process;"
        "raise SystemExit(run_managed_process("
        "[sys.executable,'-c',sys.argv[2],sys.argv[1]],cwd=Path(sys.argv[3])))"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            helper_code,
            str(pid_path),
            child_code,
            str(REPOSITORY),
        ],
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pid_path.is_file():
        if parent.poll() is not None:
            _stdout, stderr = parent.communicate()
            pytest.fail(f"Managed trainer parent exited early: {stderr}")
        time.sleep(0.05)
    if not pid_path.is_file():
        parent.kill()
        _stdout, stderr = parent.communicate()
        pytest.fail(f"Managed trainer did not start: {stderr}")
    child_pid = int(pid_path.read_text(encoding="utf-8"))

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    def is_active(pid: int) -> bool:
        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)

    assert is_active(child_pid)
    parent.kill()
    parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and is_active(child_pid):
        time.sleep(0.05)

    assert not is_active(child_pid)


@pytest.mark.skipif(os.name != "nt", reason="Windows owner-mutex behavior")
def test_closing_desktop_owner_kills_worker_and_trainer_tree(
    tmp_path: Path,
) -> None:
    import ctypes
    from ctypes import wintypes

    lifetime = DesktopOwnerLifetime()
    pid_path = tmp_path / "trainer.pid"
    child_code = (
        "import os,sys,time;"
        "open(sys.argv[1],'w',encoding='utf-8').write(str(os.getpid()));"
        "time.sleep(120)"
    )
    worker_code = (
        "import sys;"
        "from pathlib import Path;"
        "from src.process_control import "
        "arm_desktop_owner_guard,run_managed_process;"
        "arm_desktop_owner_guard();"
        "raise SystemExit(run_managed_process("
        "[sys.executable,'-c',sys.argv[2],sys.argv[1]],cwd=Path(sys.argv[3])))"
    )
    environment = dict(os.environ)
    environment[DESKTOP_OWNER_MUTEX_ENV] = lifetime.name
    worker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            worker_code,
            str(pid_path),
            child_code,
            str(REPOSITORY),
        ],
        cwd=REPOSITORY,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pid_path.is_file():
        if worker.poll() is not None:
            _stdout, stderr = worker.communicate()
            lifetime.close()
            pytest.fail(f"Pipeline worker exited early: {stderr}")
        time.sleep(0.05)
    if not pid_path.is_file():
        lifetime.close()
        worker.kill()
        _stdout, stderr = worker.communicate()
        pytest.fail(f"Managed trainer did not start: {stderr}")
    trainer_pid = int(pid_path.read_text(encoding="utf-8"))

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    def is_active(pid: int) -> bool:
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            return bool(
                kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                and exit_code.value == 259
            )
        finally:
            kernel32.CloseHandle(handle)

    assert is_active(trainer_pid)
    lifetime.close()
    worker.wait(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and is_active(trainer_pid):
        time.sleep(0.05)

    assert not is_active(trainer_pid)
