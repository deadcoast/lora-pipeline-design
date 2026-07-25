"""Child-process ownership helpers for long-running trainer workloads.

On Windows, a child process normally survives when its console parent is
forcibly terminated. Training must not keep consuming the GPU after MIVIM's
run lease disappears, so managed children are placed in a kill-on-close Job
Object. Windows then terminates the complete child tree if MIVIM exits, even
when normal Python cleanup cannot run.
"""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class ManagedProcessError(RuntimeError):
    """Raised when MIVIM cannot establish safe child-process ownership."""


DESKTOP_OWNER_MUTEX_ENV = "MIVIM_DESKTOP_OWNER_MUTEX"
_DESKTOP_OWNER_MUTEX_PREFIX = "Local\\MIVIM-LoRA-Studio-Owner-"
_desktop_owner_guard_armed = False


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
    _EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    _INFINITE = 0xFFFFFFFF
    _WAIT_FAILED = 0xFFFFFFFF
    _WAIT_OBJECT_0 = 0
    _WAIT_ABANDONED = 0x00000080
    _SYNCHRONIZE = 0x00100000
    _ERROR_ALREADY_EXISTS = 183

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _StartupInfoW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class _StartupInfoExW(ctypes.Structure):
        _fields_ = [
            ("StartupInfo", _StartupInfoW),
            ("lpAttributeList", ctypes.c_void_p),
        ]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _KERNEL32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
    _KERNEL32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _KERNEL32.UpdateProcThreadAttribute.restype = wintypes.BOOL
    _KERNEL32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    _KERNEL32.DeleteProcThreadAttributeList.restype = None
    _KERNEL32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(_StartupInfoW),
        ctypes.POINTER(_ProcessInformation),
    ]
    _KERNEL32.CreateProcessW.restype = wintypes.BOOL
    _KERNEL32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _KERNEL32.WaitForSingleObject.restype = wintypes.DWORD
    _KERNEL32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _KERNEL32.GetExitCodeProcess.restype = wintypes.BOOL
    _KERNEL32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    _KERNEL32.CreateMutexW.restype = wintypes.HANDLE
    _KERNEL32.OpenMutexW.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    _KERNEL32.OpenMutexW.restype = wintypes.HANDLE
    _KERNEL32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    _KERNEL32.ReleaseMutex.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL


class DesktopOwnerLifetime:
    """A Windows mutex that makes pipeline workers die with the Studio UI."""

    def __init__(self) -> None:
        self.name = ""
        self._handle: Any = None
        if os.name != "nt":
            return
        self.name = f"{_DESKTOP_OWNER_MUTEX_PREFIX}{uuid.uuid4()}"
        handle = _KERNEL32.CreateMutexW(None, True, self.name)
        error_code = ctypes.get_last_error()
        if not handle or error_code == _ERROR_ALREADY_EXISTS:
            if handle:
                _KERNEL32.CloseHandle(handle)
            raise ManagedProcessError(
                "Windows could not establish Studio-to-worker crash cleanup: "
                f"{ctypes.WinError(error_code)}"
            )
        self._handle = handle

    def close(self) -> None:
        if os.name == "nt" and self._handle is not None:
            # A normal close signals workers immediately. A hard process kill
            # abandons the owned mutex, which wakes them through the same path.
            _KERNEL32.ReleaseMutex(self._handle)
            _KERNEL32.CloseHandle(self._handle)
            self._handle = None


def arm_desktop_owner_guard(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Exit this worker immediately when its owning Studio process disappears."""

    global _desktop_owner_guard_armed
    environment = os.environ if environ is None else environ
    name = environment.get(DESKTOP_OWNER_MUTEX_ENV, "")
    if not name or _desktop_owner_guard_armed:
        return
    try:
        uuid.UUID(name.removeprefix(_DESKTOP_OWNER_MUTEX_PREFIX))
    except ValueError as exc:
        raise ManagedProcessError(
            "The Studio worker ownership token is invalid."
        ) from exc
    if not name.startswith(_DESKTOP_OWNER_MUTEX_PREFIX) or os.name != "nt":
        raise ManagedProcessError(
            "The Studio worker ownership token is unsupported on this platform."
        )
    handle = _KERNEL32.OpenMutexW(_SYNCHRONIZE, False, name)
    if not handle:
        raise ManagedProcessError(
            "The Studio that started this pipeline is no longer running. "
            "No training work was started."
        )
    _desktop_owner_guard_armed = True

    def watch_owner() -> None:
        result = _KERNEL32.WaitForSingleObject(handle, _INFINITE)
        _KERNEL32.CloseHandle(handle)
        if result in {_WAIT_OBJECT_0, _WAIT_ABANDONED, _WAIT_FAILED}:
            # os._exit is deliberate: it closes the worker-owned trainer Job
            # Object even if a Python thread is blocked in native training I/O.
            os._exit(70)

    threading.Thread(
        target=watch_owner,
        name="mivim-desktop-owner",
        daemon=True,
    ).start()


class _WindowsKillJob:
    """A kill-on-close Windows Job Object owned by the current process."""

    def __init__(self) -> None:
        self._handle: Any = None
        if os.name != "nt":
            return
        handle = _KERNEL32.CreateJobObjectW(None, None)
        if not handle:
            raise ManagedProcessError(
                f"Windows could not create a trainer safety job: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        information = _JobObjectExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not _KERNEL32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            _KERNEL32.CloseHandle(handle)
            raise ManagedProcessError(
                f"Windows could not enable trainer crash cleanup: {error}"
            )
        self._handle = handle

    def run(self, command: Sequence[str], *, cwd: Path) -> int:
        """Create a child inside the job before its first instruction runs."""

        if os.name != "nt" or self._handle is None:
            raise ManagedProcessError("Windows crash cleanup is unavailable.")

        attribute_size = ctypes.c_size_t()
        _KERNEL32.InitializeProcThreadAttributeList(
            None,
            1,
            0,
            ctypes.byref(attribute_size),
        )
        if attribute_size.value == 0:
            raise ManagedProcessError(
                "Windows could not size the trainer crash-cleanup attributes: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )
        attribute_buffer = ctypes.create_string_buffer(attribute_size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not _KERNEL32.InitializeProcThreadAttributeList(
            attribute_list,
            1,
            0,
            ctypes.byref(attribute_size),
        ):
            raise ManagedProcessError(
                "Windows could not initialize trainer crash cleanup: "
                f"{ctypes.WinError(ctypes.get_last_error())}"
            )

        process_information = _ProcessInformation()
        try:
            job_list = (wintypes.HANDLE * 1)(self._handle)
            if not _KERNEL32.UpdateProcThreadAttribute(
                attribute_list,
                0,
                _PROC_THREAD_ATTRIBUTE_JOB_LIST,
                ctypes.cast(job_list, ctypes.c_void_p),
                ctypes.sizeof(job_list),
                None,
                None,
            ):
                raise ManagedProcessError(
                    "Windows could not make trainer ownership atomic: "
                    f"{ctypes.WinError(ctypes.get_last_error())}"
                )
            startup = _StartupInfoExW()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.lpAttributeList = attribute_list
            command_line = ctypes.create_unicode_buffer(
                subprocess.list2cmdline([str(part) for part in command])
            )
            if not _KERNEL32.CreateProcessW(
                None,
                command_line,
                None,
                None,
                True,
                _EXTENDED_STARTUPINFO_PRESENT,
                None,
                str(cwd),
                ctypes.cast(
                    ctypes.byref(startup),
                    ctypes.POINTER(_StartupInfoW),
                ),
                ctypes.byref(process_information),
            ):
                raise ManagedProcessError(
                    "MIVIM could not start the trainer inside Windows crash "
                    f"cleanup: {ctypes.WinError(ctypes.get_last_error())}"
                )
        finally:
            _KERNEL32.DeleteProcThreadAttributeList(attribute_list)

        try:
            wait_result = _KERNEL32.WaitForSingleObject(
                process_information.hProcess,
                _INFINITE,
            )
            if wait_result == _WAIT_FAILED:
                raise ManagedProcessError(
                    "Windows could not wait for the trainer process: "
                    f"{ctypes.WinError(ctypes.get_last_error())}"
                )
            exit_code = wintypes.DWORD()
            if not _KERNEL32.GetExitCodeProcess(
                process_information.hProcess,
                ctypes.byref(exit_code),
            ):
                raise ManagedProcessError(
                    "Windows could not read the trainer exit status: "
                    f"{ctypes.WinError(ctypes.get_last_error())}"
                )
            return ctypes.c_long(exit_code.value).value
        finally:
            if process_information.hThread:
                _KERNEL32.CloseHandle(process_information.hThread)
            if process_information.hProcess:
                _KERNEL32.CloseHandle(process_information.hProcess)

    def close(self) -> None:
        if os.name == "nt" and self._handle is not None:
            _KERNEL32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def run_managed_process(command: Sequence[str], *, cwd: Path) -> int:
    """Run a child and ensure Windows kills its tree if MIVIM is terminated."""

    if not command:
        raise ValueError("Managed process command cannot be empty.")
    if os.name == "nt":
        with _WindowsKillJob() as job:
            return job.run(command, cwd=cwd)
    try:
        process = subprocess.Popen(list(command), cwd=cwd)
    except OSError as exc:
        raise ManagedProcessError(
            f"MIVIM could not start the trainer process: {exc}"
        ) from exc
    return process.wait()
