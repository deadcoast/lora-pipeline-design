"""Cross-process leases that keep MIVIM runs from sharing mutable work.

The operating system owns the actual lock. A generation marker in the lock file
links it to an atomic JSON owner sidecar used only for user-facing context, so a
killed process cannot leave a permanently busy lease: Windows byte-range locks
and POSIX ``flock`` locks are released when the process handle closes.
"""

from __future__ import annotations

import errno
import getpass
import hashlib
import importlib
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self, cast

from .io_utils import atomic_write_bytes

RUN_GUARD_DIRECTORY_ENV = "MIVIM_RUN_GUARD_DIR"
_LOCK_FILE_SIZE = 4096
_LEASE_MARKER_OFFSET = 1
_LEASE_MARKER_SIZE = 64
_LEASE_MARKER_PREFIX = b"MIVIM1:"
_METADATA_READ_ATTEMPTS = 5
_METADATA_READ_DELAY_SECONDS = 0.01

LeaseKind = Literal["accelerator", "project", "resource"]
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunLeaseMetadata:
    """Human-readable owner details written while an OS lease is held."""

    lease_id: str
    pid: int
    project: str
    started_at: str
    mode: str
    workspace: str

    @classmethod
    def from_json(cls, value: object) -> RunLeaseMetadata | None:
        if not isinstance(value, dict):
            return None
        fields = {
            key: value.get(key)
            for key in (
                "lease_id",
                "pid",
                "project",
                "started_at",
                "mode",
                "workspace",
            )
        }
        if (
            not _valid_lease_id(fields["lease_id"])
            or not isinstance(fields["pid"], int)
            or isinstance(fields["pid"], bool)
            or fields["pid"] <= 0
            or not all(
                isinstance(fields[key], str) and bool(fields[key])
                for key in ("project", "started_at", "mode", "workspace")
            )
        ):
            return None
        return cls(
            lease_id=cast(str, fields["lease_id"]),
            pid=cast(int, fields["pid"]),
            project=cast(str, fields["project"]),
            started_at=cast(str, fields["started_at"]),
            mode=cast(str, fields["mode"]),
            workspace=cast(str, fields["workspace"]),
        )


@dataclass(slots=True)
class _ActiveRunGuard:
    metadata: RunLeaseMetadata
    project_identity: str
    workspace_identity: str
    use_accelerator: bool
    project_scope_identities: frozenset[str]
    resource_identities: frozenset[str]
    is_held: bool = True


_ACTIVE_RUN_GUARD: ContextVar[_ActiveRunGuard | None] = ContextVar(
    "mivim_active_run_guard",
    default=None,
)


def active_run_guard() -> RunLeaseMetadata | None:
    """Return the guard held by this execution context, if one is active."""

    active = _ACTIVE_RUN_GUARD.get()
    return active.metadata if active is not None and active.is_held else None


class RunLeaseError(RuntimeError):
    """Base class for run-safety failures suitable for direct UI display."""


class RunLeaseBusyError(RunLeaseError):
    """Raised when another process currently owns a requested lease."""

    def __init__(
        self,
        kind: LeaseKind,
        requested_project: str,
        holder: RunLeaseMetadata | None,
    ) -> None:
        self.kind = kind
        self.requested_project = requested_project
        self.holder = holder
        if kind == "accelerator":
            subject = "Another MIVIM run is already using the accelerator"
        elif kind == "resource":
            subject = (
                f"Another MIVIM run is already using a file or folder required "
                f'by project "{requested_project}"'
            )
        else:
            subject = (
                f"Another MIVIM run is already working on project "
                f'"{requested_project}" in this workspace'
            )
        if holder is not None:
            detail = (
                f' Project "{holder.project}" is owned by process {holder.pid}, '
                f"started {holder.started_at}."
            )
        else:
            detail = "."
        super().__init__(
            f"{subject}{detail} Wait for that run to finish, or close it safely, "
            "then try again."
        )


class RunLeaseSelfConflictError(RunLeaseError):
    """Raised when one process attempts to acquire a lease it already owns."""

    def __init__(
        self,
        kind: LeaseKind,
        requested_project: str,
        holder: RunLeaseMetadata,
    ) -> None:
        self.kind = kind
        self.requested_project = requested_project
        self.holder = holder
        super().__init__(
            "MIVIM detected a duplicate run-safety guard inside its own process "
            f'({holder.pid}) for project "{requested_project}". The existing run '
            "is still protected. This is an internal coordination error; close the "
            "duplicate action and retry from the original run."
        )


def _valid_lease_id(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def _user_guard_key() -> str:
    """Return a filesystem-safe per-user key for the machine-wide temp folder."""

    user_id = str(getattr(os, "getuid", lambda: "")())
    identity = f"{user_id}\0{getpass.getuser()}".encode("utf-8", errors="replace")
    return hashlib.sha256(identity).hexdigest()[:16]


def _canonical_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RunLeaseError(
            f"MIVIM could not resolve the run-safety path {path}: {exc}"
        ) from exc


def _path_identity(path: Path) -> str:
    """Return the host filesystem's canonical comparison identity for a path."""

    return os.path.normcase(str(_canonical_path(path)))


def _project_identity(project: str) -> str:
    return os.path.normcase(project)


def active_run_guard_covers(
    *,
    project: str,
    workspace: Path,
    use_accelerator: bool,
    project_scopes: Iterable[Path] = (),
    resource_paths: Iterable[Path] = (),
) -> bool:
    """Return whether this context already holds every requested run lease."""

    active = _ACTIVE_RUN_GUARD.get()
    if active is None or not active.is_held:
        return False
    workspace_identity = _path_identity(workspace)
    if (
        active.project_identity != _project_identity(project)
        or active.workspace_identity != workspace_identity
        or (use_accelerator and not active.use_accelerator)
    ):
        return False

    requested_scopes = {workspace_identity}
    requested_scopes.update(_path_identity(scope) for scope in project_scopes)
    if not requested_scopes.issubset(active.project_scope_identities):
        return False

    requested_resources = {_path_identity(resource) for resource in resource_paths}
    return requested_resources.issubset(active.resource_identities)


def _guard_root(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    override = values.get(RUN_GUARD_DIRECTORY_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            raise RunLeaseError(
                f"{RUN_GUARD_DIRECTORY_ENV} must be an absolute folder path; "
                f"received {override!r}."
            )
        return _canonical_path(candidate)

    if os.name == "nt" and values.get("LOCALAPPDATA", "").strip():
        candidate = (
            Path(values["LOCALAPPDATA"]) / "MIVIM" / "LoRA Pipeline" / "run-guardian"
        )
    else:
        candidate = (
            Path(tempfile.gettempdir()) / f"mivim-run-guardian-{_user_guard_key()}"
        )
    return _canonical_path(candidate)


def _lease_digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8", errors="surrogatepass")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def accelerator_lease_path(
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the per-user lease shared by every MIVIM workspace."""

    return _guard_root(environ) / "accelerator.lock"


def project_lease_path(
    workspace: Path,
    project: str,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return a central lease for one project identity in one workspace."""

    project_identity = _project_identity(project)
    digest = _lease_digest("project", _path_identity(workspace), project_identity)
    return _guard_root(environ) / "projects" / f"{digest}.lock"


def resource_lease_path(
    resource: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return a central lease keyed only by a canonical file/folder identity."""

    digest = _lease_digest("resource", _path_identity(resource))
    return _guard_root(environ) / "resources" / f"{digest}.lock"


def _prepare_lock_file(path: Path) -> int:
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        if os.fstat(descriptor).st_size < _LOCK_FILE_SIZE:
            os.ftruncate(descriptor, _LOCK_FILE_SIZE)
        return descriptor
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise RunLeaseError(
            f"MIVIM could not create its run-safety lease at {path}: {exc}"
        ) from exc


def _try_lock(descriptor: int) -> bool:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True

    fcntl = cast(Any, importlib.import_module("fcntl"))

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        raise
    return True


def _unlock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return

    fcntl = cast(Any, importlib.import_module("fcntl"))

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _write_descriptor_bytes(descriptor: int, offset: int, value: bytes) -> None:
    os.lseek(descriptor, offset, os.SEEK_SET)
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("The run-safety lease marker could not be written.")
        remaining = remaining[written:]


def _write_lease_marker(descriptor: int, lease_id: str | None) -> None:
    if lease_id is None:
        encoded = b""
    else:
        if not _valid_lease_id(lease_id):
            raise ValueError(f"Invalid run-safety lease identifier: {lease_id!r}")
        encoded = _LEASE_MARKER_PREFIX + lease_id.encode("ascii")
    if len(encoded) > _LEASE_MARKER_SIZE:
        raise ValueError("Run-safety lease identifier exceeded its storage limit.")
    _write_descriptor_bytes(
        descriptor,
        _LEASE_MARKER_OFFSET,
        encoded.ljust(_LEASE_MARKER_SIZE, b"\x00"),
    )


def _read_lease_marker(lock_path: Path) -> str | None:
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_RDONLY)
        os.lseek(descriptor, _LEASE_MARKER_OFFSET, os.SEEK_SET)
        raw = os.read(descriptor, _LEASE_MARKER_SIZE)
    except OSError:
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    encoded = raw.split(b"\x00", 1)[0]
    if not encoded.startswith(_LEASE_MARKER_PREFIX):
        return None
    try:
        lease_id = encoded.removeprefix(_LEASE_MARKER_PREFIX).decode("ascii")
    except UnicodeError:
        return None
    return lease_id if _valid_lease_id(lease_id) else None


def _metadata_path(lock_path: Path) -> Path:
    return lock_path.with_name(f"{lock_path.name}.owner.json")


def _write_metadata(lock_path: Path, metadata: RunLeaseMetadata) -> None:
    try:
        encoded = (
            json.dumps(asdict(metadata), ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, UnicodeError) as exc:
        raise RunLeaseError(
            f"MIVIM could not encode the run-safety lease owner: {exc}"
        ) from exc
    if len(encoded) > _LOCK_FILE_SIZE:
        raise RunLeaseError("MIVIM run-safety metadata exceeded its storage limit.")
    try:
        # Windows byte-range locks are mandatory and make the locked byte
        # unreadable to the contender. Keep owner context in an atomically
        # replaced sibling while the lock file remains the sole authority.
        atomic_write_bytes(_metadata_path(lock_path), encoded)
    except OSError as exc:
        raise RunLeaseError(
            f"MIVIM could not record the run-safety lease owner: {exc}"
        ) from exc


def _read_metadata(lock_path: Path) -> RunLeaseMetadata | None:
    path = _metadata_path(lock_path)
    for attempt in range(_METADATA_READ_ATTEMPTS):
        lease_id = _read_lease_marker(lock_path)
        try:
            raw = path.read_bytes()[:_LOCK_FILE_SIZE].rstrip(b" \t\r\n\x00")
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            parsed = None
        metadata = RunLeaseMetadata.from_json(parsed)
        if (
            lease_id is not None
            and metadata is not None
            and metadata.lease_id == lease_id
            and _read_lease_marker(lock_path) == lease_id
        ):
            return metadata
        if attempt + 1 < _METADATA_READ_ATTEMPTS:
            time.sleep(_METADATA_READ_DELAY_SECONDS)
    return None


def _remove_metadata(lock_path: Path) -> None:
    _metadata_path(lock_path).unlink(missing_ok=True)


@dataclass(slots=True)
class RunLease:
    """One held OS lease, released explicitly or automatically at process exit."""

    path: Path
    kind: LeaseKind
    metadata: RunLeaseMetadata
    _descriptor: int | None = field(default=None, init=False, repr=False)

    @classmethod
    def acquire(
        cls,
        path: Path,
        kind: LeaseKind,
        metadata: RunLeaseMetadata,
    ) -> RunLease:
        descriptor = _prepare_lock_file(path)
        try:
            locked = _try_lock(descriptor)
        except OSError as exc:
            os.close(descriptor)
            raise RunLeaseError(
                f"MIVIM could not check its run-safety lease at {path}: {exc}"
            ) from exc
        if not locked:
            os.close(descriptor)
            holder = _read_metadata(path)
            if holder is not None and holder.pid == os.getpid():
                raise RunLeaseSelfConflictError(kind, metadata.project, holder)
            raise RunLeaseBusyError(kind, metadata.project, holder)

        lease = cls(path=path, kind=kind, metadata=metadata)
        lease._descriptor = descriptor
        try:
            # A previous process can die before retiring its sidecar. Blank the
            # in-file generation marker first so contenders cannot attribute that
            # stale sidecar to this newly acquired OS lease.
            _write_lease_marker(descriptor, None)
        except (OSError, ValueError) as exc:
            try:
                try:
                    _unlock(descriptor)
                except OSError as unlock_error:
                    _LOGGER.warning(
                        "MIVIM could not explicitly unlock a failed lease: %s",
                        unlock_error,
                    )
            finally:
                lease._descriptor = None
                try:
                    os.close(descriptor)
                except OSError as close_error:
                    _LOGGER.warning(
                        "MIVIM could not close a failed lease handle: %s",
                        close_error,
                    )
            raise RunLeaseError(
                f"MIVIM could not initialize its run-safety lease at {path}: {exc}"
            ) from exc
        try:
            _remove_metadata(path)
        except OSError as exc:
            _LOGGER.warning(
                "MIVIM could not remove stale run-safety owner metadata: %s",
                exc,
            )
        try:
            _write_metadata(path, metadata)
        except RunLeaseError as exc:
            # The OS lease is authoritative. Owner context is useful, but a
            # sidecar permission/replace failure must not disable run safety.
            _LOGGER.warning("%s", exc)
        else:
            try:
                # Publish the generation only after the complete sidecar is
                # visible. Readers accept metadata only while both copies match.
                _write_lease_marker(descriptor, metadata.lease_id)
            except (OSError, ValueError) as exc:
                _LOGGER.warning(
                    "MIVIM could not publish run-safety owner metadata: %s",
                    exc,
                )
        return lease

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            try:
                _write_lease_marker(descriptor, None)
            except (OSError, ValueError) as exc:
                _LOGGER.warning(
                    "MIVIM could not retire its run-safety owner marker: %s",
                    exc,
                )
            try:
                _remove_metadata(self.path)
            except OSError as exc:
                _LOGGER.warning(
                    "MIVIM could not remove run-safety owner metadata: %s",
                    exc,
                )
            try:
                _unlock(descriptor)
            except OSError as exc:
                # Closing the descriptor still releases the kernel lease. Do not
                # replace a successful run (or its real exception) with cleanup
                # noise that a casual user cannot act upon.
                _LOGGER.warning("MIVIM could not explicitly unlock its lease: %s", exc)
        finally:
            try:
                os.close(descriptor)
            except OSError as exc:
                _LOGGER.warning("MIVIM could not close its lease handle: %s", exc)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def pipeline_run_guard(
    *,
    project: str,
    workspace: Path,
    mode: str,
    use_accelerator: bool,
    project_scopes: Iterable[Path] = (),
    resource_paths: Iterable[Path] = (),
    environ: Mapping[str, str] | None = None,
) -> Iterator[RunLeaseMetadata]:
    """Hold every cross-process lease required by one pipeline invocation."""

    canonical_workspace = _canonical_path(workspace)
    metadata = RunLeaseMetadata(
        lease_id=str(uuid.uuid4()),
        pid=os.getpid(),
        project=project,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
        mode=mode,
        workspace=str(canonical_workspace),
    )
    with ExitStack() as leases:
        if use_accelerator:
            leases.enter_context(
                RunLease.acquire(
                    accelerator_lease_path(environ),
                    "accelerator",
                    metadata,
                )
            )
        unique_scopes: dict[str, Path] = {}
        for scope in (canonical_workspace, *project_scopes):
            resolved = _canonical_path(scope)
            unique_scopes.setdefault(os.path.normcase(str(resolved)), resolved)
        for identity in sorted(unique_scopes):
            scope = unique_scopes[identity]
            leases.enter_context(
                RunLease.acquire(
                    project_lease_path(scope, project, environ),
                    "project",
                    metadata,
                )
            )
        unique_resources: dict[str, Path] = {}
        for resource in resource_paths:
            resolved = _canonical_path(resource)
            unique_resources.setdefault(os.path.normcase(str(resolved)), resolved)
        for identity in sorted(unique_resources):
            resource = unique_resources[identity]
            leases.enter_context(
                RunLease.acquire(
                    resource_lease_path(resource, environ),
                    "resource",
                    metadata,
                )
            )
        active = _ActiveRunGuard(
            metadata=metadata,
            project_identity=_project_identity(project),
            workspace_identity=os.path.normcase(str(canonical_workspace)),
            use_accelerator=use_accelerator,
            project_scope_identities=frozenset(unique_scopes),
            resource_identities=frozenset(unique_resources),
        )
        active_token = _ACTIVE_RUN_GUARD.set(active)
        try:
            yield metadata
        finally:
            # Clear context state before ExitStack releases the first lease so an
            # observing stage never sees a guard that is only partially held.
            # The mutable liveness bit also invalidates copied Context instances
            # that would otherwise retain the old ContextVar value after teardown.
            active.is_held = False
            _ACTIVE_RUN_GUARD.reset(active_token)
