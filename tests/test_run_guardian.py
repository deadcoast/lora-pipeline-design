from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import copy_context
from pathlib import Path

import pytest

from src import config, run_guardian
from src import pipeline as pipeline_module
from src.run_guardian import (
    RUN_GUARD_DIRECTORY_ENV,
    RunLeaseBusyError,
    RunLeaseError,
    RunLeaseSelfConflictError,
    accelerator_lease_path,
    active_run_guard,
    active_run_guard_covers,
    pipeline_run_guard,
    project_lease_path,
    resource_lease_path,
)

REPOSITORY = Path(__file__).resolve().parents[1]
_HOLDER_SCRIPT = """
import os
import sys
import time
from pathlib import Path

from src.run_guardian import pipeline_run_guard

workspace = Path(sys.argv[1])
guard_root = Path(sys.argv[2])
project = sys.argv[3]
use_accelerator = sys.argv[4] == "1"
scope_count = int(sys.argv[5])
scope_values = sys.argv[6 : 6 + scope_count]
resource_values = sys.argv[6 + scope_count :]
project_scopes = tuple(Path(value) for value in scope_values)
resource_paths = tuple(Path(value) for value in resource_values)
with pipeline_run_guard(
    project=project,
    workspace=workspace,
    mode="validate" if use_accelerator else "prepare",
    use_accelerator=use_accelerator,
    project_scopes=project_scopes,
    resource_paths=resource_paths,
    environ={"MIVIM_RUN_GUARD_DIR": str(guard_root)},
):
    print(f"READY:{os.getpid()}", flush=True)
    time.sleep(120)
"""


def _start_holder(
    workspace: Path,
    guard_root: Path,
    project: str,
    *,
    use_accelerator: bool,
    project_scopes: tuple[Path, ...] = (),
    resource_paths: tuple[Path, ...] = (),
) -> tuple[subprocess.Popen[str], int]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _HOLDER_SCRIPT,
            str(workspace),
            str(guard_root),
            project,
            "1" if use_accelerator else "0",
            str(len(project_scopes)),
            *(str(path) for path in project_scopes),
            *(str(path) for path in resource_paths),
        ],
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    ready = process.stdout.readline().strip()
    if not ready.startswith("READY:"):
        try:
            _stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            _stdout, stderr = process.communicate(timeout=3)
        pytest.fail(f"Run-lease holder did not start: {ready}\n{stderr}")
    return process, int(ready.removeprefix("READY:"))


def _stop_holder(process: subprocess.Popen[str], *, hard: bool = False) -> None:
    if process.poll() is not None:
        return
    process.kill() if hard else process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait_for_lease_release(path: Path, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        descriptor = run_guardian._prepare_lock_file(path)
        locked = False
        try:
            locked = run_guardian._try_lock(descriptor)
            if locked:
                run_guardian._unlock(descriptor)
                return
        finally:
            os.close(descriptor)
        if time.monotonic() >= deadline:
            pytest.fail(f"Kernel lease was not released after process exit: {path}")
        time.sleep(0.01)


def test_cross_process_leases_separate_projects_and_accelerator(
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "global-guard"
    first_workspace = tmp_path / "workspace-a"
    second_workspace = tmp_path / "workspace-b"
    holder, holder_pid = _start_holder(
        first_workspace,
        guard_root,
        "portrait_a",
        use_accelerator=True,
    )
    try:
        # Preparation in another project does not use the accelerator and remains
        # available while a different project validates or trains.
        with pipeline_run_guard(
            project="portrait_b",
            workspace=second_workspace,
            mode="prepare",
            use_accelerator=False,
            environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
        ):
            pass

        with (
            pytest.raises(RunLeaseBusyError) as accelerator_error,
            pipeline_run_guard(
                project="portrait_b",
                workspace=second_workspace,
                mode="validate",
                use_accelerator=True,
                environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
            ),
        ):
            pass
        assert accelerator_error.value.kind == "accelerator"
        assert accelerator_error.value.holder is not None
        assert accelerator_error.value.holder.pid == holder_pid
        assert accelerator_error.value.holder.project == "portrait_a"
        assert "already using the accelerator" in str(accelerator_error.value)
        assert f"process {holder_pid}" in str(accelerator_error.value)
        assert accelerator_error.value.holder.started_at in str(accelerator_error.value)

        with (
            pytest.raises(RunLeaseBusyError) as project_error,
            pipeline_run_guard(
                project="portrait_a",
                workspace=first_workspace,
                mode="prepare",
                use_accelerator=False,
                environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
            ),
        ):
            pass
        assert project_error.value.kind == "project"
        assert project_error.value.holder is not None
        assert project_error.value.holder.pid == holder_pid
        assert 'project "portrait_a"' in str(project_error.value)
    finally:
        _stop_holder(holder)


def test_process_death_releases_leases_and_replaces_stale_metadata(
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "global-guard"
    workspace = tmp_path / "workspace"
    environ = {RUN_GUARD_DIRECTORY_ENV: str(guard_root)}
    holder, holder_pid = _start_holder(
        workspace,
        guard_root,
        "hard_exit_v1",
        use_accelerator=True,
    )
    lease_paths = (
        accelerator_lease_path(environ),
        project_lease_path(workspace, "hard_exit_v1", environ),
    )
    active_metadata = [run_guardian._read_metadata(path) for path in lease_paths]
    assert all(metadata is not None for metadata in active_metadata)
    old_lease_ids = {
        metadata.lease_id for metadata in active_metadata if metadata is not None
    }
    assert len(old_lease_ids) == 1
    assert {metadata.pid for metadata in active_metadata if metadata is not None} == {
        holder_pid
    }

    _stop_holder(holder, hard=True)
    assert holder.returncode is not None
    assert all(run_guardian._metadata_path(path).is_file() for path in lease_paths)
    for path in lease_paths:
        _wait_for_lease_release(path)

    with pipeline_run_guard(
        project="hard_exit_v1",
        workspace=workspace,
        mode="validate",
        use_accelerator=True,
        environ=environ,
    ) as metadata:
        assert metadata.pid == os.getpid()
        assert metadata.pid != holder_pid
        assert metadata.project == "hard_exit_v1"
        assert metadata.lease_id not in old_lease_ids
        current_metadata = [run_guardian._read_metadata(path) for path in lease_paths]
        assert all(owner is not None for owner in current_metadata)
        assert {owner.lease_id for owner in current_metadata if owner is not None} == {
            metadata.lease_id
        }
        assert {owner.pid for owner in current_metadata if owner is not None} == {
            os.getpid()
        }

    assert all(run_guardian._read_metadata(path) is None for path in lease_paths)
    assert all(not run_guardian._metadata_path(path).exists() for path in lease_paths)


def test_shared_mutable_scope_blocks_same_project_across_config_workspaces(
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "global-guard"
    shared_scope = tmp_path / "shared-model-parent"
    holder, _holder_pid = _start_holder(
        tmp_path / "workspace-a",
        guard_root,
        "portrait_v1",
        use_accelerator=False,
        project_scopes=(shared_scope,),
    )
    try:
        with (
            pytest.raises(RunLeaseBusyError) as conflict,
            pipeline_run_guard(
                project="portrait_v1",
                workspace=tmp_path / "workspace-b",
                mode="prepare",
                use_accelerator=False,
                project_scopes=(shared_scope,),
                environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
            ),
        ):
            pass
        assert conflict.value.kind == "project"

        with pipeline_run_guard(
            project="portrait_v2",
            workspace=tmp_path / "workspace-b",
            mode="prepare",
            use_accelerator=False,
            project_scopes=(shared_scope,),
            environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
        ):
            pass
    finally:
        _stop_holder(holder)


def test_canonical_resource_lease_blocks_aliases_across_projects(
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "global-guard"
    resource = tmp_path / "datasets" / "portrait_b"
    alias_parent = resource.parent / "alias-parent"
    resource.mkdir(parents=True)
    alias_parent.mkdir()
    resource_alias = alias_parent / ".." / resource.name
    holder, holder_pid = _start_holder(
        tmp_path / "workspace-a",
        guard_root,
        "portrait_a",
        use_accelerator=False,
        resource_paths=(resource,),
    )
    try:
        with (
            pytest.raises(RunLeaseBusyError) as conflict,
            pipeline_run_guard(
                project="portrait_b",
                workspace=tmp_path / "workspace-b",
                mode="prepare",
                use_accelerator=False,
                resource_paths=(resource_alias,),
                environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
            ),
        ):
            pass
        assert conflict.value.kind == "resource"
        assert conflict.value.holder is not None
        assert conflict.value.holder.pid == holder_pid
        assert conflict.value.holder.project == "portrait_a"

        with pipeline_run_guard(
            project="portrait_b",
            workspace=tmp_path / "workspace-b",
            mode="prepare",
            use_accelerator=False,
            resource_paths=(tmp_path / "datasets" / "portrait_c",),
            environ={RUN_GUARD_DIRECTORY_ENV: str(guard_root)},
        ):
            pass
    finally:
        _stop_holder(holder)


def test_project_and_resource_leases_use_central_canonical_paths(
    tmp_path: Path,
) -> None:
    guard_root = (tmp_path / "global-guard").resolve()
    environ = {RUN_GUARD_DIRECTORY_ENV: str(guard_root)}
    workspace = tmp_path / "workspace"
    resource = tmp_path / "datasets" / "portrait"
    resource.mkdir(parents=True)
    alias_parent = resource.parent / "nested"
    alias_parent.mkdir()
    resource_alias = alias_parent / ".." / resource.name

    project_path = project_lease_path(workspace, "portrait_v1", environ)
    resource_path = resource_lease_path(resource, environ)

    assert project_path.parent == guard_root / "projects"
    assert resource_path.parent == guard_root / "resources"
    assert resource_lease_path(resource_alias, environ) == resource_path

    alternate_case = resource.with_name(resource.name.swapcase())
    case_insensitive = os.path.normcase("A") == os.path.normcase("a")
    if case_insensitive:
        assert resource_lease_path(alternate_case, environ) == resource_path
    else:
        assert resource_lease_path(alternate_case, environ) != resource_path

    with pipeline_run_guard(
        project="portrait_v1",
        workspace=workspace,
        mode="prepare",
        use_accelerator=False,
        project_scopes=(tmp_path / "shared-output",),
        resource_paths=(resource, resource_alias),
        environ=environ,
    ):
        assert not (workspace / ".mivim-run-locks").exists()
        assert project_path.is_file()
        assert resource_path.is_file()


def test_relative_guard_directory_override_is_rejected() -> None:
    with pytest.raises(RunLeaseError, match="must be an absolute folder path"):
        accelerator_lease_path({RUN_GUARD_DIRECTORY_ENV: "relative-guard"})

    with (
        pytest.raises(RunLeaseError, match="must be an absolute folder path"),
        pipeline_run_guard(
            project="portrait_v1",
            workspace=Path("workspace"),
            mode="prepare",
            use_accelerator=False,
            environ={RUN_GUARD_DIRECTORY_ENV: "relative-guard"},
        ),
    ):
        pass


def test_same_process_nested_guard_has_explicit_error(tmp_path: Path) -> None:
    environ = {RUN_GUARD_DIRECTORY_ENV: str((tmp_path / "global-guard").resolve())}
    assert active_run_guard() is None

    with pipeline_run_guard(
        project="nested_v1",
        workspace=tmp_path / "workspace",
        mode="train",
        use_accelerator=True,
        environ=environ,
    ) as owner:
        assert active_run_guard() is owner
        with (
            pytest.raises(
                RunLeaseSelfConflictError,
                match="duplicate run-safety guard inside its own process",
            ) as conflict,
            pipeline_run_guard(
                project="nested_v1",
                workspace=tmp_path / "workspace",
                mode="validate",
                use_accelerator=True,
                environ=environ,
            ),
        ):
            pass
        assert conflict.value.kind == "accelerator"
        assert conflict.value.holder.pid == os.getpid()
        assert conflict.value.holder.lease_id == owner.lease_id
        assert active_run_guard() is owner
    assert active_run_guard() is None


def test_active_run_guard_resets_after_success_and_exception(
    tmp_path: Path,
) -> None:
    environ = {RUN_GUARD_DIRECTORY_ENV: str((tmp_path / "global-guard").resolve())}
    workspace = tmp_path / "workspace"
    assert active_run_guard() is None
    assert not active_run_guard_covers(
        project="active_v1",
        workspace=workspace,
        use_accelerator=False,
    )

    with pipeline_run_guard(
        project="active_v1",
        workspace=workspace,
        mode="prepare",
        use_accelerator=False,
        environ=environ,
    ) as metadata:
        assert active_run_guard() is metadata
    assert active_run_guard() is None

    with (
        pytest.raises(RuntimeError, match="stage sentinel"),
        pipeline_run_guard(
            project="active_v1",
            workspace=workspace,
            mode="prepare",
            use_accelerator=False,
            environ=environ,
        ) as metadata,
    ):
        assert active_run_guard() is metadata
        raise RuntimeError("stage sentinel")
    assert active_run_guard() is None
    assert not active_run_guard_covers(
        project="active_v1",
        workspace=workspace,
        use_accelerator=False,
    )


def test_active_run_guard_coverage_is_exact_and_context_local(
    tmp_path: Path,
) -> None:
    environ = {RUN_GUARD_DIRECTORY_ENV: str((tmp_path / "guard-root").resolve())}
    workspace = tmp_path / "workspace"
    held_scope = tmp_path / "shared" / "held"
    missing_scope = tmp_path / "shared" / "missing"
    held_resource = tmp_path / "datasets" / "held"
    missing_resource = tmp_path / "datasets" / "missing"
    alias_parent = held_resource.parent / "alias-parent"
    alias_parent.mkdir(parents=True)
    held_resource_alias = alias_parent / ".." / held_resource.name

    def covers(
        *,
        project: str = "coverage_v1",
        requested_workspace: Path = workspace,
        require_accelerator: bool = False,
        scopes: tuple[Path, ...] = (held_scope,),
        resources: tuple[Path, ...] = (held_resource_alias,),
    ) -> bool:
        return active_run_guard_covers(
            project=project,
            workspace=requested_workspace,
            use_accelerator=require_accelerator,
            project_scopes=scopes,
            resource_paths=resources,
        )

    assert not covers()

    with pipeline_run_guard(
        project="coverage_v1",
        workspace=workspace,
        mode="prepare",
        use_accelerator=False,
        project_scopes=(held_scope,),
        resource_paths=(held_resource,),
        environ=environ,
    ):
        assert covers()
        copied_active_context = copy_context()
        assert copied_active_context.run(covers)
        assert not covers(require_accelerator=True)
        assert not covers(scopes=(missing_scope,))
        assert not covers(resources=(missing_resource,))
        assert not covers(project="different_v1")
        assert not covers(requested_workspace=tmp_path / "other-workspace")
    assert not covers()
    assert not copied_active_context.run(covers)
    assert copied_active_context.run(active_run_guard) is None

    with pipeline_run_guard(
        project="coverage_v1",
        workspace=workspace,
        mode="validate",
        use_accelerator=True,
        project_scopes=(held_scope,),
        resource_paths=(held_resource,),
        environ=environ,
    ):
        assert covers(require_accelerator=True)
    assert not covers()


def test_active_guard_prevents_public_stages_from_nesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparer = pipeline_module.prep
    trainer = pipeline_module.trainer
    validator = pipeline_module.validator
    toolkit = tmp_path / "ai-toolkit"
    toolkit.mkdir()
    (toolkit / "run.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "AI_TOOLKIT_DIR", str(toolkit))
    run_configs = tmp_path / "workspace" / "configs"
    dataset = tmp_path / "datasets" / "active_stages_v1"
    checkpoints = tmp_path / "training" / "checkpoints"
    grids = tmp_path / "validation" / "grids"
    comfy_loras = tmp_path / "comfy" / "loras"
    monkeypatch.setattr(config, "RUN_CONFIGS_DIR", run_configs)
    monkeypatch.setattr(config, "DATASETS_DIR", dataset.parent)
    monkeypatch.setattr(config, "CHECKPOINTS_DIR", checkpoints)
    monkeypatch.setattr(config, "GRIDS_DIR", grids)
    monkeypatch.setattr(config, "COMFY_LORA_DIR", comfy_loras)

    @contextmanager
    def unexpected_nested_guard(**_kwargs: object) -> Iterator[None]:
        pytest.fail("A public stage nested another pipeline guard.")
        yield

    orphan_checks: list[str] = []
    for module in (preparer, trainer, validator):
        monkeypatch.setattr(module, "pipeline_run_guard", unexpected_nested_guard)
        monkeypatch.setattr(
            module,
            "check_orphaned_comfy_prompts",
            lambda module=module: orphan_checks.append(module.__name__),
        )
    monkeypatch.setattr(
        preparer,
        "_prepare_with_guard",
        lambda *_args, **_kwargs: dataset,
    )
    monkeypatch.setattr(trainer, "_train_with_guard", lambda *_args, **_kwargs: 0)
    expected_grid = tmp_path / "validation" / "grid.png"
    monkeypatch.setattr(
        validator,
        "_validate_with_guard",
        lambda *_args, **_kwargs: expected_grid,
    )

    with pipeline_run_guard(
        project="active_stages_v1",
        workspace=run_configs.parent,
        mode="full",
        use_accelerator=True,
        project_scopes=(
            dataset.parent,
            checkpoints.parent,
            grids.parent,
            comfy_loras.parent,
        ),
        resource_paths=(
            dataset,
            checkpoints,
            run_configs / "active_stages_v1.yaml",
            grids,
            comfy_loras,
        ),
        environ={RUN_GUARD_DIRECTORY_ENV: str((tmp_path / "global-guard").resolve())},
    ):
        assert (
            preparer.prepare(
                tmp_path / "photos",
                "active_stages_v1",
                caption_mode="blip",
            )
            == dataset
        )
        assert (
            trainer.train(
                tmp_path / "template.yaml",
                run_name="active_stages_v1",
                dataset_dir=dataset,
                output_dir=checkpoints,
            )
            == 0
        )
        assert (
            validator.validate(
                lora="active_stages_v1.safetensors",
                strengths=[0.8],
                prompts=["portrait"],
                trigger="ohwx person",
                seed=42,
                run_name="active_stages_v1",
            )
            == expected_grid
        )
    assert orphan_checks == [
        preparer.__name__,
        trainer.__name__,
        validator.__name__,
    ]


def test_failed_late_scope_releases_earlier_project_and_accelerator_leases(
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "global-guard"
    environ = {RUN_GUARD_DIRECTORY_ENV: str(guard_root)}
    free_scope = tmp_path / "a-free-scope"
    blocked_scope = tmp_path / "z-blocked-scope"
    holder, _holder_pid = _start_holder(
        tmp_path / "holder-workspace",
        guard_root,
        "portrait_v1",
        use_accelerator=False,
        project_scopes=(blocked_scope,),
    )
    try:
        assert active_run_guard() is None
        with (
            pytest.raises(RunLeaseBusyError) as conflict,
            pipeline_run_guard(
                project="portrait_v1",
                workspace=free_scope,
                mode="train",
                use_accelerator=True,
                project_scopes=(blocked_scope,),
                environ=environ,
            ),
        ):
            pass
        assert conflict.value.kind == "project"
        assert active_run_guard() is None

        # The failed attempt acquired both of these first. A fresh invocation can
        # take them immediately only if ExitStack unwound the partial acquisition.
        with pipeline_run_guard(
            project="portrait_v1",
            workspace=free_scope,
            mode="train",
            use_accelerator=True,
            environ=environ,
        ):
            assert active_run_guard() is not None
        assert active_run_guard() is None
    finally:
        _stop_holder(holder)


def test_failed_resource_releases_project_and_accelerator_leases(
    tmp_path: Path,
) -> None:
    guard_root = tmp_path / "global-guard"
    environ = {RUN_GUARD_DIRECTORY_ENV: str(guard_root)}
    free_workspace = tmp_path / "free-workspace"
    blocked_resource = tmp_path / "datasets" / "shared"
    holder, _holder_pid = _start_holder(
        tmp_path / "holder-workspace",
        guard_root,
        "holder_v1",
        use_accelerator=False,
        resource_paths=(blocked_resource,),
    )
    try:
        assert active_run_guard() is None
        with (
            pytest.raises(RunLeaseBusyError) as conflict,
            pipeline_run_guard(
                project="portrait_v1",
                workspace=free_workspace,
                mode="train",
                use_accelerator=True,
                resource_paths=(blocked_resource,),
                environ=environ,
            ),
        ):
            pass
        assert conflict.value.kind == "resource"
        assert active_run_guard() is None

        with pipeline_run_guard(
            project="portrait_v1",
            workspace=free_workspace,
            mode="train",
            use_accelerator=True,
            environ=environ,
        ):
            assert active_run_guard() is not None
        assert active_run_guard() is None
    finally:
        _stop_holder(holder)


def test_windows_default_guard_uses_stable_local_app_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_guardian.os, "name", "nt")

    path = accelerator_lease_path({"LOCALAPPDATA": str(tmp_path)})

    assert path == (
        tmp_path / "MIVIM" / "LoRA Pipeline" / "run-guardian" / "accelerator.lock"
    )


def test_owner_metadata_failure_does_not_disable_the_os_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_root = tmp_path / "global-guard"
    environ = {RUN_GUARD_DIRECTORY_ENV: str(guard_root)}
    workspace = tmp_path / "workspace"
    lock_path = project_lease_path(workspace, "metadata_v1", environ)

    with pipeline_run_guard(
        project="metadata_v1",
        workspace=workspace,
        mode="prepare",
        use_accelerator=False,
        environ=environ,
    ) as stale_metadata:
        pass

    # Recreate the durable artifacts left by a process that died before cleanup.
    run_guardian._write_metadata(lock_path, stale_metadata)
    descriptor = run_guardian._prepare_lock_file(lock_path)
    try:
        run_guardian._write_lease_marker(descriptor, stale_metadata.lease_id)
    finally:
        os.close(descriptor)
    assert run_guardian._read_metadata(lock_path) == stale_metadata

    def reject_metadata(*_args: object, **_kwargs: object) -> None:
        raise RunLeaseError("metadata unavailable")

    monkeypatch.setattr(run_guardian, "_write_metadata", reject_metadata)

    with pipeline_run_guard(
        project="metadata_v1",
        workspace=workspace,
        mode="prepare",
        use_accelerator=False,
        environ=environ,
    ) as metadata:
        assert metadata.project == "metadata_v1"
        assert metadata.lease_id != stale_metadata.lease_id
        assert run_guardian._read_metadata(lock_path) is None
        with (
            pytest.raises(RunLeaseBusyError) as conflict,
            pipeline_run_guard(
                project="metadata_v1",
                workspace=workspace,
                mode="prepare",
                use_accelerator=False,
                environ=environ,
            ),
        ):
            pass
        assert conflict.value.kind == "project"
        assert conflict.value.holder is None

    # A metadata failure must not strand the authoritative kernel lease.
    with pipeline_run_guard(
        project="metadata_v1",
        workspace=workspace,
        mode="prepare",
        use_accelerator=False,
        environ=environ,
    ):
        pass


@pytest.mark.parametrize(
    ("arguments", "expected_mode", "expected_accelerator"),
    [
        (
            {
                "skip_prep": False,
                "skip_train": True,
                "skip_validate": True,
                "lora": None,
                "vision": False,
            },
            "prepare",
            False,
        ),
        (
            {
                "skip_prep": False,
                "skip_train": True,
                "skip_validate": True,
                "lora": None,
                "vision": True,
            },
            "prepare",
            True,
        ),
        (
            {
                "skip_prep": True,
                "skip_train": True,
                "skip_validate": False,
                "lora": "portrait.safetensors",
                "vision": False,
            },
            "validate",
            True,
        ),
        (
            {
                "skip_prep": True,
                "skip_train": True,
                "skip_validate": False,
                "lora": None,
                "vision": False,
            },
            "validate",
            True,
        ),
        (
            {
                "skip_prep": False,
                "skip_train": False,
                "skip_validate": False,
                "lora": None,
                "vision": False,
                "base_model": Path("models/test-base-model.safetensors"),
            },
            "full",
            True,
        ),
    ],
)
def test_pipeline_acquires_expected_guard_before_any_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, object],
    expected_mode: str,
    expected_accelerator: bool,
) -> None:
    observed: list[dict[str, object]] = []
    stage_started = False

    @contextmanager
    def reject_before_stage(**kwargs: object) -> Iterator[None]:
        observed.append(kwargs)
        raise RuntimeError("guard sentinel")
        yield

    def unexpected_stage(*_args: object, **_kwargs: object) -> None:
        nonlocal stage_started
        stage_started = True

    monkeypatch.setattr(pipeline_module, "pipeline_run_guard", reject_before_stage)
    monkeypatch.setattr(pipeline_module.prep, "prepare", unexpected_stage)
    monkeypatch.setattr(pipeline_module.trainer, "train", unexpected_stage)
    monkeypatch.setattr(pipeline_module.validator, "validate", unexpected_stage)
    monkeypatch.setattr(config, "RUN_CONFIGS_DIR", tmp_path / "workspace" / "configs")
    monkeypatch.setattr(config, "DATASETS_DIR", tmp_path / "photos" / "datasets")
    monkeypatch.setattr(
        config,
        "CHECKPOINTS_DIR",
        tmp_path / "training" / "checkpoints",
    )
    monkeypatch.setattr(config, "GRIDS_DIR", tmp_path / "validation" / "grids")
    monkeypatch.setattr(config, "COMFY_LORA_DIR", tmp_path / "comfy" / "loras")

    values: dict[str, object] = {
        "name": "guarded_v1",
        "src": str(tmp_path / "photos"),
        "config": "configs/person_sdxl.yaml",
        "trigger": "ohwx person",
        "size": 1024,
        "preset": "balanced",
        "base_model": None,
        "strengths": [0.0, 0.8],
        "seed": 42,
        "lora": None,
        "vision": False,
        "overwrite": False,
        "skip_prep": True,
        "skip_train": False,
        "skip_validate": False,
        "attempt_id": "11111111-1111-4111-8111-111111111111",
    }
    values.update(arguments)

    with pytest.raises(RuntimeError, match="guard sentinel"):
        pipeline_module.run(argparse.Namespace(**values))

    assert not stage_started
    expected_scopes = {
        (tmp_path / "workspace").resolve(),
        (tmp_path / "photos").resolve(),
    }
    if not bool(values["skip_train"]):
        expected_scopes.add((tmp_path / "training").resolve())
    if not bool(values["skip_validate"]):
        expected_scopes.add((tmp_path / "validation").resolve())
        expected_scopes.add((tmp_path / "comfy").resolve())
    config_dir = tmp_path / "workspace" / "configs"
    expected_resources = {
        (config_dir / "guarded_v1.yaml").resolve(),
        (config_dir / "guarded_v1-recovery.json").resolve(),
        (config_dir / "guarded_v1-result.json").resolve(),
    }
    if not bool(values["skip_prep"]) or not bool(values["skip_train"]):
        expected_resources.add(
            (tmp_path / "photos" / "datasets" / "guarded_v1").resolve()
        )
    if not bool(values["skip_train"]):
        expected_resources.add((tmp_path / "training" / "checkpoints").resolve())
    if not bool(values["skip_validate"]):
        expected_resources.add((tmp_path / "validation" / "grids").resolve())
        expected_resources.add((tmp_path / "comfy" / "loras").resolve())
        if bool(values["skip_train"]) and not values["lora"]:
            expected_resources.add((tmp_path / "training" / "checkpoints").resolve())
    base_model_value = values["base_model"]
    if base_model_value is not None:
        assert isinstance(base_model_value, Path)
        expected_resources.add(base_model_value.resolve())
    assert observed == [
        {
            "project": "guarded_v1",
            "workspace": (tmp_path / "workspace").resolve(),
            "mode": expected_mode,
            "use_accelerator": expected_accelerator,
            "project_scopes": expected_scopes,
            "resource_paths": expected_resources,
        }
    ]
