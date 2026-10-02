# Choose a saved training checkpoint

Results now offers **Compare training checkpoints**. The dialog displays two
saved training previews at the same prompt index. Both training steps are
selectable, and missing previews are explicitly labelled. **Validate this
checkpoint** runs a separate baseline/strength sweep without preparing photos
or training again. It produces the normal ready-to-use ComfyUI workflow.

Selection is manual. Studio does not score likeness or declare a winning step.
This first version opens from a completed result and reads the original named
training folder. Checkpoints without readable training-step metadata are omitted.
Trainer previews are guidance; they are not presented as ComfyUI validation.

The chosen checkpoint is fingerprinted during discovery and checked again before
installation. Each choice receives a fresh version name, preserving the final
model and previous results. Its durable recovery and completed result retain the
original training project, including after restarting Studio. Changed checkpoint
bytes are rejected instead of silently substituting a different model.

Implementation: `src/training_checkpoints.py`, `src/checkpoint_review.py`, and the
existing desktop, service, pipeline, and settings modules.

## Local build and verification

The new standalone Windows package is
`dist/checkpoint-selection/MIVIMLoRAStudio.dist/MIVIMLoRAStudio.exe`.
Use `dist/checkpoint-selection/Open Studio.cmd` for the configured photo project.
The earlier Windows/GPU acceptance package remains in its original location.

Executable SHA-256:
`75471B51A39BFF9503CE3202C38CDB61C4EEA63DFAAE7D36F2647C8BA0BE10BC`.

- The rebuilt native app discovered eight real checkpoints, steps 250 through
  2,000, with four previews each, from `hd_sdxl_acceptance_20260916_v2`.
- The new native selection path launched the packaged worker and installed the
  selected 1,750-step weights under separate version names. Recovery records
  contain the original project and exact selected content fingerprint.
- **New live GPU validation is pending.** ComfyUI at `127.0.0.1:8188` was offline.
  Both attempted choices retain `validation_pending` recovery; neither has a
  completed result. The previous successful 2,000-step acceptance remains intact.
- 304 tests passed using isolated app-data and run-lease directories. Tests cover
  matching preview indexes, explicit UI selection, exact chosen weights reaching
  the real validation/workflow code with a mocked image client, preservation of
  the original result, changed-weight rejection, and recovery after interruption.
  These tests do not prove new GPU generation.
- Ruff, Mypy, and Pyright passed. A separate repeat of the native suite collided
  with an earlier mock server on port 8765; that failure was preserved, the owned
  test server was stopped, and the affected integration test then passed.

Evidence lives in `dist/checkpoint-selection/`: build and test logs, captured
native UI structure, settings, and the configured launcher. Live recovery records
are in `dist/milestone-20260916/workspace/configs/` under names beginning
`hd_sdxl_acceptance_20260916_v2_step1750_`.
