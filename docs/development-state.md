# Development review — 2026-09-16

The review started from clean commit `d7875dd` (MIVIM 0.6). The repository
already implements the guided SDXL desktop journey: connection discovery,
photo preparation, trainer orchestration, validation, recovery, and ComfyUI
workflow publication. The next useful work is tightening that existing journey.
Video and automatic strength selection remain outside the current delivered
scope. Checkpoint comparison was subsequently implemented; see the follow-up
at the end of this document.

## Selected entry point: prepared-photo integrity

**P2 correctness defect, reproduced and fixed:** an extra image/caption pair
could enter training without appearing in the preparation record.

- **Required behavior:** prepared reuse and training must verify the images
  against the preparation record, while allowing users to edit captions.
- **Trigger:** prepare a project, then add `extra.png`, `002.jpg`, or
  `holiday.WEBP` with a valid matching caption to its prepared folder.
- **Disagreement:** `prepare_dataset.prepared_output_manifest_issue` checked
  only numbered lowercase PNGs. `dataset_pairs.scan_dataset_pairs` and
  `train._validate_dataset_pairs` accepted other supported images too.
- **Consequence:** `app_service.inspect_prepared_dataset` approved reuse and
  the trainer could materialize a configuration for the expanded dataset.
  Valid captions and unchanged source photos did not prevent this.
- **Related rebuild behavior:** rebuild replaced numbered generated pairs but
  retained the additional photos, so rebuilding did not remove the mismatch.

The integrity check now uses the trainer's shared image inventory. Added
supported images are rejected by readiness and again before the training
configuration is written. Preparation also refuses unmanaged photos before
changing existing output, including when rebuild is requested. Its message
explains how to move those photos or include them through source preparation.
The desktop's existing next-version action provides a recovery that preserves
the original project. Editable captions and ordinary manifest-free CLI
datasets keep their existing behavior.

## Verification

- Eight new service/core regression cases failed against the original source
  with the expected incorrect acceptance, then passed after the fix.
- A ninth regression clicks through the desktop review, checks the named
  photo error and disabled Start, uses **Use the next version name**, and
  verifies the previous files remain unchanged.
- Source suite: **288 passed, 1 deselected**. The deselected test runs an
  existing packaged executable; that binary was not rebuilt for this change.
- Ruff lint and formatting, Mypy, Pyright, and `git diff --check` passed.
- Local regression and suite output is retained under `dist` in
  `review-regression-before.txt`, `review-regression-after.txt`, and
  `review-suite.txt`.

Early test attempts encountered a shared temporary-directory collision and
sandbox restrictions on user-level caches and run leases. The successful runs
used separate repository-local temporary, cache, and lease directories; those
environment errors are not counted as product regressions.

## Review boundary and next work

This was a correctness review of preparation, manifest verification, readiness,
and the training input boundary, supported by the broader source suite. The
shared pair scanner, preservation of editable captions, and existing desktop
recovery actions provided useful foundations for the correction. It was not an
exhaustive review of every asynchronous connection or cancellation path.

Live Comfy Desktop generation, GPU training, manual workflow import, and a new
Windows package remain unverified by this change. The next acceptance milestone
is one complete person-SDXL run through the desktop, including result handoff,
before expanding the supported model or media scope.

## Next development slice: caption review ownership

**P2 correctness defect, reproduced and fixed:** the prepared-caption editor
did not participate in the dataset leases held by preparation and training.
Its atomic file writes prevented partial captions but did not stop another
Studio window from changing training inputs during a run.

`MainWindow.review_prepared_dataset` now acquires the displayed dataset's
lease before constructing the editor and holds it until the modal review
returns. Canceling an unsaved-close prompt keeps the lease. Done, Escape,
window close, and opening failures release it. Busy datasets show an error
instead of opening for editing. A competing run names the caption review it
needs the user to close. The lease targets the displayed result's path even
if the current workspace and project form have changed.

Five new desktop cases reproduced the missing exclusion and unhandled opening
failure before the correction, then passed. They use real OS leases and a
separate process entering the public trainer; only its GPU execution body and
remote-prompt check are replaced. They verify both contention orderings,
unchanged files on rejection, retry after release, unrelated-dataset access,
and the editor's close/cancel paths. This proves process coordination, not a
GPU training run. Evidence is in `dist/caption-review-before.txt` and
`dist/caption-review-after.txt`.

After this slice, the source suite passes **293 tests, with 1 packaged-binary
test deselected**. Ruff lint/formatting, Mypy, Pyright, and the whitespace check
also pass. The full suite output is retained in `dist/caption-review-suite.txt`.
At that point the executable had not been rebuilt; live Desktop/GPU acceptance
remained a separate milestone.

## Completed milestone: Windows package and Desktop/GPU acceptance

The September 16 acceptance run is now complete. Native execution found and
fixed a packaged worker-launch defect and an SDXL sample negative-prompt
default. The rebuilt Studio completed 2,000 training steps on the RTX 5090,
installed the resulting LoRA, rendered all 16 validation cells, and verified
the result. Its published workflow opened and generated another image through
Comfy Desktop's native Run button. All 14 source photos remained unchanged.

The combined source/package suite passes **297 tests**, including a new native
Windows test that launches the packaged preparation worker through Studio's
controls. Mypy, Pyright, Ruff, formatting, and whitespace checks pass.
See [the acceptance report](windows-gpu-acceptance-20260916.md) for exact scope,
environment, retained failures, artifacts, and the configured project launcher.
# Checkpoint comparison follow-up

Implemented the next user-facing checkpoint-selection capability: Results can
compare earlier/final previews and validate a deliberate choice under a separate
version name. The Windows package has been rebuilt with this feature. Native
selection and installation were exercised; new GPU generation remains pending
because ComfyUI was offline. See [checkpoint selection](checkpoint-selection.md)
for the exact behavior and verification boundary.
