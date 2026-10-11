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

## Model profiles and research correction — 2026-10-11

Added explicit NoobAI Epsilon and V-pred profiles to Connect, persisted worker
settings, training configuration, validation graphs, editable ComfyUI export,
cache signatures, and recovery binding. Standard SDXL remains the default.
The initial V-pred compatibility gate was corrected in **0.6.1**: an isolated
import adapter changes the known image-tensor velocity target to encoded
latents without editing the external checkout. Unknown implementations remain
blocked, and source drift is rejected before executing the trainer module.
The filename mismatch check now ignores parent-directory names.

The corrected source suite passes **324 tests**, with **2 packaged tests
deselected**. The new executable separately passes **4 packaging tests** after
adding V-pred result-handoff coverage. Native preparation preserves the V-pred
profile; standard and V-pred results render correctly with fixture artifacts.
Ruff, formatting, Mypy, Pyright, and whitespace checks pass. The actual installed
trainer reproduced the original cached-latent failure; its corrected loss and
backward gradients pass four analytical cases using synthetic CPU tensors.

Package: `dist/noobai-0.6.1/MIVIMLoRAStudio.dist/MIVIMLoRAStudio.exe`.
The earlier package is preserved. A moved-environment build-launcher defect was
also fixed. The original old-binary schema failure and first rebuild failure
remain documented. Real NoobAI-weight training, ComfyUI generation, and model
quality remain unqualified. See [model support](model-support.md) for sources,
reproduction commands, and evidence paths.

## Comfy Desktop connection flow correction — 2026-10-11

The reported blocked connection was reproduced with Desktop 1.1.6 open at its
instance dashboard. The selected local `ComfyUI` instance had no running server
or runtime port record. MIVIM's generic **NOT RUNNING** state did not make the
next action clear, and its launch retry loop stopped after 18 checks. Repeated
launch clicks also created overlapping retry chains.

**0.6.2** names the selected instance card to open in Desktop, distinguishes
**START INSTANCE**, **CONNECTING**, and **CONNECTED**, and keeps one automatic
check active while connecting. It preserves the selected instance boundary;
connection requires that instance's actual API response. Setup checks cannot
overlap, background retries preserve the visible state, and the logs retain
candidate addresses and probe failures. Readiness and Results use the same
launch guidance and resume after connection succeeds.

Live acceptance opened the installed local instance through Desktop's own UI.
MIVIM then detected `127.0.0.1:8188`, one checkpoint, and three LoRAs. The rebuilt
Windows application showed **CONNECTED** with those values. A second normal
launch using the user's settings was verified through Windows accessibility
and left open, with ComfyUI still running. No model generation was performed.

- Source suite: **326 passed, 3 packaged cases deselected**.
- Rebuilt package suite: **4 passed**, covering the icon, native photo
  preparation, and standard/V-pred result handoff.
- Ruff lint/format, Mypy, Pyright, whitespace checks, wheel build, and static
  UI audit passed. Native screenshots cover the startup instruction at
  1040 × 740, its sibling Project page, and the live connected package.
- Package: `dist/comfy-connection-0.6.2/MIVIMLoRAStudio.dist/MIVIMLoRAStudio.exe`.
- SHA-256: `782EDF6819586C0901201DDC118E38E214A62682815CA11FB33B47FBC3A63221`.
- Evidence: `dist/comfy-connect-source-final.log`,
  `dist/comfy-connection-package-tests.log`, `dist/comfy-connection-live.png`,
  `dist/comfy-connection-start-instance.png`, and
  `dist/comfy-connection-sibling-project.png`.

The earlier targeted run caught two remaining references to the removed retry
counter in Results; these were corrected before the passing source/package
runs. Its failure remains in `dist/comfy-connect-regression.log`. Previous
package directories were preserved. No manual edits were made to Comfy Desktop
source or settings; Desktop managed its own normal startup state.

## Anime workflow feature review — 2026-10-11, 0.7.0

The review found three product gaps: source tag captions were discarded,
NoobAI still used photographic review prompts, and training length could not
be changed independently of the plan. All three are now implemented in the
desktop and underlying pipeline:

- Import matching UTF-8 sidecars before caption generation, retain tag order,
  and prepend only a missing trigger. Project shows caption coverage; the
  existing editor remains the place to adjust prepared captions.
- Use matching anime face/full-body/profile/action shots during bundled
  training and final comparison, at 832×1216, with profile-specific sampling.
  Custom training YAML preserves its own prompts.
- Persist a per-project step override, pass it through both CLI entry points,
  and shorten save/sample intervals for short experiments. The form estimates
  presentations per source image, and zero restores the current plan default.
- Skip photographic blur warnings in anime mode, preserve aspect ratio, and
  label the longer plan **Extended training** rather than promising quality.

The built Windows app prepared a 12-image mixed-caption fixture through its
normal controls: ten tag captions imported (including a UTF-8 BOM and a caption
already containing the trigger), two trigger-only fallbacks, unchanged source
files, and preserved aspect ratios. Its caption editor displayed the imported
tags. Native step editing showed 400 → 600 → Plan default (2,000) → 400.
Both NoobAI config outputs use 400 steps, 100-step snapshots, and the same
four prompts/dimensions/negative prompt as their exported ComfyUI workflows.
The model-file placeholder used for config checks is explicitly not weights;
no GPU training or generated-image quality claim is made.

Existing source checks: 326 passed, three packaged cases deselected. Rebuilt
package checks: four passed. Ruff, Mypy, Pyright, wheel build and whitespace
checks passed. Project screenshots were inspected at the normal size and at
1040×740 with the form scrolled to its actions. No new test suite or security
gate was added. Earlier checks caught outdated copy assertions and accidentally
ran the older default package; those logs remain available.

- Executable: `dist/anime-core-0.7.0-final/MIVIMLoRAStudio.dist/MIVIMLoRAStudio.exe`.
- SHA-256: `13B9C3FE4319509D666CC78171764CE3A41CA869088F0B06C1F77C3D2A0E1D9A`.
- Feature record: `dist/anime-core-acceptance/feature-results.json`.
- UI: `dist/anime-core-native-controls.log`, `dist/anime-core-project-final.png`,
  `dist/anime-core-project-minimum.png`.
- Checks: `dist/anime-core-source-release.log`, `dist/anime-core-package.log`.

The next useful feature gaps are automatic anime tagging and editable preview
prompt sets. Existing source imports and the four built-in shots do not imply
those features are already present. Actual NoobAI training quality still needs
a run with model weights and visual review of its saved checkpoints.
