# Changelog

One entry per release. Build outputs, logs, and screenshots that earlier
versions of this file cited under `dist/` were removed on 2026-10-13; only
`dist/release` (the current package) and `dist/milestone-20260916` (the
September 16 GPU acceptance evidence) remain.

## 0.9.0 — 2026-10-13

- **Library page**, the home page once a run exists: every run grouped by
  character with an art-style chip, base model and profile, trigger, steps,
  date, grid thumbnail, and whether the LoRA is still installed. Open any
  result, train the next version with the same setup, or tag older runs that
  wait under Unsorted. Built from the result records on disk; nothing is hashed.
- **Character** and **Art style** fields on New project: one word each, with
  autocomplete from every word used before and a hint that says whether the
  word is new. Both are required to review or start a run. The project name
  fills as `<character>-<style>-mk<NN>` until you type your own.
- Connect names each checkpoint's purpose (**MODEL :: Anime · NoobAI-XL ·
  Epsilon-pred**) in its tooltip and under the selected model, labels unknown
  models in one word, and selects the profile the filename calls for.
- Result records (schema 5) carry `character`, `artstyle`, and `source_dir`;
  `src.pipeline` accepts `--character` and `--artstyle`. `library.json` in the
  workspace holds the vocabulary, a style per checkpoint, and tag overrides.
- Fix the Review & run nav label, which rendered without its ampersand.
- Stop tracking the `ruth_db` training images and remove them from history.
  The folder stays on disk and is ignored.
- Replace the development-state narrative with this changelog.

## 0.8.2 — 2026-10-11

- Recognize SDXL base architecture from the local checkpoint's safetensors
  header; manual confirmation is only needed when it cannot be identified.
- Discover the installed trainer under the per-user application-data folder.
  Trainer probes run in isolated mode from the trainer's own directory, so the
  packaged Python 3.13 no longer leaks into the trainer's Python 3.12 environment.
- Build to `dist/release`. `scripts/install_windows.ps1` installs to
  `%LOCALAPPDATA%\Programs\MIVIM LoRA Studio`, creates shortcuts, and can copy
  an existing ai-toolkit checkout with `-TrainerDirectory`.
- Readiness cards place long action buttons below their explanation.

## 0.8.1 — 2026-10-11

- Encode long SDXL captions in complete CLIP sections inside the training
  worker instead of discarding trailing tags. Applies to the standard and
  V-pred launchers without editing the ai-toolkit checkout.

## 0.8.0 — 2026-10-11

- Automatic anime tagging with WD ViT v3 (ONNX on CPU) for uncaptioned images
  in NoobAI projects, on by default, with a Project checkbox to disable it. The
  model downloads once into the user's MIVIM cache.
- CLI: `--caption-mode auto|anime|template|blip` on `src.pipeline` and
  `src.prepare_dataset`.

## 0.7.0 — 2026-10-11

- Import existing UTF-8 `.txt` captions beside source images, preserving tag
  order and adding the identity phrase only when missing. Project shows
  caption coverage.
- Anime preview and comparison shots (face, full body, profile, action at
  832×1216) for NoobAI profiles.
- Training steps control independent of the plan; `--steps` on `src.train`
  and `src.pipeline`.
- Skip photographic blur heuristics and preserve aspect ratio for anime
  profiles.

## 0.6.2 — 2026-10-11

- Connect names the Desktop instance card to open and distinguishes
  START INSTANCE, CONNECTING, and CONNECTED. One automatic check stays active
  while connecting instead of stopping after 18 retries.

## 0.6.1 — 2026-10-11

- Model profiles: SDXL Standard, NoobAI-XL Epsilon-pred, and NoobAI-XL V-pred,
  selected in Connect and carried through training, validation, exported
  workflows, and recovery.
- V-pred: an isolated import adapter corrects ai-toolkit's image-tensor
  velocity target and enables zero-terminal-SNR. Unrecognized trainer sources
  are blocked. See `docs/model-support.md`.

## 0.6.0 — 2026-09-16

- Windows package and real GPU acceptance: a 2,000-step SDXL run, LoRA
  install, 16-cell validation grid, and workflow publication. See
  `docs/windows-gpu-acceptance-20260916.md`.
- Compare saved training checkpoints and validate a chosen checkpoint under a
  new version name. See `docs/checkpoint-selection.md`.
- Prepared-dataset integrity: added images are rejected before reuse and
  training; caption review holds the dataset lease.
- Packaged worker launch and SDXL sample negative-prompt defaults fixed.

## 0.3 to 0.6 — 2026-07-24

- Casual-first PySide6 desktop, read-only Comfy Desktop discovery, verified
  ComfyUI handoff, and workflow publishing on top of the original CLI pipeline.
