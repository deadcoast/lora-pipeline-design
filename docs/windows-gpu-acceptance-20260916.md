# Windows package and SDXL acceptance — September 16, 2026

> Historical build outputs and logs under `dist/` cited below were removed on
> 2026-10-13. Only `dist/release` and `dist/milestone-20260916` remain. See
> `CHANGELOG.md` for the release history.

## Scope and current result

**Passed on this Windows machine.** The rebuilt Studio completed preparation,
2,000 GPU training steps, LoRA installation, all 16 validation images, verified
result display, and workflow publication using the user-selected
`C:\Users\ryanf\Desktop\hd` folder. The published workflow also opened in
Comfy Desktop and generated an additional image through its native Run button.

The current project is `hd_sdxl_acceptance_20260916_v2`, using the recommended
Balanced plan: 2,000 steps, rank 16, 1024 px preparation, and validation at
strengths 0, 0.6, 0.8, and 1.0. The input contains 14 readable images with no
exact duplicates. Eight images require enlargement. Source hashes were saved
before preparation; all 14 originals matched their paths, sizes, and SHA256
hashes after the run.

The final model contains 2,166 finite tensors, records step 2,000 / epoch 71,
and matches its installed copy byte for byte. All 16 validation cells are
1024×1024 and match their manifest hashes. The 1816×1642 comparison grid and
embedded API/UI workflows were inspected. This verifies the local execution
and handoff path; the grid remains the review artifact for likeness and prompt
flexibility, rather than a quantitative identity-quality score.

The final LoRA is `hd_sdxl_acceptance_20260916_v2.safetensors` (85,438,532 bytes),
raw SHA256 `95086854f7a615229bdc7e5cd45b9e65721ae08ebb00ef9493139fc943dc1930`.
The native Comfy Desktop run completed as prompt
`cb5378c8-82e9-4a9e-9cf5-22f6235249a2` and produced `validate_00017_.png`.
Its KSampler executed; only checkpoint loading and the empty latent were cached.
Observed GPU memory peaked at 16,420 MiB during the monitored pipeline.

Reopen this configured Studio/project with
[`Open Studio.cmd`](../dist/milestone-20260916/Open%20Studio.cmd).
The result is in
[`hd_sdxl_acceptance_20260916_v2_grid.png`](../dist/milestone-20260916/workspace/grids/hd_sdxl_acceptance_20260916_v2_grid.png).
The complete verification receipt is
[`acceptance-receipt.json`](../dist/milestone-20260916/acceptance-receipt.json).

## Defects found through native execution

1. **Packaged worker launch:** readiness passed, but Prepare photos failed
   with `FailedToStart`. Nuitka 4.0 standalone sets `sys.executable` to a
   `python.exe` inside the distribution, which this package does not ship.
   Packaged preparation, training, and validation retry now relaunch the
   actual executable named by `sys.argv[0]`. Source execution continues to
   use Python. The support log now records the worker program and directory.
2. **SDXL preview defaults:** the first real trainer attempt loaded the base
   model and cached image latents, then failed when generating baseline
   previews. Current ai-toolkit defaults an omitted sample negative prompt
   to `None`, which its SDXL tokenizer rejects. The shipped template now sets
   `sample.neg` to an empty string. The corrected run generated four baseline
   previews and entered the optimization loop.

The failed attempts and their outputs remain available. The successful retry
uses a separate versioned project. No failed run is counted as acceptance.

## Environment and provenance

- Windows package: `dist/MIVIMLoRAStudio.dist/MIVIMLoRAStudio.exe`, version
  0.6.0.0. The whole distribution folder is required. This local build is
  unsigned.
- Trainer: isolated uv environment at `dist/acceptance-tools/ai-toolkit`,
  [upstream revision e65c4d0](https://github.com/ostris/ai-toolkit/tree/e65c4d0fb69251e692390574c49873297dc4bae5),
  Python 3.12.10.
  Its `pyproject.toml` and `uv.lock` retain the resolved dependencies.
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB. The exact trainer environment
  executed a CUDA kernel before training: PyTorch 2.13.0+cu130, CUDA 13.0.
- SDXL base: official [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), revision
  `462165984030d82259a11f4367a4eed129e94a7b`;
  checkpoint SHA256
  `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b`.
  The downloaded file and installed ComfyUI copy both match the published
  LFS object hash.
- Comfy Desktop 1.0.47: local `ComfyUI` instance `inst-1786443116240`, API at
  `127.0.0.1:8188`, ComfyUI 0.36.0 / PyTorch 2.12.1+cu130 / Python 3.13.12.
  Training and validation use the same verified checkpoint.

The first trainer dependency sync failed because the official CUDA wheel
index omitted the Windows torchvision wheel hash while exposing Linux hashes.
The isolated uv project pins the official Windows wheel URLs and records their
actual content hashes. The failed lock and install logs are retained. No Comfy
Desktop Python environment was modified.

## Evidence and verification boundaries

Local evidence lives in `dist/milestone-20260916/`: build receipts, saved
settings, source hashes, native window captures, GPU samples, Studio logs, and
the separate project workspace. Initial build/install failures are also kept
under `dist/milestone-*.log`.

The complete test suite passes **297 tests**, including all three package
tests. Mypy and Pyright pass for source and tests; Ruff lint, formatting, and
the Git whitespace check also pass. `dist/milestone-all-tests.txt` retains the
combined run. The package's external training template was updated to match
the corrected source template byte for byte after the executable rebuild;
the build receipt records both executable and template hashes.

The package tests now include native Windows UI automation that starts Studio,
clicks through review and preparation, waits for its real worker to produce
the result, and verifies prepared image/caption pairs and unchanged originals.
This test needs the compiled package and a Windows desktop. Its synthetic input
proves the packaged worker path; the user's photo run supplies GPU acceptance.

An offline GUI test also exposed reliance on a real ComfyUI server being
absent. Its API response is now explicitly mocked offline so a running Desktop
instance does not change that test's result.
