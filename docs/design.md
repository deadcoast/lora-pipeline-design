# LoRA Training Pipeline — Design Document

**Goal:** Build a reusable, guided Windows application to train LoRAs on
**photos** (Phase 1), then extend the same pipeline to **video** (Phase 2).
**Hardware:** Ryzen 9800X3D · 64GB DDR5 · RTX 5090 (32GB VRAM, Blackwell) · Windows 11 — this is top-tier for local training; VRAM is not a constraint.
**Primary audience:** Casual users who need the application to discover,
explain, validate, and recover the workflow for them. Technical controls remain
available without defining the default experience.

---

## Product direction

The supported product surface is **MIVIM LoRA Studio**, a PySide6 Windows
companion for Comfy Desktop. CLI modules remain automation adapters and the
domain implementation, but casual users are not expected to invoke them.

The default journey is:

1. Detect and connect a local Comfy Desktop instance.
2. Choose and review source photos.
3. Select a plain-language training plan.
4. Resolve readiness failures through user-facing actions.
5. Review prepared thumbnails and edit captions when preparation is separated.
6. Monitor or stop the real pipeline.
7. Review the verified baseline grid and add the exact workflow to ComfyUI.

Comfy Desktop state is read-only discovery input. The studio stores its own
approved connection and workspace settings, validates the live local API, and
never modifies Desktop's installation registry or managed application files.
Publishing a result uses ComfyUI's supported profile-scoped user-data API,
never a guessed Desktop filesystem path.

---

## 1. Mental model: what each tool actually does

The single most important thing to internalize is that **ComfyUI is an inference engine, not a trainer.** It's brilliant at *running* diffusion models through a node graph and at *validating* your LoRA. Training is a different job, best handled by a dedicated trainer.

So the pipeline has two engines that talk to each other:

| Job                        | Tool                                    | Why                                                                                                                                                                                      |
|----------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training** the LoRA      | **ai-toolkit** (ostris)                 | One tool covers SDXL, Flux, *and* Wan 2.2 video. YAML configs, a web UI, and a `run.py` you can script. MIT licensed. This is what makes photos→video a small step instead of a rewrite. |
| **Inference / validation** | **ComfyUI Desktop** (already installed) | Generate test images from your trained LoRA, run XY comparison grids, build the "does this LoRA work?" loop.                                                                             |
| **Orchestration**          | **MIVIM LoRA Studio**                   | Guides setup and photo review, owns run safety and recovery, launches training, then drives ComfyUI's API to generate and publish verified validation work.                              |

> **Why not train inside ComfyUI directly?** You *can* — nodes like `comfyUI-Realtime-Lora` and `Lora-Training-in-Comfy` wrap trainers as nodes. It's great for quick experiments and worth trying early to see training happen visually. But for a *reusable application pipeline* that scales to video, a dedicated trainer + scripted orchestration is cleaner, more reproducible, and far easier to debug. We'll use the ComfyUI training nodes as a learning aid, and ai-toolkit as the production trainer.

### The flow

```text
   photos/             shared local SDXL model              ComfyUI (API)
 ┌───────────┐   prep    ┌────────────────────┐  .safetensors ┌──────────────┐
 │ raw images│ ───────▶ │ dataset (img+txt)  │ ───────▶      │ load LoRA +  │
 │           │  caption  │ + training config  │   train       │ generate     │
 └───────────┘           └────────────────────┘               │ validation   │
       ▲                          │                           │ grid         │
       │                          ▼                           └──────┬───────┘
       │                    checkpoints/                             │
       └──────────────── review grid, adjust, retrain ◀─────────────┘
                          (your Python orchestrator drives this whole loop)
```

---

## 2. Environment setup — the Blackwell (RTX 5090) check

RTX 5090 support requires a PyTorch wheel built with CUDA 12.8 or newer.
Stable PyTorch has supported `sm_120` since 2.7, so use the current stable
command from PyTorch's install selector rather than pinning this project to an
old nightly URL.

**Checks that matter for you:**

1. Install a current CUDA 12.8+ PyTorch build in every environment that touches
   CUDA.
2. Confirm `torch.cuda.get_arch_list()` contains `sm_120`.
3. Avoid optional compiled attention packages unless their wheels explicitly
   match the installed PyTorch/CUDA versions; PyTorch SDPA is the safe default.
4. Keep NVIDIA drivers current.
5. Keep training and ComfyUI in separate Python venvs. They have conflicting
   dependencies; isolating them prevents one from breaking the other. The Comfy
   Desktop app manages its own runtime — leave it alone and run ai-toolkit in
   its own venv.

> Windows note: everything below runs natively on Windows 11. If you ever hit a dependency that's Linux-only, **WSL2** is the fallback — but you shouldn't need it for SDXL/Flux/Wan LoRA training.

---

## 3. Repository layout

The repository contains the desktop product and CLI core. User projects default
to a versioned writable application-data workspace; ai-toolkit and Comfy
Desktop remain separate installations selected by the user.

```text
lora-pipeline-design/
├─ pyproject.toml / uv.lock            # app, test, and locked deploy dependencies
├─ configs/
│  └─ person_sdxl.yaml                 # reusable SDXL training template
├─ src/
│  ├─ desktop.py                       # guided application and real worker lifecycle
│  ├─ app_models.py / app_settings.py  # typed state and atomic persistence
│  ├─ app_service.py                   # readiness, repairs, and command construction
│  ├─ comfy_desktop.py                 # Desktop discovery and exact model roots
│  ├─ run_guardian.py                  # OS project/resource/GPU leases
│  ├─ process_control.py               # UI → worker → trainer crash ownership
│  ├─ comfy_activity.py                # durable remote ComfyUI prompt ownership
│  ├─ prepare_dataset.py / train.py    # prepared pairs and ai-toolkit launch
│  ├─ comfy_client.py / validate.py    # API execution and verified result workflow
│  └─ pipeline.py                      # prep → train → install → validate
├─ workflows/
│  └─ validate_sdxl_api.json           # shipped API-format execution template
└─ scripts/
   └─ build_windows.ps1                # locked standalone Windows package
```

---

## 4. Phase 1 — Photos, step by step

### Step 1: Dataset preparation (the part that actually determines quality)

LoRA quality is ~80% dataset, ~20% settings. For a **person/identity** LoRA aim for 15–40 sharp, varied photos (angles, lighting, expressions, backgrounds); for a **style** LoRA, 30–100 images that share the look but vary the subject.

`prepare_dataset.py` should:

- Convert everything to PNG, strip alpha, fix orientation.
- Resize so the long edge is ~1024px (SDXL) — ai-toolkit can bucket mixed aspect ratios, so you don't have to hard-crop to squares.
- **Auto-caption** each image to a matching `.txt` file. For a person LoRA use a **rare trigger token** (e.g. `ohwx person`) plus light description; for style, describe *contents* and let the style be the constant. A vision model (Florence-2 / BLIP via a ComfyUI node, or a captioning script) gets you 90% there; hand-edit the rest.
- Print a report: count, resolution spread, any corrupt files, caption length distribution.

### Step 2: Training config (ai-toolkit)

Copy an example from `ai-toolkit/config/examples/` and adapt. A sensible SDXL starting point:

```yaml
# configs/person_sdxl.yaml  (abridged — real file has more comments)
job: extension
config:
  name: "subject_v1"
  process:
    - type: sd_trainer
      training_folder: "outputs/checkpoints"
      device: cuda:0
      trigger_word: "{trigger}"
      network:
        type: lora
        linear: 16          # rank — 16 is a solid default; 8 for style, 32 for hard subjects
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250     # checkpoint cadence -> lets you pick the best, not the last
      datasets:
        - folder_path: "datasets/subject_v1"
          caption_ext: "txt"
          resolution: [1024]
      train:
        batch_size: 1
        steps: 2000         # ~1500–2500 for a person; watch for overfitting
        lr: 1.0e-4
        optimizer: adamw8bit
        dtype: bf16
      model:
        name_or_path: "stabilityai/stable-diffusion-xl-base-1.0"
        arch: "sdxl"
        is_xl: true
      sample:
        sample_every: 250   # ai-toolkit renders preview images DURING training
        samples:
          - prompt: "{trigger}, cinematic portrait, soft light"
          - prompt: "{trigger}, standing in a city street, 35mm"
```

Two habits that save you: **save checkpoints often** (`save_every`) so you can pick the epoch *before* it overfit, and use ai-toolkit's built-in **sampling during training** to watch it learn in real time. Runtime varies materially with the model, trainer version, plan, dataset, GPU, and memory pressure; the application does not promise a fixed duration.

You can drive training either from the **ai-toolkit web UI** (`http://localhost:8675` — start/stop/monitor jobs visually) or from code:

```python
# src/train.py
exit_code = run_managed_process(
    [trainer_python, "run.py", "configs/person_sdxl.yaml"],
    cwd=AI_TOOLKIT_DIR,
)
```

The production path creates the trainer atomically inside a Windows
kill-on-close Job Object. The pipeline worker also watches a Studio-owned
lifetime mutex, so closing or hard-killing the UI releases the complete process
tree rather than leaving an invisible GPU job.

### Step 3: Validation via ComfyUI's API

The repository ships `workflows/validate_sdxl_api.json`, an API-format template
using only core nodes. `validate.py` injects the exact checkpoint, LoRA, prompt,
strength, and one fixed seed, then submits the prompt through the implemented
client in `src/comfy_client.py`. It waits for this prompt's WebSocket completion,
reads its history, and downloads its images.

Successful validation stitches the prompt × strength sweep into a comparison
grid. It also generates `<project>_use_in_comfyui.json` as an editable ComfyUI
Workflow JSON v0.4. Before approval, the code reconstructs the API prompt from
that UI graph and requires exact equality with what ran. The PNG embeds the
executed API prompt under `prompt` and the editable graph under `workflow`.
Workflow JSON 1.0 is the newest documented schema, so 1.0 export is not claimed.

### Step 4: The orchestrator

```bash
python -m src.pipeline \
  --src "C:/photos/me_raw" \
  --name person_v1 \
  --config configs/person_sdxl.yaml
```

The orchestrator prepares the named dataset, materializes a concrete ai-toolkit
config, launches training, accepts only a checkpoint created or changed by that
run, copies it into ComfyUI's LoRA folder, and renders the comparison grid.
Everything after this is tuning knobs (rank, LR, steps, dataset) against the
grids.

---

## 5. Phase 2 — Video (the payoff of choosing ai-toolkit)

Because ai-toolkit already trains **Wan 2.2** (and LTX-2) video LoRAs, the
orchestration structure can be reused. What changes:

- **Dataset**: short clips (or image→video pairs) instead of stills; captions describe motion. `prepare_dataset.py` grows a video branch (frame extraction, clip length, fps normalization).
- **Config**: point ai-toolkit at a Wan 2.2 config template instead of SDXL; expect longer runs and heavier VRAM use (your 32GB handles Wan 2.2 14B LoRA with offload settings).
- **Validation**: build a second ComfyUI workflow for Wan 2.2 video generation,
  export it as API JSON, generalize `comfy_client.py` to collect video artifacts,
  and add video-specific validation output instead of a PNG contact sheet.

The orchestrator and folder structure stay stable, while artifact collection
and validation become media-aware.

---

## 6. Suggested learning roadmap (milestones)

1. **Prove the hardware.** Get ComfyUI Desktop generating a plain SDXL image with a CUDA 12.8+ PyTorch build whose architecture list includes `sm_120`. No LoRA yet — just confirm the 5090 renders without kernel errors.
2. **Train one LoRA the easy way.** Install `comfyUI-Realtime-Lora` and train a tiny SDXL LoRA *inside* ComfyUI on ~10 photos. Goal: see training happen and understand the knobs, not quality.
3. **Stand up ai-toolkit.** Separate venv, Blackwell-compatible stable PyTorch, web UI reachable at :8675. Retrain the same 10 photos via a YAML config. Compare.
4. **Automate validation.** Export an API-format workflow, write `comfy_client.py`, auto-generate a validation grid from a trained LoRA.
5. **Close the loop.** Wire `pipeline.py` so one command does prep → train → validate. This is your v1 application.
6. **Then, and only then, video.** Swap in a Wan 2.2 config and a video validation workflow.

Ship each milestone before starting the next — it keeps failures isolated to one new variable.

---

## 7. Common pitfalls to avoid

- **Skipping the Blackwell fix** → CUDA kernel errors that look like broken installs. Do §2 first.
- **`xformers` sneaking back in** via a custom node and downgrading PyTorch. Pin PyTorch; prefer SDPA/SageAttention.
- **Shared venv** for ComfyUI + trainer → dependency hell. Keep them separate.
- **Weak/inconsistent dataset** → no amount of setting-tuning fixes it. Invest here.
- **Only keeping the final checkpoint** → you can't recover the pre-overfit sweet spot. Save every N steps.
- **Bad/duplicated captions** → trigger token doesn't "bind." Rare token + clean captions.
- **Editing the desktop app's bundled Python** for dev work → fragile. Develop custom nodes against a clean install if you go that route.

---

## Current scope decisions

- Phase 1 is implemented as a guided desktop application for a single
  person/identity LoRA on SDXL.
- Current Comfy Desktop 2 registration, executable-drive/default storage,
  multi-instance state, shared/private model roots, launch arguments, generated
  model YAML, install-local `extra_model_paths.yaml`, output model folders,
  live-process port locks, and legacy layouts are discovered read-only.
  Dead-process locks and unprovable running snapshots are rejected.
- The app owns versioned per-user settings and a writable project workspace.
- Fast, Balanced, and High Quality plans change real rank, steps, resolution,
  preview cadence, and validation breadth.
- Setup checks, preparation-only operation, background logging, OS-level
  resource/GPU leases, Studio-owned worker lifetime, atomic Windows trainer
  Job membership, result verification, and last-result restoration are
  implemented in the user-facing application.
- Validation emits a prompt, client, and host ownership marker. Stop verifies
  that identity against ComfyUI's live queue before deleting a pending prompt.
  A running prompt receives a prompt-ID-targeted interrupt only on ComfyUI
  0.3.56 or newer. Older/unknown versions and absent/mismatched ownership are
  safe no-ops toward ComfyUI; the worker remains attached with instructions,
  and closing is refused until a safe stop can be confirmed.
- Prompt reservations are persisted before submission and rechecked at every
  accelerator entry point. Unreadable markers, incomplete queue schemas, and
  unidentified queue entries fail closed; same-client submissions use distinct
  records so one cleanup cannot erase another.
- Background logging is a bounded, rotating, local-only support trail beside
  the user's settings. It records setup/readiness outcomes, recoverable worker
  failures, and real pipeline output without dumping environment variables or
  photo contents. The run monitor exposes the folder through **Open local
  logs**; no upload is automatic.
- Prepared projects are validated against their generation manifest, exposed in
  a thumbnail/caption editor, and resumed with `--skip-prep` after preparation
  or a later-stage failure. The review and trainer consume one shared supported
  image/`.txt` inventory, and filename-specific structural failures block
  process launch. The authoritative trainer re-hashes the reviewed generated
  images immediately before config materialization.
- LoRA installation writes a versioned recovery record before validation starts.
  If validation fails or the studio closes, the next launch restores a
  validation-only retry and never repeats completed preparation or training.
  Recovery is attempt-bound to the original Desktop installation, LoRA and
  checkpoint paths/fingerprints, strengths, seed, and validation signature.
  The launch-time host is recorded as execution evidence, while a later dynamic
  port is accepted only after the durable installation and model identities
  match. Startup and Retry hash large models outside the GUI thread and keep
  Retry disabled until they match.
- Validation includes a no-LoRA baseline and plain-language prompt/strength
  axes. Each verified cell is resumable from an attempt-isolated atomic
  manifest only when the full signature, including both model identities,
  matches; a changed input receives an isolated attempt. The verified result
  record uses the immutable launch snapshot and carries the installed LoRA,
  identity phrase, checkpoint, tested default strength nearest 0.8, and an
  editable core-node Workflow JSON v0.4. The PNG separately embeds the exact
  API prompt and equivalent UI workflow. Model hashing and result-action gates
  run outside the GUI thread. The Results page publishes the verified JSON
  through ComfyUI's profile-scoped user-data API, preserves conflicting files
  with content-fingerprinted names, reads the saved bytes back, and retains a
  manual JSON fallback.
- Training configs are reusable templates; the orchestrator injects each run's
  name, dataset path, selected Comfy Desktop checkpoint file, output path, and
  selected preset before launching ai-toolkit.
- Readiness executes the selected trainer Python, identifies the installed
  GPU's compute capability, and completes a real CUDA kernel before approving
  training. This catches CUDA-visible but architecture-incompatible wheels
  (notably Blackwell `sm_120`) while retaining support for older GPUs.
- Output collision checks cover the prepared dataset, exact project
  checkpoints, validation work/grid, trainer and result records, and the
  installed ComfyUI LoRA. The app requires a new version name or an explicit
  rebuild decision, while a successful prepare-only result remains resumable.
- Readiness write-probes output locations and invalidates stale reviews whenever
  a user-facing choice changes.
- Desktop selection binds only its identity-matched live runtime, effective
  model-root set, exact checkpoint file, and LoRA destination—even when only
  one installation exists. Missing ownership or a path outside those effective
  roots blocks a full run instead of falling through to another instance.
  Missing saved IDs remain visibly unavailable until the user chooses, and
  duplicate instance names block runtime matching rather than guessing.
- A missing LoRA directory can be created from the GUI only when it exactly
  matches the selected Desktop instance's verified model-root declaration; the
  repair never creates arbitrary or parent paths.
- Video/Wan support remains an explicit Phase 2 roadmap item.
- Face-count QC, multi-checkpoint winner selection, and remote model upload
  remain explicit future product work rather than partially exposed controls.
  Blur and likely near-duplicate checks are local review recommendations in
  Phase 1.

---

## Model-profile interaction contract — 2026-10-11

The NoobAI addition extends the existing Connect model card. This document and
the [quality-of-life map](quality-of-life-map.md) remain the maintained design
and behavior context; this feature introduces no new visual system. The audience
is the existing English-language Windows desktop user.

| Capability | Canonical owner | Model-profile behavior | Verification |
| --- | --- | --- | --- |
| Select/Listbox | Native Qt `QComboBox`, styled by `src/desktop_theme.py` | Same keyboard, popup, focus, and selection behavior as the Desktop-instance selector; visible label and accessible name | `tests/test_model_profiles.py` keyboard selection and persistence |
| Form | `MainWindow._collect_settings`, `_invalidate_readiness`, application-service readiness | Persist the explicit profile; changing it invalidates approval; unknown saved profiles remain visible and blocked | Profile/settings tests and source desktop suite |
| Feedback/recovery | Typed `ReadinessActionKind` and existing readiness cards | Profile errors open the actual profile selector; trainer incompatibility disables a full run but not photo preparation | Service gate and trainer compatibility tests |
| Layout/scrolling | Existing scrollable Connect page and shared Qt theme | Add the selector inside the existing model card; no new fixed-height page container or screen-local colors | Desktop screenshot and popup inspection |

`src/desktop_theme.py` remains the runtime token owner: `APP_STYLE` supplies the
dark navy surface (`#0c1220`), card surface (`#141e33`), foreground (`#eef2ff`),
Segoe UI typography, and shared combo/field states. The new selector uses those
rules directly, with the existing `_hint` label and `_body` explanatory text.
The sibling project form keeps the same visual language and scroll behavior.

## Connection startup correction — 2026-10-11

The Connect page and readiness repair share `comfy_start_instruction` so both
name the selected local instance card in Desktop. `MainWindow` owns a single
retry timer and permits one setup check at a time. Connect keeps checking while
offline; an explicit Desktop launch also keeps the readiness/result continuation
pending until connection succeeds or the window closes. Returning to Connect
restarts checking. Background checks retain the displayed state while in flight.

The existing status badge, body text, logs action, and native buttons retain
their theme and keyboard behavior. **START INSTANCE** gives the card-opening
instruction; **CONNECTING** names the selected instance's runtime address and
startup-terminal recovery; **CONNECTED** requires a successful API response.
Unavailable or ambiguous installation identities retain their existing repair
states. Connection logs include candidate addresses and the actual probe error.

Verification covers repeated launch clicks, startup taking more than the old
18 attempts, one check in flight, late API success, failed-check retry, readiness
continuation, and result workflow publication. Live acceptance uses the installed
Comfy Desktop instance and a rebuilt Windows app; model generation is separate.

See [model support](model-support.md) for source authority, variant-specific
settings, the V-pred training gate, and the qualification boundary.

## Anime project controls — 2026-10-11

The Project form retains the existing native Qt grid and theme. Training plan
and Training steps sit side by side: plan selects rank/resolution and the
comparison sweep; an optional step count controls run length. Zero displays
**Plan default**, including that plan's numeric value. Explicit overrides
survive plan changes and project reloads. The source summary supplies the
per-image presentation estimate and imported-caption coverage; neither is a
new approval gate. **Extended training** replaces the misleading **High quality** label.

NoobAI profiles show their four anime review shots and portrait canvas on the
Project page. The same prompt definitions feed bundled ai-toolkit previews,
the ComfyUI strength grid, and the exported workflow. Standard SDXL keeps its
photo suite. Existing source captions are always preferred, with editable
prepared captions as the normal follow-up. Flat illustrations skip the photo
blur heuristic. The form remains scrollable at the minimum window size.

## Automatic anime tagging — 2026-10-11

The NoobAI Project form now defaults **Tag uncaptioned anime images automatically**
to on. A short hint explains the one-time download and local processing. The
checkbox and hint appear only for anime profiles; BLIP is disabled while anime
tagging is selected. The persisted draft carries this choice into the actual
preparation worker. Automatic tagging does not add a confirmation gate.

The run view shows model-download progress and the current image count. Existing
source captions are imported unchanged apart from a missing trigger; otherwise
WD ViT v3 produces general tags locally. Generated captions feed the same dataset
editor, training files, and reuse record as imported captions. Ready-state copy
distinguishes a cached tagger from its first download, without requiring users
to install Python or a vision runtime in the packaged application.
