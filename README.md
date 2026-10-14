# MIVIM LoRA Studio

MIVIM LoRA Studio is a guided Windows companion for training a person or anime
character LoRA and using it with Comfy Desktop. The normal workflow is visual: choose images,
review readiness, start the run, and inspect the comparison grid. Casual users
do not need to edit YAML, `.env`, API ports, or ComfyUI node IDs.

The current release covers SDXL identity LoRAs, including NoobAI anime profiles:

```text
photos → quality checks → preparation → ai-toolkit training
       → install in ComfyUI → validation grid → open in ComfyUI
```

## Start the studio

Installed Windows app: open **MIVIM LoRA Studio** from the desktop or Start menu.
The installed executable stays at
`%LOCALAPPDATA%\Programs\MIVIM LoRA Studio\MIVIMLoRAStudio.exe`.
`Start MIVIM LoRA Studio.cmd` also opens that installation when present.
Projects and settings remain in `%LOCALAPPDATA%\MIVIM\LoRA Pipeline`.

For a developer-built package, `scripts/build_windows.ps1` writes the current
release to one location, `dist/release`. Run `Setup MIVIM LoRA Studio.cmd` to
install that release and create the shortcuts. The installer can also accept
`-TrainerDirectory` to copy an existing ai-toolkit environment into Studio's
per-user trainer folder; Studio discovers that folder automatically.

For the easiest source checkout setup on Windows:

1. Double-click `Setup MIVIM LoRA Studio.cmd` once.
2. Double-click `Start MIVIM LoRA Studio.cmd` afterward.

The setup creates a private `.venv` and installs the studio dependencies. It
does not install anything into Comfy Desktop's Python environment.

Developers using `uv` can run:

```powershell
uv sync --group dev
uv run mivim-lora-studio
```

You can also launch it directly:

```powershell
.venv\Scripts\python.exe -m src.desktop
```

## What the app handles

Already tagged your images? Keep each image beside its matching `.txt` file.
Studio imports those captions automatically, preserving their tags and order.
From 0.8.1, the training worker encodes long SDXL captions in complete sections
instead of silently discarding the trailing tags. No retagging or extra setup
is needed; automatic tagging only fills missing captions.

### Library

- Opens as the home page once at least one run exists. Every run is read from
  the result and recovery records already on disk, paired with its grid,
  checkpoint folder, and installed LoRA. Nothing is hashed on this page.
- Groups runs by **character**, shows each run's **art style** chip, base
  checkpoint and profile, trigger, step count, finish date, and a status:
  Ready in ComfyUI, LoRA missing, Result missing, Validation pending,
  Prepared only, or Incomplete. A style filter and a search box narrow the list.
- **Open result** shows any run on the Result page, not only the last one.
  **Train next version** carries that run's checkpoint, profile, trigger, plan,
  tags, and photo folder into New project with the next version name.
  **Tag** files runs that predate the tag fields; they wait under **Unsorted**.
- Stores the tag vocabulary, a style label per checkpoint, and tag overrides in
  `library.json` beside the workspace folders. An unreadable file is left
  untouched and reported on the page.

### Connect

- Detects current Comfy Desktop 2 and legacy Desktop installations.
- Detects the registered Windows application before its first Desktop launch,
  then guides the user to create a local instance instead of offering the
  installer again.
- Reads Desktop configuration and live-process runtime port hints without
  modifying them; stale locks from closed instances are ignored.
- Supports Desktop's multi-instance, shared-model, and dynamic-port layout.
- Reconstructs Desktop's executable-drive and persisted shared-storage
  defaults, configured model libraries, startup path arguments, generated
  per-instance model config, install-local `extra_model_paths.yaml`, and output
  model folders in ComfyUI's actual first-match order.
- Uses the launch snapshot for a running instance. If Desktop storage changed
  after launch or the snapshot cannot be proven, the app asks the user to
  restart that instance instead of approving next-launch paths as current.
- Shows an instance selector when more than one local Desktop installation
  exists, then binds that instance's runtime port, model roots, checkpoint, and
  LoRA destination as one connection. It never falls through to another
  instance's runtime when the selected instance is offline.
- Preserves a missing saved instance as **Original instance unavailable** until
  the user explicitly chooses another. Duplicate Desktop display names are
  blocked because current runtime locks cannot distinguish them safely.
- Probes only explicit loopback addresses; it never scans the network.
- Lists the checkpoints and LoRAs reported by the running ComfyUI API. Each
  checkpoint carries a purpose tooltip such as **MODEL :: Anime · NoobAI-XL ·
  Epsilon-pred**; the same line appears under the selected model. NoobAI files
  are labelled anime and a detected SDXL base is labelled photo; anything else
  shows **Unlabelled** with a one-word field to label it. Picking a checkpoint
  selects the profile its filename calls for, and the profile list stays
  available for override.
- Resolves the selected checkpoint back to its local Desktop model file so
  training and validation use the same SDXL base.
- Recognizes SDXL base architecture from the local checkpoint's tensor header;
  manual model-type confirmation is only needed when it cannot be identified.
- Offers to create a missing `loras` folder only when it is the exact folder
  declared by the selected, verified Desktop model root.
- Opens Comfy Desktop, the verified ComfyUI URL, and result folders.
- Names the local instance card to open in Desktop, keeps checking while you
  connect, and shows **CONNECTING** while its server starts. Repeated opens
  reuse one connection check; the wait no longer stops after 90 seconds.
- Keeps ai-toolkit in its own environment.
- Discovers the installed trainer under the per-user application-data folder.
  Trainer checks run in that Python's isolated mode and directory, preventing
  the packaged app's Python libraries from interfering with the trainer.

Comfy Desktop remains responsible for starting and stopping its own instances.
There is no supported external API for selecting and launching a particular
Desktop instance, so the studio opens Desktop and explains the required action
instead of pretending it started one.

### Model profiles

Connect includes separate **NoobAI-XL · Epsilon-pred** and **NoobAI-XL · V-pred**
profiles alongside standard SDXL. The selection controls training configuration,
validation, and the editable ComfyUI workflow. The V-pred launcher corrects the
known ai-toolkit image-target bug in the trainer process and applies its required
noise schedule. Unrecognized trainer implementations remain blocked. See
[model support and the research corrections](docs/model-support.md) for verified
releases and acceptance limits.

### New project

- Imports existing UTF-8 `.txt` captions beside source images, preserving tag
  order and adding the identity phrase only when it is missing. Images without
  captions in NoobAI projects are tagged automatically with local WD ViT v3.
  Prepared captions remain editable. Standard SDXL keeps template/optional BLIP captions.
- **Automatic anime tagging is on by default** for both NoobAI profiles, in
  source and packaged Windows versions. The model downloads once (361 MB), then
  works offline on CPU. Download and per-image progress appear in the run view;
  no images are uploaded. The Project checkbox can disable automatic tags.
- Offers a **Training steps** control independent of the rank/resolution plan.
  **Plan default** restores the preset length; the image count shows approximate
  presentations per source at batch size 1. Short runs still save intermediate
  checkpoints and previews.
- NoobAI training previews and final comparisons share four shots: face, full
  body, profile, and action at 832×1216. Photographic prompts remain available
  with standard SDXL; custom training YAML retains its own prompts.
- Disables photographic blur heuristics for anime profiles so flat shading is
  not treated as a blurry photograph. Aspect ratio is preserved during preparation.
- Native folder picker and drag-and-drop photo selection.
- **Character** and **Art style** fields above the project name. One word
  each, lowercased as you type, with autocomplete from every word used before
  so a misspelling never creates a second style. The hint under each field says
  **Existing style: anime** or **New style: watercolor will be added**. Both are
  required to review or start a run; a new word joins the vocabulary when the
  run starts.
- Fills the project name as `<character>-<style>-mk<NN>` with the next free
  number in that group until you type your own name. After tagging
  `ruth-mk02` as ruth / anime, the next run is offered as `ruth-anime-mk03`.
- Plain-language project name and identity phrase.
- Checks unreadable files, exact duplicates, image count, and undersized photos
  before a long run.
- Flags likely blur and very similar photos as recommendations, and names every
  affected file in an in-app photo review without changing originals.
- Keeps rebuild controls optional. Advanced BLIP captioning is shown only in a
  source environment where its optional runtime can be installed and verified.
- Stores projects in a writable per-user workspace by default.

### Review and run

The application checks the actual paths and services it will use:

- source photos and project name;
- running ComfyUI API;
- selected checkpoint as reported by ComfyUI;
- selected checkpoint file as the ai-toolkit training base;
- writable ComfyUI LoRA folder;
- ai-toolkit checkout plus an actual CUDA test kernel in its isolated Python;
  the result names the GPU compute architecture and blocks incompatible wheels,
  including a build that cannot execute on Blackwell `sm_120`;
- writable workspace and available storage.

Photo preparation remains available when the trainer or ComfyUI is not ready.
A full run is enabled only when every required capability passes.
Each readiness card has one exact repair operation behind its visible button:
choosing photos never opens review by surprise, model and trainer actions return
to the relevant setup control, and successful settings changes are saved before
the checks run again. Low-storage recovery opens Windows Storage settings;
changing the workspace remains a separate, clearly labelled choice.
The project name is also checked across prepared data, trainer checkpoints and
config, validation work and grid, the completed-run record, and the installed
ComfyUI LoRA. Existing outputs require a new version name or explicit rebuild
approval, preventing old and new runs from being mixed silently.

The three UI presets change real training configuration:

Every plan includes a no-LoRA baseline for an honest before/after comparison:

| Plan         | Rank | Steps | Preparation | LoRA strengths            |
|--------------|-----:|------:|------------:|---------------------------|
| Fast draft   |    8 |   900 |      768 px | 0.8                       |
| Balanced     |   16 | 2,000 |    1,024 px | 0.6, 0.8, 1.0             |
| High quality |   32 | 3,000 |    1,024 px | 0.5, 0.65, 0.8, 0.95, 1.1 |

The run monitor streams the real pipeline output. Cross-process leases prevent
overlapping GPU, project, workspace, dataset, checkpoint, grid, and ComfyUI
model-library mutations—even across multiple Studio windows and direct CLI
stage calls. On Windows, the Studio owns its worker lifetime and trainers are
created atomically inside a kill-on-close Job Object, so a normal stop, crash,
or hard-killed window cannot silently leave training on the GPU. During
validation, durable prompt reservations survive a worker crash. Pending owned
prompts are removed individually; a running prompt receives a targeted
interrupt only when ComfyUI reports version 0.3.56 or newer. Older or unknown
versions, unreadable queue state, and unverified ownership fail closed: MIVIM
explains the exact ComfyUI action and never stops another user's work. Failed
starts and interrupted runs expose direct edit, output-folder, and retry
actions.
An out-of-memory recovery can offer the Fast preset, but it explains that a
fresh rebuild replaces the current project's generated work and requires
explicit approval before enabling that rebuild.
Prepared data and completed checkpoints are retained. If training and
installation finish but ComfyUI validation fails,
the monitor offers a validation-only retry instead of repeating training. That
post-install handoff is recorded atomically, so reopening the studio after a
failed or interrupted validation restores the same validation-only recovery.
The retry remains bound to the exact Desktop installation, address, LoRA bytes,
checkpoint file, prompts, strengths, and seed saved with that run. Verified
cells resume individually; changed inputs get an isolated attempt instead of
mixing old and new images.
The Technical details header also opens the durable local support logs, so a
user can inspect or share the relevant details without reproducing a failure in
a terminal.

Prepared projects are manifest-validated and reused automatically. The app
offers thumbnails and an editable caption for every prepared photo; continuing
the full run uses `--skip-prep`, so a prepare-only run or a failure after
preparation does not force users to repeat that work. Caption review reserves
the same dataset used by preparation and training: another Studio window or
CLI run cannot change or train that dataset until the review closes. A busy
dataset cannot open for editing, and failed opens release the reservation.
Reuse and training share one image/`.txt` pair contract; duplicate stems,
missing partners, and unsupported image files are named before Start can
launch a process. The
manifest also binds reuse to the requested description method and the exact
generated image bytes. The trainer rechecks those hashes immediately before
materializing its run config. Captions remain editable, but replaced or deleted
prepared images require a visible rebuild instead of being trusted silently.
The preparation record covers every supported image the trainer can consume,
including added photos with different names or formats. Unrecorded photos block
reuse and training. Rebuild preserves photos outside the generated numbered
pairs and asks the user to move them out or choose a new project name; photos
intended for training belong in the source folder before preparation.

### Result

- Verifies that this run created or changed the expected dataset or grid before
  reporting success.
- Displays the validation grid inside the app.
- Labels the grid with a no-LoRA baseline, plain-language prompt columns, and
  LoRA-strength rows.
- Persists and restores only the most recent verified result record, rather
  than guessing from stale files with the same project name.
- Writes `<project>_use_in_comfyui.json` as an editable, core-node ComfyUI
  Workflow JSON v0.4. The grid PNG embeds both the exact API prompt that ran and
  its equivalent editable UI workflow.
- Shows and copies the exact LoRA filename, identity phrase, base checkpoint,
  ready-workflow path, and tested default strength nearest 0.8.
- Hashes the exact LoRA and checkpoint in the background before enabling
  handoff actions, then requires the same Desktop installation and model paths.
  A normal Desktop restart may use a new dynamic port without invalidating that
  durable identity.
- Adds the exact verified JSON to the selected ComfyUI user profile through the
  supported user-data API under **Workflows → MIVIM**. Existing workflows are
  never overwritten: each new file has a content-fingerprinted name, identical
  bytes are accepted as already added, and every write is read back before
  approval.
- Keeps **Show JSON file** and **Workflow → Open** as a visible fallback. If the
  Workflows sidebar was already open, the app tells the user to click Refresh.

## Required external tools

### Comfy Desktop

Install the current Windows application from the
[official ComfyUI Windows guide](https://docs.comfy.org/installation/desktop/windows).
Create or launch a local instance from Desktop.

Install an SDXL-compatible checkpoint through ComfyUI. The studio reads the live
checkpoint list, resolves the selected file through Desktop's configured model
roots, and injects that same local file into ai-toolkit. If an external model
path cannot be resolved automatically, the Connect page provides a checkpoint
file picker, but training remains blocked until that exact file resolves inside
the selected instance's effective roots. Finished LoRAs are written only to the
selected user/shared model library; a same-named prior LoRA is preserved with a
`.previous` suffix.

### ai-toolkit

ai-toolkit performs training and must use a separate Python environment:

```powershell
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
git submodule update --init --recursive
py -3 -m venv venv
# Install the current CUDA build from the official PyTorch selector first.
venv\Scripts\python.exe -m pip install -r requirements.txt
```

For an RTX 5090, install a current PyTorch/CUDA build that includes `sm_120` in
the ai-toolkit environment. The studio's own environment intentionally does not
contain PyTorch unless optional vision captions are installed.

Return to the Connect page and choose the ai-toolkit folder. Its venv Python is
detected automatically when it uses the conventional `venv` or `.venv` name.

## Storage and privacy

Application settings are versioned, non-secret JSON under:

```text
%LOCALAPPDATA%\MIVIM\LoRA Pipeline\settings.json
```

The default project workspace is:

```text
%LOCALAPPDATA%\MIVIM\LoRA Pipeline\workspace
```

The workspace can be changed from the Connect page. Source photos, prepared
datasets, checkpoints, and grids remain local. The studio does not upload them.
Comfy Cloud and arbitrary remote ComfyUI hosts are intentionally not treated as
local model destinations.

Setup checks, photo/readiness failures, and the real pipeline output are kept in
size-limited rotating support logs under:

```text
%LOCALAPPDATA%\MIVIM\LoRA Pipeline\logs
```

The current log plus four backups use at most about 5 MB. Logs are never
uploaded automatically and never include environment-variable dumps or photo
contents, but they can include local paths and filenames needed to diagnose a
run. Use **Open local logs** beside Technical details to reach them.

## Advanced command-line use

The supported CLI remains available for automation:

```powershell
# Verify a configured environment.
.venv\Scripts\python.exe -m src.doctor

# Full run.
.venv\Scripts\python.exe -m src.pipeline `
  --src "C:\Photos\person" `
  --name person_v1 `
  --preset balanced

# Prepare only.
.venv\Scripts\python.exe -m src.pipeline `
  --src "C:\Photos\person" `
  --name person_v1 `
  --preset fast `
  --skip-train `
  --skip-validate
```

`.env` remains an advanced CLI configuration mechanism. The desktop app passes
its saved settings directly to the child pipeline and does not require `.env`.

## Development and validation

```powershell
uv sync --group dev
uv run pytest -q
uv run mypy src tests
uv run pyright src tests
uv run ruff check src tests
uv run ruff format --check src tests
```

The test suite includes an offscreen Qt interaction that clicks through the
actual desktop workflow and runs the real photo-preparation pipeline in its
background process. A real initialized window can also be rendered for visual
inspection:

```powershell
uv run mivim-lora-studio `
  --settings "$env:TEMP\mivim-smoke-settings.json" `
  --screenshot "dist\mivim-lora-studio.png"
```

Build a standalone Windows application with Qt's official deployment tool:

```powershell
.\scripts\build_windows.ps1
```

The build exports the runtime and deployment tools from the checked-in
`uv.lock`, verifies package hashes, and synchronizes the private
`.venv-deploy` before packaging. Run `uv lock` and commit the updated lockfile
when intentionally changing application or deployment dependencies;
`requirements.txt` supports the source-checkout launcher and is not the release
build's dependency authority.

`pysidedeploy.spec` includes the training config and ComfyUI validation
workflow. The packaged executable reuses itself as a hidden-console pipeline
worker, so it does not depend on a system Python interpreter after deployment.
Distribute the complete `dist\release\MIVIMLoRAStudio.dist` folder, not the
executable by itself.

After building, `uv run python -m pytest tests/test_packaging.py -q` checks the package,
including native Windows UI actions that start its real photo-preparation
worker. This acceptance test requires a Windows desktop. For a separate build,
set `MIVIM_TEST_PACKAGED_APP` to its full executable path before running the tests.

The script permits unsigned local test builds but prints a release warning.
Before publishing the folder to casual users, Authenticode-sign
`MIVIMLoRAStudio.exe` with the MIVIM publisher certificate and confirm
`Get-AuthenticodeSignature` reports `Valid`. The certificate and private key
are release credentials and are intentionally not stored in this repository.

## Architecture

See the [MIVIM quality-of-life map](docs/quality-of-life-map.md) for the
casual-user friction, correction path, and compatibility boundary behind each
desktop feature.

```text
PySide6 desktop
  ├─ application service (readiness + commands)
  ├─ versioned user settings
  ├─ read-only Comfy Desktop discovery + live launch snapshot
  ├─ bounded local support log
  ├─ Studio-owned worker lifetime
  └─ background pipeline worker
       ├─ cross-process project/resource/GPU leases
       ├─ prepare_dataset
       ├─ crash-owned ai-toolkit process tree
       ├─ atomic LoRA handoff
       └─ durable ComfyUI API validation ownership
```

Important source files:

| File                     | Responsibility                                          |
|--------------------------|---------------------------------------------------------|
| `src/desktop.py`         | Guided UI, background process, cancellation, results    |
| `src/library.py`         | Library scan, tag vocabulary, `library.json`, next name |
| `src/library_view.py`    | Library page, tag fields with autocomplete, tag dialog  |
| `src/tags.py`            | One-word character and art-style tag rule               |
| `src/dataset_review.py`  | Prepared-photo preview and atomic caption editing       |
| `src/photo_review.py`    | Filename-level source quality review                    |
| `src/app_service.py`     | Readiness, source review, command construction          |
| `src/app_models.py`      | Typed UI-independent settings and project state         |
| `src/app_settings.py`    | Atomic per-user persistence                             |
| `src/studio_logging.py`  | Local rotating setup, recovery, and pipeline logs       |
| `src/comfy_desktop.py`   | Desktop 2/legacy discovery and live API models          |
| `src/run_guardian.py`    | Cross-process project, resource, and accelerator leases |
| `src/process_control.py` | Studio/worker/trainer crash-lifetime ownership          |
| `src/comfy_activity.py`  | Durable ownership of remotely queued ComfyUI prompts    |
| `src/pipeline.py`        | Preparation → training → installation → validation      |
| `src/train.py`           | Concrete ai-toolkit config and real preset overrides    |
| `src/comfy_client.py`    | ComfyUI prompt/WebSocket/history client                 |

## Deliberate current limits

- Phase 1 supports person/identity LoRAs on SDXL.
- Video/Wan training, remote ComfyUI uploads, and Comfy Cloud are not presented
  as working features.
- BLIP captions are optional in source installations and are hidden in the
  packaged build until a managed optional-component installer exists.
- Face-count review still requires a future local vision component. The current
  pass covers corruption, exact and likely near-duplicates, likely blur, count,
  and resolution.
- Each completed validation produces a comparison grid plus a tested,
  core-node-only ComfyUI Workflow JSON v0.4. The PNG separately embeds the API
  prompt and equivalent editable UI workflow. Publication of the verified 0.4
  workflow through ComfyUI's supported user-data API is implemented; Workflow
  JSON 1.0 export is not claimed.
- Automated API integration uses a mock ComfyUI API. The separate
  [September 16 Windows/GPU acceptance](docs/windows-gpu-acceptance-20260916.md)
  completed a real 2,000-step SDXL run, 16 validation images, workflow
  publication, and generation from the opened workflow in Comfy Desktop.
- **Result → Compare training checkpoints** now compares saved training previews
  at the same prompt index. Choose **Validate this checkpoint** to generate a
  separate baseline/strength grid and ready ComfyUI workflow without retraining.
  The original result stays intact. Interrupted choices retain validation-only
  recovery, and the comparison remains linked to the original training project.
  Automatic checkpoint ranking and strength-winner selection remain future work.

The local ComfyUI routes used by the integration are documented in the
[official server route reference](https://docs.comfy.org/development/comfyui-server/comms_routes).
