# MIVIM LoRA Studio

MIVIM LoRA Studio is a guided Windows companion for training a person LoRA and
using it with Comfy Desktop. The normal workflow is visual: choose photos,
review readiness, start the run, and inspect the comparison grid. Casual users
do not need to edit YAML, `.env`, API ports, or ComfyUI node IDs.

The current release covers one-person SDXL LoRAs:

```text
photos → quality checks → preparation → ai-toolkit training
       → install in ComfyUI → validation grid → open in ComfyUI
```

## Start the studio

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

### Connect

- Detects current Comfy Desktop 2 and legacy Desktop installations.
- Reads Desktop configuration and runtime port hints without modifying them.
- Supports Desktop's multi-instance, shared-model, and dynamic-port layout.
- Probes only explicit loopback addresses; it never scans the network.
- Lists the checkpoints and LoRAs reported by the running ComfyUI API.
- Resolves the selected checkpoint back to its local Desktop model file so
  training and validation use the same SDXL base.
- Opens Comfy Desktop, the verified ComfyUI URL, and result folders.
- Keeps ai-toolkit in its own environment.

Comfy Desktop remains responsible for starting and stopping its own instances.
There is no supported external API for selecting and launching a particular
Desktop instance, so the studio opens Desktop and explains the required action
instead of pretending it started one.

### New project

- Native folder picker and drag-and-drop photo selection.
- Plain-language project name and identity phrase.
- Checks unreadable files, exact duplicates, image count, and undersized photos
  before a long run.
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
- ai-toolkit checkout plus a real PyTorch/CUDA probe in its isolated Python;
- writable workspace and available storage.

Photo preparation remains available when the trainer or ComfyUI is not ready.
A full run is enabled only when every required capability passes.

The three UI presets change real training configuration:

| Plan         | Rank | Steps | Preparation | Validation strengths      |
|--------------|-----:|------:|------------:|---------------------------|
| Fast draft   |    8 |   900 |      768 px | 0.8                       |
| Balanced     |   16 | 2,000 |    1,024 px | 0.6, 0.8, 1.0             |
| High quality |   32 | 3,000 |    1,024 px | 0.5, 0.65, 0.8, 0.95, 1.1 |

The run monitor streams the real pipeline output. It prevents concurrent GPU
runs, and Stop cancels the complete Windows process tree so a trainer process
is not silently left running. Failed starts and interrupted runs expose direct
edit, output-folder, and retry actions. Prepared data and completed checkpoints
are retained. If training and installation finish but ComfyUI validation fails,
the monitor offers a validation-only retry instead of repeating training.

### Result

- Verifies that this run created or changed the expected dataset or grid before
  reporting success.
- Displays the validation grid inside the app.
- Restores the most recent result after reopening.
- Opens the result folder and the connected ComfyUI instance.

## Required external tools

### Comfy Desktop

Install the current Windows application from the
[official ComfyUI Windows guide](https://docs.comfy.org/installation/desktop/windows).
Create or launch a local instance from Desktop.

Install an SDXL-compatible checkpoint through ComfyUI. The studio reads the live
checkpoint list, resolves the selected file through Desktop's configured model
roots, and injects that same local file into ai-toolkit. If an external model
path cannot be resolved automatically, the Connect page provides a checkpoint
file picker. Finished LoRAs are written only to the selected user/shared model
library; a same-named prior LoRA is preserved with a `.previous` suffix.

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

`pysidedeploy.spec` includes the training config and ComfyUI validation
workflow. The packaged executable reuses itself as a hidden-console pipeline
worker, so it does not depend on a system Python interpreter after deployment.
Distribute the complete `dist\MIVIMLoRAStudio.dist` folder, not the executable
by itself.

## Architecture

```text
PySide6 desktop
  ├─ application service (readiness + commands)
  ├─ versioned user settings
  ├─ read-only Comfy Desktop discovery
  └─ background pipeline worker
       ├─ prepare_dataset
       ├─ ai-toolkit subprocess
       ├─ atomic LoRA handoff
       └─ ComfyUI API validation
```

Important source files:

| File                   | Responsibility                                       |
|------------------------|------------------------------------------------------|
| `src/desktop.py`       | Guided UI, background process, cancellation, results |
| `src/app_service.py`   | Readiness, source review, command construction       |
| `src/app_models.py`    | Typed UI-independent settings and project state      |
| `src/app_settings.py`  | Atomic per-user persistence                          |
| `src/comfy_desktop.py` | Desktop 2/legacy discovery and live API models       |
| `src/pipeline.py`      | Preparation → training → installation → validation   |
| `src/train.py`         | Concrete ai-toolkit config and real preset overrides |
| `src/comfy_client.py`  | ComfyUI prompt/WebSocket/history client              |

## Deliberate current limits

- Phase 1 supports person/identity LoRAs on SDXL.
- Video/Wan training, remote ComfyUI uploads, and Comfy Cloud are not presented
  as working features.
- BLIP captions are optional in source installations and are hidden in the
  packaged build until a managed optional-component installer exists.
- Face-count, blur, and near-duplicate review require a future local vision/QC
  component; the current readiness pass covers corruption, exact duplicates,
  count, and resolution.
- The first complete review artifact is one validation grid. Multi-checkpoint
  winner selection and partial validation resume are subsequent product work,
  not hidden placeholders in this release.

The local ComfyUI routes used by the integration are documented in the
[official server route reference](https://docs.comfy.org/development/comfyui-server/comms_routes).
