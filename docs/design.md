# LoRA Training Pipeline — Design Document

**Goal:** Build a reusable, code-driven pipeline to train LoRAs on **photos** (Phase 1), then extend the same pipeline to **video** (Phase 2).
**Hardware:** Ryzen 9800X3D · 64GB DDR5 · RTX 5090 (32GB VRAM, Blackwell) · Windows 11 — this is top-tier for local training; VRAM is not a constraint.
**Assumptions used (tell me to change these):** you want to *learn the whole pipeline end-to-end*, and you want a *programmatic, code-driven* design rather than clicking nodes by hand.

---

## 1. Mental model: what each tool actually does

The single most important thing to internalize is that **ComfyUI is an inference engine, not a trainer.** It's brilliant at *running* diffusion models through a node graph and at *validating* your LoRA. Training is a different job, best handled by a dedicated trainer.

So the pipeline has two engines that talk to each other:

| Job                        | Tool                                    | Why                                                                                                                                                                                      |
|----------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Training** the LoRA      | **ai-toolkit** (ostris)                 | One tool covers SDXL, Flux, *and* Wan 2.2 video. YAML configs, a web UI, and a `run.py` you can script. MIT licensed. This is what makes photos→video a small step instead of a rewrite. |
| **Inference / validation** | **ComfyUI Desktop** (already installed) | Generate test images from your trained LoRA, run XY comparison grids, build the "does this LoRA work?" loop.                                                                             |
| **Orchestration**          | **Your own Python** (in VSCode)         | Prepares datasets, launches training, then drives ComfyUI's API to auto-generate validation images. This *is* your "application."                                                        |

> **Why not train inside ComfyUI directly?** You *can* — nodes like `comfyUI-Realtime-Lora` and `Lora-Training-in-Comfy` wrap trainers as nodes. It's great for quick experiments and worth trying early to see training happen visually. But for a *reusable application pipeline* that scales to video, a dedicated trainer + scripted orchestration is cleaner, more reproducible, and far easier to debug. We'll use the ComfyUI training nodes as a learning aid, and ai-toolkit as the production trainer.

### The flow

```text
   photos/                     ai-toolkit                    ComfyUI (API)
 ┌───────────┐   prep   ┌────────────────────┐  .safetensors ┌──────────────┐
 │ raw images│ ───────▶ │ dataset (img+txt)  │ ───────▶     │ load LoRA +  │
 │           │  caption │ + training config  │   train       │ generate     │
 └───────────┘          └────────────────────┘               │ validation   │
       ▲                          │                          │ grid         │
       │                          ▼                          └──────┬───────┘
       │                    checkpoints/                            │
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

One project folder in VSCode, trainer and ComfyUI referenced (not copied) from it:

```text
lora-pipeline/
├─ pyproject.toml / requirements.txt   # orchestrator + self-test dependencies
├─ .env                                # paths to ComfyUI, ai-toolkit, model dirs
├─ configs/
│  └─ person_sdxl.yaml                 # reusable SDXL training template
├─ datasets/
│  └─ subject_v1/
│     ├─ 01.png   01.txt               # image + caption pairs
│     ├─ 02.png   02.txt
│     └─ ...
├─ src/
│  ├─ prepare_dataset.py               # resize, auto-caption, sanity-check
│  ├─ train.py                         # thin wrapper that calls ai-toolkit run.py
│  ├─ comfy_client.py                  # ComfyUI API client (queue + websocket)
│  ├─ validate.py                      # generate a validation grid from a trained LoRA
│  └─ pipeline.py                      # ties it all together: prep → train → validate
├─ workflows/
│  └─ validate_sdxl_api.json           # ComfyUI workflow in "API format"
└─ outputs/
   ├─ checkpoints/                     # LoRA .safetensors land here
   └─ grids/                           # validation images per epoch
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

Two habits that save you: **save checkpoints often** (`save_every`) so you can pick the epoch *before* it overfit, and use ai-toolkit's built-in **sampling during training** to watch it learn in real time. On a 5090 an SDXL run like this is roughly 20–40 minutes.

You can drive training either from the **ai-toolkit web UI** (`http://localhost:8675` — start/stop/monitor jobs visually) or from code:

```python
# src/train.py
subprocess.run(
    ["python", "run.py", "configs/person_sdxl.yaml"], cwd=AI_TOOLKIT_DIR, check=True
)
```

### Step 3: Validation via ComfyUI's API (this is the "application" part)

Build a validation workflow **once** in the ComfyUI Desktop UI: `Load Checkpoint → Load LoRA → CLIP Text Encode (pos/neg) → KSampler → VAE Decode → Save Image`. Then **Save (API Format)** to `workflows/validate_sdxl_api.json`. That JSON is a template your code fills in and submits.

Minimal client:

```python
# src/comfy_client.py  (implemented client)
import json, uuid, urllib.request, websocket  # websocket-client

COMFY = "127.0.0.1:8188"  # ComfyUI server (enable API / run server mode)


def queue_prompt(graph: dict) -> str:
    body = json.dumps({"prompt": graph, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(f"http://{COMFY}/prompt", data=body)
    return json.load(urllib.request.urlopen(req))["prompt_id"]


def run_and_wait(graph: dict):
    ws = websocket.WebSocket()
    ws.connect(f"ws://{COMFY}/ws?clientId={CLIENT_ID}")
    pid = queue_prompt(graph)
    while True:  # block until this prompt finishes
        msg = ws.recv()
        if isinstance(msg, str):
            m = json.loads(msg)
            if (
                m.get("type") == "executing"
                and m["data"]["node"] is None
                and m["data"]["prompt_id"] == pid
            ):
                break
    # then fetch images from /history/<pid>
```

`validate.py` loads the workflow JSON, swaps in the new LoRA filename + a fixed set of seeds/prompts, runs them, and stitches an XY grid (LoRA strength × prompt) into `outputs/grids/`. Now every training run ends with a comparable contact sheet — that's your feedback loop.

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

- Phase 1 is implemented for a single person/identity LoRA on SDXL.
- Training configs are reusable templates; the orchestrator injects each run's
  name, dataset path, and checkpoint path before launching ai-toolkit.
- Video/Wan support remains an explicit Phase 2 roadmap item.

---
