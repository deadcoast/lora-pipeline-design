# lora-pipeline

A small, code-driven pipeline to train **person LoRAs on SDXL** and validate them
automatically through ComfyUI. Built for Windows 11 + RTX 5090. Designed so the
same structure extends to **video (Wan 2.2)** later with minimal changes.

```
raw photos ──prepare_dataset──▶ dataset (img+txt) ──train (ai-toolkit)──▶ LoRA
                                                                            │
   validation grid ◀──validate (ComfyUI API)◀───────────────────────────────┘
```

Three engines, one orchestrator:

| Job | Tool |
|---|---|
| Train | **ai-toolkit** (its own venv) |
| Inference / validation | **ComfyUI Desktop** (its own runtime) |
| Glue (prep, launch, validate) | **this repo** (its own venv) |

---

## 1. One-time setup (do the Blackwell step FIRST)

Your RTX 5090 (Blackwell, `sm_120`) is **not** supported by stable PyTorch wheels.
Skipping this causes `"CUDA error: no kernel image is available for execution on the
device"` that looks like a broken install. In **each** venv that touches CUDA
(ai-toolkit, and ComfyUI if you manage it manually):

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

Also: **do not install `xformers`** (its wheels top out at Ada/sm_89 and will
downgrade your good PyTorch). Use PyTorch SDPA or SageAttention for attention.
Update NVIDIA drivers to a recent production branch (R580+).

### This repo's venv (lightweight — no PyTorch needed here)

```bash
cd lora-pipeline
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # then edit paths
```

### ai-toolkit (the trainer) — separate venv

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
python -m venv venv
venv\Scripts\activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
pip install -r requirements.txt
```

Point `.env` at it: `AI_TOOLKIT_DIR` and `AI_TOOLKIT_PYTHON` (the venv's python.exe).

### ComfyUI

You already have Comfy Desktop. Make sure the SDXL base checkpoint is in
`models/checkpoints`, and that the API server is reachable at `COMFY_HOST`
(default `127.0.0.1:8188`). Trained LoRAs must land in `models/loras`
(`COMFY_LORA_DIR`) for validation to see them.

Sanity check paths any time:

```bash
python -m src.config
```

---

## 2. Verify before you run

Two commands prove things work before you spend time on data or training:

```bash
# (a) Self-test: boots a MOCK ComfyUI and runs the real client + validation
#     code end-to-end. No GPU, no models needed. Proves the integration works.
python -m tests.test_integration

# (b) Preflight: connects to YOUR running ComfyUI, confirms the checkpoint is
#     loadable, the LoRA folder is writable, and your ai-toolkit paths are valid.
python -m src.doctor
```

`doctor` exits non-zero until your environment is actually ready, so you never
start a training run into a broken setup.

## 3. Run it

```bash
# Full pipeline: prep photos -> train -> auto-validate
python -m src.pipeline --src "C:/photos/me_raw" --name person_v1

# Or one stage at a time:
python -m src.prepare_dataset --src "C:/photos/me_raw" --name person_v1
python -m src.train  --config configs/person_sdxl.yaml
python -m src.validate --lora person_v1.safetensors
```

Outputs:
- `datasets/person_v1/` — processed `NNN.png` + `NNN.txt` caption pairs
- ai-toolkit writes checkpoints (`person_v1_<step>.safetensors`) to its `output/`
- `outputs/grids/person_v1_grid.png` — the validation contact sheet

---

## 3. The tuning loop

1. Build a clean dataset (15–40 varied photos of one person). This matters most.
2. Train with the defaults in `configs/person_sdxl.yaml`.
3. Look at the validation grid. Reading it:
   - **Doesn't look like the person** → train longer, raise rank (16→32), or add data.
   - **Looks *exactly* like training photos, ignores the prompt (overfit)** → fewer
     steps, lower LR (1e-4→5e-5), or pick an earlier checkpoint.
   - **Right likeness, poor flexibility** → lower LoRA strength at inference (0.6–0.8).
4. Because `save_every: 250` keeps intermediate checkpoints, validate a few and
   keep the best — not necessarily the last.

---

## 4. Files

```
configs/person_sdxl.yaml     ai-toolkit training config (edit the dataset path)
workflows/validate_sdxl_api.json  ComfyUI workflow in API format (the validation graph)
src/config.py                loads .env, resolves paths
src/prepare_dataset.py       resize/convert/caption + QC report
src/caption.py               trigger-token template captions (optional BLIP)
src/comfy_client.py          ComfyUI REST/WebSocket client
src/validate.py              sweeps prompt x strength -> stitched grid
src/train.py                 wraps ai-toolkit run.py
src/pipeline.py              prep -> train -> validate
```

### Regenerating the validation workflow
`workflows/validate_sdxl_api.json` is a plain SDXL+LoRA text2img graph. To change
it, rebuild it in ComfyUI and use **Save (API Format)** — the code fills in the
checkpoint, LoRA, prompts, strength and seed at runtime, so keep those node IDs.

---

## 5. Extending to video (later)

ai-toolkit also trains **Wan 2.2** video LoRAs, so the jump is small:
- `prepare_dataset.py` grows a clip/frame branch (motion-focused captions).
- Add a `configs/person_wan22.yaml` pointing at a Wan template.
- Export a Wan video-generation workflow from ComfyUI as API JSON; `comfy_client.py`
  drives it unchanged (it just returns a video instead of a PNG).

The orchestrator, client, and folder layout stay the same.
```
