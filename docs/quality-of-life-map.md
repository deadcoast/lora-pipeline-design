# MIVIM quality-of-life map

This map keeps the casual-user promise testable. A feature is considered
implemented only when the desktop application exposes it and its production
path is exercised. Packaged UI and mock ComfyUI API coverage do not prove a
live Desktop workflow import or GPU training run; those remain environment
acceptance checks.

## Current user journey

| Area | Frustration prevented | User-facing correction or recovery |
| --- | --- | --- |
| Comfy Desktop | Wrong instance, port, or model library | Bind the selected Desktop installation, live runtime, checkpoint roots, and LoRA destination as one context; block ambiguous or stale state |
| First setup | Raw paths and ports dominate onboarding | Show model selection and guided trainer setup first; keep paths and custom ports under **Advanced** |
| Base model | Training and validation silently use different architectures | Require the exact same local checkpoint and an explicit SDXL-base confirmation |
| Trainer | A CUDA-enabled-looking environment fails after training starts | Run a real CUDA kernel in the trainer's own Python; show GPU architecture and guided repair |
| Source photos | Bad files are reported only as counts | Name and preview unreadable, small, blurry, duplicate, and likely near-duplicate files |
| Exact copies | Users must manually clean harmless duplicates | Preserve originals and automatically prepare one copy |
| Captions | A removed identity phrase silently ruins training | Require the phrase in every caption and offer to restore it automatically |
| Prepared work | A failed later stage forces photo preparation again | Validate the source manifest and caption pairs, then continue with `--skip-prep` |
| Project names | Old checkpoints, grids, or LoRAs mix with a new run | Detect the whole project artifact namespace; require a new version name or explicit rebuild |
| Long training | Progress appears frozen, errors are opaque, or Stop affects another ComfyUI user | Show elapsed/indeterminate training progress and durable local logs; remove an owned pending prompt, or target a running prompt only on ComfyUI 0.3.56+ |
| Validation | Users cannot compare against the unchanged model | Include a clearly labelled no-LoRA baseline and plain-language prompt/strength axes |
| Failed validation | Restarting causes unnecessary retraining or regeneration | Persist the exact handoff; retry missing or corrupt cells only when the full signature, including both model content identities, matches |
| Result handoff | Casual users do not know what to enter or build in ComfyUI | Export editable Workflow JSON v0.4; embed both the API prompt and UI workflow in the PNG; show and copy the exact LoRA, phrase, checkpoint, and tested strength |
| Restored result | Large files freeze the UI or stale model bytes are trusted | Hash the LoRA and checkpoint in a worker; keep handoff actions disabled until the original Desktop context and exact bytes are verified |
| Offline restart | A temporary Desktop discovery failure forgets a valid setup | Retain the known installation and checkpoint identity, then ask the user to reconnect that exact context |

## Comfy Desktop compatibility boundary

The integration is read-only toward Desktop configuration. It understands the
current installation records, shared/private model roots,
`extra_model_paths.yaml`, identity-bearing live port locks, and the legacy
Desktop layout. MIVIM never edits Desktop's installation registry.

A selected Desktop instance is an identity boundary. If its runtime cannot be
matched safely, a full run is blocked instead of falling through to another
running instance.

MIVIM can open Desktop or its browser UI, but Windows does not expose a reliable
external command to select/start one particular Desktop instance or import a
workflow into it. The app therefore verifies the selected context, opens the
destination, and gives the user a ready PNG/JSON to drag in or open.

## Recovery policy

- Correct automatically when the result is deterministic and source data stays
  untouched, such as ignoring exact copies.
- Offer one clear repair action when user judgment is required, such as
  confirming an SDXL base model or repairing the trainer.
- Preserve completed work before attempting the next fallible stage.
- Keep technical output and rotating logs available locally without making
  terminal commands part of the normal workflow.
- Never report a full run as complete unless its immutable launch snapshot,
  attempt ID, grid, workflow, installation, host, model paths, LoRA, checkpoint,
  and content fingerprints agree. Rehash both models after validation and keep
  the grid but refuse approval if either changed during the sweep.

## Explicit next roadmap

These items remain future work and are not represented as complete:

- face-count and multi-person source-photo checks;
- guided selection of the best intermediate training checkpoint;
- managed ai-toolkit installation and updates;
- video, remote, and cloud training paths.
