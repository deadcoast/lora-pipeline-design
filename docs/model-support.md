# Model profiles and compatibility

Reviewed against primary sources on **2026-10-11**. This is a compatibility
reference, not a ranking of image quality. No checkpoint weights are downloaded
or bundled by Studio.

## NoobAI-XL addition

In **Connect → SDXL base model**, choose the checkpoint and then its **Model
profile**. Confirm the variant on that exact release's model card; a filename
is only a useful mismatch warning. Renamed files still need an explicit profile.
Training and validation continue to use the same local checkpoint.

| Profile | Reference release | Validation and exported workflow | Training boundary |
| --- | --- | --- | --- |
| SDXL · Standard | Existing SDXL path | Existing sampler and schedule | Existing behavior |
| NoobAI-XL · Epsilon-pred | [Laxhar 1.1](https://huggingface.co/Laxhar/noobai-XL-1.1) | Explicit Epsilon prediction; Euler ancestral, normal schedule, 28 steps, CFG 5.5 | SDXL LoRA configuration; DDPM training and Euler ancestral previews |
| NoobAI-XL · V-pred | [Laxhar V-pred 1.0](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0) | Explicit V-prediction and zero-terminal-SNR; Euler, normal schedule, 30 steps, CFG 4.5 | DDPM training and Euler previews; isolated adapter corrects the known image-target bug |

The upstream Epsilon card recommends Euler ancestral, 25–30 steps and CFG 5–6.
The V-pred card instead recommends Euler, 28–35 steps and CFG 4–5, and its
Diffusers example enables zero-terminal-SNR. These profiles target the named
releases, not every derivative bearing the NoobAI name. The normal ComfyUI
schedule is a Studio integration choice, not a universal model-card mandate.

Both NoobAI profiles export the core `ModelSamplingDiscrete` node. CLIP and VAE
continue to come from the selected checkpoint. Studio does not add a CLIP-layer
override. Training previews use 1024 × 1024; the existing identity-comparison
prompts and validation framing are unchanged. Fast/Balanced/Quality still
control the LoRA rank, training duration, and dataset preparation.

The chosen profile is saved with settings, worker outcomes, and interrupted-run
recovery. Changing it invalidates readiness, separates validation caches, and
prevents retrying or comparing checkpoints under a different profile. The
exported workflow retains its tested prediction settings independently of later
Connect selections. Older records default to standard SDXL; newly written
settings/outcomes use newer schema versions and require the updated application.

The CLI exposes the same selection through `--model-profile` on `src.pipeline`,
`src.train`, and `src.validate`, or `MIVIM_MODEL_PROFILE` in `.env`. Values are
`sdxl`, `noobai-xl-epsilon`, and `noobai-xl-vpred`. NoobAI training also requires
`--base-model` pointing to the selected local checkpoint.

### V-pred trainer correction

The inspected [ai-toolkit trainer](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/sd_trainer/SDTrainer.py)
passes `batch.tensor` to its V-pred velocity target. Its
[batch loader](https://github.com/ostris/ai-toolkit/blob/main/toolkit/data_transfer_object/data_loader.py)
uses that field for image pixels and leaves it absent with cached latents.
The target must use encoded latents; the [Diffusers SDXL training example](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_sdxl.py)
uses that input for its velocity target. Studio recognizes the exact upstream
V-pred assignment and replaces only its `batch.tensor` argument with
`batch.latents` through an in-memory import adapter. An already corrected
assignment is also accepted. Other velocity calls and loss branches are left
intact. Unknown control flow or arguments remain blocked before preparation.

The child checks the trainer file's SHA-256 against the inspected bytes before
executing it. A changed checkout requires another setup check. The loader never
reads or writes that module's bytecode cache, and no external source is edited.
The same isolated launcher patches ai-toolkit's scheduler factory to enable
zero-terminal-SNR for both DDPM training and Euler previews. It checks prediction
type, schedule support, and DDPM's terminal alpha before model loading. These
checks are not a substitute for training qualification with actual weights.
Validation-only CLI runs remain available without the trainer. Relabeling
V-pred weights as Epsilon is not a repair.

## Corrections to the older model notes

| Claim or family | Research result and project consequence |
| --- | --- |
| NoobAI vs. Illustrious vs. Pony | NoobAI's official cards identify Illustrious ancestry. A family label does not guarantee every LoRA's behavior across fine-tunes, merges, or prediction variants. Record the exact release and its declared training base. |
| Hassaku XL as a single Illustrious/Pony hybrid | The supplied name does not identify a release or mixing recipe. A creator-owned source for the exact claimed hybrid was not established in this review. Keep it unverified; no automatic profile or lineage claim has been added. |
| WAI and Nova as interchangeable replacements | The supplied names do not identify exact versions or files. No particular release or automatic substitution was qualified here; verify each artifact's own lineage and configuration. |
| FLUX anime as another XL checkpoint | FLUX.1 uses a different architecture and conditioning path. ComfyUI's [official guide](https://docs.comfy.org/tutorials/flux/flux-1-text-to-image) distinguishes packaged checkpoints from separate diffusion-model, CLIP-L/T5, and autoencoder loading. FLUX is outside this SDXL trainer/workflow; an SDXL VAE or SDXL LoRA is not a general substitute. |
| Animagine XL 4.x | The verified reference is [Animagine XL 4.0](https://huggingface.co/cagliostrolab/animagine-xl-4.0), an SDXL fine-tune. Its card documents its own tag order and settings. It does not establish the supplied face-quality or content-filtering ranking. This addition does not qualify a dedicated Animagine training profile. |
| Clip skip 2 everywhere | Neither inspected NoobAI card prescribes a universal CLIP-skip override. Follow the exact model's instructions and the host's layer convention; do not copy a numeric setting indiscriminately between UIs. |
| A square image cannot show a whole figure | This is not an architectural rule. The cards include square and non-square resolutions. Composition and output quality require actual evaluation; larger dimensions do not guarantee framing. |
| Grey output always needs a replacement VAE; muddy output always means high CFG | These are diagnostic hypotheses, not established causes. Check the exact checkpoint, prediction mode, decoder, precision, and workflow before attributing a fault. |
| A 200 MB file proves a poor checkpoint | File size alone does not establish artifact type or quality. It may be an adapter or another packaging format. Verify the publisher, artifact type, base, and hash. |
| “Quality king,” best current merges, or universal LoRA weights | No controlled comparison was supplied or established. Popularity, sample galleries, and anecdotal impressions do not prove a general ranking. No such ranking or fixed LoRA stack is encoded in Studio. |

The model cards are authoritative for their documented configuration and stated
lineage, not independent proof of comparative quality. The cited releases are
verified reference points; this review does not claim they are the newest or
most popular releases in every hosting catalog.

## Verification boundary

Source tests cover profile persistence, keyboard selection, readiness
invalidation, prediction/sampler consistency, editable workflow round-trips,
cache separation, recovery binding, trainer rejection, and the isolated target
correction. The adapter tests cover both upstream and already corrected targets,
cached and uncached inputs, unchanged Epsilon behavior, source drift rejection,
and preservation of the trainer file and bytecode cache. A regression also
confirms that checkpoint filename hints ignore parent directory names.

The corrected source suite passed **324 tests** (the then-current **2 packaged
UI tests were deselected**). The rebuilt **0.6.1** executable subsequently passed
**4 packaging tests**: icon validity, native photo preparation retaining a V-pred
profile, and result handoff for both standard SDXL and V-pred. The latter uses
fixture images and model files; it proves the packaged UI/schema path, not model
generation. Ruff lint/format, Mypy, Pyright, and whitespace checks passed.

Package: `dist/noobai-0.6.1/MIVIMLoRAStudio.dist/MIVIMLoRAStudio.exe`.
Distribute its entire sibling folder. The earlier package remains in place.
Executable SHA-256:
`FA1D727B7F10F73C62661D691BB125A247718E04B064E9C51BCECD9CA53948DF`.
The matching wheel is
`dist/noobai-0.6.1/lora_pipeline_design-0.6.1-py3-none-any.whl`.
The build launcher now uses the deployment interpreter directly, avoiding stale
console-launcher paths after a repository move. Logs are retained at
`dist/noobai-corrections-source-tests.log`,
`dist/noobai-corrections-package-profiles.log`, and
`dist/noobai-0.6.1-build-retry.log`. Native result screenshots are in
`dist/noobai-corrections-package-profiles`; the earlier Connect/profile popup/
Project captures remain at `dist/noobai-*.png`.

`scripts/verify_vpred_trainer.py` exercised the installed ai-toolkit's real loss
method and DDPM/Euler schedulers on synthetic CPU tensors, at checkout revision
`e65c4d0fb69251e692390574c49873297dc4bae5`. The original cached
V-pred path raised `AttributeError: 'NoneType' object has no attribute 'device'`.
The corrected loss and backward gradients matched the analytical target for
cached and uncached V-pred and Epsilon inputs, including the terminal timestep.
Evidence: `dist/noobai-vpred-numerical.log`. Reproduce with:

```powershell
uv run python -m scripts.verify_vpred_trainer `
  --toolkit dist/acceptance-tools/ai-toolkit `
  --python dist/acceptance-tools/ai-toolkit/.venv/Scripts/python.exe
```

Live NoobAI training with actual weights, ComfyUI image generation, and visual
quality remain unqualified. No model weights were downloaded for these checks.
The initial old-executable handoff failure on the new settings schema is
preserved as historical evidence; the new executable passes that acceptance
case. An initial rebuild also failed on the moved environment's stale launcher;
`dist/noobai-0.6.1-build.log` retains that failure before the launcher repair.
