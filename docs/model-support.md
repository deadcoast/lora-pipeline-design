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
| NoobAI-XL · V-pred | [Laxhar V-pred 1.0](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0) | Explicit V-prediction and zero-terminal-SNR; Euler, normal schedule, 30 steps, CFG 4.5 | Requires a compatible ai-toolkit latent-target implementation; the inspected local/upstream implementation is blocked |

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

### V-pred trainer limitation

The inspected [ai-toolkit trainer](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/sd_trainer/SDTrainer.py)
passes `batch.tensor` to its V-pred velocity target. Its
[batch loader](https://github.com/ostris/ai-toolkit/blob/main/toolkit/data_transfer_object/data_loader.py)
uses that field for image pixels and leaves it absent with cached latents.
Studio detects this implementation and blocks full V-pred runs before preparing
photos or launching the trainer. Unknown target implementations also fail closed.
Selecting a different profile is only correct when the checkpoint changes to
match; relabeling V-pred weights as Epsilon is not a repair.

For a compatible trainer using `batch.latents`, the isolated V-pred launcher
patches ai-toolkit's scheduler factory in that child process to enable
zero-terminal-SNR for both DDPM training and Euler previews. It checks prediction
type, schedule support, and DDPM's terminal alpha before model loading. It never
edits the external checkout. This compatibility check is not a substitute for
GPU training qualification. Validation-only CLI runs remain available without
the trainer.

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
cache separation, recovery binding, and incompatible trainer rejection.
The final source suite passed **318 tests**, with **2 tests of existing Windows
binaries deselected**. Ruff lint/format, Mypy, Pyright, and `git diff --check`
passed. A wheel was built under `dist/noobai-wheel`. Native Windows Qt captures
of Connect, the open profile popup, and the sibling Project page are retained
as `dist/noobai-connect.png`, `dist/noobai-popup.png`, and
`dist/noobai-project.png`. The static UI audit found no violations; the native
captures and keyboard test provide the actual desktop interaction evidence.

The installed ai-toolkit scheduler probe successfully constructed actual DDPM
and Euler V-pred schedulers with zero-terminal-SNR before showing `run.py --help`.
That probe did not load NoobAI weights or train a LoRA.

Live NoobAI GPU training, ComfyUI image generation, visual quality, and a rebuilt
Windows executable remain unqualified. An initial suite invocation accidentally
included the old packaged-result test: the existing executable rejected the new
settings schema. That failure is retained as a package-version boundary, not
counted as a passing package test. Use the updated source application until a
new Windows build is qualified.
