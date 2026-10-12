"""Check caption preservation with local pretrained SDXL encoders, offline.

Uses the production worker payload and reads source captions without editing
them. This checks encoding behavior, not trained image quality.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from src.toolkit_captions import caption_runner

PROBE = r"""
import json
from safetensors import safe_open
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from diffusers.loaders.single_file_utils import create_diffusers_clip_model_from_ldm
from toolkit.stable_diffusion_model import StableDiffusion

torch.set_num_threads(4)
checkpoint_path, config_path, caption_path, output_path = sys.argv[2:]
with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint_file:
    weights = {key: checkpoint_file.get_tensor(key) for key in checkpoint_file.keys()
               if key.startswith("conditioner.")}
encoders = [create_diffusers_clip_model_from_ldm(
    cls, weights, config=config_path, subfolder=folder,
    torch_dtype=torch.float32, local_files_only=True,
) for cls, folder in ((CLIPTextModel, "text_encoder"), (CLIPTextModelWithProjection, "text_encoder_2"))]
del weights
tokenizers = [CLIPTokenizer.from_pretrained(config_path, subfolder=folder, local_files_only=True)
              for folder in ("tokenizer", "tokenizer_2")]
sd = StableDiffusion.__new__(StableDiffusion)
sd.arch = "sdxl"
sd.tokenizer = tokenizers
sd.text_encoder = encoders

captions = [path.read_text(encoding="utf-8-sig").strip()
            for path in sorted(Path(caption_path).glob("*.txt"))]
assert captions, "No source captions to check"
long_count = 0
for tokenizer in tokenizers:
    for caption in captions:
        ids = tokenizer(caption, add_special_tokens=False, truncation=False, verbose=False).input_ids
        capacity = tokenizer.model_max_length - 2
        count = max(1, (len(ids) + capacity - 1) // capacity)
        tokens = caption_sections(tokenizer, [ids], count)[0]
        restored = []
        for index, section in enumerate(tokens.tolist()):
            size = min(capacity, max(0, len(ids) - index * capacity))
            assert section[0] == tokenizer.bos_token_id
            assert section[size + 1] == tokenizer.eos_token_id
            restored.extend(section[1:size + 1])
        assert restored == ids
        if tokenizer is tokenizers[0] and count > 1:
            long_count += 1

with torch.no_grad():
    short = ["a red circle on a white background"]
    expected = original_encode_prompts_xl(tokenizers, encoders, short, None)
    actual = sd.encode_prompt(short)
    torch.testing.assert_close(actual.text_embeds, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.pooled_embeds, expected[1], rtol=0, atol=0)

    prefix = "red circle, " * 30
    prompts = [prefix + "blue square", prefix + "green triangle"]
    old, _ = original_encode_prompts_xl(tokenizers, encoders, prompts, None)
    torch.testing.assert_close(old[0], old[1], rtol=0, atol=0)
    full = sd.encode_prompt(prompts)
    assert full.text_embeds.shape[1] > 77
    torch.testing.assert_close(full.text_embeds[0, :77], full.text_embeds[1, :77], rtol=0, atol=0)
    assert not torch.equal(full.text_embeds[0, 77:], full.text_embeds[1, 77:])

    mixed = sd.encode_prompt(["", max(captions, key=len)], num_images_per_prompt=2)
    assert mixed.text_embeds.shape[0] == mixed.pooled_embeds.shape[0] == 4
    torch.testing.assert_close(mixed.text_embeds[0], mixed.text_embeds[1])
    torch.testing.assert_close(mixed.text_embeds[2], mixed.text_embeds[3])
    distinct = sd.encode_prompt(short, prompt2=prompts[:1])
    assert distinct.text_embeds.shape[1] == full.text_embeds.shape[1]
    blank = sd.encode_prompt([""], max_length=full.text_embeds.shape[1])
    assert blank.text_embeds.shape[1] == full.text_embeds.shape[1]
    dropped, _ = train_tools.encode_prompts_xl(tokenizers, encoders, prompts[:1], None, dropout_prob=1)
    torch.testing.assert_close(dropped, blank.text_embeds)
    very_long = sd.encode_prompt(["red circle, " * 120])
    assert very_long.text_embeds.shape[1] > 4 * 77
    assert torch.isfinite(very_long.text_embeds).all()

# Frozen text encoders are the shipped default; preserve gradient flow for a
# user config that trains them. Limit this check to CLIP-L's token embeddings.
for encoder in encoders:
    encoder.requires_grad_(False)
embedding = encoders[0].get_input_embeddings().weight
embedding.requires_grad_(True)
encoded = sd.encode_prompt(prompts[:1])
encoded.text_embeds[:, 77:, :768].square().mean().backward()
assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
assert embedding.grad.abs().sum().item() > 0

result = {
    "source_captions": len(captions), "captions_over_one_section": long_count,
    "all_original_token_ids_retained": True, "short_caption_identical": True,
    "original_encoder_ignored_changed_tail": True, "full_encoder_uses_changed_tail": True,
    "mixed_batch_shape": list(mixed.text_embeds.shape),
    "distinct_encoder_prompts": True, "blank_and_dropout": True,
    "beyond_four_sections_shape": list(very_long.text_embeds.shape),
    "tail_gradient": True, "checkpoint": checkpoint_path,
}
Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2), flush=True)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "toolkit",
        "python",
        "checkpoint",
        "sdxl-config",
        "captions",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vpred", action="store_true")
    args = parser.parse_args()
    toolkit = args.toolkit.resolve()
    runner = caption_runner(toolkit, v_prediction=args.vpred)
    runner = runner.rsplit("sys.argv = [str(run_path)", 1)[0]
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")
    return subprocess.run(
        [
            str(args.python.resolve()),
            "-c",
            runner + PROBE,
            str(toolkit / "run.py"),
            str(args.checkpoint.resolve()),
            str(args.sdxl_config.resolve()),
            str(args.captions.resolve()),
            str(args.output.resolve()),
        ],
        cwd=toolkit,
        env=environment,
        timeout=180,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
