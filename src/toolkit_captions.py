"""Full-caption SDXL encoding inside the selected ai-toolkit worker.

The worker payload also works in the frozen Windows application. It patches only
the worker's SDXL encoder, leaving the trainer checkout and caption files intact.
Short captions retain the trainer's existing implementation. Long captions use
complete CLIP sections, as ComfyUI does, with pooled output from the first section.
"""

from __future__ import annotations

from pathlib import Path

from .toolkit_vpred import vpred_runner


def caption_runner(toolkit_dir: Path, *, v_prediction: bool = False) -> str:
    if v_prediction:
        return vpred_runner(toolkit_dir, before_run=CAPTION_ADAPTER)
    return _BOOTSTRAP + CAPTION_ADAPTER + _RUN_TOOLKIT


_BOOTSTRAP = r"""
import runpy
import sys
from pathlib import Path
run_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(run_path.parent))
"""

_RUN_TOOLKIT = r"""
sys.argv = [str(run_path), *sys.argv[2:]]
runpy.run_path(str(run_path), run_name="__main__")
"""

CAPTION_ADAPTER = r"""
import functools
import inspect
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from toolkit import train_tools

original_encode_prompts_xl = train_tools.encode_prompts_xl

def caption_sections(tokenizer, token_ids, sections):
    width = tokenizer.model_max_length
    capacity = width - 2
    rows = []
    for ids in token_ids:
        chunks = []
        for index in range(sections):
            payload = ids[index * capacity:(index + 1) * capacity]
            chunk = [tokenizer.bos_token_id, *payload, tokenizer.eos_token_id]
            chunk += [tokenizer.pad_token_id] * (width - len(chunk))
            chunks.append(chunk)
        rows.append(chunks)
    return torch.tensor(rows, dtype=torch.long)

def encode_full_prompts_xl(
    tokenizers, text_encoders, prompts, prompts2=None, num_images_per_prompt=1,
    use_text_encoder_1=True, use_text_encoder_2=True, truncate=True,
    max_length=None, dropout_prob=0.0,
):
    prompt_lists = [prompts, prompts if prompts2 is None else prompts2]
    all_ids = [
        tokenizer(text, add_special_tokens=False, truncation=False, verbose=False).input_ids
        for tokenizer, text in zip(tokenizers, prompt_lists)
    ]
    sections = max(
        max(1, (len(ids) + tokenizer.model_max_length - 3) // (tokenizer.model_max_length - 2))
        for tokenizer, rows in zip(tokenizers, all_ids) for ids in rows
    )
    if max_length is not None:
        sections = max(sections, (max_length + tokenizers[0].model_max_length - 1) // tokenizers[0].model_max_length)
    if sections == 1:
        return original_encode_prompts_xl(
            tokenizers, text_encoders, prompts, prompts2,
            num_images_per_prompt=num_images_per_prompt,
            use_text_encoder_1=use_text_encoder_1, use_text_encoder_2=use_text_encoder_2,
            truncate=truncate, max_length=max_length, dropout_prob=dropout_prob,
        )

    embeddings = []
    pooled = None
    for encoder_index, (tokenizer, encoder, rows) in enumerate(zip(tokenizers, text_encoders, all_ids)):
        enabled = (use_text_encoder_1, use_text_encoder_2)[encoder_index]
        rows = [ids if enabled else [] for ids in rows]
        if dropout_prob > 0:
            rows = [ids if torch.rand(1).item() > dropout_prob else [] for ids in rows]
        tokens = caption_sections(tokenizer, rows, sections).to(encoder.device)
        encoded_sections = []
        for index in range(sections):
            result = encoder(tokens[:, index, :], output_hidden_states=True)
            encoded_sections.append(result.hidden_states[-2])
            if encoder_index == 1 and index == 0:
                pooled = result[0]
        embeddings.append(torch.cat(encoded_sections, dim=1).repeat_interleave(num_images_per_prompt, dim=0))
    return torch.cat(embeddings, dim=-1), pooled.repeat_interleave(num_images_per_prompt, dim=0)

train_tools.encode_prompts_xl = encode_full_prompts_xl

def align_preview_captions(pipeline_class):
    original_call = pipeline_class.__call__
    signature = inspect.signature(original_call)

    @functools.wraps(original_call)
    def generate(self, *args, **kwargs):
        call = signature.bind(self, *args, **kwargs)
        positive = call.arguments.get("prompt_embeds")
        negative = call.arguments.get("negative_prompt_embeds")
        if positive is not None and negative is not None:
            # Match ai-toolkit's concat_prompt_embeds padding for CFG batches.
            length = max(positive.shape[1], negative.shape[1])
            for key, embeds in (("prompt_embeds", positive), ("negative_prompt_embeds", negative)):
                if embeds.shape[1] < length:
                    call.arguments[key] = torch.nn.functional.pad(embeds, (0, 0, 0, length - embeds.shape[1]))
        return original_call(*call.args, **call.kwargs)

    pipeline_class.__call__ = generate

align_preview_captions(StableDiffusionXLPipeline)
align_preview_captions(StableDiffusionXLImg2ImgPipeline)
print("[train] Full caption encoding enabled; trailing tags are retained.", flush=True)
"""
