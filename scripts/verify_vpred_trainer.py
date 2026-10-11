"""Exercise the installed trainer's loss and backward pass without model weights.

Run with Studio's Python; --python selects the ai-toolkit environment. This is
a synthetic numerical compatibility check, not model or image-quality evidence.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.toolkit_vpred import vpred_runner

PROBE = r"""
import json
from types import SimpleNamespace
import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig

torch.manual_seed(731)
trainer = SDTrainer.__new__(SDTrainer)
trainer.device_torch = torch.device("cpu")
trainer.train_config = TrainConfig(dtype="float32", loss_type="mse", loss_target="noise")
trainer.dfe = None
trainer.adapter = None
trainer.sd = SimpleNamespace(
    prediction_type="v_prediction", is_flow_matching=False,
    noise_scheduler=sampler.get_sampler("ddpm", {"prediction_type": "v_prediction"}),
    scale_loss=lambda loss: loss,
)
trainer.additional_logs = {}
latents = torch.randn(2, 4, 8, 8)
noise = torch.randn_like(latents)
timesteps = torch.tensor([0, 999], dtype=torch.int64)
noisy_latents = trainer.sd.noise_scheduler.add_noise(latents, noise, timesteps)
batch = SimpleNamespace(
    tensor=None, latents=latents, mask_tensor=None,
    loss_multiplier_list=[1.0, 1.0], get_is_reg_list=lambda: [False, False],
)

# Compile the original, unmodified upstream method in its real module globals.
tree = ast.parse(trainer_path.read_bytes())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SDTrainer")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_loss")
namespace = dict(SDTrainer.calculate_loss.__globals__)
exec(compile(ast.Module(body=[method], type_ignores=[]), str(trainer_path), "exec"), namespace)
original = namespace["calculate_loss"]
try:
    original(trainer, torch.zeros_like(latents), noise, noisy_latents, timesteps, batch)
    original_failure = None
except (AttributeError, TypeError, RuntimeError) as error:
    original_failure = f"{type(error).__name__}: {error}"

checks = []
for mode in ("v_prediction", "epsilon"):
    trainer.sd.prediction_type = mode
    for cached in (True, False):
        batch.tensor = None if cached else torch.randn(2, 3, 64, 64)
        pixels = batch.tensor
        prediction = torch.randn_like(latents, requires_grad=True)
        alpha = trainer.sd.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latents if mode == "v_prediction" else noise
        expected = ((prediction - target) ** 2).mean()
        actual = trainer.calculate_loss(prediction, noise, noisy_latents, timesteps, batch)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - target) / prediction.numel())
        assert torch.isfinite(prediction.grad).all()
        assert batch.tensor is pixels
        checks.append({"prediction": mode, "cached_latents": cached, "loss": actual.item(), "backward": "passed"})
print(json.dumps({"original_cached_vpred_failure": original_failure, "checks": checks}, indent=2))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    toolkit = args.toolkit.resolve()
    runner = vpred_runner(toolkit)
    # Use the production import hook and scheduler adapter, replacing only the
    # final run.py invocation with this bounded, weight-free numerical probe.
    runner = runner.rsplit('runpy.run_path(str(run_path), run_name="__main__")', 1)[0]
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        NO_ALBUMENTATIONS_UPDATE="1",
    )
    completed = subprocess.run(
        [str(args.python.resolve()), "-c", runner + PROBE, str(toolkit / "run.py")],
        cwd=toolkit,
        env=environment,
        timeout=120,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
