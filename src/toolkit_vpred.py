"""Isolated ai-toolkit compatibility adapter for NoobAI V-prediction.

Passed to the trainer's own Python with -c, including in frozen Studio builds.
The selected checkout is never edited. Patch before trainer modules import
get_sampler, and verify scheduler behavior before run.py can load the model.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

TRAINER_SOURCE = Path("extensions_built_in/sd_trainer/SDTrainer.py")


def velocity_target(source: str) -> ast.Attribute:
    """Recognize only the known direct V-pred target in SDTrainer's loss.

    Other velocity calls, such as the unaugmented-image loss, are unrelated.
    Accept the upstream image-target bug and the corresponding latent fix;
    changed control flow or argument conventions require a new review.
    """
    tree = ast.parse(source)
    test = ast.dump(
        ast.parse("self.sd.prediction_type == 'v_prediction'", mode="eval").body
    )
    matches = [
        branch
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "SDTrainer"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "calculate_loss"
        for branch in ast.walk(method)
        if isinstance(branch, ast.If) and ast.dump(branch.test) == test
    ]
    if len(matches) == 1:
        body = matches[0].body
        for field in ("tensor", "latents"):
            expected = ast.parse(
                f"target = self.sd.noise_scheduler.get_velocity(batch.{field}, noise, timesteps)"
            ).body
            if [ast.dump(node) for node in body] == [
                ast.dump(node) for node in expected
            ]:
                assignment = body[0]
                assert isinstance(assignment, ast.Assign)
                assert isinstance(assignment.value, ast.Call)
                target = assignment.value.args[0]
                assert isinstance(target, ast.Attribute)
                return target
    raise ValueError("Unrecognized ai-toolkit V-pred velocity-target implementation.")


def vpred_runner(toolkit_dir: Path) -> str:
    """Bind the child adapter to the exact inspected source bytes and AST node."""
    source = (toolkit_dir / TRAINER_SOURCE).read_bytes()
    target = velocity_target(source.decode("utf-8"))
    approved = (hashlib.sha256(source).hexdigest(), target.lineno, target.col_offset)
    return f"approved_target = {approved!r}\n" + _VPRED_RUNNER


_VPRED_RUNNER = r"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import runpy
import sys
from pathlib import Path

run_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(run_path.parent))
trainer_name = "extensions_built_in.sd_trainer.SDTrainer"
trainer_path = run_path.parent / "extensions_built_in/sd_trainer/SDTrainer.py"

class LatentTargetLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = trainer_path.read_bytes()
        digest, line, column = approved_target
        if hashlib.sha256(source).hexdigest() != digest:
            raise RuntimeError("ai-toolkit changed after the V-pred compatibility check. Run Check setup again.")
        tree = ast.parse(source, filename=str(trainer_path))
        targets = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute)
                   and node.lineno == line and node.col_offset == column
                   and isinstance(node.value, ast.Name) and node.value.id == "batch"
                   and node.attr in ("tensor", "latents")]
        if len(targets) != 1:
            raise RuntimeError("The verified V-pred velocity target is unavailable.")
        targets[0].attr = "latents"
        print("[train] NoobAI V-pred: verified latent velocity target", flush=True)
        # Compile in memory; never read or write the trainer's bytecode cache.
        return compile(tree, str(trainer_path), "exec")

class LatentTargetFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == trainer_name:
            return importlib.util.spec_from_loader(
                fullname, LatentTargetLoader(fullname, str(trainer_path)))
        return None

if trainer_name in sys.modules:
    raise RuntimeError("The trainer was imported before the V-pred adapter.")
sys.meta_path.insert(0, LatentTargetFinder())
from toolkit import sampler
from toolkit.config_modules import ModelConfig

if not ModelConfig(name_or_path="probe", arch="sdxl", is_v_pred=True).is_v_pred:
    raise RuntimeError("This ai-toolkit does not support V-prediction. Update ai-toolkit.")
original_get_sampler = sampler.get_sampler

def noobai_get_sampler(name, kwargs=None, arch="sd"):
    options = dict(kwargs or {})
    if arch == "sd" and options.get("prediction_type") == "v_prediction":
        options["rescale_betas_zero_snr"] = True
    scheduler = original_get_sampler(name, options, arch=arch)
    if options.get("rescale_betas_zero_snr"):
        if not scheduler.config.rescale_betas_zero_snr:
            raise RuntimeError("The trainer scheduler cannot apply NoobAI zero-terminal-SNR.")
    return scheduler

sampler.get_sampler = noobai_get_sampler
for name in ("ddpm", "euler"):
    scheduler = noobai_get_sampler(name, {"prediction_type": "v_prediction"})
    if scheduler.config.prediction_type != "v_prediction":
        raise RuntimeError("The trainer scheduler cannot apply V-prediction.")
    if name == "ddpm":
        if not callable(getattr(scheduler, "get_velocity", None)):
            raise RuntimeError("The trainer scheduler has no velocity target.")
        if float(scheduler.alphas_cumprod[-1]) > 1e-7:
            raise RuntimeError("The trainer schedule does not end at zero SNR.")
print("[train] NoobAI V-pred: verified DDPM/Euler zero-terminal-SNR schedulers", flush=True)
sys.argv = [str(run_path), *sys.argv[2:]]
runpy.run_path(str(run_path), run_name="__main__")
"""
