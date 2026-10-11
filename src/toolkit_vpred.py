"""Isolated ai-toolkit entry point for NoobAI's zero-terminal-SNR schedule.

Passed to the trainer's own Python with -c, including in frozen Studio builds.
The selected checkout is never edited. Patch before trainer modules import
get_sampler, and verify scheduler behavior before run.py can load the model.
"""

VPRED_RUNNER = r"""
import runpy
import sys
from pathlib import Path

run_path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(run_path.parent))
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
