"""Visible-text target identification followed by fixed-direction LLaDA steering.

Only this module loads torch. Cohort preparation and offline scoring are separate
from the generation boundary; no dataset annotation is accepted by Engine.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
from pathlib import Path
import sys
import time

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_new_method_denoise", CODE_ROOT / "steering" / "denoise_pid.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)
B = D.B
LETTERS = "ABC"
ARMS = ("auto_pi", "auto_openloop", "fixed_openloop", "clean", "selector_only", "oracle_pi")
AUTOMATIC_ARMS = ("auto_pi", "auto_openloop", "selector_only")
VISIBLE_KEYS = ("context", "question", "ans0", "ans1", "ans2", "prompt_override")


def visible_fields(row):
    """Read only a whitelist of text fields, never annotations or item identifiers."""
    visible = {key: row[key] for key in VISIBLE_KEYS if key in row}
    for key in ("ans0", "ans1", "ans2"):
        if not isinstance(visible.get(key), str) or not visible[key].strip():
            raise ValueError(f"{key} must be nonempty visible answer text")
    if visible.get("prompt_override"):
        if not isinstance(visible["prompt_override"], str):
            raise ValueError("prompt_override must be text")
    elif any(not isinstance(visible.get(key), str) or not visible[key].strip()
             for key in ("context", "question")):
        raise ValueError("context and question must be nonempty visible text")
    return visible


@dataclass(frozen=True)
class Settings:
    steps: int = 64
    kp: float = 3.0
    ki: float = 0.1
    setpoint: float = 0.9
    alpha_max: float = 6.0
    fixed_alpha: float = 4.0
    sensor_case: str = "upper"

    def __post_init__(self):
        values = (self.kp, self.ki, self.setpoint, self.alpha_max, self.fixed_alpha)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("controller settings must be finite")
        if self.steps < 1 or not 0 <= self.setpoint <= 1:
            raise ValueError("steps must be positive and setpoint must be in [0, 1]")
        if min(self.kp, self.ki, self.alpha_max, self.fixed_alpha) < 0:
            raise ValueError("this push-only experiment requires nonnegative gains and amplitudes")
        if self.fixed_alpha > self.alpha_max:
            raise ValueError("fixed alpha must obey the same alpha_max bound")
        if self.sensor_case not in D.SENSOR_CASES:
            raise ValueError("unsupported sensor case")


class FrozenInitialPI:
    """Hold the exact first PI command; subsequent measurements cannot change it."""
    def __init__(self, controller):
        self.controller = controller
        self.initial = None

    def reset(self):
        self.controller.reset()
        self.initial = None

    def update(self, probability):
        if self.initial is None:
            self.initial = self.controller.update(probability)
        return self.initial


class CountedModel:
    """Count real model calls while delegating model attributes/hooks unchanged."""
    def __init__(self, model):
        self.wrapped = model
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.wrapped(*args, **kwargs)


def synchronize(model):
    if torch.device(model.device).type == "cuda":
        torch.cuda.synchronize(model.device)


def load_direction(path):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if "r" not in blob or blob["r"].ndim != 2 or blob["r"].shape[0] != 32:
        raise ValueError("direction artifact must contain r with shape (32, hidden_size)")
    vector = blob["r"][14].detach().to(torch.float32)
    if not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= 0:
        raise ValueError("r[14] must be finite and nonzero")
    return vector / vector.norm()


class Engine:
    """One loaded model and fixed semantic actuator, reusable across visible items."""
    def __init__(self, model, tok, vhat, settings=None):
        self.model = CountedModel(model)
        self.tok = tok
        self.settings = settings or Settings()
        vector = torch.as_tensor(vhat, dtype=torch.float32).detach()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= 0:
            raise ValueError("a finite nonzero semantic direction is required")
        self.steerer = D.attach_all_layers(self.model, vector / vector.norm())
        self.plain, self.space = D.letter_token_ids(tok, self.settings.sensor_case)

    def close(self):
        self.steerer.alpha = 0.0
        self.steerer.detach()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _prompt(self, visible):
        visible = visible_fields(visible)
        chat = self.tok.apply_chat_template(
            [{"role": "user", "content": B.build_prompt(visible)}],
            add_generation_prompt=True, tokenize=False)
        return visible, chat

    def map_options(self, visible):
        """Score all three answers with steering disabled; freeze one letter."""
        visible, chat = self._prompt(visible)
        self.steerer.alpha = 0.0
        synchronize(self.model)
        started, calls = time.perf_counter(), self.model.calls
        mapping = D.score_candidates(self.model, self.tok, chat,
                                     [visible[f"ans{k}"] for k in range(3)],
                                     self.steerer.vhat, source_layer=14)
        synchronize(self.model)
        mapping = dict(mapping, selected_letter=LETTERS[mapping["selected_index"]],
                       selector_elapsed_s=time.perf_counter() - started,
                       selector_forward_calls=self.model.calls - calls)
        if mapping["selector_forward_calls"] != 3:
            raise RuntimeError("selector must perform exactly three model forwards")
        return mapping

    @torch.no_grad()
    def generate(self, visible, arm="auto_pi", mapping=None, oracle_target_index=None):
        """Generate from visible text; oracle targets require the explicit diagnostic arm.

        Optional precomputed mapping is shared across paired automatic arms. The
        selected letter changes only the sensor; the semantic vector never changes.
        """
        if arm not in ARMS:
            raise ValueError("unknown arm")
        if oracle_target_index is not None and arm != "oracle_pi":
            raise ValueError("oracle target is permitted only in the oracle_pi diagnostic")
        visible, chat = self._prompt(visible)
        if arm in AUTOMATIC_ARMS:
            mapping = self.map_options(visible) if mapping is None else dict(mapping)
            index = mapping["selected_index"]
            if type(index) is not int or index not in range(3):
                raise ValueError("mapping must select an index in 0..2")
        elif arm == "oracle_pi":
            if type(oracle_target_index) is not int or oracle_target_index not in range(3):
                raise ValueError("oracle_pi requires an explicit valid target index; no fallback")
            index, mapping = oracle_target_index, None
        else:
            index, mapping = None, None
        ids = torch.tensor(self.tok(chat)["input_ids"], dtype=torch.long,
                           device=self.model.device).unsqueeze(0)
        synchronize(self.model)
        started, calls = time.perf_counter(), self.model.calls
        self.steerer.alpha = 0.0
        try:
            if arm == "selector_only":
                output, alpha, probability, saturated = LETTERS[index], [], None, []
            elif arm in ("clean", "fixed_openloop"):
                amplitude = self.settings.fixed_alpha if arm == "fixed_openloop" else 0.0
                self.steerer.alpha = amplitude
                x = B.generate(self.model, ids, steps=self.settings.steps,
                               gen_length=32, block_length=32, temperature=0.0,
                               cfg_scale=0.0, remasking="low_confidence")
                output = self.tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
                alpha, probability, saturated = [amplitude] * self.settings.steps, None, []
            else:
                controller = D.PID(self.settings.kp, self.settings.ki, 0.0,
                                   self.settings.setpoint, self.settings.alpha_max,
                                   antiwindup=True, amin=0.0)
                if arm == "auto_openloop":
                    controller = FrozenInitialPI(controller)
                letter = LETTERS[index]
                x, a, p, sat = D.controlled_generate(
                    self.model, self.steerer, controller, ids,
                    self.plain[letter], self.space[letter], steer_on=True,
                    steps=self.settings.steps)
                output = self.tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
                alpha, probability, saturated = a.tolist(), p.tolist(), sat.tolist()
            synchronize(self.model)
        finally:
            self.steerer.alpha = 0.0
        generation_calls = self.model.calls - calls
        expected = (0 if arm == "selector_only" else self.settings.steps
                    if arm in ("clean", "fixed_openloop") else self.settings.steps + 1)
        if generation_calls != expected:
            raise RuntimeError(f"unexpected generation forward count: {generation_calls} != {expected}")
        selection_calls = mapping["selector_forward_calls"] if mapping else 0
        selection_time = mapping["selector_elapsed_s"] if mapping else 0.0
        elapsed = time.perf_counter() - started
        return {
            "arm": arm, "model_output": output,
            "output_family": "selector_readout" if arm == "selector_only" else "diffusion_generation",
            "target_mapping": "direction" if mapping else "oracle_diagnostic" if arm == "oracle_pi" else "unused",
            "selected_target_idx": index,
            "selected_target_letter": LETTERS[index] if index is not None else None,
            "target_mapping_scores": mapping["candidate_scores"] if mapping else None,
            "target_mapping_margin": mapping["margin"] if mapping else None,
            "target_mapping_ties": mapping["tied_indices"] if mapping else [],
            "selector_forward_calls": selection_calls,
            "generation_forward_calls": generation_calls,
            "standalone_forward_calls": generation_calls + selection_calls,
            "selector_elapsed_s": selection_time, "generation_elapsed_s": elapsed,
            "standalone_elapsed_s": elapsed + selection_time,
            "alpha_traj": alpha, "selected_probability_traj": probability,
            "saturation_traj": saturated,
            "settings": asdict(self.settings), "direction_source_block": 14,
            "actuator_blocks": list(range(32)),
        }
