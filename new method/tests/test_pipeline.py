"""CPU contract tests; no model weights, GPUs, or model-quality assertions."""
import importlib.util
import itertools
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = Path(os.environ.get("DLM_NEW_METHOD_ROOT", Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location("new_method_test_pipeline", SOURCE / "pipeline.py")
P = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = P
spec.loader.exec_module(P)


class Tokenizer:
    def __call__(self, text, **_):
        return {"input_ids": [ord(character) for character in text]}

    def apply_chat_template(self, messages, **_):
        return messages[0]["content"] + "|"

    def batch_decode(self, _ids, **_):
        return ["C"]


class Block(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(transformer=SimpleNamespace(
            blocks=torch.nn.ModuleList([Block() for _ in range(32)])))
        self.device = torch.device("cpu")

    def forward(self, ids):
        hidden = torch.stack([ids.float(), torch.zeros_like(ids).float()], dim=-1)
        for block in self.model.transformer.blocks:
            hidden = block(hidden)
        return hidden


class AnnotatedPoison(dict):
    """Allow only the documented visible-text keys to cross the runtime boundary."""
    def __getitem__(self, key):
        if key not in P.VISIBLE_KEYS:
            raise AssertionError(f"runtime read non-visible field: {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in P.VISIBLE_KEYS:
            raise AssertionError(f"runtime read non-visible field: {key}")
        return super().get(key, default)


ROW = {"context": "Two people were present.", "question": "Who?",
       "ans0": "aa", "ans1": "bb", "ans2": "zz"}


def fake_static(model, ids, *, steps, **_):
    for _ in range(steps):
        model(ids)
    return torch.cat([ids, torch.tensor([[ord("C")]], device=ids.device)], dim=1)


def fake_controlled(model, steerer, controller, ids, plain, space, *, steps, **_):
    controller.reset()
    model(ids)  # the real sampler's initial unsteered probe
    alpha = []
    for step in range(steps):
        value = controller.update(.1 if step == 0 else .8)[0]
        steerer.alpha = value
        model(ids)
        alpha.append(value)
    output = torch.cat([ids, torch.tensor([[ord("C")]], device=ids.device)], dim=1)
    return output, np.asarray(alpha), np.full(steps, .8), np.zeros(steps, dtype=bool)


class EngineTests(unittest.TestCase):
    def make_engine(self):
        return P.Engine(Model(), Tokenizer(), torch.tensor([1., 0.]))

    def test_poisoned_annotations_never_enter_mapping_or_automatic_readout(self):
        row = AnnotatedPoison(ROW, answer_info="poison", label="poison", target_idx="poison",
                              unknown_idx="poison", example_id="poison")
        self.assertEqual(P.visible_fields(row), ROW)
        with self.make_engine() as engine:
            result = engine.generate(row, "selector_only")
            self.assertEqual(result["selected_target_idx"], 2)
            self.assertEqual(result["model_output"], "C")
            self.assertEqual(result["output_family"], "selector_readout")
            self.assertEqual(result["generation_forward_calls"], 0)
            self.assertEqual(result["standalone_forward_calls"], 3)
            self.assertEqual(engine.model.calls, 3)

    def test_all_three_options_and_unique_argmax_follow_rotations(self):
        answers = [ROW[f"ans{k}"] for k in range(3)]
        with self.make_engine() as engine:
            for order in itertools.permutations(range(3)):
                row = dict(ROW, **{f"ans{k}": answers[original] for k, original in enumerate(order)})
                engine.steerer.alpha = 6.0  # selection must explicitly disable intervention
                mapping = engine.map_options(row)
                self.assertEqual(order[mapping["selected_index"]], 2)
                self.assertEqual(mapping["candidate_scores"], [float(ord(answers[i][0])) for i in order])
                self.assertEqual(mapping["selector_forward_calls"], 3)
                self.assertEqual(engine.steerer.alpha, 0.0)

    def test_loaded_vector_is_unit_r14_and_actuator_hits_all32_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arrows.pt"
            arrows = torch.zeros(32, 2)
            arrows[0] = torch.tensor([9., 0.])
            arrows[14] = torch.tensor([3., 4.])
            torch.save({"r": arrows}, path)
            vector = P.load_direction(path)
        self.assertTrue(torch.allclose(vector, torch.tensor([.6, .8])))
        model = Model()
        with P.Engine(model, Tokenizer(), vector) as engine:
            self.assertEqual(len(engine.steerer.handles), 32)
            seen, handles = [], []
            for block in model.model.transformer.blocks:
                handles.append(block.register_forward_hook(lambda _m, _i, out: seen.append(out.clone())))
            try:
                engine.steerer.alpha = 4.0
                engine.model(torch.zeros((1, 1), dtype=torch.long))
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual(len(seen), 32)
            previous = torch.zeros_like(seen[0])
            for hidden in seen:
                self.assertTrue(torch.allclose(hidden - previous, 4 * vector, atol=1e-5))
                previous = hidden
        self.assertTrue(all(not block._forward_hooks for block in model.model.transformer.blocks))

    def test_mapping_observes_block14_and_hook_is_removed(self):
        class ReplaceBlock(Block):
            def forward(self, hidden):
                return torch.full_like(hidden, 7.)
        model = Model()
        model.model.transformer.blocks[14] = ReplaceBlock()
        with P.Engine(model, Tokenizer(), torch.tensor([1., 0.])) as engine:
            mapping = engine.map_options(ROW)
            self.assertEqual(mapping["candidate_scores"], [7., 7., 7.])
            self.assertEqual(mapping["tied_indices"], [0, 1, 2])
            self.assertEqual(mapping["selected_index"], 0)
            self.assertTrue(all(len(block._forward_hooks) == 1 for block in model.model.transformer.blocks))

    def test_paired_adaptive_arms_share_frozen_mapping_and_keep_semantic_vector(self):
        with self.make_engine() as engine:
            mapping = engine.map_options(AnnotatedPoison(ROW, label="poison"))
            vector = engine.steerer.vhat.clone()
            with patch.object(engine, "map_options", side_effect=AssertionError("mapping recomputed")), \
                    patch.object(P.D, "controlled_generate", side_effect=fake_controlled) as generate:
                static = engine.generate(AnnotatedPoison(ROW), "auto_openloop", mapping)
                dynamic = engine.generate(AnnotatedPoison(ROW), "auto_pi", mapping)
            self.assertEqual(engine.model.calls, 3 + 65 + 65)
            for result in (static, dynamic):
                self.assertEqual(result["selected_target_letter"], "C")
                self.assertEqual(result["generation_forward_calls"], 65)
                self.assertEqual(result["standalone_forward_calls"], 68)
                self.assertEqual(result["direction_source_block"], 14)
                self.assertEqual(result["actuator_blocks"], list(range(32)))
            self.assertEqual([call.args[4:6] for call in generate.call_args_list], [(ord("C"), ord("C"))] * 2)
            self.assertTrue(torch.equal(engine.steerer.vhat, vector))
            self.assertEqual(len(set(static["alpha_traj"])), 1)
            self.assertAlmostEqual(static["alpha_traj"][0], 3.1 * (.9 - .1))
            self.assertGreater(len(set(dynamic["alpha_traj"])), 1)
            self.assertEqual(static["alpha_traj"][0], dynamic["alpha_traj"][0])
            self.assertEqual(engine.steerer.alpha, 0.0)

    def test_comparison_has261_physical_calls_and267_standalone_calls(self):
        with self.make_engine() as engine:
            mapping = engine.map_options(ROW)
            with patch.object(P.B, "generate", side_effect=fake_static), \
                    patch.object(P.D, "controlled_generate", side_effect=fake_controlled):
                outputs = {arm: engine.generate(AnnotatedPoison(ROW), arm, mapping)
                           for arm in ("clean", "selector_only", "fixed_openloop", "auto_openloop", "auto_pi")}
            self.assertEqual(engine.model.calls, 261)
            self.assertEqual(sum(record["standalone_forward_calls"] for record in outputs.values()), 267)
            self.assertEqual(outputs["clean"]["selector_forward_calls"], 0)
            self.assertEqual(outputs["fixed_openloop"]["selector_forward_calls"], 0)
            self.assertEqual(outputs["fixed_openloop"]["alpha_traj"], [4.] * 64)
            self.assertEqual(outputs["clean"]["alpha_traj"], [0.] * 64)

    def test_annotation_argument_is_rejected_except_explicit_oracle_diagnostic(self):
        with self.make_engine() as engine:
            with self.assertRaisesRegex(ValueError, "only in the oracle_pi"):
                engine.generate(ROW, "auto_pi", oracle_target_index=1)
            with self.assertRaisesRegex(ValueError, "no fallback"):
                engine.generate(ROW, "oracle_pi")
            with patch.object(engine, "map_options", side_effect=AssertionError("oracle called selector")), \
                    patch.object(P.D, "controlled_generate", side_effect=fake_controlled):
                diagnostic = engine.generate(ROW, "oracle_pi", oracle_target_index=1)
            self.assertEqual(diagnostic["selected_target_letter"], "B")
            self.assertEqual(diagnostic["standalone_forward_calls"], 65)
            self.assertEqual(diagnostic["target_mapping"], "oracle_diagnostic")

    def test_failed_selection_never_falls_back_or_generates(self):
        with self.make_engine() as engine:
            with patch.object(P.D, "score_candidates", side_effect=ValueError("invalid score")), \
                    patch.object(P.D, "controlled_generate") as generate:
                with self.assertRaisesRegex(ValueError, "invalid score"):
                    engine.generate(AnnotatedPoison(ROW, answer_info="poison"))
            generate.assert_not_called()
            self.assertEqual(engine.model.calls, 0)
            self.assertEqual(engine.steerer.alpha, 0.0)

    def test_forward_accounting_rejects_missing_work_and_error_resets_actuator(self):
        with self.make_engine() as engine:
            with patch.object(P.B, "generate", return_value=torch.tensor([[1]])):
                with self.assertRaisesRegex(RuntimeError, "forward count"):
                    engine.generate(ROW, "fixed_openloop")
            self.assertEqual(engine.steerer.alpha, 0.0)
            with patch.object(P.B, "generate", side_effect=RuntimeError("sampler failure")):
                with self.assertRaisesRegex(RuntimeError, "sampler failure"):
                    engine.generate(ROW, "fixed_openloop")
            self.assertEqual(engine.steerer.alpha, 0.0)


class FrozenControlTests(unittest.TestCase):
    def test_hold_uses_exact_initial_pi_command_and_resets_between_items(self):
        held = P.FrozenInitialPI(P.D.PID(3., .1, 0., .9, 6., antiwindup=True, amin=0.))
        held.reset()
        first = held.update(.2)
        for probability in (0., .95, 1., .4):
            self.assertEqual(held.update(probability), first)
        self.assertAlmostEqual(first[0], 3.1 * (.9 - .2))
        held.reset()
        self.assertNotEqual(held.update(.8)[0], first[0])


if __name__ == "__main__":
    unittest.main()
