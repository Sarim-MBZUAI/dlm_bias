"""CPU tests for inference-time isolation, selection, and backward compatibility."""
import importlib
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(os.environ.get("DLM_TEST_SELECTOR_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "steering"))
selector = importlib.import_module("target_selector")
decode = importlib.import_module("denoise_pid")


class Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in text]}

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"] + "|"


class Block(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, hidden):
        return (hidden, None) if self.tuple_output else hidden


class Model(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.model = SimpleNamespace(transformer=SimpleNamespace(blocks=torch.nn.ModuleList([Block(tuple_output)])))
        self.device = torch.device("cpu")
        self.calls = 0

    def forward(self, ids):
        self.calls += 1
        hidden = torch.stack((ids.float(), torch.ones_like(ids).float()), dim=-1)
        for block in self.model.transformer.blocks:
            output = block(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        return output


class PoisonedMetadata(dict):
    forbidden = {"answer_info", "label", "black_idx", "target_idx", "biased_idx", "example_id",
                 "question_index", "unknown_idx", "additional_metadata"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"inference accessed annotation {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"inference accessed annotation {key}")
        return super().get(key, default)


class SelectorTests(unittest.TestCase):
    def test_unique_max_is_permutation_equivariant(self):
        scores = [-2, 9, 1]
        for permutation in itertools.permutations(range(3)):
            result = selector.select_target([scores[i] for i in permutation])
            self.assertEqual(permutation[result["selected_index"]], 1)
            self.assertEqual(result["margin"], 8)

    def test_ties_are_explicit_and_nonfinite_scores_fail(self):
        result = selector.select_target([2, 2, 1])
        self.assertEqual((result["selected_index"], result["tied_indices"], result["margin"]), (0, [0, 1], 0))
        for scores in ([1, 2], [1, float("nan"), 3], [1, 2, float("inf")]):
            with self.assertRaises(ValueError):
                selector.select_target(scores)

    def test_all_candidates_raw_projection_answer_only_and_three_forwards(self):
        for tuple_output in (False, True):
            model = Model(tuple_output)
            result = selector.score_candidates(model, Tokenizer(), "PROMPT|", ["aa", "zz", "bbb"],
                                               torch.tensor([7.0, 0]), source_layer=0)
            self.assertEqual(result["candidate_scores"], [97.0, 122.0, 98.0])
            self.assertEqual(result["selected_index"], 1)
            self.assertEqual(result["selector_forward_calls"], 3)
            self.assertEqual(model.calls, 3)
            self.assertFalse(model.model.transformer.blocks[0]._forward_hooks)

    def test_all_three_candidates_include_uncertainty_text(self):
        model = Model()
        result = selector.score_candidates(model, Tokenizer(), "P|", ["aa", "bb", "zz"],
                                           torch.tensor([1., 0]), source_layer=0)
        self.assertEqual(result["selected_index"], 2)  # no privileged exclusion of slot C

    def test_source_layer14_uses_transformer_block14_not_embedding_or_block0(self):
        class LastBlock(Block):
            def forward(self, hidden):
                return torch.full_like(hidden, 7.)
        model = Model()
        model.model.transformer.blocks.extend([Block() for _ in range(13)] + [LastBlock()])
        result = selector.score_candidates(model, Tokenizer(), "P|", ["aa", "bb", "cc"],
                                           torch.tensor([1., 0]), source_layer=14)
        self.assertEqual(result["candidate_scores"], [7., 7., 7.])
        self.assertTrue(all(not block._forward_hooks for block in model.model.transformer.blocks))

    def test_invalid_direction_or_empty_answer_fails_without_forwards(self):
        for direction, answers in [(torch.zeros(2), ["a", "b", "c"]),
                                   (torch.tensor([float("nan"), 0]), ["a", "b", "c"]),
                                   (torch.ones(2), ["a", "", "c"])]:
            model = Model()
            with self.assertRaises(ValueError):
                selector.score_candidates(model, Tokenizer(), "P|", answers, direction, source_layer=0)
            self.assertEqual(model.calls, 0)

    def test_changed_token_prefix_fails_without_fallback(self):
        class MergeTokenizer(Tokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2] if text == "P" else [1, 9, 3]}
        model = Model()
        with self.assertRaisesRegex(ValueError, "prefix"):
            selector.score_candidates(model, MergeTokenizer(), "P", ["a", "b", "c"], torch.ones(2), 0)
        self.assertEqual(model.calls, 0)

    def test_capture_hook_removed_after_forward_failure(self):
        model = Model()
        with patch.object(model, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                selector.score_candidates(model, Tokenizer(), "P|", ["a", "b", "c"], torch.ones(2), 0)
        self.assertFalse(model.model.transformer.blocks[0]._forward_hooks)


class GenerationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(device=torch.device("cpu"))
        self.tok = Tokenizer()
        self.steerer = SimpleNamespace(alpha=5.9, vhat=torch.tensor([1., 0]))
        self.row = {"context": "A person and another person.", "question": "Who?",
                    "ans0": "First person", "ans1": "Second person", "ans2": "Cannot tell"}
        self.generated = (torch.tensor([[1, 2]]), np.array([1., 2.]), np.array([.2, .4]), np.array([False, False]))

    def test_direction_branch_never_reads_annotations_and_freezes_mapping(self):
        poisoned = PoisonedMetadata(self.row, answer_info="poison", label="poison", black_idx="poison")
        def select(*args, **kwargs):
            self.assertEqual(self.steerer.alpha, 0)
            self.assertEqual(args[3], [self.row[f"ans{k}"] for k in range(3)])
            return {"selected_index": 2, "candidate_scores": [1., 2., 3.], "margin": 1.,
                    "tied_indices": [2], "selector_forward_calls": 3}
        with patch.object(decode, "score_candidates", side_effect=select) as select_mock, \
                patch.object(decode, "controlled_generate", return_value=self.generated) as generate:
            result = decode.generate_item(self.model, self.tok, self.steerer, object(), poisoned,
                                          True, target_mapping="direction", steps=64)
        self.assertEqual(select_mock.call_count, 1)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args.args[4:6], (ord("C"), ord("C")))
        self.assertEqual(result[-1]["selected_letter"], "C")
        self.assertEqual(result[-1]["total_forward_calls"], 68)

    def test_selector_failure_does_not_use_annotation_fallback(self):
        with patch.object(decode, "score_candidates", side_effect=ValueError("bad score")), \
                patch.object(decode, "controlled_generate") as generate:
            with self.assertRaisesRegex(ValueError, "bad score"):
                decode.generate_item(self.model, self.tok, self.steerer, object(),
                                     PoisonedMetadata(self.row), True, target_mapping="direction")
        generate.assert_not_called()

    def test_default_remains_oracle_without_selector_calls(self):
        row = dict(self.row, answer_info={"ans0": ["first", "white"],
                                         "ans1": ["second", "Black"], "ans2": ["unknown", "unknown"]})
        with patch.object(decode, "score_candidates", side_effect=AssertionError("oracle called selector")), \
                patch.object(decode, "controlled_generate", return_value=self.generated) as generate:
            implicit = decode.generate_item(self.model, self.tok, self.steerer, object(), row, True)
            first_args = generate.call_args.args[4:6]
            explicit = decode.generate_item(self.model, self.tok, self.steerer, object(), row, True,
                                            target_mapping="oracle")
        self.assertEqual(implicit[-1], explicit[-1])
        self.assertEqual(first_args, (ord("B"), ord("B")))
        self.assertEqual(implicit[-1]["selector_forward_calls"], 0)
        self.assertEqual(implicit[-1]["total_forward_calls"], 65)


if __name__ == "__main__":
    unittest.main()
