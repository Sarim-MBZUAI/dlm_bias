"""Experimental answer-option mapping from a fixed LLaDA semantic direction.

This module accepts visible text and a frozen direction, never dataset labels.
For each displayed answer, it pools the answer-token residual at the direction's
source layer and takes its raw dot product with the unit direction. All three
answers participate, including any uncertainty answer. Scores and their margin
are projection units, not calibrated probabilities of correct identification.
"""
import math

import torch


def select_target(scores):
    """Select the largest of three finite scores; exact ties use A/B/C order."""
    values = [float(value) for value in scores]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("target mapping requires exactly three finite scores")
    largest = max(values)
    ties = [index for index, value in enumerate(values) if value == largest]
    ordered = sorted(values, reverse=True)
    return {
        "selected_index": ties[0],
        "candidate_scores": values,
        "margin": ordered[0] - ordered[1],
        "tied_indices": ties,
    }


@torch.no_grad()
def score_candidates(model, tok, prompt_text, answer_texts, direction,
                     source_layer=14):
    """Run three clean completed-answer forwards and freeze their argmax.

    ``prompt_text`` is the chat-templated generation prefix. The caller must
    disable any steering hooks before calling. Pooling mirrors build_arrows,
    with strict prefix/span checks instead of its last-token fallback. Invalid
    text, token boundaries, activations, or direction abort the run; no candidate
    is discarded and there is no annotation-based fallback.
    """
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("target mapping requires a nonempty prompt string")
    if len(answer_texts) != 3 or any(
        not isinstance(answer, str) or not answer.strip() for answer in answer_texts
    ):
        raise ValueError("target mapping requires three nonempty answer strings")
    vector = torch.as_tensor(direction, dtype=torch.float32).detach()
    if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
        raise ValueError("target mapping direction must be a finite vector")
    norm = vector.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise ValueError("target mapping direction must have a positive finite norm")
    vector = vector / norm

    prompt_ids = tok(prompt_text)["input_ids"]
    if not prompt_ids:
        raise ValueError("target mapping prompt has no tokens")
    candidates = []
    for answer in answer_texts:
        ids = tok(prompt_text + answer.strip())["input_ids"]
        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("candidate tokenization changed the prompt prefix")
        if len(ids) <= len(prompt_ids):
            raise ValueError("candidate has no answer tokens after the prompt")
        candidates.append(ids)

    # Same path as bbq_eval.BLOCKS_PATH, resolved from the AutoModel wrapper.
    blocks = model.model.transformer.blocks
    if not 0 <= source_layer < len(blocks):
        raise ValueError("target mapping source layer is outside the model")
    captured = {}

    def capture(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden

    handle = blocks[source_layer].register_forward_hook(capture)
    scores = []
    try:
        for token_ids in candidates:
            captured.clear()
            ids = torch.tensor(token_ids, dtype=torch.long,
                               device=model.device).unsqueeze(0)
            model(ids)
            if "hidden" not in captured:
                raise RuntimeError("target mapping source-layer hook did not fire")
            hidden = captured.pop("hidden")
            if (hidden.ndim != 3 or hidden.shape[0] != 1
                    or hidden.shape[1] != len(token_ids)
                    or hidden.shape[2] != vector.numel()):
                raise ValueError("target mapping received unexpected hidden-state shape")
            pooled = hidden[0, len(prompt_ids):].to(torch.float32).mean(dim=0)
            score = torch.dot(pooled, vector.to(pooled.device))
            scores.append(float(score))
    finally:
        handle.remove()
    result = select_target(scores)
    result["selector_forward_calls"] = len(candidates)
    return result
