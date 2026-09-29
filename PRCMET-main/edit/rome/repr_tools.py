"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )
def get_inputs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_templates: List[str],
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_inputs_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_templates,
        track,
    )

def get_words_idxs_in_templates(
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    subtoken: str,
) -> List[List[int]]:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"

    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)

    # Pre-process tokens
    for i, prefix in enumerate(prefixes):
        if len(prefix) > 0:
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)
    # Tokenizers such as Llama-3 add a BOS token automatically. Tokenizing all
    # three fragments with the default settings counts that BOS three times,
    # even though the filled context contains it only once. In particular,
    # subject_last for the template "{}" then points one past the sequence.
    # Keep special tokens on the prefix only and tokenize the remaining
    # fragments as continuations of the same sequence.
    prefixes_tok = tok(prefixes, add_special_tokens=True)["input_ids"]
    words_tok = tok(words, add_special_tokens=False)["input_ids"]
    suffixes_tok = tok(suffixes, add_special_tokens=False)["input_ids"]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, attention_mask, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        sequence_length = cur_repr.shape[1]
        valid_lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
        for i, idx_list in enumerate(batch_idxs):
            valid_length = int(valid_lengths[i])
            padding_offset = (
                sequence_length - valid_length if tok.padding_side == "left" else 0
            )
            resolved_idxs = []
            for idx in idx_list:
                if idx < -valid_length or idx >= valid_length:
                    raise IndexError(
                        "Representation token index is outside the unpadded "
                        f"sequence: sample={i}, index={idx}, "
                        f"valid_length={valid_length}."
                    )
                unpadded_idx = idx if idx >= 0 else valid_length + idx
                resolved_idxs.append(padding_offset + unpadded_idx)
            to_return[key].append(cur_repr[i][resolved_idxs].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, contexts_tok["attention_mask"], "in")
        if tout:
            _process(tr.output, batch_idxs, contexts_tok["attention_mask"], "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]

def get_inputs_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_templates: List[str],
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_names = [module_template.format(layer) for module_template in module_templates]
    to_return = {module_name: [] for module_name in module_names}

    def _process(cur_repr, batch_idxs, attention_mask, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        sequence_length = cur_repr.shape[1]
        valid_lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
        for i, idx_list in enumerate(batch_idxs):
            valid_length = int(valid_lengths[i])
            padding_offset = (
                sequence_length - valid_length if tok.padding_side == "left" else 0
            )
            resolved_idxs = []
            for idx in idx_list:
                if idx < -valid_length or idx >= valid_length:
                    raise IndexError(
                        "Representation token index is outside the unpadded "
                        f"sequence: sample={i}, index={idx}, "
                        f"valid_length={valid_length}."
                    )
                unpadded_idx = idx if idx >= 0 else valid_length + idx
                resolved_idxs.append(padding_offset + unpadded_idx)
            to_return[key].append(cur_repr[i][resolved_idxs].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=module_names,
                retain_input=tin
            ) as tr:
                model(**contexts_tok)

        if tin:
            for module_name in module_names:
                _process(
                    tr[module_name].input,
                    batch_idxs,
                    contexts_tok["attention_mask"],
                    module_name,
                )


    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        raise ValueError(f" len(to_return) ={ len(to_return) }  not equal to 2")
    else:
        return to_return[module_names[0]], to_return[module_names[1]]
