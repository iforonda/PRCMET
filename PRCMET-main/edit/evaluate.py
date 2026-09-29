import hashlib
import json
import shutil
from itertools import islice
from time import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from baselines.ft import FTHyperParams, apply_ft_to_model
# from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from util.eval_utils.eval_utils_counterfact import compute_rewrite_quality_counterfact
from util.eval_utils.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from prcmet import PRCMETHyperParams, apply_prcmet_to_model, reset_preservation_state
from util import nethook
from util.globals import *

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PRCMET": (PRCMETHyperParams, apply_prcmet_to_model),
    # "ROME": (ROMEHyperParams, apply_rome_to_model),
    # "FT": (FTHyperParams, apply_ft_to_model),
    # "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
}


def _prompt_template_relation_id(prompt: str) -> str:
    """Create a stable zsRE relation surrogate from its edit prompt only."""

    normalized_prompt = " ".join(prompt.casefold().split())
    if "{}" not in normalized_prompt:
        raise ValueError(
            "zsRE relation anchors require an edit prompt containing '{}'."
        )
    digest = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
    return f"zsre_prompt_template:{digest}"


def _relation_anchor_request(
    record: Mapping[str, Any], relation_strategy: str = "dataset"
) -> Dict[str, Any]:
    """Project a dataset record onto fields allowed for relation anchors."""

    if relation_strategy not in {"dataset", "prompt_template"}:
        raise ValueError(f"Unknown relation identity strategy: {relation_strategy}")
    if "case_id" not in record:
        raise ValueError("Relation anchor records require case_id.")
    rewrite = record.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise ValueError("Relation anchor records require requested_rewrite.")
    for field in ("prompt", "subject"):
        if field not in rewrite:
            raise ValueError(
                f"Relation anchor records require requested_rewrite.{field}."
            )
    if not isinstance(rewrite["prompt"], str):
        raise ValueError("Relation anchor prompt must be a string.")
    relation_id = rewrite.get("relation_id")
    if relation_id is None or relation_id == "":
        if relation_strategy == "prompt_template":
            relation_id = _prompt_template_relation_id(rewrite["prompt"])
        else:
            raise ValueError(
                "Relation anchor preservation requires relation_id in the dataset."
            )
    return {
        "case_id": record["case_id"],
        "relation_id": relation_id,
        "prompt": rewrite["prompt"],
        "subject": rewrite["subject"],
    }


def _build_held_out_relation_anchor_pool(
    full_records: Sequence[Mapping[str, Any]],
    experiment_records: Sequence[Mapping[str, Any]],
    relation_strategy: str = "dataset",
    allow_empty: bool = False,
) -> List[Dict[str, Any]]:
    """Exclude every case scheduled for editing, including future batches."""

    edit_case_ids = {record.get("case_id") for record in experiment_records}
    if None in edit_case_ids:
        raise ValueError("Every edit dataset record must have a case_id.")
    # Validate the edit side now so missing relation metadata fails before editing.
    for record in experiment_records:
        _relation_anchor_request(record, relation_strategy)
    pool = [
        _relation_anchor_request(record, relation_strategy)
        for record in full_records
        if record.get("case_id") not in edit_case_ids
    ]
    if not pool and not allow_empty:
        raise ValueError(
            "Relation anchor preservation has an empty held-out pool after "
            "excluding all experiment edit case_ids."
        )
    return pool


def _normalize_anchor_subject(subject: str) -> str:
    """Strip ends, collapse all whitespace runs, then Unicode casefold."""

    return " ".join(subject.split()).casefold()


def _neutral_qa_anchor_request(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Project onto case_id, requested_rewrite.prompt and subject only."""

    if record.get("case_id") is None:
        raise ValueError("Neutral QA anchor records require case_id.")
    rewrite = record.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise ValueError("Neutral QA anchor records require requested_rewrite.")
    prompt, subject = rewrite.get("prompt"), rewrite.get("subject")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Neutral QA anchor prompt must be a non-empty string.")
    if not isinstance(subject, str) or not _normalize_anchor_subject(subject):
        raise ValueError("Neutral QA anchor subject must be a non-empty string.")
    return {"case_id": record["case_id"], "prompt": prompt, "subject": subject}


def _build_held_out_neutral_qa_anchor_pool(
    full_records: Sequence[Mapping[str, Any]],
    experiment_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Exclude case_ids and normalized subjects across the entire experiment."""

    edit_requests = [
        _neutral_qa_anchor_request(record) for record in experiment_records
    ]
    edit_case_ids = {request["case_id"] for request in edit_requests}
    edit_subjects = {
        _normalize_anchor_subject(request["subject"]) for request in edit_requests
    }
    pool = []
    for record in full_records:
        request = _neutral_qa_anchor_request(record)
        if request["case_id"] in edit_case_ids:
            continue
        if _normalize_anchor_subject(request["subject"]) in edit_subjects:
            continue
        pool.append(request)
    if not pool:
        raise ValueError(
            "zsRE neutral QA anchor candidate_count=0 after excluding case_id "
            "and normalized-subject overlaps with all "
            f"{len(experiment_records)} experiment edit records; "
            "an independent held-out QA pool is required."
        )
    return pool


def _algorithm_edit_request(
    record: Mapping[str, Any], relation_strategy: Optional[str] = None
) -> Dict[str, Any]:
    request = {"case_id": record["case_id"], **record["requested_rewrite"]}
    if relation_strategy is not None:
        request["relation_id"] = _relation_anchor_request(
            record, relation_strategy
        )["relation_id"]
    return request


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    model_path: str = None,
    single_shot: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if single_shot:
        if dataset_size_limit is None or dataset_size_limit <= 0:
            raise ValueError(
                "--single_shot requires a positive --dataset_size_limit."
            )
        if (
            alg_name == "PRCMET"
            and getattr(hparams, "preservation_constraint", False)
            and getattr(hparams, "preservation_mode", "history") == "history"
        ):
            raise ValueError(
                "History covariance is empty during the only solve and cannot "
                "affect a single-shot edit. Use preservation_mode='anchors'."
            )
    # Module caches intentionally persist across edit batches in one invocation.
    # Reset here so separate experiments (including repeated main() calls in the
    # same Python process) cannot accidentally share preservation history.
    if alg_name == "PRCMET":
        reset_preservation_state()
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        if model_path:
            print(f"Instantiating model: {model_name} from {model_path}")
            if "neox" in model_name:
                model = AutoModelForCausalLM.from_pretrained(model_path + model_name).half().cuda()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path + model_name).cuda()
            tok = AutoTokenizer.from_pretrained(model_path + model_name)
        else:
            print(f"Instantiating model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
            tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)
    requested_edit_count = len(ds)
    neutral_anchor_requests = None
    neutral_anchor_context_mode = "neutral_templates"
    neutral_anchor_enabled = (
        alg_name == "PRCMET"
        and getattr(hparams, "preservation_constraint", False) is True
        and getattr(hparams, "preservation_mode", "history") == "anchors"
    )
    if neutral_anchor_enabled and ds_name == "zsre":
        neutral_anchor_context_mode = "zsre_held_out_qa"
    if alg_name == "PRCMET":
        print(
            f"Dataset: {ds_name}; neutral anchor context mode: "
            f"{neutral_anchor_context_mode if neutral_anchor_enabled else 'inactive'}",
            flush=True,
        )
        print(f"Experiment edit set size: {requested_edit_count}", flush=True)
    if neutral_anchor_enabled and ds_name == "zsre":
        # This loader projects the complete dataset onto edit questions only;
        # it does not construct or consume locality evaluation inputs.
        full_qa_records = MENDQADataset.load_qa_anchor_records(DATA_DIR)
        neutral_anchor_requests = _build_held_out_neutral_qa_anchor_pool(
            full_qa_records, ds
        )
        print(
            "Held-out neutral QA candidate count after excluding all experiment "
            f"edit case_id/normalized-subject overlaps: {len(neutral_anchor_requests)}",
            flush=True,
        )
        del full_qa_records
    preservation_anchor_requests = None
    relation_anchor_strategy = None
    relation_anchor_enabled = (
        alg_name == "PRCMET"
        and getattr(hparams, "relation_anchor_preservation", False) is True
    )
    if relation_anchor_enabled:
        relation_anchor_strategy = (
            "prompt_template" if ds_name == "zsre" else "dataset"
        )
        full_ds = ds_class(DATA_DIR, tok=tok, size=None)
        preservation_anchor_requests = _build_held_out_relation_anchor_pool(
            full_ds, ds, relation_anchor_strategy,
            allow_empty=ds_name == "mcf",
        )
        del full_ds
        if ds_name == "mcf" and not preservation_anchor_requests:
            relation_anchor_enabled = False
            relation_anchor_strategy = None
            print(
                "mcf relation anchors skipped: all dataset case_ids are in "
                "the experiment edit set, so no independent held-out "
                "relation candidates remain.",
                flush=True,
            )
        else:
            print(
                "Held-out relation anchor pool: "
                f"{len(preservation_anchor_requests)} records after excluding "
                f"all {requested_edit_count} experiment edit case_ids",
                flush=True,
            )
            print(
                "Relation identity strategy: "
                + (
                    "normalized zsRE edit-prompt template"
                    if relation_anchor_strategy == "prompt_template"
                    else "dataset relation_id"
                ),
                flush=True,
            )
    if single_shot:
        if num_edits != requested_edit_count:
            raise ValueError(
                "--single_shot requires --num_edits to equal the loaded dataset "
                f"length ({requested_edit_count}), got {num_edits}."
            )
        print("Single-shot editing enabled", flush=True)
        print(f"Requested edits: {requested_edit_count}", flush=True)
        print("Edit protocol: single_shot", flush=True)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_{{}}_clamp_{{}}_case_{{}}.npz"
        )
        print(f"Will load cache from {cache_template}")
    print(f"kvs cache template: {cache_template}")
    apply_algorithm_calls = 0
    # Iterate through dataset
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["MEMIT", "PRCMET"]) else dict()
        if neutral_anchor_enabled:
            etc_args["neutral_anchor_context_mode"] = neutral_anchor_context_mode
            etc_args["neutral_anchor_requests"] = neutral_anchor_requests
        if relation_anchor_enabled:
            etc_args["preservation_anchor_requests"] = (
                preservation_anchor_requests
            )
        elif alg_name == "PRCMET" and ds_name == "mcf" and getattr(
            hparams, "relation_anchor_preservation", False
        ) is True:
            etc_args["skip_relation_anchors"] = True

        start = time()
        apply_algorithm_calls += 1
        if single_shot and apply_algorithm_calls > 1:
            raise RuntimeError(
                "Single-shot editing attempted more than one algorithm application."
            )
        edited_model, weights_copy = apply_algo(
            model,
            tok,
            [
                _algorithm_edit_request(record, relation_anchor_strategy)
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args,
        )
        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        print("Start evaluation")
        start = time()
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "edit_protocol": "single_shot" if single_shot else "batched",
                "requested_edit_count": requested_edit_count,
                "apply_algorithm_calls": apply_algorithm_calls,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
        print("Evaluation took", time() - start)

    if single_shot:
        if apply_algorithm_calls != 1:
            raise RuntimeError(
                "A single-shot run must apply the editing algorithm exactly once; "
                f"observed {apply_algorithm_calls} calls."
            )
        print(f"Algorithm applications: {apply_algorithm_calls}", flush=True)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "PRCMET"],
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=False,
    )
    parser.add_argument(
        "--model_path",
        default=None
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "EleutherAI/gpt-j-6B",
            "gpt2-xl",
            "meta-llama/Meta-Llama-3-8B",
        ],
        default="EleutherAI/gpt-j-6B",
        help="Model to edit.",
        required=False,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="EleutherAI_gpt-j-6B.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=False,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=100,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        default=True,
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--single_shot",
        action="store_true",
        help=(
            "Require the full loaded dataset to be passed to exactly one editing "
            "algorithm application."
        ),
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        model_path=args.model_path,
        single_shot=args.single_shot,
    )
