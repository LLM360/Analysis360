import argparse
import json
import logging
import fnmatch
import os
import re
from pathlib import Path
from lm_eval import tasks, evaluator, utils
from lm_eval.utils import make_table
logging.getLogger("openai").setLevel(logging.WARNING)


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--log_samples", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_eval_examples", action="store_true")
    parser.add_argument("--max_eval_examples_per_task", type=int, default=None)
    parser.add_argument("--use_cache", type=str, default=None)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def main():
    args = parse_args()
    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")

    tasks.initialize_tasks(args.verbosity)

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    if task_names:
        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=args.model_args,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            use_cache=args.use_cache,
            limit=args.limit,
            log_samples=args.log_samples,
            #description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
        )
    else:
        raise RuntimeError(f"No tasks found for {args.tasks}")

    if args.log_samples:
        samples = results.pop("samples")
    dumped = json.dumps(results, indent=2)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)
        if args.log_samples:
            for task_name, config in results["configs"].items():
                output_name = "{}_{}".format(
                    re.sub("/|=", "__", args.model_args), task_name
                )
                output_path = Path(args.output_path).parent.joinpath(f"{output_name}.jsonl")
                samples_dumped = json.dumps(
                    samples[task_name],
                    indent=2,
                    default=_handle_non_serializable,
                    ensure_ascii=False,
                )
                output_path.open("w", encoding='utf-8').write(samples_dumped)
    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    print(make_table(results))


if __name__ == "__main__":
    main()
