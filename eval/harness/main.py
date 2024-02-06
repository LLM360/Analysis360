import argparse
import json
import logging
import fnmatch
import os

from lm_eval_v0_3 import evaluator as v0_3_evaluator
from lm_eval_v0_3 import tasks as v0_3_tasks
from lm_eval import tasks, evaluator, utils
from lm_eval.utils import make_table
logging.getLogger("openai").setLevel(logging.WARNING)


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

    dir_path = os.path.dirname(os.path.realpath(__file__))

    print("initializzing tasks")
    tasks.initialize_tasks(args.verbosity)

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    print("tasks initialized")

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
        v0_3_task_names = v0_3_tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)
        v0_3_task_names = pattern_match(args.tasks.split(","), v0_3_tasks.ALL_TASKS)

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
            #description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
        )
    elif v0_3_task_names:
        description_dict = {}
        if args.description_dict_path:
            with open(args.description_dict_path, "r") as f:
                description_dict = json.load(f)
        if args.model == "vllm":
            model = "hf-causal-experimental"
            print('cannot use vllm with the old lm-eval harness. Replacing with "hf-causal-experimental".')
        else:
            model = args.model
        results = v0_3_evaluator.simple_evaluate(
            model=model,
            model_args=args.model_args,
            tasks=v0_3_task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            no_cache=not(args.use_cache),
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
        )
    else:
        raise RuntimeError(f"No tasks found for {args.tasks}")

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(
        f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
        f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
    )
    if task_names:
        print(make_table(results))
    else:
        print(v0_3_evaluator.make_table(results))


if __name__ == "__main__":
    main()
