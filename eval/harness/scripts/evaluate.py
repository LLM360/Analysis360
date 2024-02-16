import argparse
import glob
import os
import subprocess
import time
import wandb

from collections import defaultdict
from utils import tasks, extract_all_scores
task_info = dict(tasks)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_ckpt",
        default="",
    )
    parser.add_argument(
        "--ckpt_samples",
        nargs="+",
        default=None
    )
    parser.add_argument(
        '--slurm',
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--experiment_name",
        default="",
    )
    parser.add_argument(
        "--wandb_experiment_id",
        default="",
    )
    parser.add_argument(
        "--output_folder",
        default="",
    )
    parser.add_argument(
        "--run_every",
        default=5,
    )
    parser.add_argument(
        "--log_samples",
        action="store_true"
    )
    parser.add_argument(
        "--use_cache",
        default=None)
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        choices=["CRITICAL","ERROR","WARNING","INFO","DEBUG"]
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="hf-causal-experimental",
        choices=["vllm", "hf-causal-experimental"]
    )

    # set slurm args
    slurm_parser = parser.add_argument_group('slurm_args')
    parse_slurm_args(slurm_parser)

    args = parser.parse_args()
    return args

def parse_slurm_args(slurm_parser):
    """These args are all for slurm launch."""
    slurm_parser.add_argument(
        '--partition',
        default='gpumid',
    )
    slurm_parser.add_argument(
        '--gres',
        default='gpu:1',
    )
    slurm_parser.add_argument(
        '--reservation',
        default='analysis',
    )
    slurm_parser.add_argument(
        '--mem',
        default='100G',
    )
    slurm_parser.add_argument(
        '--nodes',
        default=1,
    )
    slurm_parser.add_argument(
        '--cpus-per-task',
        default=4,
    )
    slurm_parser.add_argument(
        '--ntasks',
        default=1,
    )


def get_checkpoints_with_result(path):
    ckpts = defaultdict(list)
    for file in glob.glob(path):
        if file.split('/')[-1].split('_')[0].isdigit():
            ckpt = int(file.split('/')[-1].split('_')[0])
            task = file.split('/')[-1].split('_')[1]
            ckpts[ckpt].append(task)
    final_ckpts = []
    for key in ckpts.keys():
        if len(ckpts[key]) == 11:
            final_ckpts.append(key)
    final_ckpts.sort()
    return final_ckpts


def main():

    args = parse_args()

    # initialize wandb
    if args.wandb_experiment_id != '':
        wandb.init(
          project="Amber_evaluations_HF",
          name=f"{args.experiment_name}",
          id=args.wandb_experiment_id,
          resume="must",
          config={})
    else:
        wandb.init(
          project="Amber_evaluations_HF",
          name=f"{args.experiment_name}",
          config={})

    # define our custom x axis metric
    wandb.define_metric("custom_step")
    # define which metrics will be plotted against it
    wandb.define_metric("*", step_metric="custom_step")


    # Check existing folder output
    COMPLETED_CKPTS = []
    final_ckpts = get_checkpoints_with_result(f"{args.output_folder}/*")

    # Submit job
    for ckpt in final_ckpts:
        log_scores = extract_all_scores(args.output_folder, ckpt)
        COMPLETED_CKPTS.append(ckpt)
        wandb.log(log_scores)


    # Start the process
    os.makedirs(args.output_folder, exist_ok = True)
    current_dir = os.getcwd()
    work_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    log_scores = {}

    while (True):
        # Check if the new checkpoint exists
        files = glob.glob(f"{args.experiment_ckpt}/ckpt_*")
        files_to_check = []
        if args.ckpt_samples:
            ckpt_samples = [int(ckpt) for ckpt in args.ckpt_samples]
            for file in files:
                ckpt = int(file.split('/')[-1].split('_')[1])
                if ckpt in ckpt_samples and ckpt not in COMPLETED_CKPTS:
                    files_to_check.append(file)
        else:
            for file in files:
                ckpt = int(file.split('/')[-1].split('_')[1])
                if ckpt not in COMPLETED_CKPTS:
                    files_to_check.append(file)

        if len(files_to_check) == 0:
            # sleep for 4 hours
            print('There is no new checkpoint')
            time.sleep(14400)
        else:
            # run the evaluation by submitting all job
            print('Found', len(files_to_check), 'checkpoints')
            for file in files_to_check:
                print('Submitting evaluation jobs of', file)
                ckpt = int(file.split('/')[-1].split('_')[1])
                for task in task_info.keys():
                    task_map, num_fewshot, task_args = task_info[task]
                    out_file = f'{args.output_folder}/{ckpt}_{task}.json'
                    if not os.path.isfile(out_file):
                        # Run Locally
                        device = "cuda"
                        cache_str = f" --use_cache={args.use_cache}" if args.use_cache else ""
                        task_cmd = f"python {work_dir}/main.py --model {args.model} --model_args pretrained={file} --tasks {task_map} --num_fewshot {num_fewshot} --output_path {out_file}  --device {device} --batch_size 16 --verbosity {args.verbosity}{cache_str}"
                        if args.log_samples:
                            task_cmd += " --log_samples"
                        cmd = task_cmd
                        # Slurm runner
                        if args.slurm:
                            slurm_cmd = f"srun --nodes={args.nodes} --reservation={args.reservation} --partition={args.partition} --ntasks={args.ntasks} --cpus-per-task={args.cpus-per-task}"\
                            "--gres={args.gres} --mem={args.mem} --output={current_dir}/stdout/slurm-%j.out --job-name={ckpt}_{task}"
                            cmd = slurm_cmd + task_cmd + ' &'
                        result = subprocess.run(cmd, shell=True, text=True,)

            # check if the run is all complete
            while (True):
                current_completed_ckpts = get_checkpoints_with_result(f"{args.output_folder}/*")

                new_ckpts = [int(file.split('/')[-1].split('_')[1]) for file in files_to_check if (file.split('/')[-1].split('_')[1].isdigit())]
                new_ckpts.sort()
                if len(set(new_ckpts) - set(current_completed_ckpts)) == 0:
                    # All job is finish, push logs scores to wandb
                    print('All evaluations are finished')
                    for ckpt in new_ckpts:
                        log_scores = extract_all_scores(args.output_folder, ckpt)
                        COMPLETED_CKPTS.append(ckpt)
                        wandb.log(log_scores)
                    break
                else:
                    # sleep for 2 hours
                    print("sleeping")
                    time.sleep(200)


if __name__ == '__main__':
    main()
