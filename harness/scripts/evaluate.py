import argparse
import glob
import os
import subprocess
import time
import wandb
from collections import defaultdict
from utils import tasks, extract_all_scores

task_info = dict(tasks)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_ckpt",
    default="",
)
parser.add_argument("--ckpt_samples", nargs="+", default=None)
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
args = parser.parse_args()


# initialize wandb

if args.wandb_experiment_id != '':
    wandb.init(
      project="yuki_evaluations_HF",
      name=f"{args.experiment_name}",
      id=args.wandb_experiment_id,
      resume="must",
      config={})
else:
    wandb.init(
      project="yuki_evaluations_HF",
      name=f"{args.experiment_name}",
      config={})

# define our custom x axis metric
wandb.define_metric("custom_step")
# define which metrics will be plotted against it
wandb.define_metric("*", step_metric="custom_step")


# Check existing folder output

COMPLETED_CKPTS = []

def get_checkpoints_with_result():
    path = f"{args.output_folder}/*"
    ckpts = defaultdict(list)
    for file in glob.glob(path):
        ckpt = int(file.split('/')[-1].split('_')[0])
        task = file.split('/')[-1].split('_')[1]
        ckpts[ckpt].append(task)
    final_ckpts = []
    for key in ckpts.keys():
        if len(ckpts[key]) == 11:
            final_ckpts.append(key)
    final_ckpts.sort()
    return final_ckpts

final_ckpts = get_checkpoints_with_result()

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
            model_param = ["hf-causal-experimental", f"pretrained={file}", '--gres=gpu:1 --mem=100G', []]
            platform_args = ' --reservation=analysis --partition=gpumid --time=48:00:00 --ntasks=1 --cpus-per-task=4'
            model_type, model_args, slurm_args, _ = model_param
            
            for task in task_info.keys():
                task_map, num_fewshot, task_args = task_info[task]
                out_file = f'{args.output_folder}/{ckpt}_{task}.json'
                if not os.path.isfile(out_file):
                    slurm_cmd = f"srun --nodes=1  --output={current_dir}/stdout/slurm-%j.out {platform_args} --job-name={task} {slurm_args}"
                    task_cmd = f' python {work_dir}/main.py --model {model_type} --model_args {model_args} --tasks {task_map} --num_fewshot {num_fewshot} --output_path {out_file}  --device cuda --batch_size 16'
                    cmd = slurm_cmd + task_cmd + ' &'
                    result = subprocess.run(cmd, shell=True, text=True,)
        
        # check if the run is all complete
        while (True):
            current_completed_ckpts = get_checkpoints_with_result()
            new_ckpts = [int(file.split('/')[-1].split('_')[1]) for file in files_to_check]
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
                time.sleep(2000)
#wandb.finish()

