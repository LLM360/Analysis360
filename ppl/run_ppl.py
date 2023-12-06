import argparse
import wandb
import glob
import os
import time
import subprocess
import pandas as pd
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_ckpt",
    default="",
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
    default=2,
)
args = parser.parse_args()


# initialize wandb

if args.wandb_experiment_id != '':
    wandb.init(
      project="evaluations_PPL",
      name=f"{args.experiment_name}",
      id=args.wandb_experiment_id,
      resume="must",
      config={})
else:
    wandb.init(
      project="evaluations_PPL",
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
    ckpts = defaultdict(dict)
    for file in glob.glob(path):
        ckpt = int(file.split('/')[-1].replace('.csv', ''))
        ckpts[ckpt]['custom_step'] = ckpt
        df = pd.read_csv(file)        
        for _, row in df.iterrows():
            ckpts[ckpt][row['type']] = row['ppl']
    return ckpts


# Submit existing result to wandb
completed_ckpts = get_checkpoints_with_result()
for ckpt in completed_ckpts.keys():
    COMPLETED_CKPTS.append(ckpt)
    wandb.log(completed_ckpts[ckpt])

# Now begin
os.makedirs(args.output_folder, exist_ok=True)
current_dir = os.getcwd()
work_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
log_scores = {}
while (True):
    # Check if the new checkpoint exists
    files = glob.glob(f"{args.experiment_ckpt}/ckpt_*")
    files_to_check = []
    for file in files:
        ckpt = int(file.split('/')[-1].split('_')[1])
        if ckpt % int(args.run_every) == 0 and ckpt not in COMPLETED_CKPTS:
            files_to_check.append(file)

    if len(files_to_check) == 0:
        # sleep for 4 hours
        print('There is no new checkpoint')
        time.sleep(14400)
    else:
        # run the evaluation by submitting all job
        print('Found', len(files_to_check), 'checkpoints')
        for file in files_to_check:
            ckpt = int(file.split('/')[-1].split('_')[1])
            platform_args = ' --reservation=bowen --partition=gpumid --time=48:00:00 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=100G'
            
            out_file = f'{args.output_folder}/{ckpt}.csv'
            if not os.path.isfile(out_file):
                print('Submitting evaluation jobs of', file)
                slurm_cmd = f"srun --nodes=1  --output={current_dir}/stdout/slurm-%j.out {platform_args} --job-name=ppl"
                task_cmd = f' python main_ppl.py --model_id {file} --batch_size 16 --output_folder {args.output_folder}'
                cmd = slurm_cmd + task_cmd + ' &'
                result = subprocess.run(cmd, shell=True, text=True,)
        
        # check if the run is all complete
        while (True):
            current_completed_ckpts = get_checkpoints_with_result()
            new_ckpts = [int(file.split('/')[-1].split('_')[1]) for file in files_to_check]
            new_ckpts.sort()
            if len(set(new_ckpts) - set(current_completed_ckpts.keys())) == 0:
                # All job is finish, push logs scores to wandb
                print('All evaluations are finished')
                for ckpt in current_completed_ckpts.keys():
                    COMPLETED_CKPTS.append(ckpt)
                    wandb.log(current_completed_ckpts[ckpt])
                break
            else:
                # sleep for 2 hours
                time.sleep(200)
