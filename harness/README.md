# MBZUAI LLM Evaluation

Before running this script, please do `wandb login`.

```
cd script
python evaluate.py\
    --experiment_ckpt /lustre/scratch/users/william.neiswanger/mbzuai_llm_wikipedia5x_60p/workdir_7b/\
    --experiment_name eval7b_wikipedia5x_60p \
    --output_folder ../output/output60p_0shot \
    --run_every 5
```

Parameter definition:
* `experiment_ckpt` is the path to the list of all experiment checkpoints.
* `experiment_name` is the experiment name for wandb.
* `output_folder` is the path to the output folder
* `run_every` means to evaluate the checkpoint for every certain multiple of checkpoint

This script needs to be run in a `tmux` session. It will run in an infinite loop to check new checkpoints and regularly update the `wandb` for the evaluation scores.
