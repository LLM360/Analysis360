# LLM360 Evaluation

## Amber

Before running this script, please do `wandb login`.

```
cd scripts
python evaluate.py\
    --experiment_ckpt /lustre/scratch/users/<home directory>/<model directory>/workdir_7b/\
    --experiment_name <experiment name> \
    --output_folder ../output/<output folder> \
    --run_every 5
```

Parameter definition:
* `experiment_ckpt` is the path to the list of all experiment checkpoints.
* `experiment_name` is the experiment name for wandb.
* `output_folder` is the path to the output folder
* `run_every` means to evaluate the checkpoint for every certain multiple of checkpoint

This script needs to be run in a `tmux` session. It will run in an infinite loop to check new checkpoints and regularly update the `wandb` for the evaluation scores.

## CrystalCoder

We rely on [Bigcode harness](https://github.com/bigcode-project/bigcode-evaluation-harness) and [EleutherAI harness](https://github.com/EleutherAI/lm-evaluation-harness) to run evaluations for CrystalCoder.  `crystalcoder_eval.py` records all the configuration for the tests we have ran so far.

Sample commands:
- For bigcode harness:
`CUDA_VISIBLE_DEVICES=0 python bigcode-evaluation-harness/main.py --model <MODEL> --batch_size=1 --max_length_generation <MAX_LEN> --n_samples 1 --temperature <TEMP> --tasks humaneval --allow_code_execution --trust_remote_code --save_generations --save_generations_path <YOUR_PATH>/<SOME_OUTPUT_NAME>.json --metric_output_path <YOUR_PATH>/<SOME_OUTPUT_NAME>.json --precision bf16`

- For lm-harness:
`CUDA_VISIBLE_DEVICES=0 python lm-evaluation-harness/main.py --no_cache --model=hf-causal-experimental --batch_size=2 --model_args="pretrained=<MODEL>,trust_remote_code=True,dtype=bfloat16" --tasks=<TASK> --num_fewshot=<KSHOT> --output_path=<YOUR_PATH>/<SOME_OUTPUT_NAME>.json`
