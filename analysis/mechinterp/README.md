# Mechanistic Interpretability

This folder contains instructions to run [LLM360](https://github.com/LLM360) models with
packages implementing mechanistic intepretability methods and visualizations, such as
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## Table of Contents 
- [Installation](#installation)
- [Quick Start](#quick-start)

## Installation

> [!NOTE]
> This folder is a work-in-progress, more demos and instructions are coming soon.

Note that this installation is more involved than other analyses in this repository. It
requires cloning and modifying the code within TranformerLens, as we must extend it for
use with the LLM360 models.

To install TransformerLens for LLM360 models, please follow the steps below.

#### 1. Clone TransformerLens repository and install dependencies.

Clone TransformerLens from github and install all dependencies via:

```bash
git clone https://github.com/TransformerLensOrg/TransformerLens.git
cd TransformerLens
pip install -e .
```

#### 2. Edit TransformerLens code to add LLM360 models.

Next, edit the file: `transformer_lens/loading_from_pretrained.py`.

In the `OFFICIAL_MODEL_NAMES` list, add the line:
```python
    "LLM360/Amber",
```

In the `get_pretrained_state_dict` method, on line 1730, add the code block:
```python
            elif official_model_name.startswith("LLM360/Amber"):
                hf_model = AutoModelForCausalLM.from_pretrained(
                    official_model_name,
                    revision=f"ckpt_{cfg.checkpoint_value}",
                    torch_dtype=dtype,
                    token=huggingface_token,
                    **kwargs,
                )
```

#### 3. Run TransformerLens.

Following the TransformerLens [Getting
Started](https://transformerlensorg.github.io/TransformerLens/content/getting_started.html)
and [Main Demo
Notebook](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb),
import `transformer_lens` from this repo and run the example code.
