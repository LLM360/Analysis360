# Harness
This folder contains instructions to evaluate [LLM360](https://github.com/LLM360) models in alignment with the [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).

## Table of Contents 
- [Installation](#installation)
- [Quick Start](#quick-start)

## Installation
Please run the following commands or visit the [official documentation](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about): 
```bash
git clone https://github.com/huggingface/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout adding_all_changess
pip install -e .[math,ifeval,sentencepiece]
```

## Quick Start
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.