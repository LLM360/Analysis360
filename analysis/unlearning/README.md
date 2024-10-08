# Unlearning
This folder contains implementations for [machine unlearning](https://arxiv.org/abs/2402.08787) methods on [LLM360](https://www.llm360.ai/) models. Machine unlearning is a pre-deployment safety measure designed to remove hazardous knowledge from language models. Unlearned models are inherently safe, as they lack the knowledge to be misused. 

## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

Here's a list of unlearning methods we have implemented so far. 

<div align="center">

|Method|Model|
|------|-----|
|[max_entropy](https://arxiv.org/abs/2408.00761)|[CrystalChat](https://huggingface.co/LLM360/CrystalChat)|
|[min_posterior](https://arxiv.org/abs/2408.00761)|[CrystalChat](https://huggingface.co/LLM360/CrystalChat)|
|[random_matching](https://arxiv.org/abs/2408.00761)|[CrystalChat](https://huggingface.co/LLM360/CrystalChat)|
|[RMU](https://arxiv.org/abs/2408.00761)|[CrystalChat](https://huggingface.co/LLM360/CrystalChat)|

</div>

### Directory Structure

``unlearn.py`` is the main entrypoint for running unlearning methods. It uses python modules in ``methods/`` and ``utils/`` folders.

The ``methods/`` folder contains the implementations for unlearning methods:
- ``training.py``: All training loop implementations
- ``utils.py``: Loss functions and other method-related utils

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``data_utils.py``: Dataloader for text datasets
- ``model_utils.py``: Model IO utils

By default, unlearned models are saved to ``models/`` folder. Please store all training datasets to the ``data/`` folder. 

> [!NOTE]
> This project uses the [bio-forget-corpus](https://huggingface.co/datasets/cais/wmdp-corpora) from the [WMDP Benchmark](https://www.wmdp.ai/) for unlearning training. Access to this dataset requires a separate request. Please follow the instructions provided [here](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform) to obtain the necessary permissions. By default, the dataloader is configured to load the dataset from ``data/bio_forget.jsonl``.

## Installation
1. Clone and enter the repo:
    ```bash
    git clone https://github.com/LLM360/Analysis360.git
    cd Analysis360/analysis/unlearning
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. To install ``lm-eval``, please check the [installation instructions](../metrics/harness) in the ``metrics/harness`` folder.

## Quick Start

### Training and Evaluation
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.
