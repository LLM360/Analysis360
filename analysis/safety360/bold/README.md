# BOLD
This folder contains implementations to evaluate [LLM360](https://www.llm360.ai/) models on [BOLD](https://arxiv.org/abs/2101.11718) dataset, which evaluates social biases in language models across five domains: profession, gender, race, religion, and political ideology.

## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

The folder contains sentiment analysis for BOLD dataset. [Amber](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f) and [Crystal](https://huggingface.co/collections/LLM360/crystal-65e733d14e6a0786c4f5a606) models are currently supported. 

### Directory Structure

``single_ckpt_bold_eval.py`` is the main entrypoint for running BOLD evaluation on a single model. It uses python modules in ``utils/`` folder.

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``data_utils.py``: Prompt dataset utils
- ``model_utils.py``: Model loader

The BOLD prompts are stored in ``./data/prompts/``. By default, the model generations are saved in ``./{prompt_file_name}_with_responses.jsonl``, and the evaluation results are saved in ``./{model_name}_results.jsonl``.

## Installation
1. Clone and enter the folder:
    ```bash
    git clone https://github.com/LLM360/Analysis360.git
    cd Analysis360/analysis/safety360/bold
    ```
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### BOLD Evaluation
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.