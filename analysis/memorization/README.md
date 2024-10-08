# Memorization
This folder contains implementations to evaluate [LLM360](https://www.llm360.ai/) model memorization of the training data. Such memorization raises privacy concerns by potentially leaking private training data and can degrade LLM performance when the data includes unintended duplicates or anomalies. This folder adopts the _memorization score_ introduced in [Biderman et al. 2023](https://arxiv.org/abs/2304.11158) to measure model memorization. Please refer to section 4.3 of the [LLM360 paper](https://arxiv.org/pdf/2312.06550) for more implementation details. 


## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

The folder contains training data memorization evaluation for the [Amber](https://huggingface.co/LLM360/Amber) checkpoints. LLM360 project releases 360 [pretraining model checkpoints](https://huggingface.co/LLM360/Amber/tree/main) and corresponding [training data chunks](https://huggingface.co/datasets/LLM360/AmberDatasets/tree/main/train) to support transparent and reproducible research on LLM training process. 

### Directory Structure

``single_ckpt_memorization_eval.py`` is the main entrypoint for running memorization evaluation on a single model. It uses python modules in ``utils/`` folder.

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``data_utils.py``: Dataloader utils
- ``model_utils.py``: Checkpoint loader

By default, the training data chunks are saved in ``./data/train_{data_id}.jsonl``, and the evaluation results are saved in ``./result_ckpt-{ckpt_id}/data-{data_id}.json``.

## Installation
1. Clone and enter the folder:
    ```bash
    git clone https://github.com/LLM360/Analysis360.git
    cd Analysis360/analysis/memorization
    ```
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation
    ```

## Quick Start

### Memorization Evaluation
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.