# ToxiGen
This folder contains implementations to evaluate [LLM360](https://github.com/LLM360) models on [ToxiGen](https://arxiv.org/abs/2203.09509) dataset, which evaluates language model's toxicity on text generation.

## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

The folder contains code for model response generation and evaluation for ToxiGen dataset. [Amber](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f) and [Crystal](https://huggingface.co/collections/LLM360/crystal-65e733d14e6a0786c4f5a606) models are currently supported. 

### Directory Structure

``single_ckpt_toxigen.py`` is the main entrypoint for running ToxiGen on a single model. It uses python modules in ``utils/`` folder.

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``data_utils.py``: Dataset IO utils
- ``model_utils.py``: Model loader

By default, the model generations are saved in ``./{model_name}_{prompt_key}_with_responses.jsonl``, and the evaluation results are saved in ``./{model_name}_results.jsonl``.

## Installation
1. Clone and enter the folder:
    ```bash
    git clone https://github.com/LLM360/Analysis360.git
    cd analysis360/analysis/safety360/toxigen
    ```
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### ToxiGen evaluation
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.