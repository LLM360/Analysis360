# Perplexity
This folder contains implementations to measure model's per-token perplexity. 

## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

The folder contains code for model perplexity measurement. [Amber](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f) and [Crystal](https://huggingface.co/collections/LLM360/crystal-65e733d14e6a0786c4f5a606) models are currently supported. 

### Directory Structure

``single_ckpt_ppl_eval.py`` is the main entrypoint for calculating perplexity on a single model. It uses python modules in ``utils/`` folder.

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``data_utils.py``: Dataset IO utils
- ``model_utils.py``: Model loader

 We provide a sample dataset at ``./data/wikitext.txt``, which contains a 1,000-line random sample from the [wikitext-2-v1](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-v1) train split. By default, the perplexity results are saved in ``./results.josn``.

## Installation
1. Clone and enter the folder:
    ```bash
    git clone https://github.com/LLM360/Analysis360.git
    cd Analysis360/analysis/metrics/ppl
    ```
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Perplexity evaluation
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.