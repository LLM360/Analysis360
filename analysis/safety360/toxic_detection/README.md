# Toxicity Detection
This folder contains implementations for toxicity detection benchmarks on [LLM360](https://github.com/LLM360) models. The benchmark measures model's capability on identifying toxic text.

## Table of Contents 
- [Overview](#overview)
  - [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training and Evaluation](#training-and-evaluation)

## Overview

Here's a list of toxicity detection benchmarks we have implemented so far. 

<div align="center">

|Benchmark|Model|
|------|-----|
|[Social Bias Frames](https://aclanthology.org/2020.acl-main.486.pdf)|[Crystal](https://huggingface.co/collections/LLM360/crystal-65e733d14e6a0786c4f5a606)<br>[Amber](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f)|
|[Dynahate](https://arxiv.org/abs/2012.15761)|[Crystal](https://huggingface.co/collections/LLM360/crystal-65e733d14e6a0786c4f5a606)<br>[Amber](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f)|
|[Toxigen](https://arxiv.org/abs/2203.09509)|[Crystal](https://huggingface.co/collections/LLM360/crystal-65e733d14e6a0786c4f5a606)<br>[Amber](https://huggingface.co/collections/LLM360/amber-65e7333ff73c7bbb014f2f2f)|

</div>

### Directory Structure

``single_ckpt_toxic_detection.py`` is the main entrypoint for evaluating toxicity detection on a single model. It uses python modules in ``utils/`` folder.

The ``utils/`` folder contains helper functions for model/dataset IO:
- ``data_utils.py``: Dataset preparation for all benchmarks
- ``model_utils.py``: Model loader

By default, the evaluation results are saved in ``./{model_name}_results.jsonl``.

## Installation
1. Clone and enter the folder:
    ```bash
    git clone https://github.com/LLM360/Analysis360.git
    cd analysis360/analysis/safety360/toxic_detection
    ```
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### Evaluation and
An example usage is provided in the [demo.ipynb](demo.ipynb), which can be executed with a single ``A100 80G`` GPU.