# Directory Overview

Welcome to the ``analysis/`` directory! This folder contains various analysis implementations for [LLM360](https://www.llm360.ai/) models. Each subfolder is an independent and self-contained module with setup instructions, relying soley on the code within the subfolder.

1. [Data memorization (`memorization/`)](memorization/) evaluates model memorization of the training data.
2. [LLM Unlearning (`unlearn/`)](unlearning) implements machine unlearning methods to remove an LLM's hazardous knowledge.
3. [Safety360 (`safety360/`)](safety360/) contains modules to measure model safety:
    - [`bold/`](safety360/bold/) provides sentiment analysis with [BOLD](https://arxiv.org/abs/2101.11718) dataset.
    - [`toxic_detection/`](safety360/toxic_detection/) measures model's capability to identify toxic text.
    - [`toxigen/`](safety360/toxigen/) evaluate model's toxicity on text generation.
    - [`wmdp/`](safety360/wmdp/) evaluate model's hazardous knowledge.
4. [Mechanistic Interpretability (`mechinterp/`)](mechinterp/) contains packages visualizing algorithms executed by LLMs during inference.
5. [Evaluation metrics (`metrics/`)](metrics/) contains modules for model evaluation:
    - [`harness/`](metrics/harness/) provides instructions to evaluate models following the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard).
    - [`ppl/`](metrics/ppl/) evaluates model per-token perplexity
