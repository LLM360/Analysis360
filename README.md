# Analysis360: Analyze LLMs in 360 degrees

<div align="center">
   <img src="./docs/imgs/llm360-big.png" height=50% width=50%><br><br>
</div>

---

<p align="center">
   <a href="https://github.com/LLM360/Analysis360/blob/dev/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="license"></a>
</p>
<p align="center">
  <a href="">Blogpost[Amber]</a> •
  <a href="">wandb dashboard[Amber]</a> •
  <a href="">Blogpost[CrystalCoder]</a> •
  <a href="">wandb dashboard[CrystalCoder]</a> •
  <a href="">Publication</a>
</p>
Welcome to LLM360!

This repo contains all the code that we used for model evaluation and analysis. It serves as the single source of truth for all evaluation metrics and provides in-depth analysis from many different angles.

## Our Approach 

We run evaluations on a variety of benchmarks, including the conventional benchmarks like MMLU, Hellaswag, user-preference aligned benchmarks like MT-bench, long-context evaluations like LongEval, and additional studies on safety benchmarks for truthfulness, toxicity and bias. Moreover, we report results on the model samples we preselectedfrom a suite of LLMs where they all trained on same data seen in the exact same order to better observe and understand how our models develop and evolve over the training process. We also provide public access to all checkpoints, all code and all wandb dashboards for detailed training and evaluation curves.

## List of Analysis and Metrics

Here's a full list of analysis/metrics we have collected so far. We will keep expanding the list as our study proceeds, please stay tuned on the upcoming changes!
| Metrics/Analysis                                                           | Description                                                                                                                                                 |                                                  Amber                                                 | CrystalCoder |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------:|--------------|
| [mmlu](https://arxiv.org/abs/2009.03300)                                   | a test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more |      [&check;](https://wandb.ai/mbzuai-llm/test/reports/mmlu-23-11-28-16-08-14---Vmlldzo2MTA5MTY3)     |              |
| [race](https://arxiv.org/abs/1704.04683)                                   | a test to measure reading comprehension ablity                                                                                                              |      [&check;](https://wandb.ai/mbzuai-llm/test/reports/race-23-11-28-16-12-11---Vmlldzo2MTA5MTky)     |              |
| [arc_challenge](https://arxiv.org/abs/1803.05457)                          | a set of grade-school science questions                                                                                                                     | [&check;](https://wandb.ai/mbzuai-llm/test/reports/arc_challenge-23-11-28-16-13-34---Vmlldzo2MTA5MjAx) |              |
| [boolq](https://arxiv.org/abs/1905.10044)                                  | a question answering dataset for yes/no questions containing 15942 examples                                                                                 |     [&check;](https://wandb.ai/mbzuai-llm/test/reports/boolq-23-11-28-16-14-02---Vmlldzo2MTA5MjA0)     |              |
| [hellaswag](https://arxiv.org/abs/1905.07830)                              | a test of commonsense inference                                                                                                                             |   [&check;](https://wandb.ai/mbzuai-llm/test/reports/hellaswag-23-11-28-16-14-34---Vmlldzo2MTA5MjEw)   |              |
| [openbookqa](https://arxiv.org/abs/1809.02789)                             | a question-answering dataset modeled after open book exams for assessing human understanding of a subject                                                   |   [&check;](https://wandb.ai/mbzuai-llm/test/reports/openbookqa-23-11-28-16-16-31---Vmlldzo2MTA5MjIz)  |              |
| [piqa](https://arxiv.org/abs/1911.11641)                                   | a test to measure physical commonsense and reasoning                                                                                                        |      [&check;](https://wandb.ai/mbzuai-llm/test/reports/piqa-23-11-28-16-17-02---Vmlldzo2MTA5MjI5)     |              |
| [siqa](https://arxiv.org/abs/1904.09728)                                   | a test to measure commonsense reasoning about social interactions                                                                                           |      [&check;](https://wandb.ai/mbzuai-llm/test/reports/siqa-23-11-28-16-21-40---Vmlldzo2MTA5MjUy)     |              |
| [winogrande](https://arxiv.org/abs/1907.10641)                             | an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning                                                                         |   [&check;](https://wandb.ai/mbzuai-llm/test/reports/winogrande-23-11-28-16-22-16---Vmlldzo2MTA5MjU2)  |              |
| [crowspairs](https://arxiv.org/abs/2010.00133)                             | a challenge set for evaluating what language models (LMs) on their tendency to generate biased outputs                                                      |   [&check;](https://wandb.ai/mbzuai-llm/test/reports/crowspairs-23-11-28-16-22-45---Vmlldzo2MTA5MjYw)  |              |
| [truthfulqa](https://arxiv.org/abs/2109.07958)                             | a test to measure a model’s propensity to reproduce falsehoods commonly found online                                                                        |   [&check;](https://wandb.ai/mbzuai-llm/test/reports/truthfulqa-23-11-28-16-23-20---Vmlldzo2MTA5MjY3)  |              |
| [pile](https://pile.eleuther.ai/)                                          | a test to measure model's perplexity, we covered 18/22 sub datasets                                                                                         |                        [&check;](https://wandb.ai/mbzuai-llm/test/runs/8odaqd7f)                       |              |
| [toxigen](https://arxiv.org/abs/2203.09509)                                | a test to measure model's toxicity on text generation                                                                                                       |                                                                                                        |              |
| [toxicity identification](https://arxiv.org/abs/2305.13169)                | a test to measure model's capability on identifying toxic text                                                                                              |                                                                                                        |              |
| [bold](https://arxiv.org/abs/2101.11718)                                   | a test to evaluate fairness in open-ended language generation in English language                                                                           |                                                                                                        |              |
| [calibration analysis](https://arxiv.org/abs/2207.05221)                   | an analysis to evaluate the tradeoffs between bias generation and identification                                                                            |                                                                                                        |              |
| [memorization and token orders analysis](https://arxiv.org/abs/2202.07646) | an analysis to understand model's memorization abilities                                                                                                    |                                                                                                        |              |
| [mt-bench](https://arxiv.org/abs/2306.05685)                               | a challenging multi-turn question set designed to evaluate the conversational and instruction-following ability of models                                   |                                                                                                        |              |
| [longeval](https://lmsys.org/blog/2023-06-29-longchat/#evaluation-toolkits-longeval)         | a test to measure how long can LLMs truly promise on context length                                                                       |                                                                                                        |              |
## How to reproduce our results
Most of our evaluations are built based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)'s core `lm_eval` module. We reused the metrics that were supported by harness and added in our own to support more. Please follow the instructions [here](./harness/README.md) to get started. For any metric that's not included, users should be able to find a dedicated folder for that metric in the root level of the repo and follow the instructions there.