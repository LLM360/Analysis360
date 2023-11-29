# Analysis360: Analyze LLMs in 360 degrees

<div align="center">
   <img src="./docs/imgs/llm360-big.png" height=50% width=50%><br><br>
</div>

---

<p align="center">
   <a href="https://github.com/LLM360/Analysis360/blob/dev/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="license"></a>
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

We run evaluations on a variety of benchmarks, including the conventional benchmarks like MMLU, Hellaswag, user-preference aligned benchmarks like MT-bench, long-context evaluations like LongEval, and additional studies on safety benchmarks for truthfulness, toxicity and bias. Moreover, we report results on the model samples we preselectedfrom a suite of LLMs where they all trained on same data seen in the exact same order to better observe and understand how our models develop and evolve over the training process. We also provide public access to all checkpoints, all wandb dashboards for detailed training and evaluation curves.

## List of Analysis and Metrics

Here's a full list of analysis/metrics we have collected so far. We will keep expanding the list as our study proceeds, please stay tuned on the upcoming changes!
| Metrics/Analysis                                                                                                                    | Description                                                                                                                                                 | Amber                                                                                                  | CrystalCoder |
|-------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------:|:------------:|
| mmlu                                                                                                                                | a test to measure a text model’s multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more | [&check;](https://wandb.ai/mbzuai-llm/test/reports/mmlu-23-11-28-16-08-14---Vmlldzo2MTA5MTY3)          |              |
| race                                                                                                                                | a test to measure reading comprehension ablity                                                                                                              | [&check;](https://wandb.ai/mbzuai-llm/test/reports/race-23-11-28-16-12-11---Vmlldzo2MTA5MTky)          |              |
| arc_challenge                                                                                                                       | a set of grade-school science questions                                                                                                                     | [&check;](https://wandb.ai/mbzuai-llm/test/reports/arc_challenge-23-11-28-16-13-34---Vmlldzo2MTA5MjAx) |              |
| boolq                                                                                                                               | a question answering dataset for yes/no questions containing 15942 examples                                                                                 | [&check;](https://wandb.ai/mbzuai-llm/test/reports/boolq-23-11-28-16-14-02---Vmlldzo2MTA5MjA0)         |              |
| hellaswag                                                                                                                           | a test of commonsense inference                                                                                                                             | [&check;](https://wandb.ai/mbzuai-llm/test/reports/hellaswag-23-11-28-16-14-34---Vmlldzo2MTA5MjEw)     |              |
| openbookqa                                                                                                                          | a question-answering dataset modeled after open book exams for assessing human understanding of a subject                                                   | [&check;](https://wandb.ai/mbzuai-llm/test/reports/openbookqa-23-11-28-16-16-31---Vmlldzo2MTA5MjIz)    |              |
| piqa                                                                                                                                | a test to measure physical commonsense and reasoning                                                                                                        | [&check;](https://wandb.ai/mbzuai-llm/test/reports/piqa-23-11-28-16-17-02---Vmlldzo2MTA5MjI5)          |              |
| siqa                                                                                                                                | a test to measure commonsense reasoning about social interactions                                                                                           | [&check;](https://wandb.ai/mbzuai-llm/test/reports/siqa-23-11-28-16-21-40---Vmlldzo2MTA5MjUy)          |              |
| winogrande                                                                                                                          | an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning                                                                         | [&check;](https://wandb.ai/mbzuai-llm/test/reports/winogrande-23-11-28-16-22-16---Vmlldzo2MTA5MjU2)    |              |
| crowspairs                                                                                                                          | a challenge set for evaluating what language models (LMs) on their tendency to generate biased outputs                                                      | [&check;](https://wandb.ai/mbzuai-llm/test/reports/crowspairs-23-11-28-16-22-45---Vmlldzo2MTA5MjYw)    |              |
| truthfulqa                                                                                                                          | a test to measure a model’s propensity to reproduce falsehoods commonly found online                                                                        | [&check;](https://wandb.ai/mbzuai-llm/test/reports/truthfulqa-23-11-28-16-23-20---Vmlldzo2MTA5MjY3)    |              |
| pile                                                                                                                                | a test to measure models’ perplexity, we covered 18/22 sub datasets                                                                                         | [&check;](https://wandb.ai/mbzuai-llm/test/runs/8odaqd7f)                                              |              |
| bias generation vs identification ([bold](https://arxiv.org/abs/2101.11718))                                                        | an analysis to evaluate the tradeoffs between bias generation and identification                                                                            |                                                                                                        |              |
| toxicity generation vs identification ([toxigen](https://arxiv.org/abs/2203.09509) and [dynahate](https://arxiv.org/abs/2012.15761) | an analysis to evaluate the tradeoffs between toxicity generation and identification                                                                        |                                                                                                        |              |
| [Memorization and token orders](https://arxiv.org/abs/2202.07646)                                                                   | an analysis to understand models’ memorization abilities                                                                                                    |                                                                                                        |              |