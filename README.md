# Analysis360: Analyze LLMs in 360 degrees
<div align="center">
   <img src="./docs/imgs/llm360-icon.webp"><br><br>
</div>

-----------------
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
We run evaluations on a variety of benchmarks, including the conventional benchmarks like MMLU, Hellaswag, user-preference aligned benchmarks like MT-bench, long-context evaluations like LongEval, and additional studies on safety benchmarks for truthfulness, toxicity and bias. Moreover, we report results on the model samples we preselected from a suite of LLMs where they all trained on same data seen in the exact same order to better observe and understand how our models develop and evolve over the training process. We also provide public access to all checkpoints, all wandb dashboards for detailed training and evaluation curves.

## List of Analysis
Here's a full list of analysis/metrics we have collected so far. We will keep expanding the list as our study proceeds, please stay tuned on the upcoming changes!
| Analysis/Metrics       | Description                                                                                                                                                 |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mmlu                   | a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more |
| race                   | a test to measure reading comprehension ablity                                                                                                              |
| arc_challenge          | a set of grade-school science questions                                                                                                                     |
| boolq                  | a question answering dataset for yes/no questions containing 15942 examples                                                                                 |
| hellaswag              | a test of commonsense inference                                                                                                                             |
| openbookqa             | a question-answering dataset modeled after open book exams for assessing human understanding of a subject                                                   |
| piqa                   | a test to measure physical commonsense and reasoning                                                                                                        |
| siqa                   | a test to measure commonsense reasoning about social interactions                                                                                           |
| winogrande             | an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning                                                                         |
| crowspairs             | a challenge set for evaluating what language models (LMs) on their tendency to generate biased outputs                                                      |
| truthfulqa             | a test to measure a model's propensity to reproduce falsehoods commonly found online                                                                        |
| vicuna-instructions-80 | 80 questions cover 9 tasks including generic instructions, knowledge, math, Fermi, counterfactual, roleplay, generic, coding and writing common-sense.      |
| bold                   | a test to evaluate fairness in open-ended language generation in English language                                                                           |
| toxigen                | a test to classify input text as either hateful or not hateful                                                                                              |
### Common LM eval metrics:
- link to code
- link to wandb
### Additional model characteristics
#### toxigen
#### bold
#### more to come
