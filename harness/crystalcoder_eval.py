task_fewshot_pairs = [
    # ("toxigen", 0),
    # ("arc_easy", 25),
    # ("gsm8k", 8),
    # ("bigbench*", 3),
    # ("wsc273", 0),
    # ("xstory_cloze*", 0),
    # ("pawsx*", 0),
    
    ("arc_challenge", 25),
    ("hendrycksTest*", 5),
    ("triviaqa", 5),
    ("hellaswag", 10),

    ("arc_easy", 0),
    ("arc_challenge", 0),
    ("boolq", 0),
    ("winogrande", 0),
    ("openbookqa", 0),
    ("piqa", 0),
    ("truthfulqa_mc", 0),
    ("race", 0),
    ("swag", 0),
    ("hellaswag", 0),
    ("copa", 0),
    ("hendrycksTest*", 0),
    
    # new ones added by LM leaderboard
    ("gsm8k", 5),
    ("winogrande", 5),
    ("drop", 3),

]
# tasks, max_len, n_samples, temperature
task_code_args = [
    ("humaneval", 512, 20, 0.2, 5),
    ("humaneval", 512, 20, 0.8, 5),
    ("humaneval", 512, 1, 0.01, 1), # greedy

    ("mbpp", 512, 16, 0.1, 4),
    ("mbpp", 512, 16, 0.8, 4),
    ("mbpp", 512, 1, 0.01, 1), # greedy
]
def _make_multipl_e_tasks (max_len, n_samples, temperature, batchsize):
    LANGUAGES = [
        "py",
        "sh",
        "cpp",
        "cs",
        "d",
        "go",
        "java",
        "js",
        "jl",
        "lua",
        "pl",
        "php",
        "r",
        "rkt",
        "rb",
        "rs",
        "scala",
        "swift",
        "ts",
    ]
    multipl_e_tasks = []
    for language in LANGUAGES:
        multipl_e_tasks.append((f"multiple-{language}", max_len, n_samples, temperature, batchsize))
    return multipl_e_tasks

task_code_args += _make_multipl_e_tasks(max_len=650, n_samples=20, temperature=0.8, batchsize=4)
task_code_args += _make_multipl_e_tasks(max_len=650, n_samples=1, temperature=0.01, batchsize=1)
