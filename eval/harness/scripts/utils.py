import json
import numpy as np


# Format  Task_name: [--tasks, --num_fewshot, 'task_specific_args']
tasks = {
#    # knowledge
    "mmlu": ["hendrycksTest-*", 5, '-'],
    "race": ["race", 0, '-'],
#
#    # commonsense reasoning
    "hellaswag": ["hellaswag", 10, '-'],
    "piqa": ["piqa", 0, '-'],
    "boolq": ["boolq", 0, '-'],
    "siqa": ["siqa", 0, "-"],
    "arc_challenge": ["arc_challenge", 25, '-'],
    "openbookqa": ["openbookqa", 0, '-'],
    "winogrande": ["winogrande", 5, '-'],
#
#    # misinformation, bias
    "truthfulqa": ["truthfulqa_mc2", 0, '-'],
    "crowspairs": ["crows_pairs_english_*", 0, '-'],

    # Language & overall
#    'vicuna80':["vicuna80",0, '--save_eval_examples'],
}


mmlu_subject_groups = {
    'math': [
        'hendrycksTest-abstract_algebra',
        'hendrycksTest-college_mathematics',
        'hendrycksTest-elementary_mathematics',
        'hendrycksTest-formal_logic',
        'hendrycksTest-high_school_mathematics',
        'hendrycksTest-high_school_statistics',   
    ],
    'computer_science': [
        'hendrycksTest-college_computer_science',
        'hendrycksTest-computer_security',
        'hendrycksTest-high_school_computer_science',
        'hendrycksTest-machine_learning' 
    ],
    'medical': [
        'hendrycksTest-anatomy',
        'hendrycksTest-clinical_knowledge',
        'hendrycksTest-college_biology',
        'hendrycksTest-college_medicine',
        'hendrycksTest-high_school_biology',
        'hendrycksTest-human_aging',
        'hendrycksTest-medical_genetics',
        'hendrycksTest-nutrition',
        'hendrycksTest-professional_medicine',
        'hendrycksTest-virology',
    ],
    'physics': [
        'hendrycksTest-astronomy',
        'hendrycksTest-college_physics',
        'hendrycksTest-conceptual_physics',
        'hendrycksTest-electrical_engineering',
        'hendrycksTest-high_school_physics'
    ],
    'chemistry': [
        'hendrycksTest-college_chemistry',
        'hendrycksTest-high_school_chemistry',
    ],
    'economic_business': [
        'hendrycksTest-business_ethics',
        'hendrycksTest-econometrics',
        'hendrycksTest-high_school_macroeconomics',
        'hendrycksTest-high_school_microeconomics',
        'hendrycksTest-management',
        'hendrycksTest-marketing',
        'hendrycksTest-professional_accounting'
    ],
    'history_and_general': [
        'hendrycksTest-global_facts',
        'hendrycksTest-high_school_european_history',
        'hendrycksTest-high_school_us_history',
        'hendrycksTest-high_school_world_history',
        'hendrycksTest-prehistory',
        'hendrycksTest-world_religions'
    ],
    'phsychology': [
        'hendrycksTest-high_school_psychology',
        'hendrycksTest-human_sexuality',
        'hendrycksTest-professional_psychology',
    ],
    'sociology': [
        'hendrycksTest-public_relations',
        'hendrycksTest-sociology',
    ],
    'legal_gov': [
        'hendrycksTest-high_school_government_and_politics',
        'hendrycksTest-international_law',
        'hendrycksTest-jurisprudence',
        'hendrycksTest-moral_disputes',
        'hendrycksTest-moral_scenarios',
        'hendrycksTest-professional_law',
        'hendrycksTest-us_foreign_policy'
    ],
    'other': [
        'hendrycksTest-logical_fallacies',
        'hendrycksTest-miscellaneous',
        'hendrycksTest-philosophy',
        'hendrycksTest-security_studies'
    ]
}


def get_score(fname, task):
    results = json.load(open(fname))
    if task in {'truthfulqa'}:
        print(results['results']["truthfulqa_mc2"].keys())
        return results['results']["truthfulqa_mc2"]['acc,none'] * 100
    if task in {'truthfulqa_mc_ar'}:
        return results['results'][f'{task}']['mc2'] * 100

    if task in ['mmlu','mmlu_ar', 'mmlu_hu_ar']:
        l1 = list(map(lambda x: x['acc'], results['results'].values()))
        return np.mean(l1) * 100
    if task in {'crowspairs', 'crowspairs_ar'}:
        return np.mean(list(map(lambda x: x['pct_stereotype,none'], results['results'].values()))) * 100
    if task in {'arc', 'arc_easy', 'arc_challenge', 'hellaswag', 'openbookqa', 'piqa', 'mathqa', 'siqa', \
            'exams_ar', 'digitised_ar', 'hellaswag_ar', 'piqa_ar', 'siqa_ar', 'arc_challenge_ar',\
            'openbookqa_ar'}:
        print(results['results'][task].keys())
        if 'acc_norm,none' in results['results'][task]:
            return results['results'][task]['acc_norm,none'] * 100
        else:
            return results['results'][task]['acc_norm'] * 100
    if task in ['race', 'winogrande', 'boolq', 'triviaqa', 'boolq_ar']:
        return results['results'][task]['acc,none'] * 100
    else:
        print(f"Error: Key not found: {task}")
        return None


def get_one_score_mmlu(fname, subject_group):
    result = json.load(open(fname))
    scores = []
    for subject in mmlu_subject_groups[subject_group]:
        scores.append(result['results'][subject]['acc_norm'] * 100)
    return np.mean(scores)


def extract_all_scores(output_folder, ckpt):
    log_scores = {}
    log_scores['custom_step'] = ckpt
    # for all tasks
    for task in tasks.keys():
        log_scores[task] = get_score(f"{output_folder}/{ckpt}_{task}.json", task)
    # for MMLU subjects
    for subject_group in mmlu_subject_groups:
        log_scores[subject_group] = get_one_score_mmlu(f"{output_folder}/{ckpt}_mmlu.json", subject_group)
    return log_scores
