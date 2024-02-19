import json
import numpy as np


# Format  Task_name: [--tasks, --num_fewshot, 'task_specific_args']
tasks = {
    # knowledge
    "mmlu": ["mmlu_*", 5, '-'],
    "race": ["race", 0, '-'],

    # commonsense reasoning
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
        'mmlu_abstract_algebra',
        'mmlu_college_mathematics',
        'mmlu_elementary_mathematics',
        'mmlu_formal_logic',
        'mmlu_high_school_mathematics',
        'mmlu_high_school_statistics',   
    ],
    'computer_science': [
        'mmlu_college_computer_science',
        'mmlu_computer_security',
        'mmlu_high_school_computer_science',
        'mmlu_machine_learning' 
    ],
    'medical': [
        'mmlu_anatomy',
        'mmlu_clinical_knowledge',
        'mmlu_college_biology',
        'mmlu_college_medicine',
        'mmlu_high_school_biology',
        'mmlu_human_aging',
        'mmlu_medical_genetics',
        'mmlu_nutrition',
        'mmlu_professional_medicine',
        'mmlu_virology',
    ],
    'physics': [
        'mmlu_astronomy',
        'mmlu_college_physics',
        'mmlu_conceptual_physics',
        'mmlu_electrical_engineering',
        'mmlu_high_school_physics'
    ],
    'chemistry': [
        'mmlu_college_chemistry',
        'mmlu_high_school_chemistry',
    ],
    'economic_business': [
        'mmlu_business_ethics',
        'mmlu_econometrics',
        'mmlu_high_school_macroeconomics',
        'mmlu_high_school_microeconomics',
        'mmlu_management',
        'mmlu_marketing',
        'mmlu_professional_accounting'
    ],
    'history_and_general': [
        'mmlu_global_facts',
        'mmlu_high_school_european_history',
        'mmlu_high_school_us_history',
        'mmlu_high_school_world_history',
        'mmlu_prehistory',
        'mmlu_world_religions'
    ],
    'phsychology': [
        'mmlu_high_school_psychology',
        'mmlu_human_sexuality',
        'mmlu_professional_psychology',
    ],
    'sociology': [
        'mmlu_public_relations',
        'mmlu_sociology',
    ],
    'legal_gov': [
        'mmlu_high_school_government_and_politics',
        'mmlu_international_law',
        'mmlu_jurisprudence',
        'mmlu_moral_disputes',
        'mmlu_moral_scenarios',
        'mmlu_professional_law',
        'mmlu_us_foreign_policy'
    ],
    'other': [
        'mmlu_logical_fallacies',
        'mmlu_miscellaneous',
        'mmlu_philosophy',
        'mmlu_security_studies'
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
