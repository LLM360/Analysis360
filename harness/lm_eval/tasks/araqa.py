from math import exp
from functools import partial
from lm_eval.base import rf, Task
from lm_eval.utils import ARA_DATA_DIR
from lm_eval.tasks.utils import en2ar
from datasets import load_dataset, load_metric
import os.path as osp


def _squad_metric(predictions, references):
    squad_metric = load_metric("squad")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references).get(key, 0)


class AraQA(Task):
    VERSION = 0
    DATASET_PATH = "araqa"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    # Not implement
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return (
            "Question: " + doc["question"] + "\n\n" +
            "Answer: "
        )

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        answer_list = doc["answers"]
        if len(answer_list) > 0:
            answer = answer_list[0]['text']
        else:
            answer = "unanswerable"
        return answer

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": None})
        print(continuation)
        return continuation

    def process_results(self, doc, results, save_all=False):
        continuation = results

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
        }

        references = {
            "id": doc["id"],
            "answers": {
                'text': list(map(lambda x: x["text"], doc["answers"])),
                'answer_start': list(map(lambda x: x["answer_start"], doc["answers"])),
                }
        }

        results = {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            )
        }
        if save_all:
            results['example'] = {
                "input":self.doc_to_text(doc),
                "id":doc["id"],
                "pred":predictions,
                "ref":references,
            }

        return results

    def higher_is_better(self):
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }

    def aggregation(self):
        return {
            "exact": partial(
                _squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
        }

class AraQA_AR(AraQA): # Change prompt lang to ar
    DATASET_PATH = "araqa_ar"
    def doc_to_text(self, doc):
        return (
            f"{en2ar['Question']}: " + doc["question"] + "\n\n" +
            f"{en2ar['Answer']}: "
        )

class AraRC(AraQA): # Change style to RC
    DATASET_PATH = "ararc"

    def doc_to_text(self, doc):
        return (
            "Title: " + doc["title"] + "\n\n" +
            "Document: " + doc["context"] + "\n\n" +
            "Question: " + doc["question"] + "\n\n" +
            "Answer: "
        )

class AraRC_AR(AraQA):
    DATASET_PATH = "ararc_ar"

    def doc_to_text(self, doc):
        return (
            f"{en2ar['Title']}: " + doc["title"] + "\n\n" +
            f"{en2ar['Document']}: " + doc["context"] + "\n\n" +
            f"{en2ar['Question']}: " + doc["question"] + "\n\n" +
            f"{en2ar['Answer']}: "
        )


class TydiQA_RC(AraRC):
    DATASET_NAME = "tydiqa_rc"

class MLQA_RC(AraRC):
    DATASET_NAME = "mlqa_rc"

class TydiQA(AraQA):
    DATASET_NAME = "tydiqa"

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset = load_dataset("json", data_files={"test":osp.join(data_dir, "tydiqa_dev.json"),
                                                   "validation":osp.join(data_dir, "tydiqa_train.json")})
        self.dataset = dataset

    def has_validation_docs(self):
        return True
    def validation_docs(self):
        return self.dataset["validation"]


class MLQA(AraQA):
    DATASET_NAME = "mlqa"

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        dataset = load_dataset("json", data_files={"test":osp.join(data_dir, "mlqa_dev.json"),
                                                   "validation":osp.join(data_dir, "mlqa_test.json")})
        self.dataset = dataset

    def has_validation_docs(self):
        return True
    def validation_docs(self):
        return self.dataset["validation"]



class TydiQA_RC_AR(AraRC_AR):
    DATASET_NAME = "tydiqa_rc_ar"

class MLQA_RC_AR(AraRC_AR):
    DATASET_NAME = "mlqa_rc_ar"

class TydiQA_AR(AraQA_AR):
    DATASET_NAME = "tydiqa_ar"

class MLQA_AR(AraQA_AR):
    DATASET_NAME = "mlqa_ar"
