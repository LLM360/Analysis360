from math import exp
from functools import partial
from ..base import rf, Task
from ..utils import ARA_DATA_DIR
from datasets import load_dataset, load_metric
import os.path as osp


class AraQA(Task):
    VERSION = 0
    DATASET_PATH = "araqa"

    def __init__(self, data_dir=None, file_name=None, cache_dir=None, download_mode=None):
        self.download(data_dir, file_name, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, file_name=None, cache_dir=None, download_mode=None):
        dataset = load_dataset("json", data_files={"test":osp.join(data_dir, file_name)})
        # For test
        # dataset["test"] = dataset["test"].select(range(10))
        self.dataset = dataset

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return (
            "Title: " + doc["title"] + "\n\n" +
            "Document: " + doc["context"] + "\n\n" +
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
        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        return continuation

    def process_results(self, doc, results):
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

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            )
        }

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

class AraRC(AraQA):
    DATASET_PATH = "ararc"

    def doc_to_text(self, doc):
        return (
            "Question: " + doc["question"] + "\n\n" +
            "Answer: "
        )

    def process_results(self, doc, results):
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

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            )
        }



class TydiQA_RC(AraRC):
    DATASET_NAME = "tydiqa_rc"
    def __init__(self, data_dir="tydiqa_dev.json"):
        AraRC.__init__(self, ARA_DATA_DIR, data_dir)

class MLQA_RC(AraRC):
    DATASET_NAME = "mlqa_rc"
    def __init__(self, data_dir="mlqa_dev.json"):
        AraRC.__init__(self, ARA_DATA_DIR, data_dir)

class TydiQA(AraQA):
    DATASET_NAME = "tydiqa"
    def __init__(self, data_dir="tydiqa_dev.json"):
        AraQA.__init__(self, ARA_DATA_DIR, data_dir)

class MLQA(AraQA):
    DATASET_NAME = "mlqa"
    def __init__(self, data_dir="mlqa_dev.json"):
        AraQA.__init__(self, ARA_DATA_DIR, data_dir)
