"""
SIQA
"""
from ..base import MultipleChoiceTask
from datasets import load_dataset


_CITATION = """
@inproceedings{sap-etal-2019-social,
    title = "Social {IQ}a: Commonsense Reasoning about Social Interactions",
    author = "Sap, Maarten  and
      Rashkin, Hannah  and
      Chen, Derek  and
      Le Bras, Ronan  and
      Choi, Yejin",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-1454",
    doi = "10.18653/v1/D19-1454",
    pages = "4463--4473",
}
"""


class SiQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "social_i_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["context"] + '\n' + doc['question'],
            "choices": [doc["answerA"], doc["answerB"], doc["answerC"]],
            "gold": int(doc["label"])-1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["query"] + "\nAnswer: "

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["goal"]
