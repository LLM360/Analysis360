"""
TODO

"""
import re
from ..base import PerplexityTask
import pandas as pd

_CITATION = """
"""

DATA_DIR = r"/l/users/haonan.li/iiai_llm/llm-eval/datasets/ardc"

class ARDC(PerplexityTask):
    VERSION = 0
    DATASET_PATH = ""
    DATASET_NAME = ""

    def __init__(self, data_dir=None, file_name=None, cache_dir=None, download_mode=None):
        self.download(data_dir, file_name, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, file_name=None, cache_dir=None, download_mode=None):
        dataset = pd.read_pickle(f"{DATA_DIR}/test_data.pkl")
        self.dataset = dataset

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            for doc in self.dataset["text"]:
                yield doc

    def should_decontaminate(self):
        return True
