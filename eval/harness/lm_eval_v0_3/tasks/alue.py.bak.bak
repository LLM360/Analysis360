import numpy as np
from ..base import rf, Task
from ..metrics import mean, matthews_corrcoef, f1_score, yesno, f1_score_multiclass
from ..utils import general_detokenize, ARA_DATA_DIR
from ..tasks.utils import en2ar
from datasets import load_dataset
import os.path as osp

# Single-Sentence Tasks


class IDAT(Task):
    VERSION = 0
    DATASET_PATH = "alue"
    DATASET_NAME = "idat"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"test":osp.join(data_dir, "idat_test.csv")})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        prompt = "Is this sentence contain irony?"
        text = f"{doc['text']}\n{prompt} yes or no?\nAnswer: "
        return text

    def should_decontaminate(self):
        return False

    def doc_to_target(self, doc):
        return {1:"yes",0:"no"}[doc["label"]]

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, "yes")
        ll_false, _ = rf.loglikelihood(ctx, "no")
        return ll_true, ll_false

    def process_results(self, doc, results, save_all=False):
        ll_true, ll_false = results
        pred = ll_true > ll_false
        gold = doc["label"]
        result = {
            "acc": pred == gold,
            "f1": (gold, pred),
        }
        if save_all:
            doc['pred'] = pred
            doc['ref'] = gold
            result['example'] = doc
        return result

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class IDAT_AR(IDAT):
    DATASET_NAME = "idat_ar"

    def doc_to_text(self, doc):
        prompt = "هل هذه الجملة تحتوي على سخرية؟"
        text = f"{doc['text']}\n{prompt} {en2ar['yes or no?']}\n{en2ar['Answer']}: "
        return text

    def doc_to_target(self, doc):
        return {1:en2ar["yes"],0:en2ar["no"]}[doc["label"]]

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, en2ar["yes"])
        ll_false, _ = rf.loglikelihood(ctx, en2ar["no"])
        return ll_true, ll_false



# Inference Tasks
class XNLI(Task):
    VERSION = 0
    DATASET_PATH = "alue"
    DATASET_NAME = "xnli"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"test":osp.join(data_dir, "xnli_ar_dev.tsv")}, delimiter='\t')


    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True


    def doc_to_text(self, doc):
        return "Sentence1: {}\nSentence2: {} Question: If Sentence1 is True, then Sentence2 is (True, False or Neither)?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"]
        )

    def doc_to_target(self, doc):
        return " {}".format({"entailment": "True",
                             "neutral": "Neither",
                             "contradiction": "False"}[doc["gold_label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results, save_all=False):
        gold = {"entailment": 0,
                 "neutral": 1,
                 "contradiction": 2}[doc["gold_label"]]
        pred = np.argmax(results)
        result = {"acc": pred == gold,
                "f1": (gold, pred),
                }
        if save_all:
            doc['pred'] = pred
            doc['ref'] = gold
            result['example'] = doc
        return result

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score_multiclass}


# class XNLI_AR(XNLI):
#     DATASET_NAME = "xnli_ar"
#
#     def doc_to_text(self, doc):
#         return "Sentence1: {}\nSentence2: {} Question: If Sentence1 is True, then Sentence2 is (True, False or Neither)?\nAnswer:".format(
#             doc["sentence1"],
#             doc["sentence2"]
#         )
#
#     def doc_to_target(self, doc):
#         return " {}".format({"entailment": "True",
#                              "neutral": "Neither",
#                              "contradiction": "False"}[doc["gold_label"]])
#
#     def construct_requests(self, doc, ctx):
#         ll_true, _ = rf.loglikelihood(ctx, " True")
#         ll_neither, _ = rf.loglikelihood(ctx, " Neither")
#         ll_false, _ = rf.loglikelihood(ctx, " False")
#         return ll_true, ll_neither, ll_false


class OOLD(Task):
    VERSION = 1
    DATASET_PATH = "alue"
    DATASET_NAME = "oold"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"test":osp.join(data_dir, "oold_val.txt")}, delimiter='\t')


    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return "{}\n Is this sentence contains offensive content? yes or no?\nAnswer: ".format(
            doc["sentence"],
        )

    def doc_to_target(self, doc):
        return " {}".format({"NOT_OFF": "no", "OFF": "yes"}[doc["off_label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, "yes")
        ll_false, _ = rf.loglikelihood(ctx, "no")
        return ll_true, ll_false

    def process_results(self, doc, results, save_all=False):
        ll_true, ll_false = results
        pred = ll_true > ll_false
        gold = {"NOT_OFF": 1, "OFF": 0}[doc["off_label"]]
        result= {
            "acc": pred == gold,
            "f1": (gold, pred),
        }
        if save_all:
            doc['pred'] = pred
            doc['ref'] = gold
            result['example'] = doc
        return result

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class OOLD_AR(OOLD):
    DATASET_NAME = "oold_ar"

    def doc_to_text(self, doc):
        return f"{doc['sentence']}\n هل هذه الجملة تحتوي على محتوى مسيء؟ {en2ar['yes or no?']}\n{en2ar['Answer']}: "
    def doc_to_target(self, doc):
        return " {}".format({"NOT_OFF": en2ar["no"], "OFF": en2ar["yes"]}[doc["off_label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, en2ar["yes"])
        ll_false, _ = rf.loglikelihood(ctx, en2ar["no"])
        return ll_true, ll_false


class OHSD(OOLD):
    DATASET_NAME = "ohsd"

    def doc_to_text(self, doc):
        return "{}\n Is this sentence contains hate speech? yes or no?\nAnswer: ".format(
            doc["sentence"],
        )

    def doc_to_target(self, doc):
        return " {}".format({"NOT_HS": "no", "HS": "yes"}[doc["hs_label"]])

    def process_results(self, doc, results, save_all=False):
        ll_true, ll_false = results
        pred = ll_false > ll_true
        gold = {"NOT_HS": 1, "HS": 0}[doc["hs_label"]]
        result = {
            "acc": pred == gold,
            "f1": (gold, pred),
        }
        if save_all:
            doc['pred'] = pred
            doc['ref'] = gold
            result['example'] = doc
        return result


class OHSD_AR(OHSD):
    DATASET_NAME = "ohsd_ar"

    def doc_to_text(self, doc):
        return f"{doc['sentence']}\n هل تحتوي هذه الجملة على كلام يحض على الكراهية؟ {en2ar['yes or no?']}'\n{en2ar['Answer']}: "
    def doc_to_target(self, doc):
        return " {}".format({"NOT_HS": en2ar["no"], "HS": en2ar["yes"]}[doc["hs_label"]])


# Similarity and Paraphrase Tasks

class MQ2Q(Task):
    VERSION = 1
    DATASET_PATH = "alue"
    DATASET_NAME = "mq2q"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        self.download(ARA_DATA_DIR, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"test":osp.join(data_dir, "mq2q_train.tsv")}, delimiter='\t')

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return "Sentence 1: {}\nSentence 2: {}\nQuestion: Do both sentences mean the same thing? yes or no?\nAnswer: ".format(
            doc["question1"],
            doc["question2"],
        )

    def doc_to_target(self, doc):
        return {0:'no',1:'yes'}[doc["label"]]

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, "yes")
        ll_no, _ = rf.loglikelihood(ctx, "no")
        return ll_yes, ll_no

    def process_results(self, doc, results, save_all=False):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        result = {
            "acc": pred == gold,
            "f1": (gold, pred),
        }
        if save_all:
            doc['pred'] = pred
            doc['ref'] = gold
            result['example'] = doc
        return result

    def higher_is_better(self):
        return {"acc": True, "f1": True}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score}


class MQ2Q_AR(MQ2Q):
    DATASET_NAME = "mq2q"

    def doc_to_text(self, doc):
        return "جملة 1: {doc['question1']}\nجملة 2: {doc['question2']}\nسؤال: هل الجملتين تعنيان نفس الشيء؟ {en2ar['yes or no?']}\n{en2ar['Answer']}: "

    def doc_to_target(self, doc):
        return {0:en2ar['no'],1:en2ar['yes']}[doc["label"]]

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, en2ar["yes"])
        ll_no, _ = rf.loglikelihood(ctx, en2ar["no"])
        return ll_yes, ll_no


