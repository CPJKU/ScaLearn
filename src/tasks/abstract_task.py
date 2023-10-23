"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format used in JOINT MTL."""

import functools
import logging
from collections import defaultdict
from typing import Callable, Dict, List

import datasets
import evaluate
import numpy as np
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer"),
}


class TaskDataset:
    def __init__(self, name: str = None):
        self.name = name
        if self.name.lower() in GLUE_DATASETS:
            self.dataset_name = "glue"
        elif self.name.lower() in SUPERGLUE_DATASETS:
            self.dataset_name = "super_glue"
        self.metrics = evaluate.load(self.dataset_name, self.name)
        self.multiple_choice = self.name in ["copa"]
        self.is_regression = self.name == "stsb"
        self.split_to_data_split = {
            "train": "train",
            "validation": "train",
            "test": "validation",
        }

    def load_dataset(self, split: int, name: str = None):
        if name is None:
            name = self.name
        return datasets.load_dataset(self.dataset_name, name, split=split)

    def get_dataset(self, split, n_obs=None):
        print("Loading dataset", self.name, split)
        if self.name == "record":
            # take train as train, validation as validation, validation as test
            if split == "test":
                mapped_split = "validation"
            elif split == "validation":
                mapped_split = "validation"
            else:
                mapped_split = "train"
            dataset = self.load_dataset(split=mapped_split)
            if n_obs:
                dataset = dataset.select(range(n_obs))
        else:
            mapped_split = self.split_to_data_split[split]
            if "mnli" in self.name:
                if split == "test" and self.name == "mnli_mismatched":
                    mapped_split = "validation_mismatched"

                elif split == "test" and self.name == "mnli":
                    mapped_split = "validation_matched"
                dataset = self.load_dataset(split=mapped_split, name="mnli")
            else:
                dataset = self.load_dataset(split=mapped_split)
            if n_obs:
                dataset = dataset.select(range(n_obs))
            if split in ["train", "validation"]:
                if self.name in ["stsb", "record", "cb"]:
                    dataset_dict = dataset.train_test_split(
                        test_size=0.1, shuffle=True, seed=42
                    )
                else:
                    dataset_dict = dataset.train_test_split(
                        test_size=0.1, shuffle=True, seed=42, stratify_by_column="label"
                    )
                if split == "train":
                    dataset = dataset_dict["train"]
                else:
                    dataset = dataset_dict["test"]
            else:
                if n_obs:
                    dataset = dataset.select(range(n_obs))

        if self.name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif not self.multiple_choice and not self.is_regression:
            self.label_list = dataset.features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        if not self.multiple_choice and not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")
        print("Number of samples:", len(dataset))

        if self.name == "record":
            return dataset.map(
                self.record_preprocess_function,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=True,
            )
        else:
            return dataset.map(
                self.preprocessor,
                batched=True,
                remove_columns=dataset.column_names,
                load_from_cache_file=True,
            )

    def preprocessor(self, examples):
        # WSC
        if self.name == "wsc":
            examples["span2_word_text"] = []
            for text, span2_index, span2_word in zip(
                examples["text"], examples["span2_index"], examples["span2_text"]
            ):
                words_a = text.split()
                words_a[span2_index] = "*" + words_a[span2_index] + "*"
                examples["span2_word_text"].append(" ".join(words_a))

        # WiC
        if self.name == "wic":
            examples["processed_sentence1"] = []
            self.sentence2_key = "processed_sentence2"
            examples["processed_sentence2"] = []
            for sentence1, sentence2, word, start1, end1, start2, end2 in zip(
                examples["sentence1"],
                examples["sentence2"],
                examples["word"],
                examples["start1"],
                examples["end1"],
                examples["start2"],
                examples["end2"],
            ):
                examples["processed_sentence1"].append(word + ": " + sentence1)
                examples["processed_sentence2"].append(word + ": " + sentence2)

        # MultiRC
        if self.name == "multirc":
            examples["question_answer"] = []
            for question, answer in zip(examples["question"], examples["answer"]):
                examples["question_answer"].append(f"{question} {answer}")

        # COPA
        if self.name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"
                examples["text_a"].append(text_a)

        # examples to list
        # {"premise": [item1, item2, ...], "hypothesis": [item1, item2, ...]}
        # --> [{"premise": item1, "hypothesis": item1}, {"premise": item2, "hypothesis": item2}, ...]
        n_examples = len(examples["label"])
        examples = [{k: v[i] for k, v in examples.items()} for i in range(n_examples)]
        if self.name == "mnli_mismatched":
            task_list = ["mnli"] * n_examples
        else:
            task_list = [self.name] * n_examples

        return {"example": examples, "task": task_list}

    def record_preprocess_function(self, examples):
        results = {
            "index": list(),
            "question_id": list(),
            "label": list(),
            "entity": list(),
            "answers": list(),
            "question": list(),
            "passage": list(),
        }
        for idx, passage in enumerate(examples["passage"]):
            query, entities, answers = (
                examples["query"][idx],
                examples["entities"][idx],
                examples["answers"][idx],
            )
            index = examples["idx"][idx]
            passage = passage.replace("@highlight\n", "- ")

            for ent_idx, ent in enumerate(entities):
                question = query.replace("@placeholder", ent)
                label = 1 if ent in answers else 0

                results["label"].append(label)
                results["index"].append(index)
                results["question_id"].append(index["query"])
                results["entity"].append(ent)
                results["answers"].append(answers)
                results["question"].append(question)
                results["passage"].append(passage)
        n_examples = len(results["answers"])
        results = [{k: v[i] for k, v in results.items()} for i in range(n_examples)]
        task_list = [self.name] * n_examples
        return {"example": results, "task": task_list}


def build_compute_metrics_fn(
    task_names: List[str],
) -> Callable[[EvalPrediction], Dict]:
    """Builds a dictionary from each task to the task metric."""

    def compute_metrics(self, p: EvalPrediction, task: str, metrics):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if task == "stsb" else np.argmax(preds, axis=1)

        # preds = np.argmax(preds, axis=1)
        print(preds.sum())

        if task == "record":
            return record_compute_metrics(p)

        if task == "multirc":
            from sklearn.metrics import f1_score

            return {"f1": f1_score(preds, p.label_ids)}

        if task is not None:
            result = metrics.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif task == "stsb":
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def tasks_metrics(task) -> Dict:
        return functools.partial(
            compute_metrics,
            task,
            metrics=TaskDataset(task).metrics,
        )

    return {task: tasks_metrics(task) for task in task_names}


def record_compute_metrics(p: EvalPrediction):
    from tasks.superglue.utils import (
        exact_match_score,
        f1_score,
        metric_max_over_ground_truths,
    )
    from training.get_trainer_multi import AutoTask

    probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    examples = AutoTask.get("record").get_dataset(
        split="test",
    )
    examples = [example["example"] for example in examples]
    qid2pred = defaultdict(list)
    qid2ans = {}
    for prob, example in zip(probs, examples):
        qid = example["question_id"]
        qid2pred[qid].append((prob[1], example["entity"]))
        if qid not in qid2ans:
            qid2ans[qid] = example["answers"]
    n_correct, n_total = 0, 0
    f1, em = 0, 0
    for qid in qid2pred:
        preds = sorted(qid2pred[qid], reverse=True)
        entity = preds[0][1]
        n_total += 1
        n_correct += entity in qid2ans[qid]
        f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
        em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
    acc = n_correct / n_total
    f1 = f1 / n_total
    em = em / n_total
    return {"f1": f1, "exact_match": em}
