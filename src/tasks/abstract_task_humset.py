"""Implements different HUMSET tasks and defines the processors to convert each dataset
to a sequence to sequence format used int JOINT MTL"""

import functools
import logging
from typing import Callable, Dict, List

import datasets
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tasks.utils import HUMSET_DATASETS
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


class HumsetDataset:
    def __init__(self, name: str = None):
        self.name = name
        if self.name.lower() in HUMSET_DATASETS:
            self.dataset_name = "humset"
        self.multiple_choice = self.name in ["copa"]
        self.is_regression = self.name == "stsb"
        self.split_to_data_split = {
            "train": "train",
            "validation": "train",
            "test": "validation",
        }

    def load_dataset(self, split: int, name: str = None):
        return datasets.load_dataset("nlp-thedeep/humset", "1.0.0", split=split)

    def get_dataset(self, split, n_obs=None):
        print("Loading dataset", self.name, split)
        dataset = self.load_dataset(split, self.name)

        if not self.is_regression:
            self.label_list = list(
                set(label for example in dataset[self.name] for label in example)
            )
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        if not self.multiple_choice and not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")

        print("Number of samples:", len(dataset))

        processed_examples = self.preprocess_examples(dataset)
        return Dataset.from_dict(processed_examples)

    def preprocess_examples(self, examples):
        # Define new lists to hold our processed examples
        new_texts = []
        new_labels = []

        # Iterate over all examples
        for labels, text in zip(examples[self.name], examples["excerpt"]):
            # Initialize a binary label vector
            binary_label_vector = [0] * len(self.label_list)

            # For each label in the instance, set the corresponding entry in the binary label vector to 1
            for label in labels:
                binary_label_vector[self.label_list.index(label)] = 1

            new_texts.append(text)
            new_labels.append(binary_label_vector)

        return {
            "label": new_labels,
            "excerpt": new_texts,
            "lang": examples["lang"],
            "entry_id": examples["entry_id"],
            "n_tokens": examples["n_tokens"],
            "document": examples["document"],
            "task": [self.name] * len(examples),
        }


def build_compute_metrics_fn_humset(
    task_names: List[str],
) -> Callable[[EvalPrediction], Dict]:
    """Builds a dictionary from each task to the task metric."""

    def compute_metrics(self, p: EvalPrediction, task: str, metrics):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = torch.sigmoid(torch.from_numpy(preds.astype(np.float32))).numpy()

        preds = (preds > 0.5).astype(int)
        # Calculate precision, recall, f1-score (macro), and accuracy
        precision = precision_score(p.label_ids, preds, average="macro")
        recall = recall_score(p.label_ids, preds, average="macro")
        f1 = f1_score(p.label_ids, preds, average="macro")
        accuracy = accuracy_score(p.label_ids, preds)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def tasks_metrics(task) -> Dict:
        return functools.partial(compute_metrics, task, metrics=None)

    return {task: tasks_metrics(task) for task in task_names}
