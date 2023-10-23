import logging

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from datasets.load import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)

logger = logging.getLogger(__name__)


class HumsetDataset:
    def __init__(
        self,
        tokenizer: AutoTokenizer.from_pretrained,
        data_args,
        training_args,
        name=None,
    ) -> None:
        super().__init__()
        if name:
            self.name = name
            data_args.task_name = name
            label_col = name
        else:
            self.name = data_args.task_name
            label_col = data_args.task_name
        raw_datasets = load_dataset("nlp-thedeep/humset", "1.0.0")
        self.tokenizer = tokenizer
        self.data_args = data_args
        # labels
        self.is_regression = False
        self.multiple_choice = False

        if not self.is_regression:
            self.label_list = list(
                set(
                    label
                    for example in raw_datasets["train"][label_col]
                    for label in example
                )
            )
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # Apply the preprocessing function to each example in the dataset
        ds = DatasetDict()
        for split in ["train", "validation", "test"]:
            processed_examples = self.preprocess_examples(
                raw_datasets[split], label_col
            )
            ds[split] = Dataset.from_dict(processed_examples)

        self.train_dataset = ds["train"]
        self.eval_dataset = ds["validation"]
        self.test_dataset = ds["test"]
        if data_args.max_train_samples:
            max_train_samples = min(
                len(self.train_dataset), data_args.max_train_samples
            )
            self.train_dataset = self.train_dataset.select(range(max_train_samples))
        if data_args.max_eval_samples:
            max_eval_samples = min(len(self.eval_dataset), data_args.max_eval_samples)
            self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change
        # it if we already did the padding.
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=8
            )
        else:
            self.data_collator = None

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (examples["text"],)
        result = self.tokenizer(
            *args,
            padding=self.padding,
            max_length=self.max_seq_length,
            truncation=True,
        )

        return result

    def preprocess_examples(self, examples, label_col):
        # Define new lists to hold our processed examples
        new_texts = []
        new_labels = []

        # Iterate over all examples
        for labels, text in zip(examples[label_col], examples["excerpt"]):
            # Initialize a binary label vector
            binary_label_vector = [0] * len(self.label_list)

            # For each label in the instance, set the corresponding entry in the binary label vector to 1
            for label in labels:
                binary_label_vector[self.label_list.index(label)] = 1

            new_texts.append(text)
            new_labels.append(binary_label_vector)

        # Tokenize the texts and get the result
        result = self.preprocess_function({"text": new_texts})

        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
            "label": new_labels,
            "lang": examples["lang"],
            "entry_id": examples["entry_id"],
            "n_tokens": examples["n_tokens"],
            "document": examples["document"],
        }

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = (
            np.squeeze(preds)
            if self.is_regression
            else torch.sigmoid(torch.from_numpy(preds.astype(np.float32))).numpy()
        )

        preds = (preds > 0.5).astype(int)
        if self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            # Calculate precision, recall, f1-score (samples), and accuracy
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
