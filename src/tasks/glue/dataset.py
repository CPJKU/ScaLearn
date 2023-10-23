import logging

import evaluate
import numpy as np
from datasets.load import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


class GlueDataset:
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
        raw_datasets = load_dataset("glue", data_args.task_name)
        self.tokenizer = tokenizer
        self.data_args = data_args
        # labels
        self.is_regression = data_args.task_name == "stsb"
        self.multiple_choice = False

        if not self.is_regression:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.task_name]

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

        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]

            if data_args.max_train_pct:
                # create dict of percentiles in 5%-steps for stratification
                percentiles = {100: train_dataset}
                percentiles_idx = {}
                # for each 5 % increment, take how many samples are needed to reach that percentage
                for i in range(5, 100, 5):
                    percentiles_idx[i] = int(len(train_dataset) * i / 100)
                if data_args.task_name == "stsb":
                    # regression --> no stratification
                    percentiles[95] = train_dataset.train_test_split(
                        train_size=percentiles_idx[95], shuffle=True, seed=42
                    )["train"]
                    for i in range(90, 0, -5):
                        percentiles[i] = percentiles[i + 5].train_test_split(
                            train_size=percentiles_idx[i], shuffle=True, seed=42
                        )["train"]
                else:
                    percentiles[95] = train_dataset.train_test_split(
                        train_size=percentiles_idx[95],
                        shuffle=True,
                        seed=42,
                        stratify_by_column="label",
                    )["train"]
                    for i in range(90, 0, -5):
                        percentiles[i] = percentiles[i + 5].train_test_split(
                            train_size=percentiles_idx[i],
                            shuffle=True,
                            seed=42,
                            stratify_by_column="label",
                        )["train"]
                train_dataset = percentiles[data_args.max_train_pct]

            # select and stratify
            if data_args.task_name == "stsb":
                # regression --> no stratification
                dataset_dict = train_dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=42
                )
            else:
                dataset_dict = train_dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=42, stratify_by_column="label"
                )
            if data_args.max_train_samples:
                # check for max_train_samples > len(train_dataset)
                if data_args.max_train_samples > len(dataset_dict["train"]):
                    n_train_samples = len(dataset_dict["train"])
                else:
                    n_train_samples = data_args.max_train_samples
                dataset_dict["train"] = dataset_dict["train"].select(
                    range(n_train_samples)
                )
            self.train_dataset = dataset_dict["train"]
            logger.warn(f"train_dataset: {len(self.train_dataset)}")
            self.eval_dataset = dataset_dict["test"]

            if data_args.max_eval_samples or data_args.max_eval_pct:
                # samples
                if data_args.max_eval_samples:
                    max_eval_samples = data_args.max_eval_samples
                # pct
                elif data_args.max_eval_pct:
                    max_eval_samples = int(
                        len(self.eval_dataset) * data_args.max_eval_pct
                    )
                max_eval_samples = min(len(self.eval_dataset), max_eval_samples)
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))
                logger.warn(f"eval_dataset: {len(self.eval_dataset)}")

        if training_args.do_eval:
            if (
                "validation" not in raw_datasets
                and "validation_matched" not in raw_datasets
            ):
                raise ValueError("--do_eval requires a validation dataset")
            if data_args.task_name == "mnli":
                self.test_dataset = raw_datasets["validation_matched"]
                self.test_dataset_mm = raw_datasets["validation_mismatched"]
            else:
                self.test_dataset = raw_datasets["validation"]

            logger.warn(f"test_dataset: {len(self.test_dataset)}")

        self.metric = evaluate.load("glue", data_args.task_name)

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
        args = (
            (examples[self.sentence1_key],)
            if self.sentence2_key is None
            else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(
            *args, padding=self.padding, max_length=self.max_seq_length, truncation=True
        )

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        if self.data_args.task_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
