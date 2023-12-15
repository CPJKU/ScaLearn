from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments
from transformers.adapters import AdapterArguments

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
    "sectors": None,
    "pillars_1d": None,
    "pillars_2d": None,
    "subpillars_1d": None,
    "subpillars_2d": None,
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_seq_length_multirc: int = field(
        default=324,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    template_id: Optional[int] = field(
        default=1,
        metadata={"help": "The specific prompt string to use"},
    )
    pilot: Optional[str] = field(
        default=None, metadata={"help": "do the pilot experiments."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_train_pct: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "percentage of the dataset if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_eval_pct: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "percentage of the dataset if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    eval_adapter: Optional[str] = field(
        default=False,
        metadata={"help": "The adapter to evaluate."},
    )
    train_probing_head: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train a probing head."},
    )
    source_task: Optional[str] = field(
        default=None,
        metadata={"help": "The target task for probing."},
    )
    omega_grid: Optional[str] = field(
        default=None,
        metadata={"help": "The grid of omega values to use for probing."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )

    early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping or not."},
    )

    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "Patience for early stopping."},
    )

    omega: Optional[float] = field(
        default=None, metadata={"help": "Static value of omega to use for probing"}
    )
    sparsity: Optional[float] = field(
        default=1.0,
        metadata={"help": "sparsity of the subnet for ProPETL"},
    )
    share_adapter: bool = field(
        default=False,
        metadata={"help": "Whether share the adapter across the layers for ProPETL."},
    )
    head_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of layers in the classification head."},
    )


@dataclass
class FusionArguments:
    """
    Arguments pertaining to what data we are going to input our model Fusion

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_fusion: bool = field(
        default=False, metadata={"help": "Whether to train fusion or not."}
    )
    fusion_type: str = field(
        default="dynamic", metadata={"help": "Type of fusion to perform."}
    )
    fusion_with_head: bool = field(
        default=False, metadata={"help": "Whether to include the head in the fusion."}
    )

    fusion_adapter_config: str = field(
        default="pfeiffer",
        metadata={"help": "Type of adapter config to use for fusion."},
    )

    fusion_load_dir: str = field(
        default="scripts/st-a_fusion/af_config.json",
        metadata={"help": "Json specifying paths to adapters to be loaded fur fusion."},
    )

    fusion_unfreeze_adapters: str = field(
        default=None, metadata={"help": "Whether to unfreeze adapters."}
    )

    learn_omega: bool = field(
        default=False, metadata={"help": "Whether to learn omega or not."}
    )
    soup_topK: int = field(
        default=5, metadata={"help": "Topk to use for AdapterSoup."}
    )
    soup_include_target: bool = field(
        default=False, metadata={"help": "Whether to include target task addapter in AdapterSoup."}
    )
    soup_sim_file: str = field(
        default=None, metadata={"help": "Path to SentSim file for AdapterSoup."}
    )


@dataclass
class TwoStageMTLArguments:
    scalearn_type: str = field(
        default=None, metadata={"help": "Type of scalearn to perform."}
    )
    train_adapters: bool = field(
        default=False, metadata={"help": "Whether to train adapters or not."}
    )


def get_args():
    """Parse all the args."""
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            AdapterArguments,
            FusionArguments,
            TwoStageMTLArguments,
        )
    )

    args = parser.parse_args_into_dataclasses()

    return args
