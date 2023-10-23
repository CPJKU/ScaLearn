from dataclasses import dataclass, field
from typing import List, Optional

from adapters_propetl import ADAPTER_CONFIG_MAPPING
from transformers import HfArgumentParser, TrainingArguments
from transformers.adapters import AdapterArguments

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
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": ("passage", "question"),
    "multirc": ("paragraph", "question_answer"),
    "sectors": ("excerpt", None),
    "pillars_1d": ("excerpt", None),
    "pillars_2d": ("excerpt", None),
    "subpillars_1d": ("excerpt", None),
    "subpillars_2d": ("excerpt", None),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Task name from the list of registered tasks."},
    )
    eval_tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Evaluation task name from the list of registered tasks."},
    )
    train_tasks: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Training task name from the list of registered tasks."},
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
    max_train_pct: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "percentage of the dataset if set."
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
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "# training examples. -1 means use all."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "# validation examples. -1 means use all."}
    )
    max_test_samples: Optional[int] = field(
        default=None, metadata={"help": "# test examples. -1 means use all."}
    )

    temperature: Optional[int] = field(
        default=1,
        metadata={
            "help": "Defines the temperature"
            "value for sampling across the multiple datasets."
        },
    )

    downsample_last_dataset: Optional[int] = field(
        default=0,
        metadata={
            "help": "Whether to downsample the last dataset for adapter-based MTL."
        },
    )


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

    omega: float = field(
        default=1.0, metadata={"help": "Static value of omega to use for t-sigmoid"}
    )

    freeze_base_model: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the base model or not for adapter-based MTL."
        },
    )

    separate_task_adapters: bool = field(
        default=True,
        metadata={
            "help": "Whether to use separate task adapters for adapter-based MTL."
        },
    )
    freeze_model_but_task_embeddings: bool = field(
        default=False, metadata={"help": "freezes the whole model but task-embedding."}
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


@dataclass
class MTLArguments:
    scalearn_type: str = field(
        default=None, metadata={"help": "Type of scalearn to perform."}
    )

    train_adapters: bool = field(
        default=False, metadata={"help": "Whether to train adapters or not."}
    )

    train_multi_mask_adapter: bool = field(
        default=False, metadata={"help": "Whether to train multi-mask adapter or not."}
    )
    adapters: Optional[List[str]] = field(
        default=None, metadata={"help": "List of adapters to be used."}
    )

    adapter_config_name: Optional[str] = field(
        default="pfeiffer",
        metadata={
            "help": "config name for the adapter layers, should be selected "
            f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."
        },
    )
    add_layer_norm_before_adapter: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )
    add_layer_norm_after_adapter: Optional[bool] = field(
        default=True, metadata={"help": "whether to have layer-norm after adapter."}
    )
    reduction_factor: Optional[int] = field(
        default=16,
        metadata={
            "help": "defines the default reduction factor for " "adapter layers."
        },
    )
    non_linearity: Optional[str] = field(
        default="relu", metadata={"help": "Defines nonlinearity for adapter layers."}
    )
    sparsity: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "defines the default reduction factor for " "adapter layers."
        },
    )
    share_adapter: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )
    share_encoder_decoder_single_adapter: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )
    mask_extreme_mode: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )
    mask_extreme_mode_combine_method: Optional[str] = field(
        default="or", metadata={"help": "whether to have layer-norm before adapter."}
    )
    use_multilingual: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )

    mt_mode: Optional[bool] = field(
        default=False, metadata={"help": "whether to have layer-norm before adapter."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."},
    )
    task_embedding_dim: Optional[int] = field(
        default=512, metadata={"help": "task embedding dimensions."}
    )

    train_task_embeddings: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If specified learns the tasks "
            "embeddings from given task seedings."
        },
    )
    projected_task_embedding_dim: Optional[int] = field(
        default=64,
        metadata={
            "help": "Defines the task embedding dimension" " after projection layer. "
        },
    )
    hidden_dim: Optional[int] = field(
        default=128,
        metadata={
            "help": "defines the default hidden dimension for " "adapter layers."
        },
    )
    task_hidden_dim: Optional[int] = field(
        default=128,
        metadata={"help": "defines the hidden dimension for task embedding projector."},
    )
    conditional_layer_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Implements conditional layer norms "
            "modulated based on task embeddings."
        },
    )
    train_adapters_blocks: bool = field(
        default=False, metadata={"help": "If set, uses adapter blocks."}
    )
    unique_hyper_net: bool = field(
        default=False,
        metadata={
            "help": "If set, uses one hyper network"
            "to generates the adapter weights"
            "for all the layers."
        },
    )
    efficient_unique_hyper_net: bool = field(
        default=True,
        metadata={
            "help": "If set, uses one hyper network" "for all adapters in each layer."
        },
    )
    unique_hyper_net_layer_norm: bool = field(
        default=True,
        metadata={
            "help": "If set, applies a layer"
            "norm after computing the "
            "embeddings for the unique "
            "hyper-net."
        },
    )
    original_layer_norm_before: bool = field(
        default=True,
        metadata={
            "help": "If set, applies a layer"
            "norm before computing the adapter,"
            "Pfeiffer style"
        },
    )

    original_layer_norm_after: bool = field(
        default=True,
        metadata={
            "help": "If set, applies a layer"
            "norm after computing the adapter,"
            "instead of only summing,"
            "Pfeiffer style"
        },
    )
    hf_dropout: float = field(
        default=False,
        metadata={
            "help": "If set, applies dropout"
            "to the final generated adapter output in HF"
        },
    )

    adp_after_self: bool = field(
        default=False,
        metadata={
            "help": "If set, adds additional Adapter module after self-attention"
            "in encoder layers."
        },
    )
    input_dim: int = field(
        default=768,
        metadata={"help": "hidden dim of model, 1024 for RoBERTa-large"},
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
            MTLArguments,
        )
    )

    args = parser.parse_args_into_dataclasses()

    return args
