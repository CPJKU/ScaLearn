"""Implements the adapters' configurations."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Union

import torch.nn as nn


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""

    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    train_adapters_blocks = True
    tasks: Optional[List] = None
    input_dim: int = 768
    task_to_embeddings: Optional[dict] = None
    task_to_adapter: Optional[dict] = None
    adapter_config_name: Optional[str] = "adapter"
    device: str = None
    original_layer_norm_before: bool = True
    original_layer_norm_after: bool = True
    hf_dropout: float = 0.0
    adp_after_self: bool = False


@dataclass
class MetaAdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""

    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = True
    non_linearity: str = "swish"
    reduction_factor: int = 16
    weight_init_range = 1e-2
    # Whether to use conditional layer norms for adapters.
    conditional_layer_norm = False
    hidden_dim: int = 128
    # Whether to add adapter blocks, this is used in case we need
    # to tune only layer norms.
    train_adapters_blocks: bool = True
    tasks: Optional[List] = None
    input_dim: int = 768
    task_to_embeddings: Optional[dict] = None
    task_to_adapter: Optional[dict] = None
    adapter_config_name: Optional[str] = "adapter"
    device: str = None
    """Implements Meta adapter in which a hyper-network generates the parameters of
     adapter layers. In this case we have a task embeddings which is feed to the
     hyper-network to allow it generate the weights for the adapter layers."""
    task_embedding_dim: int = 512
    task_embedding_dir: Union[bool, None] = None
    hidden_dim: int = 128
    train_task_embeddings: bool = False
    projected_task_embedding_dim: int = 64
    task_hidden_dim: int = 128
    parametric_task_embedding: bool = False
    # If Specified, uses one hypernet to generates the adapters weights.
    unique_hyper_net: bool = False
    unique_hyper_net_layer_norm: bool = True
    # We consider only one hyper-net for all the blocks of transformer.
    efficient_unique_hyper_net: bool = False
    adapter_config_name: Optional[str] = "meta-adapter"
    original_layer_norm_before: bool = True
    original_layer_norm_after: bool = True
    hf_dropout: float = 0.0
    adp_after_self: bool = False
    num_tasks: Optional[int] = None


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [("adapter", AdapterConfig), ("meta-adapter", MetaAdapterConfig)]
)


class AutoAdapterConfig(nn.Module):
    """Generic Adapter config class to instantiate different adapter configs."""

    @classmethod
    def get(cls, config_name: str):
        if config_name in ADAPTER_CONFIG_MAPPING:
            return ADAPTER_CONFIG_MAPPING[config_name]()
        raise ValueError(
            "Unrecognized adapter config type identifier: {}. Should contain one of {}".format(
                config_name, ", ".join(ADAPTER_CONFIG_MAPPING.keys())
            )
        )
