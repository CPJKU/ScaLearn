"""Implements the adapters' configurations."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Union

import torch.nn as nn


@dataclass
class AdapterConfig(object):
    """Implements the adapter configuration proposed by Houlsby et. al, 2019
    in https://arxiv.org/abs/1902.00751."""

    add_layer_norm_before_adapter: bool = False
    add_layer_norm_after_adapter: bool = False
    non_linearity: str = "relu"
    reduction_factor: float = 16.0
    weight_init_range: float = 1e-2
    sparsity: float = 1.0
    share_adapter: bool = False
    share_encoder_decoder_single_adapter = True
    adapter_config_name: str = "Houlsby"
    mask_extreme_mode: bool = False
    mask_extreme_mode_combine_method: str = "or"
    use_multilingual: bool = False
    mt_mode: bool = False
    input_dim: Union[int, None] = None
    mask_extreme_mode: Union[bool, None] = None
    mask_extreme_mode_combine_method: Union[str, None] = None
    task_to_adapter: Union[dict, None] = None
    tasks: Union[List, None] = None


ADAPTER_CONFIG_MAPPING = OrderedDict(
    [
        ("adapter", AdapterConfig),
    ]
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
