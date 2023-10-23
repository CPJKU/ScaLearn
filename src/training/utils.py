import inspect

from adapters_hf import AdapterController as AdapterControllerHF
from adapters_hf import (
    AdapterLayersHyperNetController,
    AdapterLayersOneHyperNetController,
    MetaAdapterController,
)
from adapters_propetl import AdapterController as AdapterControllerMasking
from torch import nn


def map_omega_grid(
    config: dict, seed, adapter_type: str = "pfeiffer"
) -> dict[str, float]:
    """Gets all possible path combinations for a given adapter config, used in pairwise probing."""
    config = eval(config)
    if adapter_type == "pfeiffer":
        adapter_path = "st-a-3e-4"

    paths = []
    for adapter in config.keys():
        path = "runs/" + adapter_path + "/" + adapter + "/roberta-base/100/" + str(seed)
        paths.append(path)

    # create dict: {path: omega}
    paths = {path: omega for path, omega in zip(paths, config.values())}

    return paths


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freezing_params(model, training_args, model_args, mtl_args):
    """
    Freezes the model parameters based on the given setting in the arguments.
    Args:
      model: the given model.
      training_args: defines the training arguments.
      model_args: defines the model arguments.
      adapter_args: defines the adapters arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # parameters of computing task embeddings and adapter controllers.
    if mtl_args.train_adapters or mtl_args.train_multi_mask_adapter:
        freeze_params(model)
        print("Freezing all model parameters.")
        for name, sub_module in model.named_modules():
            if isinstance(
                sub_module,
                (AdapterControllerMasking, AdapterControllerHF, MetaAdapterController),
            ):
                for param_name, param in sub_module.named_parameters():
                    print("Unfreezing param: ", param_name)
                    param.requires_grad = True
        if mtl_args.train_adapters:
            if mtl_args.adapter_config_name == "meta-adapter":
                for param in model.task_embedding_controller.parameters():
                    print("Unfreezing task embedding.")
                    param.requires_grad = True
            if mtl_args.unique_hyper_net:
                for name, sub_module in model.named_modules():
                    if isinstance(
                        sub_module,
                        (AdapterLayersHyperNetController, AdapterControllerHF),
                    ):
                        for param_name, param in sub_module.named_parameters():
                            print("Unfreezing param: ", param_name)
                            param.requires_grad = True
            if mtl_args.efficient_unique_hyper_net:
                for name, sub_module in model.named_modules():
                    if isinstance(sub_module, (AdapterLayersOneHyperNetController)):
                        for param_name, param in sub_module.named_parameters():
                            print("Unfreezing param: ", param_name)
                            param.requires_grad = True

    # Unfreeze model heads
    for n, p in model.named_parameters():
        print(n)
        if "head" in n or "pooler" in n:
            print("Unfreezing param: ", n)
            p.requires_grad = True
