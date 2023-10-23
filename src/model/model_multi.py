import functools

import torch.nn as nn
from transformers import PretrainedConfig, RobertaPreTrainedModel


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


class MultitaskModel(RobertaPreTrainedModel):
    """Used for Full FT-m"""
    def __init__(self, encoder, shared_params, taskmodels_dict):
        super().__init__(PretrainedConfig())

        self.encoder = encoder
        self.shared_params = shared_params
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict, own_params=None):
        shared_encoder = None
        taskmodels_dict = {}

        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )

            if shared_encoder is None:
                shared_encoder = model.base_model
                if own_params is not None:
                    shared_params = set(model.base_model.state_dict().keys()) - set(
                        own_params
                    )
                else:
                    shared_params = set(model.base_model.state_dict().keys())
                print(len(shared_params))
            else:
                for param_name, param in model.base_model.named_parameters():
                    if param_name in shared_params:
                        # print(param_name)
                        weights = rgetattr(shared_encoder, param_name)
                        # set the shared param to the new model's param
                        rsetattr(model.base_model, param_name, weights)

            taskmodels_dict[task_name] = model

        return cls(
            encoder=shared_encoder,
            shared_params=shared_params,
            taskmodels_dict=taskmodels_dict,
        )

    def forward(self, **kwargs):
        task = kwargs.pop("task")
        return self.taskmodels_dict[task](**kwargs)

    def print_parameter_info(self):
        print("Shared Parameters:")
        print("==================")
        shared_param_count = 0
        for param_name, param in self.encoder.named_parameters():
            if param_name in self.shared_params:
                shared_param_count += param.numel()
                trainable = param.requires_grad
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )

        print("\nNon-Shared Parameters:")
        print("======================")
        non_shared_param_count = 0
        for param_name, param in self.encoder.named_parameters():
            if param_name not in self.shared_params:
                non_shared_param_count += param.numel()
                trainable = param.requires_grad
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )
        for param_name, param in self.named_parameters():
            if param_name not in self.shared_params and "classifier" in param_name:
                non_shared_param_count += param.numel()
                trainable = param.requires_grad
                print(
                    f"Parameter: {param_name}, params: {param.numel()}, Trainable: {trainable}"
                )

        print("\nParameter Counts:")
        print("=================")
        print(f"Shared Parameters: {shared_param_count}")
        print(f"Non-Shared Parameters: {non_shared_param_count}")
        print("Total Parameters: ", shared_param_count + non_shared_param_count)
