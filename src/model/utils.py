from enum import Enum

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.adapters import AutoAdapterModel


class TaskType(Enum):
    TOKEN_CLASSIFICATION = (1,)
    SEQUENCE_CLASSIFICATION = (2,)
    QUESTION_ANSWERING = (3,)
    MULTIPLE_CHOICE = 4


AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}


def get_model(
    args,
    task_type: TaskType,
    config: AutoConfig.from_pretrained,
    fix_bert: bool = False,
):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        fusion_args,
        mtl_args,
    ) = args

    if adapter_args.train_adapter:
        # We use the AutoAdapterModel class here for better adapter support.
        model = AutoAdapterModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    bert_param = 0
    if fix_bert:
        if config.model_type == "bert":
            for param in model.bert.parameters():
                param.requires_grad = False
            for _, param in model.bert.named_parameters():
                bert_param += param.numel()
        elif config.model_type == "roberta":
            for param in model.roberta.parameters():
                param.requires_grad = False
            for _, param in model.roberta.named_parameters():
                bert_param += param.numel()
        elif config.model_type == "deberta":
            for param in model.deberta.parameters():
                param.requires_grad = False
            for _, param in model.deberta.named_parameters():
                bert_param += param.numel()
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    total_param = all_param - bert_param
    print("***** total param is {} *****".format(total_param))
    return model
