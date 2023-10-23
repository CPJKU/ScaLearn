import dataclasses
import logging

from adapters_hf import AutoAdapterConfig as AutoAdapterConfigHF
from adapters_propetl import AutoAdapterConfig as AutoAdapterConfigMasking
from arguments_multi import get_args
from model.model_adapter_multi import MultitaskAdapterModel
from model.model_multi import MultitaskModel
from model.utils import AUTO_MODELS, TaskType
from tasks.abstract_task import TaskDataset, build_compute_metrics_fn
from tasks.abstract_task_humset import HumsetDataset, build_compute_metrics_fn_humset
from tasks.multitask_collator import TaskCollator
from tasks.utils import GLUE_DATASETS, HUMSET_DATASETS, SUPERGLUE_DATASETS
from torch.utils.data import ConcatDataset
from torchinfo import summary
from training.multi_trainer import MultiTrainer
from training.utils import freezing_params
from transformers import AutoConfig, AutoTokenizer, EarlyStoppingCallback
from transformers.adapters import AutoAdapterModel
from transformers.models.roberta.modeling_roberta import MultiTaskRobertaModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    MultiTaskXLMRobertaModel,
)

logger = logging.getLogger(__name__)


class AutoTask:
    @classmethod
    def get(self, task_name: int):
        if task_name in GLUE_DATASETS or task_name in SUPERGLUE_DATASETS:
            return TaskDataset(task_name)
        elif task_name in HUMSET_DATASETS:
            return HumsetDataset(task_name)
        raise ValueError("Task not found")


def get_trainer(args):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        _,
        mtl_args,
    ) = get_args()
    if len(data_args.tasks) == 1:
        # convert ["['rte', 'mrpc', ...]"] to ['rte', 'mrpc', ...]
        data_args.tasks = eval(data_args.tasks[0])
    if len(data_args.eval_tasks) == 1:
        data_args.eval_tasks = eval(data_args.eval_tasks[0])
    if mtl_args.adapters is not None:
        if len(mtl_args.adapters) == 1:
            mtl_args.adapters = eval(mtl_args.adapters[0])
    if data_args.train_tasks is None or len(data_args.train_tasks) == 0:
        data_args.train_tasks = data_args.tasks

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Gets the training/test/validation datasets
    train_datasets_cls = [AutoTask.get(task) for task in data_args.train_tasks]
    train_datasets = [
        ds.get_dataset(
            split="train",
            n_obs=data_args.max_train_samples,
        )
        for ds in train_datasets_cls
    ]
    dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
    # for record only
    if data_args.downsample_last_dataset:
        dataset_sizes = dataset_sizes[:-1] + [data_args.downsample_last_dataset]
    train_dataset = ConcatDataset(train_datasets)
    training_args.remove_unused_columns = False
    eval_datasets = {
        task: AutoTask.get(task).get_dataset(
            split="validation",
            n_obs=data_args.max_eval_samples,  ## for cross lingual transfer, some task only have test set.
        )
        for task in data_args.eval_tasks
    }
    logger.warn(
        f"Eval dataset sizes: {[len(eval_dataset) for eval_dataset in eval_datasets.values()]}"
    )

    test_datasets = {
        task: AutoTask.get(task).get_dataset(
            split="test", n_obs=data_args.max_test_samples
        )
        for task in data_args.eval_tasks
    }
    # handle MNLI double eval
    if "mnli" in data_args.eval_tasks:
        test_datasets["mnli_mismatched"] = AutoTask.get("mnli_mismatched").get_dataset(
            split="test",
            n_obs=data_args.max_test_samples,
        )

    if data_args.tasks[0] in HUMSET_DATASETS:
        compute_metrics_fn = build_compute_metrics_fn_humset(data_args.eval_tasks)
    else:
        compute_metrics_fn = build_compute_metrics_fn(data_args.eval_tasks)
    logger.warn(f"Train dataset sizes: {dataset_sizes}")
    logger.warn(
        f"Eval dataset sizes: {[len(eval_dataset) for eval_dataset in eval_datasets.values()]}"
    )
    logger.warn(
        f"Test dataset sizes: {[len(test_dataset) for test_dataset in test_datasets.values()]}"
    )

    # ProPETL-m
    if mtl_args.train_multi_mask_adapter:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if data_args.tasks[0] in HUMSET_DATASETS:
            config.problem_type = "multi_label_classification"
        task_specific_config = {
            task.name: {
                "num_labels": task.num_labels,
                "id2label": task.id2label,
                "label2id": task.label2id,
            }
            if hasattr(task, "label2id")
            else {
                "num_labels": 1,
            }
            for task in train_datasets_cls
        }
        for task in train_datasets_cls:
            adapter_config = AutoAdapterConfigMasking.get("adapter")
            adapter_config.input_dim = config.hidden_size
            adapter_config.tasks = data_args.tasks
            adapter_config.task_to_adapter = (
                {
                    task: adapter
                    for task, adapter in zip(data_args.tasks, mtl_args.adapters)
                }
                if mtl_args.adapters is not None
                else None
            )
            extra_adapter_params = (
                "adapter_config_name",
                "add_layer_norm_before_adapter",
                "add_layer_norm_after_adapter",
                "reduction_factor",
                "non_linearity",
                "sparsity",
                "share_adapter",
                "share_encoder_decoder_single_adapter",
                "mask_extreme_mode",
                "mask_extreme_mode_combine_method",
                "use_multilingual",
            )

            for p in extra_adapter_params:
                if hasattr(mtl_args, p) and hasattr(adapter_config, p):
                    setattr(adapter_config, p, getattr(mtl_args, p))
                else:
                    logger.warning(
                        f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute"
                    )
            adapter_config.device = training_args.device
            assert adapter_config.adapter_config_name == "pfeiffer"

        if data_args.tasks[0] in HUMSET_DATASETS:
            model_cls = MultiTaskXLMRobertaModel
        else:
            model_cls = MultiTaskRobertaModel
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            task_specific_config=task_specific_config,
            propetl_adapter_config=dataclasses.asdict(adapter_config),
            hf_adapter_config=None,
        )
        freezing_params(model, training_args, model_args, mtl_args)

    # HYPERFORMER(++)
    elif mtl_args.train_adapters:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        if data_args.tasks[0] in HUMSET_DATASETS:
            config.problem_type = "multi_label_classification"
        task_specific_config = {
            task.name: {
                "num_labels": task.num_labels,
                "id2label": task.id2label,
                "label2id": task.label2id,
            }
            if hasattr(task, "label2id")
            else {
                "num_labels": 1,
            }
            for task in train_datasets_cls
        }
        mtl_args.adapter_config_name = "meta-adapter"
        adapter_config = AutoAdapterConfigHF.get(mtl_args.adapter_config_name)
        adapter_config.task_to_adapter = (
            {task: adapter for task, adapter in zip(data_args.tasks, mtl_args.adapters)}
            if mtl_args.adapters is not None
            else None
        )

        adapter_config.task_to_embeddings = (
            {
                task: embedding
                for task, embedding in zip(data_args.tasks, mtl_args.task_embeddings)
            }
            if (mtl_args.task_embeddings is not None)
            else None
        )
        extra_adapter_params = (
            "task_embedding_dim",
            "add_layer_norm_before_adapter",
            "add_layer_norm_after_adapter",
            "reduction_factor",
            "hidden_dim",
            "non_linearity",
            "train_task_embeddings",
            "projected_task_embedding_dim",
            "task_hidden_dim",
            "conditional_layer_norm",
            "train_adapters_blocks",
            "unique_hyper_net",  # HF
            "unique_hyper_net_layer_norm",
            "efficient_unique_hyper_net",  # HF++
            "original_layer_norm_before",
            "original_layer_norm_after",
            "hf_dropout",
            "adp_after_self",
            "input_dim",
        )
        hf_adapter_config = dataclasses.asdict(adapter_config)
        for p in extra_adapter_params:
            if hasattr(mtl_args, p) and hasattr(adapter_config, p):
                setattr(adapter_config, p, getattr(mtl_args, p))
                hf_adapter_config[p] = getattr(mtl_args, p)
            else:
                logger.warning(
                    f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute"
                )
        hf_adapter_config["device"] = training_args.device.type
        hf_adapter_config["adapter_config_name"] = "meta-adapter"
        hf_adapter_config["input_dim"] = config.hidden_size
        hf_adapter_config["tasks"] = data_args.tasks
        if data_args.tasks[0] in HUMSET_DATASETS:
            model_cls = MultiTaskXLMRobertaModel
        else:
            model_cls = MultiTaskRobertaModel
        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            task_specific_config=task_specific_config,
            hf_adapter_config=hf_adapter_config,
            propetl_adapter_config=None,
        )
        freezing_params(model, training_args, model_args, mtl_args)

    # Adapter-m and Full FT-m
    else:
        model_type_dict = {
            task.name: AUTO_MODELS[TaskType.SEQUENCE_CLASSIFICATION]
            if not task.multiple_choice
            else AUTO_MODELS[TaskType.MULTIPLE_CHOICE]
            for task in train_datasets_cls
        }
        model_config_dict = {
            task.name: AutoConfig.from_pretrained(
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path,
                num_labels=task.num_labels,
                finetuning_task=task.name,
                revision=model_args.model_revision,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            for task in train_datasets_cls
        }
        for config in model_config_dict.values():
            if data_args.tasks[0] in HUMSET_DATASETS:
                config.problem_type = "multi_label_classification"
        # Adapter-m
        if adapter_args.train_adapter:
            if data_args.tasks[0] in HUMSET_DATASETS:
                multi_label = True
            else:
                multi_label = False
            model_type_dict = {
                task.name: AutoAdapterModel for task in train_datasets_cls
            }
            model = MultitaskAdapterModel.create(
                model_name=model_args.model_name_or_path,
                model_type_dict=model_type_dict,
                model_config_dict=model_config_dict,
                dataset_cls=train_datasets_cls,
                freeze_base_model=model_args.freeze_base_model,
                separate_task_adapters=model_args.separate_task_adapters,
                adapter_config_name=mtl_args.adapter_config_name,
                multilabel=multi_label,
            )

        # Full FT-m
        else:
            model_type_dict = {
                task.name: AUTO_MODELS[TaskType.SEQUENCE_CLASSIFICATION]
                if not task.multiple_choice
                else AUTO_MODELS[TaskType.MULTIPLE_CHOICE]
                for task in train_datasets_cls
            }
            model = MultitaskModel.create(
                model_name=model_args.model_name_or_path,
                model_type_dict=model_type_dict,
                model_config_dict=model_config_dict,
            )
        model.print_parameter_info()
        # check if the embedding layer is indeed shared
        for i, task in enumerate(train_datasets_cls):
            if i == 0:
                ptr_0 = model.taskmodels_dict[
                    task.name
                ].base_model.embeddings.word_embeddings.weight.data_ptr()
            else:
                assert (
                    ptr_0
                    == model.taskmodels_dict[
                        task.name
                    ].base_model.embeddings.word_embeddings.weight.data_ptr()
                )

    # some introspection
    param_optimizer = list(model.named_parameters())
    logger.warn("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")

    if model_args.early_stopping:
        logger.info(
            "Early stopping is enabled with patience %d",
            model_args.early_stopping_patience,
        )
        early_stopping_callback = [
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience
            )
        ]
    else:
        early_stopping_callback = []

    logger.info(summary(model, depth=10))

    trainer = MultiTrainer(
        model=model,
        config=list(model_config_dict.values())[0]
        if (not mtl_args.train_multi_mask_adapter and not mtl_args.train_adapters)
        else config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=TaskCollator(
            tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores
        ),
        compute_metrics=None,
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        dataset_sizes=dataset_sizes,
        callbacks=early_stopping_callback,
    )

    return trainer, model, train_dataset, eval_datasets, test_datasets
