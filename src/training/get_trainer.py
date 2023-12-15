import json
import logging
import os
from collections import OrderedDict

from arguments import get_args
from model.utils import TaskType, get_model
from tasks.glue.dataset import GlueDataset
from tasks.humset.dataset import HumsetDataset
from tasks.superglue.dataset import SuperGlueDataset
from tasks.utils import GLUE_DATASETS, SUPERGLUE_DATASETS
from torchinfo import summary
from training.utils import get_default_args, map_omega_grid
from transformers import (
    AdapterTrainer,
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
)
from transformers.adapters.configuration import AdapterConfig, PfeifferConfig
from transformers.adapters.training import setup_adapter_training


logger = logging.getLogger(__name__)


def get_trainer(args):
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        fusion_args,
        mtl_2_args,
    ) = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if data_args.task_name.lower() in GLUE_DATASETS:
        dataset = GlueDataset(tokenizer, data_args, training_args)
    elif data_args.task_name.lower() in SUPERGLUE_DATASETS:
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)
    elif data_args.dataset_name == "humset":
        dataset = HumsetDataset(tokenizer, data_args, training_args)
    logger.info(dataset.train_dataset, dataset.eval_dataset, dataset.test_dataset)

    if not dataset.is_regression and not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.task_name,
            revision=model_args.model_revision,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    if data_args.dataset_name == "humset":
        config.problem_type = "multi_label_classification"

    # ProPETL related args
    config.share_adapter = model_args.share_adapter
    config.sparsity = model_args.sparsity

    if not dataset.multiple_choice:
        model = get_model(
            args=args, task_type=TaskType.SEQUENCE_CLASSIFICATION, config=config
        )
    else:
        model = get_model(args=args, task_type=TaskType.MULTIPLE_CHOICE, config=config)

    adapter_setup = None
    # ScaLearn + AdapterFusion
    if fusion_args.train_fusion:
        af_config = json.load(open(fusion_args.fusion_load_dir))

        # ScaLearn only: set data_args.task_name as first task
        if fusion_args.fusion_type != "dynamic":
            # as OrderedDict to preserve order.
            # then, set data_args.task_name as first task
            # used in layer.py - last adapter is needed as task adapter
            # for visualization
            af_config = OrderedDict(af_config)
            af_config.move_to_end(data_args.task_name, last=True)

        if fusion_args.fusion_adapter_config == "pfeiffer":
            adapter_config = PfeifferConfig()
        else:
            raise ValueError(
                "Only pfeiffer is currently supported for Two-Staege MTL training."
                "Set --fusion_adapter_config to pfeiffer"
            )

        for t, task in af_config.items():
            if t == "SELF":
                # replace SELF with task name
                af_config[data_args.task_name] = af_config["SELF"].replace(
                    "SELF", data_args.task_name
                )
                del af_config["SELF"]
                break

        # few-shot: change loading path
        for _, task in af_config.items():
            task = task.split("/")[-3]
            seed = af_config[task][-1]
            logger.info(task)
            # use max_train_pct for task, else 100
            if task == data_args.task_name:
                if not data_args.max_train_samples:
                    af_config[task] = (
                        af_config[task][:-1] + str(data_args.max_train_pct) + "/" + seed
                    )
                elif data_args.max_train_samples:
                    af_config[task] = (
                        af_config[task][:-1]
                        + str(data_args.max_train_pct)
                        + "/few/"
                        + str(data_args.max_train_samples)
                        + "/"
                        + seed
                    )
            else:
                af_config[task] = af_config[task][:-1] + "100" + "/" + seed

        logger.info(af_config)
        if not fusion_args.fusion_type == "soup":
            for _, adapter_dir in af_config.items():
                adapter_config.sparsity = model_args.sparsity
                logger.info(adapter_dir)
                model.load_adapter(
                    adapter_dir,
                    config=adapter_config,
                    with_head=fusion_args.fusion_with_head,
                )
            adapter_setup = [list(af_config.keys())]

        # SOUP
        if fusion_args.fusion_type == "soup":
            logger.info("*** Applying weight space averaging of the adapters ***")
            print("*** Applying weight space averaging of the adapters ***")
            state_dicts = {}
            with open(fusion_args.soup_sim_file) as f:
                all_weight_dict = json.load(f) 
            weight_dict = all_weight_dict[data_args.task_name]
            print("*** Weight vector ***")
            print(weight_dict)
            if not fusion_args.soup_include_target:
                weight_dict.pop(data_args.task_name)
            # sort and select topK most similar adapters
            weight_dict = {
                k: v
                for k, v in sorted(
                    weight_dict.items(), key=lambda item: item[1], reverse=True
                )[:fusion_args.soup_topK]
            }
            logger.info(weight_dict)
            print("*** Weight vector after topK selection ***")
            print(weight_dict)
            # normalize: sum of weights = 1
            weight_dict = {
                k: v / sum(weight_dict.values()) for k, v in weight_dict.items()
            }
            logger.info(weight_dict)
            print("*** Weight vector after normalization ***")
            print(weight_dict)
            for task, _ in weight_dict.items():
                current_adapter_dir = af_config[task]
                print(task, current_adapter_dir)
                model.load_adapter(
                    current_adapter_dir,
                    config=adapter_config,
                    load_as="adapter",
                    with_head=False,
                )
                model.set_active_adapters("adapter")
                state_dicts[task] = model.state_dict()

                # Deactivate all adapters
                model.set_active_adapters(None)
                # Delete the added adapter
                model.delete_adapter("adapter")
            
            print("*** Adapters are weighted according to the weight vector ***")
            new_state_dict = list(state_dicts.values())[0].copy()
            for key in new_state_dict:
                sum_state_dicts_of_key = 0
                if "adapter" in key:
                    print(key)
                    for i in state_dicts.keys():
                        sum_state_dicts_of_key += (state_dicts[i][key] * weight_dict[i])
                        print(i, weight_dict[i])
                    new_state_dict[key] = sum_state_dicts_of_key
            
            model.load_adapter(
                af_config[data_args.task_name],
                config=adapter_config,
                with_head=False,
                load_as="adapter",
            )
            model.load_state_dict(new_state_dict)
            model.train_adapter(
                "adapter",
            )
            model.freeze_model(True)   
                    
        # Add a fusion layer and tell the model to train a transfer layer
        elif mtl_2_args.scalearn_type:
            model.add_scalearn(adapter_setup[0], fusion_args.fusion_type)
            model.train_transfer_layer(
                adapter_setup, unfreeze_adapters=fusion_args.fusion_unfreeze_adapters
            )
        else:
            model.add_adapter_fusion(adapter_setup[0], fusion_args.fusion_type)
            model.train_transfer_layer(
                adapter_setup, unfreeze_adapters=fusion_args.fusion_unfreeze_adapters
            )

    # Single-task adapters (Pfeiffer, Compacter++, ProPETL)
    elif adapter_args.train_adapter:
        # PROBING RUNS
        # 2 omega values; summed up
        if data_args.omega_grid and mtl_2_args.scalearn_type == "omega_grid":
            omega_grid = map_omega_grid(
                config=data_args.omega_grid,
                seed=training_args.seed,
                adapter_type=adapter_args.adapter_config,
            )
            for adapter_dir, _ in omega_grid.items():
                logger.info(adapter_dir)
                model.load_adapter(
                    f"{os.path.expanduser('~')}/ScaLearn/src/" + adapter_dir,
                    with_head=False,
                )
            # get tasks from omega_grid
            source_tasks = [l.split("/")[2] for l in list(omega_grid.keys())]
            # create dict: task -> omega
            omega_grid = {l.split("/")[2]: i for l, i in list(omega_grid.items())}
            model.add_scalearn(
                source_tasks, mtl_2_args.scalearn_type, grid_values=omega_grid
            )
            model.train_transfer_layer([source_tasks], unfreeze_adapters=False)
        # 1/2 omega values
        if data_args.eval_adapter:
            config = AdapterConfig.load(
                training_args.output_dir + "/adapter_config.json"
            )
            config.omega = model_args.omega
            if data_args.train_probing_head:
                model.load_adapter(
                    training_args.output_dir,
                    config=config,
                    with_head=False,
                )
                model.train()
                if data_args.source_task:
                    model.train_adapter([data_args.source_task])
                    model.set_active_adapters(data_args.source_task)
                else:
                    model.train_adapter([data_args.task_name])
                    model.set_active_adapters(data_args.task_name)
                model.freeze_model(True)
                # add a head with corresponding name.
                if data_args.source_task:
                    head_name = data_args.source_task
                else:
                    head_name = data_args.task_name
                if dataset.multiple_choice:
                    model.add_multiple_choice_head(
                        head_name, num_choices=2, overwrite_ok=True
                    )
                else:
                    model.add_classification_head(
                        head_name,
                        num_labels=dataset.num_labels,
                        overwrite_ok=True,
                        id2label={i: v for i, v in enumerate(dataset.label_list)}
                        if not dataset.is_regression
                        else None,
                    )
            else:
                model.load_adapter(
                    training_args.output_dir,
                    config=config,
                    with_head=True,
                )
                model.train_adapter([data_args.task_name])
                model.set_active_adapters(data_args.task_name)
        # Regular training mode
        else:
            if dataset.multiple_choice:
                model.add_multiple_choice_head(data_args.task_name, num_choices=2)
            else:
                if data_args.dataset_name == "humset":
                    multi_label = True
                else:
                    multi_label = False
                model.add_classification_head(
                    data_args.task_name,
                    num_labels=dataset.num_labels,
                    id2label={i: v for i, v in enumerate(dataset.label_list)}
                    if not dataset.is_regression
                    else None,
                    layers=model_args.head_n_layers
                    if model_args.head_n_layers
                    else get_default_args(model.add_classification_head)["layers"],
                    multilabel=multi_label,
                )
            # Setup adapters
            if not data_args.omega_grid:
                setup_adapter_training(
                    model,
                    adapter_args,
                    data_args.task_name,
                    # for propetl
                    adapter_config_kwargs={
                        "sparsity": model_args.sparsity,
                        "share_adapter": model_args.share_adapter,
                    },
                )
    else:
        if adapter_args.load_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    param_optimizer = list(model.named_parameters())
    logger.info("Trainable parameters:")
    for n, p in param_optimizer:
        if p.requires_grad:
            logger.info(f"{n}")

    trainer_cls = (
        AdapterTrainer
        if (adapter_args.train_adapter or fusion_args.train_fusion)
        else Trainer
    )

    # early stopping
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

    logger.info(summary(model, depth=5))

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        callbacks=early_stopping_callback,
    )

    return trainer, model, dataset, adapter_setup
