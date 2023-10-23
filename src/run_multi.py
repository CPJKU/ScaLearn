import logging
import os
import random
import sys

import datasets
import torch
import transformers
from arguments_multi import get_args
from training.get_trainer_multi import get_trainer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "MULTI"
os.environ["WANDB_WATCH"] = "false"
os.environ["WANDB_LOG_MODEL "] = "false"
logger = logging.getLogger(__name__)


def train_fn(trainer, training_args, last_checkpoint=None):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if not training_args.no_cuda:
        torch.cuda.synchronize()  # wait for move to complete
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset)

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if not training_args.no_cuda:
        torch.cuda.synchronize()  # wait for all_reduce to complete
        end.record()
        total_time = {"total_time": start.elapsed_time(end)}
        print("###### total_time ", total_time["total_time"])
        return total_time
    return None


def evaluate_fn(trainer, data_args, training_args, eval_dataset, total_time):
    all_metrics = {}
    for eval_dataset_name, eval_dataset in eval_dataset.items():
        metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"test_{eval_dataset_name}",
        )

        max_test_samples = (
            data_args.max_test_samples
            if data_args.max_test_samples is not None
            else len(eval_dataset)
        )
        metrics[f"test_{eval_dataset_name}_samples"] = min(
            max_test_samples, len(eval_dataset)
        )
        all_metrics.update(metrics)

    if not training_args.no_cuda:
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        all_metrics["peak_memory_mb"] = peak_memory
        all_metrics["total_time_s"] = total_time["total_time"] / 1000
        all_metrics["total_time_min"] = total_time["total_time"] / 1000 / 60
        all_metrics["total_time_h"] = total_time["total_time"] / 1000 / 60 / 60
    trainer.log_metrics("test", all_metrics)
    trainer.save_metrics("test", all_metrics)


def detect_last_checkpoint(training_arguments: transformers.TrainingArguments) -> str:
    checkpoint = None
    if (
        os.path.isdir(training_arguments.output_dir)
        and training_arguments.do_train
        and not training_arguments.overwrite_output_dir
    ):
        checkpoint = get_last_checkpoint(training_arguments.output_dir)
        if checkpoint is None and len(os.listdir(training_arguments.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_arguments.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

        elif (
            checkpoint is not None and training_arguments.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return checkpoint


def setup_logging(training_args: transformers.TrainingArguments) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def main() -> None:
    args = get_args()
    (
        model_args,
        data_args,
        training_args,
        adapter_args,
        fusion_args,
        mtl_args,
    ) = args

    os.environ["WANDB_PROJECT"] = "MULTI"
    os.environ["WANDB_WATCH"] = "false"
    os.environ["WANDB_LOG_MODEL "] = "false"

    setup_logging(training_args)

    if data_args.train_tasks is None or len(data_args.train_tasks) == 0:
        data_args.train_tasks = data_args.tasks

    if not data_args.eval_adapter:
        last_checkpoint = detect_last_checkpoint(training_arguments=training_args)
    else:
        last_checkpoint = None

    set_seed(training_args.seed)

    if len(data_args.tasks) == 1:
        # convert ["['rte', 'mrpc', ...]"] to ['rte', 'mrpc', ...]
        data_args.tasks = eval(data_args.tasks[0])
    if len(data_args.eval_tasks) == 1:
        data_args.eval_tasks = eval(data_args.eval_tasks[0])
    trainer, model, train_dataset, eval_datasets, test_datasets = get_trainer(args=args)

    if training_args.do_train:
        # Log a few random samples from the training set:
        for ds in train_dataset.datasets:
            logger.info(f"Dataset: {ds[0]['task']}")
            for index in random.sample(range(len(ds)), 3):
                logger.info(f"Sample {index} of the training set: {ds[index]}.")
            logger.info("-" * 100)

        total_time = train_fn(trainer, training_args, last_checkpoint)

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        evaluate_fn(trainer, data_args, training_args, test_datasets, total_time)

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.eval_tasks is not None:
        kwargs["language"] = "en"
        kwargs["dataset_args"] = data_args.eval_tasks

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
