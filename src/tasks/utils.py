from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
HUMSET_DATASETS = [
    "sectors",
    "pillars_1d",
    "pillars_2d",
    "subpillars_1d",
    "subpillars_2d",
]


TASKS = ["glue", "superglue", "humset"]

DATASETS = GLUE_DATASETS

ADD_PREFIX_SPACE = {
    "bert": False,
    "roberta": True,
    "deberta": True,
    "gpt2": True,
    "deberta-v2": True,
}

USE_FAST = {
    "bert": True,
    "roberta": True,
    "deberta": True,
    "gpt2": True,
    "deberta-v2": False,
}
