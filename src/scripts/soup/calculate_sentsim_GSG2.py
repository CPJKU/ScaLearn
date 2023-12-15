import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import datasets

sent_model = SentenceTransformer("all-mpnet-base-v2")

task_to_keys_glue = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}
task_to_keys_superglue = {
    "wic": ("sentence1", "sentence2"),
    "record": (None, None),
    "copa": (None, None),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wsc": ("text", "span1_text"),
    "multirc": ("paragraph", "question"),
}

task_to_keys = {
    "wic": ("sentence1", "sentence2"),
    "record": (None, None),
    "copa": (None, None),
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "wsc": ("text", "span1_text"),
    "multirc": ("paragraph", "question"),
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}
raw_datasets_superglue = {
    dataset_name: datasets.load_dataset("super_glue", dataset_name)
    for dataset_name in task_to_keys_superglue.keys()
}
raw_datasets_glue = {
    dataset_name: datasets.load_dataset("glue", dataset_name)
    for dataset_name in task_to_keys_glue.keys()
}
raw_datasets = {**raw_datasets_superglue, **raw_datasets_glue}


all_cos_sims = {}

N_SAMPLES = 10000

for target_task in task_to_keys.keys():
    print("Target task: ", target_task)
    target_dataset = raw_datasets[target_task]
    
    current_n_samples = N_SAMPLES
    # ensure to not select more than the number of samples
    if len(target_dataset) < N_SAMPLES:
        current_n_samples = len(target_dataset)
    sequences_test_100 = target_dataset["train"].shuffle(seed=42).select(range(current_n_samples))

    # concat key 0 and 1
    # ensure to not index if key2 is None
    if target_task == "copa":
        sequences_test_100 = [
            a + " " + b
            for a, b in zip(
                sequences_test_100["premise"],
                sequences_test_100["choice1"],
            )
        ]
    elif target_task == "record":
        sequences_test_100 = [
            a + " " + b
            for a, b in zip(
                sequences_test_100["passage"],
                sequences_test_100["query"],
            )
        ]
    elif task_to_keys[target_task][1] is not None:
        sequences_test_100 = [
            a + " " + b
            for a, b in zip(
                sequences_test_100[task_to_keys[target_task][0]],
                sequences_test_100[task_to_keys[target_task][1]],
            )
        ]
    else:
        sequences_test_100 = sequences_test_100[task_to_keys[target_task][0]]


    embeddings_test = sent_model.encode(sequences_test_100)

    cos_sims = {}
    # Load train sentences
    for source_task in task_to_keys.keys():
        # if source_task == TARGET_TASK:
        #     continue
        print(source_task)
        train_data = raw_datasets[source_task]["train"]

        current_n_samples = N_SAMPLES
        # ensure to not select more than the number of samples
        if len(train_data) < N_SAMPLES:
            current_n_samples = len(train_data)
        sequences_train_100 = train_data.shuffle(seed=42).select(range(current_n_samples))
        # concat key 0 and 1
        # ensure to not index if key2 is None
        if source_task == "copa":
            sequences_train_100 = [
                a + " " + b
                for a, b in zip(
                    sequences_train_100["premise"],
                    sequences_train_100["choice1"],
                )
            ]
        elif source_task == "record":
            sequences_train_100 = [
                a + " " + b
                for a, b in zip(
                    sequences_train_100["passage"],
                    sequences_train_100["query"],
                )
            ]
        elif task_to_keys[source_task][1] is not None:
            sequences_train_100 = [
                a + " " + b
                for a, b in zip(
                    sequences_train_100[task_to_keys[source_task][0]],
                    sequences_train_100[task_to_keys[source_task][1]],
                )
            ]
        else:
            sequences_train_100 = sequences_train_100[task_to_keys[source_task][0]]
        embeddings_train = sent_model.encode(sequences_train_100)

        all_cosine = 0
        counter = 0
        for a in embeddings_train:
            for b in embeddings_test:
                cosine = np.dot(a, b) / (norm(a) * norm(b))
                all_cosine = all_cosine + cosine
                counter += 1
        avg_cos = all_cosine / counter
        if avg_cos < 0:
            avg_cos = 0
        cos_sims[source_task] = avg_cos

    print(cos_sims)
    all_cos_sims[target_task] = cos_sims
    
    # save to json
    with open(f"GSG2_cos_sims_{N_SAMPLES}.json", "w") as outfile:
        json.dump(all_cos_sims, outfile)
        
    