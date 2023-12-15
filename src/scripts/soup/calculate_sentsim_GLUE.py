import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import datasets

sent_model = SentenceTransformer("all-mpnet-base-v2")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}
raw_datasets = {
    dataset_name: datasets.load_dataset("glue", dataset_name)
    for dataset_name in task_to_keys.keys()
}

all_cos_sims = {}

N_SAMPLES = 10_000

for target_task in task_to_keys.keys():
    print("Target task: ", target_task)
    target_dataset = raw_datasets[target_task]

    # ensure to not select more than the number of samples
    if len(target_dataset) < N_SAMPLES:
        current_n_samples = len(target_dataset)
    sequences_test_100 = target_dataset["train"].shuffle(seed=42).select(range(current_n_samples))
    # concat key 0 and 1
    # ensure to not index if key2 is None
    if task_to_keys[target_task][1] is not None:
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

        if len(train_data) < N_SAMPLES:
            current_n_samples = len(train_data)
        sequences_train_100 = train_data.shuffle(seed=42).select(range(current_n_samples))
        # concat key 0 and 1
        # ensure to not index if key2 is None
        if task_to_keys[source_task][1] is not None:
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
    with open(f"GLUE_cos_sims_{N_SAMPLES}.json", "w") as outfile:
        json.dump(all_cos_sims, outfile)
        
    