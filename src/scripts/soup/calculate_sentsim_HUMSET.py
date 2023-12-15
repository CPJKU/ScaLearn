import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import datasets

sent_model = SentenceTransformer("all-mpnet-base-v2")

task_to_keys = {
    "sectors": ("excerpt", None),
    "pillars_1d": ("excerpt", None),
    "pillars_2d": ("excerpt", None),
    "subpillars_1d": ("excerpt", None),
    "subpillars_2d": ("excerpt", None),
}
raw_datasets = {
    dataset_name: datasets.load_dataset("nlp-thedeep/humset", "1.0.0")
    for dataset_name in task_to_keys.keys()
}

all_cos_sims = {}

N_SAMPLES = 10000

for t, target_task in enumerate(task_to_keys.keys()):
    print("Target task: ", target_task)
    target_dataset = raw_datasets[target_task]

    # sample 100 randomly
    sequences_test_100 = target_dataset["train"].shuffle(seed=t).select(range(N_SAMPLES))
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
    for s, source_task in enumerate(task_to_keys.keys()):
        # if source_task == TARGET_TASK:
        #     continue
        print(source_task)
        train_data = raw_datasets[source_task]["train"]

        sequences_train_100 = train_data.shuffle(seed=s).select(range(N_SAMPLES))
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
    with open(f"HUMSET_cos_sims_{N_SAMPLES}.json", "w") as outfile:
        json.dump(all_cos_sims, outfile)
        
    