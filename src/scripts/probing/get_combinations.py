datasets = [
    "QQP",
    "MNLI",
]  # Add more dataset names as needed
values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]


def generate_value_combinations(dataset_names, values):
    if not dataset_names:
        return [{}]

    current_dataset = dataset_names[0]
    remaining_datasets = dataset_names[1:]
    combinations = []

    for value in values:
        remaining_combinations = generate_value_combinations(remaining_datasets, values)
        for combination in remaining_combinations:
            combination[current_dataset] = value
        combinations.extend(remaining_combinations)

    return combinations
