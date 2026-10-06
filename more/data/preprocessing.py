"""Dataset preprocessing functions for tokenization and coordinate alignment."""

from more.data.collator import SCENE_TO_IDX


def preprocess_train_val(
    examples, indices, tokenizer, max_source_length, max_target_length,
    history_column, future_column, padding,
    other_coord_system, other_coord_datasets, split_name,
):
    """Tokenize train/val examples and attach trajectory metadata."""
    inputs = examples[history_column]
    targets = examples[future_column]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )
    labels = tokenizer(
        text_target=targets, max_length=max_target_length, padding=padding, truncation=True
    )

    if padding == "max_length":
        labels["input_ids"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in labels["input_ids"]
        ]

    # Preserve trajectory arrays
    for key in ("obs_traj", "pred_traj"):
        if key in examples:
            model_inputs[key] = examples[key]

    if "original_ped_id" in examples:
        model_inputs["original_ped_id"] = examples["original_ped_id"]

    # Prefer raw 'scene' (string) over 'scene_id' which can be None in some datasets.
    # If 'scene_id' exists but its first value is None, fall back to 'scene'.
    if "scene" in examples and any(s is not None for s in examples["scene"]):
        scene_values = examples["scene"]
    elif "scene_id" in examples and any(s is not None for s in examples["scene_id"]):
        scene_values = examples["scene_id"]
    else:
        scene_values = None

    if scene_values is not None:
        model_inputs["scene_id"] = scene_values
        model_inputs["scene_idx"] = [SCENE_TO_IDX.get(s, 0) for s in scene_values]

    # Attach alternative coordinate system trajectories
    if other_coord_system and other_coord_datasets is not None:
        obs_traj_other_list = []
        pred_traj_other_list = []
        split_mapping = {"train": "train", "validation": "val", "test": "test"}
        dataset_split = split_mapping.get(split_name, split_name)
        if dataset_split in other_coord_datasets:
            for idx in indices:
                if 0 <= idx < len(other_coord_datasets[dataset_split]):
                    sample = other_coord_datasets[dataset_split][idx]
                    obs_traj_other_list.append(sample["obs_traj"])
                    pred_traj_other_list.append(sample["pred_traj"])
                else:
                    obs_traj_other_list.append(None)
                    pred_traj_other_list.append(None)
        else:
            obs_traj_other_list = [None] * len(indices)
            pred_traj_other_list = [None] * len(indices)

        model_inputs["obs_traj_other"] = obs_traj_other_list
        model_inputs["pred_traj_other"] = pred_traj_other_list

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_test(
    examples, tokenizer, max_source_length, max_target_length,
    history_column, future_column, padding,
):
    """Tokenize test examples with minimal metadata."""
    inputs = examples[history_column]
    targets = examples[future_column]
    model_inputs = tokenizer(
        inputs, max_length=max_source_length, padding=padding, truncation=True
    )
    labels = tokenizer(
        text_target=targets, max_length=max_target_length, padding=padding, truncation=True
    )

    if padding == "max_length":
        labels["input_ids"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    for key in ("obs_traj", "pred_traj"):
        if key in examples:
            model_inputs[key] = examples[key]

    if "original_ped_id" in examples:
        model_inputs["original_ped_id"] = examples["original_ped_id"]

    if "scene" in examples and any(s is not None for s in examples["scene"]):
        model_inputs["scene_id"] = examples["scene"]
    elif "scene_id" in examples and any(s is not None for s in examples["scene_id"]):
        model_inputs["scene_id"] = examples["scene_id"]

    return model_inputs
