"""Expert prediction caching: save/load precomputed expert trajectories."""

import os
import pickle
import time
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


def save_expert_predictions(expert_predictions, save_path, dataset_name, phase):
    """Save expert predictions to a pickle file.

    Args:
        expert_predictions: {model_name: {ped_id: prediction_array}}
        save_path: Directory to save the file.
        dataset_name: Dataset identifier.
        phase: 'train', 'val', or 'test'.

    Returns:
        Path to the saved file.
    """
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"expert_predictions_{dataset_name}_{phase}.pkl")
    temp_path = file_path + ".tmp"

    # Convert tensors to numpy
    to_save = {}
    for model_name, pred_dict in expert_predictions.items():
        to_save[model_name] = {}
        for ped_id, pred in pred_dict.items():
            to_save[model_name][ped_id] = (
                pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
            )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(to_save, f)
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rename(temp_path, file_path)
            logger.info(f"Expert predictions saved to {file_path} "
                        f"({os.path.getsize(file_path)} bytes)")
            return file_path
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Save attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(1.0)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise RuntimeError(
                    f"Failed to save expert predictions after {max_retries} attempts: {e}"
                )


def load_expert_predictions(load_path, dataset_name, phase):
    """Load cached expert predictions from a pickle file.

    Args:
        load_path: Directory containing the pickle file.
        dataset_name: Dataset identifier.
        phase: 'train', 'val', or 'test'.

    Returns:
        {model_name: {ped_id: prediction_array}} or None if not found.
    """
    file_path = os.path.join(load_path, f"expert_predictions_{dataset_name}_{phase}.pkl")

    if not os.path.exists(file_path):
        logger.info(f"Expert predictions not found: {file_path}")
        return None

    with open(file_path, 'rb') as f:
        expert_predictions = pickle.load(f)

    # Keep (ped_id, scene) tuple keys as-is for proper lookup
    total = {k: len(v) for k, v in expert_predictions.items()}
    logger.info(f"Loaded expert predictions from {file_path}: {total}")

    # Detect key format for downstream code
    for model_name, pred_dict in expert_predictions.items():
        if pred_dict:
            first_key = next(iter(pred_dict))
            if isinstance(first_key, tuple):
                logger.info(f"  {model_name}: using (ped_id, scene) tuple keys")
            break
    return expert_predictions
