"""Custom data collator for trajectory prediction with metadata fields."""

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq


# Scene name <-> index mapping
SCENE_TO_IDX = {
    'biwi_hotel': 1, 'crowds_zara01': 2, 'crowds_zara02': 3,
    'crowds_zara03': 4, 'students001': 5, 'students003': 6,
    'uni_examples': 7, 'biwi_eth': 8,
}
IDX_TO_SCENE = {v: k for k, v in SCENE_TO_IDX.items()}


class TrajectoryDataCollator(DataCollatorForSeq2Seq):
    """DataCollator that preserves trajectory and metadata fields alongside token tensors."""


    def __call__(self, features, return_tensors=None):
        # Extract custom trajectory fields
        custom_fields = {}
        for key in ("obs_traj", "pred_traj", "obs_traj_other", "pred_traj_other"):
            if key in features[0]:
                custom_fields[key] = [f[key] for f in features]

        # Extract metadata fields
        metadata_fields = {}
        for key in ("original_ped_id", "scene_id", "scene", "scene_idx"):
            if key in features[0]:
                metadata_fields[key] = [f[key] for f in features]

        # accelerate.prepare(dataloader) silently drops list-of-string fields
        # (only tensor-like fields survive distribution). Keep scene_idx as int
        # tensor — train.py recovers the scene string via IDX_TO_SCENE.
        if "scene_idx" not in metadata_fields:
            if "scene" in metadata_fields:
                metadata_fields["scene_idx"] = [
                    SCENE_TO_IDX.get(s, 0) for s in metadata_fields["scene"]
                ]
            elif "scene_id" in metadata_fields:
                metadata_fields["scene_idx"] = [
                    SCENE_TO_IDX.get(s, 0) for s in metadata_fields["scene_id"]
                ]
        # Drop string-only fields that won't survive accelerate.prepare
        metadata_fields.pop("scene", None)
        metadata_fields.pop("scene_id", None)

        # Strip non-standard keys before calling parent collator
        features_for_super = [
            {k: v for k, v in f.items()
             if k not in ("original_ped_id", "scene_id", "scene", "scene_idx",
                          "obs_traj", "pred_traj", "obs_traj_other", "pred_traj_other")}
            for f in features
        ]

        batch = super().__call__(features_for_super, return_tensors=return_tensors)

        # Re-attach custom trajectory fields
        for key, values in custom_fields.items():
            if values and values[0] is not None:
                try:
                    if isinstance(values[0], torch.Tensor):
                        batch[key] = torch.stack(values) if values[0].dim() > 0 else torch.tensor(values)
                    elif isinstance(values[0], list):
                        batch[key] = torch.tensor(np.array(values))
                    else:
                        batch[key] = torch.tensor(values)
                except Exception:
                    batch[key] = values
            else:
                batch[key] = None

        # Re-attach metadata fields. Only tensor-like fields survive accelerate's
        # distributed dataloader; string lists are silently dropped, so anything
        # we want available downstream must be encoded as an int tensor here.
        for key, values in metadata_fields.items():
            if not values:
                batch[key] = None
                continue
            if key == "scene_idx":
                batch[key] = torch.tensor([int(v) for v in values], dtype=torch.long)
            elif isinstance(values[0], (int, float, np.number)):
                batch[key] = torch.tensor(values, dtype=torch.float32)
            elif isinstance(values[0], str):
                batch[key] = values  # may be dropped by accelerate
            else:
                batch[key] = values

        return batch
