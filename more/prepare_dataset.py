"""Build the preprocessed prompt/answer datasets consumed by ``more.train``.

Reads the raw ETH/UCY trajectory files under ``<dataset_path>/<dataset>/<phase>/``
and writes one JSON-lines file per (dataset, phase, coordinate system) into
``<dataset_path>/preprocessed/``, using the exact filenames ``more.train`` and
``more.evaluate`` expect:

    <dataset>-train-<obs_len>-<pred_len>-<coord>-multimodal.json
    <dataset>-val-<obs_len>-<pred_len>-<coord>.json
    <dataset>-test-<obs_len>-<pred_len>-<coord>.json

Train split carries all six question types (forecast, destination, direction,
group, collision, mimicry) and is augmented; val/test carry forecast only.

Usage:
    python -m more.prepare_dataset --config_file configs/default.json
    python -m more.prepare_dataset --config_file configs/default.json \
        --dataset_name eth hotel univ zara1 zara2
"""

import os
import copy
import json
import argparse

import numpy as np
from tqdm import tqdm

from more.utils.config import get_exp_config
from more.utils.dataloader import get_dataloader, read_file
from more.utils.misc import reproducibility_settings, round_floats
from more.data.converter import batch_traj2txt, change_template
from more.data.homography import world2image, generate_homography


hist_ped_template = "Pedestrian {} moved along the trajectory {} for {} frames."
question_answer_template = "question: {} context: {} answer:"
question_template = {"forecast": "What trajectory does pedestrian {} follow for the next {} frames?",
                     "destination": "At which coordinates does pedestrian {} arrive after the next {} frames?",
                     "direction": "In which direction will pedestrian {} move in the future?",
                     "group": "With which pedestrians does pedestrian {} form a group?",
                     "collision": "With which pedestrian does pedestrian {} have a collision risk?",
                     "mimicry": "Which pedestrian seems to walk similarly to pedestrian {}?"}
answer_template = {"forecast": "Pedestrian {} will move along the trajectory {} for the next {} frames.",
                   "destination": "Pedestrian {} will arrive at coordinate {} after the next {} frames.",
                   "direction": "Pedestrian {} will {}.",
                   "group": "Pedestrian {} forms a group with pedestrian {}.",
                   "collision": "Pedestrian {} has a collision risk with pedestrian {}.",
                   "mimicry": "Pedestrian {} walks similarly to pedestrian {}.",
                   "direction_answer_list": ["move forward", "move backward", "move left", "move right", "stop"],
                   "group_false": "Pedestrian {} will walk alone.",
                   "collision_false": "Pedestrian {} has no collision risk.",
                   "mimicry_false": "Pedestrian {} will walk alone."}

ALL_MODALITIES = ["forecast", "destination", "direction", "group", "collision", "mimicry"]
ALL_AUGMENTATIONS = ["shuffle", "shift", "flip", "swap", "reverse"]
VALID_DATASETS = ["eth", "hotel", "univ", "zara1", "zara2"]
VALID_PHASES = ["train", "val", "test"]

AUGMENTATION_SHIFT = [-2, -1, 0, 1, 2]
NUMPED_THRESHOLD = 5
COORD_PRECISION = {"meter": "{:.2f}", "pixel": "{:d}"}


def get_direction_type(obs_traj, pred_traj):
    """Classify the target pedestrian's future heading into one of 5 classes."""
    DIRECTION_RADIUS_THRESHOLD = 0.2
    DIRECTION_BACKWARD_THRESHOLD = 90
    DIRECTION_FORWARD_THRESHOLD = 30
    obs_len, pred_len = obs_traj.shape[1], pred_traj.shape[1]
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)[0]
    full_traj_disp = full_traj[1:] - full_traj[:-1]
    # Filter out static people
    if np.linalg.norm(full_traj[obs_len] - full_traj[pred_len], ord=2, axis=-1) < DIRECTION_RADIUS_THRESHOLD:
        return 4
    # Normalize rotation
    dir = full_traj[obs_len] - full_traj[obs_len - 2]
    rot = np.arctan2(dir[1], dir[0])
    traj_rot = np.array([[np.cos(rot), np.sin(-rot)],
                         [np.sin(rot), np.cos(rot)]])
    full_traj_disp_norm = full_traj_disp @ traj_rot
    future_norm = full_traj_disp_norm[obs_len:].mean(axis=0)
    future_dir = np.arctan2(future_norm[1], future_norm[0]) * 180 / np.pi
    # Filter out moving backward people
    if future_dir < -DIRECTION_BACKWARD_THRESHOLD or future_dir > DIRECTION_BACKWARD_THRESHOLD:
        return 1
    # Filter out moving left people
    if future_dir > DIRECTION_FORWARD_THRESHOLD:
        return 2
    # Filter out moving right people
    if future_dir < -DIRECTION_FORWARD_THRESHOLD:
        return 3
    # Moving forward
    return 0


def get_group_id(obs_traj, pred_traj):
    """Return the index of the nearest neighbor that stays within group distance."""
    GROUP_DIST_THRESHOLD = 1
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)
    full_traj_target = full_traj[[0]]
    full_traj_neighbor = full_traj[1:]
    distance = np.linalg.norm(full_traj_neighbor - full_traj_target, ord=2, axis=-1)
    mask = np.all(distance < GROUP_DIST_THRESHOLD, axis=-1)
    nearest = distance.mean(axis=-1).argsort()
    for id in nearest:
        if mask[id]:
            return id + 1
    return None


def get_collision_id(obs_traj, pred_traj):
    """Return the index of the neighbor that comes closest during the future horizon."""
    COLLISION_DIST_THRESHOLD = 0.3
    full_traj = pred_traj
    full_traj_target = full_traj[[0]]
    full_traj_neighbor = full_traj[1:]
    distance = np.linalg.norm(full_traj_neighbor - full_traj_target, ord=2, axis=-1)
    mask = np.any(distance < COLLISION_DIST_THRESHOLD, axis=-1)
    nearest = distance.min(axis=-1).argsort()
    for id in nearest:
        if mask[id]:
            return id + 1
    return None


def get_mimicry_id(obs_traj, pred_traj):
    """Return the index of the neighbor whose displacement pattern matches the target."""
    MIMICRY_DISP_THRESHOLD = 0.1
    full_traj = np.concatenate([obs_traj, pred_traj], axis=1)
    full_traj_disp = full_traj[:, 1:] - full_traj[:, :-1]
    full_traj_disp_target = full_traj_disp[[0]]
    full_traj_disp_neighbor = full_traj_disp[1:]
    distance = np.linalg.norm(full_traj_disp_neighbor - full_traj_disp_target, ord=2, axis=-1)
    mask = distance.mean(axis=-1) < MIMICRY_DISP_THRESHOLD
    nearest = distance.mean(axis=-1).argsort()
    for id in nearest:
        if mask[id]:
            return id + 1
    return None


def load_original_txt_data(data_dir, phase, delim='\t'):
    """Load the raw ``.txt`` annotations per scene.

    Returns:
        dict: {scene_id: np.array([frame_id, ped_id, x, y])}
    """
    original_data = {}
    phase_dir = os.path.join(data_dir, phase)

    if not os.path.exists(phase_dir):
        return original_data

    all_files = sorted(os.listdir(phase_dir))
    for filename in all_files:
        if filename.endswith('.txt'):
            filepath = os.path.join(phase_dir, filename)
            # Derive the scene name by stripping the phase suffix from the filename
            scene_name, _ = os.path.splitext(filename)
            scene_name = scene_name.replace('_' + phase, '')
            original_data[scene_name] = read_file(filepath, delim)

    return original_data


def find_original_ped_ids(obs_traj, pred_traj, frame, scene_id, original_data, coord_tolerance=0.01):
    """Recover each pedestrian's ID as it appears in the raw ``.txt`` annotations.

    The dataloader renumbers pedestrians per sequence; the expert models index
    their cached predictions by the original IDs, so both are stored.

    Args:
        obs_traj: (n_ped, obs_len, 2) observed trajectories
        pred_traj: (n_ped, pred_len, 2) future trajectories
        frame: (n_ped,) start frame of each pedestrian
        scene_id: (n_ped,) scene name of each pedestrian
        original_data: dict {scene_id: np.array([frame_id, ped_id, x, y])}
        coord_tolerance: coordinate matching tolerance

    Returns:
        list: original pedestrian ID per pedestrian, or None when unmatched
    """
    original_ped_ids = []

    if scene_id[0] not in original_data:
        return [None] * len(frame)

    scene_data = original_data[scene_id[0]]

    for ped_idx in range(len(frame)):
        start_frame = frame[ped_idx]
        first_coord = obs_traj[ped_idx, 0, :]  # (2,)

        # Match on (start frame, first observed coordinate)
        frame_mask = scene_data[:, 0] == start_frame
        frame_data = scene_data[frame_mask]

        if len(frame_data) == 0:
            original_ped_ids.append(None)
            continue

        coord_diff = np.abs(frame_data[:, 2:4] - first_coord[None, :])
        coord_match = np.all(coord_diff < coord_tolerance, axis=1)

        if np.any(coord_match):
            matched_idx = np.where(coord_match)[0][0]
            original_ped_ids.append(float(frame_data[matched_idx, 1]))
        else:
            original_ped_ids.append(None)

    return original_ped_ids


def preprocess_dataset(dataset, phase, obs_len, pred_len, dataset_path="./datasets",
                       coord_system="meter", use_scene_context=True,
                       image_scale_down=0.25,
                       multimodal=ALL_MODALITIES,
                       augment=ALL_AUGMENTATIONS,
                       postfix="-multimodal"):
    """Convert one (dataset, phase) split into a JSON-lines prompt/answer file."""
    data_dir = os.path.join(dataset_path, dataset)
    out_dir = os.path.join(dataset_path, "preprocessed")
    dst_file = os.path.join(
        out_dir, f"{dataset}-{phase}-{obs_len}-{pred_len}-{coord_system}{postfix}.json")

    if phase != "train":
        augment = []

    os.makedirs(out_dir, exist_ok=True)
    reproducibility_settings()

    print(f"Loading original txt data for {dataset} {phase}...")
    original_data = load_original_txt_data(data_dir, phase)
    print(f"Loaded {len(original_data)} scene(s) from original txt files")

    dataloader = get_dataloader(data_dir, phase, obs_len, pred_len, batch_size=1)

    homography = dataloader.dataset.homography
    scene_img = dataloader.dataset.scene_img
    scene_desc = dataloader.dataset.scene_desc

    # Change coordinate precision
    if coord_system == "meter":
        change_template({"coord_template": COORD_PRECISION[coord_system]})
    elif coord_system == "pixel":
        change_template({"coord_template": COORD_PRECISION[coord_system]})
        # Scale down the scene
        for k, v in homography.items():
            homography[k] = v.copy() @ generate_homography(scale=image_scale_down)
    else:
        raise NotImplementedError(f"Unknown coord_system: {coord_system}")

    processed_data = {m: [] for m in multimodal}

    for batch in tqdm(dataloader, desc=f"Processing {dataset} {phase} dataset..."):
        obs_traj = batch['obs_traj'].numpy()
        pred_traj = batch['pred_traj'].numpy()
        non_linear_ped = batch['non_linear_ped'].numpy()
        frame = batch['frame'].numpy()
        scene_id = batch['scene_id']
        n_ped = obs_traj.shape[0]

        original_ped_ids = find_original_ped_ids(obs_traj, pred_traj, frame, scene_id, original_data)

        # Map batch-local index -> original pedestrian ID
        original_ped_id_map = {i: orig_id for i, orig_id in enumerate(original_ped_ids)}

        aug_param = []
        if "shift" in augment:
            for x_shift in AUGMENTATION_SHIFT:
                for y_shift in AUGMENTATION_SHIFT:
                    aug_param.append({"shift": np.array([x_shift, y_shift]), "flip": False, "swap": False, "reverse": False})
        else:
            aug_param.append({"shift": np.array([0, 0]), "flip": False, "swap": False, "reverse": False})

        if "flip" in augment:
            aug_param_temp = copy.deepcopy(aug_param)
            for ap in aug_param_temp:
                ap["flip"] = True
            aug_param.extend(aug_param_temp)

        if "swap" in augment:
            aug_param_temp = copy.deepcopy(aug_param)
            for ap in aug_param_temp:
                ap["swap"] = True
            aug_param.extend(aug_param_temp)

        if "reverse" in augment:
            aug_param_temp = copy.deepcopy(aug_param)
            for ap in aug_param_temp:
                ap["reverse"] = True
            aug_param.extend(aug_param_temp)

        # Generate questions and answers for each pedestrian.
        for ped_id in range(n_ped):
            obs_traj_trunc = obs_traj.copy()
            pred_traj_trunc = pred_traj.copy()

            # Drop the farthest pedestrians if the scene is crowded.
            obs_traj_target = obs_traj_trunc[[ped_id]]
            obs_nearest = np.linalg.norm(obs_traj_trunc - obs_traj_target, ord=2, axis=-1)[:, -1].argsort()
            obs_dropout = obs_nearest[1:NUMPED_THRESHOLD]
            if "shuffle" in augment:
                np.random.shuffle(obs_dropout[1:])
            obs_dropout = np.concatenate([np.array([ped_id]), obs_dropout])
            obs_traj_trunc = obs_traj_trunc[obs_dropout]
            pred_traj_trunc = pred_traj_trunc[obs_dropout]
            n_ped_temp = obs_traj_trunc.shape[0]

            # Calculate multimodal answers
            direction_type = get_direction_type(obs_traj_trunc, pred_traj_trunc) if "direction" in multimodal else None
            group_id = get_group_id(obs_traj_trunc, pred_traj_trunc) if "group" in multimodal else None
            collision_id = get_collision_id(obs_traj_trunc, pred_traj_trunc) if "collision" in multimodal else None
            mimicry_id = get_mimicry_id(obs_traj_trunc, pred_traj_trunc) if "mimicry" in multimodal else None

            # Pixel scale transformation
            if coord_system == "pixel":
                H = homography[scene_id[ped_id]]
                obs_traj_trunc = world2image(obs_traj_trunc, H).astype(np.int32)
                pred_traj_trunc = world2image(pred_traj_trunc, H).astype(np.int32)

            original_ped_id = original_ped_ids[ped_id] if ped_id < len(original_ped_ids) else None
            original_neighbor_indices = [original_ped_id_map.get(idx, None) for idx in obs_dropout]

            # Augment the trajectory (forecast only).
            for ap in aug_param:
                obs_traj_aug = obs_traj_trunc.copy()
                pred_traj_aug = pred_traj_trunc.copy()

                obs_traj_aug += ap["shift"][None, None, :]
                pred_traj_aug += ap["shift"][None, None, :]
                if ap["flip"]:
                    if coord_system == "pixel":
                        scene_img_size = (np.array(scene_img[scene_id[ped_id]].size)[None, None, :] * image_scale_down).astype(np.int32)
                    elif coord_system == "meter":
                        mid = [3.0, 5.0] if scene_id[ped_id] == "biwi_eth" else [2.5, -3.0] if scene_id[ped_id] == "biwi_hotel" else [7.5, 7.5]
                        scene_img_size = (np.array(mid)[None, None, :])
                    obs_traj_aug = -obs_traj_aug + scene_img_size
                    pred_traj_aug = -pred_traj_aug + scene_img_size
                if ap["swap"]:
                    obs_traj_aug = obs_traj_aug[:, :, [1, 0]]
                    pred_traj_aug = pred_traj_aug[:, :, [1, 0]]
                if ap["reverse"]:
                    full_traj_aug = np.concatenate([obs_traj_aug, pred_traj_aug], axis=1)[:, ::-1]
                    obs_traj_aug = full_traj_aug[:, :obs_len]
                    pred_traj_aug = full_traj_aug[:, obs_len:]

                # Prompt generation
                scene_context = scene_desc[scene_id[0]]
                obs_traj_text_list = batch_traj2txt(obs_traj_aug)
                pred_traj_text_list = batch_traj2txt(pred_traj_aug)
                traj_context = " ".join([hist_ped_template.format(i, obs_traj_text_list[i], obs_len) for i in range(n_ped_temp)])
                if use_scene_context:
                    context = traj_context + " " + scene_context
                else:
                    context = traj_context

                if "forecast" in multimodal:
                    question = question_template["forecast"].format(0, pred_len)
                    obs_prompt = question_answer_template.format(question, context)
                    pred_prompt = answer_template["forecast"].format(0, pred_traj_text_list[0], pred_len)

                    processed_data["forecast"].append({"id": len(processed_data["forecast"]),
                                                       "type": "forecast",
                                                       "scene": scene_id[ped_id],
                                                       "frame": int(frame[ped_id]),
                                                       "n_ped": n_ped,
                                                       "ped_id": ped_id,  # batch-local index
                                                       "original_ped_id": original_ped_id,  # ID in the raw txt file
                                                       "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                                       "obs_traj": obs_traj_aug[0].tolist(),
                                                       "pred_traj": pred_traj_aug[0].tolist(),
                                                       "observation": obs_prompt,
                                                       "forecast": pred_prompt,
                                                       "neighbor_indices": obs_dropout.tolist(),
                                                       "original_neighbor_indices": original_neighbor_indices})

            # Non-forecast modalities use the unaugmented trajectory.
            scene_context = scene_desc[scene_id[0]]
            obs_traj_text_list = batch_traj2txt(obs_traj_trunc)
            traj_context = " ".join([hist_ped_template.format(i, obs_traj_text_list[i], obs_len) for i in range(n_ped_temp)])
            context = scene_context + " " + traj_context

            def _record(bucket, prompt_q, prompt_a):
                processed_data[bucket].append({"id": len(processed_data[bucket]),
                                               "type": bucket,
                                               "scene": scene_id[ped_id],
                                               "frame": int(frame[ped_id]),
                                               "n_ped": n_ped,
                                               "ped_id": ped_id,
                                               "original_ped_id": original_ped_id,
                                               "non_linear_ped": non_linear_ped[ped_id].tolist(),
                                               "obs_traj": obs_traj_trunc[0].tolist(),
                                               "pred_traj": pred_traj_trunc[0].tolist(),
                                               "observation": prompt_q,
                                               "forecast": prompt_a,
                                               "neighbor_indices": obs_dropout.tolist(),
                                               "original_neighbor_indices": original_neighbor_indices})

            if "destination" in multimodal:
                question = question_template["destination"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                dest_coord = "(" + ", ".join(COORD_PRECISION[coord_system].format(pred_traj_trunc[0, -1, j]) for j in range(2)) + ")"
                _record("destination", obs_prompt,
                        answer_template["destination"].format(0, dest_coord, pred_len))

            if "direction" in multimodal:
                question = question_template["direction"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                direction = answer_template["direction_answer_list"][direction_type]
                _record("direction", obs_prompt, answer_template["direction"].format(0, direction))

            if "group" in multimodal:
                question = question_template["group"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                answer = (answer_template["group"].format(0, group_id) if group_id is not None
                          else answer_template["group_false"].format(0))
                _record("group", obs_prompt, answer)

            if "collision" in multimodal:
                question = question_template["collision"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                answer = (answer_template["collision"].format(0, collision_id) if collision_id is not None
                          else answer_template["collision_false"].format(0))
                _record("collision", obs_prompt, answer)

            if "mimicry" in multimodal:
                question = question_template["mimicry"].format(0, pred_len)
                obs_prompt = question_answer_template.format(question, context)
                answer = (answer_template["mimicry"].format(0, mimicry_id) if mimicry_id is not None
                          else answer_template["mimicry_false"].format(0))
                _record("mimicry", obs_prompt, answer)

    processed_data_cat = []
    for m in multimodal:
        processed_data_cat += processed_data[m]

    with open(dst_file, encoding="utf-8", mode="w") as file:
        for i in tqdm(processed_data_cat, desc=f"Writing {dataset} {phase} dataset..."):
            file.write(json.dumps(round_floats(i)) + "\n")

    print(f"Wrote {len(processed_data_cat)} samples -> {dst_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build preprocessed prompt/answer datasets for MoRE training")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset_name", nargs="+", default=None,
                        help=f"One or more of {VALID_DATASETS}. Defaults to the config value.")
    parser.add_argument("--phase", nargs="+", default=None,
                        help=f"One or more of {VALID_PHASES}. Defaults to all three.")
    parser.add_argument("--coord_system", nargs="+", default=None,
                        help="One or more of ['meter', 'pixel']. Defaults to the config's "
                             "metric plus other_model_coordinate when they differ.")
    args = parser.parse_args()

    cfg = get_exp_config(args.config_file)

    datasets = args.dataset_name or [cfg.dataset_name]
    phases = [p if p != "valid" else "val" for p in (args.phase or VALID_PHASES)]

    primary_coord = cfg.metric
    if args.coord_system:
        coord_systems = args.coord_system
    else:
        coord_systems = [primary_coord]
        other = getattr(cfg, "other_model_coordinate", None)
        if other and other != primary_coord:
            coord_systems.append(other)

    for d in datasets:
        assert d in VALID_DATASETS, f"Invalid dataset: {d}. Must be one of {VALID_DATASETS}"
    for p in phases:
        assert p in VALID_PHASES, f"Invalid phase: {p}. Must be one of {VALID_PHASES}"
    for c in coord_systems:
        assert c in COORD_PRECISION, f"Invalid coord_system: {c}. Must be one of {list(COORD_PRECISION)}"

    use_scene_context = getattr(cfg, "use_scene_context", True)
    image_scale_down = getattr(cfg, "image_scale_down", 0.25)

    for coord in coord_systems:
        # more.train expects "-multimodal-nocontext" only for the primary coordinate
        # system when scene context is disabled; the auxiliary one is always plain.
        if coord == primary_coord:
            train_postfix = "-multimodal" if use_scene_context else "-multimodal-nocontext"
        else:
            train_postfix = "-multimodal"

        for d in datasets:
            for p in phases:
                print(f"\n=== {d} / {p} / {coord} ===")
                if p == "train":
                    preprocess_dataset(d, p, cfg.obs_len, cfg.pred_len,
                                       dataset_path=cfg.dataset_path,
                                       coord_system=coord,
                                       use_scene_context=use_scene_context,
                                       image_scale_down=image_scale_down,
                                       postfix=train_postfix)
                else:
                    preprocess_dataset(d, p, cfg.obs_len, cfg.pred_len,
                                       dataset_path=cfg.dataset_path,
                                       coord_system=coord,
                                       use_scene_context=use_scene_context,
                                       image_scale_down=image_scale_down,
                                       multimodal=["forecast"],
                                       augment=[],
                                       postfix="")


if __name__ == "__main__":
    main()
