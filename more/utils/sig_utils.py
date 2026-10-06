import os
import math
import torch
import numpy as np
from more.data.homography import image2world

def read_file(_path, delim='\t'):
    """Read trajectory data file"""
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

# Global cache for trajectories: (data_dir, phase, obs_len, pred_len, skip) -> {ped_id: {'obs_traj': ..., 'pred_traj': ..., 'scene_id': ...}}
_ped_trajectories_full_cache = {}

# Global cache for SingularTrajectory predictions: {cache_key: {ped_id: pred_traj}}
# cache_key = (data_dir, phase, obs_len, pred_len, num_samples)
_EXPERT_SIG_PREDICTIONS_CACHE = {}

def parse_sequences_from_ped_ids(original_ped_ids, data_dir, obs_len=8, pred_len=12, 
                                  skip=1, delim='\t'):
    """
    Parse sequences from ped_ids by finding them in original data files
    
    Args:
        original_ped_ids: (num_ped,) array of ped IDs
        data_dir: directory containing dataset files
        obs_len: observation length
        pred_len: prediction length
        skip: skip frames
        delim: delimiter
        
    Returns:
        seq_start_end: list of [start, end] indices for each sequence
        ped_id_to_idx: dict mapping ped_id to index in the final array
    """
    seq_len = obs_len + pred_len
    target_ped_ids = set(original_ped_ids.astype(float))
    
    # Get only files (not directories) from data_dir
    all_items = sorted(os.listdir(data_dir))
    all_files = []
    for item in all_items:
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            all_files.append(item_path)
    
    sequences = []  # List of sequences, each containing list of ped_ids
    ped_id_to_info = {}  # ped_id -> (file_idx, frame_range, sequence_idx)
    
    for file_idx, path in enumerate(all_files):
        data = read_file(path, delim)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
        
        for seq_idx in range(0, num_sequences * skip + 1, skip):
            if seq_idx + seq_len > len(frames):
                break
            
            curr_seq_data = np.concatenate(frame_data[seq_idx:seq_idx + seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            peds_in_curr_seq_set = set(peds_in_curr_seq.astype(float))
            
            # Check if any target ped_id is in this sequence
            if not (target_ped_ids & peds_in_curr_seq_set):
                continue
            
            # Find target ped_ids in this sequence
            target_peds_in_seq = list(target_ped_ids & peds_in_curr_seq_set)
            
            if len(target_peds_in_seq) > 0:
                sequences.append({
                    'ped_ids': target_peds_in_seq,
                    'file_idx': file_idx,
                    'frame_start': frames[seq_idx],
                    'frame_end': frames[seq_idx + seq_len - 1]
                })
                
                for ped_id in target_peds_in_seq:
                    if ped_id not in ped_id_to_info:
                        ped_id_to_info[ped_id] = []
                    ped_id_to_info[ped_id].append(len(sequences) - 1)
    
    # Group ped_ids by sequence (same frame range)
    sequence_groups = []
    used_ped_ids = set()
    
    for ped_id in original_ped_ids:
        if float(ped_id) in used_ped_ids:
            continue
        
        if float(ped_id) not in ped_id_to_info:
            # If ped_id not found, create a single-ped sequence
            sequence_groups.append([float(ped_id)])
            used_ped_ids.add(float(ped_id))
            continue
        
        # Find all ped_ids in the same sequence(s)
        seq_indices = ped_id_to_info[float(ped_id)]
        if len(seq_indices) == 0:
            continue
        
        # Use the first sequence (could be improved to handle multiple sequences)
        seq_idx = seq_indices[0]
        seq_ped_ids = sequences[seq_idx]['ped_ids']
        
        # Add all ped_ids from this sequence that are in original_ped_ids
        group = [pid for pid in seq_ped_ids if pid in target_ped_ids and pid not in used_ped_ids]
        if len(group) > 0:
            sequence_groups.append(group)
            used_ped_ids.update(group)
    
    # Create seq_start_end and ped_id_to_idx mapping
    seq_start_end = []
    ped_id_to_idx = {}
    current_idx = 0
    
    for group in sequence_groups:
        start_idx = current_idx
        for ped_id in group:
            ped_id_to_idx[ped_id] = current_idx
            current_idx += 1
        end_idx = current_idx
        seq_start_end.append([start_idx, end_idx])
    
    # Handle any remaining ped_ids not found in sequences
    for ped_id in original_ped_ids:
        if float(ped_id) not in ped_id_to_idx:
            ped_id_to_idx[float(ped_id)] = current_idx
            seq_start_end.append([current_idx, current_idx + 1])
            current_idx += 1
    
    return seq_start_end, ped_id_to_idx


class BatchDatasetWrapper:
    """Wrapper class to make batch data look like a dataset for anchor calculation"""
    def __init__(self, obs_traj, pred_traj, scene_id, vector_field=None, homography=None):
        """
        Args:
            obs_traj: (num_ped, obs_len, 2) torch.Tensor
            pred_traj: (num_ped, pred_len, 2) torch.Tensor  
            scene_id: (num_ped,) list or np.array of scene names/ids
            vector_field: dict {scene_name: vector_field_array}, optional
            homography: dict {scene_name: homography_matrix}, optional
        """
        self.obs_traj = obs_traj
        self.pred_traj = pred_traj
        self.scene_id = np.array(scene_id) if isinstance(scene_id, list) else scene_id
        self.vector_field = vector_field if vector_field is not None else {}
        self.homography = homography if homography is not None else {}
        self.anchor = None


def load_trajectories_from_ped_ids(original_ped_ids, data_dir, phase, obs_len=8, pred_len=12,
                                    skip=1, delim='\t', original_frames=None):
    """
    Load trajectories from original data files using ped_ids

    Args:
        original_ped_ids: (num_ped,) array of ped IDs
        data_dir: directory containing dataset files
        obs_len: observation length
        pred_len: prediction length
        skip: skip frames
        delim: delimiter
        original_frames: (num_ped,) array of start frames for (ped_id, frame) matching

    Returns:
        obs_traj: (num_ped, obs_len, 2) numpy array in meter coordinates
        pred_traj: (num_ped, pred_len, 2) numpy array in meter coordinates
        scene_id: (num_ped,) array of scene names
        seq_start_end: list of [start, end] indices for each sequence
    """
    # Check cache first
    cache_key = (data_dir, phase, obs_len, pred_len, skip)
    if cache_key in _ped_trajectories_full_cache:
        # Cache HIT: use cached trajectories (keyed by (ped_id, frame) when frames available)
        ped_trajectories_all = _ped_trajectories_full_cache[cache_key]
        ped_to_seq_info_all = _ped_trajectories_full_cache[cache_key + ('seq_info',)]

        # Extract requested ped_ids from cache
        obs_traj_list = []
        pred_traj_list = []
        scene_id_list = []

        for i, ped_id in enumerate(original_ped_ids):
            ped_id_float = float(ped_id)
            # Try (ped_id, frame) key first, fall back to ped_id only, then try any (ped_id, frame) key
            if original_frames is not None:
                traj_key = (ped_id_float, float(original_frames[i]))
            else:
                traj_key = ped_id_float
            
            # Try to find trajectory: first try exact key, then try ped_id-only, then try any (ped_id, frame) key
            found_traj = None
            if traj_key in ped_trajectories_all:
                found_traj = ped_trajectories_all[traj_key]
            elif ped_id_float in ped_trajectories_all:
                found_traj = ped_trajectories_all[ped_id_float]
            else:
                # Try to find any (ped_id, frame) key for this ped_id
                for key in ped_trajectories_all.keys():
                    if isinstance(key, tuple) and len(key) > 0 and key[0] == ped_id_float:
                        found_traj = ped_trajectories_all[key]
                        break
            
            if found_traj is None:
                raise KeyError(f"ped_id {ped_id_float} (frame={original_frames[i] if original_frames is not None else 'N/A'}) not found in cached trajectories")
            
            obs_traj_list.append(found_traj['obs_traj'])
            pred_traj_list.append(found_traj['pred_traj'])
            scene_id_list.append(found_traj['scene_id'])
        
        obs_traj = np.stack(obs_traj_list, axis=0)
        pred_traj = np.stack(pred_traj_list, axis=0)
        scene_id = np.array(scene_id_list)
        
        # Build seq_start_end from cached seq_info
        num_ped = len(original_ped_ids)
        seq_start_end = []
        used_ped_indices = set()

        for i, ped_id in enumerate(original_ped_ids):
            if i in used_ped_indices:
                continue

            ped_id_float = float(ped_id)
            # Use (ped_id, frame) key for seq_info lookup
            if original_frames is not None:
                seq_key = (ped_id_float, float(original_frames[i]))
            else:
                seq_key = ped_id_float
            if seq_key not in ped_to_seq_info_all:
                seq_key = ped_id_float  # fallback
            if seq_key not in ped_to_seq_info_all:
                seq_start_end.append([i, i + 1])
                used_ped_indices.add(i)
                continue

            seq_info = ped_to_seq_info_all[seq_key]
            start_idx = i
            end_idx = i + 1

            for j in range(i + 1, num_ped):
                if j in used_ped_indices:
                    continue
                other_ped_id = float(original_ped_ids[j])
                if original_frames is not None:
                    other_key = (other_ped_id, float(original_frames[j]))
                else:
                    other_key = other_ped_id
                if other_key not in ped_to_seq_info_all:
                    other_key = other_ped_id
                if other_key in ped_to_seq_info_all:
                    other_seq_info = ped_to_seq_info_all[other_key]
                    if (other_seq_info['file_idx'] == seq_info['file_idx'] and
                        other_seq_info['frame_start'] == seq_info['frame_start'] and
                        other_seq_info['frame_end'] == seq_info['frame_end']):
                        end_idx = j + 1
                        used_ped_indices.add(j)

            seq_start_end.append([start_idx, end_idx])
            used_ped_indices.add(i)

        return obs_traj, pred_traj, scene_id, seq_start_end
    
    # Cache MISS: load all trajectories and cache them
    print(f"[CACHE MISS] Loading ALL trajectories from {data_dir} (phase={phase})... This may take a while...")
    
    seq_len = obs_len + pred_len
    target_ped_ids = set(original_ped_ids.astype(float))
    
    # Get only files (not directories) from data_dir
    all_items = sorted(os.listdir(data_dir))
    all_files = []
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            all_files.append(item_path)
    
    # Dictionary to store trajectories for each ped_id (load ALL ped_ids, not just target ones)
    ped_trajectories = {}  # ped_id -> {'obs_traj': ..., 'pred_traj': ..., 'scene_id': ...}
    ped_to_seq_info = {}  # ped_id -> sequence info (for seq_start_end)
    
    for file_idx, path in enumerate(all_files):
        if file_idx % 10 == 0 and file_idx > 0:
            print(f"[CACHE LOAD] Processing file {file_idx}/{len(all_files)}: {os.path.basename(path)}")
        # Extract scene name from path
        parent_dir, scene_name = os.path.split(path)
        parent_dir, phase = os.path.split(parent_dir)
        parent_dir, dataset_name = os.path.split(parent_dir)
        scene_name, _ = os.path.splitext(scene_name)
        scene_name = scene_name.replace('_' + phase, '')
        
        # Load data
        data = read_file(path, delim)
        if len(data) == 0:
            continue
            
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        
        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))
        
        for seq_idx in range(0, num_sequences * skip + 1, skip):
            if seq_idx + seq_len > len(frames):
                break
            
            curr_seq_data = np.concatenate(frame_data[seq_idx:seq_idx + seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            
            # Load ALL ped_ids in this sequence — use (ped_id, frame) key to avoid overwrites
            start_frame = float(frames[seq_idx])
            for ped_id in peds_in_curr_seq:
                ped_id_float = float(ped_id)
                traj_key = (ped_id_float, start_frame)
                if traj_key in ped_trajectories:
                    continue  # Already loaded for this (ped_id, frame)

                # Extract trajectory for this ped_id
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                # Sort by frame
                curr_ped_seq = curr_ped_seq[curr_ped_seq[:, 0].argsort()]

                # Check if we have full sequence
                ped_frames = curr_ped_seq[:, 0]
                expected_frames = frames[seq_idx:seq_idx + seq_len]

                # Find matching frames
                frame_indices = []
                for exp_frame in expected_frames:
                    matching = np.where(ped_frames == exp_frame)[0]
                    if len(matching) > 0:
                        frame_indices.append(matching[0])

                if len(frame_indices) != seq_len:
                    continue  # Skip if we don't have full sequence

                # Extract coordinates (x, y) - these are in meter coordinates
                traj_coords = curr_ped_seq[frame_indices, 2:4]  # (seq_len, 2)

                # Split into obs and pred
                obs_traj = traj_coords[:obs_len]  # (obs_len, 2)
                pred_traj = traj_coords[obs_len:]  # (pred_len, 2)

                # Store trajectory with (ped_id, frame) key
                ped_trajectories[traj_key] = {
                    'obs_traj': obs_traj,
                    'pred_traj': pred_traj,
                    'scene_id': scene_name
                }

                ped_to_seq_info[traj_key] = {
                    'file_idx': file_idx,
                    'frame_start': frames[seq_idx],
                    'frame_end': frames[seq_idx + seq_len - 1]
                }
    
    # Cache the loaded trajectories
    _ped_trajectories_full_cache[cache_key] = ped_trajectories
    _ped_trajectories_full_cache[cache_key + ('seq_info',)] = ped_to_seq_info
    print(f"[CACHE] Cached {len(ped_trajectories)} trajectories for {cache_key}")
    
    # Build output arrays in the order of original_ped_ids
    num_ped = len(original_ped_ids)
    obs_traj_list = []
    pred_traj_list = []
    scene_id_list = []

    for i, ped_id in enumerate(original_ped_ids):
        ped_id_float = float(ped_id)
        if original_frames is not None:
            traj_key = (ped_id_float, float(original_frames[i]))
        else:
            traj_key = ped_id_float
        
        # Try to find trajectory: first try exact key, then try ped_id-only, then try any (ped_id, frame) key
        found_traj = None
        if traj_key in ped_trajectories:
            found_traj = ped_trajectories[traj_key]
        elif ped_id_float in ped_trajectories:
            found_traj = ped_trajectories[ped_id_float]
        else:
            # Try to find any (ped_id, frame) key for this ped_id
            for key in ped_trajectories.keys():
                if isinstance(key, tuple) and len(key) > 0 and key[0] == ped_id_float:
                    found_traj = ped_trajectories[key]
                    break
        
        if found_traj is None:
            raise KeyError(f"ped_id {ped_id_float} (frame={original_frames[i] if original_frames is not None else 'N/A'}) not found in loaded trajectories")
        
        obs_traj_list.append(found_traj['obs_traj'])
        pred_traj_list.append(found_traj['pred_traj'])
        scene_id_list.append(found_traj['scene_id'])

    
    obs_traj = np.stack(obs_traj_list, axis=0)  # (num_ped, obs_len, 2)
    pred_traj = np.stack(pred_traj_list, axis=0)  # (num_ped, pred_len, 2)
    scene_id = np.array(scene_id_list)
    
    # Build seq_start_end (group by scene and frame range)
    seq_start_end = []
    used_ped_indices = set()

    for i, ped_id in enumerate(original_ped_ids):
        if i in used_ped_indices:
            continue

        ped_id_float = float(ped_id)
        if original_frames is not None:
            seq_key = (ped_id_float, float(original_frames[i]))
        else:
            seq_key = ped_id_float
        if seq_key not in ped_to_seq_info:
            seq_key = ped_id_float
        if seq_key not in ped_to_seq_info:
            seq_start_end.append([i, i + 1])
            used_ped_indices.add(i)
            continue

        seq_info = ped_to_seq_info[seq_key]
        start_idx = i
        end_idx = i + 1

        for j in range(i + 1, num_ped):
            if j in used_ped_indices:
                continue
            other_ped_id = float(original_ped_ids[j])
            if original_frames is not None:
                other_key = (other_ped_id, float(original_frames[j]))
            else:
                other_key = other_ped_id
            if other_key not in ped_to_seq_info:
                other_key = other_ped_id
            if other_key in ped_to_seq_info:
                other_seq_info = ped_to_seq_info[other_key]
                if (other_seq_info['file_idx'] == seq_info['file_idx'] and
                    other_seq_info['frame_start'] == seq_info['frame_start'] and
                    other_seq_info['frame_end'] == seq_info['frame_end']):
                    end_idx = j + 1
                    used_ped_indices.add(j)

        seq_start_end.append([start_idx, end_idx])
        used_ped_indices.add(i)

    return obs_traj, pred_traj, scene_id, seq_start_end


def load_scene_resources(dataset_name, base_dir, unique_scenes=None):
    """
    Load vector_field and homography for given dataset
    
    Args:
        dataset_name: dataset name (e.g., 'eth', 'hotel')
        base_dir: base directory for datasets
        unique_scenes: list of unique scene names (optional)
        
    Returns:
        vector_field: dict {scene_name: vector_field_array}
        homography: dict {scene_name: homography_matrix}
    """
    scene_img_map = {
        'biwi_eth': 'seq_eth', 
        'biwi_hotel': 'seq_hotel',
        'students001': 'students003', 
        'students003': 'students003', 
        'uni_examples': 'students003',
        'crowds_zara01': 'crowds_zara01', 
        'crowds_zara02': 'crowds_zara02', 
        'crowds_zara03': 'crowds_zara02'
    }
    
    vector_field = {}
    homography = {}
    
    if unique_scenes is None:
        unique_scenes = [f'biwi_{dataset_name}']
    
    for scene_name in unique_scenes:
        if scene_name in scene_img_map:
            vector_field_path = os.path.join(
                base_dir, "vectorfield", 
                scene_img_map[scene_name] + "_vector_field.npy"
            )
            homography_path = os.path.join(
                base_dir, "homography", 
                scene_name + "_H.txt"
            )
            
            if os.path.exists(vector_field_path):
                vector_field[scene_name] = np.load(vector_field_path)
            
            if os.path.exists(homography_path):
                homography[scene_name] = np.loadtxt(homography_path)
        else:
            # Try direct dataset_name matching for standard datasets
            if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2"]:
                scene_name_from_dataset = f'biwi_{dataset_name}'
                if scene_name_from_dataset not in vector_field:
                    img_name = scene_img_map.get(scene_name_from_dataset, f'seq_{dataset_name}')
                    vector_field_path = os.path.join(
                        base_dir, "vectorfield", 
                        img_name + "_vector_field.npy"
                    )
                    homography_path = os.path.join(
                        base_dir, "homography", 
                        scene_name_from_dataset + "_H.txt"
                    )
                    
                    if os.path.exists(vector_field_path):
                        vector_field[scene_name_from_dataset] = np.load(vector_field_path)
                    
                    if os.path.exists(homography_path):
                        homography[scene_name_from_dataset] = np.loadtxt(homography_path)
    
    return vector_field, homography

def predict_SingularTrajectory_trajectories(hyper_params, device, batch, model, phase='test', obs_traj=None, pred_traj=None, 
                                    scene_id=None, dataset_name=None, 
                                    base_dir=None, data_dir=None, obs_len=8, pred_len=12):
    # Set data_dir from dataset_name and phase if not provided
    if data_dir is None:
        if dataset_name is not None and base_dir is not None:
            data_dir = os.path.join(base_dir, dataset_name, phase)
        else:
            raise ValueError("Either 'data_dir' or ('dataset_name' and 'base_dir') must be provided")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"data_dir does not exist: {data_dir}")
    
    cache_key = (data_dir, phase, obs_len, pred_len, id(hyper_params))
    
    if cache_key not in _EXPERT_SIG_PREDICTIONS_CACHE:
        _EXPERT_SIG_PREDICTIONS_CACHE[cache_key] = {}
    
    # Extract original_ped_id
    original_ped_ids = batch.get('original_ped_id', None)
    if original_ped_ids is None:
        raise ValueError("batch must contain 'original_ped_id'")

    if isinstance(original_ped_ids, torch.Tensor):
        original_ped_ids = original_ped_ids.cpu().numpy()

    # Extract frame info from batch
    batch_frames = batch.get('original_frame', None)
    if batch_frames is not None:
        if isinstance(batch_frames, torch.Tensor):
            batch_frames = batch_frames.cpu().numpy()
        else:
            batch_frames = np.array(batch_frames)

    num_ped = len(original_ped_ids)

    cache = _EXPERT_SIG_PREDICTIONS_CACHE[cache_key]
    original_ped_ids_float = original_ped_ids.astype(float)
    missing_ped_ids = []
    missing_indices = []

    for idx, ped_id in enumerate(original_ped_ids_float):
        if batch_frames is not None:
            ck = (float(ped_id), float(batch_frames[idx]))
        else:
            ck = float(ped_id)
        if ck not in cache:
            missing_ped_ids.append(float(ped_id))
            missing_indices.append(idx)

    if len(missing_ped_ids) == 0:
        result = []
        for i, pid in enumerate(original_ped_ids_float):
            if batch_frames is not None:
                ck = (float(pid), float(batch_frames[i]))
            else:
                ck = float(pid)
            result.append(cache[ck])
        return result
    
    # Load trajectories from original data files using ped_ids (always in meter coordinates)
    if data_dir and os.path.exists(data_dir):
        obs_traj_meter, pred_traj_meter, scene_id_loaded, seq_start_end = load_trajectories_from_ped_ids(
            original_ped_ids, data_dir, phase, obs_len, pred_len,
            original_frames=batch_frames
        )
        
        # Use loaded trajectories (in meter coordinates)
        obs_traj = obs_traj_meter
        pred_traj = pred_traj_meter
        
        # Check for NaN/Inf in obs_traj
        if np.isnan(obs_traj).any() or np.isinf(obs_traj).any():
            raise ValueError(f"obs_traj loaded from original data contains NaN or Inf!")
    else:
        raise ValueError(f"data_dir must be provided and exist: {data_dir}")
    
    # Parse sequences from ped_ids if seq_start_end not provided
    if seq_start_end is None or len(seq_start_end) == 0:
        # Fallback: treat all as single sequence
        seq_start_end = [[0, num_ped]]
    
    # Ensure tensors are on CPU for anchor calculation
    # obs_traj and pred_traj are now numpy arrays from original data (meter coordinates)
    obs_traj_cpu = torch.from_numpy(obs_traj).float()
    if pred_traj is None:
        pred_traj_cpu = torch.zeros((num_ped, pred_len, 2), dtype=torch.float32)
    else:
        pred_traj_cpu = torch.from_numpy(pred_traj).float()
    
    # Determine device
    if isinstance(device, str):
        device = torch.device(device)
    elif device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get scene_id (use loaded scene_id from original data)
    if scene_id is None:
        # Use scene_id loaded from original data files
        scene_id = scene_id_loaded
    
    # Ensure scene_id is numpy array
    if isinstance(scene_id, torch.Tensor):
        scene_id = scene_id.cpu().numpy()
    elif isinstance(scene_id, list):
        scene_id = np.array(scene_id)
    
    # Get vector_field and homography
    # Get vector_field and homography
    vector_field = batch.get('vector_field', None)
    homography = batch.get('homography', None)
    
    if vector_field is None or homography is None:
        if dataset_name and base_dir:
            vector_field, homography = load_scene_resources(
                dataset_name, base_dir, unique_scenes=np.unique(scene_id)
            )
        else:
            vector_field = {}
            homography = {}
    
    # Create wrapper dataset for anchor calculation
    wrapper_dataset = BatchDatasetWrapper(
        obs_traj=obs_traj_cpu,
        pred_traj=pred_traj_cpu,
        scene_id=scene_id,
        vector_field=vector_field,
        homography=homography
    )
    
    # Calculate anchor
    model.eval()
    with torch.no_grad():
        anchor = model.calculate_adaptive_anchor(wrapper_dataset)
    
    # Check for NaN/Inf in anchor
    if torch.isnan(anchor).any() or torch.isinf(anchor).any():
        raise ValueError(f"anchor contains NaN or Inf! NaN: {torch.isnan(anchor).sum()}, Inf: {torch.isinf(anchor).sum()}")
    
    # Prepare output batch (obs_traj and pred_traj are in meter coordinates from original data)
    output_batch = {
        'obs_traj': obs_traj_cpu.to(device),
        'anchor': anchor.to(device),
        'original_ped_id': batch.get('original_ped_id'),
        'scene_id': scene_id,
    }
    
    # Add pred_traj only once
    if pred_traj is not None:
        output_batch['pred_traj'] = pred_traj_cpu.to(device)
    
    scene_mask = torch.zeros((num_ped, num_ped), dtype=torch.bool, device=device)
    for start, end in seq_start_end:
        scene_mask[start:end, start:end] = True

    scene_mask.fill_diagonal_(True)


    output_batch['scene_mask'] = scene_mask
    output_batch['seq_start_end'] = torch.tensor(seq_start_end, dtype=torch.long, device=device)
    



    obs_traj = output_batch['obs_traj'].to(device)  # (num_ped, obs_len, 2)
    pred_traj_gt = output_batch['pred_traj'].to(device)  # (num_ped, pred_len, 2)
    adaptive_anchor = output_batch['anchor'].to(device)  # (num_ped, k, num_samples)
    scene_mask = output_batch['scene_mask'].to(device)
    seq_start_end = output_batch['seq_start_end']
    
    addl_info = {
        "scene_mask": scene_mask,
        "num_samples": hyper_params.num_samples,
        #"anchor": adaptive_anchor
    }


    output = model(obs_traj, adaptive_anchor, addl_info=addl_info)
    output = extract_best_trajectory(output["recon_traj"], pred_traj_gt)
    
    # Convert to list of numpy arrays (same format as dmrgcn_preds and stgcnn_preds)
    # output shape: (num_ped, pred_len, 2) - already in correct coordinates
    sig_preds = []
    for ped_idx in range(output.shape[0]):
        pred_coords = output[ped_idx]  # (pred_len, 2) - numpy array
        sig_preds.append(pred_coords)
        
        ped_id_float = float(original_ped_ids[ped_idx])
        if batch_frames is not None:
            cache_entry_key = (ped_id_float, float(batch_frames[ped_idx]))
        else:
            cache_entry_key = ped_id_float
        if cache_entry_key not in cache:
            cache[cache_entry_key] = pred_coords.copy()

    result = []
    for i, pid in enumerate(original_ped_ids_float):
        if batch_frames is not None:
            cache_entry_key = (float(pid), float(batch_frames[i]))
        else:
            cache_entry_key = float(pid)
        result.append(cache[cache_entry_key])

    return result

def extract_best_trajectory(pred, gt):
    r"""Extract best trajectory per pedestrian (lowest ADE across full trajectory).

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (num_ped, seq_len, 2) or (1, num_ped, seq_len, 2)

    Returns:
        best_pred (np.ndarray): (num_ped, seq_len, 2) - numpy array
    """
    if gt.dim() == 4:
        gt = gt.squeeze(0)  # (1, num_ped, seq_len, 2) -> (num_ped, seq_len, 2)

    # ADE per sample per ped: mean over timesteps of L2 distance
    # (num_samples, num_ped, seq_len)
    dist = (pred - gt.unsqueeze(0)).norm(p=2, dim=-1)
    ade = dist.mean(dim=-1)  # (num_samples, num_ped)

    # Best sample index per ped (lowest ADE)
    best_idx = ade.argmin(dim=0)  # (num_ped,)

    num_ped = pred.shape[1]
    best_pred = np.zeros((num_ped, pred.shape[2], 2), dtype=np.float32)
    for ped_idx in range(num_ped):
        best_pred[ped_idx] = pred[best_idx[ped_idx], ped_idx].detach().cpu().numpy()

    return best_pred


def extract_best_pred_per_timestep(pred, gt):
    r"""Extract best prediction per timestep (closest to GT) - DEPRECATED, use extract_best_trajectory.

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (num_ped, seq_len, 2) or (1, num_ped, seq_len, 2)

    Returns:
        best_pred (np.ndarray): (num_ped, seq_len, 2) - numpy array
    """
    if gt.dim() == 4:
        gt = gt.squeeze(0)

    temp = (pred - gt).norm(p=2, dim=-1)  # (num_samples, num_ped, seq_len)
    best_idx = temp.argmin(dim=0)  # (num_ped, seq_len)

    num_ped, seq_len = pred.shape[1], pred.shape[2]
    best_pred = np.zeros((num_ped, seq_len, 2), dtype=np.float32)

    for ped_idx in range(num_ped):
        for t_idx in range(seq_len):
            best_sample_idx = best_idx[ped_idx, t_idx]
            best_pred[ped_idx, t_idx] = pred[best_sample_idx, ped_idx, t_idx, :].detach().cpu().numpy()

    return best_pred