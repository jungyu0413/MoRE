"""
GP-Graph: Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction (ECCV 2022)
"""

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def loc_pos(seq_):
    """Add position encoding to sequence (GPGraph/SGCN style)"""
    obs_len = seq_.shape[0]
    num_ped = seq_.shape[1]

    pos_seq = np.arange(1, obs_len + 1)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat(num_ped, axis=1)

    result = np.concatenate((pos_seq, seq_), axis=-1)
    return result


def seq_to_graph_gpgraph(seq_, seq_rel, pos_enc=False):
    """Convert sequence to graph format for GPGraph (SGCN baseline)
    
    Args:
        seq_: (num_peds, 2, seq_len) - absolute positions
        seq_rel: (num_peds, 2, seq_len) - relative positions
        pos_enc: whether to add position encoding
    
    Returns:
        V: (seq_len, num_peds, 2 or 3) - node features
    """
    if isinstance(seq_, torch.Tensor):
        seq_ = seq_.cpu().numpy()
    if isinstance(seq_rel, torch.Tensor):
        seq_rel = seq_rel.cpu().numpy()
    
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]

    if pos_enc:
        V = loc_pos(V)

    return torch.from_numpy(V).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
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


# Global cache for GPGraph graphs
_GPGRAPH_CACHE = {}

# Global cache for GPGraph predictions
_EXPERT_GPGRAPH_PREDICTIONS_CACHE = {}


def preload_gpgraph_graphs(data_dir, phase, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=1, delim='\t'):
    """
    """
    cache_key = (data_dir, phase, obs_len, pred_len, skip, threshold, min_ped)
    if cache_key in _GPGRAPH_CACHE:
        return _GPGRAPH_CACHE[cache_key]
    
    import torch.distributed as dist
    import pickle
    import hashlib
    
    cache_key_str = f"gpgraph_{data_dir}_{phase}_{obs_len}_{pred_len}_{skip}_{threshold}_{min_ped}"
    cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = f"/tmp/gpgraph_cache_{cache_key_hash}.pkl"
    
    is_main_process = True
    if dist.is_initialized():
        is_main_process = (dist.get_rank() == 0)
    
    if is_main_process:
        print(f"[GPGRAPH CACHE] Preloading ALL graphs from {data_dir} (phase={phase})...")
    else:
        import time
        max_wait = 3600
        wait_time = 0
        while not os.path.exists(cache_file) and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            if wait_time % 10 == 0:
                print(f"[GPGRAPH CACHE] Rank {dist.get_rank()}: Waiting for cache file... ({wait_time}s)")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_store = pickle.load(f)
            _GPGRAPH_CACHE[cache_key] = cache_store
            print(f"[GPGRAPH CACHE] Rank {dist.get_rank()}: Loaded cache from file")
            return cache_store
        else:
            print(f"[GPGRAPH CACHE] Rank {dist.get_rank()}: Cache file not found, proceeding with preloading...")
    
    cache_store = {}
    seq_len = obs_len + pred_len

    all_files = sorted(os.listdir(data_dir))
    all_files = [os.path.join(data_dir, _path) for _path in all_files]

    # === First pass: collect all sequences to compute scale_factor ===
    # (Same logic as GPGraph/utils.py TrajectoryDataset)
    all_seq_list = []
    all_seq_meta = []  # Store metadata for second pass

    for path in tqdm(all_files, desc="Loading GPGraph graphs (pass 1: collect)"):
        data = read_file(path, delim)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])

        num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

        for idx in range(0, num_sequences * skip + 1, skip):
            if idx + seq_len > len(frames):
                break
            curr_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])

            curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), seq_len))

            num_peds_considered = 0
            _non_linear_ped = []
            ped_ids_in_graph = []

            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                if len(curr_ped_seq) == 0:
                    continue
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                if pad_end - pad_front != seq_len:
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])

                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                _idx = num_peds_considered

                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                ped_ids_in_graph.append(float(ped_id))
                num_peds_considered += 1

            if num_peds_considered > min_ped:
                curr_seq = curr_seq[:num_peds_considered]
                curr_seq_rel = curr_seq_rel[:num_peds_considered]
                curr_loss_mask = curr_loss_mask[:num_peds_considered]
                _non_linear_ped = np.asarray(_non_linear_ped)
                all_seq_list.append(curr_seq)
                all_seq_meta.append((curr_seq, curr_seq_rel, curr_loss_mask, _non_linear_ped, ped_ids_in_graph, frames[idx]))

    # === Compute scale_factor (same as GPGraph/utils.py) ===
    if len(all_seq_list) > 0:
        all_seq_concat = np.concatenate(all_seq_list, axis=0)
        coord_range = all_seq_concat.max() - all_seq_concat.min()
        if coord_range > 100:
            scale_factor = coord_range / 10.0
        else:
            scale_factor = 1.0
    else:
        scale_factor = 1.0

    if is_main_process:
        print(f"[GPGRAPH CACHE] Scale factor = {scale_factor:.4f} (coord_range = {coord_range:.2f})")

    # Store scale_factor in cache for later unscaling
    cache_store['__scale_factor__'] = scale_factor

    # === Second pass: apply scaling and build graphs ===
    for curr_seq, curr_seq_rel, curr_loss_mask, _non_linear_ped, ped_ids_in_graph, start_frame in all_seq_meta:
        # Apply scale_factor to coordinates
        curr_seq_scaled = curr_seq / scale_factor
        curr_seq_rel_scaled = curr_seq_rel / scale_factor

        # Convert to torch tensor
        obs_traj = torch.from_numpy(curr_seq_scaled[:, :, :obs_len]).type(torch.float)
        pred_traj = torch.from_numpy(curr_seq_scaled[:, :, obs_len:]).type(torch.float)
        obs_traj_rel = torch.from_numpy(curr_seq_rel_scaled[:, :, :obs_len]).type(torch.float)
        pred_traj_rel = torch.from_numpy(curr_seq_rel_scaled[:, :, obs_len:]).type(torch.float)
        loss_mask = torch.from_numpy(curr_loss_mask).type(torch.float)
        non_linear_ped = torch.from_numpy(_non_linear_ped).type(torch.float)

        # GPGraph uses seq_to_graph with position encoding
        V_obs = seq_to_graph_gpgraph(obs_traj, obs_traj_rel, pos_enc=True)
        V_pred = seq_to_graph_gpgraph(pred_traj, pred_traj_rel, pos_enc=False)

        graph_data = [
            obs_traj, pred_traj,
            obs_traj_rel, pred_traj_rel,
            non_linear_ped, loss_mask,
            V_obs, V_pred,
            ped_ids_in_graph,
            float(start_frame)  # NEW: start_frame for (ped_id, frame) matching
        ]

        for ped_id in ped_ids_in_graph:
            if ped_id not in cache_store:
                cache_store[ped_id] = []
            cache_store[ped_id].append(graph_data)

    _GPGRAPH_CACHE[cache_key] = cache_store
    print(f"[GPGRAPH CACHE] Loaded {len(cache_store) - 1} unique ped_ids (scale_factor={scale_factor:.4f})")
    
    if is_main_process:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_store, f)
            print(f"[GPGRAPH CACHE] Saved cache to {cache_file}")
        except Exception as e:
            print(f"[GPGRAPH CACHE] Warning: Failed to save cache file: {e}")
    
    if dist.is_initialized():
        dist.barrier()
    
    return cache_store


def create_graphs_from_ped_ids_gpgraph(ped_id_input, data_dir=None, dataset_name=None, phase='test',
                                        base_dir='./dataset/', obs_len=8, pred_len=12, skip=1,
                                        threshold=0.002, min_ped=1, delim='\t'):
    """
    """
    if dataset_name is not None:
        data_dir = os.path.join(base_dir, dataset_name, phase)
    
    if data_dir is None:
        raise ValueError("Either 'data_dir' or 'dataset_name' must be provided")
    
    if isinstance(ped_id_input, dict):
        if 'original_ped_id' in ped_id_input:
            ped_id_tensor = ped_id_input['original_ped_id']
        else:
            raise ValueError("batch dict must contain 'original_ped_id' key")
    else:
        ped_id_tensor = ped_id_input
    
    if isinstance(ped_id_tensor, torch.Tensor):
        ped_id_list = ped_id_tensor.cpu().numpy()
    else:
        ped_id_list = np.array(ped_id_tensor)
    
    target_ped_ids = set(ped_id_list.astype(float))
    
    cache_store = preload_gpgraph_graphs(data_dir, phase, obs_len, pred_len, skip, threshold, min_ped, delim)
    
    filtered_results = []
    seen_graph_ids = set()
    
    for ped_id in ped_id_list:
        ped_id_float = float(ped_id)
        if ped_id_float not in cache_store:
            raise ValueError(f"GPGRAPH: ped_id {ped_id_float} not found in cache. All original_ped_ids must have graphs.")

        for graph_data in cache_store[ped_id_float]:
            graph_id = id(graph_data)
            if graph_id not in seen_graph_ids:
                last_elem = graph_data[-1]
                if isinstance(last_elem, (int, float)):
                    ped_ids_in_graph = graph_data[-2]
                else:
                    ped_ids_in_graph = graph_data[-1]
                ped_ids_in_graph_set = set([float(pid) for pid in ped_ids_in_graph])
                if target_ped_ids & ped_ids_in_graph_set:
                    filtered_results.append(graph_data)
                    seen_graph_ids.add(graph_id)

    return filtered_results


class GPGraphListDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def generate_statistics_matrices_gpgraph(V, device='cuda'):
    """Generate mean and covariance matrices from the network output (GPGraph/SGCN style)
    
    Args:
        V: (pred_len, num_peds, 5) - network output [mu_x, mu_y, sigma_x, sigma_y, corr]
    
    Returns:
        mu: (pred_len, num_peds, 2) - mean positions
        cov: (pred_len, num_peds, 2, 2) - covariance matrices
    """
    mu = V[:, :, 0:2]
    sx = V[:, :, 2].exp()
    sy = V[:, :, 3].exp()
    corr = V[:, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), 2, 2, device=device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy

    return mu, cov


def nodes_rel_to_nodes_abs(nodes, init_node):
    """Convert relative positions to absolute positions
    
    Args:
        nodes: (pred_len, num_peds, 2) - relative positions
        init_node: (num_peds, 2) - initial absolute positions
    
    Returns:
        nodes_abs: (pred_len, num_peds, 2) - absolute positions
    """
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]
    return nodes_


def predict_gpgraph_trajectories(
    batch,
    gpgraph_model,
    dataset_name,
    phase='test',
    obs_len=8,
    pred_len=12,
    base_dir=None,
    n_samples=20,
    device='cuda',
    verbose=False,
    use_simple_sampler=False
):
    """


    """
    from sklearn.cluster import KMeans
    
    cache_key = (dataset_name, phase, obs_len, pred_len, n_samples)
    
    if cache_key not in _EXPERT_GPGRAPH_PREDICTIONS_CACHE:
        _EXPERT_GPGRAPH_PREDICTIONS_CACHE[cache_key] = {}
        if verbose:
            print(f"[GPGRAPH PRED CACHE] Initialized cache for {cache_key}")
    
    original_ped_ids = batch['original_ped_id']
    if isinstance(original_ped_ids, torch.Tensor):
        original_ped_ids = original_ped_ids.cpu().numpy()
    else:
        original_ped_ids = np.array(original_ped_ids)
    
    # Extract frame info from batch (for cache key)
    _batch_frames_for_cache = batch.get('original_frame', None)
    if _batch_frames_for_cache is not None:
        if isinstance(_batch_frames_for_cache, torch.Tensor):
            _batch_frames_for_cache = _batch_frames_for_cache.cpu().numpy()
        else:
            _batch_frames_for_cache = np.array(_batch_frames_for_cache)

    cache = _EXPERT_GPGRAPH_PREDICTIONS_CACHE[cache_key]
    original_ped_ids_float = original_ped_ids.astype(float)

    missing_ped_ids = []
    missing_indices = []
    for idx, ped_id in enumerate(original_ped_ids_float):
        if _batch_frames_for_cache is not None:
            ck = (float(ped_id), float(_batch_frames_for_cache[idx]))
        else:
            ck = float(ped_id)
        if ck not in cache:
            missing_ped_ids.append(float(ped_id))
            missing_indices.append(idx)

    if len(missing_ped_ids) == 0:
        if verbose:
            print(f"[GPGRAPH PRED CACHE] Cache hit: {len(original_ped_ids)} predictions retrieved from cache")
        result = []
        for i, pid in enumerate(original_ped_ids_float):
            if _batch_frames_for_cache is not None:
                ck = (float(pid), float(_batch_frames_for_cache[i]))
            else:
                ck = float(pid)
            result.append(cache[ck])
        return result
    
    if verbose:
        print(f"[GPGRAPH PRED CACHE] Cache miss: computing predictions for {len(missing_ped_ids)} ped_ids")
    
    graphs = create_graphs_from_ped_ids_gpgraph(
        original_ped_ids,
        dataset_name=dataset_name,
        phase=phase,
        obs_len=obs_len,
        pred_len=pred_len,
        base_dir=base_dir
    )

    # Retrieve scale_factor from cache for unscaling predictions
    _cache_key = None
    for k, v in _GPGRAPH_CACHE.items():
        if isinstance(v, dict) and '__scale_factor__' in v:
            gpgraph_scale_factor = v['__scale_factor__']
            break
    else:
        gpgraph_scale_factor = 1.0

    if verbose:
        print(f"Number of graphs created: {len(graphs)}, scale_factor={gpgraph_scale_factor:.4f}")
    
    gpgraph_preds = [None] * len(original_ped_ids)
    
    graph_dataset = GPGraphListDataset(graphs)
    gpgraph_loader = DataLoader(graph_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    class SimpleRandomSampler:
        """Simple Gaussian sampler — direct sampling without KMeans."""
        def __init__(self, scale=0.8):
            self.scale = scale

        def randn(self, n, k, d):
            """
            Args:
            """
            return np.random.randn(n, k, d) * self.scale

    class RandomSampler:
        def __init__(self, stack_n=1000, fast_sample=True):
            self.stack_n = stack_n
            self.pre_samples = []
            self.fast_sample = fast_sample

        def randn(self, n, k, d):
            if self.fast_sample and len(self.pre_samples) > self.stack_n:
                self.pre_samples = np.array(self.pre_samples)
                return self.pre_samples[np.random.choice(np.arange(self.stack_n), n)]
            randn_sample = []
            for _ in range(n):
                k_samples = KMeans(n_clusters=k).fit(np.random.normal(size=(1000, d)))
                randn_sample.append(k_samples.cluster_centers_ * 0.8)
            self.pre_samples.extend(randn_sample) if self.fast_sample else None
            return np.array(randn_sample)

    if use_simple_sampler:
        random_sampler = SimpleRandomSampler(scale=0.8)
        if verbose:
            print("[GPGRAPH] Using SimpleRandomSampler (fast, np.random.randn)")
    else:
        random_sampler = RandomSampler()
        if verbose:
            print("[GPGRAPH] Using original RandomSampler (KMeans-based)")
    
    # Extract frame info from batch (None if not provided → backward compat)
    batch_frames = batch.get('original_frame', None)
    if batch_frames is not None:
        if isinstance(batch_frames, torch.Tensor):
            batch_frames = batch_frames.cpu().numpy()
        else:
            batch_frames = np.array(batch_frames)

    for batch_idx, batch_data in enumerate(gpgraph_loader):
        # Auto-detect current layout: last element is scalar (start_frame)
        last_elem = batch_data[-1]
        _has_start_frame = isinstance(last_elem, torch.Tensor)
        if _has_start_frame:
            graph_start_frame = float(last_elem.item()) if isinstance(last_elem, torch.Tensor) else float(last_elem)
            ped_ids_in_graph_raw = batch_data[-2]
        else:
            graph_start_frame = None
            ped_ids_in_graph_raw = batch_data[-1]

        obs_traj = batch_data[0].to(device)
        pred_traj_gt = batch_data[1].to(device)
        obs_traj_rel = batch_data[2].to(device)
        pred_traj_rel = batch_data[3].to(device)
        V_obs = batch_data[6].to(device)
        V_tr = batch_data[7].to(device)
        ped_ids_in_graph = ped_ids_in_graph_raw

        if isinstance(ped_ids_in_graph, list):
            if len(ped_ids_in_graph) > 0 and isinstance(ped_ids_in_graph[0], torch.Tensor):
                ped_ids_list = []
                for pid in ped_ids_in_graph:
                    if pid.numel() == 1:
                        ped_ids_list.append(float(pid.item()))
                    else:
                        ped_ids_list.extend([float(x.item()) for x in pid.flatten()])
                ped_ids_in_graph = np.array(ped_ids_list)
            else:
                ped_ids_in_graph = np.array([float(pid) for pid in ped_ids_in_graph])
        elif isinstance(ped_ids_in_graph, torch.Tensor):
            ped_ids_in_graph = ped_ids_in_graph.cpu().numpy().flatten()
        else:
            ped_ids_in_graph = np.array(ped_ids_in_graph).flatten()
        
        # GPGraph forward
        # V_obs shape from DataLoader: (1, obs_len, num_nodes, 3) or (obs_len, num_nodes, 3)
        V_obs_abs = obs_traj.permute(0, 2, 3, 1)  # (1, 2, num_ped, obs_len) -> (1, num_ped, obs_len, 2)
        
        # Handle batch dimension from DataLoader
        if V_obs.dim() == 4:
            # V_obs is (1, obs_len, num_nodes, 3), squeeze batch dim first
            V_obs_tmp = V_obs.squeeze(0)  # (obs_len, num_nodes, 3)
        else:
            V_obs_tmp = V_obs  # (obs_len, num_nodes, 3)
        
        # Now add batch dimension and permute: (obs_len, num_nodes, 3) -> (1, 3, obs_len, num_ped)
        V_obs_tmp = V_obs_tmp.unsqueeze(0).permute(0, 3, 1, 2)  # (1, 3, obs_len, num_ped)
        
        with torch.no_grad():
            V_pred, indices = gpgraph_model(V_obs_abs, V_obs_tmp)
        
        V_pred = V_pred.permute(0, 2, 3, 1)  # (1, pred_len, num_ped, 5)
        V_pred = V_pred.squeeze()  # (pred_len, num_ped, 5)
        
        # Generate statistics matrices
        mu, cov = generate_statistics_matrices_gpgraph(V_pred.squeeze(dim=0) if V_pred.dim() > 3 else V_pred, device=device)
        
        # Get observation trajectory for relative to absolute conversion
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)  # (obs_len, num_ped, 2)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)  # (pred_len, num_ped, 2)
        
        ped_ids_in_graph_float = ped_ids_in_graph.astype(float)

        target_indices = []
        original_ped_indices = []
        for orig_idx, ped_id in enumerate(original_ped_ids_float):
            # Frame-based filtering: skip if this graph is from a different time window
            if batch_frames is not None and graph_start_frame is not None:
                if not np.isclose(batch_frames[orig_idx], graph_start_frame, atol=0.5):
                    continue
            matches = np.isclose(ped_ids_in_graph_float, ped_id, rtol=1e-5, atol=1e-8)
            if np.any(matches):
                graph_idx = np.where(matches)[0][0]
                target_indices.append(graph_idx)
                original_ped_indices.append(orig_idx)
        
        if len(target_indices) > 0:
            target_indices_np = np.array(target_indices)
            
            # Following original GPGraph/test.py implementation
            sample_level = 'group'
            n_ped = V_pred.size(1) if V_pred.dim() >= 2 else 1
            
            if sample_level == 'group' and indices is not None and not use_simple_sampler:
                n_groups = indices.unique().size(0)
                r_sample = random_sampler.randn(n_groups, n_samples, 2)
                # Use .take() like original GPGraph code for proper indexing
                r_sample = np.take(r_sample, indices.detach().cpu().numpy(), axis=0)
            else:
                r_sample = random_sampler.randn(n_ped, n_samples, 2)
            
            r_sample = torch.Tensor(r_sample).to(dtype=mu.dtype, device=device)
            # Original GPGraph: permute(1,0,2) then unsqueeze(dim=1), NOT dim=0
            r_sample = r_sample.permute(1, 0, 2).unsqueeze(dim=1).expand((n_samples,) + mu.shape)
            
            # Cholesky decomposition for sampling
            try:
                L = torch.linalg.cholesky(cov)
                V_pred_sample = mu + (L @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)
            except RuntimeError:
                V_pred_sample = mu.unsqueeze(0).expand((n_samples,) + mu.shape)
            
            # Relative to absolute conversion (in scaled space)
            V_obs_last = V_obs_traj[-1, :, :].cpu().numpy()  # (num_ped, 2)
            V_absl = V_pred_sample.cumsum(dim=1) + V_obs_traj[[-1], :, :].unsqueeze(0)  # (n_samples, pred_len, num_ped, 2)

            # Unscale back to original pixel coordinates
            V_absl = V_absl * gpgraph_scale_factor

            V_pred_traj_gt_unscaled = V_pred_traj_gt * gpgraph_scale_factor
            V_pred_traj_gt_expanded = V_pred_traj_gt_unscaled.unsqueeze(0)  # (1, pred_len, num_ped, 2)
            temp = (V_absl - V_pred_traj_gt_expanded).norm(p=2, dim=-1)  # (n_samples, pred_len, num_ped)
            
            ade_per_ped = temp.mean(dim=1)  # (n_samples, num_ped)
            best_sample_indices = ade_per_ped.argmin(dim=0)  # (num_ped,)
            
            for local_idx, orig_idx in enumerate(original_ped_indices):
                graph_idx = target_indices[local_idx]
                best_idx = best_sample_indices[graph_idx].item()
                pred_coords = V_absl[best_idx, :, graph_idx, :].cpu().numpy()  # (pred_len, 2)
                gpgraph_preds[orig_idx] = pred_coords
                
                ped_id_float = float(original_ped_ids[orig_idx])
                if batch_frames is not None:
                    cache_entry_key = (ped_id_float, float(batch_frames[orig_idx]))
                else:
                    cache_entry_key = ped_id_float
                if cache_entry_key not in cache:
                    cache[cache_entry_key] = pred_coords.copy()

    missing_indices = [i for i, pred in enumerate(gpgraph_preds) if pred is None]
    if missing_indices:
        missing_ped_ids = [original_ped_ids[i] for i in missing_indices]
        raise ValueError(
            f"GPGRAPH: {len(missing_indices)} predictions are missing for ped_ids: "
            f"{missing_ped_ids[:10]}{'...' if len(missing_ped_ids) > 10 else ''}. "
            f"This should not happen - all original_ped_ids must have predictions."
        )

    result = []
    for i, pid in enumerate(original_ped_ids_float):
        if batch_frames is not None:
            cache_entry_key = (float(pid), float(batch_frames[i]))
        else:
            cache_entry_key = float(pid)
        result.append(cache[cache_entry_key])
    
    if verbose:
        print(f"[GPGRAPH PRED CACHE] Cached {len(missing_ped_ids)} new predictions. Total cached: {len(cache)}")
    
    return result
