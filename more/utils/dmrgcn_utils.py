import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions import multivariate_normal

def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return NORM


def seq_to_graph(seq, seq_rel):
    assert seq.shape == seq_rel.shape

    num_nodes = seq.shape[0]
    seq_len = seq.shape[2]

    V = torch.zeros((seq_len, num_nodes, 2), dtype=torch.float)
    A_dist = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)
    A_disp = torch.zeros((seq_len, num_nodes, num_nodes), dtype=torch.float)

    for t in range(seq_len):
        for n in range(num_nodes):
            V[t, n, :] = seq_rel[n, :, t]
            for l in range(n + 1, num_nodes):
                A_dist[t, n, l] = A_dist[t, l, n] = anorm(seq[n, :, t], seq[l, :, t])
                A_disp[t, n, l] = A_disp[t, l, n] = anorm(seq_rel[n, :, t], seq_rel[l, :, t])

    return V, torch.stack([A_disp, A_dist], dim=0)


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

# Global cache for DMRGCN graphs: (data_dir, phase, obs_len, pred_len, skip, threshold, min_ped) -> {ped_id: [graph1, graph2, ...]}
_DMRGCN_CACHE = {}

def preload_dmrgcn_graphs(data_dir, phase, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t'):
    cache_key = (data_dir, phase, obs_len, pred_len, skip, threshold, min_ped)
    if cache_key in _DMRGCN_CACHE:
        return _DMRGCN_CACHE[cache_key]
    
    import torch.distributed as dist
    import pickle
    import hashlib
    
    cache_key_str = f"{data_dir}_{phase}_{obs_len}_{pred_len}_{skip}_{threshold}_{min_ped}"
    cache_key_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
    cache_file = f"/tmp/dmrgcn_cache_{cache_key_hash}.pkl"
    
    is_main_process = True
    if dist.is_initialized():
        is_main_process = (dist.get_rank() == 0)
    
    if is_main_process:
        print(f"[DMRGCN CACHE] Preloading ALL graphs from {data_dir} (phase={phase})...")
    else:
        import time
        max_wait = 3600
        wait_time = 0
        while not os.path.exists(cache_file) and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            if wait_time % 10 == 0:
                print(f"[DMRGCN CACHE] Rank {dist.get_rank()}: Waiting for cache file... ({wait_time}s)")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_store = pickle.load(f)
            _DMRGCN_CACHE[cache_key] = cache_store
            print(f"[DMRGCN CACHE] Rank {dist.get_rank()}: Loaded cache from file")
            return cache_store
        else:
            print(f"[DMRGCN CACHE] Rank {dist.get_rank()}: Cache file not found, proceeding with preloading...")
    
    cache_store = {}
    seq_len = obs_len + pred_len
    
    all_files = sorted(os.listdir(data_dir))
    all_files = [os.path.join(data_dir, _path) for _path in all_files]
    
    for path in tqdm(all_files, desc="Loading graphs"):
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
                
                # Make coordinates relative
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                _idx = num_peds_considered
                
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                
                # Linear vs Non-Linear Trajectory
                _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                ped_ids_in_graph.append(float(ped_id))
                num_peds_considered += 1
            
            if num_peds_considered > min_ped:
                curr_seq = curr_seq[:num_peds_considered]
                curr_seq_rel = curr_seq_rel[:num_peds_considered]
                curr_loss_mask = curr_loss_mask[:num_peds_considered]
                _non_linear_ped = np.asarray(_non_linear_ped)
                
                # Convert to torch tensor
                obs_traj = torch.from_numpy(curr_seq[:, :, :obs_len]).type(torch.float)
                pred_traj = torch.from_numpy(curr_seq[:, :, obs_len:]).type(torch.float)
                obs_traj_rel = torch.from_numpy(curr_seq_rel[:, :, :obs_len]).type(torch.float)
                pred_traj_rel = torch.from_numpy(curr_seq_rel[:, :, obs_len:]).type(torch.float)
                loss_mask = torch.from_numpy(curr_loss_mask).type(torch.float)
                non_linear_ped = torch.from_numpy(_non_linear_ped).type(torch.float)
                
                # Convert to graphs
                V_obs, A_obs = seq_to_graph(obs_traj, obs_traj_rel)
                V_pred, A_pred = seq_to_graph(pred_traj, pred_traj_rel)
                
                graph_data = [
                    obs_traj, pred_traj,
                    obs_traj_rel, pred_traj_rel,
                    non_linear_ped, loss_mask,
                    V_obs, A_obs,
                    V_pred, A_pred,
                    ped_ids_in_graph,
                    float(frames[idx])  # NEW: start_frame for (ped_id, frame) matching
                ]
                
                for ped_id in ped_ids_in_graph:
                    if ped_id not in cache_store:
                        cache_store[ped_id] = []
                    cache_store[ped_id].append(graph_data)
    
    _DMRGCN_CACHE[cache_key] = cache_store
    print(f"[DMRGCN CACHE] Loaded {len(cache_store)} unique ped_ids")
    
    if is_main_process:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_store, f)
            print(f"[DMRGCN CACHE] Saved cache to {cache_file}")
        except Exception as e:
            print(f"[DMRGCN CACHE] Warning: Failed to save cache file: {e}")
    
    if dist.is_initialized():
        dist.barrier()
    
    return cache_store

def create_graphs_from_ped_ids(ped_id_input, data_dir=None, dataset_name=None, phase='test', 
                                base_dir='./datasets/', obs_len=8, pred_len=8, skip=1, 
                                threshold=0.002, min_ped=1, delim='\t'):   
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
    
    cache_store = preload_dmrgcn_graphs(data_dir, phase, obs_len, pred_len, skip, threshold, min_ped, delim)
    
    filtered_results = []
    seen_graph_ids = set()
    
    for ped_id in ped_id_list:
        ped_id_float = float(ped_id)
        if ped_id_float not in cache_store:
            raise ValueError(f"DMRGCN: ped_id {ped_id_float} not found in cache. All original_ped_ids must have graphs.")

        for graph_data in cache_store[ped_id_float]:
            graph_id = id(graph_data)
            if graph_id not in seen_graph_ids:
                # Auto-detect layout: [-1] is scalar (current) or array (old)
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



class GraphListDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]



def generate_statistics_matrices(V):
    r"""generate mean and covariance matrices from the network output."""

    mu = V[:, :, 0:2]
    sx = V[:, :, 2].exp()
    sy = V[:, :, 3].exp()
    corr = V[:, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), 2, 2, device=V.device)
    cov[:, :, 0, 0] = sx * sx
    cov[:, :, 0, 1] = corr * sx * sy
    cov[:, :, 1, 0] = corr * sx * sy
    cov[:, :, 1, 1] = sy * sy

    return mu, cov


# Global cache for DMRGCN predictions: {cache_key: {ped_id: pred_traj}}
# cache_key = (dataset_name, phase, obs_len, pred_len, n_samples)
_EXPERT_DMRGCN_PREDICTIONS_CACHE = {}



def predict_dmrgcn_trajectories(
    batch,
    dmrgcn_model,
    dataset_name,
    phase='test',
    obs_len=8,
    pred_len=12,
    base_dir=None,
    n_samples=20,
    device='cuda',
    verbose=False
):
    cache_key = (dataset_name, phase, obs_len, pred_len, n_samples)
    
    if cache_key not in _EXPERT_DMRGCN_PREDICTIONS_CACHE:
        _EXPERT_DMRGCN_PREDICTIONS_CACHE[cache_key] = {}
        if verbose:
            print(f"[DMRGCN PRED CACHE] Initialized cache for {cache_key}")
    
    original_ped_ids = batch['original_ped_id'].cpu().numpy()

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

    cache = _EXPERT_DMRGCN_PREDICTIONS_CACHE[cache_key]
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
            print(f"[DMRGCN PRED CACHE] Cache hit: {len(original_ped_ids)} predictions retrieved from cache")
        result = []
        for i, pid in enumerate(original_ped_ids_float):
            if _batch_frames_for_cache is not None:
                ck = (float(pid), float(_batch_frames_for_cache[i]))
            else:
                ck = float(pid)
            result.append(cache[ck])
        return result
    
    if verbose:
        print(f"[DMRGCN PRED CACHE] Cache miss: computing predictions for {len(missing_ped_ids)} ped_ids")
    
    graphs = create_graphs_from_ped_ids(
        original_ped_ids,
        dataset_name=dataset_name,
        phase=phase,
        obs_len=obs_len,
        pred_len=pred_len,
        base_dir=base_dir
    )
    
    if verbose:
        print(f"Number of graphs created: {len(graphs)}")
        print(f"Original ped_ids (first 10): {original_ped_ids[:10]}")
    
    dmrgcn_preds = [None] * len(original_ped_ids)
    
    graph_dataset = GraphListDataset(graphs)
    dmrgcn_loader = DataLoader(graph_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    # Extract frame info from batch (None if not provided → backward compat)
    batch_frames = batch.get('original_frame', None)
    if batch_frames is not None:
        if isinstance(batch_frames, torch.Tensor):
            batch_frames = batch_frames.cpu().numpy()
        else:
            batch_frames = np.array(batch_frames)

    for batch_idx, batch_data in enumerate(dmrgcn_loader):
        # Auto-detect current layout: last element is scalar (start_frame)
        last_elem = batch_data[-1]
        _has_start_frame = isinstance(last_elem, torch.Tensor)
        if _has_start_frame:
            graph_start_frame = float(last_elem.item()) if isinstance(last_elem, torch.Tensor) else float(last_elem)
            ped_ids_in_graph_raw = batch_data[-2]
            V_obs, A_obs, V_tr, A_tr = [tensor.to(device) for tensor in batch_data[-6:-2]]
        else:
            graph_start_frame = None
            ped_ids_in_graph_raw = batch_data[-1]
            V_obs, A_obs, V_tr, A_tr = [tensor.to(device) for tensor in batch_data[-5:-1]]

        obs_traj, pred_traj_gt = [tensor.to(device) for tensor in batch_data[:2]]
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
        
        if verbose and batch_idx == 0:
            print(f"First graph ped_ids: {ped_ids_in_graph}")
            print(f"Original ped_ids (first 10): {original_ped_ids[:10]}")
            ped_ids_set = set(ped_ids_in_graph.astype(float))
            original_set = set(original_ped_ids.astype(float))
            intersection = ped_ids_set & original_set
            print(f"Intersection: {intersection}")
        
        V_obs_ = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = dmrgcn_model(V_obs_, A_obs)
        V_pred = V_pred.permute(0, 2, 3, 1)
        
        V_pred = V_pred.squeeze()
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)
        
        mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
        mv_normal = multivariate_normal.MultivariateNormal(mu, cov)
        V_pred_sample = mv_normal.sample((n_samples,))
        
        # Relative trajectories to absolute trajectories
        V_absl = []
        for t in range(V_pred_sample.size(1)):
            V_absl.append(V_pred_sample[:, 0:t + 1, :, :].sum(dim=1, keepdim=True) + V_obs_traj[-1, :, :])
        V_absl = torch.cat(V_absl, dim=1)
        
        # Calculate ADEs and FDEs for each trajectory
        temp = V_absl - V_pred_traj_gt
        temp = (temp ** 2).sum(dim=-1).sqrt()
        
        ped_ids_in_graph_float = ped_ids_in_graph.astype(float)
        original_ped_ids_float = original_ped_ids.astype(float)

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
            target_indices = torch.tensor(target_indices, dtype=torch.long).to(device)
            # V_absl shape: (n_samples, pred_len, num_nodes, 2)
            V_absl_target = V_absl[:, :, target_indices, :]  # (n_samples, pred_len, num_target_peds, 2)
            V_pred_traj_gt_target = V_pred_traj_gt[:, target_indices, :]  # (pred_len, num_target_peds, 2)
            
            temp_target = V_absl_target - V_pred_traj_gt_target.unsqueeze(0)  # (n_samples, pred_len, num_target_peds, 2)
            temp_target = (temp_target ** 2).sum(dim=-1).sqrt()  # (n_samples, pred_len, num_target_peds)
            
            mean_errors = temp_target.mean(dim=1)  # (n_samples, num_target_peds)
            best_sample_indices = mean_errors.argmin(dim=0)  # (num_target_peds,)
            
            for local_idx, orig_idx in enumerate(original_ped_indices):
                best_idx = best_sample_indices[local_idx].item()
                pred_coords = V_absl_target[best_idx, :, local_idx, :].cpu().numpy()
                dmrgcn_preds[orig_idx] = pred_coords
                
                ped_id_float = float(original_ped_ids[orig_idx])
                if batch_frames is not None:
                    cache_entry_key = (ped_id_float, float(batch_frames[orig_idx]))
                else:
                    cache_entry_key = ped_id_float
                if cache_entry_key not in cache:
                    cache[cache_entry_key] = pred_coords.copy()

    missing_indices = [i for i, pred in enumerate(dmrgcn_preds) if pred is None]
    if missing_indices:
        missing_ped_ids = [original_ped_ids[i] for i in missing_indices]
        error_msg = (
            f"DMRGCN: {len(missing_indices)} predictions are missing for ped_ids: "
            f"{missing_ped_ids[:10]}{'...' if len(missing_ped_ids) > 10 else ''}. "
            f"This should not happen - all original_ped_ids must have predictions. "
            f"Total requested: {len(original_ped_ids)}, Got: {len(original_ped_ids) - len(missing_indices)}"
        )
        print(f"[DMRGCN PRED] ERROR: {error_msg}")
        import sys
        sys.stdout.flush()
        raise ValueError(error_msg)

    result = []
    for i, pid in enumerate(original_ped_ids_float):
        if batch_frames is not None:
            cache_entry_key = (float(pid), float(batch_frames[i]))
        else:
            cache_entry_key = float(pid)
        result.append(cache[cache_entry_key])
    
    if verbose:
        print(f"[DMRGCN PRED CACHE] Cached {len(missing_ped_ids)} new predictions. Total cached: {len(cache)}")
    
    return result