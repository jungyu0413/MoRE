"""Direct ExpertTraj prediction following the original test_ethucy.py logic.

Instead of mapping ped_ids to sequences, we iterate through the DataLoader
sequentially (like the original test code) and reconstruct absolute coordinates
to match with our (ped_id, scene) keyed predictions.
"""
import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn.functional as Func
from torch.utils.data import DataLoader

# Ensure expert_traj is importable
_WORKSPACE_DIR = os.environ.get('MORE_WORKSPACE_DIR', '/workspace')
_EXPERT_TRAJ_DIR = os.path.join(_WORKSPACE_DIR, 'expert_traj')
if _EXPERT_TRAJ_DIR not in sys.path:
    sys.path.insert(0, _EXPERT_TRAJ_DIR)

from utils_expert import TrajectoryDataset
from metrics import seq_to_nodes, nodes_rel_to_nodes_abs
from gmm2d import GMM2D


def predict_expert_traj_all(model, dataset_name, phase, device,
                            base_dir=None,
                            expert_goals_path=None,
                            obs_len=8, pred_len=12, n_samples=20,
                            use_gt_goals=False):
    """Run ExpertTraj prediction on all sequences and return {(ped_id, scene): pred}.

    Follows the same logic as test_ethucy.py but collects predictions keyed by
    (ped_id, scene) via absolute coordinate fingerprinting.

    Args:
        model: Loaded ExpertTraj model
        dataset_name: e.g., 'eth', 'hotel', etc.
        phase: 'train', 'val', 'test'
        device: torch device
        base_dir: ExpertTraj datasets directory (contains dataset_name/phase/)
        expert_goals_path: Path to pre-computed expert goals .npy file
        obs_len: observation length
        pred_len: prediction length
        n_samples: number of samples to draw
        use_gt_goals: If True, use GT endpoint as expert goal (for training data)

    Returns:
        dict: {(ped_id, scene): np.array (n_samples, pred_len, 2)} in absolute meter coords
    """
    data_dir = os.path.join(base_dir, dataset_name, phase)
    if not os.path.exists(data_dir):
        print(f"[EXPERT_TRAJ_DIRECT] Data dir not found: {data_dir}")
        return {}

    # Use "test" symlink trick to disable 24x angle augmentation
    test_link_dir = tempfile.mkdtemp(prefix="expert_traj_direct_")
    test_link_path = os.path.join(test_link_dir, "test_data")
    os.symlink(os.path.abspath(data_dir), test_link_path)

    dataset = TrajectoryDataset(
        test_link_path, obs_len=obs_len, pred_len=pred_len,
        skip=1, norm_lap_matr=True, grad_eff=0.4,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load pre-computed expert goals (for test data)
    expert_dest = None
    if expert_goals_path and os.path.exists(expert_goals_path):
        expert_dest = np.load(expert_goals_path)
        print(f"[EXPERT_TRAJ_DIRECT] Loaded expert goals: {expert_dest.shape}")

    # Build lookup: absolute obs trajectory → (ped_id, scene)
    # For each ped in each scene, for each possible obs window, create a fingerprint
    raw_trajs = {}  # {scene: {ped_id: [(frame, x, y), ...]}}
    scene_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
    for fname in scene_files:
        scene = fname.replace(f'_{phase}.txt', '').replace('_train.txt', '').replace('_test.txt', '').replace('_val.txt', '').replace('.txt', '')
        raw = np.loadtxt(os.path.join(data_dir, fname))
        raw_trajs[scene] = {}
        for row in raw:
            pid = float(row[1])
            if pid not in raw_trajs[scene]:
                raw_trajs[scene][pid] = []
            raw_trajs[scene][pid].append((row[0], row[2], row[3]))

    # Build lookup with ALL possible obs windows per ped (not just first window)
    obs_key_to_ped = {}
    for scene, peds in raw_trajs.items():
        for pid, points in peds.items():
            points_sorted = sorted(points, key=lambda x: x[0])
            coords = [(p[1], p[2]) for p in points_sorted]
            # Create key for each possible obs window
            for start_idx in range(len(coords) - obs_len + 1):
                window = coords[start_idx:start_idx + obs_len]
                k = (round(window[0][0], 3), round(window[0][1], 3),
                     round(window[-1][0], 3), round(window[-1][1], 3))
                obs_key_to_ped[k] = (pid, scene)

    model.eval()
    total_num_of_objs = 0
    all_predictions = {}  # {(ped_id, scene): np.array (n_samples, pred_len, 2)}
    matched = 0
    unmatched = 0

    with torch.no_grad():
        for step, batch_data in enumerate(loader):
            batch = [tensor.to(device) for tensor in batch_data]
            (obs_traj_norm, obs_traj, obs_traj_rel, pred_traj_gt, pred_traj_gt_rel,
             V_obs, A_obs, V_tr, A_tr, inp_mask, out_mask,
             velocity_obs, velocity_pred, acc_obs, acc_pred, seq_start) = batch

            num_of_objs = int(inp_mask[0, 0].sum().item())

            # Apply expert goals
            if use_gt_goals:
                # Use GT endpoint as goal (for training data)
                gt_endpoint = pred_traj_gt[0, -1, :num_of_objs, :]  # (num_of_objs, 2)
                rst = gt_endpoint.unsqueeze(0).unsqueeze(0).repeat(1, 8, 1, 1)
                obs_traj_norm[:, :, :num_of_objs] = obs_traj_norm[:, :, :num_of_objs] - rst
            elif expert_dest is not None and total_num_of_objs + num_of_objs <= len(expert_dest):
                rst = expert_dest[total_num_of_objs:total_num_of_objs + num_of_objs]
                rst = torch.from_numpy(rst).unsqueeze(0).to(device)
                rst = rst.permute(0, 2, 1, 3)
                rst = rst.view(1, 8, num_of_objs, 2)
                obs_traj_norm[:, :, :num_of_objs] = obs_traj_norm[:, :, :num_of_objs] - rst

            total_num_of_objs += num_of_objs

            # Model forward
            V_obs_tmp = torch.cat([obs_traj_norm, velocity_obs, acc_obs], dim=-1)
            V_obs_tmp = V_obs_tmp.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs, inp_mask, out_mask)
            V_pred = V_pred.squeeze()
            if V_pred.dim() == 2:
                V_pred = V_pred.unsqueeze(1)

            # Trim to valid peds
            V_pred = V_pred[:, :num_of_objs, :]
            obs_traj_trimmed = obs_traj.squeeze()[:, :num_of_objs, :]
            seq_start_trimmed = seq_start.squeeze()
            if seq_start_trimmed.dim() == 1:
                seq_start_trimmed = seq_start_trimmed.unsqueeze(0)
            seq_start_trimmed = seq_start_trimmed[:num_of_objs, :]

            # GMM sampling
            log_pis = torch.ones(V_pred[..., -2:-1].shape).to(device)
            gmm2d = GMM2D(
                log_pis,
                V_pred[..., 0:2],
                V_pred[..., 2:4],
                Func.tanh(V_pred[..., -1]).unsqueeze(-1),
            )

            # Get observation in centered coords for rel->abs conversion
            V_x = seq_to_nodes(obs_traj_trimmed.data.cpu().numpy().copy())

            for ped_idx in range(num_of_objs):
                # Reconstruct absolute coords of this ped's observation
                obs_centered = obs_traj_trimmed[:, ped_idx, :].cpu().numpy()  # (obs_len, 2)
                start_pos = seq_start_trimmed[ped_idx, :].cpu().numpy()       # (2,)
                obs_abs = obs_centered + start_pos  # (obs_len, 2) absolute meter coords

                # Match to (ped_id, scene) using coordinate fingerprint
                k = (round(float(obs_abs[0, 0]), 3), round(float(obs_abs[0, 1]), 3),
                     round(float(obs_abs[-1, 0]), 3), round(float(obs_abs[-1, 1]), 3))
                ped_info = obs_key_to_ped.get(k)

                if ped_info is None:
                    unmatched += 1
                    continue

                matched += 1

                # Sample predictions and convert to absolute coords
                # Use nodes_rel_to_nodes_abs (cumsum + last_obs) like test_ethucy.py
                sample_preds = []
                last_obs_centered = V_x[-1, ped_idx, :]  # last obs in centered coords

                for _ in range(n_samples):
                    V_pred_sample = gmm2d.rsample()
                    V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                        V_pred_sample.data.cpu().numpy().copy(),
                        V_x[-1, :, :].copy(),
                    )
                    if len(V_pred_rel_to_abs.shape) < 3:
                        V_pred_rel_to_abs = np.expand_dims(V_pred_rel_to_abs, 1)

                    # This gives centered coords; add seq_start to get absolute
                    pred_centered = V_pred_rel_to_abs[:, ped_idx, :]  # (pred_len, 2)
                    pred_abs = pred_centered + start_pos  # absolute meter coords
                    sample_preds.append(pred_abs)

                sample_preds = np.stack(sample_preds, axis=0)  # (n_samples, pred_len, 2)

                # Store first match (or overwrite with the latest - for training we might
                # have multiple windows for the same ped, take any valid one)
                if ped_info not in all_predictions:
                    all_predictions[ped_info] = sample_preds

    print(f"[EXPERT_TRAJ_DIRECT] Generated {len(all_predictions)} predictions "
          f"(matched={matched}, unmatched={unmatched})")
    return all_predictions


def verify_expert_traj_predictions(predictions, base_dir, dataset_name, phase,
                                   obs_len=8, pred_len=12):
    """Verify predictions against GT by computing ADE/FDE.

    Uses raw data files to get GT trajectories in absolute coordinates.
    """
    data_dir = os.path.join(base_dir, dataset_name, phase)
    scene_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])

    # Build (ped_id, scene) -> full trajectory mapping
    gt_trajs = {}
    for fname in scene_files:
        scene = fname.replace(f'_{phase}.txt', '').replace('_train.txt', '').replace('_test.txt', '').replace('_val.txt', '').replace('.txt', '')
        raw = np.loadtxt(os.path.join(data_dir, fname))
        peds = {}
        for row in raw:
            pid = float(row[1])
            if pid not in peds:
                peds[pid] = []
            peds[pid].append((row[0], row[2], row[3]))

        for pid, points in peds.items():
            points_sorted = sorted(points, key=lambda x: x[0])
            coords = np.array([(p[1], p[2]) for p in points_sorted])
            if len(coords) >= obs_len + pred_len:
                # Take first valid window's GT
                gt = coords[obs_len:obs_len + pred_len]  # (pred_len, 2)
                gt_trajs[(pid, scene)] = gt

    ades, fdes = [], []
    for key, preds in predictions.items():
        if key not in gt_trajs:
            continue
        gt = gt_trajs[key]  # (pred_len, 2)

        # Best-of-N ADE/FDE
        best_ade = float('inf')
        best_fde = float('inf')
        for s in range(preds.shape[0]):
            err = np.linalg.norm(preds[s] - gt, axis=-1)  # (pred_len,)
            ade = err.mean()
            fde = err[-1]
            if ade < best_ade:
                best_ade = ade
                best_fde = fde
        ades.append(best_ade)
        fdes.append(best_fde)

    if ades:
        print(f"[EXPERT_TRAJ_VERIFY] Matched {len(ades)}/{len(predictions)} predictions with GT")
        print(f"[EXPERT_TRAJ_VERIFY] Best-of-{preds.shape[0]} ADE: {np.mean(ades):.4f}, FDE: {np.mean(fdes):.4f}")
    else:
        print("[EXPERT_TRAJ_VERIFY] No matches found with GT")
    return np.mean(ades) if ades else None, np.mean(fdes) if fdes else None
