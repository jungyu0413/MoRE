"""Expert prediction precomputation for the K=5 expert ensemble.

Precomputes and caches predictions from each enabled trajectory expert model
for every pedestrian (keyed by original_ped_id) in a dataset split, so that
training and reward computation can look up expert outputs without
re-invoking the experts on every batch.

Experts: SingularTrajectory, DMRGCN, GPGraph, Social-STGCNN, ExpertTraj.
"""

import os
import pickle
import torch
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from more.utils.dmrgcn_utils import predict_dmrgcn_trajectories
from more.utils.gpgraph_utils import predict_gpgraph_trajectories
from more.utils.sig_utils import predict_SingularTrajectory_trajectories
from more.utils.stgcnn_utils import predict_stgcnn_trajectories
from more.utils.expert_traj_predict_direct import predict_expert_traj_all
import tempfile
import shutil


def _create_scene_data_dir(base_dir, dataset_name, phase, scene_name):
    """
    Create a temporary directory mimicking base_dir/dataset_name/phase/ structure
    but containing only the data file for one scene.
    This prevents ped_id collisions when models read all files in a directory.

    Returns: (tmp_base_dir, needs_cleanup)
    - Use tmp_base_dir as base_dir parameter for predict functions.
    - The predict function will access tmp_base_dir/dataset_name/phase/ internally.
    """
    data_dir = os.path.join(base_dir, dataset_name, phase)
    if not os.path.exists(data_dir):
        return base_dir, False

    # Find the file matching this scene
    all_files = os.listdir(data_dir)
    scene_file = None
    for f in all_files:
        fname_no_ext = f.replace(f'_{phase}.txt', '').replace('_train.txt', '').replace('_val.txt', '').replace('_test.txt', '').replace('.txt', '')
        if fname_no_ext == scene_name:
            scene_file = f
            break

    if scene_file is None:
        return base_dir, False

    # Create: tmp_base/dataset_name/phase/ with symlink to single scene file
    tmp_base = tempfile.mkdtemp(prefix=f"expert_scene_{scene_name}_")
    tmp_data_dir = os.path.join(tmp_base, dataset_name, phase)
    os.makedirs(tmp_data_dir, exist_ok=True)
    os.symlink(os.path.join(data_dir, scene_file), os.path.join(tmp_data_dir, scene_file))
    return tmp_base, True


def _save_expert_predictions(expert_predictions, save_path, dataset_name, phase):
    """
    
    Args:
        expert_predictions: dict - {model_name: {ped_id: prediction}}
        phase: str - 'train', 'val', 'test'
    
    
    """
    try:
        os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {save_path}: {e}")
    file_path = os.path.join(save_path, f"expert_predictions_{dataset_name}_{phase}.pkl")
    temp_file_path = file_path + ".tmp"
    
    predictions_to_save = {}
    for model_name, pred_dict in expert_predictions.items():
        predictions_to_save[model_name] = {}
        for ped_id, pred in pred_dict.items():
            if isinstance(pred, torch.Tensor):
                predictions_to_save[model_name][ped_id] = pred.cpu().numpy()
            else:
                predictions_to_save[model_name][ped_id] = pred
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(temp_file_path, 'wb') as f:
                pickle.dump(predictions_to_save, f)
            
            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                raise RuntimeError(f"Temporary file {temp_file_path} is empty or does not exist")
            
            if os.path.exists(file_path):
                os.remove(file_path)
            os.rename(temp_file_path, file_path)
            
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                raise RuntimeError(f"Final file {file_path} is empty or does not exist")
            
            print(f"[EXPERT SAVE] Successfully saved expert predictions to {file_path} (size: {os.path.getsize(file_path)} bytes)")
            return file_path
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[EXPERT SAVE] Attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(1.0)
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
            else:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
                raise RuntimeError(f"Failed to save expert predictions to {file_path} after {max_retries} attempts: {e}")


def precompute_expert_predictions(
    raw_datasets,
    singulartrajectory_model,
    dmrgcn_model,
    gpgraph_model,
    dataset_name,
    cfg,
    hyper_params,
    accelerator,
    phase='train',
    stgcnn_model=None,
    expert_traj_model=None,  # Expert Trajectory (Goal-Example Model)
):
    """
    
    Args:
        hyper_params: SingularTrajectory hyper_params
    
    Returns:
        expert_predictions: dict - {model_name: {ped_id: prediction}}
    """
    if not accelerator.is_local_main_process:
        return
    
    print(f"\n{'='*80}")
    print(f"[EXPERT PRECOMPUTE] Starting expert predictions precomputation for phase: {phase}")
    print(f"[EXPERT PRECOMPUTE] Using K=5 experts: STGCNN, DMRGCN, GPGraph, SingularTrajectory, ExpertTraj")
    print(f"{'='*80}\n")
    
    print(f"[EXPERT PRECOMPUTE] Step 1/6: Collecting all original_ped_ids from {phase} dataset...")
    
    save_path = getattr(cfg, 'expert_tmp_path', None)
    ped_ids_cache_path = None
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ped_ids_cache_path = os.path.join(save_path, f"ped_ids_{dataset_name}_{phase}.pkl")
    
    all_ped_ids = None
    if ped_ids_cache_path and os.path.exists(ped_ids_cache_path):
        try:
            with open(ped_ids_cache_path, 'rb') as f:
                all_ped_ids = pickle.load(f)
            print(f"[EXPERT PRECOMPUTE] Loaded {len(all_ped_ids)} ped_ids from cache: {ped_ids_cache_path}")
        except Exception as e:
            print(f"[EXPERT PRECOMPUTE] WARNING: Failed to load ped_ids cache: {e}, will collect from dataset")
            all_ped_ids = None
    
    all_ped_id_frame_pairs = None
    all_ped_id_scene_pairs = None

    if all_ped_ids is None:
        all_ped_ids_set = set()
        all_ped_id_frame_set = set()
        all_ped_id_scene_set = set()
        dataset = raw_datasets[phase] if phase in raw_datasets else raw_datasets["train"]

        batch_size_collect = 1000
        num_samples = len(dataset)
        num_batches_collect = (num_samples + batch_size_collect - 1) // batch_size_collect

        for i in tqdm(range(num_batches_collect), desc=f"Collecting ped_ids from {phase}"):
            start_idx = i * batch_size_collect
            end_idx = min((i + 1) * batch_size_collect, num_samples)
            batch_data = dataset[start_idx:end_idx]

            if "original_ped_id" in batch_data:
                ped_ids_batch = batch_data["original_ped_id"]
                frames_batch = batch_data.get("frame", None)
                scenes_batch = batch_data.get("scene", None)
                if isinstance(ped_ids_batch, list):
                    for idx, ped_id in enumerate(ped_ids_batch):
                        pid = float(ped_id) if not isinstance(ped_id, list) else float(ped_id[0])
                        all_ped_ids_set.add(pid)
                        if frames_batch is not None:
                            frame_val = frames_batch[idx]
                            fval = float(frame_val) if not isinstance(frame_val, list) else float(frame_val[0])
                            all_ped_id_frame_set.add((pid, fval))
                        if scenes_batch is not None:
                            scene_val = scenes_batch[idx]
                            all_ped_id_scene_set.add((pid, scene_val))

        all_ped_ids = sorted(list(all_ped_ids_set))
        all_ped_id_frame_pairs = sorted(list(all_ped_id_frame_set))
        all_ped_id_scene_pairs = sorted(list(all_ped_id_scene_set))
        print(f"[EXPERT PRECOMPUTE] Found {len(all_ped_ids)} unique ped_ids, "
              f"{len(all_ped_id_frame_pairs)} (ped_id, frame) pairs, "
              f"{len(all_ped_id_scene_pairs)} (ped_id, scene) pairs")

        if ped_ids_cache_path:
            try:
                with open(ped_ids_cache_path, 'wb') as f:
                    pickle.dump(all_ped_ids, f)
                frame_cache_path = ped_ids_cache_path.replace('ped_ids_', 'ped_id_frame_pairs_')
                with open(frame_cache_path, 'wb') as f:
                    pickle.dump(all_ped_id_frame_pairs, f)
                scene_cache_path = ped_ids_cache_path.replace('ped_ids_', 'ped_id_scene_pairs_')
                with open(scene_cache_path, 'wb') as f:
                    pickle.dump(all_ped_id_scene_pairs, f)
                print(f"[EXPERT PRECOMPUTE] Saved ped_ids, frame pairs, and scene pairs to cache")
            except Exception as e:
                print(f"[EXPERT PRECOMPUTE] WARNING: Failed to save cache: {e}")
    else:
        if ped_ids_cache_path:
            frame_cache_path = ped_ids_cache_path.replace('ped_ids_', 'ped_id_frame_pairs_')
            if os.path.exists(frame_cache_path):
                try:
                    with open(frame_cache_path, 'rb') as f:
                        all_ped_id_frame_pairs = pickle.load(f)
                    print(f"[EXPERT PRECOMPUTE] Loaded {len(all_ped_id_frame_pairs)} (ped_id, frame) pairs from cache")
                except:
                    pass
            scene_cache_path = ped_ids_cache_path.replace('ped_ids_', 'ped_id_scene_pairs_')
            if os.path.exists(scene_cache_path):
                try:
                    with open(scene_cache_path, 'rb') as f:
                        all_ped_id_scene_pairs = pickle.load(f)
                    print(f"[EXPERT PRECOMPUTE] Loaded {len(all_ped_id_scene_pairs)} (ped_id, scene) pairs from cache")
                except:
                    pass
        if all_ped_id_frame_pairs is None or all_ped_id_scene_pairs is None:
            all_ped_id_frame_set = set()
            all_ped_id_scene_set = set()
            dataset = raw_datasets[phase] if phase in raw_datasets else raw_datasets["train"]
            for i in range(len(dataset)):
                sample = dataset[i]
                pid = float(sample["original_ped_id"])
                if "frame" in sample:
                    fval = float(sample["frame"])
                    all_ped_id_frame_set.add((pid, fval))
                if "scene" in sample:
                    all_ped_id_scene_set.add((pid, sample["scene"]))
            if all_ped_id_frame_pairs is None:
                all_ped_id_frame_pairs = sorted(list(all_ped_id_frame_set))
            if all_ped_id_scene_pairs is None:
                all_ped_id_scene_pairs = sorted(list(all_ped_id_scene_set))
            print(f"[EXPERT PRECOMPUTE] Collected {len(all_ped_id_frame_pairs)} (ped_id, frame), "
                  f"{len(all_ped_id_scene_pairs)} (ped_id, scene) pairs from dataset")

    print(f"[EXPERT PRECOMPUTE] Using {len(all_ped_ids)} unique ped_ids, "
          f"{len(all_ped_id_scene_pairs) if all_ped_id_scene_pairs else 0} (ped_id, scene) pairs\n")
    
    if len(all_ped_ids) == 0:
        print(f"[EXPERT PRECOMPUTE] WARNING: No ped_ids found! Skipping precomputation.")
        accelerator.wait_for_everyone()
        return
    
    batch_size = 256
    num_batches = (len(all_ped_ids) + batch_size - 1) // batch_size

    device = accelerator.device

    scene_to_ped_ids = {}
    if all_ped_id_scene_pairs:
        for pid, scene in all_ped_id_scene_pairs:
            if scene not in scene_to_ped_ids:
                scene_to_ped_ids[scene] = []
            scene_to_ped_ids[scene].append(pid)
        for scene in scene_to_ped_ids:
            scene_to_ped_ids[scene] = sorted(list(set(scene_to_ped_ids[scene])))
        print(f"[EXPERT PRECOMPUTE] Scene-based grouping: {', '.join(f'{s}({len(p)})' for s, p in scene_to_ped_ids.items())}")
    expert_predictions = {
        'singular': {},
        'dmrgcn': {},
        'gpgraph': {},
        'stgcnn': {},
        'expert_traj': {},  # Goal-Example Model
    }
    
    saved_file_path = None
    
    use_scene_isolation = bool(scene_to_ped_ids)

    try:
        if dmrgcn_model is not None:
            print(f"[EXPERT PRECOMPUTE] Step 2/6: Computing DMRGCN predictions (scene_isolation={use_scene_isolation})...")
            workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
            dmrgcn_base_dir = getattr(cfg, 'dmrgcn_base_dir', None) or os.path.join(workspace_dir, 'DMRGCN', 'datasets')
            try:
                if use_scene_isolation:
                    for scene_name, scene_ped_ids in scene_to_ped_ids.items():
                        tmp_base, needs_cleanup = _create_scene_data_dir(dmrgcn_base_dir, dataset_name, phase, scene_name)
                        scene_num_batches = (len(scene_ped_ids) + batch_size - 1) // batch_size
                        try:
                            for i in range(scene_num_batches):
                                start_idx = i * batch_size
                                end_idx = min((i + 1) * batch_size, len(scene_ped_ids))
                                batch_ped_ids = scene_ped_ids[start_idx:end_idx]
                                batch = {"original_ped_id": torch.tensor(batch_ped_ids, dtype=torch.float32)}
                                preds = predict_dmrgcn_trajectories(
                                    batch, dmrgcn_model=dmrgcn_model,
                                    dataset_name=dataset_name, phase=phase,
                                    obs_len=cfg.obs_len, pred_len=cfg.pred_len,
                                    base_dir=tmp_base if needs_cleanup else dmrgcn_base_dir,
                                    n_samples=20, device=device, verbose=False
                                )
                                if preds is not None:
                                    for j, ped_id in enumerate(batch_ped_ids):
                                        if j < len(preds) and preds[j] is not None:
                                            expert_predictions['dmrgcn'][(float(ped_id), scene_name)] = preds[j]
                        finally:
                            if needs_cleanup:
                                shutil.rmtree(tmp_base, ignore_errors=True)
                        # Clear internal cache between scenes to avoid stale data
                        from utils.dmrgcn_utils import _DMRGCN_CACHE, _EXPERT_DMRGCN_PREDICTIONS_CACHE
                        _DMRGCN_CACHE.clear()
                        _EXPERT_DMRGCN_PREDICTIONS_CACHE.clear()
                else:
                    for i in tqdm(range(num_batches), desc="DMRGCN"):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, len(all_ped_ids))
                        batch_ped_ids = all_ped_ids[start_idx:end_idx]
                        batch = {"original_ped_id": torch.tensor(batch_ped_ids, dtype=torch.float32)}
                        try:
                            preds = predict_dmrgcn_trajectories(
                                batch, dmrgcn_model=dmrgcn_model,
                                dataset_name=dataset_name, phase=phase,
                                obs_len=cfg.obs_len, pred_len=cfg.pred_len,
                                base_dir=dmrgcn_base_dir, n_samples=20, device=device, verbose=False
                            )
                            if preds is not None and len(preds) > 0:
                                for j, ped_id in enumerate(batch_ped_ids):
                                    if j < len(preds) and preds[j] is not None:
                                        expert_predictions['dmrgcn'][float(ped_id)] = preds[j]
                        except Exception as e:
                            import traceback
                            print(f"[EXPERT PRECOMPUTE] ERROR: DMRGCN prediction failed for batch {i}: {e}")
                            traceback.print_exc()
                            continue
            except Exception as e:
                import traceback
                print(f"[EXPERT PRECOMPUTE] CRITICAL ERROR: DMRGCN prediction loop failed: {e}")
                traceback.print_exc()

            print(f"[EXPERT PRECOMPUTE] ✓ DMRGCN predictions completed ({len(expert_predictions['dmrgcn'])} predictions)\n")
        else:
            print(f"[EXPERT PRECOMPUTE] Step 2/6: DMRGCN model not enabled, skipping...\n")
        
        def _predict_per_scene(model_name, predict_fn, predict_kwargs_fn, base_dir, cache_modules=None):
            """Run per-scene prediction with isolated dirs. predict_kwargs_fn(batch, tmp_base) -> kwargs dict."""
            total = 0
            if use_scene_isolation:
                for scene_name, scene_ped_ids in scene_to_ped_ids.items():
                    tmp_base, needs_cleanup = _create_scene_data_dir(base_dir, dataset_name, phase, scene_name)
                    scene_num_batches = (len(scene_ped_ids) + batch_size - 1) // batch_size
                    try:
                        for i in range(scene_num_batches):
                            s = i * batch_size
                            e = min((i + 1) * batch_size, len(scene_ped_ids))
                            bpids = scene_ped_ids[s:e]
                            batch = {"original_ped_id": torch.tensor(bpids, dtype=torch.float32)}
                            try:
                                preds = predict_fn(**predict_kwargs_fn(batch, tmp_base if needs_cleanup else base_dir))
                                if preds is not None:
                                    for j, pid in enumerate(bpids):
                                        if j < len(preds) and preds[j] is not None:
                                            expert_predictions[model_name][(float(pid), scene_name)] = preds[j]
                                            total += 1
                            except Exception as ex:
                                import traceback
                                traceback.print_exc()
                                continue
                    finally:
                        if needs_cleanup:
                            shutil.rmtree(tmp_base, ignore_errors=True)
                    # Clear internal caches between scenes
                    if cache_modules:
                        for cm in cache_modules:
                            cm.clear()
            else:
                for i in tqdm(range(num_batches), desc=model_name):
                    s = i * batch_size
                    e = min((i + 1) * batch_size, len(all_ped_ids))
                    bpids = all_ped_ids[s:e]
                    batch = {"original_ped_id": torch.tensor(bpids, dtype=torch.float32)}
                    try:
                        preds = predict_fn(**predict_kwargs_fn(batch, base_dir))
                        if preds is not None:
                            for j, pid in enumerate(bpids):
                                if j < len(preds) and preds[j] is not None:
                                    expert_predictions[model_name][float(pid)] = preds[j]
                                    total += 1
                    except Exception as ex:
                        import traceback
                        traceback.print_exc()
                        continue
            return total

        if singulartrajectory_model is not None:
            print(f"[EXPERT PRECOMPUTE] Step 3/6: Computing SingularTrajectory predictions...")
            workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
            sig_base_dir = getattr(cfg, 'sig_base_dir', None) or os.path.join(workspace_dir, 'SingularTrajectory', 'datasets')
            from utils.sig_utils import _ped_trajectories_full_cache, _EXPERT_SIG_PREDICTIONS_CACHE
            n = _predict_per_scene(
                'singular',
                predict_SingularTrajectory_trajectories,
                lambda batch, bd: dict(
                    hyper_params=hyper_params, device=device, batch=batch,
                    model=singulartrajectory_model, dataset_name=dataset_name, phase=phase,
                    base_dir=bd, data_dir=None, obs_len=8, pred_len=12
                ),
                sig_base_dir,
                cache_modules=[_ped_trajectories_full_cache, _EXPERT_SIG_PREDICTIONS_CACHE]
            )
            print(f"[EXPERT PRECOMPUTE] ✓ SingularTrajectory: {n} predictions\n")
        else:
            print(f"[EXPERT PRECOMPUTE] Step 3/6: SingularTrajectory not enabled, skipping...\n")

        if gpgraph_model is not None:
            print(f"[EXPERT PRECOMPUTE] Step 4/6: Computing GPGraph predictions...")
            workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
            gpgraph_base_dir = getattr(cfg, 'gpgraph_base_dir', None) or os.path.join(workspace_dir, 'GPGraph', 'dataset')
            gpgraph_n_samples = getattr(cfg, 'gpgraph_n_samples', None) or 20
            gpgraph_use_simple_sampler = getattr(cfg, 'gpgraph_use_simple_sampler', False) or False
            from utils.gpgraph_utils import _GPGRAPH_CACHE, _EXPERT_GPGRAPH_PREDICTIONS_CACHE
            n = _predict_per_scene(
                'gpgraph',
                predict_gpgraph_trajectories,
                lambda batch, bd: dict(
                    batch=batch, gpgraph_model=gpgraph_model,
                    dataset_name=dataset_name, phase=phase,
                    obs_len=cfg.obs_len, pred_len=cfg.pred_len,
                    base_dir=bd, n_samples=gpgraph_n_samples, device=device,
                    verbose=False, use_simple_sampler=gpgraph_use_simple_sampler
                ),
                gpgraph_base_dir,
                cache_modules=[_GPGRAPH_CACHE, _EXPERT_GPGRAPH_PREDICTIONS_CACHE]
            )
            print(f"[EXPERT PRECOMPUTE] ✓ GPGraph: {n} predictions\n")
        else:
            print(f"[EXPERT PRECOMPUTE] Step 4/6: GPGraph not enabled, skipping...\n")

        if stgcnn_model is not None:
            print(f"[EXPERT PRECOMPUTE] Step 5/6: Computing Social-STGCNN predictions...")
            workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
            stgcnn_base_dir = getattr(cfg, 'stgcnn_base_dir', None) or os.path.join(workspace_dir, 'Social-STGCNN', 'datasets')
            stgcnn_n_samples = getattr(cfg, 'stgcnn_n_samples', 20)
            from utils.stgcnn_utils import _STGCNN_CACHE, _EXPERT_STGCNN_PREDICTIONS_CACHE
            n = _predict_per_scene(
                'stgcnn',
                predict_stgcnn_trajectories,
                lambda batch, bd: dict(
                    batch=batch, stgcnn_model=stgcnn_model,
                    dataset_name=dataset_name, phase=phase,
                    obs_len=cfg.obs_len, pred_len=cfg.pred_len,
                    base_dir=bd, n_samples=stgcnn_n_samples, device=accelerator.device,
                    verbose=False
                ),
                stgcnn_base_dir,
                cache_modules=[_STGCNN_CACHE, _EXPERT_STGCNN_PREDICTIONS_CACHE]
            )
            print(f"[EXPERT PRECOMPUTE] ✓ Social-STGCNN: {n} predictions\n")
        else:
            print(f"[EXPERT PRECOMPUTE] Step 5/6: Social-STGCNN not enabled, skipping...\n")

        if expert_traj_model is not None:
            print(f"[EXPERT PRECOMPUTE] Step 6/6: Computing Expert Trajectory predictions...")
            workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
            expert_traj_base_dir = getattr(cfg, 'expert_traj_base_dir', None) or os.path.join(workspace_dir, 'expert_traj', 'datasets')
            expert_traj_model_path = getattr(cfg, 'expert_traj_model_path', None) or os.path.join(workspace_dir, 'expert_traj', 'checkpoint_ethucy')
            expert_traj_n_samples = getattr(cfg, 'expert_traj_n_samples', None) or 20

            # ETH/UCY: direct DataLoader-based approach (like test_ethucy.py)
            expert_goals_path = os.path.join(
                expert_traj_model_path, f"test_{dataset_name}_expert.npy"
            )
            try:
                direct_preds = predict_expert_traj_all(
                    expert_traj_model, dataset_name, phase, device,
                    base_dir=expert_traj_base_dir,
                    expert_goals_path=expert_goals_path,
                    obs_len=cfg.obs_len, pred_len=cfg.pred_len,
                    n_samples=expert_traj_n_samples,
                    use_gt_goals=(phase == 'train'),  # GT goals for training data
                )
                for key, pred in direct_preds.items():
                    expert_predictions['expert_traj'][key] = pred
            except Exception as e:
                import traceback
                traceback.print_exc()

            print(f"[EXPERT PRECOMPUTE] ✓ Expert Trajectory: {len(expert_predictions['expert_traj'])} predictions\n")
        else:
            print(f"[EXPERT PRECOMPUTE] Step 6/6: Expert Trajectory not enabled, skipping...\n")

    except Exception as e:
        import traceback
        print(f"[EXPERT PRECOMPUTE] ERROR: Exception during prediction computation: {e}")
        traceback.print_exc()
        print(f"[EXPERT PRECOMPUTE] Will attempt to save computed predictions anyway...")
    
    finally:
        save_path = getattr(cfg, 'expert_tmp_path', None)
        if save_path:
            try:
                all_ped_ids_set = set(all_ped_ids)
                missing_predictions = {}

                if dmrgcn_model is not None:
                    dmrgcn_ped_ids = set(expert_predictions['dmrgcn'].keys())
                    missing_dmrgcn = all_ped_ids_set - dmrgcn_ped_ids
                    if missing_dmrgcn:
                        missing_predictions['dmrgcn'] = sorted(list(missing_dmrgcn))

                if singulartrajectory_model is not None:
                    singular_ped_ids = set(expert_predictions['singular'].keys())
                    missing_singular = all_ped_ids_set - singular_ped_ids
                    if missing_singular:
                        missing_predictions['singular'] = sorted(list(missing_singular))

                if gpgraph_model is not None:
                    gpgraph_ped_ids = set(expert_predictions['gpgraph'].keys())
                    missing_gpgraph = all_ped_ids_set - gpgraph_ped_ids
                    if missing_gpgraph:
                        missing_predictions['gpgraph'] = sorted(list(missing_gpgraph))

                if stgcnn_model is not None:
                    stgcnn_ped_ids = set(expert_predictions['stgcnn'].keys())
                    missing_stgcnn = all_ped_ids_set - stgcnn_ped_ids
                    if missing_stgcnn:
                        missing_predictions['stgcnn'] = sorted(list(missing_stgcnn))

                if expert_traj_model is not None:
                    expert_traj_ped_ids = set(expert_predictions['expert_traj'].keys())
                    missing_expert_traj = all_ped_ids_set - expert_traj_ped_ids
                    if missing_expert_traj:
                        missing_predictions['expert_traj'] = sorted(list(missing_expert_traj))

                if missing_predictions:
                    warn_msg = f"[EXPERT PRECOMPUTE] WARNING: Incomplete predictions - missing predictions for:\n"
                    for model_name, missing_ids in missing_predictions.items():
                        warn_msg += f"  {model_name}: {len(missing_ids)} missing ped_ids (first 10: {missing_ids[:10]})\n"
                    print(warn_msg)
                    # Report which models succeeded
                    complete_models = []
                    for model_name, pred_dict in expert_predictions.items():
                        if len(pred_dict) == len(all_ped_ids_set):
                            complete_models.append(f"{model_name}({len(pred_dict)})")
                    if complete_models:
                        print(f"[EXPERT PRECOMPUTE] ✓ Complete models: {', '.join(complete_models)}")
                    print(f"[EXPERT PRECOMPUTE] Saving partial predictions (available models will work for ablation)...")

                has_predictions = any(len(pred_dict) > 0 for pred_dict in expert_predictions.values())
                if has_predictions:
                    saved_file_path = _save_expert_predictions(expert_predictions, save_path, dataset_name, phase)
                    if not os.path.exists(saved_file_path) or os.path.getsize(saved_file_path) == 0:
                        raise RuntimeError(f"Saved file {saved_file_path} is empty or does not exist")
                    print(f"[EXPERT PRECOMPUTE] ✓ Expert predictions saved successfully to {saved_file_path}")
                else:
                    print(f"[EXPERT PRECOMPUTE] WARNING: No predictions computed, skipping save")
            except Exception as e:
                import traceback
                print(f"[EXPERT PRECOMPUTE] ERROR: Failed to save expert predictions: {e}")
                traceback.print_exc()
                raise RuntimeError(f"Expert predictions saving failed: {e}")
    
    print(f"{'='*80}")
    print(f"[EXPERT PRECOMPUTE] All expert predictions precomputed for {phase} phase!")
    if saved_file_path:
        print(f"[EXPERT PRECOMPUTE] Saved file: {saved_file_path}")
    print(f"{'='*80}\n")
    
    accelerator.wait_for_everyone()

    return expert_predictions
