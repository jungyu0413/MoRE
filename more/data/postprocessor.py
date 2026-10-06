"""Trajectory post-processing: wall avoidance, abnormal movement filtering, clustering."""

import warnings
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from more.data.homography import world2image


def postprocess_trajectory(traj, obs_traj, seq_start_end, scene_id,
                           homography, scene_map, cfg):
    """Post-process predicted trajectories.

    Handles wall collision avoidance (scaling trajectories toward valid regions),
    abnormal movement filtering, and KMeans clustering for stochastic predictions.

    Config parameters used:
        cfg.deterministic: Whether deterministic (single sample) mode.
        cfg.best_of_n: Number of samples to keep after clustering.
        cfg.pred_len: Prediction horizon T_pred.
        cfg.image_scale_down: Scale factor for scene map lookup.
        cfg.dataset_name: Dataset name for ETH-UCY special handling.
        cfg.abnormal_thresh: Abnormal movement threshold (default 100).
        cfg.wall_avoidance_steps: Number of scale steps for wall avoidance (default 100).
    """

    abnormal_thresh = getattr(cfg, 'abnormal_thresh', 100)
    wall_avoidance_steps = getattr(cfg, 'wall_avoidance_steps', 100)
    wall_avoidance_min_scale = getattr(cfg, 'wall_avoidance_min_scale', 0.01)
    image_scale_down = getattr(cfg, 'image_scale_down', 0.25)

    def get_value(S, i, j):
        try:
            return S[i, j]
        except IndexError:
            return 0

    def obs_to_image(ped_id):
        return world2image(obs_traj[ped_id], homography[scene_id[ped_id]])

    if cfg.deterministic:
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess")):
            map_temp = scene_map.get(scene_id[s]) if isinstance(scene_map, dict) else scene_map[scene_id[s]]
            if map_temp is None:
                continue
            for ped_id in range(s, e):
                if ped_id >= traj.shape[0]:
                    continue
                sample = 0
                endpoint = (traj[ped_id, sample, -1] / image_scale_down).astype(np.int32)
                if get_value(map_temp, endpoint[1], endpoint[0]) != 1:
                    obs_traj_temp = obs_to_image(ped_id)
                    startpoint = obs_traj_temp[-1].copy()

                    scale = np.linspace(1.0, wall_avoidance_min_scale, wall_avoidance_steps)
                    traj_temp = traj[ped_id, sample].copy() - startpoint
                    traj_temps = np.tile(traj_temp[None, :, :], [len(scale), 1, 1])
                    traj_temps *= np.tile(scale[:, None, None], [1, *traj[ped_id, sample].shape])
                    traj_temps += startpoint
                    endpoints = (traj_temps[:, -1] / image_scale_down).astype(np.int32)

                    for i in range(len(endpoints)):
                        if get_value(map_temp, endpoints[i, 1], endpoints[i, 0]) == 1:
                            break
                    traj[ped_id, sample] = traj_temps[i]
    else:
        new_traj = np.zeros([traj.shape[0], cfg.best_of_n, cfg.pred_len, 2])
        for s_id, (s, e) in enumerate(tqdm(seq_start_end, desc="Postprocess")):
            map_temp = scene_map.get(scene_id[s]) if isinstance(scene_map, dict) else scene_map[scene_id[s]]

            for ped_id in range(s, e):
                if ped_id >= traj.shape[0]:
                    continue

                if cfg.metric == 'pixel' and scene_id[ped_id] in homography:
                    startpoint = world2image(obs_traj[ped_id], homography[scene_id[ped_id]])[-1].copy()
                else:
                    startpoint = obs_traj[ped_id, -1].copy()

                # Filter abnormal movements (displacement > abnormal_thresh per step)
                mask = np.diff(traj[ped_id, :, :, :], n=1, axis=1)
                mask = np.linalg.norm(mask, ord=2, axis=-1)
                mask = np.any(np.greater(mask, abnormal_thresh), axis=1)
                traj_filtered = traj[ped_id].copy()
                traj_filtered[mask, :, 0] = startpoint[0]
                traj_filtered[mask, :, 1] = startpoint[1]
                max_samples_filtered = traj_filtered.shape[0]

                # Wall avoidance (only when scene map is available)
                if map_temp is not None:
                    obs_traj_temp = obs_to_image(ped_id)
                    startpoint = obs_traj_temp[-1].copy()

                    for sample in range(max_samples_filtered):
                        endpoint = (traj_filtered[sample, -1] / image_scale_down).astype(np.int32)
                        if get_value(map_temp, endpoint[1], endpoint[0]) != 1:
                            scale = np.linspace(1.0, wall_avoidance_min_scale, wall_avoidance_steps)
                            traj_temp = traj_filtered[sample].copy() - startpoint
                            traj_temps = np.tile(traj_temp[None, :, :], [len(scale), 1, 1])
                            traj_temps *= np.tile(scale[:, None, None], [1, cfg.pred_len, 2])
                            traj_temps += startpoint
                            endpoints = (traj_temps[:, -1] / image_scale_down).astype(np.int32)

                            for i in range(len(endpoints)):
                                if get_value(map_temp, endpoints[i, 1], endpoints[i, 0]) == 1:
                                    break
                            traj_filtered[sample] = traj_temps[i]

                # Clustering: always reduce to best_of_n diverse samples via KMeans
                if max_samples_filtered > cfg.best_of_n:
                    temp = traj_filtered.reshape(max_samples_filtered, -1)
                    centroids = KMeans(n_clusters=cfg.best_of_n, random_state=0,
                                      init='k-means++', n_init=1).fit(temp).cluster_centers_
                    traj_filtered = centroids.reshape(cfg.best_of_n, cfg.pred_len, -1)

                new_traj[ped_id, :traj_filtered.shape[0]] = traj_filtered

        traj = new_traj
    return traj
