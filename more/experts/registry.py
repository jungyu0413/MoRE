"""Expert model registry: dynamic loading of external trajectory prediction models.

Each expert model is loaded from an external repository. Users should clone the
required repos and set the paths in the config file.

The five expert types defined in the paper (Section 3.2):
  (i)   local interaction   → Social-STGCNN  : https://github.com/abduallahmohamed/Social-STGCNN
  (ii)  structural relation → DMRGCN (AAAI 2021) : https://github.com/InhwanBae/DMRGCN
  (iii) social grouping     → GPGraph (ECCV 2022) : https://github.com/InhwanBae/GPGraph
  (iv)  motion pattern      → SingularTrajectory (CVPR 2024): https://github.com/InhwanBae/SingularTrajectory
  (v)   goal expert         → ExpertTraj (ICCV 2021): https://github.com/JoeHEZHAO/expert_traj
"""

import os
import sys
import pickle
import logging

import torch

logger = logging.getLogger(__name__)


_EXTERNAL_MODULE_NAMES = (
    'model', 'models', 'utils', 'baseline', 'SingularTrajectory',
    'model_baseline', 'model_groupwrapper', 'model_lstm',
)


def _swap_sys_path(target_dir):
    """Temporarily swap sys.path to import from an external repo."""
    original_path = sys.path[:]
    saved_modules = {}
    for mod_name in list(sys.modules.keys()):
        # Save and remove known external modules and their sub-modules
        if mod_name in _EXTERNAL_MODULE_NAMES or any(
            mod_name.startswith(base + '.') for base in _EXTERNAL_MODULE_NAMES
        ):
            saved_modules[mod_name] = sys.modules.pop(mod_name)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    return original_path, saved_modules


def _restore_sys_path(original_path, saved_modules):
    """Restore sys.path after importing from an external repo."""
    # Remove any newly imported external modules before restoring saved ones
    for mod_name in list(sys.modules.keys()):
        if mod_name in _EXTERNAL_MODULE_NAMES or any(
            mod_name.startswith(base + '.') for base in _EXTERNAL_MODULE_NAMES
        ):
            if mod_name not in saved_modules:
                del sys.modules[mod_name]
    for mod_name, mod_obj in saved_modules.items():
        sys.modules[mod_name] = mod_obj
    sys.path[:] = original_path


def load_gpgraph(cfg, device):
    """Load GPGraph model from external repo."""
    workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
    gpgraph_dir = os.path.join(workspace_dir, 'GPGraph')

    original_path, saved_modules = _swap_sys_path(gpgraph_dir)

    from model_baseline import TrajectoryModel as GPGraphTrajectoryModel
    from model_groupwrapper import GPGraph

    _restore_sys_path(original_path, saved_modules)

    base_model = GPGraphTrajectoryModel(
        number_asymmetric_conv_layer=7, embedding_dims=64,
        number_gcn_layers=1, dropout=0,
        obs_len=cfg.obs_len, pred_len=cfg.pred_len, n_tcn=5, out_dims=5,
    ).to(device)

    model = GPGraph(
        baseline_model=base_model, in_channels=2, out_channels=5,
        obs_seq_len=cfg.obs_len, pred_seq_len=cfg.pred_len,
        d_type='learned_l2norm', d_th='learned',
        mix_type='mlp', group_type=(True, True, True), weight_share=True,
    ).to(device)

    model_path = getattr(cfg, 'gpgraph_model_path', None)
    if model_path:
        model_path = model_path.replace("dataset_name", cfg.dataset_name)
    else:
        model_path = os.path.join(gpgraph_dir, 'checkpoints', 'GPGraph-SGCN',
                                  cfg.dataset_name, 'val_best.pth')

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"GPGraph loaded from {model_path}")
    else:
        logger.warning(f"GPGraph weights not found: {model_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_dmrgcn(cfg, device):
    """Load DMRGCN model from external repo."""
    workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
    dmrgcn_dir = os.path.join(workspace_dir, 'DMRGCN')

    original_path, saved_modules = _swap_sys_path(dmrgcn_dir)
    from model import social_dmrgcn
    _restore_sys_path(original_path, saved_modules)

    model = social_dmrgcn(
        n_stgcn=getattr(cfg, 'dmrgcn_n_stgcn', 1),
        n_tpcnn=getattr(cfg, 'dmrgcn_n_tpcnn', 4),
        output_feat=getattr(cfg, 'dmrgcn_output_feat', 5),
        kernel_size=getattr(cfg, 'dmrgcn_kernel_size', 3),
        seq_len=cfg.obs_len,
        pred_seq_len=cfg.pred_len,
    ).to(device)

    dmrgcn_model_path = getattr(cfg, 'dmrgcn_model_path', None)
    model_path = dmrgcn_model_path.replace("dataset_name", cfg.dataset_name) \
        if dmrgcn_model_path else os.path.join(dmrgcn_dir, 'checkpoints',
            f'social-dmrgcn-{cfg.dataset_name}-experiment_tp4_de80', f'{cfg.dataset_name}_best.pth')
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"DMRGCN loaded from {model_path}")
    else:
        logger.warning(f"DMRGCN weights not found: {model_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_singulartrajectory(cfg, device):
    """Load SingularTrajectory model from external repo."""
    workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
    st_dir = os.path.join(workspace_dir, 'SingularTrajectory')

    original_path, saved_modules = _swap_sys_path(st_dir)

    import baseline
    from SingularTrajectory import SingularTrajectory
    from utils import DotDict, get_exp_config

    # Resolve model path while SingularTrajectory's sys.path is active
    model_path = getattr(cfg, 'singulartrajectory_model_path', None)
    if model_path:
        model_path = model_path.replace("dataset_name", cfg.dataset_name)

    # Load config.pkl (needs SingularTrajectory's DotDict on sys.path)
    hyper_params = None
    if model_path and os.path.exists(model_path):
        config_pkl = os.path.join(os.path.dirname(model_path), 'config.pkl')
        if os.path.exists(config_pkl):
            with open(config_pkl, 'rb') as f:
                hyper_params = pickle.load(f)
                if not isinstance(hyper_params, DotDict):
                    hyper_params = DotDict(hyper_params)

    if hyper_params is None:
        task = getattr(cfg, 'singulartrajectory_task', 'stochastic')
        config_path = os.path.join(st_dir, 'config', task,
                                   f'singulartrajectory-transformerdiffusion-{cfg.dataset_name}.json')
        if os.path.exists(config_path):
            hyper_params = get_exp_config(config_path)
        else:
            hyper_params = DotDict({
                'obs_len': cfg.obs_len, 'pred_len': cfg.pred_len,
                'k': getattr(cfg, 'singulartrajectory_k', 4),
                'num_samples': getattr(cfg, 'singulartrajectory_num_samples', 20),
                'obs_svd': True, 'pred_svd': True,
                'traj_dim': 2, 'static_dist': getattr(cfg, 'singulartrajectory_static_dist', 0.4),
                'baseline': getattr(cfg, 'singulartrajectory_baseline', 'transformerdiffusion'),
            })

    baseline_cfg = DotDict({
        'scheduler': getattr(cfg, 'singulartrajectory_diffusion_scheduler', 'ddim'),
        'steps': getattr(cfg, 'singulartrajectory_diffusion_steps', 10),
        'beta_start': getattr(cfg, 'singulartrajectory_beta_start', 1e-4),
        'beta_end': getattr(cfg, 'singulartrajectory_beta_end', 5e-2),
        'beta_schedule': getattr(cfg, 'singulartrajectory_beta_schedule', 'linear'),
        'k': hyper_params.k, 's': hyper_params.num_samples,
    })
    PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
    predictor_model = PredictorModel(baseline_cfg)

    hook_func = DotDict({
        "model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
        "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
        "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook,
    })

    model = SingularTrajectory(
        baseline_model=predictor_model,
        hook_func=hook_func,
        hyper_params=hyper_params,
    )
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"SingularTrajectory loaded from {model_path}")

    # Restore sys.path AFTER all SingularTrajectory imports and pickle loads
    _restore_sys_path(original_path, saved_modules)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, hyper_params


def load_stgcnn(cfg, device):
    """Load Social-STGCNN model from external repo."""
    workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
    stgcnn_dir = os.path.join(workspace_dir, 'Social-STGCNN')

    original_path, saved_modules = _swap_sys_path(stgcnn_dir)
    from model import social_stgcnn
    _restore_sys_path(original_path, saved_modules)

    model_path_cfg = getattr(cfg, 'stgcnn_model_path', None)
    if model_path_cfg:
        checkpoint_dir = model_path_cfg.replace("dataset_name", cfg.dataset_name)
        if os.path.isdir(checkpoint_dir):
            model_path = os.path.join(checkpoint_dir, 'val_best.pth')
        else:
            model_path = checkpoint_dir
            checkpoint_dir = os.path.dirname(model_path)
    else:
        checkpoint_dir = os.path.join(stgcnn_dir, 'checkpoint', f'social-stgcnn-{cfg.dataset_name}')
        model_path = os.path.join(checkpoint_dir, 'val_best.pth')

    args_path = os.path.join(checkpoint_dir, 'args.pkl')
    if os.path.exists(args_path):
        with open(args_path, 'rb') as f:
            stgcnn_args = pickle.load(f)
    else:
        output_size = getattr(cfg, 'stgcnn_output_feat', None) or getattr(cfg, 'stgcnn_output_size', 5)
        stgcnn_args = type('Args', (), {
            'n_stgcnn': getattr(cfg, 'stgcnn_n_stgcnn', 1),
            'n_txpcnn': getattr(cfg, 'stgcnn_n_txpcnn', 4),
            'output_size': output_size,
            'obs_seq_len': cfg.obs_len, 'pred_seq_len': cfg.pred_len,
            'kernel_size': getattr(cfg, 'stgcnn_kernel_size', 3),
        })()

    model = social_stgcnn(
        n_stgcnn=stgcnn_args.n_stgcnn, n_txpcnn=stgcnn_args.n_txpcnn,
        output_feat=stgcnn_args.output_size,
        seq_len=stgcnn_args.obs_seq_len, pred_seq_len=stgcnn_args.pred_seq_len,
        kernel_size=stgcnn_args.kernel_size,
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Social-STGCNN loaded from {model_path}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_expert_traj(cfg, device):
    """Load Expert Trajectory (Goal-Example) model from external repo (ETH/UCY)."""
    workspace_dir = getattr(cfg, 'workspace_dir', '/workspace')
    expert_dir = os.path.join(workspace_dir, 'expert_traj')

    original_path, saved_modules = _swap_sys_path(expert_dir)

    from model_lstm import Goal_Example_Model
    _restore_sys_path(original_path, saved_modules)
    model = Goal_Example_Model(
        n_stgcnn=getattr(cfg, 'expert_traj_n_stgcnn', 1),
        n_txpcnn=getattr(cfg, 'expert_traj_n_txpcnn', 5),
        input_feat=getattr(cfg, 'expert_traj_input_feat', 6),
        output_feat=getattr(cfg, 'expert_traj_output_feat', 128),
        seq_len=cfg.obs_len,
        kernel_size=getattr(cfg, 'expert_traj_kernel_size', 3),
        pred_seq_len=cfg.pred_len,
    ).to(device)

    model_path = getattr(cfg, 'expert_traj_model_path', None) or os.path.join(expert_dir, 'checkpoint_ethucy')
    ckpt = os.path.join(model_path, f"{cfg.dataset_name}_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f"ExpertTraj loaded from {ckpt}")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ─── Convenience: load all enabled experts ───────────────────────────

EXPERT_LOADERS = {
    'gpgraph': ('use_gpgraph', load_gpgraph),
    'dmrgcn': ('use_dmrgcn', load_dmrgcn),
    'singulartrajectory': ('use_singulartrajectory', load_singulartrajectory),
    'stgcnn': ('use_stgcnn', load_stgcnn),
    'expert_traj': ('use_expert_traj', load_expert_traj),
}


def load_all_experts(cfg, device):
    """Load all enabled expert models based on config flags.

    Returns:
        experts: dict with keys matching expert names and values being the loaded models.
        extra: dict with auxiliary data (e.g., hyper_params for SingularTrajectory).
    """
    experts = {}
    extra = {}

    for name, (flag, loader) in EXPERT_LOADERS.items():
        if getattr(cfg, flag, False):
            logger.info(f"Loading expert: {name}")
            result = loader(cfg, device)
            if isinstance(result, tuple):
                experts[name] = result[0]
                extra[name] = result[1:]
            else:
                experts[name] = result
        else:
            logger.info(f"Expert {name} disabled ({flag}=False)")

    return experts, extra
