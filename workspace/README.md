# Expert models (`workspace/`)

MoRE uses five frozen numerical trajectory predictors as reward experts **during training only**
(evaluation does not need them). They are not bundled here. Clone each official repository into this
folder and use its pretrained ETH/UCY checkpoints. The folder location is `workspace_dir` in
`configs/default.json` (default `./workspace`).

```bash
cd workspace
git clone https://github.com/abduallahmohamed/Social-STGCNN.git
git clone https://github.com/InhwanBae/DMRGCN.git
git clone https://github.com/InhwanBae/GPGraph.git
git clone https://github.com/InhwanBae/SingularTrajectory.git
git clone https://github.com/JoeHEZHAO/expert_traj.git
```

| Expert (role in the paper) | Official repository | Checkpoint path MoRE loads (`<ds>` = eth/hotel/univ/zara1/zara2) | Config key to override |
|---|---|---|---|
| Social-STGCNN (local interaction) | https://github.com/abduallahmohamed/Social-STGCNN | `Social-STGCNN/checkpoint/social-stgcnn-<ds>/val_best.pth` (+ `args.pkl`) | `stgcnn_model_path` |
| DMRGCN (structural relation) | https://github.com/InhwanBae/DMRGCN | `DMRGCN/checkpoints/social-dmrgcn-<ds>-experiment_tp4_de80/<ds>_best.pth` | `dmrgcn_model_path` |
| GPGraph (social grouping) | https://github.com/InhwanBae/GPGraph | `GPGraph/checkpoints/GPGraph-SGCN/<ds>/val_best.pth` | `gpgraph_model_path` |
| SingularTrajectory (motion pattern) | https://github.com/InhwanBae/SingularTrajectory | `SingularTrajectory/checkpoints/SingularTrajectory-stochastic/<ds>/model_best.pth` (+ `config.pkl`) | `singulartrajectory_model_path` |
| Expert-Traj / Goal-Example (goal) | https://github.com/JoeHEZHAO/expert_traj | `expert_traj/checkpoint_ethucy/<ds>_best.pth` | `expert_traj_model_path` |

- Pretrained weights: follow each repository's README (GPGraph, Social-STGCNN and Expert-Traj ship them in the
  repository; DMRGCN and SingularTrajectory provide them through their release pages / download scripts).
  A `*_model_path` entry may contain the literal word `dataset_name`, which is replaced by the current dataset.
- The expert repositories' own `datasets/` folders (ETH/UCY in each repo's format) are read when expert
  predictions are precomputed (`scripts/prepare_experts.sh`).
- Experts can be switched off individually with `use_stgcnn`, `use_dmrgcn`, `use_gpgraph`,
  `use_singulartrajectory`, `use_expert_traj`.
