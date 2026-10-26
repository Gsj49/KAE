# Reproducibility

- [environment.txt](environment.txt): Isambard GH200 software versions and Slurm environment.
- [MSE_REPRODUCTION.md](MSE_REPRODUCTION.md): saved models, samples and commands
  for the estimation experiments.

Installation and the main entry points are described in the
[repository README](../README.md). The code is based on official VERL commit
`4da0d3d3188072772cb2ec817b3d6cf4a463821f` and retains its Apache-2.0 license.
The reference system is the Isambard cluster, using Slurm and NVIDIA GH200
GPUs on Linux/aarch64. Set up the matching CUDA/PyTorch/vLLM environment, then
install this checkout with `pip install -e . --no-deps`.

[data/README.md](../data/README.md) describes the prepared datasets and saved
samples. Optional file inventories are stored in `data/*_assets.json`.
Checkpoint records and sampled completion indices used
by gradient MSE are in `experiments/gradient_mse/`.

[experiments/COVERAGE.md](../experiments/COVERAGE.md) lists experiment entry
points and their configurations.
