# Paper experiment entry points

Entry points for the paper's experiments.

| Paper experiment | Entry point | Configuration |
|---|---|---|
| One-shot illustration, Figure `fig:oneshot` | `train_scripts/prepare_oneshot.py`, `one_shot.py` | Three baselines, explicit copy/pool/history IDs, B128/G8/100 steps |
| Value MSE, Section `sec:exp-baseline-mse` | `experiments/run.py --experiment value_mse` | CPU comparison, saved responses, 100-response reference, GRPO/RF++/KAE plus JS/Kalman |
| Kernel/bandwidth sensitivity, Figure `fig:mse_kernel_sensitivity` | `experiments/kernel_sensitivity/`, common `run.py` entry | Paired draws, triangular/exponential grids, 100 repeats, full-batch RF++ LOO, hashes and plots |
| Gradient MSE, Section `sec:exp-gradvar` | `experiments/run.py --experiment gradient_mse` | MATH last 5 blocks + model norm + output layer; GSM8K all parameters; saved draws, full-batch RF++ LOO and shared forward |
| DAPO and MATH three-run policy tables | `train_scripts/train.py`, `test_scripts/evaluate_math.py` | Seven methods, common scheduler, three data seeds, mathematical evaluation/aggregation |
| G=1 policy curves, Figure `fig:qwen15b_n1_val_curves` | `train_scripts/single_stream.py` | Both 1.5B models, REINFORCE/KAE/scheduled REINFORCE, G1/B1024/400 steps |
| KAE with reward normalization | `train_scripts/train.py --method kae_norm` | KAE baseline with current-group std and fallback 0.5; same evaluation |
| HumanEval+/MBPP+ | `test_scripts/code/{generate,score,aggregate}.py` | Greedy, avg@16, pass@16; shared GPU generation and isolated parallel CPU scoring |
| Assumption 6 and KAE cross term | `experiments/run.py --experiment assumption6` | Full-parameter score norms, per-prompt correlation, optimal baseline/value difference, signed cross term |
| Monte Carlo reference error, Table `tab:MCSE` | `experiments/mc_error/run.py` | Included reference success counts; CPU calculation of mean, median and 90th percentile SE |
| Computing cost, Table `tab:response-resources` | `experiments/computing_cost/run.py` | Included per-step timing/token records and sampled GPU memory; CPU CSV/LaTeX export |

See [MSE_REPRODUCTION.md](../reproducibility/MSE_REPRODUCTION.md) for saved
models, sample files and commands.
