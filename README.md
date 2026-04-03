# GIPO: Gaussian Importance Sampling Policy Optimization

This repository contains a lightweight, single-process implementation of Reinforcement Learning for **MetaWorld** robotic manipulation tasks, featuring the **GIPO** clipping mechanism. It uses a discretized action space and a simple MLP architecture to process low-dimensional state observations.

---

## Overview

The script `metaworld_ppo_discrete_simple_complete.py` provides an end-to-end training pipeline for robotic control. Key features include:

* **Discrete Action Space:** Maps continuous MetaWorld actions into $N$ discrete bins (default 256) per dimension to simplify the policy output.
* **Actor-Critic MLP:** A shared encoder architecture with independent policy and value heads for efficient feature extraction.
* **GIPO Clipping:** Implements Gaussian Importance Sampling Policy Optimization, which uses kernels (Gaussian, Laplacian, or Cauchy) to provide a smoother optimization landscape than standard PPO.
* **Efficiency:** Designed for $39$-dimensional state inputs and optimized with `bfloat16` support for high-performance training on modern GPUs.
* **Dynamic Schedulers:** Built-in cosine decay for learning rates with configurable warmup periods for both actor and critic.

---

## Architecture



The model follows a classic Actor-Critic structure with a shared representation layer:

1.  **Shared Encoder:** A 2-layer MLP with a hidden dimension of 512, utilizing ReLU activations and LayerNorm for stable training.
2.  **Policy Head:** Processes encoder features to output logits for each action dimension ($4 \times 256$ bins).
3.  **Value Head:** Detaches from the shared features to provide a baseline state-value estimate $V(s)$.

---

## Requirements

The project requires a MuJoCo-compatible environment and the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Note: The environment variables are pre-configured in the script to use `osmesa` for headless rendering.*

---

## Usage

### Training
To initiate training on a specific MetaWorld task (e.g., `reach-v3`):

```bash
python metaworld_ppo_discrete_simple_complete.py --task-name reach-v3 --clip-mode gipo
```

### Key Hyperparameters
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--clip-mode` | `gipo` | Clipping strategy choice: `ppo`, `sapo`, or `gipo`. |
| `--kernel-type` | `gaussian` | The kernel used for GIPO (options: `gaussian`, `laplacian`, `cauchy`). |
| `--train-iters` | `1000` | Total number of training iterations. |
| `--policy-lr` | `3e-4` | Learning rate for the policy head. |
| `--sigma` | `1.0` | The $\sigma$ parameter controlling the GIPO soft-clipping width. |
| `--seed` | `42` | Random seed for reproducibility (Torch, NumPy, Python). |
### Monitoring
Progress is logged via **TensorBoard**. You can monitor returns, success rates, and the ratio of policy updates:

```bash
tensorboard --logdir runs/
```

---

## Implementation Details

### Action Mapping
MetaWorld typically requires continuous actions in the range $[-1, 1]$. This wrapper converts discrete token IDs back to continuous values using the following logic:
$$continuous = -1.0 + 2.0 \cdot \frac{token}{bins - 1}$$

### GIPO Mechanism
When `--clip-mode gipo` is active with the Gaussian kernel, the surrogate loss is weighted by a soft-clipping coefficient:
$$coeff = \exp\left(-0.5 \cdot \left(\frac{\log(ratio)}{\sigma}\right)^2\right)$$

This approach ensures that the policy does not deviate too aggressively from the old distribution while maintaining gradient information across a wider range than standard PPO clipping.

---

## Checkpointing
The system saves comprehensive experiment data to the `log-dir`:
* **`args.json`**: Saves all CLI arguments for experiment reproduction.
* **`simple_state_iter_N.pt`**: Contains model state, optimizer state, and all random seeds (Torch, NumPy, Python).
* **`policy_prob_pairs_latest.csv`**: Logs the $P_{old}$ and $P_{new}$ probabilities for the latest updates to analyze distribution shifts.
