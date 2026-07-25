# Curious-Agent-RL

An experimental reinforcement-learning project inspired by Jürgen
Schmidhuber's 1991 work on curious model-building control systems. The project
compares tabular and neural agents whose intrinsic reward comes from
improvement in predicting their environment—not simply from encountering
surprising transitions.

The agents learn inside a configurable grid world containing deterministic,
static, noisy, and changing regions. This makes it possible to study whether
learning progress draws an agent toward useful novelty while allowing it to
lose interest in transitions that are already understood or remain
unpredictable.

## Curiosity mechanism

Each agent contains three learning components:

- **World model (M):** predicts the next state from the current state and action.
- **Confidence model (C):** predicts the world model's error.
- **Controller (Q):** learns action values from external and intrinsic rewards.

For a state-action pair, the confidence model's prediction is recorded before
and after the online model update:

```text
r_curiosity = C_before - C_after
r_total     = r_external + beta * r_curiosity
```

A decrease in predicted model error produces positive curiosity reward. The
combined reward is then used by either tabular Q-learning or DQN.

At each environment step, the pipeline:

1. selects an action with an epsilon-greedy policy;
2. observes the environment transition and external reward;
3. measures and updates the world model;
4. updates the confidence model using the observed prediction error;
5. computes curiosity and total reward;
6. updates the tabular Q-function or neural Q-network.

## Implementations

| Component | Tabular agent | DQN agent |
| --- | --- | --- |
| World model | State-action lookup table | Multilayer perceptron |
| Confidence model | Predicted-error lookup table | Multilayer perceptron |
| Controller | Q-table | Q-network or dueling Q-network |
| Stabilization | Direct Q-learning updates | Replay buffer and a target-network copy |
| Checkpoint | NumPy `.npz` | PyTorch `.pt` |

The unified runner executes the tabular experiment first and the DQN experiment
second by default. Either implementation can also be run independently.

## Environment

The default environment is a `10 x 10` grid with four actions: up, down, left,
and right. Its cells are divided among five configurable zone types:

| Zone | Transition behavior | Experimental purpose |
| --- | --- | --- |
| Static | Agent remains in place | Already-boring behavior |
| Deterministic A | Fixed learnable rule | Primary learnable region |
| Deterministic B | A different fixed rule | Alternative learnable region |
| Noisy | Random transition | Persistently unpredictable region |
| Dynamic | Switches between rules | Tests renewed curiosity |

The agent begins at the top-left corner. The default external goal is at the
bottom-right corner and provides a reward of `1.0`.

## Requirements

- Python 3.10 or newer
- NumPy
- PyTorch
- Pygame
- PyYAML
- Matplotlib
- TensorBoard

Development and test tools are provided by the `dev` optional dependency.

## Installation

Clone the repository and create a local virtual environment:

```bash
git clone <repository-url>
cd Curious-Agent-RL

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
venv\Scripts\Activate.ps1
```

The remaining examples use the virtual environment's Python executable
directly, so activating it is optional.

## Quick start

Inspect the resolved experiment plan without starting training:

```bash
venv/bin/python main.py --dry-run
```

Run a small end-to-end experiment:

```bash
venv/bin/python main.py \
  --episodes 5 \
  --max-steps 20 \
  --output-dir runs/smoke
```

Run the complete experiment using the defaults from both configuration files:

```bash
venv/bin/python main.py
```

The default run is substantially longer: 1,000 tabular episodes followed by
2,000 DQN episodes, with at most 200 steps per episode.

## Command-line options

```text
--agent {tabular,dqn,both}  Select an implementation; default is both
--tabular-config PATH       Use a custom tabular YAML configuration
--dqn-config PATH           Use a custom DQN YAML configuration
--episodes N                Override episodes for all selected agents
--max-steps N               Override maximum steps per episode
--seed N                    Override the configured random seed
--output-dir PATH           Place agent artifacts below a common directory
--render                    Render training through Pygame
--dry-run                   Resolve and print the plan without training
```

For the complete built-in help:

```bash
venv/bin/python main.py --help
```

### Run one agent

```bash
venv/bin/python main.py --agent tabular
venv/bin/python main.py --agent dqn
```

### Use custom configurations

```bash
venv/bin/python main.py \
  --tabular-config configs/tabular.yaml \
  --dqn-config configs/dqn.yaml
```

Command-line values for episodes, maximum steps, seed, and output directory
override the corresponding YAML values without changing the files.

### Render training

```bash
venv/bin/python main.py --agent tabular --render
```

Rendering requires a graphical desktop and is considerably slower than
headless training.

## Configuration

The default experiment definitions are:

- [`configs/tabular.yaml`](configs/tabular.yaml)
- [`configs/dqn.yaml`](configs/dqn.yaml)

Both files configure:

- grid size, goal, zone proportions, colors, and transition rules;
- exploration rate, decay, curiosity weight, and discount factor;
- episode count, maximum episode length, logging interval, and save interval;
- checkpoint, log, and result paths;
- rendering size and frame rate.

The DQN configuration additionally controls the world-model, confidence-model,
and Q-network architectures, optimizer learning rates, replay buffer, and
target-network updates.

The most important curiosity parameter is:

```yaml
agent:
  beta: 1.0
```

`beta` controls the contribution of intrinsic reward to the controller's total
reward. A value of `0.0` disables curiosity and provides a useful external-only
baseline.

## Outputs

Without `--output-dir`, final checkpoints are written to:

```text
checkpoints/
├── tabular/
│   └── agent_final.npz
└── dqn/
    └── agent_final.pt
```

With `--output-dir runs/experiment-01`, they are written to:

```text
runs/experiment-01/
├── tabular/checkpoints/agent_final.npz
└── dqn/checkpoints/agent_final.pt
```

Periodic checkpoints are controlled by `training.save_interval`. Training
progress and summary metrics are currently emitted to the console. The YAML
`logs` and `results` directories are reserved for expanded experiment tracking.

## Standalone training scripts

The original per-agent entry points remain available:

```bash
venv/bin/python scripts/train_tabular.py \
  --config configs/tabular.yaml \
  --seed 42

venv/bin/python scripts/train_dqn.py \
  --config configs/dqn.yaml \
  --seed 42
```

Add `--render` to either command to open the Pygame environment.

## Visualization

The visualization script currently focuses on the tabular agent:

```bash
# Train and display a live tabular agent
venv/bin/python scripts/visualize.py \
  --config configs/tabular.yaml \
  --mode live \
  --episodes 10

# Load the default final tabular checkpoint and plot C/Q heatmaps
venv/bin/python scripts/visualize.py \
  --config configs/tabular.yaml \
  --mode heatmap

# Plot metrics parsed from a standalone tabular training log
venv/bin/python scripts/visualize.py \
  --mode curves \
  --log-file training_tabular.log
```

The heatmap command looks for the checkpoint path specified in the selected
configuration. Matplotlib modes create PNG files and open an interactive plot
window.

## Testing

Run the complete suite with the repository virtual environment:

```bash
venv/bin/pytest
```

Run an individual test module:

```bash
venv/bin/pytest tests/test_agents.py
venv/bin/pytest tests/test_main.py
```

The suite covers the environment, tabular curiosity behavior, neural models,
DQN target-network construction, and unified experiment orchestration.

## Project structure

```text
Curious-Agent-RL/
├── main.py                         # Unified experiment pipeline
├── configs/
│   ├── tabular.yaml                # Tabular experiment configuration
│   └── dqn.yaml                    # Neural experiment configuration
├── scripts/
│   ├── train_tabular.py            # Standalone tabular trainer
│   ├── train_dqn.py                # Standalone DQN trainer
│   └── visualize.py                # Live and analytical visualization
├── src/curious_agent/
│   ├── agents/
│   │   ├── tabular_curious.py      # Tabular M, C, and Q agent
│   │   └── dqn_curious.py          # Neural M, C, and Q agent
│   ├── env/grid_world.py           # Multi-zone grid environment
│   ├── models/
│   │   ├── world_model.py
│   │   ├── confidence_net.py
│   │   └── q_network.py
│   └── utils/replay_buffer.py
└── tests/                           # Unit and integration tests
```

## Current scope and limitations

This repository is an early experimental implementation rather than a
benchmark-ready RL framework.

- The curiosity signal is an online approximation based on the confidence
  model's change in predicted error.
- The tabular world model stores one next-state prediction per state-action
  pair, so it cannot represent a full stochastic transition distribution.
- The DQN agent maintains and soft-updates a target-network copy, but its
  current bootstrap loss still uses the online Q-network.
- Visualization and checkpoint analysis are currently more complete for the
  tabular agent than for DQN.
- Zone-distribution visualization and tabular model-accuracy reporting still
  contain placeholder logic.
- The unified runner saves checkpoints, but it does not yet aggregate repeated
  seeds, evaluate trained policies, or generate comparison reports.

These constraints make the project most useful for learning, prototyping, and
extending curiosity-driven exploration experiments.

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
