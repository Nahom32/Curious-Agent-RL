# Curious-Agent-RL
An implementation inspired by schmidhuber's paper: "Curious Model Building Control Systems" for RL and intrinsic motivation control

## Running the experiment pipeline

Use the repository virtual environment to run both the tabular and DQN
experiments in sequence:

```bash
venv/bin/python main.py
```

The runner uses `configs/tabular.yaml` and `configs/dqn.yaml` by default. Useful
options include:

```bash
# Inspect the resolved pipeline without starting training
venv/bin/python main.py --dry-run

# Quick end-to-end run with isolated artifacts
venv/bin/python main.py --episodes 5 --max-steps 20 --output-dir runs/smoke

# Run only one implementation
venv/bin/python main.py --agent tabular
venv/bin/python main.py --agent dqn

# Render the environment while training
venv/bin/python main.py --agent tabular --render
```

Run `venv/bin/python main.py --help` for all configuration, seed, and output
options.
