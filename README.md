# sim2sim

Sim-to-sim policy validation with deploy.json configuration.

## Overview

`sim2sim` enables validating trained policies in a different simulator before deploying to real hardware. It reads the `deploy.json` configuration exported during training and runs the policy in MuJoCo (or other backends) for validation.

## Features

- **deploy.json driven**: Reads observation/action configurations from deploy.json (deploy_std standard)
- **Torque control default**: Simulation uses motor torque interface, matching real robot behavior
- **Built-in controllers**: PD and MIT Cheetah-style controllers with gravity/friction compensation
- **History support**: Stacks observation frames for encoder-based policies
- **rll_rl integration**: Loads PPO/AMP/FastSAC/TDMPC2 checkpoints directly

## Install

```bash
# From robot_lab workspace
uv pip install --python .venv/bin/python -e ./source/sim2sim --no-build-isolation

# With MuJoCo support
uv pip install --python .venv/bin/python -e './source/sim2sim[mujoco]' --no-build-isolation

# Full install
uv pip install --python .venv/bin/python -e './source/sim2sim[all]' --no-build-isolation
```

## Quick Start

### 1. Export deploy.json during training play

```bash
cd source/rll_rl
python examples/gymnasium/ppo_play.py checkpoint.pt --env-id Pendulum-v1 --export-deploy --deploy-output deploy.json
```

### 2. Run sim2sim validation

```bash
# Basic torque control with PD controller
sim2sim -d deploy.json -c checkpoint.pt --mjcf robot.xml

# MIT Cheetah controller
sim2sim -d deploy.json -c checkpoint.pt --mjcf robot.xml --controller mit_cheetah

# Custom PD gains
sim2sim -d deploy.json -c checkpoint.pt --mjcf robot.xml --kp 50 --kd 2.0

# Render
sim2sim -d deploy.json -c checkpoint.pt --mjcf robot.xml --render
```

## Architecture

```
source/sim2sim/
├── pyproject.toml
├── src/sim2sim/
│   ├── __init__.py
│   ├── config.py              # DeployConfig dataclass + loader
│   ├── history.py             # Observation history buffer
│   ├── controllers/
│   │   ├── base.py            # Abstract MotorController
│   │   ├── pd.py              # PD position controller
│   │   └── mit.py             # MIT Cheetah controller
│   ├── backends/
│   │   ├── base.py            # Abstract SimulatorBackend
│   │   ├── mujoco.py          # MuJoCo backend
│   │   └── factory.py         # Backend registry
│   ├── policy/
│   │   ├── base.py            # Abstract PolicyLoader
│   │   └── rll_rl.py          # rll_rl checkpoint loader
│   ├── wrappers/
│   │   └── deploy_wrapper.py  # deploy.json obs/action processing
│   └── main.py                # CLI entry point
```

## Controllers

### PD Controller

Standard PD position control:
```
torque = Kp * (target_pos - current_pos) - Kd * current_vel
```

```bash
sim2sim --controller pd --kp 30 --kd 0.8
```

### MIT Cheetah Controller

Torque controller with feedforward, gravity compensation, and virtual spring-damper:

```bash
sim2sim --controller mit_cheetah --kp 40 --kd 1.0 --virtual-spring-k 50
```

Features:
- Gravity compensation via MuJoCo's `mj_rne`
- Coulomb + viscous friction model
- Optional virtual spring-damper for balance

## API Usage

```python
from sim2sim import load_deploy_config
from sim2sim.backends import create_backend
from sim2sim.controllers import MITCheetahController
from sim2sim.policy import RllRlPolicyLoader
from sim2sim.wrappers import DeployWrapper

# Load config
cfg = load_deploy_config("deploy.json")

# Create backend with MIT controller
backend = create_backend("mujoco")
controller = MITCheetahController(kp=40.0, kd=1.0)
backend.load("robot.xml", cfg, controller=controller)

# Load policy
loader = RllRlPolicyLoader()
loader.load("checkpoint.pt")
policy = loader.get_inference_policy()

# Create wrapper
wrapper = DeployWrapper(cfg, obs_dim=backend.n_joints * 2)

# Run
raw_obs = backend.reset()
obs = wrapper.reset(raw_obs)

for _ in range(1000):
    action = policy(obs)
    raw_obs = backend.step(action)
    obs, _ = wrapper.step(raw_obs, action)
```

## deploy.json Format

Follows the deploy_std specification:

```json
{
  "format_version": 1,
  "step_dt": 0.02,
  "actuator": {
    "stiffness": [30.0, ...],
    "damping": [0.8, ...]
  },
  "actions": {
    "concat": {
      "all_joints": {"dim": 1, "keys": ["JointPositionAction"]}
    },
    "JointPositionAction": {
      "dim": [12],
      "type": "joint_position",
      "scale": [0.25, ...],
      "offset": [0.0, ...],
      "joints": ["FL_hip_joint", ...]
    }
  },
  "observations": {
    "concat": {
      "obs": {"dim": 1, "keys": ["joint_pos", "joint_vel", "base_ang_vel"]}
    },
    "joint_pos": {"dim": [12], "type": "joint_position", "scale": [1.0, ...]},
    "joint_vel": {"dim": [12], "type": "joint_velocity", "scale": [0.05, ...]},
    "base_ang_vel": {"dim": [3], "type": "base_ang_vel", "scale": [0.25, ...]}
  }
}
```

## Development

```bash
cd source/sim2sim
ruff check src tests
pytest tests
```

## License

BSD-3-Clause
