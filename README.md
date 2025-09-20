# GymTORAX

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyPI Version](https://img.shields.io/pypi/v/gymtorax.svg)](https://pypi.org/project/gymtorax/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/antoine-mouchamps/gymtorax/tests.yml?branch=main&label=tests)](https://github.com/antoine-mouchamps/gymtorax/actions)
[![Documentation](https://img.shields.io/badge/docs-sphinx-brightgreen.svg)](https://gymtorax.readthedocs.io)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A Gymnasium environment for reinforcement learning in tokamak plasma control**

GymTORAX transforms the [TORAX plasma simulator](https://torax.readthedocs.io/) into a set of reinforcement learning (RL) environments, bridging the gap between plasma physics simulation and RL research. It provides ready-to-use Gymnasium-compliant environments for training RL agents on realistic plasma control problems, and allows the creation of new environments.

The documentation is available at [https://gymtorax.readthedocs.io](https://gymtorax.readthedocs.io)

## Key Features

- **Gymnasium Complience**: Seamless compatibility with popular RL libraries
- **Physics Model**: Powered by TORAX 1D transport equations solver
- **Flexible Environment Design**: Easily define custom action spaces, observation spaces, and reward functions

## What is TORAX?

TORAX is an open-source plasma simulator that models the time evolution of plasma quantities (temperatures, densities, magnetic flux, ...) using 1D transport equations. GymTORAX transforms TORAX from an open-loop simulator into a closed-loop control environment suitable for reinforcement learning.

More information in the official TORAX documentation at [https://torax.readthedocs.io/](https://torax.readthedocs.io/).

## Quick Start

### Prerequisites

- **Python 3.10+** (required for modern typing features)

### Installation

Install from PyPI (recommended):

```bash
pip install gymtorax
```

For development installation:

```bash
git clone https://github.com/antoine-mouchamps/gymtorax
cd gymtorax
pip install -e ".[dev,docs]"
```

### Verify Installation

```python
import gymtorax
print(f"GymTORAX version: {gymtorax.__version__}")

# Quick test
import gymnasium as gym
env = gym.make('gymtorax/Test-v0')
env.reset()
env.close()
```

### Basic Usage

```python
import gymnasium as gym
import gymtorax

# Create environment
env = gym.make('gymtorax/IterHybrid-v0')

# Reset environment
observation, info = env.reset()

# Run episode
for step in range(100):
    # Random action (replace with your RL agent)
    action = env.action_space.sample()
    
    # Execute action
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()
```

### Custom Environment

Create your own plasma control task:

```python
from gymtorax import BaseEnv
from gymtorax.action_handler import IpAction, EcrhAction
from gymtorax.observation_handler import AllObservation

class CustomPlasmaEnv(BaseEnv):
    """Custom environment for beta_N control with current and heating."""
    
    def _define_action_space(self):
        return [
            IpAction(
                min=[1e6], max=[15e6], 
                ramp_rate=[0.2e6]  # MA/s ramp limit
            ),
            EcrhAction(
                min=[0.0, 0.1, 0.01], 
                max=[20e6, 0.9, 0.5]   # [W, r/a, width]
            ),
        ]
    
    def _define_observation_space(self):
        return AllObservation()
    
    def _get_torax_config(self):
        return {
            "config": YOUR_TORAX_CONFIG,  # See docs for config examples
            "discretization": "auto", 
            "delta_t_a": 1.0  # 1 second control timestep
        }
    
    def _compute_reward(self, state, next_state, action):
        """Multi-objective reward for beta_N control."""
        # Performance: track normalized beta
        target_beta = 0.02
        current_beta = next_state["/"]["plasma_geometry"]["beta_N"][0]
        performance = -abs(current_beta - target_beta)
        
        # Safety: penalize q < 2 (kink instability)
        q_min = next_state["/"]["plasma_physics"]["q_safety_factor"].min()
        safety = -max(0, 2.0 - q_min) * 10
        
        # Efficiency: penalize excessive control effort
        control_effort = -0.01 * sum(np.sum(a**2) for a in action.values())
        
        return performance + safety + control_effort

# Register and use
import gymnasium as gym
gym.register(id='MyPlasma-v0', entry_point=CustomPlasmaEnv)
env = gym.make('MyPlasma-v0')
```

## Advanced Usage

### Logging and Debugging

```python
# Configure comprehensive logging
env = gym.make('gymtorax/IterHybrid-v0', 
               log_level="debug",           # debug, info, warning, error
               logfile="simulation.log",    # File output
               store_history=True)          # Keep full simulation history

# Access simulation data
env.reset()
env.step(env.action_space.sample())
torax_state = env.torax_app.get_state_data()  # Raw TORAX state
history = env.torax_app.get_output_datatree()  # Full xarray DataTree
```

### Visualization and Monitoring
WIP
```python
WIP
```

## Contributing

We welcome contributions!

### Development Workflow

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create** a feature branch: `git checkout -b feature/new_feature`
4. **Set up** development environment:
   ```bash
   pip install -e ".[dev,docs]"
   pre-commit install  # Optional: auto-formatting
   ```
5. **Make** your changes with tests
6. **Run** quality checks:
   ```bash
   pytest                    # Run test suite
   ruff check && ruff format # Linting and formatting
   pytest --cov=gymtorax     # Coverage report
   ```
7. **Commit** and **push** changes
8. **Open** a Pull Request with description

## Citation

If you use GymTORAX in your research, please cite our work:

```bibtex
@software{gym_torax_2024,
    title={Gym-TORAX: A Gymnasium Environment for Reinforcement Learning in Tokamak Plasma Control},
    author={Antoine Mouchamps and Arthur Malherbe and Adrien Bolland and Damien Ernst},
    year={2024},
    url={https://github.com/antoine-mouchamps/gymtorax},
    version={0.1.0},
    note={Software package for reinforcement learning in fusion plasma control}
}
```

**Research Article**: A publication describing GymTORAX is in preparation. This citation will be updated upon publication.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
