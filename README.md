# GymTORAX

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![PyPI Version](https://img.shields.io/pypi/v/gymtorax.svg)](https://pypi.org/project/gymtorax/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/antoine-mouchamps/gymtorax/ci.yml?branch=main&label=tests)](https://github.com/antoine-mouchamps/gymtorax/actions)
[![Documentation](https://img.shields.io/badge/docs-sphinx-brightgreen.svg)](https://gymtorax.readthedocs.io)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A Gymnasium environment for reinforcement learning in tokamak plasma control**

GymTORAX transforms the [TORAX plasma simulator](https://torax.readthedocs.io/) into a set of reinforcement learning (RL) environments, bridging the gap between plasma physics simulation and RL research. It provides ready-to-use Gymnasium-compliant environments for training RL agents on realistic plasma control problems, and allows the creation of new environments.

In its current version, one environment is readily available, based on a ramp-up scenario of the International Thermonuclear Experimental Reactor (ITER).

The documentation of the package is available at [https://gymtorax.readthedocs.io](https://gymtorax.readthedocs.io)

## Key Features

- **Gymnasium Complience**: Seamless compatibility with popular RL libraries
- **Physics Model**: Powered by TORAX 1D transport equations solver
- **Flexible Environment Design**: Easily define custom action spaces, observation spaces, and reward functions

## What is TORAX?

TORAX is an open-source plasma simulator that models the time evolution of plasma quantities (temperatures, densities, magnetic flux, ...) using 1D transport equations. GymTORAX transforms TORAX from an open-loop simulator into a closed-loop control environment suitable for reinforcement learning.

More information about TORAX are available in the official documentation at [https://torax.readthedocs.io/](https://torax.readthedocs.io/).

## Quick Start

### Prerequisites

- **Python 3.10+**

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

Out of the box, Gym-TORAX current provides a single environment based on the Iter-Hybrid ramp-up scenario. The environment is named `IterHybrid-v0` and can be used in the following way:

```python
import gymnasium as gym
import gymtorax

# Create environment
env = gym.make('gymtorax/IterHybrid-v0')

# Reset environment
observation, info = env.reset()

# Run episode
terminated = False
while not terminated:
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
To create a custom plasma control environment, four abstract methods need to be implemented:
- `_get_torax_config`: specifies the TORAX configuration file and the discretization to use.
- `_define_action_space`: defines which actions are considered in this enviroment, and optional bounds and ramp-rates contraints by returning a list of `Action` objects.
- `_define_observation_space`: defines the variables present in the observation and optional bounds by returning an `Observation` object.
- `_compute_reward`: computes the reward base on `state`, `next_state` and `action`.

```python
from gymtorax import BaseEnv
from gymtorax.action_handler import IpAction, EcrhAction
from gymtorax.observation_handler import AllObservation

class CustomPlasmaEnv(BaseEnv):
    """Custom environment for beta_N control with current and heating."""
    def _get_torax_config(self):
        return {
            "config": YOUR_TORAX_CONFIG,  # See docs for config examples
            "discretization": "auto", 
            "delta_t_a": 1.0  # 1 second control timestep
        }

    def _define_action_space(self):
        return [ # [A]
            IpAction(
                min=[1e6], max=[15e6], 
                ramp_rate=[0.2e6]  # MA/s ramp limit
            ),
            EcrhAction( # [W, r/a, width]
                min=[0.0, 0.1, 0.01], 
                max=[20e6, 0.9, 0.5]   
            ),
        ]
    
    def _define_observation_space(self):
        return AllObservation(
            expect={'profiles': ['n_e']} # Remove data from the observation 
        )
    
    def _compute_reward(self, state, next_state, action):
        """Multi-objective reward for plasma control."""
        def _is_H_mode():  # Rought estimate of the LH transition
            if (
                next_state["profiles"]["T_e"][0] > 10
                and next_state["profiles"]["T_i"][0] > 10
            ):
                return True
            else:
                return False

        def _r_fusion_gain(): # Reward based on the fusion gain in H mode
            fusion_gain = reward.get_fusion_gain(next_state) / 10  # Normalize with ITER target
            if _is_H_mode():
                return fusion_gain
            else:
                return 0

        def _r_q_min(): # Reward if safety factor is always > 1
            q_min = reward.get_q_min(next_state)
            if q_min <= 1:
                return q_min
            elif q_min > 1:
                return 1

        def _r_q_95(): # Reward if edge safety factor is > 3
            q_95 = reward.get_q95(next_state)
            if q_95 / 3 <= 1:
                return q_95 / 3
            else:
                return 1

        # Normalize reward components
        r_fusion_gain = weight_list[0] * _r_fusion_gain() / 50
        r_q_min = weight_list[2] * _r_q_min() / 150
        r_q_95 = weight_list[3] * _r_q_95() / 150

        return r_fusion_gain r_q_min + r_q_95 # Return total reward

# Register and use
import gymnasium as gym
gym.register(id='MyPlasmaEnv-v0', entry_point=CustomPlasmaEnv)
env = gym.make('MyPlasmaEnv-v0')
```

## Advanced Usage

### Logging and Debugging

```python
# Configure comprehensive logging
env = gym.make('gymtorax/IterHybrid-v0', 
               log_level="debug",           # debug, info, warning, error
               log_file="simulation.log",    # Log output
               store_history=True)          # Keep full simulation history for postprocessing

# Access simulation data
env.reset()
env.step(env.action_space.sample())

env.save_file("output.nc")
```

### Visualization and Monitoring

GymTORAX provides real-time visualization capabilities for plasma simulation monitoring and analysis.

#### Custom Visualization Configuration

Customize the visualization layout and content using either a default configuration name or a custom TORAX `FigureProperties` object:

```python
# Using default configuration
env = gym.make('gymtorax/IterHybrid-v0', 
               render_mode="human",
               plot_config="default")  # Built-in TORAX plot configuration

# Using custom TORAX FigureProperties object
from torax._src.plotting.plotruns_lib import FigureProperties
custom_config = FigureProperties(...)  # Define custom plot layout
env = gym.make('gymtorax/IterHybrid-v0', 
               render_mode="human",
               plot_config=custom_config)
```

#### Video Recording

Record simulation videos for analysis, presentations, or documentation:

```python
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gymtorax

# Setup video recording wrapper
env = gym.make('gymtorax/IterHybrid-v0', render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda x: True,  # Record every episode
    name_prefix="plasma_simulation"
)

# Run simulation with automatic video recording
observation, info = env.reset()
terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
# Video saved automatically to ./videos/plasma_simulation-episode-0.mp4
```

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
