# almgren-chriss

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/acrl)
![GitHub](https://img.shields.io/github/license/brilhana/almgren-chriss)

Deep reinforcement learning for optimal execution of portfolio transactions.

## Installation
```bash
pip install acrl
```

## Usage
```python
from collections import deque

import numpy as np

import acrl as sca
from acrl.agent import Agent

env = sca.MarketEnvironment()

agent = Agent(
    state_size=env.observation_space_dimension(),
    action_size=env.action_space_dimension(),
    random_seed=0,
)

liquidation_time = 60
n_trades = 60
risk_aversion = 1e-6
episodes = 10000

shortfall_hist = np.array([])
shortfall_deque = deque(maxlen=100)

for episode in range(episodes):
    current_state = env.reset(
        seed=episode,
        liquid_time=liquidation_time,
        num_trades=n_trades,
        lamb=risk_aversion,
    )

    env.start_transactions()

    for i in range(n_trades + 1):
        action = agent.act(current_state, add_noise=True)
        new_state, reward, done, info = env.step(action)
        agent.step(current_state, action, reward, new_state, done)
        current_state = new_state

        if info.done:
            shortfall_hist = np.append(shortfall_hist, info.implementation_shortfall)
            shortfall_deque.append(info.implementation_shortfall)
            break

    if (episode + 1) % 100 == 0:
        print(
            "\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}".format(
                episode + 1, episodes, np.mean(shortfall_deque)
            )
        )

print("\nAverage Implementation Shortfall: ${:,.2f} \n".format(np.mean(shortfall_hist)))
```