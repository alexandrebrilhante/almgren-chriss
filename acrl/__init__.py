from collections import deque

import numpy as np

import acrl.synthetic as sca
from acrl.agent import Agent


class AlmgrenChriss:
    def __init__(
        self, liquidation_time=60, n_trades=60, risk_aversion=0, episodes=10000
    ):
        self.liquidation_time = liquidation_time
        self.n_trades = n_trades
        self.risk_aversion = risk_aversion
        self.episodes = episodes

        self.env = sca.MarketEnvironment()

        self.agent = Agent(
            state_size=self.env.observation_space_dimension(),
            action_size=self.env.action_space_dimension(),
            random_seed=0,
        )

        self.shortfall_hist = np.array([])
        self.shortfall_deque = deque(maxlen=100)

    def run(self):
        for episode in range(self.episodes):
            current_state = self.env.reset(
                seed=episode,
                liquid_time=self.liquidation_time,
                num_trades=self.n_trades,
                lamb=self.risk_aversion,
            )

            self.env.start_transactions()

            for i in range(self.n_trades + 1):
                action = self.agent.act(current_state, add_noise=True)
                new_state, reward, done, info = self.env.step(action)
                self.agent.step(current_state, action, reward, new_state, done)
                current_state = new_state

                if info.done:
                    shortfall_hist = np.append(
                        self.shortfall_hist, info.implementation_shortfall
                    )
                    self.shortfall_deque.append(info.implementation_shortfall)
                    break

            if (episode + 1) % 100 == 0:
                print(
                    "\rEpisode [{}/{}]\tAverage Shortfall: ${:,.2f}...".format(
                        episode + 1, self.episodes, np.mean(self.shortfall_deque)
                    )
                )

        print(
            "Average Implementation Shortfall: ${:,.2f}...\n".format(
                np.mean(shortfall_hist)
            )
        )
