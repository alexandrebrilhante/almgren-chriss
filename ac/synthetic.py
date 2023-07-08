import collections
import random

import numpy as np

ANNUAL_VOLAT = 0.12
BID_ASK_SP = 1 / 8
DAILY_TRADE_VOL = 5e6
TRAD_DAYS = 250
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)
TOTAL_SHARES = 1000000
STARTING_PRICE = 50
LLAMBDA = 1e-6
LIQUIDATION_TIME = 60
NUM_N = 60
EPSILON = BID_ASK_SP / 2
SINGLE_STEP_VARIANCE = (DAILY_VOLAT * STARTING_PRICE) ** 2
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)


class MarketEnvironment:
    def __init__(
        self, seed=0, lqd_time=LIQUIDATION_TIME, num_tr=NUM_N, llambda=LLAMBDA
    ):
        random.seed(seed)

        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT
        self.total_shares = TOTAL_SHARES
        self.starting_price = STARTING_PRICE
        self.llambda = llambda
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.single_step_variance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA
        self.tau = self.liquidation_time / self.num_n
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)

        self.kappa_hat = np.sqrt(
            (self.llambda * self.single_step_variance) / self.eta_hat
        )

        self.kappa = (
            np.arccosh((((self.kappa_hat**2) * (self.tau**2)) / 2) + 1) / self.tau
        )

        self.shares_remaining = self.total_shares
        self.time_horizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))
        self.prev_impact_price = self.starting_price
        self.transacting = False
        self.k = 0

    def reset(
        self, seed=0, liquid_time=LIQUIDATION_TIME, num_trades=NUM_N, lamb=LLAMBDA
    ):
        self.__init__(
            randomSeed=seed, lqd_time=liquid_time, num_tr=num_trades, lambd=lamb
        )

        self.initial_state = np.array(
            list(self.logReturns)
            + [
                self.time_horizon / self.num_n,
                self.shares_remaining / self.total_shares,
            ]
        )

        return self.initial_state

    def start_transactions(self):
        self.transacting = True
        self.tolerance = 1
        self.total_capture = 0
        self.prev_price = self.starting_price
        self.totalSSSQ = 0
        self.totalSRSQ = 0
        self.prev_utility = self.compute_ac_utility(self.total_shares)

    def step(self, action):
        class Info:
            pass

        info = Info()
        info.done = False

        if self.transacting and (
            self.time_horizon == 0 or abs(self.shares_remaining) < self.tolerance
        ):
            self.transacting = False

            info.done = True
            info.slippage = self.total_shares * self.starting_price - self.total_capture

            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)

            info.expected_variance = (
                self.single_step_variance * self.tau * self.totalSRSQ
            )

            info.utility = (
                info.expected_shortfall + self.llambda * info.expected_variance
            )

        if self.k == 0:
            info.price = self.prev_impact_price
        else:
            info.price = self.prev_impact_price + np.sqrt(
                self.single_step_variance * self.tau
            ) * random.normalvariate(0, 1)

        if self.transacting:
            if isinstance(action, np.ndarray):
                action = action.item()

            leaves_now = self.shares_remaining * action

            if self.time_horizon < 2:
                leaves_now = self.shares_remaining

            info.share_to_sell_now = np.around(leaves_now)
            info.curr_perm_impact = self.perm_impact(info.share_to_sell_now)
            info.curr_temp_impact = self.temp_impact(info.share_to_sell_now)
            info.exec_price = info.price - info.curr_temp_impact
            self.total_capture += info.share_to_sell_now * info.exec_price

            self.logReturns.append(np.log(info.price / self.prev_price))
            self.logReturns.popleft()

            self.shares_remaining -= info.share_to_sell_now

            self.totalSSSQ += info.share_to_sell_now**2
            self.totalSRSQ += self.shares_remaining**2

            self.time_horizon -= 1
            self.prev_price = info.price
            self.prev_impact_price = info.price - info.curr_perm_impact

            curr_utility = self.compute_ac_utility(self.shares_remaining)
            reward = (abs(self.prev_utility) - abs(curr_utility)) / abs(
                self.prev_utility
            )
            self.prev_utility = curr_utility

            if self.shares_remaining <= 0:
                info.slippage = (
                    self.total_shares * self.starting_price - self.total_capture
                )

                info.done = True
        else:
            reward = 0.0

        self.k += 1

        state = np.array(
            list(self.logReturns)
            + [
                self.time_horizon / self.num_n,
                self.shares_remaining / self.total_shares,
            ]
        )

        return (state, np.array([reward]), info.done, info)

    def perm_impact(self, leaves):
        return self.gamma * leaves

    def temp_impact(self, leaves):
        return (self.epsilon * np.sign(leaves)) + ((self.eta / self.tau) * leaves)

    def get_expected_shortfall(self, leaves):
        return (
            0.5 * self.gamma * (leaves**2)
            + self.epsilon * leaves
            + (self.eta_hat / self.tau) * self.totalSSSQ
        )

    def get_ac_expected_shortfall(self, leaves):
        ft = 0.5 * self.gamma * (leaves**2)

        st = self.epsilon * leaves

        tt = self.eta_hat * (leaves**2)

        nft = np.tanh(0.5 * self.kappa * self.tau) * (
            self.tau * np.sinh(2 * self.kappa * self.liquidation_time)
            + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau)
        )

        dft = 2 * (self.tau**2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)

        fot = nft / dft

        return ft + st + (tt * fot)

    def get_ac_variance(self, leaves):
        ft = 0.5 * (self.single_step_variance) * (leaves**2)

        nst = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(
            self.kappa * (self.liquidation_time - self.tau)
        ) - self.liquidation_time * np.sinh(self.kappa * self.tau)

        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(
            self.kappa * self.tau
        )

        st = nst / dst

        return ft * st

    def compute_ac_utility(self, leaves):
        if self.liquidation_time == 0:
            return 0

        return self.get_ac_expected_shortfall(
            leaves
        ) + self.llambda * self.get_ac_variance(leaves)

    def get_trade_list(self):
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares

        for i in range(1, self.num_n + 1):
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st

        trade_list *= ft

        return trade_list

    def observation_space_dimension(self):
        return 8

    def action_space_dimension(self):
        return 1

    def stop_transactions(self):
        self.transacting = False
