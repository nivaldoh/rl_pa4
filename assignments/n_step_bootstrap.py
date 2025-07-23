from typing import Iterable, Tuple
import gymnasium as gym
import numpy as np

from interfaces.policy import RandomPolicy
from interfaces.solver import Solver, Hyperparameters

from assignments.policy_deterministic_greedy import Policy_DeterministicGreedy

def on_policy_n_step_td(
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    n: int,
    alpha: float,
    initV: np.array,
    gamma: float = 1.0
) -> Tuple[np.array]:
    """
    Runs the on-policy n-step TD algorithm to estimate the value function for a given policy.

    Sutton & Barto, p. 144, "n-step TD Prediction"

    Parameters:
        trajs (list): N trajectories generated using an unknown policy. Each trajectory is a 
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n (int): The number of steps (the "n" in n-step TD)
        alpha (float): The learning rate
        initV (np.ndarray): initial V values; np array shape of [nS]
        gamma (float): The discount factor

    Returns:
        V (np.ndarray): $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = np.array(initV)

    #print(f'n={n}')
    for ep in trajs:
        #print(f'\n-------\nEp(s_t,a_t,r_[t+1],s_[t+1]): {ep}')
        T = len(ep)
        states = [s for s, _, _, _ in ep]
        rewards = [r for _, _, r, _ in ep]

        for tau in range(T):
            #print(f'tau: {tau}')
            G = 0

            for i in range(tau + 1, min(tau + n, T) + 1):
                #print(f'i: {i}')
                G += gamma**(i - tau - 1) * rewards[i - 1]
                #print(f'G updated to {G}')
                
            if tau + n < T:
                # #print('Bootstrapping with value at step {}, state {}, action {}, containing Q={}'.format(
                #     min_tau + n,
                #     states[min_tau + n],
                #     actions[min_tau + n],
                #     Q[states[min_tau + n], [actions[min_tau + n]]],
                # ))
                G += gamma**(n) * V[states[tau + n]]
                #print(f'G bootstrap update to {G}')

            # #print('Updating V at {}, state {}, with V={}'.format(
            #     iter_tau,
            #     states[iter_tau],
            #     V[states[iter_tau]],
            # ))
            V[states[tau]] += alpha * (G - V[states[tau]])
            #print('New V: {}'.format(V[states[tau]]))

    return V

# python run.py n_step_bootstrap --environment WrappedFrozenLake-v0 --num_episodes 1

class NStepSARSAHyperparameters(Hyperparameters):
    """ Hyperparameters for NStepSARSA algorithm """
    def __init__(self, gamma: float, alpha: float, n: int):
        """
        Parameters:
            gamma (float): The discount factor
            alpha (float): The learning rate
            n (int): The number of steps (the "n" in n-step SARSA)
        """
        super().__init__(gamma)
        self.alpha = alpha
        """The learning rate"""
        self.n = n
        """The number of steps (the "n" in n-step SARSA)"""

class NStepSARSA(Solver):
    """
    Solver for N-Step SARSA algorithm, good for discrete state and action spaces.

    Off-policy algorithm, using weighted importance sampling.
    """
    def __init__(self, env: gym.Env, hyperparameters: NStepSARSAHyperparameters):
        super().__init__("NStepSARSA", env, hyperparameters)
        self.pi = Policy_DeterministicGreedy(np.ones((env.observation_space.n, env.action_space.n)))



        self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.bpi = RandomPolicy(self.env.action_space.n)

    def action(self, state):
        """
        Chooses an action based on the current policy.

        Parameters:
            state (int): The current state
        
        Returns:
            int: The action to take
        """
        return self.pi.action(state)

    def train_episode(self):
        """
        Trains the agent for a single episode.

        Returns:
            float: The total (undiscounted) reward for the episode
        """

        #####################
        # TODO: Implement Off Policy n-Step SARSA algorithm
        #   - Hint: Sutton Book p. 149
        #   - Hint: You'll need to build your trajectories using a behavior policy (RandomPolicy)
        #   - Hint: You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.
        #   - Hint: Be sure to check both terminated and truncated variables.
        #####################
        
        episode_G = 0.0

        #######

        ep = []
        state, _ = self.env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = self.bpi.action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            ep.append((state, action, reward, next_state))
            episode_G += reward # for logging only
            state = next_state

        # Ep(s_t,a_t,r_[t+1],s_[t+1]): [(0, 1, 0.0, 4), (4, 1, 0.0, 8), (8, 2, 0.0, 9), (9, 2, 0.0, 10), (10, 1, 0.0, 14), (14, 2, 1.0, 15)]

        gamma = self.hyperparameters.gamma
        alpha = self.hyperparameters.alpha
        n = self.hyperparameters.n

        T = len(ep)
        states = [s for s, _, _, _ in ep]
        rewards = [r for _, _, r, _ in ep]
        # if any(r != 0 for r in rewards):
            #print(f'\n\n\n\n\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Episode has non-zero reward: {rewards}')
            #print(f'~\nEp(s_t,a_t,r_[t+1],s_[t+1]): {ep}')
        
        actions = [a for _, a, _, _ in ep]

        # tau 0, n 2
        for tau in range(T):
            #print(f'\ntau: {tau}')
            G = 0.0
            rho = 1.0

            # i 1 up to min (2, 5) + 1
            for i in range(tau + 1, min(tau + n, T) + 1):
                #print(f'i: {i}')
                G += gamma**(i - tau - 1) * rewards[i - 1]
                #print(f'G updated to {G}')

            for i in range(tau + 1, min(tau + n + 1, T)):
                #print(f'i: {i}')
                pi_prob = self.pi.action_prob(states[i], actions[i])
                bpi_prob = self.bpi.action_prob(states[i], actions[i])
                #print(f'pi_prob: {pi_prob}, bpi_prob: {bpi_prob}')
                rho *= pi_prob/bpi_prob
                #print(f'rho updated to {rho}')
                
            if tau + n < T:
                # #print('Bootstrapping with value at step {}, state {}, action {}, containing Q={}'.format(
                #     min_tau + n,
                #     states[min_tau + n],
                #     actions[min_tau + n],
                #     Q[states[min_tau + n], [actions[min_tau + n]]],
                # ))
                G += gamma**(n) * self.Q[states[tau + n], actions[tau + n]]
                #print(f'G bootstrap update to {G}')

            # #print('Updating Q at {}, state {}, action {}, with Q={}'.format(
            #     iter_tau,
            #     states[iter_tau],
            #     actions[iter_tau],
            #     Q[states[iter_tau], [actions[iter_tau]]],
            # ))

            self.Q[states[tau], actions[tau]] += alpha * rho * (G - self.Q[states[tau], actions[tau]])
            self.pi.update_Q(self.Q)
            # #print('Q: {}'.format(self.Q))
            
        return sum(rewards)
        #######

        # return episode_G
