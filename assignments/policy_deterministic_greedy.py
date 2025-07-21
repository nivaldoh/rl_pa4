import numpy as np
from typing import override
from interfaces.policy import Policy

class Policy_DeterministicGreedy(Policy):
    def __init__(self, Q: np.ndarray[np.float64]):
        """
        Parameters:
        - Q (np.ndarray): Q function; numpy array shape of [nS,nA]
        """
        self.Q = Q

    @override
    def action(self, state: int) -> int:
        """
        Chooses the action that maximizes the Q function for the given state.

        Parameters:
            - state (int): state index

        Returns:
            - int: index of the action to take
        """

        ### TODO: Implement the action method ###
        # raise NotImplementedError
        available_action_values = self.Q[state]
        return np.argmax(available_action_values)


    @override
    def action_prob(self, state: int, action: int) -> float:
        """
        Returns the probability of taking the action if we are in the given state.

        Since this is a greedy policy, this will be a 1 or a 0.

        Parameters:
            - state (int): state index
            - action (int): action index

        Returns:
            - float: the probability of taking the action in the given state
        """

        ### TODO: Implement the action_prob method ###
        # raise NotImplementedError

        available_action_values = self.Q[state]
        greedy_action = np.max(available_action_values)
        if action == greedy_action:
            return 1
        return 0