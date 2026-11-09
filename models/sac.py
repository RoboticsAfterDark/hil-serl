"""
Implementation of RLPD SAC (Soft Actor Critic)
Implementation based on:
https://spinningup.openai.com/en/latest/algorithms/sac.html
https://github.com/ikostrikov/rlpd/blob/main/rlpd/agents/sac/sac_learner.py
https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/agents/continuous/sac.py
"""

from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim:int):
        """
        Takes in state (s) and action (a) and outputs a single scalar representing the expected return
        starting in state s, taking action a, and then following the policy.

        Args:
            state_dim: len(s)
            action_dim: len(a)
        """
        super().__init__()

        self.q_value_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim+action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.q_value_network(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim:int, action_dim:int):
        """
        Takes in state (s) and outputs mean and log standard deviation for each action.
        Each mean and standard deviation represent a Gaussian Action Distribution.
        
        Args:
            state_dim: len(s)
            action_dim: len(a)
        """
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # 2 action heads
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

    def forward(self, x):
        shared_features = self.backbone(x)
        mean = self.mean_head(shared_features)
        log_std = self.log_std_head(shared_features)

        return mean, log_std