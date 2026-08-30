import numpy as np

class RolloutBuffer:
    def __init__(self, rollout_steps: int, num_envs: int, obs_dim: int, num_dices: int):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs

        # Pre-allocate arrays
        self.obs = np.zeros((rollout_steps, num_envs, obs_dim), dtype=np.float32)
        self.dice_mask = np.zeros((rollout_steps, num_envs, num_dices), dtype=np.float32)
        self.category = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.action_type = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        self.step = 0

    def add(self, obs, action, rewards, dones, values, log_probs):
        self.obs[self.step] = obs
        self.dice_mask[self.step] = action["dice_mask"]
        self.category[self.step] = action["category"]
        self.action_type[self.step] = action["action_type"]
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.log_probs[self.step] = log_probs
        
        self.step += 1

    def reset(self):
        self.step = 0

    def get_flattened_data(self):
        def flatten(arr):
            return arr.reshape(-1, *arr.shape[2:])

        return (
            flatten(self.obs),
            flatten(self.dice_mask),
            flatten(self.category),
            flatten(self.action_type),
            self.rewards,     # (T, N)
            self.dones,       # (T, N)
            self.values,      # (T, N)
            flatten(self.log_probs)
        )

def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """
    rewards: (T, N)
    values:  (T, N)
    dones:   (T, N) bool
    last_values: (N,) scalar bootstrap value after last step
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages.reshape(-1), returns.reshape(-1)
