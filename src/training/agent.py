import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical

from src.game.environment import ActionType
from src.training.ppo_network import PPOPolicy
from src.training.buffer import RolloutBuffer

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        num_dices: int,
        num_categories: int,
        hidden: int = 256,
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = device
        self.policy = PPOPolicy(
            obs_dim=obs_dim,
            num_dices=num_dices,
            num_categories=num_categories,
            hidden=hidden
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        
    def train(self, mode: bool = True):
        self.policy.train(mode)
        
    def eval(self):
        self.policy.eval()

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        obs: np.ndarray (N, obs_dim) or (obs_dim,)
        returns:
            action: dict compatible with vector envs (batched) or single env
            log_prob: np.ndarray (N,) or float
            value: np.ndarray (N,) or float
            action_type: np.ndarray (N,) or int
        """
        is_single = obs.ndim == 1
        if is_single:
            obs = np.expand_dims(obs, axis=0)

        self.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dice_logits, cat_logits, values = self.policy(obs_t)

            phases = obs[:, -1]  # action_type is the last element
            phases_int = phases.astype(int)

            dice_dist = Bernoulli(logits=dice_logits)
            cat_dist = Categorical(logits=cat_logits)

            if deterministic:
                dice_sample = (dice_logits > 0.0).float()
                cat_sample = cat_logits.argmax(dim=-1)
            else:
                dice_sample = dice_dist.sample()
                cat_sample = cat_dist.sample()

            dice_log_prob = dice_dist.log_prob(dice_sample).sum(dim=1)
            cat_log_prob = cat_dist.log_prob(cat_sample)

            N = obs.shape[0]
            log_probs = np.zeros(N, dtype=np.float32)
            dice_actions = dice_sample.cpu().numpy().astype(bool)
            cat_actions = cat_sample.cpu().numpy().astype(np.int64)

            for i in range(N):
                if phases_int[i] == int(ActionType.SELECT_DICE):
                    log_probs[i] = dice_log_prob[i].item()
                else:
                    log_probs[i] = cat_log_prob[i].item()

            action = {
                "action_type": phases_int,
                "dice_mask": dice_actions,
                "category": cat_actions,
            }

        values_np = values.cpu().numpy()

        if is_single:
            return (
                {k: v[0] for k, v in action.items()},
                float(log_probs[0]),
                float(values_np[0]),
                int(phases_int[0]),
            )
        return action, log_probs, values_np, phases_int

    def _compute_log_prob(self, obs_t, dice_mask_t, category_t, action_type_t):
        dice_logits, cat_logits, _ = self.policy(obs_t)
        N = obs_t.size(0)
        logp = torch.empty(N, device=self.device)

        dice_phase = action_type_t == int(ActionType.SELECT_DICE)
        cat_phase = action_type_t == int(ActionType.SELECT_CATEGORY)

        if dice_phase.any():
            dice_dist = Bernoulli(logits=dice_logits[dice_phase])
            logp[dice_phase] = dice_dist.log_prob(dice_mask_t[dice_phase]).sum(dim=1)

        if cat_phase.any():
            cat_dist = Categorical(logits=cat_logits[cat_phase])
            logp[cat_phase] = cat_dist.log_prob(category_t[cat_phase])

        return logp

    def _compute_entropy(self, obs_t, action_type_t):
        dice_logits, cat_logits, _ = self.policy(obs_t)
        N = obs_t.size(0)
        entropy = torch.zeros(N, device=self.device)

        dice_phase = action_type_t == int(ActionType.SELECT_DICE)
        cat_phase = action_type_t == int(ActionType.SELECT_CATEGORY)

        if dice_phase.any():
            dice_dist = Bernoulli(logits=dice_logits[dice_phase])
            entropy[dice_phase] = dice_dist.entropy().sum(dim=1)

        if cat_phase.any():
            cat_dist = Categorical(logits=cat_logits[cat_phase])
            entropy[cat_phase] = cat_dist.entropy()

        return entropy

    def update(
        self,
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
        batch_size: int = 512,
        ppo_epochs: int = 10,
        clip_eps: float = 0.1,
        value_coef: float = 0.5,
        entropy_coef: float = 0.08,
    ):
        (
            obs_arr,
            dice_arr,
            cat_arr,
            act_type_arr,
            _,
            _,
            _,
            old_logp_arr,
        ) = buffer.get_flattened_data()

        # Pre-load to device to speed up inner epoch training
        device_obs = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        device_dice = torch.as_tensor(dice_arr, dtype=torch.float32, device=self.device)
        device_cat = torch.as_tensor(cat_arr, dtype=torch.long, device=self.device)
        device_act_type = torch.as_tensor(act_type_arr, dtype=torch.long, device=self.device)
        device_old_logp = torch.as_tensor(old_logp_arr, dtype=torch.float32, device=self.device)
        device_adv = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        device_ret = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        N_flat = len(advantages)
        idxs = np.arange(N_flat)

        self.train()
        for epoch_idx in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N_flat, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = device_obs[mb_idx]
                mb_dice = device_dice[mb_idx]
                mb_cat = device_cat[mb_idx]
                mb_act_type = device_act_type[mb_idx]
                mb_old_logp = device_old_logp[mb_idx]
                mb_adv = device_adv[mb_idx]
                mb_ret = device_ret[mb_idx]

                new_logp = self._compute_log_prob(mb_obs, mb_dice, mb_cat, mb_act_type)
                _, _, values_t = self.policy(mb_obs)
                entropy = self._compute_entropy(mb_obs, mb_act_type)

                ratio = (new_logp - mb_old_logp).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * F.mse_loss(values_t, mb_ret)
                entropy_term = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
