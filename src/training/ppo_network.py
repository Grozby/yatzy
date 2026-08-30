import torch
import torch.nn as nn

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, num_dices: int, num_categories: int, hidden: int = 256):
        super().__init__()
        self.num_dices = num_dices
        self.num_categories = num_categories

        self.actor_dice = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        
        self.actor_cat = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.critic_base = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Heads
        self.dice_head = nn.Linear(hidden, num_dices)  # 5 factorized Bernoulli logits
        self.cat_head = nn.Linear(hidden, num_categories)  # 15 Categorical logits
        self.value_head = nn.Linear(hidden, 1)  # state value

    def forward(self, obs: torch.Tensor):
        """
        obs: (batch, obs_dim)
        returns:
            dice_logits: (batch, num_dices)
            cat_logits:  (batch, num_categories)
            values:      (batch,)
        """
        dice_feat = self.actor_dice(obs)
        cat_feat = self.actor_cat(obs)
        
        dice_logits = self.dice_head(dice_feat)
        cat_logits = self.cat_head(cat_feat)

        # Action Masking: Filled categories are explicitly set to heavily negative logits
        # They exist in obs space slice: [self.num_dices + 6 : self.num_dices + 6 + self.num_categories]
        filled_start = self.num_dices + 6
        filled_mask = obs[..., filled_start : filled_start + self.num_categories]
        cat_logits = cat_logits - filled_mask * 1e9

        critic_feat = self.critic_base(obs)
        values = self.value_head(critic_feat).squeeze(-1)
        
        return dice_logits, cat_logits, values