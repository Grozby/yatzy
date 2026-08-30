import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import time
import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from src.game.environment import YatzyEnvironment
from src.training.agent import PPOAgent
from src.training.buffer import RolloutBuffer, compute_gae


def make_env(env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}
    return YatzyEnvironment(**env_kwargs)


def train_ppo(
    cfg: DictConfig | dict | None = None,
    **kwargs,
):
    # If no config provided, load from default values or kwargs
    if cfg is not None:
        if isinstance(cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        else:
            cfg_dict = cfg
        
        training_cfg = cfg_dict.get("training", {})
        model_cfg = cfg_dict.get("model", {})
        env_cfg = cfg_dict.get("env", {})
    else:
        training_cfg = {}
        model_cfg = {}
        env_cfg = {}

    # Merge with explicit kwargs if passed
    num_envs = kwargs.get("num_envs", training_cfg.get("num_envs", 64))
    num_epochs = kwargs.get("num_epochs", training_cfg.get("num_epochs", 2000))
    rollout_steps = kwargs.get("rollout_steps", training_cfg.get("rollout_steps", 256))
    batch_size = kwargs.get("batch_size", training_cfg.get("batch_size", 512))
    gamma = kwargs.get("gamma", training_cfg.get("gamma", 0.995))
    lam = kwargs.get("lam", training_cfg.get("lam", 0.95))
    clip_eps = kwargs.get("clip_eps", training_cfg.get("clip_eps", 0.1))
    value_coef = kwargs.get("value_coef", training_cfg.get("value_coef", 0.5))
    entropy_coef = kwargs.get("entropy_coef", training_cfg.get("entropy_coef", 0.01))
    lr = kwargs.get("lr", training_cfg.get("lr", 3e-4))
    min_lr = kwargs.get("min_lr", training_cfg.get("min_lr", 5e-5))
    min_entropy = kwargs.get("min_entropy", training_cfg.get("min_entropy", 0.005))
    save_interval = kwargs.get("save_interval", training_cfg.get("save_interval", 100))
    eval_interval = kwargs.get("eval_interval", training_cfg.get("eval_interval", 50))
    experiments_base_dir = kwargs.get("experiments_base_dir", training_cfg.get("experiments_base_dir", "experiments"))

    hidden = kwargs.get("hidden", model_cfg.get("hidden", 256))
    
    env_kwargs = {
        "num_dices": kwargs.get("num_dices", env_cfg.get("num_dices", 5)),
        "max_rolls": kwargs.get("max_rolls", env_cfg.get("max_rolls", 3)),
        "upper_section_score_threshold": kwargs.get(
            "upper_section_score_threshold", env_cfg.get("upper_section_score_threshold", 63)
        ),
        "upper_section_bonus_points": kwargs.get(
            "upper_section_bonus_points", env_cfg.get("upper_section_bonus_points", 50)
        ),
    }

    # Setup timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(experiments_base_dir) / f"run_{timestamp}"
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")

    # Save configuration
    full_config = {
        "training": {
            "num_envs": num_envs,
            "num_epochs": num_epochs,
            "rollout_steps": rollout_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "lam": lam,
            "clip_eps": clip_eps,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "lr": lr,
            "min_lr": min_lr,
            "min_entropy": min_entropy,
            "save_interval": save_interval,
            "eval_interval": eval_interval,
            "experiments_base_dir": experiments_base_dir,
        },
        "model": {
            "hidden": hidden,
        },
        "env": env_kwargs,
        "timestamp": timestamp,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=4)

    # We use AsyncVectorEnv with context="fork" to prevent PyTorch MPS multiprocessing lockup on MacOS.
    # We must create the environments BEFORE checking torch.backends.mps.is_available()!
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(env_kwargs) for _ in range(num_envs)],
        context="fork",
    )

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Environments: {num_envs}")

    single_env = YatzyEnvironment(**env_kwargs)

    obs_dim = single_env.observation_space.shape[0]
    num_dices = single_env.num_dices
    num_categories = single_env.num_categories

    agent = PPOAgent(
        obs_dim=obs_dim,
        num_dices=num_dices,
        num_categories=num_categories,
        hidden=hidden,
        lr=lr,
        device=device,
    )

    buffer = RolloutBuffer(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        num_dices=num_dices,
    )

    global_step = 0
    obs, _ = envs.reset()
    running_mean = None
    alpha = 0.1
    history_logs = []

    best_running_mean = -float("inf")
    best_epoch = 0
    best_rollout_mean = 0.0

    for epoch in range(num_epochs):
        t0 = time.time()
        buffer.reset()
        total_scores = []

        # -----------------------------
        # 1. Rollout Phase
        # -----------------------------
        for _ in range(rollout_steps):
            action, logp, value, _ = agent.select_action(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(action)
            dones = np.logical_or(terminations, truncations)

            # Scale rewards by 0.02 to prevent massive value errors triggering clip_grad_norm constantly
            scaled_rewards = rewards / 50.0
            buffer.add(obs, action, scaled_rewards, dones, value, logp)
            obs = next_obs
            global_step += num_envs

            if isinstance(infos, dict):
                if "final_info" in infos:
                    for idx, is_term in enumerate(terminations):
                        if is_term:
                            info_dict = infos["final_info"][idx]
                            if info_dict is not None and "total_score" in info_dict:
                                total_scores.append(info_dict["total_score"])
                else:
                    for idx, is_term in enumerate(terminations):
                        if is_term:
                            if "total_score" in infos and len(infos["total_score"]) > idx:
                                has_val_array = infos.get("_total_score", None)
                                if has_val_array is None or has_val_array[idx]:
                                    total_scores.append(infos["total_score"][idx])
            elif isinstance(infos, (tuple, list)):
                for idx, is_term in enumerate(terminations):
                    if is_term and infos[idx] is not None and "total_score" in infos[idx]:
                        total_scores.append(infos[idx]["total_score"])

        current_mean = float(np.mean(total_scores)) if total_scores else 0.0
        if total_scores:
            if running_mean is None:
                running_mean = current_mean
            else:
                running_mean = (1 - alpha) * running_mean + alpha * current_mean

        display_running_mean = running_mean if running_mean is not None else 0.0
        
        # Check for best performing epoch so far
        is_new_best = False
        if running_mean is not None and running_mean > best_running_mean:
            best_running_mean = running_mean
            best_epoch = epoch + 1
            best_rollout_mean = current_mean
            is_new_best = True

            # Save best checkpoint and metadata
            best_model_path = exp_dir / "best_model.pt"
            torch.save(agent.policy.state_dict(), best_model_path)
            best_info = {
                "best_epoch": best_epoch,
                "best_running_mean": round(best_running_mean, 2),
                "best_rollout_mean": round(best_rollout_mean, 2),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(exp_dir / "best_model_info.json", "w") as f:
                json.dump(best_info, f, indent=4)

        best_tag = f" 🏆 [Best: {best_running_mean:.2f} @ Ep {best_epoch}]" if best_epoch > 0 else ""
        print(
            f"Epoch {epoch+1}/{num_epochs} | Rollout mean: {current_mean:.2f} | "
            f"running_mean: {display_running_mean:.2f}{best_tag} | Ep finishes: {len(total_scores)}"
        )

        # -----------------------------
        # 2. Advantage Computation
        # -----------------------------
        agent.eval()
        with torch.no_grad():
            _, _, last_values_t = agent.policy(torch.as_tensor(obs, dtype=torch.float32, device=device))
            last_values = last_values_t.cpu().numpy()

        advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_values, gamma, lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Learning rate and entropy schedule with minimum floor
        progress = epoch / float(num_epochs)
        current_lr = max(min_lr, lr * (1.0 - progress))
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = current_lr

        current_entropy_coef = max(min_entropy, entropy_coef * (1.0 - progress))

        # -----------------------------
        # 3. PPO Update
        # -----------------------------
        agent.update(
            buffer=buffer,
            advantages=advantages,
            returns=returns,
            batch_size=batch_size,
            ppo_epochs=4,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=current_entropy_coef,
        )

        t1 = time.time()
        N_flat = rollout_steps * num_envs
        sps = int(N_flat / (t1 - t0))
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1} done | SPS: {sps} | last_return_mean ~ {returns.mean():.2f} | lr: {current_lr:.5f}"
            )

        # Save metrics
        history_logs.append(
            {
                "epoch": epoch + 1,
                "rollout_mean": current_mean,
                "running_mean": display_running_mean,
                "best_running_mean_so_far": best_running_mean,
                "best_epoch_so_far": best_epoch,
                "ep_finishes": len(total_scores),
                "sps": sps,
                "lr": current_lr,
                "return_mean": float(returns.mean()),
            }
        )

        # Save periodic checkpoint every save_interval epochs
        if (epoch + 1) % save_interval == 0:
            ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(agent.policy.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

            # Periodically write metrics json
            with open(exp_dir / "metrics.json", "w") as f:
                json.dump(history_logs, f, indent=2)

    # Final save
    final_model_path = exp_dir / "final_model.pt"
    torch.save(agent.policy.state_dict(), final_model_path)
    torch.save(agent.policy.state_dict(), "ppo_yatzy.pt")  # Maintain root fallback
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(history_logs, f, indent=2)

    print(f"\nTraining completed! All models and metrics saved to: {exp_dir}")
    print(f"🏆 Best Performance: running_mean = {best_running_mean:.2f} achieved at Epoch {best_epoch}")
    print(f"   Best model saved at: {exp_dir / 'best_model.pt'}")
    envs.close()
    return agent


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("--- Loaded Hydra Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    train_ppo(cfg)


if __name__ == "__main__":
    main()
