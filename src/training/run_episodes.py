import torch
from src.training.agent import PPOAgent
from src.game.environment import YatzyEnvironment

def run_episodes(num_episodes: int = 3, device: str = "cpu"):
    env = YatzyEnvironment()
    obs_dim = env.observation_space.shape[0]

    agent = PPOAgent(
        obs_dim=obs_dim,
        num_dices=env.num_dices,
        num_categories=env.num_categories,
        device=device
    )

    # Optional: load trained weights
    try:
        state_dict = torch.load("ppo_yatzy.pt", map_location=device)
        agent.policy.load_state_dict(state_dict)
        print("Loaded trained policy from ppo_yatzy.pt")
    except FileNotFoundError:
        print("No trained policy found, running with random-initialized policy.")

    agent.eval()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _, _, _ = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = bool(terminated or truncated)

        print(f"Episode {ep + 1}: total reward = {total_reward}")

if __name__ == "__main__":
    run_episodes()
