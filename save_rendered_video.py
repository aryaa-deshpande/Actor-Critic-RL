import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics


def run_and_record_one_episode(env_id: str, policy, out_dir="videos", max_steps=10000):
    """
    env_id: e.g. "CartPole-v1"
    policy: callable taking observation -> greedy action (Your code implements this)
    """
    os.makedirs(out_dir, exist_ok=True)

    # MUST create env with a render_mode that returns images
    env = gym.make(env_id, render_mode="rgb_array")

    # track episode stats (reward, length) and record video (only first episode)
    env = RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder=out_dir, name_prefix="greedy_eval",
                      episode_trigger=lambda e_index: e_index == 0)

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        # Replace the following with their agent's greedy action:
        #   action = policy(obs)                 # if policy is a function
        #   action = policy.act(obs, True)       # if policy object with .act
        action = policy(obs)  # <-- adapt to your agent's API

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated or steps >= max_steps:
            break

    env.close()
    print(f"Saved video in {out_dir}. Episode reward: {total_reward}, steps: {steps}")
