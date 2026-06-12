import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.fc(x)


def preprocess(state):
    return np.array(state).flatten()


def main():
    env = gym.make("highway-v0", render_mode="human")

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    model = DQN(state_size, action_size)

    model.load_state_dict(
        torch.load("models/dqn_highway.pth")
    )

    model.eval()

    state, info = env.reset()
    state = preprocess(state)

    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        state = preprocess(next_state)
        total_reward += reward
        done = terminated or truncated

        print("Action:", action, "Reward:", reward)

    print("Evaluation finished.")
    print("Total reward:", total_reward)

    env.close()


if __name__ == "__main__":
    main()