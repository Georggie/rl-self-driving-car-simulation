import gymnasium as gym
import highway_env
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os


# =========================
# Q NETWORK
# =========================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)


# =========================
# HYPERPARAMETERS
# =========================
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 50000
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


# =========================
# MAIN TRAIN LOOP
# =========================
def main():
    env = gym.make("highway-v0")

    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    q_network = DQN(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=LR)

    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON

    def preprocess(state):
        return np.array(state).flatten()

        os.makedirs("results", exist_ok = True)
        os.makedirs("models", exist_ok = True)

        with open("results/training_rewards.csv", "w", newline = "") as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "reward", "epsilon"])

    for episode in range(50):

        state, _ = env.reset()
        state = preprocess(state)

        done = False
        total_reward = 0

        while not done:

            # =========================
            # ACTION (epsilon-greedy)
            # =========================
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state)
                with torch.no_grad():
                    action = torch.argmax(q_network(state_tensor)).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(next_state)

            done = terminated or truncated

            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            # =========================
            # TRAINING STEP
            # =========================
            if len(memory) > BATCH_SIZE:

                batch = random.sample(memory, BATCH_SIZE)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

                next_q_values = q_network(next_states).max(1)[0]

                target = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

        with open("results/training_rewards.csv", "a", newline = "") as file:
            writer = csv.writer(file)
            writer.writerow([episode, total_reward, epsilon])

        if episode % 10 == 0:
            torch.save(
                q_network.state_dict(),
                "models/dqn_highway.pth"
            )    

            print("Checkpoint saved")

    env.close()


if __name__ == "__main__":
    main()