import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "results/training_rewards.csv",
    header=None,
    names=["Episode", "Reward", "Epsilon"]
)

# 20-episode moving average
data["MovingAverage"] = data["Reward"].rolling(window=20).mean()

plt.figure(figsize=(10, 6))

# Raw rewards
plt.plot(
    data["Episode"],
    data["Reward"],
    alpha=0.25,
    label="Reward"
)

# Smoothed trend
plt.plot(
    data["Episode"],
    data["MovingAverage"],
    linewidth=3,
    label="20-Episode Average"
)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training Reward")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/reward_curve.png")

print("Graph saved to results/reward_curve.png")