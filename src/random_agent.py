import gymnasium as gym
import highway_env


# =========================
# SAFE DRIVING POLICY
# =========================
def choose_action(obs):
    """
    Rule-based driving policy using kinematics observation.
    Works with highway-env structured state.
    """

    ego = obs[0]

    ego_x = ego[1]
    ego_y = ego[2]

    min_front_distance = float("inf")
    ego_lane = ego_y  # approximate lane proxy

    left_clear = True
    right_clear = True

    # Analyze surrounding vehicles
    for car in obs[1:]:
        presence = car[0]
        if presence < 0.5:
            continue

        x = car[1]
        y = car[2]

        # car ahead
        if x > ego_x:
            dist = x - ego_x
            min_front_distance = min(min_front_distance, dist)

        # lane blocking approximation
        if abs(x - ego_x) < 0.2:
            if y < ego_y:
                left_clear = False
            elif y > ego_y:
                right_clear = False

    # =========================
    # DECISION LOGIC
    # =========================

    # 1. Emergency brake
    if min_front_distance < 0.4:
        return 4  # brake

    # 2. Lane change logic
    if not left_clear and right_clear:
        return 2  # move right

    if not right_clear and left_clear:
        return 0  # move left

    # 3. Default behavior
    return 3  # accelerate


# =========================
# MAIN LOOP
# =========================
def main():
    env = gym.make(
    "highway-v0",
    render_mode="human",
    config={
        "lanes_count": 4,          # keep moderate
        "vehicles_count": 15,      # stable baseline
        "vehicles_density": 1.0,   # IMPORTANT: stable traffic
        "observation": {
            "type": "Kinematics"
        },
        "action": {
            "type": "DiscreteMetaAction"
        }
    }
)

    obs, info = env.reset()

    print("Environment initialized!")
    print("Observation shape:", obs.shape)

    done = False

    while not done:
        action = choose_action(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        print("Reward:", reward)

    env.close()


if __name__ == "__main__":
    main()