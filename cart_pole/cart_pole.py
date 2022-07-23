from enum import Enum
import random 
import gym

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN


class Actions(Enum):
    DO_NOTHING = 0
    RIGHT_ENGINE = 1
    MAIN_ENGINE = 2
    LEFT_ENGINE = 3


def main():
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=42, return_info=True)

    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("lunarlander")


    for i in range(1000):
        env.render()

        #action = policy(observation)  # User-defined policy function
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)

        if i % 50:
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Done: {done}")
            print(f"Info: {info}")
            print("==================================================")

        if done:
            observation, info = env.reset(return_info=True)
    env.close()



if __name__ == "__main__":
    main()
