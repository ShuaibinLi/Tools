from remote_env import RemoteEnv
import parl
import argparse
import numpy as np
from parl.utils import logger


# example of RemoteEnv
# for both discrete and continuous action space environment
def discrete_env():
    logger.info("Running example of RemoteEnv in continuous_env: {}".format(
        args.discrete_env))

    parl.connect('localhost')
    env = RemoteEnv(env_name=args.discrete_env)
    env.seed(1)

    action_space = env.action_space
    act_dim = action_space.n

    obs, done = env.reset(), False
    # Run an episode with a random policy
    total_steps, episode_reward = 0, 0
    while not done:
        action = np.random.choice(act_dim)
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
    logger.info('Episode done, total_steps {}, episode_reward {}'.format(
        total_steps, episode_reward))


def continuous_env():
    logger.info("Running example of RemoteEnv in continuous_env: {}".format(
        args.continuous_env))

    parl.connect('localhost')
    env = RemoteEnv(env_name=args.continuous_env)
    env.seed(0)

    obs, done = env.reset(), False
    # Run an episode with a random policy
    total_steps, episode_reward = 0, 0
    while not done:
        total_steps += 1
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
    logger.info('Episode done, total_steps {}, episode_reward {}'.format(
        total_steps, episode_reward))


def main():
    discrete_env()
    continuous_env()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continuous_env",
        default='Pendulum-v0',
        help='OpenAI gym/mujoco environment name')
    parser.add_argument(
        "--discrete_env",
        default='MountainCar-v0',
        help='OpenAI gym/mujoco environment name')

    args = parser.parse_args()

    main()
