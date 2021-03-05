from parallel_env import ParallelEnv
import parl
import argparse
import numpy as np
from parl.utils import logger


# example of ParallelEnv
def parallel_env():
    logger.info("Running example of RemoteEnv in atari_env: {}".format(
        args.atari_env))

    parl.connect('localhost')
    env_list = ParallelEnv(env_name=args.atari_env, env_num=args.env_num)

    seed_list = [np.random.randint(0, 100) for i in range(args.env_num)]
    env_list.seed(seed_list)

    obs_list = env_list.reset()
    episode_reward_list = [0] * args.env_num
    steps_list = [0] * args.env_num
    episodes = 0
    # Run episodes with a random policy
    while episodes < args.max_episodes:

        action_list = env_list.sample_actions()
        next_obs_list, reward_list, done_list, info_list = env_list.step(
            action_list)
        for i in range(args.env_num):
            steps_list[i] += 1
            episode_reward_list[i] += reward_list[i]
            if done_list[i]:
                episodes += 1
                logger.info(
                    'Env{} done, total_steps {}, episode_reward {}'.format(
                        i, steps_list[i], episode_reward_list[i]))
        obs_list = next_obs_list


def main():
    parallel_env()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atari_env",
        default='BreakoutNoFrameskip-v4',
        help='OpenAI gym/atari environment name')
    parser.add_argument(
        "--env_num", default=2, type=int, help='number of environment')
    parser.add_argument(
        "--max_episodes", default=2, type=int, help='episode of running')

    args = parser.parse_args()

    main()
