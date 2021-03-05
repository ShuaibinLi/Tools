import parl
from parl.utils import logger
import gym
import numpy as np
from gym.spaces import Box, Discrete


class ParallelEnv(object):
    """
        To use ParallelEnv, you need download this repo, and import this file locally

        Example:
        .. code-block:: python
            import parl
            from parallel_env import ParallelEnv

            parl.connect('localhost')
            env_list = ParallelEnv(env_name='BreakoutNoFrameskip-v4', env_num=5)

        Attributes:
            env_name: gym environment name
            env_num: number of environments

        Public Functions: the same as gym (mainly sample_actions, reset, step, seed ...)
                          but the inputs and returns are list of parallel environments

                          btw: you can find a single env' ObservationSpace...
                               e.g. action_space = env_list[0].action_space

        Note:
            ``ParallelEnv`` defines a remote environment wrapper for running the environment remotely and
            enables large-scale parallel collection of environmental data.

            Support both Continuous action space and Discrete action space environments.

        """

    def __init__(self, env_name, env_num):
        self.env_list = [RemoteEnv(env_name=env_name) for _ in range(env_num)]
        self.env_num = env_num

        self.episode_steps_list = [0] * env_num
        self._max_episode_steps = self.env_list[0]._max_episode_steps

    def sample_actions(self):
        action_list = [env.sample_action() for env in self.env_list]
        action_list = [action.get() for action in action_list]
        action_list = np.array(action_list)
        return action_list

    def reset(self):
        obs_list = [env.reset() for env in self.env_list]
        obs_list = [obs.get() for obs in obs_list]
        obs_list = np.array(obs_list)
        return obs_list

    def step(self, action_list):
        return_list = [
            self.env_list[i].step(int(action_list[i]))
            for i in range(self.env_num)
        ]
        return_list = [return_.get() for return_ in return_list]
        return_list = np.array(return_list, dtype=object)

        next_obs_list = np.array([next_obs for next_obs in return_list[:, 0]])
        reward_list = np.array([reward for reward in return_list[:, 1]])
        done_list = return_list[:, 2]
        info_list = return_list[:, 3]

        for i in range(self.env_num):
            self.episode_steps_list[i] += 1
            info_list[i]['timeout'] = False

            if done_list[i] or self.episode_steps_list[
                    i] >= self._max_episode_steps:
                if self.episode_steps_list[i] >= self._max_episode_steps:
                    info_list[i]['timeout'] = True
                self.episode_steps_list[i] = 0
                obs_list_i = self.env_list[i].reset()
                next_obs_list[i] = obs_list_i.get()
                next_obs_list[i] = np.array(next_obs_list[i])
        return next_obs_list, reward_list, done_list, info_list

    def seed(self, seed_list):
        for i in range(self.env_num):
            self.env_list[i].seed(seed_list[i])

    def render(self):
        return logger.warning(
            'Can not render in remote environment, render() have been skipped.'
        )


@parl.remote_class(wait=False)
class RemoteEnv(object):
    def __init__(self, env_name):
        assert isinstance(env_name, str)

        class ActionSpace(object):
            def __init__(self,
                         action_space=None,
                         low=None,
                         high=None,
                         shape=None,
                         n=None):
                self.action_space = action_space
                self.low = low
                self.high = high
                self.shape = shape
                self.n = n

        class ObservationSpace(object):
            def __init__(self, observation_space, low, high, shape=None):
                self.observation_space = observation_space
                self.low = low
                self.high = high
                self.shape = shape

        self.env = gym.make(env_name)
        if hasattr(self.env, '_max_episode_steps'):
            self._max_episode_steps = int(self.env._max_episode_steps)
        try:
            self._elapsed_steps = int(self.env._elapsed_steps)
        except:
            logger.error('object has no attribute _elspaed_steps')

        self.observation_space = ObservationSpace(
            self.env.observation_space, self.env.observation_space.low,
            self.env.observation_space.high, self.env.observation_space.shape)
        if isinstance(self.env.action_space, Discrete):
            self.action_space = ActionSpace(n=self.env.action_space.n)
        elif isinstance(self.env.action_space, Box):
            self.action_space = ActionSpace(
                self.env.action_space, self.env.action_space.low,
                self.env.action_space.high, self.env.action_space.shape)

    def sample_action(self):
        return self.env.action_space.sample()

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)
