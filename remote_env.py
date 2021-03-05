import parl
from parl.utils import logger
import gym
from gym.spaces import Box, Discrete


@parl.remote_class
class RemoteEnv(object):
    """
    To use RemoteEnv, you need download this repo, and import this file locally

    Example:
    .. code-block:: python
        import parl
        from remote_env import RemoteEnv

        parl.connect('localhost')
        env = RemoteEnv(env_name='HalfCheetah-v1')

    Attributes:
        env_name: gym environment name

    Public Functions: the same as gym (mainly action_space, observation_space, reset, step, seed ...)

    Note:
        ``RemoteEnv`` defines a remote environment wrapper for running the environment remotely and
        enables large-scale parallel collection of environmental data.

        Support both Continuous action space and Discrete action space environments.

    """

    def __init__(self, env_name=None):
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

            def sample(self):
                return self.action_space.sample()

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

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self):
        return logger.warning(
            'Can not render in remote environment, render() have been skipped.'
        )
