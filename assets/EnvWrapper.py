import numpy as np
import cv2
import gym
from gym import spaces
from collections import deque


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, gray=False):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.gray = gray
        if(self.gray):
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                            dtype=env.observation_space.dtype)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3),
                                                dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame
        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        if(self.gray):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = frame.reshape(frame.shape[0], frame.shape[1], 1)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame


class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """Stack n_frames last frames.

    (don't use lazy frames)
    modified from:
    stable_baselines.common.atari_wrappers

    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(n_frames, shp[0], shp[1], shp[2]),
                                        dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_ob()

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.frames.append(obs)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.n_frames
    # return np.concatenate(list(self.frames), axis=3)
    ob = np.array(list(self.frames))
    ob = ob.reshape(self.observation_space.shape)
    return ob
    # return np.array(list(self.frames))
    # frameStack, H, W, C



class StackTranspose(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0]*shp[3], shp[1], shp[2]),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        return self._get_ob(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._get_ob(obs), reward, done, info


    def _get_ob(self, obs):
        # (frameStack, H, W, C)
        obs = obs.transpose((0, 3, 1, 2))
        # (frameStack, C, H, W)
        obs = obs.reshape(*self.observation_space.shape)
        # (frameStack*C, H, W)
        return obs


class ActionReshape(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        # print(action)
        action = np.argmax(action)
        # print(action)
        return self.env.step(action)
