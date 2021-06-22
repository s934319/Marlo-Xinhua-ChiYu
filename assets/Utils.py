import os
import errno
import numpy as np

from tqdm import tqdm


def CreatFolder(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        # directory already exists
        pass
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise



def random_agent_obs_mean_std(env, nsteps=10000):
    print("random_agent_obs_mean_std")
    ob = np.asarray(env.reset())
    obs = [ob]
    for _ in tqdm(range(nsteps)):
        ac = env.action_space.sample()
        ob, _, done, _ = env.step(ac)
        if done:
            ob = env.reset()
        obs.append(np.asarray(ob))
    # print(np.asarray(ob).shape)
    mean = np.mean(obs, 0).astype(np.float32)
    std = np.std(obs, 0).mean().astype(np.float32)

    return mean, std
