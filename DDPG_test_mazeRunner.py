import marlo
import os

import time
import torch

from assets.DDPG import DDPG
from assets.EnvWrapper import FrameStack, WarpFrame, StackTranspose




folder_path = "./mazeRunner"
file_path = os.path.join(folder_path, "data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


n_frameStack = 4
opticalFlow_ICM = True



# 設定環境
client_pool = [('127.0.0.1', 10000)]
join_tokens = marlo.make('MarLo-MazeRunner-v0',
                        params={
                        "client_pool": client_pool,
                        "maze_height" : 1,
                        # "retry_sleep" : 4,
                        # "step_sleep" : 0.004,
                        "prioritise_offscreen_rendering": False
                        })
# As this is a single agent scenario,
# there will just be a single token
assert len(join_tokens) == 1
join_token = join_tokens[0]

env = marlo.init(join_token)
env = WarpFrame(env)
env = FrameStack(env, n_frameStack)
env = StackTranspose(env)

OBS_DIM = env.observation_space.shape
# ACT_DIM = env.action_space.n
ACT_DIM = 7
print("OBS_DIM", OBS_DIM)
print("ACT_DIM", ACT_DIM)


agent = DDPG(
    OBS_DIM,
    ACT_DIM,
    n_frameStack,
    device,
    opticalFlow_ICM = opticalFlow_ICM,
    TESTING=True,
)

agent.load(file_path, False, folder_path)
agent.actor.eval()
agent.critic.eval()
agent.feature_extractor.eval()
if(opticalFlow_ICM):
    agent.icm.eval()


obs_cur = env.reset()

done = False
reward = 0
total_reward = 0

n_round=0
while not done:

    ts = time.time()
    action = agent.act(obs_cur)
    print("action Time used: ", time.time()-ts)
    # print(reward)

    obs_next, reward, done, info = env.step(action)

    total_reward += reward


    obs_cur = obs_next

    n_round+=1

print("Rounds passed: ", n_round)
print("agent's total reward: ", total_reward)


env.close()
