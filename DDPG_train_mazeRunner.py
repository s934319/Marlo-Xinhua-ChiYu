import marlo

import os
import time

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch

from assets.Utils import CreatFolder, random_agent_obs_mean_std
from assets.EnvWrapper import FrameStack, WarpFrame, StackTranspose
from assets.DDPG import DDPG



def main():
    folder_path = "./mazeRunner"
    CreatFolder(folder_path)
    file_path = os.path.join(folder_path, "data")
    

    # 每幾個episode儲存一次model權重
    save_steps = 5000


    # 目標訓練的episode數
    n_steps = 1000000


    update_interval = 4


    n_frameStack = 4

    GAMMA = 0.995
    TAU = 1e-2
    ACTOR_LR = 1e-4
    CRITIC_LR = 1e-3
    BATCH_SIZE = 128

    EPSILON = 0.9

    MEMORY_SIZE = 20000
    MEMORY_WARMUP_SIZE = 2000


    opticalFlow_ICM = True
    FLOW_LR = 1e-6
    EXT_R_COE = 0.5
    INT_R_COE = 0.5
    flow_update_interval = 4


    load = False




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    t = time.localtime()
    writer = SummaryWriter(os.path.join(folder_path, "runs", time.strftime("%m.%d.%Y_%H.%M.%S", t)))

    # 設定環境
    client_pool = [('127.0.0.1', 10000)]
    join_tokens = marlo.make('MarLo-MazeRunner-v0',
                            params={
                            "client_pool": client_pool,
                            "maze_height" : 1,
                            # "retry_sleep" : 4,
                            # "step_sleep" : 0.004,
                            # "prioritise_offscreen_rendering": False
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


    obs_mean = None
    obs_std = None
    if(not load):
        obs_mean, obs_std = random_agent_obs_mean_std(env)
        # print(obs_mean, obs_std)
    

    
    agent = DDPG(
        OBS_DIM,
        ACT_DIM,
        n_frameStack,
        device,
        obs_mean = obs_mean,
        obs_std = obs_std,
        GAMMA = GAMMA,
        A_LR = ACTOR_LR,
        C_LR = CRITIC_LR,
        TAU = TAU,
        MEMORY_SIZE = MEMORY_SIZE,
        BATCH_SIZE = BATCH_SIZE,
        EPSILON = EPSILON,
        opticalFlow_ICM = opticalFlow_ICM,
        FLOW_LR = FLOW_LR,
        EXT_R_COE = EXT_R_COE,
        INT_R_COE = INT_R_COE,
        flow_update_interval = flow_update_interval,
    )

    # continue training
    if(load):
        agent.load(file_path, False, folder_path)
        agent.actor.train()
        agent.critic.train()
        agent.feature_extractor.train()
        if(opticalFlow_ICM):
            agent.icm.train()


    saving = False
    total_steps = 0
    pbar = tqdm(total=n_steps)

    # for epi in tqdm(range(n_episodes)):
    while total_steps < n_steps:

        state_cur = env.reset()


        done = False

        total_reward = 0

        steps = 0
        while not done:
            # action = agent.choose_action(state_cur, steps)
            # action = agent.choose_action(state_cur)
            action = agent.choose_action(state_cur, total_steps/n_steps)
            # print(action)

            state_next, reward, done, info = env.step(action)
            # print(reward)

            
            if(done):
                agent.memory.append(state_cur, action, reward, state_cur, done)
            else:
                agent.memory.append(state_cur, action, reward, state_next, done)


            total_reward += reward
            state_cur = state_next
            

            # if(agent.memory.size() >= MEMORY_WARMUP_BATCH*BATCH_SIZE and (steps % update_interval) == 0):
            if(agent.memory.size() >= MEMORY_WARMUP_SIZE and (steps % update_interval) == 0):
                if(opticalFlow_ICM):
                    critic_loss, actor_loss, flow_loss = agent.update()
                    writer.add_scalar('step_flow_loss', flow_loss, total_steps)
                else:
                    critic_loss, actor_loss = agent.update()

                writer.add_scalar('step_critic_loss', critic_loss, total_steps)
                writer.add_scalar('step_actor_loss', actor_loss, total_steps)

            if(total_steps % save_steps == 0):
                saving = True

            steps += 1
            total_steps += 1
            pbar.update(1)


        # agent.update_noise()

        # if(epi % save_episode == 0):
        if(saving):
            agent.save(file_path + "_temp_" + str(total_steps), folder_path)
            saving = False
            
        
        writer.add_scalar('steps_per_episode', steps, total_steps)
        writer.add_scalar('total_reward', total_reward, total_steps)
        print("steps per episode: ", steps)
        print("agent's reward: ", total_reward)
       

    agent.save(file_path, folder_path)
    pbar.close()
    env.close()


if __name__ == "__main__":
    main()