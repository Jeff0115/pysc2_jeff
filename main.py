import sys
import os
import shutil
import argparse
from functools import partial
from pre_processing import Preprocessor as pro
import numpy
import time
from pysc2.lib import actions

from a2c import RandomAgent,A2C
import torch


from environment import SubprocVecEnv, make_sc2env, SingleEnv
from test import test_model
from ppo import PPO
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])




def main():
    map_name='DefeatRoaches'
    envs_num=8
    max_windows=1
    total_updates=-1
    env_args = dict(
        map_name=map_name,
        battle_net_map=False,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=32,
            feature_minimap=32,
            rgb_screen=None,
            rgb_minimap=None,
            action_space=None,
            use_feature_units=False,
            use_raw_units=False),
        step_mul=8,
        game_steps_per_episode=None,
        disable_fog=False,
        visualize=False
    )
    vis_env_args = env_args.copy()
    vis_env_args['visualize'] = True
    num_vis = min(envs_num, max_windows)
    env_fns = [partial(make_sc2env, **vis_env_args)] * num_vis
    num_no_vis = envs_num - num_vis
    if num_no_vis > 0:
      env_fns.extend([partial(make_sc2env, **env_args)] * num_no_vis)
    envs = SubprocVecEnv(env_fns)
    # 一个随机的实现方式 用来debug
    '''agents=[]
    for i in range(envs_num):
        agent=RandomAgent()
        agents.append(agent)'''

    '''observation_spec = envs.observation_spec()
    action_spec = envs.action_spec()
    processor = pro(observation_spec)
    for agent,obs_spec,act_spec in zip(agents,observation_spec,action_spec):
        agent.setup(obs_spec[0],act_spec[0])
    try:
        while True:
            num_frames=0
            timesteps= envs.reset()
            for a in agents:
                a.reset()
            while True:
                num_frames+=1
                last_timesteps=timesteps
                actions= [agent.step(timestep) for agent,timestep in zip(agents,timesteps)]
                timesteps=envs.step(actions)
                obs=processor.preprocess_obs(timesteps)
                a=1
    except KeyboardInterrupt:
        pass'''
    while True:
        test_mark=0
        better=0
        agent=PPO(envs)
        agent.reset()
        # agent.net.load_state_dict(torch.load('./save/episode311_score36.2.pkl'))
        #try:
        while True:
            agent.train()
            if agent.sum_episode%120<60:
                test_mark=0
            if agent.sum_episode%120>=60 and not test_mark:
                # print("###### I'm in!")
                test_mark=1
                mean_score, _ = test_model(agent)
                if mean_score > 36 + better:
                    better+=1
                    if better>70:
                        better=70
                    torch.save(agent.net.state_dict(), './save/episode' +
                                str(agent.sum_episode)+'_score'+str(mean_score)+'.pkl')
                if mean_score<20+0.01*agent.sum_episode:
                    print("############################\n\n\n")
                    break

    #except :
        #print(agent.last_obs['available_actions'])

    envs.close()



if __name__=='__main__':
    main()