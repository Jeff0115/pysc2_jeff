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

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])




def main():
    map_name='CollectMineralShards'
    envs_num=2
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
        agent=A2C(envs)
        agent.reset()
        # agent.net.load_state_dict(torch.load('./save/episode40_score20.pkl'))
        #try:
        while True:
            agent.train()
            if agent.sum_episode>40 or ((agent.sum_episode%6==5) and agent.last_score<30):
                print("############################\n\n\n")
                break

    #except :
        #print(agent.last_obs['available_actions'])

    envs.close()



if __name__=='__main__':
    main()