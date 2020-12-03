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




def test_model(agent):
    map_name='MoveToBeacon'
    envs_num=1
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
    processor=pro(envs.observation_spec()[0])
    #try:

    t=0
    max_score=0
    mean_score=0
    episode=10
    while True:
        policy, value = agent.step(agent.last_obs)
        actions = agent.select_actions(policy, agent.last_obs)
        actions = agent.mask_unused_action(actions)
        size = agent.last_obs['screen'].shape[2:4]
        pysc2_action = agent.functioncall_action(actions, size)
        obs_raw = agent.envs.step(pysc2_action)
        agent.last_obs = agent.processor.preprocess_obs(obs_raw)
        for i in obs_raw:
            if i.last():
                t+=1
                score = i.observation['score_cumulative'][0]
                mean_score+=score
                print("episode %d: score = %f" % (t, score))
                if score>max_score:
                    max_score=score
        if t>=episode:
            mean_score/=episode
            print('max score=%d,average score=%d\n\n\n' % (max_score,mean_score))
            break

    #except :
        #print(agent.last_obs['available_actions'])
    envs.close()
    return mean_score, max_score