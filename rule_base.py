import torch
import torch.nn as nn
import torch.optim as optim
import os
from functools import partial
from pre_processing import Preprocessor, is_spatial_action, stack_ndarray_dicts
from pysc2.lib.actions import TYPES as ACTION_TYPES
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from torch.distributions.categorical import Categorical
import random
import numpy
import torch.optim as optim
import time
from pysc2.lib import actions
from pysc2.agents import base_agent
from net import CNN
import copy


from pysc2.lib import features
# pysc2.agents.scripted_agent import CollectMineralShards
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

from environment import SubprocVecEnv, make_sc2env, SingleEnv


def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
    return {k: flatten_first_dims(v) for k, v in x.items()}


def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))
class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
      if not minerals:
        return FUNCTIONS.no_op()
      marines = _xy_locs(player_relative == _PLAYER_SELF)
      marine_xy = numpy.mean(marines, axis=0).round()  # Average location.
      distances = numpy.linalg.norm(numpy.array(minerals) - marine_xy, axis=1)
      closest_mineral_xy = minerals[numpy.argmin(distances)]
      x,y=closest_mineral_xy

      return 331,[[0],[x,y]]
    else:
      return 7,[[0]]
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['main.py'])

def RuleBase(net):
    map_name = 'CollectMineralShards'
    total_episodes=100
    total_updates = -1
    sum_score=0
    n_steps=8
    learning_rate=1e-4
    optimizer = optim.Adam(
        net.parameters(), learning_rate, weight_decay=0.01)
    env = make_sc2env(
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
        visualize=True
    )

    processor = Preprocessor(env.observation_spec()[0])
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent = CollectMineralShards()
    episodes=0
    agent.reset()
    timesteps=env.reset()
    while True:
        fn_ids=[]
        args_ids=[]
        observations=[]
        for step in range(n_steps):
            a_0,a_1=agent.step(timesteps[0])
            obs = processor.preprocess_obs(timesteps)
            observations.append(obs)
            actions=FunctionCall(a_0,a_1)
            fn_id = torch.LongTensor([a_0]).cuda()
            args_id = {}
            if a_0 == 7:
                for type in ACTION_TYPES:
                    if type.name == 'select_add':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 331:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            action = (fn_id, args_id)
            fn_ids.append(fn_id)
            args_ids.append(args_id)
            timesteps = env.step([actions])
            if timesteps[0].last():
                i=timesteps[0]
                score = i.observation['score_cumulative'][0]
                sum_score += score
                episodes += 1
                if episodes%50==0:
                    torch.save(net.state_dict(), './save/episode2' +str(episodes)
                               +str('.pkl'))
                print("episode %d: score = %f" % (episodes, score))

        observations = flatten_first_dims_dict(
            stack_ndarray_dicts(observations))

        train_fn_ids=torch.cat(fn_ids)
        train_arg_ids={}

        for k in args_ids[0].keys():
            temp=[]
            temp=[d[k] for d in args_ids]

            train_arg_ids[k]=torch.cat(temp,dim=0)

        screen = torch.FloatTensor(observations['screen']).cuda()
        minimap = torch.FloatTensor(observations['minimap']).cuda()
        flat = torch.FloatTensor(observations['flat']).cuda()
        policy, _ = net(screen, minimap, flat)

        fn_pi,args_pi=policy
        available_actions=torch.FloatTensor(observations['available_actions']).cuda()
        function_pi = available_actions * fn_pi
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        Loss=nn.CrossEntropyLoss(reduction='none')
        loss=Loss(function_pi,train_fn_ids)

        for type in train_arg_ids.keys():
            id=train_arg_ids[type]
            pi=args_pi[type]
            arg_loss_list = []
            for i, p in zip(id, pi):
                if i == -1:
                    temp = torch.zeros((1)).cuda()
                else:
                    a = torch.LongTensor([i]).cuda()
                    b = torch.unsqueeze(p, dim=0).cuda()
                    temp = Loss(b, a)
                arg_loss_list.append(temp)

            arg_loss=torch.cat(arg_loss_list)
            loss+= arg_loss
        loss=loss.mean()
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if episodes >=total_episodes:
            break
    torch.save(net.state_dict(), './save/episode1' +
               str('.pkl'))

def RuleBasevalue(net):
    map_name = 'CollectMineralShards'
    value_coef=0.25
    total_episodes = 20
    total_updates = -1
    sum_score = 0
    n_steps = 8
    learning_rate = 1e-4
    optimizer = optim.Adam(
        net.parameters(), learning_rate, weight_decay=0.01)
    env = make_sc2env(
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
        visualize=True
    )

    processor = Preprocessor(env.observation_spec()[0])
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent = CollectMineralShards()
    episodes = 0
    agent.reset()
    timesteps = env.reset()
    while True:
        fn_ids = []
        args_ids = []
        observations = []
        rewards=[]
        dones=[]
        for step in range(n_steps):
            a_0, a_1 = agent.step(timesteps[0])
            obs = processor.preprocess_obs(timesteps)
            observations.append(obs)
            actions = FunctionCall(a_0, a_1)
            fn_id = torch.LongTensor([a_0]).cuda()
            args_id = {}
            if a_0 == 7:
                for type in ACTION_TYPES:
                    if type.name == 'select_add':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            elif a_0 == 331:
                for type in ACTION_TYPES:
                    if type.name == 'queued':
                        args_id[type] = torch.LongTensor([a_1[0][0]]).cuda()
                    elif type.name == 'screen':

                        args_id[type] = torch.LongTensor([a_1[1][1] * 32 + a_1[1][0]]).cuda()
                    else:
                        args_id[type] = torch.LongTensor([-1]).cuda()
            action = (fn_id, args_id)
            fn_ids.append(fn_id)
            args_ids.append(args_id)
            timesteps = env.step([actions])
            rewards.append(torch.FloatTensor([timesteps[0].reward]).cuda())
            dones.append(torch.IntTensor([timesteps[0].last()]).cuda())

            if timesteps[0].last():
                i = timesteps[0]
                score = i.observation['score_cumulative'][0]
                sum_score += score
                episodes += 1
                if episodes%50==0:
                    torch.save(net.state_dict(), './save/episode' +str(episodes)
                               +str('.pkl'))
                print("episode %d: score = %f" % (episodes, score))
            #obs = processor.preprocess_obs(timesteps)
            #observations.append(obs)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        with torch.no_grad():
            obs = processor.preprocess_obs(timesteps)
            screen = torch.FloatTensor(obs['screen']).cuda()
            minimap = torch.FloatTensor(obs['minimap']).cuda()
            flat = torch.FloatTensor(obs['flat']).cuda()
            _,next_value = net(screen, minimap, flat)

        observations = flatten_first_dims_dict(
            stack_ndarray_dicts(observations))

        train_fn_ids = torch.cat(fn_ids)
        train_arg_ids = {}

        for k in args_ids[0].keys():
            temp = []
            temp = [d[k] for d in args_ids]

            train_arg_ids[k] = torch.cat(temp, dim=0)

        screen = torch.FloatTensor(observations['screen']).cuda()
        minimap = torch.FloatTensor(observations['minimap']).cuda()
        flat = torch.FloatTensor(observations['flat']).cuda()
        policy, value = net(screen, minimap, flat)

        returns=torch.zeros((rewards.shape[0]+1,),dtype=float)
        returns[-1] = next_value
        for i in reversed(range(rewards.shape[0])):
            next_rewards=0.999*returns[i+1]*(1-dones[i])
            returns[i]=rewards[i]+next_rewards
        returns=returns[:-1].cuda()

        fn_pi, args_pi = policy
        available_actions = torch.FloatTensor(observations['available_actions']).cuda()
        function_pi = available_actions * fn_pi
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        Loss = nn.CrossEntropyLoss(reduction='none')
        policy_loss = Loss(function_pi, train_fn_ids)

        for type in train_arg_ids.keys():
            id = train_arg_ids[type]
            pi = args_pi[type]
            arg_loss_list = []
            for i, p in zip(id, pi):
                if i == -1:
                    temp = torch.zeros((1)).cuda()
                else:
                    a = torch.LongTensor([i]).cuda()
                    b = torch.unsqueeze(p, dim=0).cuda()
                    temp = Loss(b, a)
                arg_loss_list.append(temp)

            arg_loss = torch.cat(arg_loss_list)
            policy_loss += arg_loss
        policy_loss = policy_loss.mean()
        value_loss = (returns - value).pow(2).mean()
        print(policy_loss,value_loss)
        loss=policy_loss+value_coef*value_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if episodes >= total_episodes:
            break
    torch.save(net.state_dict(), './save/episode2' +
               str('.pkl'))
