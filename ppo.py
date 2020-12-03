import torch
import torch.optim as optim
import os
from pysc2.lib.actions import TYPES as ACTION_TYPES
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from torch.distributions.categorical import Categorical
import random
import numpy as np
import time
from pysc2.lib import actions
from pysc2.agents import base_agent
from pre_processing import Preprocessor, is_spatial_action, stack_ndarray_dicts
from net import CNN
import copy

torch.cuda.set_device(0)


def printoobs_info(obs_raw):
    print(obs_raw.observation['available_actions'])
    print(obs_raw.last())


def flatten_first_dims(x):
    new_shape = [x.shape[0] * x.shape[1]] + list(x.shape[2:])
    return x.reshape(*new_shape)


def flatten_first_dims_dict(x):
    return {k: flatten_first_dims(v) for k, v in x.items()}


def stack_and_flatten_actions(lst, axis=0):
    fn_id_list, arg_dict_list = zip(*lst)
    fn_id = np.stack(fn_id_list, axis=axis)
    fn_id = flatten_first_dims(fn_id)
    arg_ids = stack_ndarray_dicts(arg_dict_list, axis=axis)
    arg_ids = flatten_first_dims_dict(arg_ids)
    return (fn_id, arg_ids)


def compute_policy_log_probs(available_actions, policy, actions):
    def compute_log_prob(probs, actions):
        cate_dist = Categorical(probs)
        log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
        return log_prob+1e-7

    available_actions = torch.FloatTensor(available_actions).cuda()
    fn_id, arg_id = actions
    fn_id = torch.FloatTensor(fn_id).cuda()
    function_pi, arg_pi = policy
    function_pi = available_actions * function_pi
    function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
    fn_log_prob = compute_log_prob(function_pi, fn_id)
    log_prob = fn_log_prob
    for type in arg_id.keys():
        id = arg_id[type]
        id = torch.FloatTensor(id).cuda()
        pi = arg_pi[type]
        arg_log_probs = torch.FloatTensor().cuda()
        for i, p in zip(id, pi):
            if i == -1:
                arg_log_prob = torch.zeros((1, 1)).cuda()
            else:
                a = torch.FloatTensor([i]).cuda()
                b = torch.unsqueeze(p, dim=0).cuda()
                arg_log_prob = compute_log_prob(b, a)
            arg_log_probs = torch.cat((arg_log_probs, arg_log_prob), dim=0)
        # mask=torch.ne(id,-1).float().unsqueeze(dim=-1).cuda()
        # arg_log_prob*=mask
        log_prob += arg_log_probs
    return log_prob


def compute_policy_entropy(available_actions, policy, actions):
    def compute_entropy(probs):
        cate_dist = Categorical(probs)
        entropy = cate_dist.entropy()
        return entropy

    available_actions = torch.FloatTensor(available_actions).cuda()
    fn_id, arg_id = actions
    fn_id = torch.FloatTensor(fn_id).cuda()
    function_pi, arg_pi = policy
    function_pi = available_actions * function_pi
    function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
    entropy = compute_entropy(function_pi).mean()

    for type in arg_id.keys():
        id = arg_id[type]
        id = torch.FloatTensor(id).cuda()
        pi = arg_pi[type]
        mask = torch.ne(id, -1).float().cuda()
        arg_entropy = (compute_entropy(pi)*mask).mean()
        entropy += arg_entropy

    return entropy




class PPO():

    def __init__(self, envs):
        self.value_loss_coefficient = 0.5
        self.entropy_coefficient = 0.05
        self.learning_rate = 1e-4
        self.envs = envs
        self.env_num=8
        self.processor = Preprocessor(self.envs.observation_spec()[0])
        self.sum_score = 0
        self.n_steps = 512
        self.gamma = 0.999
        self.clip=0.27
        self.sum_episode = 0
        self.total_updates = -1
        self.net = CNN().cuda()
        self.old_net = copy.deepcopy(self.net)
        self.old_net.cuda()
        self.epoch=4
        self.batch_size=8
        self.optimizer = optim.Adam(
            self.net.parameters(), self.learning_rate, weight_decay=0.01)

    def reset(self):
        self.obs_start = self.envs.reset()
        self.last_obs = self.processor.preprocess_obs(self.obs_start)

    def grad_step(self, observation):
        screen = torch.FloatTensor(observation['screen']).cuda()
        minimap = torch.FloatTensor(observation['minimap']).cuda()
        flat = torch.FloatTensor(observation['flat']).cuda()
        policy, value = self.net(screen, minimap, flat)
        return policy, value

    def step(self, observation):
        screen = torch.FloatTensor(observation['screen']).cuda()
        minimap = torch.FloatTensor(observation['minimap']).cuda()
        flat = torch.FloatTensor(observation['flat']).cuda()
        with torch.no_grad():
            policy, value = self.net(screen, minimap, flat)
        return policy, value
    def old_step(self,observation):
        screen = torch.FloatTensor(observation['screen']).cuda()
        minimap = torch.FloatTensor(observation['minimap']).cuda()
        flat = torch.FloatTensor(observation['flat']).cuda()
        with torch.no_grad():
            policy, value = self.old_net(screen, minimap, flat)
        return policy, value
    def select_actions(self, policy, last_obs):
        available_actions = last_obs['available_actions']

        def sample(prob):
            actions = Categorical(prob).sample()
            return actions
        function_pi, args_pi = policy
        available_actions = torch.FloatTensor(available_actions)
        function_pi = available_actions*function_pi.cpu()
        function_pi /= torch.sum(function_pi, dim=1, keepdim=True)
        try:
            function_sample = sample(function_pi)
        except:
            return 0
        args_sample = dict()
        for type, pi in args_pi.items():
            if type.name == 'queued':
                args_sample[type] = torch.zeros((self.env_num,),dtype=int)
            else:
                args_sample[type] = sample(pi).cpu()
        return function_sample, args_sample

    def mask_unused_action(self, actions):
        fn_id, arg_ids = actions
        for n in range(fn_id.shape[0]):
            a_0 = fn_id[n]
            unused_types = set(ACTION_TYPES) - \
                set(FUNCTIONS._func_list[a_0].args)
            for arg_type in unused_types:
                arg_ids[arg_type][n] = -1
        return (fn_id, arg_ids)

    def functioncall_action(self, actions, size):
        height, width = size
        fn_id, arg_ids = actions
        fn_id = fn_id.numpy().tolist()
        actions_list = []
        for n in range(len(fn_id)):
            a_0 = fn_id[n]
            a_l = []
            for arg_type in FUNCTIONS._func_list[a_0].args:
                arg_id = arg_ids[arg_type][n].detach(
                ).numpy().squeeze().tolist()
                if is_spatial_action[arg_type]:
                    arg = [arg_id % width, arg_id // height]
                else:
                    arg = [arg_id]
                a_l.append(arg)
            action = FunctionCall(a_0, a_l)

            actions_list.append(action)
        return actions_list

    def get_value(self, observation):
        screen = torch.FloatTensor(observation['screen']).cuda()
        minimap = torch.FloatTensor(observation['minimap']).cuda()
        flat = torch.FloatTensor(observation['flat']).cuda()
        with torch.no_grad():
            _, value = self.net(screen, minimap, flat)
        return value

    def train(self):
        obs_raw = self.obs_start
        shape = (self.n_steps, self.envs.n_envs)
        sample_values = np.zeros(shape, dtype=np.float32)
        sample_obersavation = []
        sample_rewards = np.zeros(shape, dtype=np.float32)
        sample_actions = []
        sample_dones = np.zeros(shape, dtype=np.float32)
        scores = []
        last_obs = self.last_obs
        for step in range(self.n_steps):
            policy, value = self.step(last_obs)

            actions = self.select_actions(policy, last_obs)
            if actions == 0:
                self.sum_episode = 7
                self.sum_score = 0
                return
            actions = self.mask_unused_action(actions)

            size = last_obs['screen'].shape[2:4]
            sample_values[step, :] = value.cpu()
            sample_obersavation.append(last_obs)
            sample_actions.append(actions)
            pysc2_action = self.functioncall_action(actions, size)

            '''fn_id, args_id = actions
            if fn_id[0].cpu().numpy().squeeze() in obs_raw[0].observation['available_actions']:
                print('1,True')
            else: print('1.False'),printoobs_info(obs_raw[0])
            if fn_id[1].cpu().numpy().squeeze() in obs_raw[1].observation['available_actions']:
                print('2,True')
            else: print('2.False'),printoobs_info(obs_raw[1])
            print(last_obs['available_actions'][0][fn_id[0]], last_obs['available_actions'][1][fn_id[1]],fn_id)'''
            obs_raw = self.envs.step(pysc2_action)
            # print("0:",pysc2_action[0].function)
            # print("1:",pysc2_action[1].function)

            last_obs = self.processor.preprocess_obs(obs_raw)
            sample_rewards[step, :] = [
                i.reward for i in obs_raw]
            sample_dones[step, :] = [i.last() for i in obs_raw]

            for i in obs_raw:
                if i.last():
                    score = i.observation['score_cumulative'][0]
                    self.sum_score += score
                    self.sum_episode += 1
                    print("episode %d: score = %f" % (self.sum_episode, score))
                    # if self.sum_episode % 10 == 0:
                    #     torch.save(self.net.state_dict(), './save/episode' +
                    #                str(self.sum_episode)+'_score'+str(score)+'.pkl')

        self.last_obs = last_obs
        next_value = self.get_value(last_obs).cpu()

        returns = np.zeros(
            [sample_rewards.shape[0]+1, sample_rewards.shape[1]])
        returns[-1, :] = next_value
        for i in reversed(range(sample_rewards.shape[0])):
            next_rewards = self.gamma*returns[i+1, :]*(1-sample_dones[i, :])
            returns[i, :] = sample_rewards[i, :]+next_rewards
        returns = returns[:-1, :]
        advantages = returns-sample_values
        self.old_net.load_state_dict(self.net.state_dict())
        actions = stack_and_flatten_actions(sample_actions)
        observation = flatten_first_dims_dict(
            stack_ndarray_dicts(sample_obersavation))
        returns = flatten_first_dims(returns)
        advantages = flatten_first_dims(advantages)
        self.learn(observation, actions, returns, advantages)

    def learn(self, observation, actions, returns, advantages):
        temp=np.arange(returns.shape[0])
        minibatch=returns.shape[0]//self.batch_size
        screen=observation['screen']
        flat=observation['flat']
        minimap=observation['minimap']
        a_actions=observation['available_actions']
        args_id=actions[1]
        for _ in range(self.epoch):
            np.random.shuffle(temp)
            for i in range(0,returns.shape[0],minibatch):
                j=i+minibatch
                shuffle=temp[i:j]
                batch_screen=screen[shuffle]
                batch_minimap=minimap[shuffle]
                batch_flat=flat[shuffle]
                batch_a_actions=a_actions[shuffle]
                batch_observation={'screen': batch_screen,
                                    'minimap': batch_minimap,
                                    'flat': batch_flat,
                                    'available_actions': batch_a_actions}
                batch_advantages=advantages[shuffle]
                batch_fn_id=actions[0][shuffle]

                batch_args_id={k:v[shuffle] for k, v in args_id.items()}
                batch_actions=(batch_fn_id,batch_args_id)
                batch_returns=returns[shuffle]

                batch_advantages = torch.FloatTensor(batch_advantages).cuda()
                batch_returns = torch.FloatTensor(batch_returns).cuda()
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                policy, batch_value = self.grad_step(batch_observation)
                log_probs = compute_policy_log_probs(
                    batch_observation['available_actions'], policy, batch_actions).squeeze()

                old_policy, _ =self.old_step(batch_observation)
                old_log_probs=compute_policy_log_probs(
                    batch_observation['available_actions'], old_policy, batch_actions).squeeze().detach()
                ratio=torch.exp(log_probs-old_log_probs)
                temp1=ratio*batch_advantages
                temp2=torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * batch_advantages

                policy_loss = -torch.min(temp1, temp2).mean()

                value_loss = (batch_returns-batch_value).pow(2).mean()
                entropy_loss = compute_policy_entropy(
                    batch_observation['available_actions'], policy, batch_actions)
                loss = policy_loss+value_loss*self.value_loss_coefficient +\
                    entropy_loss*self.entropy_coefficient
                # loss=loss.requires_grad_()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.optimizer.step()
