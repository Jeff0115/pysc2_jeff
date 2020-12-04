import torch
from torch.distributions.categorical import Categorical
from rule_base import RuleBase,RuleBasevalue
from net import CNN
def compute_log_prob(probs, actions):
        cate_dist = Categorical(probs)
        log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
        return log_prob
a=1
net=CNN().cuda()
RuleBasevalue(net)
