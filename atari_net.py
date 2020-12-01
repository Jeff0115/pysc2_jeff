import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib.actions import FunctionCall, FUNCTIONS
from pysc2.lib.actions import TYPES
import numpy as np

from pre_processing import is_spatial_action, NUM_FUNCTIONS, FLAT_FEATURES

class MyLinear(nn.Module):
    def __init__(self,out_channel):
        super(MyLinear,self).__init__()
        self.linear=nn.Linear(256,out_channel)

    def forward(self,x):
        x=self.linear(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1_screen=nn.Conv2d(27,16,kernel_size=8,stride=4,padding=2) #8*8
        self.conv1_minimap=nn.Conv2d(11,16,kernel_size=8,stride=4,padding=2)
        self.conv2_screen=nn.Conv2d(16,32,kernel_size=4,stride=2,padding=1) #4*4
        self.conv2_minimap=nn.Conv2d(16,32,kernel_size=4,stride=2,padding=1)
        self.relu=nn.ReLU()
        self.linear_1=nn.Linear(11+32*4*4+32*4*4,256)
        self.linear_2=nn.Linear(256,1)
        self.linear_3=nn.Linear(256,NUM_FUNCTIONS)
        self.linear_x=nn.Linear(256,32)
        self.linear_y=nn.Linear(256,32)
        self.conv3=nn.Conv2d(75,1,kernel_size=1,stride=1)
        self.mylinear=dict()
        for k in actions.TYPES:
            if not is_spatial_action[k]:
                self.mylinear[k]=MyLinear(k.sizes[0]).cuda()
    def forward(self,screen,minimap,flat):
        screen=self.relu(self.conv1_screen(screen))
        screen=self.relu(self.conv2_screen(screen)).cuda()
        screen=torch.flatten(screen, start_dim=1)
        minimap=self.relu(self.conv1_minimap(minimap))
        minimap=self.relu(self.conv2_minimap(minimap))
        minimap=torch.flatten(minimap, start_dim=1)
        flat=torch.tanh(flat)
        state=torch.cat((screen,minimap,flat),dim=1)
        flatten=self.relu(self.linear_1(state))
        value=self.linear_2(flatten)
        value=torch.reshape(value,(-1,))
        act_prob=self.linear_3(flatten)
        act_prob=F.softmax(act_prob,dim=1)
        args_out=dict()
        for i in actions.TYPES:
            if is_spatial_action[i]:
                arg_out_x=self.linear_x(flatten).reshape(-1,1,32)
                arg_out_y=self.linear_y(flatten).reshape(-1,32,1)
                arg_out=torch.matmul(arg_out_y,arg_out_x)
                arg_out=arg_out.flatten(1,-1)
                arg_out=F.softmax(arg_out,dim=1)
            else:
                arg_out=self.mylinear[i](flatten)
                arg_out=F.softmax(arg_out,dim=1)
            args_out[i]=arg_out
        policy=(act_prob,args_out)

        return policy,value

#下面的都是用来debug
'''ac=actions.TYPES

a=CNN().cuda()
opt=optim.Adam(a.parameters(),lr=0.0001)
screen=10*torch.randn(4,27,32,32).cuda()
minimap=10*torch.randn(4,11,32,32).cuda()
flat=10*torch.randn(4,11).cuda()
policy,value=a(screen,minimap,flat)
act_prob=policy[0]
args_out=policy[1]
loss=(torch.sum(act_prob)+value).mean()
loss0=0
for i in ac:
    loss0+=torch.sum(args_out[i])
loss+=loss0
opt.zero_grad()
loss.backward()
opt.step()
c=1'''