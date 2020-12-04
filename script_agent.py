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
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
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