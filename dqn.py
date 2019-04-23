import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simulator import *
from scheduler import *
import argparse
import json

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print('device:', device)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	def __init__(self, inputs, outputs):
		super(DQN, self).__init__()
		self.lin1 = nn.Linear(inputs, inputs*2)
		self.lin2 = nn.Linear(inputs*2, inputs)
		self.lin3 = nn.Linear(inputs, outputs)
		
		# Called with either one element to determine next action, or a batch
		# during optimization. Returns tensor([[left0exp, right0exp]...]).
	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))
		
		return x.view(x.size(0), -1)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN(5, 1).to(device)
target_net = DQN(5, 1).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

#num_actions 범위 내에서 아웃풋이 나오게 DQN 수정
#num_outputs만큼 output을 return하게 수정
def select_action(state, num_actions, num_outputs):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * \
					math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return largest column value of each row
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			y = policy_net(state)
			y[num_actions:] = -1
			_ind = list(np.argsort(y, axis=0))[::-1][:num_outputs]
			_y = []
			for ind in _ind:
				_y.append(ind)
			return torch.tensor(_y, device=device, dtype=torch.long)
	else:
		possible_actions = [i for i in range(num_actions)]
		if len(possible_actions) == 0:
			return torch.tensor([-1] * num_outputs, device=device, dtype=torch.long)
		return torch.tensor(random.sample(possible_actions, num_outputs), device=device, dtype=torch.long)

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)
	action_batch = action_batch.view(-1, 1)
	action_batch[action_batch==-1] = 0
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

class RS(object):
	def __init__(self, n_core, num_episodes, max_timestep, machine, preempt=True):
		self.n_core = n_core
		self.preempt = preempt
		self.num_episodes = num_episodes
		self.i_episode = 0
		self.max_timestep = max_timestep
		self.machine = machine

		self.prev_state = None
		self.prev_action = None

	def prios(self, p):
		if p == None:
			print("ERROR: None packet is in Q")
			exit()
		return (p.dead, p.priority, p.release, p.time, p.required, p)

	def step(self, result, ncores, time):
		cpu_util = result[3] / (ncores * time) * 100
		throughput = result[0] / time
		result[0] = 1 if result[0] == 0 else result[0]
		turnaround = result[1] / result[0]
		priority = result[2] / result[0]
		load = self.simulator.load / time

		reward = cpu_util + throughput - turnaround - load 

		return reward

	def run(self, time, packets, runQ):
		state = []
		_p = []
		for p in packets:
			if p == None:
				print("ERROR: None packet generated")
				exit()
			temp_p = self.prios(p)
			state.append(temp_p[:-1])
			_p.append(temp_p[-1])
		for p in runQ:
			if p == None:
				continue
			temp_p = self.prios(p)
			state.append(temp_p[:-1])
			_p.append(temp_p[-1])
		num_p = len(_p)
		for _ in range(10 - num_p):
			_p.append(None)
			state.append((-1, -1, -1, -1, -1))
#		print('time:', time)
#		print('state:', state)
#		print('_p:', _p)
		action = select_action(torch.tensor(state, device=device, dtype=torch.float32), num_p, self.n_core)
#		print('action selected:', action)

		if time > 0:
			prev_state = torch.tensor([self.prev_state], device=device, dtype=torch.float32)
			prev_action = torch.tensor(self.prev_action, device=device, dtype=torch.long)

		self.prev_state = state
		self.prev_action = action
		
		_p = np.array(_p)
		if time == 0:
			ind = action.tolist()
			for i in range(len(ind)):
				ind[i] = int(ind[i])
			return _p[ind]

		done = True if time == self.max_timestep else False
		result = self.machine.result()
		reward = self.step(result, self.machine.ncores, time)
		reward = torch.tensor([reward], device=device)

		state = torch.tensor([state], device=device, dtype=torch.float32)
		memory.push(prev_state, prev_action, state, reward)
		
		optimize_model()

		if done:
			if self.i_episode % TARGET_UPDATE == 0:
				target_net.load_state_dict(policy_net.state_dict())
				print('target updated')
			self.i_episode += 1
			print(self.i_episode, 'episodes done')
		
		if self.i_episode == self.num_episodes: 
			print('Complete')
		
		ind = action.tolist()
		for i in range(len(ind)):
			ind[i] = int(ind[i])
		if action[0] == -1:
			return [None] * len(action)
		return _p[ind]


with open('config.json') as data_file:
	data = json.loads(data_file.read())

num_episodes = 50
max_timestep = 10000

gens = []
for c in data["generators"]:
	gen = Generator(c['rmean'], c['rstd'], c['tmean'], c['tstd'], c['pmean'], c['pstd'], c['delay'])
	gens.append(gen)

_m = data["machine"]
machine = Machine(_m["resources"], _m["performance"], _m["ncore"])

scheduler = RS(data["machine"]["ncore"], num_episodes, max_timestep, machine)

for _ in range(num_episodes):
	simulator = Simulator(gens, scheduler, machine, max_timestep)
	scheduler.simulator = simulator
	simulator.run()


