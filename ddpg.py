import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import utils


EPS = 0.0003


def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform__(-v, v)


class critic_network(nn.Module):

	def __init__(self, state_dim, action_dim):
		super(critic_network, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		h1_dim = 256
		h2_dim = 128
		h3_dim = 128

		self.fcs1 = nn.Linear(state_dim,h1_dim)
		self.fcs2 = nn.Linear(h1_dim,h2_dim)		
		self.fca1 = nn.Linear(action_dim,h2_dim)
		self.fc2 = nn.Linear(h2_dim + h2_dim,h3_dim)
		self.fc3 = nn.Linear(h3_dim,1)

		'''
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.fc3.weight.data.uniform_(-EPS,EPS)
		'''
		nn.init.xavier_normal_(self.fcs1.weight.data)
		nn.init.xavier_normal_(self.fcs2.weight.data)
		nn.init.xavier_normal_(self.fca1.weight.data)
		nn.init.xavier_normal_(self.fc2.weight.data)
		nn.init.uniform_(self.fc3.weight.data, a=-EPS, b=EPS)

	def forward(self, sa):
		state, action = sa
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class actor_network(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		super(actor_network, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		h1_dim = 256
		h2_dim = 128
		h3_dim = 64

		self.fc1 = nn.Linear(state_dim,h1_dim)
		self.fc2 = nn.Linear(h1_dim,h2_dim)
		self.fc3 = nn.Linear(h2_dim,h3_dim)
		self.fc4 = nn.Linear(h3_dim,action_dim)

		'''
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
		self.fc4.weight.data.uniform__(-EPS,EPS)
		'''

		nn.init.xavier_normal_(self.fc1.weight.data)
		nn.init.xavier_normal_(self.fc2.weight.data)
		nn.init.xavier_normal_(self.fc3.weight.data)
		nn.init.uniform_(self.fc4.weight.data, a=-EPS, b=EPS)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = torch.tanh(self.fc4(x))
		action = action * self.action_lim

		return action


class DDPG(object):
	def __init__(self, state_dim, action_dim, action_lim):
		self.action_lim = action_lim

		self.a_net = actor_network(state_dim, action_dim, action_lim)
		self.a_net_target = actor_network(state_dim, action_dim, action_lim)
		self.c_net = critic_network(state_dim, action_dim)
		self.c_net_target = critic_network(state_dim, action_dim)

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.a_net.to(self.device)
		self.a_net_target.to(self.device)
		self.c_net.to(self.device)
		self.c_net_target.to(self.device)

		utils.hard_update(self.c_net_target, self.c_net)
		utils.hard_update(self.a_net_target, self.a_net)	

		self.tau = .001
		self.gamma = .99
		actor_learning_rate = 1e-3
		critic_learning_rate = 1e-4

		self.a_net_optimizer = optim.Adam(self.a_net.parameters(), lr=actor_learning_rate)
		self.c_net_optimizer = optim.Adam(self.c_net.parameters(), lr=critic_learning_rate)

	def action(self, s, noise=None):		
		inp = torch.from_numpy(s).unsqueeze(0).float().to(self.device)

		self.a_net.eval()
		with torch.no_grad():			
			a = self.a_net(inp).data[0].cpu().numpy() 
		self.a_net.train()

		if noise is not None:
			a = a + noise()*self.action_lim
		return a	

	def train(self, training_data):
		# important to restructure like this
		batch_s = np.vstack(training_data[:,0])
		batch_a = np.vstack(training_data[:,1])
		batch_s1 = np.vstack(training_data[:,4])
		batch_r = np.array(training_data[:,2]).astype("float")
		# batch_done = np.array(training_data[:,3]).astype("float")
		
		s1 = torch.from_numpy(batch_s).float().to(self.device)
		a1 = torch.from_numpy(batch_a).float().to(self.device)
		r1 = torch.from_numpy(batch_r).float().to(self.device)
		s2 = torch.from_numpy(batch_s1).float().to(self.device)

		with torch.no_grad():
			a2 = self.a_net_target(s2)
			next_val = torch.squeeze(self.c_net_target((s2, a2)))
		y_expected = r1 + self.gamma*next_val
		y_predicted = torch.squeeze(self.c_net.forward((s1, a1)))
		
		# compute critic loss, and update the critic
		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
		self.c_net_optimizer.zero_grad()
		loss_critic.backward()
		self.c_net_optimizer.step()

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.a_net.forward(s1)
		loss_actor = -self.c_net.forward((s1, pred_a1))
		loss_actor = loss_actor.mean()
		self.a_net_optimizer.zero_grad()
		loss_actor.backward()
		self.a_net_optimizer.step()

		utils.soft_update(self.a_net_target, self.a_net, self.tau)
		utils.soft_update(self.c_net_target, self.c_net, self.tau)
		