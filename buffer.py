from collections import deque
import random

class episode_buffer(object):
	def __init__(self, maxlen=50000):
		self.maxlen = maxlen
		self.data = deque(maxlen=self.maxlen)

	def add(self, ep):
		self.data.append(ep)	

	def sample(self, nsamples):
		if nsamples > len(self.data):
			return random.sample(self.data, len(self.data))
		else:		
			return random.sample(self.data, nsamples)	

	def display(self):
		for x in self.data:
			print(x)		