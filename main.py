import numpy as np
import gym
import torch

from buffer import episode_buffer
from ddpg import DDPG
from noise import OrnsteinUhlenbeckActionNoise


if __name__ == '__main__':
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high[0]
        
    batch_size = 1024
    tau = .001
    gamma = .99
    actor_learning_rate = 1e-3
    critic_learning_rate = 1e-4

    agent = DDPG(s_dim, a_dim, a_bound.item())
    ep_buffer = episode_buffer()
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))

    nepisodes = 1000
    i = 0
    steps = 0
    update_freq = 4
    target_update_freq = 10
    pre_trainsteps = batch_size
    rall = []

    while i < nepisodes:
        i += 1
        s = env.reset()
        actor_noise.reset()

        # total reward collected in an episode
        ep_r = 0

        j = 0
        done = False
        while not done:
            j += 1
            steps += 1
            # env.render()
            a = agent.action(s, actor_noise)            
            s1, r, done, _ = env.step(a)
            ep_r += r

            ep_buffer.add([np.reshape(s,(1,s_dim)), np.reshape(a,(1,a_dim)), r, done, np.reshape(s1,(1,s_dim))])
            if len(ep_buffer.data) > batch_size and steps % update_freq == 0:
                training_data = np.array(ep_buffer.sample(batch_size))              
                agent.train(training_data)
            s = np.copy(s1)

        print('Episode:',i,'reward:',ep_r)

        # if i%100==0:
        #     torch.save(agent.a_net.state_dict(), './saved_model/actor_'+str(i)+'.pt') 
        #     torch.save(agent.c_net.state_dict(), './saved_model/critic_'+str(i)+'.pt')    
        
