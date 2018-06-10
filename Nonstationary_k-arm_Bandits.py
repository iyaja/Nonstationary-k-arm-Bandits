# uncomment thenext line if using in jupyter notebook line 
# %matplotlib inline
# import nessecary libraries for computation and plotting
import numpy as np 
import matplotlib.pyplot as plt

# initialize k bandits with some initial value init_q for the action-value funtion q_star
def initBandits(k,init_q):
    global q_star       # q_star is an array that stores the true value for evey action. It is indexed by a, the action taken
    q_star = np.empty(k)
    q_star.fill(init_q)
    return q_star

# increment the bandits by allowing the action-value function to take a random walk with (default,can be tweaked)mean step size 0 and SD 0.01
def incBandits_Gaussian(q_prev, mu = 0, sigma = 0.01):
    q_next = q_prev + np.random.normal(mu,sigma,q_prev.shape)
    return q_next

# initialize the Q fuction, that is, the agent's estimate of the action-value function
def initAgent():
    Q_pred = np.random.randn(q_star.shape[0])
    return Q_pred

# provide a reward to the agent for performing an action. Done by looking up q_star(a)
def Reward(action):
    return q_star[action]


# some default parameters that can be tweaked
epsilon = 0.01
timesteps = 10000
alpha = 0.1 # setting alpha = 1/n results in the sample averages algorithm

initBandits(10,0.5)
Q = initAgent()
total_reward = 0
average_reward = []


# Main algorithm for the agent
for n in range (1,timesteps):
    
    a = np.random.choice([np.argmax(Q),np.random.randint(1,10)] , p = [1-epsilon,epsilon]) # epsilon-greedy selection of action
    R = Reward(a) 
    Q[a] = Q[a] + alpha*(R - Q[a])      # Update rule for Q function
    q_star = incBandits_Gaussian(q_star)
    
    total_reward = total_reward + R
    average_reward.append(total_reward/n)       # Calculate average reward for future ploting
    

# basic functions to plot the average reward over time    
plt.plot(average_reward)
plt.show()


