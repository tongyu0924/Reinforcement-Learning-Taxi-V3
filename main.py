import numpy as np
import gym
import random
from IPython.display import clear_output
import time

# create Taxi environment 
env = gym.make("Taxi-v3")

# initialize q-table
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# hyperparameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate = 0.005

# training variables
num_episodes = 1000
max_steps = 99 # per episode

# training
for episode in range(num_episodes):

  # reset the enviroment
  state = env.reset()
  done = False

  for s in range(max_steps):

    if random.uniform(0, 1) < epsilon:
      action = env.action_space.sample()
    
    else:
      action = np.argmax(q_table[state, :])
    
    # take action and observe reward
    new_state, reward, done, info = env.step(action)

    # Q-learning 
    q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]) - q_table[state, action])

    # Update to our new state
    state = new_state

    # if done, finish episode
    if done == True:
      break
    
  #　Decrease epsilon
  epsilon = np.exp(-decay_rate * episode)

print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")

total_epochs, total_penalties = 0, 0
episodes = 100
frames = []

def print_frames(frames):
  for i, frame in enumerate(frames):
      clear_output(wait=True)
      print(frame['frame'])
      print(f"Episode: {frame['episode']}")
      print(f"Timestep: {i + 1}")
      print(f"State: {frame['state']}")
      print(f"Action: {frame['action']}")
      print(f"Reward: {frame['reward']}")
      time.sleep(1)

for ep in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'episode': ep, 
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

print_frames(frames)
