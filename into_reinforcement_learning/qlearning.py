# implementation of "Introduction to Reinforcement Learning" 
# https://deepnote.com/@ken-e7bd/Intro-to-Q-learning-in-RL-4R450s6_RVKJIC2xiqs71g

import math
import numpy as np
import gym
# import matplotlib.pyplot as plt
# from pyvirtualdisplay import Display

def epsilon_greedy_policy(state, env, q_table, exploration_rate):
    """
    This is an epsilon greedy policy
    In other words, most times the agent chooses
    the action that maximises the reward given state
    (greedily). But occassionally (controlled by exploration_rate),
    the agent chooses a random action which makes sure 
    the agent balances between exploitation and exploration

    inputs:
    -------
        state: the current state the agent is at
        env: the CartPole environment
        Q_table: a table-like structure
        exploration_rate: a small number close to 0
    return:
    -------
        action to be taken in the next step
    """
    if (np.random.random() < exploration_rate):
        # Generates numbers np.random.random() uniformly between 0-1
        # This samples a random action given the environment
        return env.action_space.sample()
    else:
        # Choose greedily the action which gives the highest expected reward
        # given the current state
        return np.argmax(q_table[state])

def get_rate(e):
    """
    Get the learning rate or exploration_rate given an episode
    subject to decay (25.)

    inputs:
    -------
        e: a given episode
    return:
    -------
        a learning or an exploration rate
    """
    return max(0.1, min(1., 1. - np.log10((e + 1) / 25.)))

def update_q(q_table, state, action, reward, new_state, alpha, gamma):
    """
    Q-learning update step

    inputs:
    -------
        Q_table: a table-like structure with N rows for states and M columns for actions
        state: the current state the agent is at time step t
        action: the action taken given the previous state at time step t
        reward: reward collected as a result of that action at time step t
        new_state: the new state at time-step t+1
        alpha: learning rate factor
        gamma: discount factor
    return:
    ------- 
        updated Q-table
    """
    q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[state][action])
    return q_table


def random_play(env):
#     env = gym.make('CartPole-v0')

    # Define the number of episodes for which we want to run (just 1)
    for e in range(1):
        # Reset the environment for a new episode, get the default state S_0
        state = env.reset()
        # Define the number of timesteps for which to run the episode by default (200)
        for step in range(200):
            # Sample a random action (A_{t})
            action = env.action_space.sample()
            # Get the new state (S_{t+1}), reward (R_{t+1}), end signal and additional information
            new_state, reward, done, info = env.step(action)
            # Update the state S_{t} = S_{t+1}
            state = new_state
            if done:
                print("Episode finished after {} timesteps".format(step+1))
                break
    # Close the environment
    env.close()

def discretize_state(state, env, buckets=(1, 1, 6, 12)):
    """
    The original states in this game are continuous, 
    which does not work with Q-learning which expects 
    discrete states. This function discretize the continuous
    states into buckets. 

    inputs:
    -------
        state: current state's observation which needs discretizing
        env: the cartpole environment
        buckets: this will be used to discretize the original continuous states in this Cartpole example
    return:
    -------
        The discretized state
    """
    # The upper and the lower bounds for the discretization
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]

    # state is the native state representations produced by env
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    # state_ is discretized state representation used for Q-table later
    state_ = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    state_ = [min(buckets[i] - 1, max(0, state_[i])) for i in range(len(state))]
    return tuple(state_) 

def train(env, num_episodes):
    """
    Training the agent with Q-learning with respect to pseudocode in Algorithm 1

    inputs:
    -------
        env: the cartpole environment
        num_episodes: the number of episodes for which to train
    return:
    -------
        The optimised Q-table
        The list containing the total cummulative reward for each episode of training
    """
    # Discount factor gamma represents how much does the agent value future rewards as opposed to immediate rewards
    gamma = 0.98

    # (1, 1, 6, 12) represents the discretisation buckets
    # Initialise the q-table as full of zeros at the start
    q_table = np.zeros((1, 1, 6, 12) + (env.action_space.n,))

    # Create a list to store the accumulated reward per each episode
    total_reward = []
    for e in range(num_episodes):
        # Reset the environment for a new episode, get the default state S_0
        state = env.reset()
        state = discretize_state(state, env)

        # Adjust the alpha and the exploration rate, it is a coincidence they are the same
        alpha = exploration_rate = get_rate(e)
        
        # Initialize the current episode reward to 0 
        episode_reward = 0
        done = False
        while done is False:
            # Choose the action A_{t} based on the policy
            action = epsilon_greedy_policy(state, env, q_table, exploration_rate)

            # Get the new state (S_{t+1}), reward (R_{t+1}), end signal
            new_state, reward, done, _ = env.step(action)
            new_state = discretize_state(new_state, env)

            # Update Q-table via update_q(Q_table, S_{t}, A_{t}, R_{t+1}, S_{t+1}, alpha, gamma) 
            q_table = update_q(q_table, state, action, reward, new_state, alpha, gamma)

            # Update the state S_{t} = S_{t+1}
            state = new_state
            
            # Accumulate the reward
            episode_reward += reward
        
        total_reward.append(episode_reward)
    print('Finished training!')
    return q_table, total_reward

def play(env, q_table, max_steps=100):
    done = False
    state = env.reset()
    state = discretize_state(state, env)
    steps = 0
    while (done is False) and (steps <= max_steps):
        # Choose the action A_{t} based on the policy
        action = np.argmax(q_table[state])
        
        # Get the new state (S_{t+1}), reward (R_{t+1}), end signal
        state, reward, done, _ = env.step(action)
        state = discretize_state(state, env)
        steps += 1
    print("Episode finished after {} timesteps".format(steps))

if __name__ == "__main__":
    np.random.seed(42)
    random_play()