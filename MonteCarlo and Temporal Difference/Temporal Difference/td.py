#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import sys
from collections import defaultdict
from collections import Counter
import random
env = gym.make('CliffWalking-v0')
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v td_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
     Hints:
        You can use the function from project2-1
    """
    ############################
    # YOUR IMPLEMENTATION HERE #

    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.choice(nA)
    else:
        action = np.argmax(Q[state])
       




    ############################
    return action

def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
  
    eps_min = 0.01
    decay = 0.99
    ############################
    # YOUR IMPLEMENTATION HERE #
    nA=4
    
    for i in range(n_episodes):

        # define decaying epsilon
        ####why does this decay is important is it trying to imitate qlearning by decaying epsilon?
        epsilon = epsilon*decay
        state= env.reset()
        
        action = epsilon_greedy(Q, state,4, epsilon)
                                   
        while True:



        # get an action from policy
            next_state, reward, done, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state,4, epsilon)
             
                                    

            td_target = reward + gamma * (Q[next_state][next_action])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state
            action=next_action
            if done:
                
                break
    return Q

def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    q_values = defaultdict(lambda: np.zeros(env.action_space.n))

# Iterate over 500 episodes
    for _ in range(n_episodes):

        state = env.reset()    
        done = False

    # While episode is not over
        while not done:
        # Choose action        
           action = epsilon_greedy(q_values, state,4, epsilon=0.1)
        # Do the action
           next_state, reward, done,info= env.step(action)
        # Update q_values        
           td_target = reward + gamma * np.max(q_values[next_state])
           td_error = td_target - q_values[state][action]
           q_values[state][action] += alpha * td_error
        # Update state
           state = next_state
    return q_values



sarsa(env, n_episodes=50000, gamma=1.0, alpha=0.01, epsilon=0.1)      