# Credit for this code goes to the following three sources:
# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

import gym
import timeit
import numpy as np
from utils import evaluate_policy, extract_policy

"""
Solving FrozenLake8x8 environment using Value-Itertion.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
def value_iteration(env, gamma = 1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.observation_space.n)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)] 
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            break
    return v

def run_value_iteration(env_name, gamma=1.0):
    env = gym.make(env_name)
    env.seed(99)
    np.random.seed(99)

    start = timeit.default_timer()

    optimal_v = value_iteration(env, gamma)
    
    stop = timeit.default_timer()
    total_time = stop - start

    policy = extract_policy(env, optimal_v, gamma)
    scores = evaluate_policy(env, policy, gamma)

    return scores, total_time

if __name__ == '__main__':
    run_value_iteration('FrozenLake8x8-v0')