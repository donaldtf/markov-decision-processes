# Credit for this code goes to the following three sources:
# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

import gym
import timeit
import numpy as np
from utils import evaluate_policy, extract_policy

"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
def compute_policy_v(env, policy, gamma=1.0, eps = 1e-10):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.observation_space.n)
    while True:
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
def policy_iteration(env, gamma = 1.0, max_iterations = 200000):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))  # initialize a random policy
    for i in range(max_iterations):
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(env, old_policy_v, gamma)
        if (np.all(policy == new_policy)):
            print ('Policy-Iteration converged at iteration %d.' %(i+1))
            break
        policy = new_policy
    return policy

def run_policy_iteration(env_name, gamma = 1.0):
    env = gym.make(env_name)
    env.seed(99)
    np.random.seed(99)

    start = timeit.default_timer()

    optimal_policy = policy_iteration(env, gamma)

    stop = timeit.default_timer()
    total_time = stop - start

    scores = evaluate_policy(env, optimal_policy)

    return scores, total_time

if __name__ == '__main__':
    run_policy_iteration('FrozenLake8x8-v0')
