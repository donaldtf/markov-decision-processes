# Credit for this code goes to the following three sources:
# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

"""
Solving FrozenLake8x8 environment using Policy iteration.
Author : Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import gym
from gym import wrappers
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import timeit

def run_episode(env, policy, gamma = 1.0):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma = 1.0, n = 1000):
    scores = [run_episode(env, policy, gamma) for _ in range(n)]
    return scores

def extract_policy(env, v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

# This code was not borrowed from anywhere and is of my own making
def plot_results(result, result_label, other, other_label, title, file_name):
    plt.figure()
    plt.title(title)
    plt.xlabel(other_label)
    plt.ylabel(result_label)

    result_mean = np.mean(result, axis=1)
    result_std = np.std(result, axis=1)
    plt.grid()

    plt.fill_between(other, result_mean - result_std,
                     result_mean + result_std, alpha=0.1,
                     color="g")
    plt.plot(other, result_mean, 'o-', color="g")

    plt.savefig(file_name)