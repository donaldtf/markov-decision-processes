# Credit for this code goes to the following three sources:
# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
# https://medium.com/swlh/introduction-to-reinforcement-learning-coding-q-learning-part-3-9778366a41c0
# https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

import numpy as np
import timeit
import gym

def run_episode(env, q_table, max_steps, gamma):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(max_steps):
        action = np.argmax(q_table[obs, :])        
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break

    return total_reward

def get_q_table(env, epsilon, epis, max_steps, lr_rate, gamma):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rev_list = []

    # Start
    for i in range(epis):
        s = env.reset()
        rAll = 0
        t = 0
        
        while t < max_steps:
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

            if np.random.uniform(0, 1) < epsilon:
                a = env.action_space.sample()

            s1, r, done, _ = env.step(a)  

            Q[s,a] = Q[s,a] + lr_rate*(r + gamma*np.max(Q[s1,:]) - Q[s,a])

            s = s1
            rAll += r

            t += 1
        
            if done:
                break

        rev_list.append(rAll)

    return Q

def run_q_learning(env_name, epsilon=0, lr_rate = 0.81, gamma = 1.0, total_episodes=10000, max_steps=1000):
    env = gym.make(env_name)
    env.seed(99)
    np.random.seed(99)

    start = timeit.default_timer()

    q_table = get_q_table(env, epsilon, total_episodes, max_steps, lr_rate, gamma)

    stop = timeit.default_timer()
    total_time = stop - start

    scores = [run_episode(env, q_table, max_steps, gamma) for _ in range(1000)]
    # print("Average score of solution = ", np.mean(scores))
    # print('Time = ', total_time)

    return scores, total_time

if __name__ == '__main__':
    run_q_learning('FrozenLake-v0')