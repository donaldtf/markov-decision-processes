import timeit

from value_iteration import run_value_iteration
from policy_iteration import run_policy_iteration
from q_learning import run_q_learning
from utils import plot_results

def run_different_gammas(method, method_name, env_name):
    gammas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    overall_scores = []
    times = []

    for gamma in gammas:
        scores, time = method(env_name, gamma)
        overall_scores.append(scores)
        times.append([time])

    title = "Mean Score vs Gamma for {}, {}".format(env_name, method_name)
    file_name = "scores/{}-{}.png".format(env_name, method_name)
    plot_results(overall_scores, "Mean Score of Policy", gammas, "Gamma Value", title, file_name)

    title = "Time Taken (s) vs Gamma for {}, {}".format(env_name, method_name)
    file_name = "times/{}-{}.png".format(env_name, method_name)
    plot_results(times, "Time Taken (s)", gammas, "Gamma Value", title, file_name)

def run_different_epsilons(env_name):
    epsilons = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    overall_scores = []
    times = []

    for epsilon in epsilons:
        scores, time = run_q_learning(env_name, epsilon)
        overall_scores.append(scores)
        times.append([time])

    title = "Mean Score vs Epsilon for {}, QLearning".format(env_name)
    file_name = "scores/{}-QLearning-Epsilon.png".format(env_name)
    plot_results(overall_scores, "Mean Score of Policy", epsilons, "Epsilon Value", title, file_name)

    title = "Time Taken (s) vs Epsilon for {}, QLearning".format(env_name)
    file_name = "times/{}-QLearning-Epsilon.png".format(env_name)
    plot_results(times, "Time Taken (s)", epsilons, "Epsilon Value", title, file_name)

def run_different_learn_rates(env_name):
    learning_rates = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    overall_scores = []
    times = []

    for learning_rate in learning_rates:
        scores, time = run_q_learning(env_name, 0, learning_rate)
        overall_scores.append(scores)
        times.append([time])

    title = "Mean Score vs Learning Rate for {}, QLearning".format(env_name)
    file_name = "scores/{}-QLearning-Learning-Rate.png".format(env_name)
    plot_results(overall_scores, "Mean Score of Policy", learning_rates, "Learning Rate Value", title, file_name)

    title = "Time Taken (s) vs Learning Rate for {}, QLearning".format(env_name)
    file_name = "times/{}-QLearning-Learning-Rate.png".format(env_name)
    plot_results(times, "Time Taken (s)", learning_rates, "Learning Rate Value", title, file_name)

def run_different_q_gammas(env_name):
    gammas = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    overall_scores = []
    times = []

    for gamma in gammas:
        scores, time = run_q_learning(env_name, 0, 0.6, gamma)
        overall_scores.append(scores)
        times.append([time])

    title = "Mean Score vs Gamma for {}, QLearning".format(env_name)
    file_name = "scores/{}-QLearning-Gamma.png".format(env_name)
    plot_results(overall_scores, "Mean Score of Policy", gammas, "Gamma Value", title, file_name)

    title = "Time Taken (s) vs Gamma for {}, QLearning".format(env_name)
    file_name = "times/{}-QLearning-Gamma.png".format(env_name)
    plot_results(times, "Time Taken (s)", gammas, "Gamma Value", title, file_name)

def run_frozen_lake(env_name):
    print (env_name)

    print ("Value Iteration Results")
    run_different_gammas(run_value_iteration, "Value Iteration", env_name)
    print ()

    print ("Policy Iteration Results")
    run_different_gammas(run_policy_iteration, "Policy Iteration", env_name)
    print ()
    
    print ("Q Learning Results")
    print ()
    print ("Iterating different epsilons...")
    print ()
    run_different_epsilons(env_name)
    print ("Iterating different learning rates...")
    print ()
    run_different_learn_rates(env_name)
    print ("Iterating different gammas...")
    print ()
    run_different_q_gammas(env_name)
    print ()

    print ()

small  = 'FrozenLake-v0'
large  = 'FrozenLake8x8-v0'

start = timeit.default_timer()

run_frozen_lake(small)
run_frozen_lake(large)

stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")
