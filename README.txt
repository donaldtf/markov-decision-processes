##### Machine Learning - OMSCS Spring 2019 ####
##### Author - Donald Ford ####

### Getting Started ###

In order to recreate the results shown in my report, please clone the following repo from github.

https://github.com/donaldtf/markov-decision-processes

Or you can just use this command to clone it via ssh: `git clone git@github.com:donaldtf/markov-decision-processes.git`

This repo contains all the code needed to reproduce my results. The project structure looks like this:

/scores - holds charts that show the average score obtained vs a range of different parameters being tweaked
/times - holds charts that show the time needed to complete an algorithm vs a range of different parameters
frozen_lake.py - runs both the small and large frozen lakes with value iteration, policy iteration and q Learning
policy_iteration.py - implementation for running policy iteration for an openai frozen lake env
value_iteration.py - implementation for running value iteration for an openai frozen lake env
q_learning.py - implementation for running q learning for an openai frozen lake env
utils.py - holds shared functions, used for things such as charting results

### Install Dependencies ###

The code relies on the following dependencies in order to run. You can install them via your favorite method (conda, pip, etc.).

- numpy
- matplotlib
- openai https://github.com/openai/gym

Once these are all installed you should be ready to run the code

### Running the code ###

Running the code is simple once you have your dependencies installed. Simply run the following command

`python frozen_lake.py`

This will generate all of the plots shown in the report.

Note: Running all of the algorithms at once may take 10 - 15 minutes, depending on your machine, to complete.
