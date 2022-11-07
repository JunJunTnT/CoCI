## Introduction

This is the code for implementing different algorithms on multiagent-praticle_envs.

## Getting start

You can get all the dependencies in `requirements.txt`. Use `pip install -r requirements.txt` to install the dependencies you need.
After installing dependencies, you can run this project in `./experiments/train.py`. Experiment environment can be modified by setting different parameters below.

Note: We have refactored the code of highway-env and gym, if you want to verify each algorithm, please replace the code of them in your environment. The default path is `$PYTHON_HOME$\Lib\site-packages`, where `$PYTHON_HOME$` is the directory of your python.

## Parameters

Each parameter is fully explained in `train.py`, you can change algorithm or environment settings by modifying specific parameters.

## Code structure

- `./experiments/train.py`: contains code for training each algorithm on the highway-env
- `./experiments/Neural_Networks.py`: contains code for different neural networks used in each algorithm
- `./maddpg/trainer/GACML(ATOC, MADDPG, SchedNet, DAMA).py`: core code for each algorithm
- `./maddpg/trainer/replay_buffer.py:` replay buffer code for each algorithm
- `./maddpg/common/distributions.py`: useful distributions used in each algorithm
- `./maddpg/common/tf_util.py`: useful tensorflow functions used in each algorithm
- `./maddpg/common/attention(attention_me).py`: attention mechanism code for each algorithm
- `./multiagent_praticle_envs`: core code for mpe which is used in this experiment.
