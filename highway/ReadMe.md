## Introduction

This is the code for implementing different algorithms on highway-env.


## Dependency

You can get all the dependencies in `requirements.txt`. Use `pip install -r requirements.txt` to install the dependencies you need.

After installing dependencies, you can run this project in `./experiments/train.py`. Experiment environment can be modified by setting different parameters below.

Note: We have refactored the code of highway-env and gym, if you want to verify each algorithm, please replace the code of them in your environment. The default path is `$PYTHON_HOME$\Lib\site-packages`, where `$PYTHON_HOME$` is the directory of your python.

## Parameters

- `trainer_name`: name of the trainer
- `scenario_name`: name of the scenario
- `loss rate`: packet loss rate in communication channel
- `controlled_vehicles`: number of controlled agents (default: `4`)
- `observation_dim`: length of agents' observation vectors (default: `5`)
- `mean_step`: result is calculated every time this number of episodes has been completed (default: `50`)
- `view`: maximum number of observations of agents (default: `4`)
- `low_speed_range`: set the interval for low speed penalty (default: `[0, 3]`)
- `low_speed_punishment`: maximum value of low speed penalty (default: `1`)
- `arrived_reward`: reward for each agent if it reaches its destination (default:` 5`)
- `model_save_path`: directory where model are saved (default: ` "./res/" + scenario_name + "/"`)
- `res_save_path`: directory where iteration results are saved (default: ` model_save_path`)
- `restore`: whether to save model parameters  (default: `False`)
- `display`: whether to display the trained policy to the screen (default: `False`)
- `update_option`: whether to update the model after certain iterations(default: `true`)

## Code structure

- `./experiments/train.py`: contains code for training each algorithm on the highway-env
- `./experiments/Neural_Networks.py`: contains code for different neural networks used in each algorithm
- `./maddpg/trainer/ACML(GACML, ATOC, MADDPG, SchedNet, DAMA).py`: core code for each algorithm
- `./maddpg/trainer/replay_buffer.py:` replay buffer code for each algorithm
- `./maddpg/common/distributions.py`: useful distributions used in each algorithm
- `./maddpg/common/tf_util.py`: useful tensorflow functions used in each algorithm
- `./maddpg/common/attention(attention_me).py`: attention mechanism code for each algorithm
