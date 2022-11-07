## Introduction

This is the code for policy gradient algorithm in binary environment.

## Getting start

You can get all the dependencies in `requirements.txt`. Use `pip install -r requirements.txt` to install the dependencies you need.
After installing dependencies, you can run this project in `./train_without_com.py` and `./train_with_com.py`. Experiment environment can be modified by setting different parameters below.

## Parameters

- `GAMMA`: discount factor in policy gradient algorithm (default: `0.9`)
- `LEARNING_RATE1`: learning rate of agent1 (default: `0.05`)
- `LEARNING_RATE2`: learning rate of agent2 (default: `0.1`)
- `EPISODE`: end algorithm after certain iterations (default: `500`)
- `STEP`: calculate results after certain iterations (default: `50`)
- `discount_commu`: discount factor in reward if agent chooses to communicate (default: `0.2`)
- `parameter1`: observation ability of agent1 (default: `[0.6, 0, 0.4, 0]`)
- `parameter2`: observation ability of agent2 (default: `[0.8, 0.2, 0, 0]`)
- `theta`: parameters of action policy (default: `[0, 0, 0, 0]`)
- `com_theta`: parameters of communication policy (default: `[0, 0, 0, 0]`)

## Code structure

- `./env_binary.py`: contains code for binary environment
- `./train_without_com.py`: contains code for policy gradient algorithm in no-communication environment
- `./train_with_com.py`: contains code for policy gradient algorithm in communication environment
