# temporally_correlated_exploration_stochasticity

Code for the poster presentation at Japanese Neural Network Society Meeting 2018.

Please check the (non-peer-reviewed) conference paper (Improving exploration in reinforcement learning with temporally correlated stochasticity.pdf) and poster (poster.pdf) for details.

## Required Libraries:
- TensorFlow
- OpenAI Gym
- scipy.io

## Explanation
Please prepare a data folder '..\data\'.
By running python programs, the result will be saved into the data folder, as two .mat files, for Gaussian white exploration noise and OU exploration noise .


## Run the code
```
python cartpole.py
```
or 
```
python chain_world.py
```
for getting the results in Fig.2 in the paper.