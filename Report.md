# Report


## Learning algorithm
 
The agent used for this reinforcement learning task was a simple implementation of the vanilla deep Q-Network algorithm introduced in [the original paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf). The neural network used was a feed forward network with three fully connected hidden layers with dimensions (64, 32, 16). Thus from input to output the network has dimensions (37 = dimension of states, 64, 32, 16, 4 = dimension of actions).

### Hyperparameters 

    - Maximum steps per episode: 1000
    - Epsilon start: 1.0
    - Epsilon end: 0.01
    - Epsilon decay rate: 0.99
    - Learning rate: 0.001
    - Buffer size: 1e5
    - Batch size: 64
    - Gamma: 0.99 
    - Tau: 0.001
    - Update every: 4

## Plot of rewards

In the experiments performed, this architecture and choice of hyperparameters often achieved an average reward >= 13.0 over 100 episodes in less than 400 episodes. In the experiment shown, 370 episodes were enough. 

![plot of rewards](scores.png)

## Ideas for future work

Several improvements can be made to the vanilla DQN, including Prioritized Experience Replay and Dueling DQNs. A list of possible combinations that may lead to improvements is suggested in the [Rainbow paper](https://arxiv.org/abs/1710.02298). These improvements are expected to significantly impact performance, especially on the more challenging task of learning a good policy from raw pixels. 


