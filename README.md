# Reachinator
DDPG to solve Reacher in a Unity ML-Toolkit Environment

`Reachinator` is a deep deterministic policy gradeint (DDPG) agent trained to play [Unity ML-Agent Toolkit](https://github.com/Unity-Technologies/ml-agents)'s [Reacher](https://www.youtube.com/watch?v=2N9EoF6pQyE). We train 20 agents simultaneously to keep close to the goal location as long as possible (+0.1 per time step). The model solved the environment (scoring a 100-play moving average across all agents of 30 or above) in 104 episodes (roughly 15 minutes). The weights of trained network are saved as `actor_optimal.pth` and `critic_optimal.pth`.

## Environment

The environment consists of 33 values that describe a state and 4 values that describe an action. The state represents position, rotation, velocity, and angular velocities of the two arm Rigidbodies. The action is torque applicable to two joints. The episode ends after 1000 timesteps. These rewards sum up to a score at the end of each episode (1000 timesteps). The environment is considered solved when the average score of the last 100 episodes exceed 30.

```
Actions look like: 
[ 1.        , -0.51279938,  0.13521564,  1.        ]
Action size: 4
States look like: 
[ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726671e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
State size: 33
```

## Getting Started

0. Install dependencies.

```
pip install -r requirements.txt
```

1. Clone this repository and install `unityagents`.

```
pip -q install ./python
```

2. Import `UnityEnvironment` and load environment.

```
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="Reacher_Linux_OneAgent/Reacher.x86_64")
```

3. Follow `train_agent.ipynb` to train the agent.

4. Our implementation is divided as follows:
* `memory.py` - Experience Replay Memory
* `agent.py` - Agent
* `network` - Q-networks for local and target

## Train Agent

These are the steps you can take to train the agent with default settings.

1. Create a experience replay memory.

```
mem = VanillaMemory(int(1e5), seed = 0)
```

2. Create an agent.

```
agent = Agent(state_size=33, action_size=4, replay_memory=mem, random_seed=0, 
              nb_agent = 20, bs = 128,
              gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-4, wd_actor=0, wd_critic=0,
              clip_actor = None, clip_critic = None, update_interval = 1, update_times = 1)
```

3. Train the agent.

```
def get_update_interval(episode, start_update_interval = 1, end_update_interval = 20, gamma = 1.03):
    res = int(start_update_interval * gamma ** (episode))
    return(min(res,end_update_interval))
    
scores = []
moving_scores = []
scores_avg = deque(maxlen=100)
n_episodes = 500
nb_agent = 20

for episode in trange(n_episodes):
    #get initial states
    env_info = env.reset(train_mode=True)[brain_name]            
    states = env_info.vector_observations
    agent.reset_noise()                                             
    score = np.zeros(nb_agent)
    agent.update_interval = get_update_interval(episode)

    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]                 # env step                    
        next_states = env_info.vector_observations               # get the next state        
        rewards = env_info.rewards                               # get the reward        
        dones = env_info.local_done                              # see if episode has finished        
        agent.step(states, actions, rewards, next_states, dones) # agent step
        score += rewards                                         # update the score
        states = next_states                                     # roll over the state to next time step        
        if np.any(dones):                                        # exit loop if episode finished        
            break     
            
    #book keeping
    scores.append(np.mean(score))
    scores_avg.append(np.mean(score))
    moving_scores.append(np.mean(scores_avg))

    #print scores intermittenly
    if episode % 10 ==0: print(f'Episode: {episode} Score: {np.mean(score)} Average Score: {np.mean(scores_avg)}')

    #break if done
    if (np.mean(scores_avg) >= 30) & (len(scores_avg) == 100):
        print(f'Environment solved in {episode} episodes! Average Score: {np.mean(scores_avg)}')
        break    
```