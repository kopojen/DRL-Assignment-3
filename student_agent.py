import gym
import torch
from train import RainbowDQNAgent, MarioPreprocessor

# Do not modify the input of the 'act' function and the '__init__' function.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_PATH = "best_q_6637.pth"
TARGET_PATH = "best_t_6637.pth"

class Agent(object):
    """Agent that selects actions using a trained Rainbow DQN."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.agent = RainbowDQNAgent((4, 84, 84), 12, DEVICE)
        self.agent.load(Q_PATH, TARGET_PATH)
        self.processor = MarioPreprocessor()
        self.previous_action = None
        self.frame_skip_counter = 0

    def act(self, observation):
        if self.frame_skip_counter > 0:
            self.frame_skip_counter -= 1
            return self.previous_action
        
        state = self.processor.process(observation)
        action = self.agent.select_action(state, use_epsilon=False)
        self.frame_skip_counter = 3
        self.previous_action = action
        return action