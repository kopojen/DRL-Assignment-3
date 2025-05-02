import os
import gym
import torch
from train import RainbowDQNAgent, MarioPreprocessor   # 用原本的 Preprocessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(object):
    """Rainbow-DQN Mario agent (流程與 train/eval 完全一致)"""

    def __init__(self):
        # 12 discrete actions (COMPLEX_MOVEMENT)
        self.action_space = gym.spaces.Discrete(12)

        # — 載入最高分 checkpoint —
        ckpt = os.path.join(os.path.dirname(__file__), "best_6637.pth")
        self.agent = RainbowDQNAgent((4, 84, 84), 12, DEVICE)
        self.agent.load(ckpt)           # 內含 _sync_target()
        self.agent.epsilon = 0.0        # 評測時全 greedy

        # — 前處理器 —
        self.proc = MarioPreprocessor()   # train.py 同款 (已改成 84×84 空幀)
        self.new_episode = True

    def reset(self):
        self.proc.reset()
        self.new_episode = True

    def act(self, observation):
        # 第一次進來：buffer = 3 blank + 1 real (與訓練、eval 完全一致)
        if self.new_episode:
            self.proc.reset()
            state = self.proc.process(observation)   # push 1 次就停
            self.new_episode = False
        else:
            state = self.proc.process(observation)

        # 直接用模型選擇動作
        action = self.agent.select_action(state, explore=False)
        return action