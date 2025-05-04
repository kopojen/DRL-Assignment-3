import gym
import torch
from train import DuelingDQN, FramePreprocessor  # 注意要 match train.py 的 class 名稱

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_PATH = "best.pth"  # 替換成你要用的檔案

class Agent:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(12)
        self.q_net = DuelingDQN((4, 84, 84), 12).to(DEVICE)
        self.q_net.load_state_dict(torch.load(Q_PATH, map_location=DEVICE))
        self.q_net.eval()
        self.processor = FramePreprocessor()
        self.skip_count = 0
        self.last_action = None

    def act(self, observation):
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        state = self.processor.process(observation)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state)
        action = int(q_values.argmax().item())

        self.last_action = action
        self.skip_count = 3
        return action