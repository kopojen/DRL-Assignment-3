import random
from tqdm import tqdm
import cv2
import torch
import os
import numpy as np
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch import nn
import torch.optim as optim
from typing import Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
import pickle

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
NUM_STACKED_FRAMES = 4
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
TRAIN_ITER = 20000
TRAIN_FREQ = 4
SKIP = 4

class MarioPreprocessor:
    """
    Preprocess frames for Super Mario Bros:
      - Convert RGB to grayscale
      - Resize to (height, width)
      - Normalize pixel values to [0, 1]
      - Stack the last N frames
    """
    def __init__(
        self,
        output_size=(84, 84),
        num_frames=4,
        raw_frame_shape=(240, 256)
    ):
        """
        Args:
            output_size (tuple): Desired (height, width) of processed frames.
            num_frames (int): Number of frames to stack in the state.
            raw_frame_shape (tuple): Shape (height, width) of incoming raw frames.
        """
        self.height, self.width = output_size
        # cv2.resize expects (width, height)
        self.resize_dims = (self.width, self.height)
        self.num_frames = num_frames
        # Initialize buffer with blank frames
        blank = np.zeros(raw_frame_shape, dtype=np.uint8)
        self.frame_buffer = deque([blank.copy() for _ in range(num_frames)], maxlen=num_frames)

    def reset(self):
        """
        Clear and refill the buffer with blank frames at the start of an episode.
        """
        blank = np.zeros((self.height, self.width), dtype=np.uint8)
        self.frame_buffer.clear()
        for _ in range(self.num_frames):
            self.frame_buffer.append(blank.copy())

    def process(self, frame):
        """
        Convert a raw RGB frame to a normalized stack of frames.

        Args:
            frame (np.ndarray): Input RGB image with shape (H, W, 3) and dtype uint8.

        Returns:
            np.ndarray: Stacked frames of shape (num_frames, height, width), dtype float32.

        Raises:
            ValueError: If the input frame is not a valid RGB image.
        """
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Invalid frame shape {frame.shape}, expected (H, W, 3)")

        # 1. Grayscale conversion
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # 2. Resize
        resized = cv2.resize(
            gray,
            self.resize_dims,
            interpolation=cv2.INTER_AREA
        )
        # 3. Update buffer
        self.frame_buffer.append(resized)
        # 4. Stack and normalize
        stacked = np.stack(self.frame_buffer, axis=0).astype(np.float32) / 255.0
        return stacked

class MemoryEfficientPERBuffer:
    """
    Memory-efficient Prioritized Experience Replay (PER) buffer without a SumTree.

    Features:
      - Stores frames as uint8 to minimize memory usage.
      - Samples with direct probability normalization (O(N)).
      - Computes importance-sampling weights with annealed beta.
    """
    def __init__(
        self,
        capacity: int,
        input_shape: tuple = (4, 84, 84),
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000
    ):
        self.capacity = capacity
        self.input_shape = input_shape
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / float(beta_frames)
        self.eps = 1e-6
        self.max_priority = 1.0

        # allocate storage
        self.states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.priorities = np.zeros(capacity, dtype=np.float64)

        self.pointer = 0
        self.size = 0

    def _compress(self, state: np.ndarray) -> np.ndarray:
        """Scale float32 [0,1] to uint8 [0,255]"""
        clipped = np.clip(state, 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)

    def _decompress(self, state_uint8: np.ndarray) -> np.ndarray:
        """Scale uint8 [0,255] back to float32 [0,1]"""
        return state_uint8.astype(np.float32) / 255.0

    def _priority(self, error: float) -> float:
        """Compute priority from TD error."""
        return (abs(error) + self.eps) ** self.alpha

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Insert a new transition, overwriting oldest if full."""
        s = self._compress(state)
        ns = self._compress(next_state)

        idx = self.pointer
        self.states[idx] = s
        self.next_states[idx] = ns
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.priorities[idx] = self.max_priority

        # advance pointer
        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample a batch of experiences.

        Returns:
            batch (dict): arrays of states, actions, rewards, next_states, dones
            indices (np.ndarray): positions sampled
            is_weights (np.ndarray): importance-sampling weights
        """
        if self.size < batch_size:
            raise ValueError("Not enough samples to draw the batch")

        prios = self.priorities[: self.size]
        total = np.sum(prios)
        if total <= 0:
            probs = np.ones(self.size) / self.size
        else:
            probs = prios / total

        indices = np.random.choice(self.size, batch_size, p=probs)
        self.beta = min(1.0, self.beta + self.beta_increment)

        # compute IS weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # gather batch
        bs_uint8 = self.states[indices]
        bns_uint8 = self.next_states[indices]
        batch = {
            'states': bs_uint8.astype(np.float32) / 255.0,
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': bns_uint8.astype(np.float32) / 255.0,
            'dones': self.dones[indices]
        }
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, errors: np.ndarray):
        """Update priorities of sampled indices from their new TD errors."""
        for idx, err in zip(indices, errors):
            if idx < self.size:
                p = self._priority(err)
                self.priorities[idx] = p
                self.max_priority = max(self.max_priority, p)

    def __len__(self) -> int:
        return self.size

class DuelingCNNQNet(nn.Module):
    """
    Dueling Deep Q-Network with convolutional feature extractor for image inputs.

    Architecture:
      1. Convolutional backbone: 3 conv layers with ReLU
      2. Splits into two streams:
         - Value stream V(s)
         - Advantage stream A(s,a)
      3. Combines via Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int
    ):
        super().__init__()
        c, h, w = input_shape
        self.num_actions = num_actions

        # --- convolutional feature extractor ---
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_layers = self.feature_extractor
        # compute flattened feature size for linear layers
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_out = self.feature_extractor(dummy)
            flat_size = int(np.prod(feat_out.shape[1:]))

        # --- value stream ---
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

        # --- advantage stream ---
        self.adv_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )
        
        self.advantage_stream = self.adv_stream

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, C, H, W), dtype float32.
               Expected normalized inputs.

        Returns:
            Tensor of Q-values, shape (batch, num_actions).
        """
        # extract features
        features = self.feature_extractor(x)
        # flatten
        flat = features.view(features.size(0), -1)

        # compute streams
        value = self.value_stream(flat)              # shape: (batch, 1)
        advantages = self.adv_stream(flat)           # shape: (batch, num_actions)

        # combine into Q-values
        q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class RainbowDQNAgent:
    """
    Partial Rainbow DQN Agent with:
      - Dueling architecture
      - Double DQN
      - Prioritized Experience Replay

    Note: Excludes N-step returns, C51 distributional RL, and NoisyNets.
    """
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_actions: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.7,
        buffer_capacity: int = 50_000,
        batch_size: int = 128,
        target_update_freq: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9999
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

        # Q-network and target network
        self.q_net = DuelingCNNQNet(input_shape, num_actions).to(device)
        self.target_net = DuelingCNNQNet(input_shape, num_actions).to(device)
        self._sync_target()
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')

        # Replay buffer
        self.replay_buffer = MemoryEfficientPERBuffer(capacity=buffer_capacity,
                                                     input_shape=input_shape)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        """
        if explore and random.random() < self.epsilon:
            return random.randrange(self.q_net.num_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_t = state_t.unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self,
              state: np.ndarray,
              action: int,
              reward: float,
              next_state: np.ndarray,
              done: bool) -> None:
        """Add transition to PER buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform a single training update. Returns loss or None if skipped.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch, indices, is_weights = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Double DQN target computation
        with torch.no_grad():
            next_q_main = self.q_net(next_states)
            best_actions = next_q_main.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, best_actions)
            td_target = rewards + self.gamma * next_q_target * (1 - dones)

        # Current Q-values
        q_values = self.q_net(states).gather(1, actions)

        # Compute weighted loss
        td_error = td_target - q_values
        loss = (self.loss_fn(q_values, td_target) * weights).mean()

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update PER priorities
        errors = td_error.abs().detach().cpu().numpy().flatten()
        self.replay_buffer.update_priorities(indices, errors)

        # Target network sync and epsilon decay
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self._sync_target()
        self._decay_epsilon()

        return loss.item()

    def _sync_target(self) -> None:
        """Copy Q-network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _decay_epsilon(self) -> None:
        """Anneal exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, q_path: str, target_path: Optional[str] = None) -> None:
        """Save model weights."""
        torch.save(self.q_net.state_dict(), q_path)
        if target_path:
            torch.save(self.target_net.state_dict(), target_path)

    def load(self,
             q_path: str,
             target_path: Optional[str] = None) -> None:
        """Load model weights."""
        self.q_net.load_state_dict(torch.load(q_path, map_location=self.device))
        if target_path:
            self.target_net.load_state_dict(torch.load(target_path, map_location=self.device))
        else:
            self._sync_target()
        self.q_net.train()
        self.target_net.eval()

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done: 
                break
        return obs, total_reward, done, info
    def reset(self, **kwargs):
        # 重置時直接呼叫底層 env.reset()
        return self.env.reset(**kwargs)

def run_episode(
    env: gym.Env,
    agent,
    preprocessor,
    train_freq: int,
    train: bool = True
) -> Tuple[float, int]:
    """
    Execute a single episode in the environment.

    Args:
        env (Env): Gym environment (e.g. JoypadSpace).
        agent: Agent with methods select_action, store, train_step, decay_epsilon.
        preprocessor: Preprocessor with methods reset() and process(frame).
        train_freq (int): Number of steps between training updates.
        train (bool): Whether to perform training during the episode.

    Returns:
        total_reward (float): Sum of rewards obtained in the episode.
        total_steps (int): Number of steps taken in the episode.
    """
    # Initialize environment and buffer
    raw_state = env.reset()
    preprocessor.reset()
    state = preprocessor.process(raw_state)

    total_reward = 0.0
    total_steps = 0
    done = False

    # Run until episode termination
    while not done:
        # Choose action (epsilon-greedy if train)
        action = agent.select_action(state, explore=train)

        # Step environment
        raw_next, reward, done, _ = env.step(action)
        next_state = preprocessor.process(raw_next)

        # Store transition
        agent.store(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        total_steps += 1

        # Periodic training step
        if train and total_steps % train_freq == 0:
            agent.train_step()

    return total_reward, total_steps

def pre_eval(env, agent, processor, iter=1):
    score, _ = run_episode(env, agent, processor, TRAIN_FREQ, train=False)
    print("Pre Eval Reward: ", score)
    return score

MOVING_AVG_WINDOW = 10
LEVEL_NAME = "SuperMarioBros-v0"
SAVE_SCORE_THRESHOLD = 4000

def save_learning_curve(episodes, scores):
    """把目前為止的學習曲線存成 png"""
    plt.figure()
    plt.plot(episodes, scores, alpha=0.3, label="score")
    if len(scores) >= MOVING_AVG_WINDOW:
        ma = np.convolve(
            scores,
            np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW,
            mode="valid",
        )
        plt.plot(
            episodes[MOVING_AVG_WINDOW - 1 :],
            ma,
            linewidth=2,
            label=f"{MOVING_AVG_WINDOW}-ep moving avg",
        )
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title(f"Learning Curve - {LEVEL_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"learning_curve_{LEVEL_NAME}.png")
    plt.close()
    
def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, SKIP)
    agent = RainbowDQNAgent(input_shape=(NUM_STACKED_FRAMES, FRAME_WIDTH, FRAME_HEIGHT), num_actions=12, device=DEVICE)
    processor = MarioPreprocessor()
    
    episodes = []
    scores   = []
    best_score = 0.0
    total_steps = 0
    
    tq = tqdm(range(TRAIN_ITER), desc="Training")
    for episode in tq:
        total_score, step = run_episode(env, agent, processor, TRAIN_FREQ, train=True)
        
        # 紀錄
        episodes.append(episode + 1)
        scores.append(total_score)
        
        save_learning_curve(episodes, scores)
        
        # 動態擷取目前最佳分數、ε 和步數
        total_steps += step
        
        # 更新進度列後綴
        tq.set_postfix(
            ep=episode+1,
            score=f"{total_score:.1f}",
            best=f"{best_score:.1f}",
            eps=f"{agent.epsilon:.3f}",
            steps=f"{total_steps}"
        )
        
        if total_score >= best_score and total_score >= 5000:
            score = pre_eval(env, agent, processor)
            if score > best_score:
                best_score = max(best_score, score)
                agent.save(f"checkpoints/mario_q_ep{episode+1}_score{int(score)}.pth", f"checkpoints/mario_q_target_ep{episode+1}_score{int(score)}.pth")
                print(f"[儲存模型] 第 {episode+1} 回合，分數 {score:.1f} ≥ {SAVE_SCORE_THRESHOLD}，已存檔。")
    
if __name__ == "__main__":
    main()