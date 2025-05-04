import os
import cv2
import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Optional
from collections import deque
from torch import nn
import torch.optim as optim
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
)
STACKED_FRAMES   = 4
FRAME_H, FRAME_W = 84, 84
TOTAL_EPISODES   = 20_000
TRAIN_EVERY      = 4
FRAME_SKIP       = 4
ENV_ID           = "SuperMarioBros-v0"
CKPT_DIR         = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

class FramePreprocessor:
    """
    • RGB → gray • Resize → 84×84 • Normalize • Stack recent N frames
    """
    def __init__(self, output_size=(84, 84), n_frames=4):
        self.h, self.w = output_size
        self.n_frames  = n_frames
        self.resize_xy = (self.w, self.h)
        blank          = np.zeros((self.h, self.w), dtype=np.uint8)
        self.buffer    = deque([blank.copy() for _ in range(n_frames)],
                               maxlen=n_frames)

    def reset(self):
        """Flush frame buffer with blanks."""
        blank = np.zeros((self.h, self.w), dtype=np.uint8)
        self.buffer.clear()
        for _ in range(self.n_frames):
            self.buffer.append(blank.copy())

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return stacked (N,84,84) float32 in [0,1]."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) RGB, got {frame.shape}")
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.resize_xy, interpolation=cv2.INTER_AREA)
        self.buffer.append(resized)
        stacked = np.stack(self.buffer, axis=0).astype(np.float32) / 255.0
        return stacked

class PrioritizedReplay:
    """Simple PER without SumTree (O(N) sample)."""
    def __init__(self, capacity: int, obs_shape=(4, 84, 84),
                 alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.cap   = capacity
        self.alpha = alpha
        self.beta  = beta_start
        self.beta_inc = (1.0 - beta_start) / beta_frames
        self.eps   = 1e-6
        self.max_p = 1.0

        self.states      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros_like(self.states)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=bool)
        self.priorities  = np.zeros(capacity, dtype=np.float64)
        self.ptr = 0
        self.size = 0

    def _priority(self, td_err: float) -> float:
        return (abs(td_err) + self.eps) ** self.alpha

    def _pack(self, s: np.ndarray) -> np.ndarray:
        return (np.clip(s, 0, 1) * 255).astype(np.uint8)

    def add(self, s, a, r, ns, done):
        idx                 = self.ptr
        self.states[idx]    = self._pack(s)
        self.next_states[idx] = self._pack(ns)
        self.actions[idx]   = a
        self.rewards[idx]   = r
        self.dones[idx]     = done
        self.priorities[idx]= self.max_p

        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int):
        if self.size < batch:
            raise ValueError("not enough samples")
        probs   = self.priorities[:self.size]
        probs   = probs / probs.sum()
        idxs    = np.random.choice(self.size, batch, p=probs)
        self.beta = min(1.0, self.beta + self.beta_inc)
        weights = (self.size * probs[idxs]) ** (-self.beta)
        weights = weights / weights.max()

        to_float = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)
        batch_dict = dict(
            states      = self.states[idxs].astype(np.float32) / 255.0,
            actions     = self.actions[idxs],
            rewards     = self.rewards[idxs],
            next_states = self.next_states[idxs].astype(np.float32) / 255.0,
            dones       = self.dones[idxs]
        )
        return batch_dict, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, td_errs):
        for i, err in zip(idxs, td_errs):
            p = self._priority(err)
            self.priorities[i] = p
            self.max_p = max(self.max_p, p)

    def __len__(self):
        return self.size

class DuelingDQN(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        c, h, w = in_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        with torch.no_grad():
            flat = int(np.prod(self.features(torch.zeros(1, *in_shape)).shape[1:]))
        self.value = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, 1))
        self.adv   = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        v = self.value(f)
        a = self.adv(f)
        return v + a - a.mean(dim=1, keepdim=True)

class RainbowAgent:
    """Dueling, Double-DQN, PER (no C51/NoisyNet/N-step)."""
    def __init__(self, in_shape, n_act, device=DEVICE,
                 lr=1e-4, gamma=0.7, buf_cap=50_000,
                 batch=128, tgt_freq=1_000,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.9999):
        self.device = device
        self.gamma  = gamma
        self.batch  = batch
        self.tgt_fq = tgt_freq
        self.eps    = eps_start
        self.eps_end= eps_end
        self.eps_decay = eps_decay
        self.step   = 0

        self.policy_net = DuelingDQN(in_shape, n_act).to(device)
        self.target_net = DuelingDQN(in_shape, n_act).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.opt   = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss  = nn.MSELoss(reduction="none")
        self.buffer= PrioritizedReplay(buf_cap, in_shape)

    def select_action(self, state, explore=True):
        if explore and random.random() < self.eps:
            return random.randrange(self.policy_net.adv[-1].out_features)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(state, device=self.device).unsqueeze(0))
        return int(q.argmax())

    def store(self, s, a, r, ns, done):
        self.buffer.add(s, a, r, ns, done)

    def _sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _anneal_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def train_step(self):
        if len(self.buffer) < self.batch: return None
        batch, idxs, w = self.buffer.sample(self.batch)
        s  = torch.tensor(batch['states'],      device=self.device)
        ns = torch.tensor(batch['next_states'], device=self.device)
        a  = torch.tensor(batch['actions'],     device=self.device).unsqueeze(1)
        r  = torch.tensor(batch['rewards'],     device=self.device).unsqueeze(1)
        d = torch.tensor(batch['dones'], dtype=torch.float32, device=self.device).unsqueeze(1)
        w  = torch.tensor(w, device=self.device).unsqueeze(1)

        with torch.no_grad():
            best_a = self.policy_net(ns).argmax(1, keepdim=True)
            tgt_q  = self.target_net(ns).gather(1, best_a)
            y = r + self.gamma * tgt_q * (1 - d)

        q    = self.policy_net(s).gather(1, a)
        td_e = (y - q).detach().cpu().numpy().flatten()
        loss = (self.loss(q, y) * w).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.buffer.update_priorities(idxs, td_e)

        self.step += 1
        if self.step % self.tgt_fq == 0:
            self._sync_target()
        self._anneal_eps()
        return loss.item()

class SkipFrame(gym.Wrapper):
    """Repeat the chosen action `skip` frames and pool rewards."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_r, done = 0, False
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_r += reward
            if done: break
        return obs, total_r, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def play_episode(env, agent, prep, train=True) -> Tuple[float, int]:
    obs = env.reset()
    prep.reset()
    state = prep.process(obs)
    tot_r, steps, done = 0.0, 0, False

    while not done:
        act = agent.select_action(state, explore=train)
        nxt, r, done, _ = env.step(act)
        nxt_state = prep.process(nxt)
        agent.store(state, act, r, nxt_state, done)
        state = nxt_state
        tot_r += r
        steps += 1
        if train and steps % TRAIN_EVERY == 0:
            agent.train_step()
    return tot_r, steps

def quick_eval(env, agent, prep):
    score, _ = play_episode(env, agent, prep, train=False)
    print("Evaluation score:", score)
    return score

def main():
    env        = gym_super_mario_bros.make(ENV_ID)
    env        = JoypadSpace(env, COMPLEX_MOVEMENT)
    env        = SkipFrame(env, FRAME_SKIP)
    processor  = FramePreprocessor()
    agent      = RainbowAgent((STACKED_FRAMES, FRAME_H, FRAME_W), 12, DEVICE)

    ep_hist, score_hist, best = [], [], 0.0
    bar = tqdm(range(TOTAL_EPISODES), desc="Training")
    for ep in bar:
        sc, st = play_episode(env, agent, processor, train=True)
        ep_hist.append(ep + 1)
        score_hist.append(sc)

        if sc >= best and sc >= 4000:
            if quick_eval(env, agent, processor) > best:
                best = sc
                q_path = f"{CKPT_DIR}/mario_q_ep{ep+1}_score{int(sc)}.pth"
                agent.save(q_path)
                print(f"[Checkpoint] Episode {ep+1} – saved {q_path}")

        bar.set_postfix(ep=ep+1, score=f"{sc:.0f}", best=f"{best:.0f}",
                        epsilon=f"{agent.eps:.3f}")

        # live curve
        if (ep+1) == 0:
            plt.figure()
            plt.plot(ep_hist, score_hist, alpha=0.3, label="score")
            if len(score_hist) >= 10:
                ma = np.convolve(score_hist, np.ones(10)/10, mode="valid")
                plt.plot(ep_hist[9:], ma, label="10-ep MA")
            plt.xlabel("Episode"); plt.ylabel("Score")
            plt.title(f"Learning – {ENV_ID}"); plt.legend()
            plt.savefig(f"learning_curve_{ENV_ID}.png"); plt.close()

if __name__ == "__main__":
    main()