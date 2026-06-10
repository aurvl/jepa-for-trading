from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from jepa_trading.models.policy import PortfolioPolicy


@dataclass
class PPOConfig:
    rollout_steps: int = 252
    total_updates: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 1e-4
    minibatch_size: int = 64
    epochs_per_update: int = 4


def compute_gae(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t]
        nextvalue = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
    return adv, adv + values[: len(rewards)]


def train_ppo(env, policy: PortfolioPolicy, config: PPOConfig, device: torch.device) -> pd.DataFrame:
    policy.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []

    obs, mask = env.reset()
    for update in tqdm(range(1, config.total_updates + 1), desc="train PPO", dynamic_ncols=True):
        obs_buf, mask_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], [], []
        for _ in range(config.rollout_steps):
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask[None], dtype=torch.bool, device=device)
            with torch.no_grad():
                action_t, logp_t, _, value_t = policy.get_action_and_value(obs_t, mask_t)
            action = action_t.squeeze(0).cpu().numpy()
            next_obs, reward, done, _, next_mask = env.step(action)
            obs_buf.append(obs)
            mask_buf.append(mask)
            act_buf.append(action)
            logp_buf.append(float(logp_t.item()))
            rew_buf.append(reward)
            done_buf.append(float(done))
            val_buf.append(float(value_t.item()))
            obs, mask = next_obs, next_mask
            if done:
                obs, mask = env.reset()

        rewards = np.array(rew_buf, dtype=np.float32)
        values = np.array(val_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, config.gamma, config.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        mask_t = torch.tensor(np.array(mask_buf), dtype=torch.bool, device=device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        idx = np.arange(len(obs_buf))
        losses = []
        for _ in range(config.epochs_per_update):
            np.random.shuffle(idx)
            for start in range(0, len(idx), config.minibatch_size):
                mb = idx[start : start + config.minibatch_size]
                _, new_logp, entropy, value = policy.get_action_and_value(obs_t[mb], mask_t[mb], act_t[mb])
                ratio = torch.exp(new_logp - old_logp_t[mb])
                pg1 = -adv_t[mb] * ratio
                pg2 = -adv_t[mb] * torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = F.mse_loss(value, ret_t[mb])
                loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

        history.append(
            {
                "update": update,
                "loss": float(np.mean(losses)),
                "rollout_reward_mean": float(rewards.mean()),
                "rollout_reward_sum": float(rewards.sum()),
                "equity": float(env.state.equity),
            }
        )
    return pd.DataFrame(history)

