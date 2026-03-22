"""
Mini-3 Problem 1: Cadmium Rod Control (Tabular RL with noisy observations)

Environment (from handout):
- Hidden state: mu_t (temperature proxy)
- Dynamics:
    mu_{t+1} = 0.7*mu_t + 2*u_t - 0.1 + w_t,  w_t ~ N(0, 0.5)
- Actions: u_t in {0,1}
    u=0 : insert rod (cool)
    u=1 : withdraw rod (heat)
- Observation:
    y_t = mu_t + v_t,  v_t ~ N(0, sigma^2)
- Agent state for RL: binned observation
    z_t = clip( floor(y_t / 2), 0, 4 )  (5 bins: 0..4)
- Termination:
    meltdown if mu_t > 9  -> x_t = 0 (terminal failure)
    goal if 7 <= mu_t <= 9 -> x_t = 1 (terminal success)
- Reward:
    r_t = -1 each step, plus +1 at termination if goal reached.
    (So terminal success yields r=0 on final step if you do -1 +1.)
    We'll implement: reward=-1 always, and if terminal success, add +1 => 0 on success terminal step.

Algorithms implemented:
- Q-learning (off-policy)
- SARSA(λ) with replacing traces (on-policy)

Outputs:
- learning_curves.png (returns vs episode, for each algorithm and sigma^2)
- heatmaps_QV_*.png (Q(z,a) and V(z) for each algorithm/sigma^2)
- summary.csv (basic stats)

Usage examples:
  python mini3_p1_tabular_rl.py
  python mini3_p1_tabular_rl.py --episodes 8000 --sigmasq 0.25 2.0
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Environment
# -----------------------------
@dataclass
class CadmiumRodEnv:
    sigmasq: float = 1.0
    max_steps: int = 500
    rng: np.random.Generator = None

    # Dynamics parameters
    a: float = 0.7
    b: float = 2.0
    c: float = -0.1
    w_var: float = 0.5  # w_t ~ N(0, 0.5)

    # Termination thresholds on hidden mu
    meltdown_thresh: float = 9.0      # mu > 9 -> failure terminal
    goal_low: float = 7.0             # 7 <= mu <= 9 -> success terminal
    goal_high: float = 9.0

    # Observation binning
    bin_width: float = 2.0
    n_bins: int = 5  # z in {0,1,2,3,4}

    def reset(self) -> Tuple[int, Dict]:
        if self.rng is None:
            self.rng = np.random.default_rng(0)
        # mu_0 ~ N(0,1) as typical default if not otherwise specified
        self.mu = float(self.rng.normal(0.0, 1.0))
        self.t = 0
        z = self._observe_bin()
        return z, {"mu": self.mu, "y": self.last_y}

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        action: 0 or 1
        returns: z_next, reward, done, info
        """
        assert action in (0, 1), "Action must be 0 or 1"
        self.t += 1

        w = float(self.rng.normal(0.0, math.sqrt(self.w_var)))
        self.mu = self.a * self.mu + self.b * float(action) + self.c + w

        # termination check on NEW mu_t (after transition)
        done = False
        success = False

        if self.mu > self.meltdown_thresh:
            done = True
            success = False
        elif self.goal_low <= self.mu <= self.goal_high:
            done = True
            success = True
        elif self.t >= self.max_steps:
            done = True
            success = False  # timeout treated as failure for stats

        # reward: -1 each step, +1 if terminal success
        reward = -1.0 + (1.0 if (done and success) else 0.0)

        z_next = self._observe_bin()
        info = {
            "mu": self.mu,
            "y": self.last_y,
            "done_reason": "success" if (done and success) else ("meltdown" if self.mu > self.meltdown_thresh else ("timeout" if self.t >= self.max_steps else ""))
        }
        return z_next, reward, done, info

    def _observe_bin(self) -> int:
        v = float(self.rng.normal(0.0, math.sqrt(self.sigmasq)))
        y = self.mu + v
        self.last_y = y
        z = int(math.floor(y / self.bin_width))
        z = max(0, min(self.n_bins - 1, z))
        return z


# -----------------------------
# Policies / helpers
# -----------------------------
def epsilon_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, Q.shape[1]))
    # tie-break randomly
    best = np.max(Q[s])
    best_actions = np.flatnonzero(np.isclose(Q[s], best))
    return int(rng.choice(best_actions))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    w = min(window, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[w:] - c[:-w]) / w
    # pad to original length for plotting alignment
    pad = np.full(w - 1, ma[0])
    return np.concatenate([pad, ma])


# -----------------------------
# Q-learning
# -----------------------------
def train_q_learning(
    sigmasq: float,
    episodes: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay_frac: float,
    seed: int,
    max_steps: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    env = CadmiumRodEnv(sigmasq=sigmasq, max_steps=max_steps, rng=rng)

    nS = env.n_bins
    nA = 2
    Q = np.zeros((nS, nA), dtype=float)

    returns = np.zeros(episodes, dtype=float)
    success = np.zeros(episodes, dtype=int)
    meltdowns = np.zeros(episodes, dtype=int)

    decay_episodes = max(1, int(eps_decay_frac * episodes))

    for ep in range(episodes):
        # linear epsilon decay
        if ep < decay_episodes:
            eps = eps_start + (eps_end - eps_start) * (ep / decay_episodes)
        else:
            eps = eps_end

        s, info = env.reset()
        G = 0.0
        done = False

        while not done:
            a = epsilon_greedy(Q, s, eps, rng)
            s2, r, done, info2 = env.step(a)
            G += r

            # Q-learning update
            td_target = r + gamma * (0.0 if done else np.max(Q[s2]))
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s2

        returns[ep] = G
        if info2["done_reason"] == "success":
            success[ep] = 1
        if info2["done_reason"] == "meltdown":
            meltdowns[ep] = 1

    V = np.max(Q, axis=1)

    return {
        "algo": "Q-learning",
        "sigmasq": sigmasq,
        "Q": Q,
        "V": V,
        "returns": returns,
        "success": success,
        "meltdowns": meltdowns,
    }


# -----------------------------
# SARSA(lambda) with replacing traces
# -----------------------------
def train_sarsa_lambda(
    sigmasq: float,
    episodes: int,
    alpha: float,
    gamma: float,
    lam: float,
    eps_start: float,
    eps_end: float,
    eps_decay_frac: float,
    seed: int,
    max_steps: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    env = CadmiumRodEnv(sigmasq=sigmasq, max_steps=max_steps, rng=rng)

    nS = env.n_bins
    nA = 2
    Q = np.zeros((nS, nA), dtype=float)

    returns = np.zeros(episodes, dtype=float)
    success = np.zeros(episodes, dtype=int)
    meltdowns = np.zeros(episodes, dtype=int)

    decay_episodes = max(1, int(eps_decay_frac * episodes))

    for ep in range(episodes):
        if ep < decay_episodes:
            eps = eps_start + (eps_end - eps_start) * (ep / decay_episodes)
        else:
            eps = eps_end

        s, _ = env.reset()
        a = epsilon_greedy(Q, s, eps, rng)

        E = np.zeros_like(Q)  # eligibility traces
        G = 0.0
        done = False

        while not done:
            s2, r, done, info2 = env.step(a)
            G += r

            if done:
                td_target = r
                delta = td_target - Q[s, a]
                # replacing trace
                E[s, a] = 1.0
                Q += alpha * delta * E
                # decay traces
                E *= gamma * lam
                break

            a2 = epsilon_greedy(Q, s2, eps, rng)

            td_target = r + gamma * Q[s2, a2]
            delta = td_target - Q[s, a]

            # replacing trace for current (s,a)
            E[s, a] = 1.0

            Q += alpha * delta * E

            # decay traces
            E *= gamma * lam

            s, a = s2, a2

        returns[ep] = G
        if info2["done_reason"] == "success":
            success[ep] = 1
        if info2["done_reason"] == "meltdown":
            meltdowns[ep] = 1

    V = np.max(Q, axis=1)

    return {
        "algo": f"SARSA(λ={lam})",
        "sigmasq": sigmasq,
        "Q": Q,
        "V": V,
        "returns": returns,
        "success": success,
        "meltdowns": meltdowns,
    }


# -----------------------------
# Plotting / saving
# -----------------------------
def save_heatmaps(out_dir: str, result: Dict):
    algo = result["algo"]
    sigmasq = result["sigmasq"]
    Q = result["Q"]
    V = result["V"]

    # Q heatmap: z rows, a columns
    plt.figure()
    plt.imshow(Q, origin="lower", aspect="auto")
    plt.xlabel("action a (0=insert, 1=withdraw)")
    plt.ylabel("binned state z")
    plt.title(f"Q(z,a) heatmap — {algo}, σ²={sigmasq}")
    plt.colorbar()
    plt.yticks(range(Q.shape[0]))
    plt.xticks([0, 1], ["0", "1"])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_Q_{algo.replace(' ', '_')}_sig{sigmasq}.png"), dpi=200)
    plt.close()

    # V heatmap (1D, shown as a single-row image)
    plt.figure()
    plt.imshow(V.reshape(1, -1), origin="lower", aspect="auto")
    plt.xlabel("binned state z")
    plt.yticks([])
    plt.title(f"V(z)=max_a Q(z,a) — {algo}, σ²={sigmasq}")
    plt.colorbar()
    plt.xticks(range(len(V)))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_V_{algo.replace(' ', '_')}_sig{sigmasq}.png"), dpi=200)
    plt.close()


def save_learning_curves(out_dir: str, all_results: List[Dict], ma_window: int = 200):
    plt.figure()
    for res in all_results:
        y = res["returns"]
        y_ma = moving_average(y, ma_window)
        plt.plot(y_ma, label=f"{res['algo']} (σ²={res['sigmasq']})")
    plt.xlabel("episode")
    plt.ylabel("return (smoothed)")
    plt.title(f"Learning curves (moving avg window={ma_window})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=200)
    plt.close()


def write_summary_csv(out_dir: str, all_results: List[Dict]):
    path = os.path.join(out_dir, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([
            "algo", "sigmasq",
            "episodes",
            "avg_return_last_1000",
            "success_rate_last_1000",
            "meltdown_rate_last_1000",
        ])
        for res in all_results:
            y = res["returns"]
            s = res["success"]
            m = res["meltdowns"]
            n = len(y)
            k = min(1000, n)
            w.writerow([
                res["algo"], res["sigmasq"],
                n,
                float(np.mean(y[-k:])),
                float(np.mean(s[-k:])),
                float(np.mean(m[-k:])),
            ])


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="mini3_p1_out", help="output directory")
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=500)

    # noise levels: provide σ² values
    ap.add_argument("--sigmasq", type=float, nargs="+", default=[0.25, 2.0])

    # common hyperparams
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--eps_start", type=float, default=0.2)
    ap.add_argument("--eps_end", type=float, default=0.02)
    ap.add_argument("--eps_decay_frac", type=float, default=0.6, help="fraction of training to decay epsilon")

    # SARSA(λ)
    ap.add_argument("--lam", type=float, default=0.8)

    # plotting
    ap.add_argument("--ma_window", type=int, default=200)

    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_results: List[Dict] = []

    # For each noise level, run both algorithms
    for sig2 in args.sigmasq:
        res_q = train_q_learning(
            sigmasq=sig2,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_frac=args.eps_decay_frac,
            seed=args.seed + 1000 + int(sig2 * 100),
            max_steps=args.max_steps,
        )
        all_results.append(res_q)

        res_sarsa = train_sarsa_lambda(
            sigmasq=sig2,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            lam=args.lam,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay_frac=args.eps_decay_frac,
            seed=args.seed + 2000 + int(sig2 * 100),
            max_steps=args.max_steps,
        )
        all_results.append(res_sarsa)

    # Save learning curves
    save_learning_curves(args.out, all_results, ma_window=args.ma_window)

    # Save heatmaps for each run
    for res in all_results:
        save_heatmaps(args.out, res)

    # Save summary
    write_summary_csv(args.out, all_results)

    print("Done.")
    print(f"Outputs written to: {args.out}")
    print("Key files:")
    print("  learning_curves.png")
    print("  heatmap_Q_*.png, heatmap_V_*.png")
    print("  summary.csv")


if __name__ == "__main__":
    main()