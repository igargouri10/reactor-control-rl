# Reinforcement Learning for Reactor Control (Tabular RL)

This project implements a reinforcement learning framework for controlling a stochastic reactor system under partial observability. The environment is modeled as a Markov Decision Process (MDP) with noisy observations, and control policies are learned using tabular Q-learning and SARSA(λ).

---

## 📌 Problem Overview

The goal is to control reactor reactivity while balancing:

- ⚠️ Safety — avoiding meltdown states  
- ⚙️ Productivity — maintaining optimal operating conditions  

The agent does not observe the true reactor state directly. Instead, it receives noisy measurements, making the problem partially observable and more challenging.

---

## 🧠 Methods Implemented

### 1. Q-Learning
- Off-policy learning algorithm  
- Learns optimal action-value function  
- Faster convergence in low-noise settings  

### 2. SARSA(λ)
- On-policy learning with eligibility traces  
- More stable learning under uncertainty  
- Better robustness to noisy observations  

---

## ⚙️ Environment Design

- Continuous reactor state (hidden)
- Noisy observation model
- Discretized observation space (5 bins)
- Binary action space:
  - Increase reactivity
  - Decrease reactivity

### Reward Function
- Positive reward for stable operation  
- Large penalty for meltdown  
- Small step penalty to encourage efficiency  

---

## 📊 Experiments

Two observation noise levels were evaluated:

- Low noise: σ² = 0.25  
- High noise: σ² = 2.0  

Each algorithm was trained and evaluated under both conditions.

---

## 📈 Results

### Key Findings

- Q-learning converges faster in low-noise environments  
- SARSA(λ) demonstrates more stable behavior under high noise  
- Increased noise leads to smoother and less distinguishable value functions  
- Policies become more conservative as uncertainty increases  

---

## 📁 Repository Structure
problem1_reactor_control/
│
├── mini3_p1_tabular_rl.py # Main implementation
├── mini3_p1_out/ # Generated outputs
│ ├── learning_curves.png
│ ├── heatmap_Q_.png
│ ├── heatmap_V_.png
│ └── summary.csv
│
├── README.md
└── .gitignore

---

## ▶️ How to Run

```bash
python mini3_p1_tabular_rl.py

📊 Outputs
Running the script generates:
  📈 Learning curves
  🔥 Q-value heatmaps
  🌡️ State-value heatmaps
  📄 Summary metrics (CSV file)

🧪 Key Insights
- Observation noise significantly impacts policy quality
- Tabular methods struggle with partial observability
- SARSA(λ) provides improved stability under uncertainty
- Discretization introduces approximation error but keeps learning tractable

🚀 Future Improvements
- Deep RL (DQN) for continuous state representation
- POMDP-based approaches
- Function approximation (linear / neural networks)
- Adaptive control under system drift

🛠️ Tech Stack
- Python
- NumPy
- Matplotlib

👤 Author
Ismail Gargouri
PhD Student in Computer Science – University of North Dakota

⭐ Notes

This project is part of a reinforcement learning coursework focusing on control systems, partial observability, and policy robustness under uncertainty.
