# Reactor Control using Reinforcement Learning (Q-Learning & SARSA)

## 📌 Overview
This project models a nuclear reactor control system as a Markov Decision Process (MDP) under partial observability. The agent must maintain reactor reactivity within a safe and productive range using noisy sensor readings.

## 🎯 Objectives
- Prevent unsafe reactor states (meltdown)
- Maintain operation within productive range
- Handle uncertainty due to noisy observations

## ⚙️ Environment Details
- Hidden state: reactor reactivity (μ)
- Observations: noisy measurements z ~ N(μ, σ²)
- Discretized into 5 bins
- Action space: {insert rod, withdraw rod}

## 🧠 Algorithms Implemented
- Q-learning (off-policy)
- SARSA(λ) (on-policy with eligibility traces)

## 📊 Experiments
- Evaluated under two noise levels:
  - σ² = 0.25 (low noise)
  - σ² = 2.0 (high noise)

### Key Outputs:
- Learning curves
- Q-value heatmaps
- State-value heatmaps
- Performance metrics:
  - Average reward
  - Success rate
  - Meltdown rate

## 📈 Results
- Q-learning converges faster in low-noise environments
- SARSA(λ) provides more stable behavior during exploration
- Higher noise significantly reduces performance and increases uncertainty
- Meltdown rate reduced to <5% in low-noise setting

## 🔍 Key Insights
- Partial observability degrades policy quality
- Discretization introduces approximation error
- Higher noise leads to smoother but less informative value functions

## ▶️ How to Run

```bash
python src/q_learning.py
python src/sarsa_lambda.py