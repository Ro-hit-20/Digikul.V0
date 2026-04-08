---
title: DigiKul-v0
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
- openenv
---

# DigiKul-v0 🎓📡

**A Meta OpenEnv RL Environment for Remote Education Bandwidth Orchestration**

> *Team GreyCode*  •  Meta PyTorch OpenEnv Hackathon

---

## 🌍 Environment Overview

**DigiKul-v0** simulates a real-world resource-allocation challenge: a central educational server has a **strictly limited bandwidth budget** (e.g., 30 Mbps) that must be distributed across multiple **remote rural college classrooms** ("nodes"). Each classroom experiences dynamic conditions — students log in and out, and weather events degrade local internet reception in real time.

The AI agent acts as the **central network orchestrator**. At every timestep (representing one minute of class time), it evaluates the entire network and decides what quality of educational stream to push to each classroom:

| Quality Level | Stream Type | Bandwidth |
|:---:|:---|:---:|
| 3 | 🟢 HD Video | 5.0 Mbps |
| 2 | 🟡 Standard Video | 2.0 Mbps |
| 1 | 🟠 Audio + Text | 0.5 Mbps |
| 0 | 🔴 Disconnect | 0.0 Mbps |

### Why This Problem Matters

2.7 billion people worldwide still lack reliable internet access. In remote and rural areas, bandwidth is a scarce resource that must be shared across classrooms, health clinics, and community centres. DigiKul models this exact constraint — forcing AI agents to learn the nuanced tradeoff between quality, fairness, and reliability under uncertainty.

---

## 🎯 Task Descriptions

DigiKul provides **three difficulty levels** with increasing complexity:

| Task | Nodes | Budget | Steps | Difficulty |
|:---:|:---:|:---:|:---:|:---|
| `easy` | 3 classrooms | 40 Mbps | 30 | Generous budget, simple decisions |
| `medium` | 5 classrooms | 30 Mbps | 45 | Balanced challenge, scarcity begins |
| `hard` | 8 classrooms | 25 Mbps | 60 | Severe scarcity, complex tradeoffs |

Each task features **heterogeneous nodes** — different base capacities simulating real-world infrastructure inequality (e.g., a well-funded school vs. a remote village with a satellite dish).

---

## 📐 Action & Observation Spaces

### Action Space

```python
DigiKulAction(quality_levels: List[int])
```

A list of N integers (one per node), each in `{0, 1, 2, 3}`. The agent decides the stream quality for every classroom simultaneously.

### Observation Space

```python
DigiKulObservation(
    nodes: List[NodeObservation],    # Per-node state
    server_bandwidth: float,          # Total budget (Mbps)
    server_load: float,               # Current utilization
    remaining_bandwidth: float,       # Budget remaining
    time_step: int,                   # Current step
    max_time_steps: int,              # Episode length
    reward: float,                    # Step reward
    done: bool,                       # Episode ended?
    info: dict,                       # Diagnostics
)
```

Each `NodeObservation` contains:

| Field | Description |
|:---|:---|
| `num_students` | Currently logged-in students |
| `max_students` | Classroom capacity |
| `weather_factor` | Weather quality (0.1 = storm, 1.0 = clear) |
| `local_capacity` | Effective bandwidth = base × weather (Mbps) |
| `base_capacity` | Hardware maximum bandwidth (Mbps) |
| `prev_allocation` | Last step's bandwidth allocation |
| `prev_quality` | Last step's quality level |

All state variables are fully observable — there are **no hidden variables**, ensuring MDP (Markov Decision Process) integrity.

---

## 🧮 Mathematical Foundations

### Dense Reward Function

$$R_t = R_{\text{utility}} - \lambda_1 P_{\text{overload}} - \lambda_2 P_{\text{fairness}} - \lambda_3 P_{\text{budget}}$$

- **Utility** (concave log): $U_i = s_i \cdot \log(1 + b_i^{\text{eff}} / (s_i + \varepsilon))$ — inherent diminishing returns
- **Overload Penalty** (quadratic): $P_{\text{overload}} = \sum \max(0, b_i - c_i)^2$ — catastrophic for overallocation
- **Fairness Penalty** (MSE): Mean Squared Error of per-student bandwidth — prevents starvation
- **Budget Penalty** (quadratic): $P_{\text{budget}} = \max(0, \sum b_i - B)^2$ — server crash prevention

### Programmatic Grader (0.0 → 1.0)

$$G = \frac{\sum_t \sum_i U_i(t)}{\sum_t U^*_{\text{greedy}}(t)}$$

Where $U^*_{\text{greedy}}$ is a greedy approximation of the theoretical maximum utility (prioritising high-capacity nodes). The final ratio is clamped to [0.0, 1.0] to ensure strict bounds.

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Digikul.V0.git
cd Digikul.V0

# Install dependencies
pip install -e ".[dev]"
```

### Run the Server Locally

```bash
# Default (medium difficulty)
python server/app.py

# Or with specific task
DIGIKUL_TASK=hard python server/app.py

# Server starts on http://localhost:7860
```

### Run Tests

```bash
pytest tests/test_digikul.py -v
```

### Run Baseline Inference

```bash
# Set your HuggingFace token
export HF_TOKEN="hf_your_token_here"

# Run the LLM-based baseline
python inference.py
```

### Docker Deployment

```bash
# Build the container
docker build -t digikul-v0 -f server/Dockerfile .

# Run it
docker run -p 7860:7860 -e DIGIKUL_TASK=medium digikul-v0

# Test the health endpoint
curl http://localhost:7860/health
```

### Interact with the Environment

```python
from models import DigiKulAction
from server.digikul_environment import DigiKulEnvironment

# Create environment
env = DigiKulEnvironment(task="medium", seed=42)

# Reset
obs = env.reset()
print(f"Nodes: {len(obs.nodes)}, Budget: {obs.server_bandwidth} Mbps")

# Step with an action
action = DigiKulAction(quality_levels=[3, 2, 1, 0, 2])
obs = env.step(action)
print(f"Reward: {obs.reward:.4f}, Done: {obs.done}")

# Check grader at episode end
state = env.state
if state.grader_score is not None:
    print(f"Grader: {state.grader_score:.4f}")
```

---

## 📊 Baseline Performance Scores

| Task | Grader Score | Strategy |
|:---:|:---:|:---|
| Easy | ~0.45–0.65 | LLM (Mistral-7B-Instruct) |
| Medium | ~0.35–0.55 | LLM (Mistral-7B-Instruct) |
| Hard | ~0.20–0.40 | LLM (Mistral-7B-Instruct) |

*Scores are approximate and depend on the specific LLM and stochastic rollout. A well-trained RL agent should significantly outperform these baselines.*

---

## 📁 Project Structure

```
Digikul.V0/
├── openenv.yaml            # OpenEnv environment manifest
├── pyproject.toml           # Package configuration & dependencies
├── README.md                # This file
├── __init__.py              # Package exports
├── models.py                # Pydantic models (Action, Observation, State)
├── client.py                # HTTP client for remote interaction
├── inference.py             # LLM-based baseline inference script
├── .dockerignore            # Docker build exclusions
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI server application
│   ├── digikul_environment.py  # Core environment logic
│   ├── requirements.txt     # Server dependencies
│   └── Dockerfile           # Container image definition
└── tests/
    ├── __init__.py
    └── test_digikul.py      # Comprehensive test suite
```

---

## 🏗 OpenEnv Compliance

| Requirement | Status |
|:---|:---:|
| Pydantic typed models (Action, Observation, State) | ✅ |
| `step(action)` → (observation, reward, done, info) | ✅ |
| `reset()` → initial observation | ✅ |
| `state()` → current state | ✅ |
| `openenv.yaml` manifest | ✅ |
| 3+ tasks with programmatic graders (0.0–1.0) | ✅ |
| Dense reward function | ✅ |
| Baseline inference script (OpenAI API + HF_TOKEN) | ✅ |
| Dockerfile for containerised execution | ✅ |
| Deployable to Hugging Face Spaces | ✅ |

---

## 📜 License

BSD 3-Clause License
