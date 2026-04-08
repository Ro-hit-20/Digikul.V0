"""
DigiKul-v0 — Core Environment Implementation.

A production-grade RL environment for the Meta OpenEnv Hackathon.
Simulates bandwidth allocation across remote rural classrooms with
stochastic student churn and weather dynamics.

Mathematical foundations:
  • Dense reward: concave log-utility with quadratic overload,
    variance-based fairness, and budget-violation penalties.
  • Programmatic grader: ratio of actual to oracle utility ∈ [0,1].
  • Transition: Gaussian random-walk churn + Ornstein–Uhlenbeck weather.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# We import models from the package root for clean resolution.
# When running inside Docker the package will be installed, so
# both relative and absolute imports work.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    DigiKulAction,
    DigiKulObservation,
    DigiKulState,
    NodeObservation,
    QUALITY_BW_MAP,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Task configurations  (Easy → Medium → Hard)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NodeConfig:
    """Static configuration for one classroom node."""
    base_capacity: float      # hardware-max bandwidth (Mbps)
    max_students: int         # classroom seat capacity
    init_students: int        # students present at episode start
    init_weather: float       # weather factor at start


@dataclass(frozen=True)
class TaskConfig:
    """Immutable task-level parameters."""
    name: str
    server_bandwidth: float           # total Mbps budget
    episode_length: int               # timesteps (minutes of class)
    nodes: Tuple[NodeConfig, ...]     # per-node static configs
    # Stochastic dynamics parameters
    churn_mean: float = 0.0           # μ for student Δ
    churn_std: float = 2.0            # σ for student Δ
    weather_theta: float = 0.15       # OU mean-reversion speed
    weather_mu: float = 0.75          # OU long-run mean
    weather_sigma: float = 0.08       # OU volatility
    # Reward shaping weights
    lambda_overload: float = 2.0
    lambda_fairness: float = 0.5
    lambda_budget: float = 5.0


TASK_REGISTRY: Dict[str, TaskConfig] = {
    # ------------------------------------------------------------------
    # EASY: 3 heterogeneous nodes, generous 40 Mbps budget, 30 steps
    # ------------------------------------------------------------------
    "easy": TaskConfig(
        name="easy",
        server_bandwidth=40.0,
        episode_length=30,
        nodes=(
            NodeConfig(base_capacity=15.0, max_students=40, init_students=20, init_weather=0.9),
            NodeConfig(base_capacity=12.0, max_students=35, init_students=18, init_weather=0.85),
            NodeConfig(base_capacity=10.0, max_students=30, init_students=15, init_weather=0.8),
        ),
        churn_std=1.5,
        weather_sigma=0.05,
        lambda_overload=1.5,
        lambda_fairness=0.3,
        lambda_budget=4.0,
    ),
    # ------------------------------------------------------------------
    # MEDIUM: 5 heterogeneous nodes, strict 30 Mbps budget, 45 steps
    # ------------------------------------------------------------------
    "medium": TaskConfig(
        name="medium",
        server_bandwidth=30.0,
        episode_length=45,
        nodes=(
            NodeConfig(base_capacity=12.0, max_students=50, init_students=30, init_weather=0.85),
            NodeConfig(base_capacity=10.0, max_students=45, init_students=25, init_weather=0.9),
            NodeConfig(base_capacity=8.0,  max_students=40, init_students=28, init_weather=0.7),
            NodeConfig(base_capacity=6.0,  max_students=35, init_students=20, init_weather=0.8),
            NodeConfig(base_capacity=5.0,  max_students=30, init_students=15, init_weather=0.75),
        ),
        churn_std=2.0,
        weather_sigma=0.08,
        lambda_overload=2.0,
        lambda_fairness=0.5,
        lambda_budget=5.0,
    ),
    # ------------------------------------------------------------------
    # HARD: 8 heterogeneous nodes, severe 25 Mbps budget, 60 steps
    # ------------------------------------------------------------------
    "hard": TaskConfig(
        name="hard",
        server_bandwidth=25.0,
        episode_length=60,
        nodes=(
            NodeConfig(base_capacity=10.0, max_students=60, init_students=40, init_weather=0.8),
            NodeConfig(base_capacity=8.0,  max_students=55, init_students=35, init_weather=0.75),
            NodeConfig(base_capacity=7.0,  max_students=50, init_students=38, init_weather=0.6),
            NodeConfig(base_capacity=6.0,  max_students=45, init_students=30, init_weather=0.85),
            NodeConfig(base_capacity=5.0,  max_students=40, init_students=25, init_weather=0.7),
            NodeConfig(base_capacity=4.0,  max_students=35, init_students=28, init_weather=0.65),
            NodeConfig(base_capacity=3.5,  max_students=30, init_students=20, init_weather=0.9),
            NodeConfig(base_capacity=3.0,  max_students=25, init_students=18, init_weather=0.55),
        ),
        churn_std=2.5,
        weather_sigma=0.10,
        lambda_overload=3.0,
        lambda_fairness=0.8,
        lambda_budget=6.0,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════════════════════════════

class DigiKulEnvironment:
    """
    OpenEnv-compatible RL environment for bandwidth allocation.

    Implements the three core methods required by OpenEnv:
        • reset()        → DigiKulObservation
        • step(action)   → DigiKulObservation
        • state (property) → DigiKulState
    """

    EPS = 1e-8  # numerical stability constant

    def __init__(self, task: str = "medium", seed: Optional[int] = None):
        if task not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_REGISTRY.keys())}"
            )
        self._task_name = task
        self._cfg = TASK_REGISTRY[task]
        self._rng = np.random.default_rng(seed)

        # Mutable per-episode state (initialised in reset)
        self._num_students: np.ndarray = np.array([])
        self._weather: np.ndarray = np.array([])
        self._prev_alloc: np.ndarray = np.array([])
        self._prev_quality: np.ndarray = np.array([])
        self._step_idx: int = 0
        self._cumulative_utility: float = 0.0
        self._cumulative_max_utility: float = 0.0
        self._total_reward: float = 0.0
        self._total_overload_events: int = 0
        self._total_budget_violations: int = 0
        self._episode_id: str = ""
        self._done: bool = True

    # ── helpers ──────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return len(self._cfg.nodes)

    def _local_capacity(self, node_idx: int) -> float:
        """Effective local bandwidth = base × weather."""
        return self._cfg.nodes[node_idx].base_capacity * float(self._weather[node_idx])

    def _compute_effective_bw(
        self, alloc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute effective bandwidth per node (capped by local capacity)
        and the overload amounts.
        """
        local_caps = np.array([self._local_capacity(i) for i in range(self.num_nodes)])
        effective = np.minimum(alloc, local_caps)
        overloads = np.maximum(alloc - local_caps, 0.0)
        return effective, overloads

    # ── utility functions ────────────────────────────────────────────────

    def _node_utility(self, eff_bw: float, students: int) -> float:
        """
        U_i = s_i · log(1 + eff_bw / (s_i + ε))
        Concave, sublinear — inherent diminishing returns.
        """
        s = float(students)
        return s * math.log(1.0 + eff_bw / (s + self.EPS))

    def _oracle_utility(self, node_idx: int) -> float:
        """
        Theoretical-maximum utility for the grader's denominator.
        Give the node min(local_capacity, server_budget) — the physical max.
        """
        cap = self._local_capacity(node_idx)
        bw = min(cap, self._cfg.server_bandwidth)
        return self._node_utility(bw, int(self._num_students[node_idx]))

    # ── reward function ──────────────────────────────────────────────────

    def _compute_reward(
        self,
        effective_bw: np.ndarray,
        overloads: np.ndarray,
        total_alloc: float,
    ) -> Tuple[float, dict]:
        """
        R_t = R_utility − λ₁·P_overload − λ₂·P_fairness − λ₃·P_budget

        Returns (reward, info_dict).
        """
        cfg = self._cfg
        N = self.num_nodes

        # ── Utility ──
        utilities = np.array([
            self._node_utility(float(effective_bw[i]), int(self._num_students[i]))
            for i in range(N)
        ])
        r_utility = float(np.mean(utilities))

        # ── Overload penalty (quadratic) ──
        p_overload = float(np.sum(overloads ** 2))

        # ── Fairness penalty (variance of per-student bandwidth) ──
        active_mask = self._num_students > 0
        if np.any(active_mask):
            per_student_bw = effective_bw[active_mask] / (
                self._num_students[active_mask].astype(float) + self.EPS
            )
            mean_psb = float(np.mean(per_student_bw))
            p_fairness = float(np.mean((per_student_bw - mean_psb) ** 2))
        else:
            p_fairness = 0.0

        # ── Budget violation penalty (quadratic) ──
        budget_excess = max(0.0, total_alloc - cfg.server_bandwidth)
        p_budget = budget_excess ** 2

        # ── Combine ──
        reward = (
            r_utility
            - cfg.lambda_overload * p_overload
            - cfg.lambda_fairness * p_fairness
            - cfg.lambda_budget * p_budget
        )

        info = {
            "r_utility": r_utility,
            "p_overload": p_overload,
            "p_fairness": p_fairness,
            "p_budget": p_budget,
            "budget_excess": budget_excess,
            "total_alloc": total_alloc,
            "utilities": utilities.tolist(),
        }
        return reward, info

    # ── stochastic transitions ───────────────────────────────────────────

    def _transition(self) -> None:
        """
        Advance stochastic state to the next timestep.
        • Students: bounded Gaussian random walk
        • Weather: Ornstein–Uhlenbeck mean-reverting process
        """
        cfg = self._cfg

        # Student churn
        delta_s = self._rng.normal(cfg.churn_mean, cfg.churn_std, size=self.num_nodes)
        delta_s = np.round(delta_s).astype(int)
        self._num_students = np.clip(
            self._num_students + delta_s,
            0,
            np.array([n.max_students for n in cfg.nodes]),
        )

        # Weather (OU process)
        xi = self._rng.normal(0.0, 1.0, size=self.num_nodes)
        self._weather = np.clip(
            self._weather
            + cfg.weather_theta * (cfg.weather_mu - self._weather)
            + cfg.weather_sigma * xi,
            0.1,
            1.0,
        )

    # ── build observation ────────────────────────────────────────────────

    def _build_observation(self, reward: float, done: bool, info: dict) -> DigiKulObservation:
        nodes = []
        for i in range(self.num_nodes):
            nodes.append(NodeObservation(
                node_id=i,
                num_students=int(self._num_students[i]),
                max_students=self._cfg.nodes[i].max_students,
                weather_factor=round(float(self._weather[i]), 4),
                local_capacity=round(self._local_capacity(i), 4),
                base_capacity=self._cfg.nodes[i].base_capacity,
                prev_allocation=round(float(self._prev_alloc[i]), 4),
                prev_quality=int(self._prev_quality[i]),
            ))
        total_alloc = float(np.sum(self._prev_alloc))
        return DigiKulObservation(
            nodes=nodes,
            server_bandwidth=self._cfg.server_bandwidth,
            server_load=round(total_alloc, 4),
            remaining_bandwidth=round(max(0.0, self._cfg.server_bandwidth - total_alloc), 4),
            time_step=self._step_idx,
            max_time_steps=self._cfg.episode_length,
            reward=round(reward, 6),
            done=done,
            info=info,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  OpenEnv Interface
    # ══════════════════════════════════════════════════════════════════════

    def reset(self) -> DigiKulObservation:
        """Initialise a new episode and return the starting observation."""
        self._episode_id = str(uuid.uuid4())
        self._step_idx = 0
        self._done = False

        # Initialise per-node arrays from config
        self._num_students = np.array(
            [n.init_students for n in self._cfg.nodes], dtype=np.int64
        )
        self._weather = np.array(
            [n.init_weather for n in self._cfg.nodes], dtype=np.float64
        )
        self._prev_alloc = np.zeros(self.num_nodes, dtype=np.float64)
        self._prev_quality = np.zeros(self.num_nodes, dtype=np.int64)

        # Reset grading accumulators
        self._cumulative_utility = 0.0
        self._cumulative_max_utility = 0.0
        self._total_reward = 0.0
        self._total_overload_events = 0
        self._total_budget_violations = 0

        return self._build_observation(reward=0.0, done=False, info={})

    def step(self, action: DigiKulAction) -> DigiKulObservation:
        """Execute one timestep given the agent's quality-level decisions."""
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new one.")

        N = self.num_nodes
        levels = action.quality_levels

        # Validate / clip action length
        if len(levels) != N:
            raise ValueError(
                f"Expected {N} quality levels, got {len(levels)}."
            )

        # Clip quality levels to valid range [0, 3]
        levels = [max(0, min(3, int(q))) for q in levels]

        # ── Compute bandwidth allocation (Mbps) ──
        alloc = np.array([QUALITY_BW_MAP[q] for q in levels], dtype=np.float64)

        # ── Effective bandwidth (capped by local capacity) ──
        effective_bw, overloads = self._compute_effective_bw(alloc)
        total_alloc = float(np.sum(alloc))

        # ── Reward ──
        reward, reward_info = self._compute_reward(effective_bw, overloads, total_alloc)

        # ── Accumulate grader metrics ──
        step_utility = sum(
            self._node_utility(float(effective_bw[i]), int(self._num_students[i]))
            for i in range(N)
        )
        step_max_utility = sum(self._oracle_utility(i) for i in range(N))

        self._cumulative_utility += step_utility
        self._cumulative_max_utility += step_max_utility
        self._total_reward += reward

        if float(np.sum(overloads)) > 0:
            self._total_overload_events += 1
        if total_alloc > self._cfg.server_bandwidth:
            self._total_budget_violations += 1

        # ── Store for next observation ──
        self._prev_alloc = alloc.copy()
        self._prev_quality = np.array(levels, dtype=np.int64)

        # ── Advance step counter ──
        self._step_idx += 1
        self._done = self._step_idx >= self._cfg.episode_length

        # ── Compute grader at episode end ──
        info: Dict[str, Any] = reward_info
        if self._done:
            grader = self._compute_grader()
            info["grader_score"] = grader
            info["episode_summary"] = {
                "total_reward": self._total_reward,
                "overload_events": self._total_overload_events,
                "budget_violations": self._total_budget_violations,
            }

        # ── Transition stochastic state for the *next* step ──
        if not self._done:
            self._transition()

        return self._build_observation(reward=reward, done=self._done, info=info)

    @property
    def state(self) -> DigiKulState:
        """Return episode-level metadata (OpenEnv state endpoint)."""
        return DigiKulState(
            episode_id=self._episode_id,
            task=self._task_name,
            step_count=self._step_idx,
            cumulative_utility=round(self._cumulative_utility, 6),
            cumulative_max_utility=round(self._cumulative_max_utility, 6),
            grader_score=self._compute_grader() if self._done else None,
            total_reward=round(self._total_reward, 6),
            total_overload_events=self._total_overload_events,
            total_budget_violations=self._total_budget_violations,
        )

    # ── grader ───────────────────────────────────────────────────────────

    def _compute_grader(self) -> float:
        """
        Programmatic grader: G = Σ U_actual / Σ U_oracle  ∈ [0.0, 1.0].

        Edge case: if oracle utility is 0 (no students ever appeared),
        return 1.0 by convention (nothing to optimise).
        """
        if self._cumulative_max_utility < self.EPS:
            return 1.0
        return min(1.0, max(0.0,
            self._cumulative_utility / self._cumulative_max_utility
        ))
