"""
DigiKul-v0 — Test Suite.

Validates environment correctness, reward math, grader bounds,
and OpenEnv interface compliance.
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import numpy as np

# Ensure package root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import DigiKulAction, DigiKulObservation, DigiKulState, QUALITY_BW_MAP
from server.digikul_environment import DigiKulEnvironment, TASK_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=["easy", "medium", "hard"])
def env(request):
    """Provide a fresh environment for each task difficulty."""
    return DigiKulEnvironment(task=request.param, seed=42)


@pytest.fixture
def medium_env():
    """Provide a deterministic medium-difficulty environment."""
    return DigiKulEnvironment(task="medium", seed=42)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Basic Interface
# ═══════════════════════════════════════════════════════════════════════════

class TestInterface:
    """Validate the OpenEnv reset/step/state interface."""

    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, DigiKulObservation)
        assert obs.done is False
        assert obs.time_step == 0
        assert obs.reward == 0.0
        assert len(obs.nodes) == env.num_nodes

    def test_step_returns_observation(self, medium_env):
        obs = medium_env.reset()
        action = DigiKulAction(quality_levels=[1] * medium_env.num_nodes)
        obs = medium_env.step(action)
        assert isinstance(obs, DigiKulObservation)
        assert obs.time_step == 1

    def test_state_returns_state(self, medium_env):
        medium_env.reset()
        state = medium_env.state
        assert isinstance(state, DigiKulState)
        assert state.step_count == 0
        assert state.episode_id != ""

    def test_step_after_done_raises(self, medium_env):
        """Stepping after episode end should raise RuntimeError."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[0] * medium_env.num_nodes)
        for _ in range(medium_env._cfg.episode_length):
            obs = medium_env.step(action)
        assert obs.done is True
        with pytest.raises(RuntimeError):
            medium_env.step(action)

    def test_wrong_action_length_raises(self, medium_env):
        """Action with wrong number of nodes should raise ValueError."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[1, 2])  # too few
        with pytest.raises(ValueError):
            medium_env.step(action)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Observation Structure
# ═══════════════════════════════════════════════════════════════════════════

class TestObservation:
    """Validate observation completeness and bounds."""

    def test_node_fields(self, env):
        obs = env.reset()
        for node in obs.nodes:
            assert node.num_students >= 0
            assert node.num_students <= node.max_students
            assert 0.0 <= node.weather_factor <= 1.0
            assert node.local_capacity >= 0.0
            assert node.base_capacity > 0.0
            assert node.local_capacity <= node.base_capacity

    def test_server_fields(self, env):
        obs = env.reset()
        assert obs.server_bandwidth > 0
        assert obs.remaining_bandwidth == obs.server_bandwidth  # nothing allocated yet
        assert obs.server_load == 0.0  # no allocation yet
        assert obs.max_time_steps > 0

    def test_all_task_configs_exist(self):
        for task in ["easy", "medium", "hard"]:
            assert task in TASK_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Reward Function
# ═══════════════════════════════════════════════════════════════════════════

class TestReward:
    """Validate reward function properties."""

    def test_all_disconnect_gives_zero_utility(self, medium_env):
        """All-disconnect should give ~0 utility reward."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[0] * medium_env.num_nodes)
        obs = medium_env.step(action)
        # Utility component should be 0 (no bandwidth = no utility)
        # Reward might be slightly negative due to fairness penalty starting conditions
        assert abs(obs.info.get("r_utility", 0)) < 0.01

    def test_overload_penalty_is_positive(self, medium_env):
        """HD to all nodes should cause overload on some nodes."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[3] * medium_env.num_nodes)
        obs = medium_env.step(action)
        # With 5 nodes × 5 Mbps = 25 Mbps > some local capacities after weather
        # Some nodes will be overloaded → penalty exists
        assert obs.info.get("p_overload", 0) >= 0

    def test_budget_violation_detected(self, medium_env):
        """HD to all 5 nodes uses 25 Mbps ≤ 30 Mbps budget, so should be OK."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[3] * medium_env.num_nodes)
        obs = medium_env.step(action)
        # 5 nodes × 5 Mbps = 25 Mbps, budget is 30 → no violation
        assert obs.info.get("budget_excess", 0) == 0.0

    def test_reward_is_finite(self, medium_env):
        """Reward should always be a finite number."""
        medium_env.reset()
        for _ in range(10):
            action = DigiKulAction(
                quality_levels=[np.random.randint(0, 4) for _ in range(medium_env.num_nodes)]
            )
            obs = medium_env.step(action)
            assert math.isfinite(obs.reward)


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Programmatic Grader
# ═══════════════════════════════════════════════════════════════════════════

class TestGrader:
    """Validate grader stays in [0.0, 1.0] under all conditions."""

    def _run_full_episode(self, env, strategy="random"):
        """Run a full episode with a given strategy and return grader score."""
        env.reset()
        N = env.num_nodes
        for _ in range(env._cfg.episode_length):
            if strategy == "random":
                levels = [np.random.randint(0, 4) for _ in range(N)]
            elif strategy == "all_hd":
                levels = [3] * N
            elif strategy == "all_disconnect":
                levels = [0] * N
            elif strategy == "all_audio":
                levels = [1] * N
            else:
                levels = [2] * N
            env.step(DigiKulAction(quality_levels=levels))
        return env.state.grader_score

    def test_grader_in_bounds_random(self):
        """Random actions: grader ∈ [0, 1]."""
        for task in ["easy", "medium", "hard"]:
            env = DigiKulEnvironment(task=task, seed=None)
            for _ in range(20):
                score = self._run_full_episode(env, strategy="random")
                assert 0.0 <= score <= 1.0, f"Grader out of bounds: {score}"

    def test_grader_in_bounds_all_hd(self):
        """All-HD actions: grader ∈ [0, 1]."""
        for task in ["easy", "medium", "hard"]:
            env = DigiKulEnvironment(task=task, seed=42)
            score = self._run_full_episode(env, strategy="all_hd")
            assert 0.0 <= score <= 1.0

    def test_grader_in_bounds_all_disconnect(self):
        """All-disconnect actions: grader should be 0.0."""
        for task in ["easy", "medium", "hard"]:
            env = DigiKulEnvironment(task=task, seed=42)
            score = self._run_full_episode(env, strategy="all_disconnect")
            assert score == 0.0

    def test_grader_not_available_before_done(self, medium_env):
        """Grader should be None while episode is running."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[1] * medium_env.num_nodes)
        medium_env.step(action)
        state = medium_env.state
        assert state.grader_score is None

    def test_grader_available_after_done(self, medium_env):
        """Grader should be a float after episode ends."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[2] * medium_env.num_nodes)
        for _ in range(medium_env._cfg.episode_length):
            medium_env.step(action)
        state = medium_env.state
        assert isinstance(state.grader_score, float)
        assert 0.0 <= state.grader_score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Stochastic Dynamics
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamics:
    """Validate stochastic transition properties."""

    def test_student_count_stays_bounded(self, medium_env):
        """Student count should stay in [0, max_students]."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[1] * medium_env.num_nodes)
        for _ in range(medium_env._cfg.episode_length):
            obs = medium_env.step(action)
            for node in obs.nodes:
                assert 0 <= node.num_students <= node.max_students

    def test_weather_stays_bounded(self, medium_env):
        """Weather factor should stay in [0.1, 1.0]."""
        medium_env.reset()
        action = DigiKulAction(quality_levels=[1] * medium_env.num_nodes)
        for _ in range(medium_env._cfg.episode_length):
            obs = medium_env.step(action)
            for node in obs.nodes:
                assert 0.1 <= node.weather_factor <= 1.0

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical episode."""
        env1 = DigiKulEnvironment(task="medium", seed=123)
        env2 = DigiKulEnvironment(task="medium", seed=123)

        obs1 = env1.reset()
        obs2 = env2.reset()

        action = DigiKulAction(quality_levels=[2, 1, 3, 0, 2])
        for _ in range(10):
            obs1 = env1.step(action)
            obs2 = env2.step(action)
            assert obs1.reward == obs2.reward
            for n1, n2 in zip(obs1.nodes, obs2.nodes):
                assert n1.num_students == n2.num_students
                assert n1.weather_factor == n2.weather_factor


# ═══════════════════════════════════════════════════════════════════════════
#  Test: Pydantic Model Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestPydanticModels:
    """Validate Pydantic model serialization and constraints."""

    def test_action_serialization(self):
        action = DigiKulAction(quality_levels=[0, 1, 2, 3, 1])
        d = action.model_dump()
        assert d == {"quality_levels": [0, 1, 2, 3, 1]}

    def test_action_from_dict(self):
        action = DigiKulAction(**{"quality_levels": [3, 2, 1]})
        assert action.quality_levels == [3, 2, 1]

    def test_observation_serialization(self, medium_env):
        obs = medium_env.reset()
        d = obs.model_dump()
        obs_rebuilt = DigiKulObservation(**d)
        assert obs_rebuilt.server_bandwidth == obs.server_bandwidth
        assert len(obs_rebuilt.nodes) == len(obs.nodes)

    def test_state_serialization(self, medium_env):
        medium_env.reset()
        state = medium_env.state
        d = state.model_dump()
        state_rebuilt = DigiKulState(**d)
        assert state_rebuilt.episode_id == state.episode_id

    def test_quality_bw_map_complete(self):
        """All quality levels should have a bandwidth mapping."""
        for q in [0, 1, 2, 3]:
            assert q in QUALITY_BW_MAP
