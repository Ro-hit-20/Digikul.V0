"""
DigiKul-v0 — Pydantic Models for Action, Observation, and State.

All models use Pydantic BaseModel as required by the Meta OpenEnv
Hackathon Problem Statement. These are the typed schemas exchanged
between the agent and the environment via the OpenEnv HTTP interface.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Bandwidth quality-level constants (Mbps consumed per student at each tier)
# ---------------------------------------------------------------------------
QUALITY_DISCONNECT = 0       # No stream
QUALITY_AUDIO_TEXT = 1       # Audio + text sync
QUALITY_STANDARD_VIDEO = 2   # Standard-definition video
QUALITY_HD_VIDEO = 3         # High-definition video

QUALITY_BW_MAP: dict[int, float] = {
    QUALITY_DISCONNECT: 0.0,
    QUALITY_AUDIO_TEXT: 0.5,
    QUALITY_STANDARD_VIDEO: 2.0,
    QUALITY_HD_VIDEO: 5.0,
}

QUALITY_LABELS: dict[int, str] = {
    QUALITY_DISCONNECT: "Disconnect",
    QUALITY_AUDIO_TEXT: "Audio+Text",
    QUALITY_STANDARD_VIDEO: "SD Video",
    QUALITY_HD_VIDEO: "HD Video",
}


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class DigiKulAction(BaseModel):
    """
    The agent's decision: a quality level for each node.

    Each entry is an integer in {0, 1, 2, 3}:
        0 = Disconnect          (0.0 Mbps)
        1 = Audio + Text        (0.5 Mbps)
        2 = Standard Video      (2.0 Mbps)
        3 = HD Video            (5.0 Mbps)

    len(quality_levels) MUST equal the number of nodes in the
    current task configuration.
    """
    quality_levels: List[int] = Field(
        ...,
        description=(
            "Quality level for each node. "
            "0=Disconnect, 1=Audio+Text, 2=SD Video, 3=HD Video"
        ),
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class NodeObservation(BaseModel):
    """Per-node observable state."""
    node_id: int = Field(..., description="Index of this node (0-based).")
    num_students: int = Field(
        ..., ge=0,
        description="Current number of students logged in.",
    )
    max_students: int = Field(
        ..., gt=0,
        description="Maximum classroom capacity.",
    )
    weather_factor: float = Field(
        ..., ge=0.0, le=1.0,
        description="Weather quality factor (1.0 = clear, 0.1 = severe storm).",
    )
    local_capacity: float = Field(
        ..., ge=0.0,
        description="Effective local bandwidth = base_capacity × weather_factor (Mbps).",
    )
    base_capacity: float = Field(
        ..., gt=0.0,
        description="Hardware maximum bandwidth for this node (Mbps).",
    )
    prev_allocation: float = Field(
        0.0, ge=0.0,
        description="Bandwidth allocated to this node at the previous step (Mbps).",
    )
    prev_quality: int = Field(
        0, ge=0, le=3,
        description="Quality level assigned at the previous step.",
    )


class DigiKulObservation(BaseModel):
    """
    Full observation returned by reset() and step().

    Contains per-node states plus global server metrics, the step
    reward, and a done flag. All information needed to satisfy the
    Markov property is included — there are no hidden variables.
    """
    nodes: List[NodeObservation] = Field(
        ..., description="Per-node observable state.",
    )
    server_bandwidth: float = Field(
        ..., gt=0.0,
        description="Total server bandwidth budget (Mbps).",
    )
    server_load: float = Field(
        0.0, ge=0.0,
        description="Current total bandwidth utilised (Mbps).",
    )
    remaining_bandwidth: float = Field(
        ..., ge=0.0,
        description="Remaining available bandwidth (Mbps).",
    )
    time_step: int = Field(
        0, ge=0,
        description="Current timestep in the episode (0-indexed).",
    )
    max_time_steps: int = Field(
        ..., gt=0,
        description="Total timesteps in the episode.",
    )
    reward: float = Field(
        0.0,
        description="Dense reward for the current step.",
    )
    done: bool = Field(
        False,
        description="Whether the episode has ended.",
    )
    info: dict = Field(
        default_factory=dict,
        description="Auxiliary diagnostics (grader score at episode end, etc.).",
    )


# ---------------------------------------------------------------------------
# State (episode-level metadata for the OpenEnv state() endpoint)
# ---------------------------------------------------------------------------
class DigiKulState(BaseModel):
    """
    Internal episode state exposed via the OpenEnv state() endpoint.

    Tracks accumulated metrics required for the programmatic grader.
    """
    episode_id: str = Field(
        "",
        description="Unique identifier for this episode.",
    )
    task: str = Field(
        "medium",
        description="Difficulty level: easy / medium / hard.",
    )
    step_count: int = Field(
        0, ge=0,
        description="Number of steps taken so far.",
    )
    cumulative_utility: float = Field(
        0.0,
        description="Sum of actual utility across all steps and nodes.",
    )
    cumulative_max_utility: float = Field(
        0.0,
        description="Sum of theoretical-max utility (oracle) across all steps and nodes.",
    )
    grader_score: Optional[float] = Field(
        None,
        description=(
            "Final programmatic grade in [0.0, 1.0]. "
            "None until the episode ends."
        ),
    )
    total_reward: float = Field(
        0.0,
        description="Sum of dense rewards across the episode.",
    )
    total_overload_events: int = Field(
        0, ge=0,
        description="Count of timesteps where any node was overloaded.",
    )
    total_budget_violations: int = Field(
        0, ge=0,
        description="Count of timesteps where server budget was exceeded.",
    )
