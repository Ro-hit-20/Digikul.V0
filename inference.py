"""
DigiKul-v0 — Baseline Inference Script (LLM-based).

This script evaluates a language model within the DigiKul-v0 environment
using the OpenAI Python client, as required by the Meta OpenEnv Hackathon.

Authentication:
    Reads HF_TOKEN from environment variables and uses it to authenticate
    with a Hugging Face Inference Endpoint (OpenAI-compatible API).

Workflow:
    1. Connect to the DigiKul environment (local server)
    2. For each difficulty task (easy, medium, hard):
       a. Reset the environment
       b. At each timestep, format the observation into a text prompt
       c. Send the prompt to the LLM via the OpenAI API client
       d. Parse the LLM's text response into a DigiKulAction
       e. Step the environment with the parsed action
    3. Report the programmatic grader score for each task

Usage:
    export HF_TOKEN="hf_xxx..."
    python inference.py

    # Or with custom env URL:
    export DIGIKUL_ENV_URL="http://localhost:7860"
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import List, Optional

# ── Ensure package is importable ──
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from models import DigiKulAction, DigiKulObservation, QUALITY_BW_MAP, QUALITY_LABELS
from server.digikul_environment import DigiKulEnvironment, TASK_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("⚠️  WARNING: HF_TOKEN not set. LLM calls will fail.")
    print("   Set it with: export HF_TOKEN='hf_...'")

# HuggingFace Inference API (OpenAI-compatible endpoint)
# Uses a small, fast model for baseline — can be changed to any HF model
HF_MODEL = os.environ.get(
    "HF_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3"
)
HF_API_BASE = os.environ.get(
    "HF_API_BASE",
    "https://api-inference.huggingface.co/v1"
)

# Number of retries for LLM parsing failures
MAX_PARSE_RETRIES = 3


# ═══════════════════════════════════════════════════════════════════════════
#  Prompt Engineering
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an AI network orchestrator managing bandwidth for remote rural classrooms.

Your task: Given the current state of all classroom nodes, decide what quality of \
educational stream to push to each classroom.

Quality levels:
  0 = Disconnect     (0.0 Mbps per node)
  1 = Audio + Text   (0.5 Mbps per node)
  2 = SD Video       (2.0 Mbps per node)
  3 = HD Video       (5.0 Mbps per node)

Rules:
- Total bandwidth across ALL nodes must not exceed the server budget.
- Nodes with more students should generally get higher quality.
- Nodes with bad weather (low weather_factor) have reduced local capacity; \
  don't waste bandwidth on them if it would cause overload.
- Balance fairly: don't starve any node.

You MUST respond with ONLY a JSON object in this exact format:
{"quality_levels": [q0, q1, q2, ...]}

where each q is 0, 1, 2, or 3. The list length must equal the number of nodes.
Do NOT include any other text, explanation, or markdown formatting.\
"""


def format_observation_prompt(obs: DigiKulObservation) -> str:
    """Convert an observation into a human-readable prompt for the LLM."""
    lines = [
        f"=== TIMESTEP {obs.time_step}/{obs.max_time_steps} ===",
        f"Server Budget: {obs.server_bandwidth} Mbps",
        f"Currently Used: {obs.server_load} Mbps",
        f"Remaining: {obs.remaining_bandwidth} Mbps",
        f"Number of Nodes: {len(obs.nodes)}",
        "",
        "--- Node Status ---",
    ]

    for node in obs.nodes:
        lines.append(
            f"Node {node.node_id}: "
            f"students={node.num_students}/{node.max_students}, "
            f"weather={node.weather_factor:.2f}, "
            f"local_cap={node.local_capacity:.1f} Mbps, "
            f"base_cap={node.base_capacity:.1f} Mbps, "
            f"prev_quality={QUALITY_LABELS.get(node.prev_quality, '?')}"
        )

    lines.append("")
    lines.append(
        "Decide the quality level (0-3) for each node. "
        "Respond with ONLY the JSON: {\"quality_levels\": [...]}"
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Interaction
# ═══════════════════════════════════════════════════════════════════════════

def create_openai_client() -> OpenAI:
    """Create an OpenAI client pointed at the HuggingFace Inference API."""
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=HF_API_BASE,
    )


def query_llm(client: OpenAI, observation_text: str) -> str:
    """Send a prompt to the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_text},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def parse_llm_response(response_text: str, num_nodes: int) -> Optional[DigiKulAction]:
    """
    Parse the LLM's text response into a DigiKulAction.

    Handles common LLM quirks: markdown fences, extra text, etc.
    Returns None if parsing fails.
    """
    try:
        # Strip markdown code fences if present
        cleaned = response_text
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        # Try to extract JSON object
        match = re.search(r'\{[^}]+\}', cleaned)
        if match:
            cleaned = match.group()

        data = json.loads(cleaned)

        if "quality_levels" in data:
            levels = data["quality_levels"]
        elif isinstance(data, list):
            levels = data
        else:
            return None

        # Validate and clip
        if len(levels) != num_nodes:
            return None

        levels = [max(0, min(3, int(q))) for q in levels]
        return DigiKulAction(quality_levels=levels)

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def get_fallback_action(obs: DigiKulObservation) -> DigiKulAction:
    """
    Deterministic fallback if the LLM fails to produce a valid action.
    Uses a simple proportional heuristic based on student count and weather.
    """
    N = len(obs.nodes)
    budget = obs.server_bandwidth
    levels = []

    # Score each node by (students × weather)
    scores = []
    for node in obs.nodes:
        score = node.num_students * node.weather_factor
        scores.append(score)

    total_score = sum(scores) + 1e-8

    for i, node in enumerate(obs.nodes):
        if node.num_students == 0:
            levels.append(0)
            continue

        # Proportional share of budget
        share = (scores[i] / total_score) * budget
        local_cap = node.local_capacity

        # Pick highest quality that fits
        if share >= 5.0 and local_cap >= 5.0:
            levels.append(3)
        elif share >= 2.0 and local_cap >= 2.0:
            levels.append(2)
        elif share >= 0.5 and local_cap >= 0.5:
            levels.append(1)
        else:
            levels.append(0)

    # Verify budget constraint; downgrade greedily if over
    total_bw = sum(QUALITY_BW_MAP[q] for q in levels)
    while total_bw > budget:
        # Find the node with worst utility-per-mbps and downgrade it
        worst_idx = -1
        worst_ratio = float("inf")
        for i, q in enumerate(levels):
            if q > 0:
                s = obs.nodes[i].num_students
                ratio = s / (QUALITY_BW_MAP[q] + 1e-8)
                if ratio < worst_ratio:
                    worst_ratio = ratio
                    worst_idx = i
        if worst_idx == -1:
            break
        levels[worst_idx] -= 1
        total_bw = sum(QUALITY_BW_MAP[q] for q in levels)

    return DigiKulAction(quality_levels=levels)


# ═══════════════════════════════════════════════════════════════════════════
#  Main Inference Loop
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    env: DigiKulEnvironment,
    client: OpenAI,
    task_name: str,
    verbose: bool = True,
) -> dict:
    """Run one full episode, querying the LLM at each step."""
    obs = env.reset()
    total_reward = 0.0
    llm_calls = 0
    fallback_calls = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Task: {task_name.upper()} | Nodes: {len(obs.nodes)} | "
              f"Budget: {obs.server_bandwidth} Mbps | Steps: {obs.max_time_steps}")
        print(f"{'='*60}")

    print(f"[START] task={task_name}", flush=True)
    while not obs.done:
        # Format observation into text prompt
        prompt = format_observation_prompt(obs)

        # Query the LLM
        action = None
        for attempt in range(MAX_PARSE_RETRIES):
            try:
                llm_response = query_llm(client, prompt)
                llm_calls += 1
                action = parse_llm_response(llm_response, len(obs.nodes))
                if action is not None:
                    break
                if verbose and attempt == 0:
                    print(f"  ⚠ Step {obs.time_step}: LLM parse failed, retrying...")
            except Exception as e:
                llm_calls += 1
                if verbose:
                    print(f"  ⚠ Step {obs.time_step}: LLM error: {e}")
                break

        # Fallback to heuristic if LLM fails
        if action is None:
            action = get_fallback_action(obs)
            fallback_calls += 1
            if verbose:
                print(f"  ⚡ Step {obs.time_step}: Using fallback heuristic")

        # Step the environment
        obs = env.step(action)
        total_reward += obs.reward
        print(f"[STEP] step={obs.time_step} reward={obs.reward}", flush=True)

        if verbose and obs.time_step % 10 == 0:
            print(f"  Step {obs.time_step}/{obs.max_time_steps} | "
                  f"Reward: {obs.reward:.4f} | Load: {obs.server_load:.1f} Mbps")

    # Get final state with grader
    final_state = env.state
    grader = final_state.grader_score
    print(f"[END] task={task_name} score={grader} steps={obs.time_step}", flush=True)

    if verbose:
        print(f"\n  ── Episode Complete ──")
        print(f"  Grader Score:     {grader:.4f}")
        print(f"  Total Reward:     {total_reward:.4f}")
        print(f"  LLM Calls:        {llm_calls}")
        print(f"  Fallback Calls:   {fallback_calls}")
        print(f"  Overload Events:  {final_state.total_overload_events}")
        print(f"  Budget Violations:{final_state.total_budget_violations}")

    return {
        "task": task_name,
        "grader_score": grader,
        "total_reward": total_reward,
        "llm_calls": llm_calls,
        "fallback_calls": fallback_calls,
        "overload_events": final_state.total_overload_events,
        "budget_violations": final_state.total_budget_violations,
    }


def main():
    """Run baseline inference across all three difficulty levels."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       DigiKul-v0  •  Baseline LLM Inference            ║")
    print("║       Meta OpenEnv Hackathon                           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nModel:    {HF_MODEL}")
    print(f"API Base: {HF_API_BASE}")
    print(f"Token:    {'✓ Set' if HF_TOKEN else '✗ NOT SET'}")

    # Create OpenAI client
    llm_client = create_openai_client()

    results = []
    for task_name in ["easy", "medium", "hard"]:
        # Create environment for this task
        env = DigiKulEnvironment(task=task_name, seed=42)
        result = run_episode(env, llm_client, task_name, verbose=True)
        results.append(result)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Grader':>10} {'Reward':>10} {'LLM Calls':>10}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['task']:<10} {r['grader_score']:>10.4f} "
              f"{r['total_reward']:>10.4f} {r['llm_calls']:>10}")
    print(f"{'='*60}")

    avg_grader = sum(r["grader_score"] for r in results) / len(results)
    print(f"\n  Average Grader Score: {avg_grader:.4f}")


if __name__ == "__main__":
    main()
