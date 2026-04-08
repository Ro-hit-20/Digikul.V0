"""
DigiKul-v0 — Baseline Inference Script (LLM-based).

This script evaluates a language model within the DigiKul-v0 environment
using the OpenAI Python client, as required by the Meta OpenEnv Hackathon.

Authentication:
    Reads API_BASE_URL and API_KEY from environment variables (injected by
    the Meta/Scaler validator). Falls back to HF_API_BASE and HF_TOKEN for
    local development runs.

Workflow:
    1. Initialise the OpenAI client against the injected LiteLLM proxy.
    2. For each difficulty task (easy, medium, hard):
       a. Reset the environment and emit a [START] log line.
       b. At each timestep, format the observation into a text prompt.
       c. Send the prompt to the LLM via the OpenAI API client.
       d. Parse the LLM's text response into a DigiKulAction.
       e. Step the environment and emit a [STEP] log line.
    3. Emit an [END] log line and report the programmatic grader score.

Usage (local dev):
    export HF_TOKEN="hf_xxx..."
    export HF_API_BASE="https://api-inference.huggingface.co/v1"   # optional
    python inference.py

Usage (validator — env vars injected automatically):
    API_BASE_URL=<proxy_url> API_KEY=<proxy_key> python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Optional

# ── Ensure package root is importable ──
sys.path.insert(0, os.path.dirname(__file__))

from openai import OpenAI

from models import DigiKulAction, DigiKulObservation, QUALITY_BW_MAP, QUALITY_LABELS
from server.digikul_environment import DigiKulEnvironment, TASK_REGISTRY


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
#
#  Priority order (most-specific first):
#    1. API_BASE_URL / API_KEY   — injected by the Meta/Scaler validator
#    2. HF_API_BASE  / HF_TOKEN  — local development fallback
# ═══════════════════════════════════════════════════════════════════════════

# --- API base URL: validator proxy takes priority ---
API_BASE_URL: str = (
    os.environ.get("API_BASE_URL")          # validator-injected
    or os.environ.get("HF_API_BASE")        # local dev override
    or "https://api-inference.huggingface.co/v1"  # hard fallback
)

# --- API key: validator proxy takes priority ---
API_KEY: str = (
    os.environ.get("API_KEY")              # validator-injected
    or os.environ.get("HF_TOKEN")          # local dev fallback
    or ""
)

if not API_KEY:
    print(
        "WARNING: Neither API_KEY nor HF_TOKEN is set. "
        "LLM calls will fail.",
        flush=True,
    )

# --- Model name ---
# The validator may inject MODEL_NAME; fall back to a sensible default.
MODEL_NAME: str = (
    os.environ.get("MODEL_NAME")
    or os.environ.get("HF_MODEL")
    or "mistralai/Mistral-7B-Instruct-v0.3"
)

# Environment benchmark name (used in [START] log tag)
ENV_NAME: str = os.environ.get("DIGIKUL_ENV_NAME", "digikul-v0")

# Number of retries for LLM parsing failures
MAX_PARSE_RETRIES: int = 3


# ═══════════════════════════════════════════════════════════════════════════
#  OpenAI client
# ═══════════════════════════════════════════════════════════════════════════

def create_openai_client() -> OpenAI:
    """
    Create an OpenAI-compatible client.

    Always uses API_BASE_URL and API_KEY so that, when the validator
    injects those variables, every request flows through the LiteLLM proxy.
    """
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )


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
        'Respond with ONLY the JSON: {"quality_levels": [...]}'
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  LLM Interaction
# ═══════════════════════════════════════════════════════════════════════════

def query_llm(client: OpenAI, observation_text: str) -> str:
    """Send a prompt to the LLM and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
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
        cleaned = response_text
        cleaned = re.sub(r"```json\s*", "", cleaned)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.strip()

        # Extract the first JSON object found
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

        if len(levels) != num_nodes:
            return None

        levels = [max(0, min(3, int(q))) for q in levels]
        return DigiKulAction(quality_levels=levels)

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def get_fallback_action(obs: DigiKulObservation) -> DigiKulAction:
    """
    Deterministic fallback used when the LLM fails to produce a valid action.
    Uses a proportional heuristic based on student count and weather factor.
    """
    budget = obs.server_bandwidth
    scores = [node.num_students * node.weather_factor for node in obs.nodes]
    total_score = sum(scores) + 1e-8
    levels: List[int] = []

    for i, node in enumerate(obs.nodes):
        if node.num_students == 0:
            levels.append(0)
            continue
        share = (scores[i] / total_score) * budget
        cap = node.local_capacity
        if share >= 5.0 and cap >= 5.0:
            levels.append(3)
        elif share >= 2.0 and cap >= 2.0:
            levels.append(2)
        elif share >= 0.5 and cap >= 0.5:
            levels.append(1)
        else:
            levels.append(0)

    # Downgrade greedily until within budget
    total_bw = sum(QUALITY_BW_MAP[q] for q in levels)
    while total_bw > budget:
        worst_idx, worst_ratio = -1, float("inf")
        for i, q in enumerate(levels):
            if q > 0:
                ratio = obs.nodes[i].num_students / (QUALITY_BW_MAP[q] + 1e-8)
                if ratio < worst_ratio:
                    worst_ratio, worst_idx = ratio, i
        if worst_idx == -1:
            break
        levels[worst_idx] -= 1
        total_bw = sum(QUALITY_BW_MAP[q] for q in levels)

    return DigiKulAction(quality_levels=levels)


# ═══════════════════════════════════════════════════════════════════════════
#  Structured log helpers
#
#  The Meta/Scaler validator parses stdout for these exact tag formats.
#  Do NOT alter the field names, order, or value formatting.
# ═══════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    """
    Emit the mandatory [START] line.

    Format: [START] task=<name> env=<benchmark> model=<model>
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: DigiKulAction,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    """
    Emit one [STEP] line per environment step.

    Format: [STEP] step=<n> action=<action_str> reward=<0.00>
                   done=<true|false> error=<msg|null>

    Notes:
      - done uses lowercase JSON booleans (true/false).
      - reward is formatted to exactly 2 decimal places.
      - error is the literal string "null" when there is no error.
      - action is serialised as a compact JSON string.
    """
    action_str = json.dumps(action.quality_levels, separators=(",", ":"))
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """
    Emit the mandatory [END] line.

    Format: [END] success=<true|false> steps=<n> score=<score>
                  rewards=<r1,r2,...>

    Notes:
      - success uses lowercase JSON booleans.
      - score is formatted to 4 decimal places.
      - rewards is a comma-separated list of per-step rewards rounded to 2 dp.
    """
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Episode runner
# ═══════════════════════════════════════════════════════════════════════════

def run_episode(
    env: DigiKulEnvironment,
    client: OpenAI,
    task_name: str,
    verbose: bool = True,
) -> dict:
    """Run one full episode, querying the LLM at each step."""
    obs = env.reset()
    step_rewards: List[float] = []
    llm_calls = 0
    fallback_calls = 0
    episode_success = True

    # ── [START] ──────────────────────────────────────────────────────────
    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    if verbose:
        print(
            f"\n{'='*60}\n"
            f"  Task: {task_name.upper()} | Nodes: {len(obs.nodes)} | "
            f"Budget: {obs.server_bandwidth} Mbps | Steps: {obs.max_time_steps}\n"
            f"{'='*60}",
            flush=True,
        )

    while not obs.done:
        current_step = obs.time_step
        prompt = format_observation_prompt(obs)

        # ── Query LLM with retries ────────────────────────────────────────
        action: Optional[DigiKulAction] = None
        step_error: Optional[str] = None

        for attempt in range(MAX_PARSE_RETRIES):
            try:
                llm_response = query_llm(client, prompt)
                llm_calls += 1
                action = parse_llm_response(llm_response, len(obs.nodes))
                if action is not None:
                    break
                if verbose and attempt == 0:
                    print(
                        f"  ⚠ Step {current_step}: LLM parse failed, retrying...",
                        flush=True,
                    )
            except Exception as exc:
                llm_calls += 1
                step_error = str(exc)
                if verbose:
                    print(
                        f"  ⚠ Step {current_step}: LLM error: {exc}",
                        flush=True,
                    )
                break

        # ── Fallback if LLM failed ────────────────────────────────────────
        if action is None:
            action = get_fallback_action(obs)
            fallback_calls += 1
            if step_error is None:
                step_error = "llm_parse_failed_used_fallback"
            if verbose:
                print(f"  ⚡ Step {current_step}: Using fallback heuristic", flush=True)

        # ── Environment step ──────────────────────────────────────────────
        obs = env.step(action)
        step_rewards.append(obs.reward)

        # ── [STEP] ────────────────────────────────────────────────────────
        log_step(
            step=obs.time_step,
            action=action,
            reward=obs.reward,
            done=obs.done,
            error=step_error,
        )

        if verbose and obs.time_step % 10 == 0:
            print(
                f"  Step {obs.time_step}/{obs.max_time_steps} | "
                f"Reward: {obs.reward:.4f} | Load: {obs.server_load:.1f} Mbps",
                flush=True,
            )

    # ── Retrieve final grader score ───────────────────────────────────────
    final_state = env.state
    grader = final_state.grader_score if final_state.grader_score is not None else 0.0
    total_reward = sum(step_rewards)

    # ── [END] ─────────────────────────────────────────────────────────────
    log_end(
        success=episode_success,
        steps=len(step_rewards),
        score=grader,
        rewards=step_rewards,
    )

    if verbose:
        print(
            f"\n  ── Episode Complete ──\n"
            f"  Grader Score:      {grader:.4f}\n"
            f"  Total Reward:      {total_reward:.4f}\n"
            f"  LLM Calls:         {llm_calls}\n"
            f"  Fallback Calls:    {fallback_calls}\n"
            f"  Overload Events:   {final_state.total_overload_events}\n"
            f"  Budget Violations: {final_state.total_budget_violations}",
            flush=True,
        )

    return {
        "task": task_name,
        "grader_score": grader,
        "total_reward": total_reward,
        "llm_calls": llm_calls,
        "fallback_calls": fallback_calls,
        "overload_events": final_state.total_overload_events,
        "budget_violations": final_state.total_budget_violations,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run baseline inference across all three difficulty levels."""
    print(
        "╔══════════════════════════════════════════════════════════╗\n"
        "║       DigiKul-v0  •  Baseline LLM Inference            ║\n"
        "║       Meta PyTorch OpenEnv Hackathon                   ║\n"
        "╚══════════════════════════════════════════════════════════╝",
        flush=True,
    )
    print(f"\nModel:    {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print(f"Key src:  {'API_KEY (validator)' if os.environ.get('API_KEY') else 'HF_TOKEN (local)'}")

    llm_client = create_openai_client()

    results = []
    for task_name in ["easy", "medium", "hard"]:
        env = DigiKulEnvironment(task=task_name, seed=42)
        result = run_episode(env, llm_client, task_name, verbose=True)
        results.append(result)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<10} {'Grader':>10} {'Reward':>10} {'LLM Calls':>10}")
    print(f"  {'-'*40}")
    for r in results:
        print(
            f"  {r['task']:<10} {r['grader_score']:>10.4f} "
            f"{r['total_reward']:>10.4f} {r['llm_calls']:>10}"
        )
    print(f"{'='*60}")

    avg_grader = sum(r["grader_score"] for r in results) / len(results)
    print(f"\n  Average Grader Score: {avg_grader:.4f}")


if __name__ == "__main__":
    main()
