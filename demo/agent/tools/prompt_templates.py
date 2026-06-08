"""Prompt templates for the single-entry, multi-role LLM agent."""

from __future__ import annotations

import json
from typing import Any


class PromptTemplates:
    @staticmethod
    def profile_messages(preferences: list[dict[str, Any]], agent_config: dict[str, Any], profile_tool_context: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are profile_agent in an online freight AI Agent. Convert visible Chinese driver preferences "
                    "into a machine-readable planning profile. The driver_profile_tool output is a deterministic safety floor. "
                    "When the driver has many preferences, build a priority map from preference_cards: critical/high hard constraints first, "
                    "then duplicate_preference_groups, soft tradeoffs, then unknown text. Repeated preferences can stack penalties, "
                    "so preserve duplicate groups instead of deduplicating them away. You may enrich missing fields, but must not remove constraints. "
                    "Compare deterministic fields with raw_preference_coverage_audit and preference_classification_tool.coverage_gaps; any gap must become a conservative planning rule, dynamic rule, or explicit unknown risk. "
                    "Use preference_classification_tool buckets to separate must-complete commitments, route/region constraints, cargo/order constraints, schedule/rest constraints and unknown items. "
                    "For long ordered commitments such as family events, wedding-car service, multi-day charter, or any text with 'first/then/until/release', produce long_sequence_commitments as an ordered step list. "
                    "Each step must include step_type, point or cargo_id, earliest_minute, deadline_minute, wait_minutes or hold_until_minute. Never collapse ordered text into one vague event. "
                    "For cumulative penalties such as '5 yuan per minute/hour', create cumulative_time_penalty_rules with rate_yuan_per_minute, time window, required point if any, and trigger. Treat tiny per-unit amounts as high-risk when repeated over many minutes. "
                    "For unknown_preferences, preserve the original text, split independent unknown likes/dislikes into unknown_attribute_tags, "
                    "group repeated unknown meaning, and add conservative planning hints instead of guessing facts. "
                    "If a preference is not covered by fixed schema fields, add dynamic_preference_rules: executable JSON rule specs with "
                    "id, source_preference_id, source_unknown_tag_ids, label, match, effect, per_violation_penalty_yuan, "
                    "expected_violations_per_month, penalty_multiplier, severity and confidence. "
                    "Use match.daily_time_window for repeated daily personal windows and match.periodic_stop_required for every-N-days rest/stop promises. "
                    "Do not treat a periodic large penalty as a single event; estimate repeated violations across the configured planning horizon and keep per-violation penalty separate. "
                    "Rules must use only visible preference text and generic cargo/status fields; never invent unavailable data. "
                    "When a penalty_anomaly_report is present, re-check whether the parsed profile or dynamic rules over-trigger. "
                    "Return strict JSON only. Use agent_config.planning_calendar for base_time, horizon_days and 0-based day indexes. Never invent coordinates."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "preferences": preferences,
                        "agent_config": agent_config,
                        "driver_profile_tool": profile_tool_context,
                        "output_schema": profile_tool_context.get("schema", {}),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

    @staticmethod
    def planner_messages(
        *,
        time_text: str,
        minute: int,
        status: dict[str, Any],
        profile: dict[str, Any],
        agent_config: dict[str, Any],
        history: list[dict[str, Any]],
        planning_board: dict[str, Any],
        evidence_board: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are planner_agent. Choose one next action from compact tool results. "
                    "Goal: maximize money. Candidate av = net after costs - current preference/task penalties. "
                    "Soft preferences are costs, not bans; hard_blocks and must_do_now are mandatory. "
                    "Prefer the highest positive-av non-hard-blocked order. Wait only when all tradeoffs are poor or a mandatory rest/task window must be protected. "
                    "Reposition only for explicit task/home/commitment anchors, never for vague market exploration. "
                    "Monthly off-days/rest are budgeted tasks: earn money early when pressure is normal; protect rest/off-day only when urgent or no profitable low-risk cargo exists. "
                    "If tools show unknown/high cumulative penalty risk, choose lower-risk positive av or request profile_recheck. "
                    "Return strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "time": time_text,
                        "minute": minute,
                        "status": status,
                        "profile": profile,
                        "agent_config": agent_config,
                        "recent_history": history,
                        "planning_board": planning_board,
                        "evidence_board": evidence_board,
                        "candidates": candidates,
                        "output": {
                            "action": "take_order|wait|reposition",
                            "cargo_id": "if take_order",
                            "duration_minutes": "if wait",
                            "latitude": "if reposition",
                            "longitude": "if reposition",
                            "reason_code": "profit_penalty_tradeoff|take_positive_order_now|wait_for_better_order|profile_recheck|future_penalty_risk",
                            "profit_penalty_tradeoff": {"net_after_return": "number", "current_action_penalty": "number", "action_value": "number"},
                            "anomaly_flags": ["optional short flags"],
                            "profile_recheck_needed": False,
                            "dynamic_rule_updates": [],
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

    @staticmethod
    def supervisor_messages(
        driver_id: str,
        time_text: str,
        status: dict[str, Any],
        profile: dict[str, Any],
        recommended: dict[str, Any],
        reason: str,
        history: list[dict[str, Any]],
        candidates: list[dict[str, Any]] | None = None,
        action_guard_report: dict[str, Any] | None = None,
        commitment_report: dict[str, Any] | None = None,
        time_task_report: dict[str, Any] | None = None,
        region_report: dict[str, Any] | None = None,
        task_penalty_report: dict[str, Any] | None = None,
        action_audit_board: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are action_review_agent. Review the recommended final action before execution using action_audit_board first. "
                    "action_audit_board is the human-readable control sheet: if it says verdict=reject, replace the action with the safest allowed wait/reposition unless there is a clearly lower-loss mandatory exception. "
                    "Check whether it harms true net return, preference penalties, future scheduled visits, rest windows, ordered commitments, or dynamic rules. "
                    "Always compare the tool interpretation with raw_preference_texts and high_priority_preference_cards; tools are evidence, not the final authority. "
                    "Use preference_classification_tool.universal_audit_checklist and coverage_gaps as a mandatory checklist before approving the action. "
                    "Pay special attention to cumulative penalties such as yuan per minute/hour, continuous wait/stay requirements, repeated rest violations, and unknown preference text. "
                    "Per-minute/per-hour uncapped penalty windows are severe constraints: during the window, approve only the required stay/wait/reposition or a lower-loss step required by the original ordered commitment. "
                    "Use action_preference_guard_tool, commitment_sequence_tool, time_task_progress_tool, region_preference_tool and task_penalty_optimizer_tool reports as hard evidence. "
                    "When tasks conflict physically, do not blindly keep the first rule; compare early-leave, late-arrival and not-at-point penalties, then choose the lower total loss if exact compliance is impossible. "
                    "For periodic visit tasks, verify whether today's visit is already credited or the monthly target is already done; repeated same-day waiting has zero extra value and should not block profitable safe orders. "
                    "If a monthly visit task is behind pace and the proposed action is idle waiting with no acceptable cargo, prefer one low-cost reposition/visit day over another long daytime wait. "
                    "For monthly full off-day/no-order tasks, reject early-month idle waiting when release_to_earn_money=true and profitable safe cargo exists; a full off-day only counts if no take_order/reposition happened today, so once today has activity do not keep waiting for that task. "
                    "For daily continuous rest, verify completed_continuous_rest_minutes and remaining_rest_minutes; approving a non-wait action near day end is unsafe unless task_penalty_optimizer_tool shows a lower total loss. "
                    "Do not reject profitable safe cargo only because of soft home/night/anchor penalties that are already included in action_value; long idle time and no-order days are also bad outcomes. "
                    "For mandatory ordered commitments, reread commitment_sequence_tool.source_text and next_step before approving the proposed action. "
                    "Do not rubber-stamp tool output: reject or replace the action with wait/reposition when the proposed next step skips an earlier step, misses required wait/stay time, is infeasible, or contradicts the source text. "
                    "If the commitment text is ambiguous, choose the safest action that preserves the sequence instead of taking a profitable cargo. "
                    "You cannot override guardrails or invent cargo. If taking an order, cargo_id must be in candidates or exactly match the commitment_sequence_tool recommended cargo. "
                    "If the proposed action can cause excessive preference loss or negative true value, reject it with wait/reposition. "
                    "Your reason_code must cite the most important action_audit_board item. Return strict JSON for take_order, wait, or reposition with reason_code and anomaly_flags."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "driver_id": driver_id,
                        "time": time_text,
                        "status": status,
                        "profile": profile,
                        "recommended_action": recommended,
                        "recommendation_reason": reason,
                        "recent_history": history,
                        "candidates": candidates or [],
                        "action_preference_guard_tool": action_guard_report or {},
                        "commitment_sequence_tool": commitment_report or {},
                        "time_task_progress_tool": time_task_report or {},
                        "region_preference_tool": region_report or {},
                        "task_penalty_optimizer_tool": task_penalty_report or {},
                        "action_audit_board": action_audit_board or {},
                        "output": {"action": "take_order|wait|reposition", "cargo_id": "if take_order", "duration_minutes": "if wait", "latitude": "if reposition", "longitude": "if reposition", "reason_code": "string", "anomaly_flags": []},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
