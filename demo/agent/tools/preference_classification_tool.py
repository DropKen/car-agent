"""Generic preference classification helper.

The tool does not contain driver-specific rules. It organizes the structured
preference cards produced by driver_profile_tool into operational buckets that
the LLM planner can reason about.
"""

from __future__ import annotations

from typing import Any


class PreferenceClassificationTool:
    """Classifies parsed preferences into generic agent work streams."""

    COMMITMENT_TYPES = {"required_cargo", "temporary_event", "scheduled_visit", "visit_frequency", "long_sequence_commitment"}
    ROUTE_TYPES = {"geo_fence", "forbidden_circle", "avoid_region", "home_return"}
    CARGO_TYPES = {"avoid_cargo", "distance_limit", "pickup_deadhead_limit", "monthly_deadhead_limit", "cargo_attribute"}
    SCHEDULE_TYPES = {"rest_or_no_action", "first_order_deadline", "daily_order_limit", "cumulative_time_penalty"}
    UNIVERSAL_AUDIT_CHECKLIST = [
        {"key": "profit_true_net", "question": "Does the action remain profitable after pickup empty cost, haul cost, return/anchor cost, current preference penalty and future preference penalty?"},
        {"key": "cumulative_penalty", "question": "Can a small per-minute/hour penalty accumulate during this action or its aftermath?"},
        {"key": "ordered_commitment", "question": "Does the action preserve every first/then/until/release sequence and required wait/stay phase?"},
        {"key": "schedule_rest", "question": "Does the action break daily rest, sleep windows, off-days, medical/family/religious/meal windows or long continuous driving limits?"},
        {"key": "monthly_periodic_goal", "question": "Does the action endanger monthly/weekly/every-N-days visit, stop, region, cargo, or home-return goals given remaining days?"},
        {"key": "route_region", "question": "Do origin, destination, corridor or reposition target conflict with forbidden cities, regions, geofences, mountains, ports, downtown, border, weather or road constraints?"},
        {"key": "cargo_attribute", "question": "Does cargo name/type/weight/volume/temperature/loading/labor/certificate/customer text conflict with any explicit or unknown cargo preference?"},
        {"key": "time_feasibility", "question": "Can the driver still catch listing remove_time, load window, unload window and all future commitment travel buffers?"},
        {"key": "unknown_text", "question": "Is there any raw preference text not covered by tools that should become a conservative dynamic rule before taking action?"},
        {"key": "repeat_violation", "question": "Would this repeat a previous small violation enough times to become a large monthly penalty?"},
        {"key": "fallback_safety", "question": "If any required information is missing, is wait/reposition safer than taking a profitable but ambiguous order?"},
    ]

    @classmethod
    def classify_profile(cls, profile: dict[str, Any]) -> dict[str, Any]:
        cards = [card for card in profile.get("preference_cards", []) or [] if isinstance(card, dict)]
        buckets = {
            "must_complete_commitments": [],
            "route_region_constraints": [],
            "cargo_order_constraints": [],
            "schedule_rest_constraints": [],
            "unknown_or_llm_required": [],
            "soft_tradeoffs": [],
        }
        for card in cards:
            types = {str(item) for item in card.get("types", []) if str(item)}
            target = cls._bucket_for(types, str(card.get("tradeoff_mode") or ""))
            buckets[target].append(cls._compact_card(card))
        return {
            "tool_name": "preference_classification_tool",
            "buckets": buckets,
            "universal_audit_checklist": cls.UNIVERSAL_AUDIT_CHECKLIST,
            "coverage_gaps": cls._coverage_gaps(profile, cards),
            "priority_order": [
                "must_complete_commitments",
                "route_region_constraints",
                "schedule_rest_constraints",
                "cargo_order_constraints",
                "unknown_or_llm_required",
                "soft_tradeoffs",
            ],
            "unknown_handling_policy": (
                "For unknown_or_llm_required, ask the LLM to map visible text to generic JSON dynamic rules; "
                "do not invent coordinates, cargo IDs, cities, or hidden driver facts. If no exact tool covers a risk, treat raw text as a conservative guardrail."
            ),
        }

    @classmethod
    def _coverage_gaps(cls, profile: dict[str, Any], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []
        high_risk_words = ("每", "分钟", "小时", "上不封顶", "必须", "不得", "禁止", "先", "再", "直到", "方可", "婚", "接亲", "包车", "驻场", "宗教", "医院", "孩子", "老人", "证件")
        covered_types = {
            "required_cargo": bool(profile.get("required_cargos")),
            "temporary_event": bool(profile.get("temporary_events")),
            "long_sequence_commitment": bool(profile.get("long_sequence_commitments")),
            "cumulative_time_penalty": bool(profile.get("cumulative_time_penalty_rules")),
            "visit_frequency": bool(profile.get("visit_frequency")),
            "scheduled_visit": bool(profile.get("scheduled_visits")),
            "geo_fence": bool(profile.get("geo_fence_bounds") or profile.get("forbidden_circles")),
            "rest_or_no_action": bool((profile.get("daily_rest") or {}).get("hours") or profile.get("required_off_days")),
        }
        for card in cards:
            content = str(card.get("content") or "")
            types = {str(item) for item in card.get("types", []) if str(item)}
            uncovered = [t for t in types if t in covered_types and not covered_types[t]]
            high_risk_untyped = any(word in content for word in high_risk_words) and not types
            if uncovered or high_risk_untyped or card.get("tradeoff_mode") == "unknown":
                gaps.append(
                    {
                        "id": card.get("id"),
                        "severity": card.get("severity"),
                        "types": sorted(types),
                        "uncovered_types": uncovered,
                        "content": content[:220],
                        "llm_required": True,
                    }
                )
        return gaps[:20]

    @classmethod
    def _bucket_for(cls, types: set[str], mode: str) -> str:
        if types & cls.COMMITMENT_TYPES:
            return "must_complete_commitments"
        if types & cls.ROUTE_TYPES:
            return "route_region_constraints"
        if types & cls.SCHEDULE_TYPES:
            return "schedule_rest_constraints"
        if types & cls.CARGO_TYPES:
            return "cargo_order_constraints"
        if mode == "unknown" or not types:
            return "unknown_or_llm_required"
        return "soft_tradeoffs"

    @staticmethod
    def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": card.get("id"),
            "types": card.get("types", []),
            "risk_key": card.get("risk_key"),
            "severity": card.get("severity"),
            "tradeoff_mode": card.get("tradeoff_mode"),
            "penalty_amount": card.get("penalty_amount"),
            "penalty_cap": card.get("penalty_cap"),
            "content": str(card.get("content") or "")[:160],
        }
