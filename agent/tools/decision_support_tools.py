"""Generic decision-support tools for profitable preference-safe dispatching."""

from __future__ import annotations

import math
import os
from typing import Any

MONTH_HORIZON_MINUTES = int(os.environ.get("AGENT_HORIZON_DAYS", "31") or 31) * 24 * 60


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _minutes_for_km(km: float, speed_kmh: float = 60.0) -> int:
    return int(math.ceil(max(0.0, km) / max(1.0, speed_kmh) * 60.0))


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class DecisionSupportTools:
    """Runs compact generic tools for one candidate action."""

    @classmethod
    def evaluate_candidate(
        cls,
        *,
        status: dict[str, Any],
        profile: dict[str, Any],
        candidate: dict[str, Any],
        history_summary: list[dict[str, Any]],
    ) -> dict[str, Any]:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        cost_per_km = float(status.get("cost_per_km", 1.5) or 1.5)
        current = cls._point_tuple([status.get("current_lat", 0.0), status.get("current_lng", 0.0)])
        start = cls._point_tuple(candidate.get("start"))
        end = cls._point_tuple(candidate.get("end"))
        pickup_km = float(candidate.get("pickup_km", 0.0) or 0.0)
        haul_km = float(candidate.get("haul_km", 0.0) or 0.0)
        price = float(candidate.get("price", 0.0) or 0.0)
        ready = int(candidate.get("ready_minute", current_minute) or current_minute)
        finish = int(candidate.get("finish_minute", current_minute) or current_minute)
        penalty = float(candidate.get("current_action_penalty", 0.0) or 0.0)
        net_after_return = float(candidate.get("net_after_return", candidate.get("estimated_net", 0.0)) or 0.0)

        cost_breakdown = cls._cost_breakdown_tool(
            price=price,
            pickup_km=pickup_km,
            haul_km=haul_km,
            end=end,
            profile=profile,
            cost_per_km=cost_per_km,
        )
        commitments = cls._future_commitments(profile, current_minute)
        time_report = cls._time_feasibility_tool(finish, end, commitments)
        future_position = cls._future_position_value_tool(end, profile, cost_per_km)
        market_wait = cls._market_wait_value_tool(history_summary, net_after_return, penalty, finish - current_minute)
        expiry = cls._cargo_expiry_recheck_tool(current_minute, ready, candidate)
        profit_floor = cls._profit_floor_tool(net_after_return, penalty, finish - current_minute)
        penalty_budget = cls._preference_penalty_budget_tool(profile, penalty)
        route_corridor = cls._route_corridor_tool(current, start, end, profile)
        rest_plan = cls._rest_planner_tool(current_minute, finish, profile)
        home_anchor = cls._home_anchor_strategy_tool(current_minute, finish, end, profile, cost_per_km)
        commitment_status = cls._commitment_status_tracker_tool(profile, commitments, current_minute, end)
        unknown_test = cls._unknown_preference_test_tool(profile, candidate)
        repeat_guard = cls._repeat_violation_guard_tool(
            current_minute=current_minute,
            finish=finish,
            end=end,
            profile=profile,
            history_summary=history_summary,
        )

        additional_future_cost = 0.0
        additional_future_cost += float(time_report.get("future_commitment_empty_cost_yuan", 0.0) or 0.0)
        additional_future_cost += float(rest_plan.get("estimated_penalty_yuan", 0.0) or 0.0)
        # Waiting value is advisory only. Do not penalize a positive-profit order
        # just because a better one might arrive later.
        if route_corridor.get("hard_block"):
            additional_future_cost += float(route_corridor.get("estimated_penalty_yuan", 0.0) or 0.0)
        additional_future_cost += float(repeat_guard.get("estimated_repeat_penalty_yuan", 0.0) or 0.0)
        additional_future_cost += float(expiry.get("estimated_time_risk_cost_yuan", 0.0) or 0.0)

        risk_flags = []
        for report in (time_report, market_wait, expiry, profit_floor, penalty_budget, route_corridor, rest_plan, home_anchor, commitment_status, unknown_test, repeat_guard):
            risk_flags.extend(report.get("risk_flags", []) if isinstance(report, dict) else [])

        return {
            "tool_name": "decision_support_tools",
            "cost_breakdown_tool": cost_breakdown,
            "market_wait_value_tool": market_wait,
            "future_position_value_tool": future_position,
            "time_feasibility_tool": time_report,
            "cargo_expiry_recheck_tool": expiry,
            "profit_floor_tool": profit_floor,
            "preference_penalty_budget_tool": penalty_budget,
            "route_corridor_tool": route_corridor,
            "rest_planner_tool": rest_plan,
            "home_anchor_strategy_tool": home_anchor,
            "commitment_status_tracker_tool": commitment_status,
            "unknown_preference_test_tool": unknown_test,
            "repeat_violation_guard_tool": repeat_guard,
            "decision_audit_tool": cls._decision_audit_tool(candidate, cost_breakdown, penalty, additional_future_cost),
            "additional_future_cost_yuan": round(additional_future_cost, 2),
            "risk_flags": list(dict.fromkeys(risk_flags)),
        }

    @staticmethod
    def _cost_breakdown_tool(*, price: float, pickup_km: float, haul_km: float, end: tuple[float, float], profile: dict[str, Any], cost_per_km: float) -> dict[str, Any]:
        anchor_km = DecisionSupportTools._nearest_profile_point_km(end, profile)
        empty_return_cost = 0.0 if anchor_km is None else anchor_km * cost_per_km
        pickup_cost = pickup_km * cost_per_km
        haul_cost = haul_km * cost_per_km
        return {
            "tool_name": "cost_breakdown_tool",
            "gross_price_yuan": round(price, 2),
            "pickup_empty_km": round(pickup_km, 2),
            "pickup_empty_cost_yuan": round(pickup_cost, 2),
            "haul_km": round(haul_km, 2),
            "haul_cost_yuan": round(haul_cost, 2),
            "empty_return_km": None if anchor_km is None else round(anchor_km, 2),
            "empty_return_cost_yuan": round(empty_return_cost, 2),
            "true_net_before_preference_yuan": round(price - pickup_cost - haul_cost - empty_return_cost, 2),
            "empty_driving_cost_included": True,
        }

    @staticmethod
    def _market_wait_value_tool(history_summary: list[dict[str, Any]], net_after_return: float, penalty: float, duration_minutes: int) -> dict[str, Any]:
        min_hourly = float(os.environ.get("AGENT_MIN_HOURLY_ACTION_VALUE_YUAN", "0") or 0)
        hours = max(1.0, duration_minutes / 60.0)
        action_value = net_after_return - penalty
        hourly_value = action_value / hours
        opportunity_cost = max(0.0, min_hourly * hours - action_value)
        return {
            "tool_name": "market_wait_value_tool",
            "hourly_action_value_yuan": round(hourly_value, 2),
            "reference_min_hourly_yuan": min_hourly,
            "opportunity_cost_yuan": round(opportunity_cost, 2),
            "wait_preferred": hourly_value < min_hourly,
            "history_count": len(history_summary),
            "risk_flags": ["low_hourly_wait_value"] if hourly_value < min_hourly else [],
        }

    @staticmethod
    def _future_position_value_tool(end: tuple[float, float], profile: dict[str, Any], cost_per_km: float) -> dict[str, Any]:
        anchor_km = DecisionSupportTools._nearest_profile_point_km(end, profile)
        if anchor_km is None:
            return {"tool_name": "future_position_value_tool", "known_anchor": False, "risk_flags": []}
        return {
            "tool_name": "future_position_value_tool",
            "known_anchor": True,
            "nearest_anchor_km": round(anchor_km, 2),
            "anchor_recovery_cost_yuan": round(anchor_km * cost_per_km, 2),
            "position_quality_score": round(max(-1000.0, 500.0 - 12.0 * anchor_km), 2),
            "risk_flags": ["far_from_anchor"] if anchor_km > float(os.environ.get("AGENT_FAR_FROM_ANCHOR_KM", "120") or 120) else [],
        }

    @staticmethod
    def _time_feasibility_tool(finish: int, end: tuple[float, float], commitments: list[dict[str, Any]]) -> dict[str, Any]:
        details = []
        future_cost = 0.0
        flags = []
        for item in commitments:
            point = item.get("point")
            target = _as_int(item.get("target_minute"))
            if not DecisionSupportTools._point(point) or target is None:
                continue
            travel = _minutes_for_km(_haversine_km(end[0], end[1], float(point[0]), float(point[1])))
            margin = target - finish - travel - 60
            details.append({"type": item.get("type"), "target_minute": target, "travel_minutes": travel, "buffered_margin_minutes": margin})
            if margin < 0:
                flags.append("time_commitment_conflict")
                future_cost += float(item.get("penalty_yuan", 1000.0) or 1000.0)
        return {
            "tool_name": "time_feasibility_tool",
            "hard_block": bool(flags),
            "future_commitment_empty_cost_yuan": round(future_cost, 2),
            "details": details[:8],
            "risk_flags": list(dict.fromkeys(flags)),
        }

    @staticmethod
    def _cargo_expiry_recheck_tool(current_minute: int, ready: int, candidate: dict[str, Any]) -> dict[str, Any]:
        load_end = _as_int(candidate.get("load_end_minute"))
        remove_minute = _as_int(candidate.get("remove_minute"))
        load_margin = None if load_end is None else load_end - ready
        listing_margin = None if remove_minute is None else remove_minute - current_minute
        pickup_minutes = max(0, ready - current_minute)
        hard_buffer = int(os.environ.get("AGENT_CARGO_TIME_HARD_BUFFER_MINUTES", "15") or 15)
        soft_buffer = int(os.environ.get("AGENT_CARGO_TIME_SOFT_BUFFER_MINUTES", "45") or 45)
        listing_hard_buffer = int(os.environ.get("AGENT_CARGO_REMOVE_HARD_BUFFER_MINUTES", "3") or 3)
        listing_soft_buffer = int(os.environ.get("AGENT_CARGO_REMOVE_SOFT_BUFFER_MINUTES", "20") or 20)
        margins = [m for m in (load_margin, listing_margin) if m is not None]
        if not margins:
            return {"tool_name": "cargo_time_availability_tool", "risk_flags": [], "time_reliability_score": 0.5}
        margin = min(margins)
        score = max(0.0, min(1.0, margin / max(1.0, max(soft_buffer, listing_soft_buffer) * 2.0)))
        risk_cost = 0.0
        flags: list[str] = []
        if load_margin is not None and load_margin < hard_buffer:
            flags.append("cargo_time_window_too_tight")
            flags.append("cargo_time_hard_block")
            risk_cost = 100000.0
        if listing_margin is not None and listing_margin < listing_hard_buffer:
            flags.append("cargo_listing_expires_too_soon")
            flags.append("cargo_time_hard_block")
            risk_cost = 100000.0
        elif listing_margin is not None and listing_margin < listing_soft_buffer:
            flags.append("cargo_listing_expires_soon")
            risk_cost = max(risk_cost, 500.0)
        if load_margin is not None and hard_buffer <= load_margin < soft_buffer:
            flags.append("cargo_time_window_tight")
            # Tight windows often turn into wasted empty driving or expired cargo.
            risk_cost = max(risk_cost, max(100.0, float(candidate.get("pickup_km", 0.0) or 0.0) * 3.0))
        return {
            "tool_name": "cargo_time_availability_tool",
            "load_window_margin_minutes": load_margin,
            "listing_remove_margin_minutes": listing_margin,
            "minimum_time_margin_minutes": margin,
            "pickup_minutes": pickup_minutes,
            "hard_buffer_minutes": hard_buffer,
            "soft_buffer_minutes": soft_buffer,
            "listing_hard_buffer_minutes": listing_hard_buffer,
            "listing_soft_buffer_minutes": listing_soft_buffer,
            "time_reliability_score": round(score, 4),
            "estimated_time_risk_cost_yuan": round(risk_cost, 2),
            "should_recheck_before_take_order": (load_margin is not None and load_margin < soft_buffer) or (listing_margin is not None and listing_margin < listing_soft_buffer),
            "risk_flags": flags,
        }

    @staticmethod
    def _profit_floor_tool(net_after_return: float, penalty: float, duration_minutes: int) -> dict[str, Any]:
        min_value = float(os.environ.get("AGENT_MIN_ACTION_VALUE_YUAN", "0") or 0)
        min_hourly = float(os.environ.get("AGENT_MIN_HOURLY_ACTION_VALUE_YUAN", "0") or 0)
        value = net_after_return - penalty
        hourly = value / max(1.0, duration_minutes / 60.0)
        flags = []
        if value < min_value:
            flags.append("below_min_action_value")
        if hourly < min_hourly:
            flags.append("below_min_hourly_value")
        return {"tool_name": "profit_floor_tool", "action_value_yuan": round(value, 2), "hourly_value_yuan": round(hourly, 2), "risk_flags": flags}

    @staticmethod
    def _preference_penalty_budget_tool(profile: dict[str, Any], penalty: float) -> dict[str, Any]:
        total_cap = 0.0
        for card in profile.get("preference_cards", []) or []:
            if isinstance(card, dict):
                total_cap += float(card.get("penalty_cap") or card.get("penalty_amount") or 0.0)
        ratio = 0.0 if total_cap <= 0 else penalty / total_cap
        return {"tool_name": "preference_penalty_budget_tool", "candidate_penalty_yuan": round(penalty, 2), "visible_penalty_budget_yuan": round(total_cap, 2), "budget_ratio": round(ratio, 4), "risk_flags": ["large_penalty_budget_use"] if ratio > 0.2 else []}

    @staticmethod
    def _route_corridor_tool(current: tuple[float, float], start: tuple[float, float], end: tuple[float, float], profile: dict[str, Any]) -> dict[str, Any]:
        flags = []
        for a, b in ((current, start), (start, end)):
            for point in DecisionSupportTools._sample_segment(a, b):
                if not DecisionSupportTools._point_allowed(point, profile):
                    flags.append("route_corridor_preference_violation")
                    break
        return {"tool_name": "route_corridor_tool", "hard_block": bool(flags), "estimated_penalty_yuan": 2000.0 if flags else 0.0, "risk_flags": list(dict.fromkeys(flags))}

    @staticmethod
    def _rest_planner_tool(current_minute: int, finish: int, profile: dict[str, Any]) -> dict[str, Any]:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        hours = _as_float(rest.get("hours")) or 0.0
        if hours <= 0:
            return {"tool_name": "rest_planner_tool", "risk_flags": []}
        touched_days = finish // 1440 - current_minute // 1440 + 1
        long_action = finish - current_minute > 1440 - int(hours * 60)
        penalty = DecisionSupportTools._penalty_for_types(profile, {"rest_or_no_action"}, 300.0) if long_action else 0.0
        return {"tool_name": "rest_planner_tool", "required_rest_hours": hours, "days_touched": touched_days, "estimated_penalty_yuan": penalty, "risk_flags": ["rest_plan_at_risk"] if long_action else []}

    @staticmethod
    def _home_anchor_strategy_tool(current_minute: int, finish: int, end: tuple[float, float], profile: dict[str, Any], cost_per_km: float) -> dict[str, Any]:
        anchor_km = DecisionSupportTools._nearest_profile_point_km(end, profile)
        if anchor_km is None:
            return {"tool_name": "home_anchor_strategy_tool", "risk_flags": []}
        return_cost = anchor_km * cost_per_km
        late = finish % 1440 >= 20 * 60
        return {"tool_name": "home_anchor_strategy_tool", "return_anchor_km": round(anchor_km, 2), "return_anchor_empty_cost_yuan": round(return_cost, 2), "recommend_return_anchor": late and anchor_km > 3, "risk_flags": ["late_far_from_anchor"] if late and anchor_km > 30 else []}

    @staticmethod
    def _commitment_status_tracker_tool(profile: dict[str, Any], commitments: list[dict[str, Any]], current_minute: int, end: tuple[float, float]) -> dict[str, Any]:
        statuses = []
        flags = []
        for item in commitments:
            target = _as_int(item.get("target_minute"))
            point = item.get("point")
            if target is None or not DecisionSupportTools._point(point):
                continue
            travel = _minutes_for_km(_haversine_km(end[0], end[1], float(point[0]), float(point[1])))
            state = "future"
            if current_minute > target:
                state = "expired"
                flags.append("commitment_expired")
            elif current_minute + travel + 60 > target:
                state = "urgent"
                flags.append("commitment_urgent")
            statuses.append({"type": item.get("type"), "state": state, "target_minute": target, "travel_minutes": travel})
        return {"tool_name": "commitment_status_tracker_tool", "statuses": statuses[:8], "risk_flags": list(dict.fromkeys(flags))}

    @staticmethod
    def _unknown_preference_test_tool(profile: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        unknown = profile.get("unknown_attribute_tags", []) or profile.get("unknown_preferences", []) or []
        if not isinstance(unknown, list) or not unknown:
            return {"tool_name": "unknown_preference_test_tool", "risk_flags": []}
        checks = []
        text = " ".join(str(candidate.get(key, "")) for key in ("cargo_name", "start_city", "end_city"))
        for item in unknown[:8]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("content_span") or item.get("content") or "")[:40]
            unsure = bool(label and any(part and part in text for part in label.split()))
            checks.append({"unknown_label": label, "candidate_match": "unsure" if unsure else "not_obvious"})
        return {"tool_name": "unknown_preference_test_tool", "checks": checks, "risk_flags": ["unknown_preference_needs_llm_check"] if checks else []}

    @staticmethod
    def _repeat_violation_guard_tool(
        *,
        current_minute: int,
        finish: int,
        end: tuple[float, float],
        profile: dict[str, Any],
        history_summary: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Avoid repeating preference violations, especially unknown or daily ones."""
        risks: list[str] = []
        details: list[dict[str, Any]] = []
        penalty = 0.0

        home = DecisionSupportTools._home_return_constraint(profile)
        if home:
            home_lat, home_lng, deadline, base_penalty = home
            travel = _minutes_for_km(_haversine_km(end[0], end[1], home_lat, home_lng))
            day = finish // 1440
            deadline_abs = day * 1440 + deadline
            if finish % 1440 > deadline:
                deadline_abs += 1440
            will_violate = finish + travel > deadline_abs - 60
            past_violations = DecisionSupportTools._recent_home_night_violations(history_summary, home_lat, home_lng, deadline)
            if will_violate:
                multiplier = min(5, 1 + past_violations)
                repeat_penalty = base_penalty * multiplier
                penalty += repeat_penalty
                risks.append("repeat_home_return_violation_risk" if past_violations else "home_return_violation_risk")
                details.append({
                    "type": "home_return",
                    "past_recent_violations": past_violations,
                    "return_minutes": travel,
                    "deadline_minute": deadline_abs,
                    "estimated_penalty_yuan": round(repeat_penalty, 2),
                })

        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        rest_hours = _as_float(rest.get("hours")) or 0.0
        if rest_hours > 0 and finish - current_minute > 1440 - int(rest_hours * 60):
            base = DecisionSupportTools._penalty_for_types(profile, {"rest_or_no_action"}, 300.0)
            penalty += base
            risks.append("repeat_daily_rest_violation_risk")
            details.append({"type": "daily_rest", "required_hours": rest_hours, "estimated_penalty_yuan": round(base, 2)})

        unknown = profile.get("unknown_attribute_tags", []) or profile.get("unknown_preferences", []) or []
        if isinstance(unknown, list) and unknown:
            current_risk = max(0, len(unknown) - 2) * 50.0
            if current_risk:
                penalty += current_risk
            risks.append("unknown_preference_conservative_check")
            details.append({"type": "unknown_preference", "unknown_items": min(len(unknown), 12), "estimated_penalty_yuan": round(current_risk, 2)})

        return {
            "tool_name": "repeat_violation_guard_tool",
            "estimated_repeat_penalty_yuan": round(penalty, 2),
            "risk_flags": list(dict.fromkeys(risks)),
            "details": details[:8],
        }

    @staticmethod
    def _decision_audit_tool(candidate: dict[str, Any], cost_breakdown: dict[str, Any], penalty: float, future_cost: float) -> dict[str, Any]:
        return {
            "tool_name": "decision_audit_tool",
            "cargo_id": candidate.get("cargo_id"),
            "net_formula": "price - pickup_empty_cost - haul_cost - empty_return_cost - preference_penalty - future_cost",
            "empty_driving_cost_included": True,
            "preference_penalty_yuan": round(penalty, 2),
            "additional_future_cost_yuan": round(future_cost, 2),
            "cost_breakdown_ref": cost_breakdown,
        }

    @staticmethod
    def _future_commitments(profile: dict[str, Any], current_minute: int) -> list[dict[str, Any]]:
        items = []
        for cargo in profile.get("required_cargos", []) or []:
            if isinstance(cargo, dict) and DecisionSupportTools._point(cargo.get("pickup_point")):
                target = _as_int(cargo.get("online_minute"))
                if target is not None and target >= current_minute:
                    items.append({"type": "required_cargo", "point": cargo.get("pickup_point"), "target_minute": target, "penalty_yuan": DecisionSupportTools._penalty_for_types(profile, {"required_cargo"}, 10000.0)})
        for event in profile.get("temporary_events", []) or []:
            if isinstance(event, dict) and DecisionSupportTools._point(event.get("pickup_point")):
                target = _as_int(event.get("pickup_minute"))
                if target is not None and target >= current_minute:
                    items.append({"type": "temporary_event", "point": event.get("pickup_point"), "target_minute": target, "penalty_yuan": DecisionSupportTools._penalty_for_types(profile, {"temporary_event"}, 9000.0)})
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                continue
            for step in sequence.get("steps", []) or []:
                if not isinstance(step, dict) or not DecisionSupportTools._point(step.get("point")):
                    continue
                target = _as_int(step.get("deadline_minute")) or _as_int(step.get("earliest_minute"))
                if target is not None and target >= current_minute:
                    items.append({"type": "long_sequence_commitment", "point": step.get("point"), "target_minute": target, "penalty_yuan": DecisionSupportTools._penalty_for_types(profile, {"temporary_event"}, 9000.0)})
                    break
        for visit in profile.get("scheduled_visits", []) or []:
            if not isinstance(visit, dict) or not DecisionSupportTools._point(visit.get("point")):
                continue
            day = _as_int(visit.get("day"))
            if day is None:
                continue
            target = day * 1440 + int(visit.get("arrive_before_minute") or 20 * 60)
            if target >= current_minute:
                items.append({"type": "scheduled_visit", "point": visit.get("point"), "target_minute": target, "penalty_yuan": 1000.0})
        return sorted(items, key=lambda item: int(item.get("target_minute", MONTH_HORIZON_MINUTES)))[:8]

    @staticmethod
    def _nearest_profile_point_km(point: tuple[float, float], profile: dict[str, Any]) -> float | None:
        points = [p for p in profile.get("preference_points", []) or [] if DecisionSupportTools._point(p)]
        if not points:
            return None
        return min(_haversine_km(point[0], point[1], float(p[0]), float(p[1])) for p in points)

    @staticmethod
    def _sample_segment(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[float, float]]:
        dist = _haversine_km(a[0], a[1], b[0], b[1])
        steps = max(2, min(16, int(dist // 50) + 2))
        return [(a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps) for i in range(steps + 1)]

    @staticmethod
    def _point_allowed(point: tuple[float, float], profile: dict[str, Any]) -> bool:
        bounds = profile.get("geo_fence_bounds")
        if isinstance(bounds, dict):
            try:
                if not (float(bounds["lat_min"]) <= point[0] <= float(bounds["lat_max"]) and float(bounds["lng_min"]) <= point[1] <= float(bounds["lng_max"])):
                    return False
            except (KeyError, TypeError, ValueError):
                pass
        for circle in profile.get("forbidden_circles", []) or []:
            if not isinstance(circle, dict) or not DecisionSupportTools._point(circle.get("center")):
                continue
            radius = float(circle.get("radius_km", 0.0) or 0.0)
            center = circle["center"]
            if radius > 0 and _haversine_km(point[0], point[1], float(center[0]), float(center[1])) <= radius:
                return False
        return True

    @staticmethod
    def _home_return_constraint(profile: dict[str, Any]) -> tuple[float, float, int, float] | None:
        for card in profile.get("preference_cards", []) or []:
            if not isinstance(card, dict) or "home_return" not in (card.get("types") or []):
                continue
            content = str(card.get("content") or "")
            import re

            coords = re.findall(r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]", content)
            if not coords:
                continue
            deadline = 23 * 60
            m = re.search(r"([0-9]{1,2})\s*点\s*前", content)
            if m:
                deadline = (int(m.group(1)) % 24) * 60
            return (float(coords[0][0]), float(coords[0][1]), deadline, float(card.get("penalty_amount") or 900.0))
        return None

    @staticmethod
    def _recent_home_night_violations(history_summary: list[dict[str, Any]], home_lat: float, home_lng: float, deadline: int) -> int:
        seen_days: set[int] = set()
        for record in history_summary[-80:]:
            if not isinstance(record, dict):
                continue
            minute = _as_int(record.get("end_minute"))
            pos = record.get("position_after")
            point = DecisionSupportTools._history_point(pos)
            if minute is None or point is None:
                continue
            day = minute // 1440
            tod = minute % 1440
            if deadline <= tod or tod < 8 * 60:
                dist = _haversine_km(point[0], point[1], home_lat, home_lng)
                if dist > 1.0:
                    seen_days.add(day)
        return len(seen_days)

    @staticmethod
    def _history_point(value: Any) -> tuple[float, float] | None:
        if isinstance(value, dict):
            lat = _as_float(value.get("lat"))
            lng = _as_float(value.get("lng"))
            if lat is not None and lng is not None:
                return (lat, lng)
        if DecisionSupportTools._point(value):
            return (float(value[0]), float(value[1]))
        return None

    @staticmethod
    def _penalty_for_types(profile: dict[str, Any], types: set[str], default: float) -> float:
        values = []
        for card in profile.get("preference_cards", []) or []:
            if isinstance(card, dict) and types.intersection(set(card.get("types") or [])):
                values.append(float(card.get("penalty_amount") or 0.0))
        return max(values) if values else default

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and _as_float(value[0]) is not None and _as_float(value[1]) is not None

    @staticmethod
    def _point_tuple(value: Any) -> tuple[float, float]:
        if DecisionSupportTools._point(value):
            return (float(value[0]), float(value[1]))
        return (0.0, 0.0)
