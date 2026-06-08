"""Cargo evaluation tools exposed to the LLM planner."""

from __future__ import annotations

import math
import os
from typing import Any


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = p2 - p1; dl = math.radians(lng2 - lng1)
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def distance_to_minutes(distance_km: float, speed_kmph: float = 60.0) -> int:
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil(distance_km / speed_kmph * 60.0))


class CargoEvaluationTool:
    """Computes profit, return cost, preference risk and one-step opportunity hints."""

    TOOL_SCHEMA = {
        "net_after_pickup_and_haul": "price - (pickup_km + haul_km) * cost_per_km",
        "return_to_anchor": "empty driving distance/time/cost from candidate destination to nearest preference point",
        "net_after_return": "net_after_pickup_and_haul - empty_return_cost",
        "future_relay_hint": "whether destination is near an anchor/preference point and may be useful for next query_cargo",
        "risk_flags": ["negative_net", "negative_after_return", "large_deadhead", "far_from_anchor", "miss_scheduled_visit"],
    }

    @classmethod
    def evaluate_candidate(
        cls,
        *,
        current_minute: int,
        finish_minute: int,
        end_lat: float,
        end_lng: float,
        net: float,
        hourly: float,
        pickup_km: float,
        cost_per_km: float,
        profile: dict[str, Any],
        next_visit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        risks: list[str] = []
        anchor = cls.nearest_anchor(end_lat, end_lng, profile)
        anchor_km = anchor["distance_km"] if anchor else None
        return_cost = None if anchor_km is None else anchor_km * cost_per_km
        return_minutes = None if anchor_km is None else distance_to_minutes(anchor_km)
        net_after_return = net if return_cost is None else net - return_cost

        visit_margin = None
        visit_return_minutes = None
        if next_visit is not None and cls._valid_point(next_visit.get("point")):
            point = next_visit["point"]
            visit_return_minutes = distance_to_minutes(haversine_km(end_lat, end_lng, float(point[0]), float(point[1])))
            deadline = int(next_visit.get("day", 0)) * 1440 + int(next_visit.get("arrive_before_minute") or 20 * 60)
            visit_margin = deadline - finish_minute - visit_return_minutes
            if visit_margin < 0:
                risks.append("miss_scheduled_visit")
            elif visit_margin < float(os.environ.get("AGENT_TIGHT_VISIT_MARGIN_MINUTES", "180") or 180):
                risks.append("tight_scheduled_visit")

        if net < 0:
            risks.append("negative_net")
        if net_after_return < 0:
            risks.append("negative_after_return")
        if pickup_km > float(os.environ.get("AGENT_LARGE_DEADHEAD_KM", "100") or 100):
            risks.append("large_deadhead")
        if anchor_km is not None and anchor_km > float(os.environ.get("AGENT_FAR_FROM_ANCHOR_KM", "120") or 120):
            risks.append("far_from_anchor")

        return {
            "tool_name": "cargo_evaluation_tool",
            "net_yuan": round(net, 2),
            "hourly_yuan": round(hourly, 2),
            "pickup_empty_km": round(pickup_km, 2),
            "nearest_anchor_km": None if anchor_km is None else round(anchor_km, 2),
            "empty_return_cost_yuan": None if return_cost is None else round(return_cost, 2),
            "empty_return_minutes": return_minutes,
            "net_after_return_yuan": round(net_after_return, 2),
            "next_visit_return_minutes": visit_return_minutes,
            "next_visit_margin_minutes": visit_margin,
            "future_relay_hint": cls._future_relay_hint(anchor_km, net_after_return, current_minute, finish_minute),
            "risk_flags": risks,
            "schema": cls.TOOL_SCHEMA,
        }

    @classmethod
    def evaluate_relay_option(cls, *, destination: list[float], anchors: list[list[float]], net_after_return: float) -> dict[str, Any]:
        """Heuristic for 'drive empty to a better area, then take a later order, then return'."""
        if not anchors:
            return {"tool_name": "route_simulation_tool", "relay_score": 0.0, "reason": "no_anchor_known"}
        best = min(haversine_km(destination[0], destination[1], p[0], p[1]) for p in anchors if cls._valid_point(p))
        relay_base = float(os.environ.get("AGENT_RELAY_ANCHOR_BASE_SCORE", "600") or 600)
        relay_km_penalty = float(os.environ.get("AGENT_RELAY_ANCHOR_KM_PENALTY", "20") or 20)
        score = net_after_return + max(0.0, relay_base - relay_km_penalty * best)
        return {"tool_name": "route_simulation_tool", "relay_score": round(score, 2), "nearest_future_anchor_km": round(best, 2)}

    @staticmethod
    def nearest_anchor(lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        points = [p for p in profile.get("preference_points", []) if CargoEvaluationTool._valid_point(p)]
        vf = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        if CargoEvaluationTool._valid_point(vf.get("point")):
            points.append(vf["point"])
        if not points:
            return None
        point = min(points, key=lambda p: haversine_km(lat, lng, float(p[0]), float(p[1])))
        return {"point": [float(point[0]), float(point[1])], "distance_km": haversine_km(lat, lng, float(point[0]), float(point[1]))}

    @staticmethod
    def _future_relay_hint(anchor_km: float | None, net_after_return: float, current_minute: int, finish_minute: int) -> str:
        if net_after_return < 0:
            return "avoid_unless_penalty_tradeoff_requires_it"
        if anchor_km is not None and anchor_km <= float(os.environ.get("AGENT_GOOD_ANCHOR_KM", "25") or 25):
            return "good_anchor_or_home_return_position"
        if finish_minute - current_minute <= int(os.environ.get("AGENT_SHORT_JOB_MINUTES", "240") or 240) and net_after_return > float(os.environ.get("AGENT_SHORT_JOB_RETURN_BUDGET_YUAN", "500") or 500):
            return "acceptable_short_job_with_return_budget"
        return "neutral_or_needs_llm_tradeoff"

    @staticmethod
    def _valid_point(point: Any) -> bool:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False
        try:
            lat = float(point[0]); lng = float(point[1])
        except (TypeError, ValueError):
            return False
        return -90 <= lat <= 90 and -180 <= lng <= 180
