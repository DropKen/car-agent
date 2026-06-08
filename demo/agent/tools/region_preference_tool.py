"""Region and anchor preference helper for LLM planning."""

from __future__ import annotations

import math
import os
from typing import Any


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class RegionPreferenceTool:
    """Summarizes regional risk, anchors and destination impact."""

    @classmethod
    def evaluate(
        cls,
        *,
        current_point: tuple[float, float],
        candidate: dict[str, Any] | None,
        profile: dict[str, Any],
        time_task_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = cls._candidate_end(candidate) or current_point
        flags: list[str] = []
        details: list[dict[str, Any]] = []
        for circle in profile.get("forbidden_circles", []) or []:
            if not isinstance(circle, dict) or not cls._point(circle.get("center")):
                continue
            center = circle.get("center")
            radius = float(circle.get("radius_km", 0.0) or 0.0)
            for label, point in (("current", current_point), ("candidate_end", target)):
                distance = _haversine_km(point[0], point[1], float(center[0]), float(center[1]))
                if radius > 0 and distance <= radius:
                    flags.append("forbidden_region_proximity")
                    details.append({"type": "forbidden_circle", "point": label, "distance_km": round(distance, 2), "radius_km": radius})
        bounds = profile.get("geo_fence_bounds")
        outside_geo_fence = isinstance(bounds, dict) and not cls._inside_bounds(target[0], target[1], bounds)
        if outside_geo_fence:
            flags.append("candidate_end_outside_geo_fence")
            details.append({"type": "geo_fence", "point": "candidate_end", "tradeoff": "economic_soft_escape"})
        anchors = cls._anchor_distances(target, profile, time_task_report or {})
        if anchors and min(item["distance_km"] for item in anchors) > float(os.environ.get("AGENT_FAR_FROM_ANCHOR_KM", "120") or 120):
            flags.append("far_from_all_preference_anchors")
        urgent_targets = cls._urgent_task_distances(target, time_task_report or {})
        if urgent_targets:
            nearest = min(item["distance_km"] for item in urgent_targets)
            if nearest > float(os.environ.get("AGENT_FAR_FROM_URGENT_TASK_KM", "80") or 80):
                flags.append("candidate_moves_far_from_urgent_time_task")
        soft_geo_escape = os.environ.get("AGENT_ALLOW_PROFITABLE_GEOFENCE_SOFT_ESCAPE", "1").strip().lower() in {"1", "true", "yes"}
        hard_flags = {"forbidden_region_proximity"}
        if not soft_geo_escape:
            hard_flags.add("candidate_end_outside_geo_fence")
        return {
            "tool_name": "region_preference_tool",
            "target_point": [round(target[0], 6), round(target[1], 6)],
            "anchor_distances": anchors[:8],
            "urgent_time_task_distances": urgent_targets[:8],
            "risk_flags": list(dict.fromkeys(flags)),
            "hard_block": any(flag in hard_flags for flag in flags),
            "details": details[:12],
            "llm_instruction": "Use this to judge whether the action helps or harms regional anchors, forbidden regions, monthly visit points and urgent time tasks.",
        }

    @classmethod
    def _anchor_distances(cls, point: tuple[float, float], profile: dict[str, Any], time_task_report: dict[str, Any]) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        monthly_released = cls._monthly_visit_released(time_task_report)
        monthly_point = None
        visit = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        if cls._point(visit.get("point")):
            raw_visit = visit.get("point")
            monthly_point = (float(raw_visit[0]), float(raw_visit[1]))
        for index, raw in enumerate(profile.get("preference_points", []) or []):
            if cls._point(raw):
                if monthly_released and monthly_point and _haversine_km(float(raw[0]), float(raw[1]), monthly_point[0], monthly_point[1]) <= 1.0:
                    continue
                anchors.append({"id": f"preference_point_{index}", "point": list(raw[:2]), "distance_km": round(_haversine_km(point[0], point[1], float(raw[0]), float(raw[1])), 2)})
        if cls._point(visit.get("point")) and not monthly_released:
            raw = visit.get("point")
            anchors.append({"id": "monthly_visit_frequency_point", "point": list(raw[:2]), "distance_km": round(_haversine_km(point[0], point[1], float(raw[0]), float(raw[1])), 2)})
        return sorted(anchors, key=lambda item: float(item["distance_km"]))

    @staticmethod
    def _monthly_visit_released(time_task_report: dict[str, Any]) -> bool:
        for item in time_task_report.get("periodic_tasks", []) or []:
            if isinstance(item, dict) and item.get("type") == "monthly_visit_frequency":
                if item.get("status") == "done" or item.get("release_to_earn_money") is True:
                    return True
        return False

    @classmethod
    def _urgent_task_distances(cls, point: tuple[float, float], report: dict[str, Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for bucket in ("overdue_tasks", "urgent_tasks"):
            for task in report.get(bucket, []) or []:
                raw = task.get("point") if isinstance(task, dict) else None
                if cls._point(raw):
                    result.append(
                        {
                            "task_id": task.get("id"),
                            "task_type": task.get("type"),
                            "bucket": bucket,
                            "point": list(raw[:2]),
                            "distance_km": round(_haversine_km(point[0], point[1], float(raw[0]), float(raw[1])), 2),
                        }
                    )
        return sorted(result, key=lambda item: float(item["distance_km"]))

    @staticmethod
    def _candidate_end(candidate: dict[str, Any] | None) -> tuple[float, float] | None:
        if not isinstance(candidate, dict):
            return None
        end = candidate.get("end")
        if isinstance(end, (list, tuple)) and len(end) >= 2:
            lat = _as_float(end[0])
            lng = _as_float(end[1])
            if lat is not None and lng is not None:
                return (lat, lng)
        return None

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and _as_float(value[0]) is not None and _as_float(value[1]) is not None

    @staticmethod
    def _inside_bounds(lat: float, lng: float, bounds: dict[str, Any]) -> bool:
        try:
            return float(bounds["lat_min"]) <= lat <= float(bounds["lat_max"]) and float(bounds["lng_min"]) <= lng <= float(bounds["lng_max"])
        except (KeyError, TypeError, ValueError):
            return True
