"""Generic cargo route and preference compliance checks."""

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


class RouteComplianceTool:
    """Checks whether an order route violates generic profile constraints."""

    @classmethod
    def evaluate(
        cls,
        *,
        cargo_name: str,
        start_city: str,
        end_city: str,
        current_point: tuple[float, float],
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        flags: list[str] = []
        details: list[dict[str, Any]] = []
        for keyword in profile.get("avoid_cargo_keywords", []) or []:
            if str(keyword) and str(keyword) in cargo_name:
                flags.append("avoid_cargo_keyword")
                details.append({"type": "avoid_cargo_keyword", "keyword": str(keyword)})
        for item in profile.get("avoid_regions", []) or []:
            if not isinstance(item, dict):
                continue
            region = str(item.get("region") or "")
            if region and (region in start_city or region in end_city):
                flags.append("avoid_region")
                details.append({"type": "avoid_region", "region": region})
        bounds = profile.get("geo_fence_bounds")
        geo_outside = False
        if isinstance(bounds, dict):
            for label, point in (("current", current_point), ("start", start_point), ("end", end_point)):
                if not cls._inside_bounds(point[0], point[1], bounds):
                    geo_outside = True
                    flags.append("geo_fence_outside")
                    details.append({"type": "geo_fence_outside", "point": label, "tradeoff": "economic_soft_escape"})
        for circle in profile.get("forbidden_circles", []) or []:
            center = circle.get("center") if isinstance(circle, dict) else None
            radius = float(circle.get("radius_km", 0.0) or 0.0) if isinstance(circle, dict) else 0.0
            if not cls._point(center) or radius <= 0:
                continue
            for label, point in (("current", current_point), ("start", start_point), ("end", end_point)):
                if _haversine_km(point[0], point[1], float(center[0]), float(center[1])) <= radius:
                    flags.append("forbidden_circle")
                    details.append({"type": "forbidden_circle", "point": label, "radius_km": radius})
        soft_geo_escape = os.environ.get("AGENT_ALLOW_PROFITABLE_GEOFENCE_SOFT_ESCAPE", "1").strip().lower() in {"1", "true", "yes"}
        hard_block = bool(flags)
        if soft_geo_escape and geo_outside:
            hard_block = any(flag != "geo_fence_outside" for flag in flags)
        return {
            "tool_name": "route_compliance_tool",
            "hard_block": hard_block,
            "risk_flags": list(dict.fromkeys(flags)),
            "details": details[:12],
        }

    @staticmethod
    def _inside_bounds(lat: float, lng: float, bounds: dict[str, Any]) -> bool:
        try:
            return float(bounds["lat_min"]) <= lat <= float(bounds["lat_max"]) and float(bounds["lng_min"]) <= lng <= float(bounds["lng_max"])
        except (KeyError, TypeError, ValueError):
            return True

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2
