"""Action-level preference guard for final LLM/tool decisions.

The guard is intentionally generic: it only consumes the visible planning
profile and the action/candidate context that the online API already exposed.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any


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


class ActionPreferenceGuardTool:
    """Checks whether a proposed action creates future preference penalties."""

    DEFAULT_BUFFER_MINUTES = 60

    @classmethod
    def evaluate_candidate(
        cls,
        *,
        status: dict[str, Any],
        profile: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        finish_minute = int(candidate.get("finish_minute", current_minute) or current_minute)
        end = candidate.get("end") if isinstance(candidate.get("end"), list) else None
        if end and len(end) >= 2:
            lat, lng = float(end[0]), float(end[1])
        else:
            lat = float(status.get("current_lat", 0.0) or 0.0)
            lng = float(status.get("current_lng", 0.0) or 0.0)
        return cls._evaluate_window(
            current_minute=current_minute,
            finish_minute=finish_minute,
            lat=lat,
            lng=lng,
            profile=profile,
            action_name="take_order",
        )

    @classmethod
    def review_action(
        cls,
        *,
        status: dict[str, Any],
        profile: dict[str, Any],
        action: dict[str, Any],
        candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)
        name = str(action.get("action", "")).lower()
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        finish_minute = current_minute
        if name == "take_order" and candidate:
            finish_minute = int(candidate.get("finish_minute", current_minute) or current_minute)
            end = candidate.get("end") if isinstance(candidate.get("end"), list) else None
            if end and len(end) >= 2:
                lat, lng = float(end[0]), float(end[1])
        elif name == "wait":
            finish_minute = current_minute + max(1, int(params.get("duration_minutes", 60) or 60))
        elif name == "reposition":
            target_lat = _as_float(params.get("latitude"))
            target_lng = _as_float(params.get("longitude"))
            if target_lat is not None and target_lng is not None:
                finish_minute = current_minute + _minutes_for_km(_haversine_km(lat, lng, target_lat, target_lng))
                lat, lng = target_lat, target_lng
        return cls._evaluate_window(
            current_minute=current_minute,
            finish_minute=finish_minute,
            lat=lat,
            lng=lng,
            profile=profile,
            action_name=name,
        )

    @classmethod
    def _evaluate_window(
        cls,
        *,
        current_minute: int,
        finish_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        action_name: str,
    ) -> dict[str, Any]:
        flags: list[str] = []
        details: list[dict[str, Any]] = []
        penalty = 0.0
        hard_block = False

        for event in profile.get("temporary_events", []) or []:
            if not isinstance(event, dict):
                continue
            pickup = event.get("pickup_point")
            home = event.get("home_point")
            pickup_minute = _as_int(event.get("pickup_minute"))
            release_minute = _as_int(event.get("release_minute"))
            if not cls._point(pickup) or not cls._point(home) or pickup_minute is None or release_minute is None:
                continue
            to_pickup = _minutes_for_km(_haversine_km(lat, lng, float(pickup[0]), float(pickup[1])))
            to_home = _minutes_for_km(_haversine_km(float(pickup[0]), float(pickup[1]), float(home[0]), float(home[1])))
            buffer_min = max(120, cls.DEFAULT_BUFFER_MINUTES)
            if current_minute < pickup_minute - cls._commitment_notice_minutes():
                continue
            if finish_minute + to_pickup + 10 + to_home + buffer_min > pickup_minute and current_minute < release_minute:
                flags.append("future_temporary_event_risk")
                hard_block = True
                p = cls._penalty_for_types(profile, {"temporary_event"}, default=9000.0)
                penalty += p
                details.append({"type": "temporary_event_sequence", "penalty_yuan": p, "must_leave_by_minute": pickup_minute - to_pickup - 10 - to_home - buffer_min})

        for item in profile.get("required_cargos", []) or []:
            if not isinstance(item, dict):
                continue
            pickup = item.get("pickup_point")
            online = _as_int(item.get("online_minute"))
            cargo_id = str(item.get("cargo_id") or "")
            if not cargo_id or not cls._point(pickup) or online is None:
                continue
            if current_minute < online - cls._commitment_notice_minutes():
                continue
            travel = _minutes_for_km(_haversine_km(lat, lng, float(pickup[0]), float(pickup[1])))
            buffer_min = max(120, cls.DEFAULT_BUFFER_MINUTES)
            if current_minute <= online and finish_minute + travel + buffer_min > online:
                flags.append("future_required_cargo_risk")
                hard_block = True
                p = cls._penalty_for_types(profile, {"required_cargo"}, default=10000.0)
                penalty += p
                details.append({"type": "required_cargo_deadline", "cargo_id": cargo_id, "penalty_yuan": p, "must_leave_by_minute": online - travel - buffer_min})

        for report in cls._long_sequence_reports(current_minute, finish_minute, lat, lng, profile):
            flags.append("future_long_sequence_risk")
            penalty += float(report.get("penalty_yuan", 0.0) or 0.0)
            hard_block = True
            details.append(report)

        home_report = cls._home_return_report(finish_minute, lat, lng, profile)
        if home_report:
            flags.append("future_home_return_risk")
            penalty += float(home_report["penalty_yuan"])
            hard_block = hard_block or bool(home_report.get("hard_block"))
            details.append(home_report)

        rest_report = cls._daily_rest_report(current_minute, finish_minute, profile)
        if rest_report:
            flags.append("future_daily_rest_risk")
            penalty += float(rest_report["penalty_yuan"])
            details.append(rest_report)

        for report in cls._cumulative_time_penalty_reports(current_minute, finish_minute, lat, lng, profile, action_name):
            flags.append("cumulative_time_penalty_risk")
            penalty += float(report.get("penalty_yuan", 0.0) or 0.0)
            hard_block = hard_block or bool(report.get("hard_block"))
            details.append(report)

        return {
            "tool_name": "action_preference_guard_tool",
            "action_name": action_name,
            "finish_minute": finish_minute,
            "estimated_future_preference_penalty_yuan": round(penalty, 2),
            "hard_block": hard_block,
            "risk_flags": list(dict.fromkeys(flags)),
            "details": details[:12],
        }

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and _as_float(value[0]) is not None and _as_float(value[1]) is not None

    @classmethod
    def _long_sequence_reports(
        cls,
        current_minute: int,
        finish_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                continue
            steps = [step for step in sequence.get("steps", []) or [] if isinstance(step, dict)]
            for step in steps:
                point = step.get("point")
                if not cls._point(point):
                    continue
                step_type = str(step.get("step_type") or "").lower()
                earliest = _as_int(step.get("earliest_minute"))
                deadline = _as_int(step.get("deadline_minute"))
                hold_until = _as_int(step.get("hold_until_minute"))
                if hold_until is not None and current_minute >= hold_until:
                    continue
                target = earliest if step_type == "visit_and_wait" and earliest is not None else (deadline if deadline is not None else earliest)
                if target is None:
                    target = hold_until
                if target is None or target < current_minute:
                    continue
                if current_minute < target - cls._commitment_notice_minutes():
                    continue
                travel = _minutes_for_km(_haversine_km(lat, lng, float(point[0]), float(point[1])))
                wait_minutes = max(0, _as_int(step.get("wait_minutes")) or 0)
                buffer_min = max(60, _as_int(sequence.get("buffer_minutes")) or 0)
                must_arrive_by = target
                if step_type == "stay_until" and deadline is not None:
                    must_arrive_by = deadline
                if finish_minute + travel + wait_minutes + buffer_min <= must_arrive_by:
                    continue
                p = cls._penalty_for_types(profile, {"temporary_event", "long_sequence"}, default=9000.0)
                reports.append(
                    {
                        "type": "long_sequence_deadline",
                        "sequence_id": sequence.get("id"),
                        "step_id": step.get("id"),
                        "step_type": step_type,
                        "penalty_yuan": p,
                        "target_minute": target,
                        "finish_minute": finish_minute,
                        "return_travel_minutes": travel,
                        "buffer_minutes": buffer_min,
                    }
                )
                break
        return reports

    @staticmethod
    def _commitment_notice_minutes() -> int:
        try:
            value = int(os.environ.get("AGENT_COMMITMENT_NOTICE_MINUTES", "1440") or 1440)
        except ValueError:
            value = 1440
        return max(60, min(7 * 1440, value))

    @classmethod
    def _home_return_report(cls, finish_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        for card in profile.get("preference_cards", []) or []:
            if not isinstance(card, dict) or "home_return" not in (card.get("types") or []):
                continue
            content = str(card.get("content") or "")
            coords = re.findall(r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]", content)
            if not coords:
                continue
            home_lat, home_lng = map(float, coords[0])
            deadline = cls._deadline_minute(content) or 23 * 60
            day = finish_minute // 1440
            deadline_abs = day * 1440 + deadline
            if finish_minute % 1440 > deadline:
                deadline_abs += 1440
            travel = _minutes_for_km(_haversine_km(lat, lng, home_lat, home_lng))
            if finish_minute + travel <= deadline_abs - 60:
                return None
            p = float(card.get("penalty_amount") or 900.0)
            return {"type": "home_return_deadline", "penalty_yuan": p, "hard_block": False, "deadline_minute": deadline_abs, "return_minutes": travel}
        return None

    @classmethod
    def _cumulative_time_penalty_reports(
        cls,
        current_minute: int,
        finish_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        action_name: str,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for rule in profile.get("cumulative_time_penalty_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            rate = _as_float(rule.get("rate_yuan_per_minute"))
            if rate is None or rate <= 0:
                continue
            window_start = _as_int(rule.get("window_start_minute"))
            window_end = _as_int(rule.get("window_end_minute"))
            if window_start is None:
                continue
            if window_end is None or window_end <= window_start:
                window_end = window_start + 24 * 60
            overlap = max(0, min(finish_minute, window_end) - max(current_minute, window_start))
            if overlap <= 0:
                continue
            point = rule.get("required_point")
            radius = _as_float(rule.get("radius_km")) or 1.0
            at_required_point = True
            if cls._point(point):
                at_required_point = _haversine_km(lat, lng, float(point[0]), float(point[1])) <= radius
            violates = action_name != "wait" or not at_required_point
            if not violates:
                continue
            penalty = overlap * rate
            hard_threshold = float(os.environ.get("AGENT_CUMULATIVE_TASK_HARD_BLOCK_YUAN", "9000") or 9000)
            reports.append(
                {
                    "type": "cumulative_time_penalty",
                    "source_text": rule.get("source_text"),
                    "rate_yuan_per_minute": round(rate, 4),
                    "overlap_minutes": overlap,
                    "penalty_yuan": round(penalty, 2),
                    "hard_block": penalty >= hard_threshold,
                    "required_point": point,
                    "trigger": rule.get("trigger"),
                }
            )
        return reports

    @classmethod
    def _daily_rest_report(cls, current_minute: int, finish_minute: int, profile: dict[str, Any]) -> dict[str, Any] | None:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        hours = _as_float(rest.get("hours")) or 0.0
        if hours <= 0:
            return None
        min_wait = int(hours * 60)
        start_day = current_minute // 1440
        end_day = finish_minute // 1440
        if finish_minute - current_minute <= 1440 - min_wait:
            return None
        p = cls._penalty_for_types(profile, {"rest_or_no_action"}, default=300.0)
        return {"type": "daily_rest_may_be_broken", "penalty_yuan": p, "days_touched": max(1, end_day - start_day + 1), "required_continuous_wait_minutes": min_wait}

    @staticmethod
    def _deadline_minute(text: str) -> int | None:
        if "23点" in text:
            return 23 * 60
        m = re.search(r"([0-9]{1,2})\s*点\s*前", text)
        return None if not m else (int(m.group(1)) % 24) * 60

    @staticmethod
    def _penalty_for_types(profile: dict[str, Any], types: set[str], default: float) -> float:
        values = []
        for card in profile.get("preference_cards", []) or []:
            if isinstance(card, dict) and types.intersection(set(card.get("types") or [])):
                values.append(float(card.get("penalty_amount") or 0.0))
        return max(values) if values else default
