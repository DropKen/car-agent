"""Driver commitment sequencing tool.

This tool extracts execution-order commitments from the planning profile and
returns deterministic next actions when timing is tight. It is intentionally
data-driven and does not know any driver-specific IDs.
"""

from __future__ import annotations

import math
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


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


class CommitmentSequenceTool:
    """Checks required cargo, family events and scheduled visits every action."""

    DEFAULT_BUFFER_MINUTES = 60

    @classmethod
    def next_required_action(
        cls,
        *,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        report = cls.commitment_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=profile,
            history=history,
        )
        return {"action": report.get("recommended_action"), "report": report} if report.get("recommended_action") else {"action": None, "report": report}

    @classmethod
    def commitment_report(
        cls,
        *,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        commitments: list[dict[str, Any]] = []
        action = cls._long_sequence_action(current_minute, lat, lng, profile, history, commitments)
        if action is None:
            action = cls._temporary_event_action(current_minute, lat, lng, profile, history, commitments)
        if action is None:
            action = cls._required_cargo_action(current_minute, lat, lng, profile, history, commitments)
        if action is None:
            action = cls._scheduled_visit_action(current_minute, lat, lng, profile, history, commitments)
        return {
            "tool_name": "commitment_sequence_tool",
            "current_minute": current_minute,
            "llm_review_required": bool(action),
            "review_instruction": "Mandatory ordered commitments are tool proposals only; LLM must verify source_text, next_step, sequence order and feasibility before execution.",
            "commitments": commitments[:12],
            "recommended_action": action,
        }

    @classmethod
    def _long_sequence_action(
        cls,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
        commitments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                continue
            steps = [step for step in sequence.get("steps", []) or [] if isinstance(step, dict)]
            if not steps:
                continue
            buffer_minutes = _as_int(sequence.get("buffer_minutes")) or cls.DEFAULT_BUFFER_MINUTES
            state = cls._sequence_state(sequence, steps, current_minute, lat, lng, history, buffer_minutes)
            commitments.append(state)
            next_step = state.get("next_step")
            if not isinstance(next_step, dict):
                continue
            recommended = cls._action_for_sequence_step(next_step, current_minute, lat, lng, buffer_minutes)
            if recommended is not None:
                recommended["reason_code"] = f"long_sequence_{next_step.get('step_type', 'step')}"
                recommended["sequence_id"] = sequence.get("id")
                recommended["sequence_step_id"] = next_step.get("id")
                return recommended
        return None

    @classmethod
    def _sequence_state(
        cls,
        sequence: dict[str, Any],
        steps: list[dict[str, Any]],
        current_minute: int,
        lat: float,
        lng: float,
        history: list[dict[str, Any]],
        buffer_minutes: int,
    ) -> dict[str, Any]:
        completed_ids: list[str] = []
        next_step: dict[str, Any] | None = None
        for step in steps:
            if cls._sequence_step_completed(step, current_minute, history):
                completed_ids.append(str(step.get("id") or step.get("label") or len(completed_ids)))
                continue
            next_step = step
            break
        summary: dict[str, Any] = {
            "type": "long_sequence_commitment",
            "sequence_id": sequence.get("id"),
            "commitment_type": sequence.get("commitment_type"),
            "completed_step_ids": completed_ids,
            "next_step": next_step,
            "source_text": sequence.get("source_text"),
        }
        if isinstance(next_step, dict) and cls._point(next_step.get("point")):
            point = next_step.get("point")
            distance = _haversine_km(lat, lng, float(point[0]), float(point[1]))
            travel = _minutes_for_km(distance)
            deadline = _as_int(next_step.get("deadline_minute"))
            earliest = _as_int(next_step.get("earliest_minute"))
            step_type = str(next_step.get("step_type") or "").lower()
            target = earliest if step_type == "visit_and_wait" and earliest is not None else (deadline if deadline is not None else earliest)
            leave_by = None if target is None else target - travel - buffer_minutes
            summary.update(
                {
                    "next_distance_km": round(distance, 2),
                    "next_travel_minutes": travel,
                    "next_leave_by_minute": leave_by,
                    "next_deadline_minute": deadline,
                    "next_earliest_minute": earliest,
                    "is_urgent": leave_by is not None and current_minute >= leave_by,
                    "is_late": deadline is not None and current_minute > deadline,
                }
            )
        return summary

    @classmethod
    def _sequence_step_completed(cls, step: dict[str, Any], current_minute: int, history: list[dict[str, Any]]) -> bool:
        step_type = str(step.get("step_type") or "").lower()
        earliest = _as_int(step.get("earliest_minute")) or 0
        point = step.get("point")
        radius = float(step.get("radius_km") or 1.0)
        if step_type == "take_cargo":
            cargo_id = str(step.get("cargo_id") or "")
            return bool(cargo_id) and cls._cargo_seen(history, cargo_id)
        if not cls._point(point):
            return False
        target = (float(point[0]), float(point[1]))
        if step_type == "visit_and_wait":
            wait_minutes = max(1, _as_int(step.get("wait_minutes")) or 1)
            if current_minute >= earliest + wait_minutes:
                return True
            return cls._waited_near_after(history, target, earliest, wait_minutes, radius)
        if step_type == "stay_until":
            hold_until = _as_int(step.get("hold_until_minute"))
            if hold_until is None or current_minute < hold_until:
                return False
            return True
            hold_start = _as_int(step.get("deadline_minute")) or earliest
            return cls._waited_near_after(history, target, hold_start, max(1, hold_until - hold_start), radius)
        return cls._visited_after(history, target, earliest, radius)

    @classmethod
    def _action_for_sequence_step(
        cls,
        step: dict[str, Any],
        current_minute: int,
        lat: float,
        lng: float,
        buffer_minutes: int,
    ) -> dict[str, Any] | None:
        step_type = str(step.get("step_type") or "").lower()
        if step_type == "take_cargo":
            cargo_id = str(step.get("cargo_id") or "")
            return {"action": "take_order", "params": {"cargo_id": cargo_id}} if cargo_id else None
        point = step.get("point")
        if not cls._point(point):
            return None
        p = (float(point[0]), float(point[1]))
        radius = float(step.get("radius_km") or 1.0)
        distance = _haversine_km(lat, lng, p[0], p[1])
        travel = _minutes_for_km(distance)
        earliest = _as_int(step.get("earliest_minute"))
        deadline = _as_int(step.get("deadline_minute"))
        target = earliest if step_type == "visit_and_wait" and earliest is not None else (deadline if deadline is not None else earliest)
        leave_by = None if target is None else target - travel - buffer_minutes
        if leave_by is not None and current_minute < leave_by and target is not None and target - current_minute > 24 * 60:
            return None
        if distance > radius:
            return {"action": "reposition", "params": {"latitude": p[0], "longitude": p[1]}}
        if earliest is not None and current_minute < earliest:
            return {"action": "wait", "params": {"duration_minutes": max(1, min(240, earliest - current_minute))}}
        if step_type == "stay_until":
            hold_until = _as_int(step.get("hold_until_minute"))
            if hold_until is not None and current_minute >= hold_until:
                return None
            duration = 60 if hold_until is None else max(1, min(240, hold_until - current_minute))
            return {"action": "wait", "params": {"duration_minutes": duration}}
        wait_minutes = max(1, _as_int(step.get("wait_minutes")) or 1)
        return {"action": "wait", "params": {"duration_minutes": wait_minutes}}

    @staticmethod
    def _record_end_minute(record: dict[str, Any]) -> int | None:
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        value = record.get("simulation_end_minute", result.get("simulation_progress_minutes"))
        return _as_int(value)

    @classmethod
    def _waited_near_after(
        cls,
        history: list[dict[str, Any]],
        point: tuple[float, float],
        minute: int,
        wait_minutes: int,
        radius_km: float,
    ) -> bool:
        run = 0
        for record in history:
            if not isinstance(record, dict):
                continue
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "wait":
                pos = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
                try:
                    if _haversine_km(float(pos.get("lat")), float(pos.get("lng")), point[0], point[1]) > radius_km:
                        run = 0
                except (TypeError, ValueError):
                    run = 0
                continue
            end_time = cls._record_end_minute(record)
            if end_time is None:
                continue
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            start_time = max(0, end_time - max(0, elapsed))
            pos_before = record.get("position_before") if isinstance(record.get("position_before"), dict) else {}
            pos_after = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            try:
                near = (
                    _haversine_km(float(pos_before.get("lat")), float(pos_before.get("lng")), point[0], point[1]) <= radius_km
                    and _haversine_km(float(pos_after.get("lat")), float(pos_after.get("lng")), point[0], point[1]) <= radius_km
                )
            except (TypeError, ValueError):
                near = False
            if not near:
                run = 0
                continue
            overlap = max(0, end_time - max(start_time, minute))
            run += overlap
            if run >= wait_minutes:
                return True
        return False

    @classmethod
    def _temporary_event_action(
        cls,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
        commitments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for event in profile.get("temporary_events", []) or []:
            if not isinstance(event, dict):
                continue
            pickup = event.get("pickup_point")
            home = event.get("home_point")
            pickup_minute = _as_int(event.get("pickup_minute"))
            release_minute = _as_int(event.get("release_minute"))
            if not cls._point(pickup) or not cls._point(home) or pickup_minute is None or release_minute is None:
                continue
            pickup_p = (float(pickup[0]), float(pickup[1]))
            home_p = (float(home[0]), float(home[1]))
            picked = cls._waited_near_after(history, pickup_p, pickup_minute, wait_minutes=10, radius_km=1.0)
            at_pickup = _haversine_km(lat, lng, pickup_p[0], pickup_p[1]) <= 1.0
            at_home = _haversine_km(lat, lng, home_p[0], home_p[1]) <= 1.0
            to_pickup = _minutes_for_km(_haversine_km(lat, lng, pickup_p[0], pickup_p[1]))
            to_home = _minutes_for_km(_haversine_km(lat, lng, home_p[0], home_p[1]))
            leave_by = pickup_minute - to_pickup - cls.DEFAULT_BUFFER_MINUTES
            commitments.append({
                "type": "temporary_event_sequence",
                "pickup_minute": pickup_minute,
                "release_minute": release_minute,
                "picked": picked,
                "at_pickup": at_pickup,
                "at_home": at_home,
                "leave_by_minute": leave_by,
            })
            if current_minute < leave_by:
                continue
            if current_minute < release_minute and not picked:
                if not at_pickup:
                    return {"action": "reposition", "params": {"latitude": pickup_p[0], "longitude": pickup_p[1]}, "reason_code": "commitment_go_pickup_first"}
                return {"action": "wait", "params": {"duration_minutes": max(1, pickup_minute + 10 - current_minute) if current_minute < pickup_minute + 10 else 10}, "reason_code": "commitment_wait_pickup"}
            if current_minute < release_minute:
                if not at_home:
                    return {"action": "reposition", "params": {"latitude": home_p[0], "longitude": home_p[1]}, "reason_code": "commitment_return_home"}
                return {"action": "wait", "params": {"duration_minutes": max(1, min(240, release_minute - current_minute))}, "reason_code": "commitment_stay_home"}
        return None

    @classmethod
    def _required_cargo_action(
        cls,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
        commitments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for item in profile.get("required_cargos", []) or []:
            if not isinstance(item, dict):
                continue
            cargo_id = str(item.get("cargo_id") or "")
            pickup = item.get("pickup_point")
            online = _as_int(item.get("online_minute"))
            if not cargo_id or not cls._point(pickup) or online is None:
                continue
            if cls._cargo_seen(history, cargo_id):
                continue
            p = (float(pickup[0]), float(pickup[1]))
            dist = _haversine_km(lat, lng, p[0], p[1])
            travel = _minutes_for_km(dist)
            buffer_minutes = max(cls.DEFAULT_BUFFER_MINUTES, _as_int(item.get("buffer_minutes")) or 0, 120)
            leave_by = online - travel - buffer_minutes
            commitments.append({"type": "required_cargo", "cargo_id": cargo_id, "online_minute": online, "distance_km": round(dist, 2), "leave_by_minute": leave_by})
            if current_minute < leave_by:
                continue
            if current_minute < online:
                if dist > 2.0:
                    return {"action": "reposition", "params": {"latitude": p[0], "longitude": p[1]}, "reason_code": "commitment_required_cargo_position"}
                return {"action": "wait", "params": {"duration_minutes": max(1, min(120, online - current_minute))}, "reason_code": "commitment_wait_required_cargo_online"}
            if dist <= 15.0:
                return {"action": "take_order", "params": {"cargo_id": cargo_id}, "reason_code": "commitment_take_required_cargo"}
            return {"action": "reposition", "params": {"latitude": p[0], "longitude": p[1]}, "reason_code": "commitment_required_cargo_late_position"}
        return None

    @classmethod
    def _scheduled_visit_action(
        cls,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
        commitments: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for item in profile.get("scheduled_visits", []) or []:
            if not isinstance(item, dict):
                continue
            day = _as_int(item.get("day"))
            point = item.get("point")
            if day is None or not cls._point(point):
                continue
            deadline = day * 1440 + int(item.get("arrive_before_minute") or 20 * 60)
            if current_minute > deadline:
                continue
            p = (float(point[0]), float(point[1]))
            if cls._visited_after(history, p, day * 1440, radius_km=1.5):
                continue
            travel = _minutes_for_km(_haversine_km(lat, lng, p[0], p[1]))
            leave_by = deadline - travel - cls.DEFAULT_BUFFER_MINUTES
            commitments.append({"type": "scheduled_visit", "target_minute": deadline, "leave_by_minute": leave_by})
            if current_minute < leave_by:
                continue
            if _haversine_km(lat, lng, p[0], p[1]) > 1.5:
                return {"action": "reposition", "params": {"latitude": p[0], "longitude": p[1]}, "reason_code": "commitment_scheduled_visit_position"}
            return {"action": "wait", "params": {"duration_minutes": max(1, int(item.get("wait_minutes") or 60))}, "reason_code": "commitment_scheduled_visit_wait"}
        return None

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2

    @staticmethod
    def _cargo_seen(history: list[dict[str, Any]], cargo_id: str) -> bool:
        for record in history:
            action = record.get("action", {}) if isinstance(record, dict) else {}
            result = record.get("result", {}) if isinstance(record, dict) else {}
            params = action.get("params", {}) if isinstance(action, dict) else {}
            if str(params.get("cargo_id") or result.get("cargo_id") or "") == cargo_id:
                return True
        return False

    @staticmethod
    def _visited_after(history: list[dict[str, Any]], point: tuple[float, float], minute: int, radius_km: float) -> bool:
        for record in history:
            if not isinstance(record, dict):
                continue
            end_time = record.get("simulation_end_minute")
            if end_time is None:
                result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
                end_time = result.get("simulation_progress_minutes")
            try:
                if int(end_time or 0) < minute:
                    continue
            except (TypeError, ValueError):
                continue
            pos = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            lat = pos.get("lat")
            lng = pos.get("lng")
            try:
                if _haversine_km(float(lat), float(lng), point[0], point[1]) <= radius_km:
                    return True
            except (TypeError, ValueError):
                continue
        return False
