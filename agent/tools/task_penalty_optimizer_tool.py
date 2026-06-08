"""Penalty optimizer for conflicting time tasks.

When commitments cannot all be satisfied exactly, this tool estimates which
small violation avoids a larger penalty. It does not execute actions; it gives
the LLM a loss-minimization board for the current action.
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


class TaskPenaltyOptimizerTool:
    """Estimates minimum-penalty tradeoffs across time/periodic tasks."""

    @classmethod
    def evaluate_action(
        cls,
        *,
        current_minute: int,
        current_point: tuple[float, float],
        action: dict[str, Any],
        profile: dict[str, Any],
        time_task_report: dict[str, Any],
        candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        finish, target = cls._action_finish_and_target(current_minute, current_point, action, candidate)
        direct_penalties = cls._direct_action_penalties(current_minute, finish, current_point, target, action, profile)
        rest_conflicts = cls._daily_rest_conflicts(current_minute, finish, action, time_task_report)
        deadline_conflicts = cls._deadline_conflicts(finish, target, profile, time_task_report)
        conflict_tradeoffs = cls._conflict_tradeoffs(current_minute, current_point, profile, time_task_report)
        total = sum(float(item.get("penalty_yuan", 0.0) or 0.0) for item in direct_penalties)
        total += sum(float(item.get("estimated_penalty_yuan", 0.0) or 0.0) for item in rest_conflicts)
        total += sum(float(item.get("estimated_late_penalty_yuan", 0.0) or 0.0) for item in deadline_conflicts)
        best = cls._best_tradeoff(conflict_tradeoffs)
        return {
            "tool_name": "task_penalty_optimizer_tool",
            "current_minute": current_minute,
            "action_finish_minute": finish,
            "action_target_point": [round(target[0], 6), round(target[1], 6)],
            "direct_action_penalties": direct_penalties[:12],
            "daily_rest_conflicts": rest_conflicts[:6],
            "deadline_conflicts": deadline_conflicts[:12],
            "conflict_tradeoffs": conflict_tradeoffs[:12],
            "best_loss_minimization_hint": best,
            "estimated_action_task_penalty_yuan": round(total, 2),
            "risk_flags": cls._risk_flags(total, direct_penalties, deadline_conflicts, conflict_tradeoffs, rest_conflicts),
            "llm_instruction": (
                "If all tasks cannot be satisfied, choose the action with the smallest total penalty after comparing early-leave, late-arrival, not-at-point and periodic-task losses."
            ),
        }

    @classmethod
    def _action_finish_and_target(
        cls,
        current_minute: int,
        current_point: tuple[float, float],
        action: dict[str, Any],
        candidate: dict[str, Any] | None,
    ) -> tuple[int, tuple[float, float]]:
        name = str(action.get("action") or "").lower()
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        if name == "take_order" and isinstance(candidate, dict):
            end = candidate.get("end")
            if cls._point(end):
                return _as_int(candidate.get("finish_minute")) or current_minute, (float(end[0]), float(end[1]))
        if name == "wait":
            return current_minute + max(1, _as_int(params.get("duration_minutes")) or 60), current_point
        if name == "reposition":
            lat = _as_float(params.get("latitude"))
            lng = _as_float(params.get("longitude"))
            if lat is not None and lng is not None:
                travel = _minutes_for_km(_haversine_km(current_point[0], current_point[1], lat, lng))
                return current_minute + travel, (lat, lng)
        return current_minute, current_point

    @classmethod
    def _direct_action_penalties(
        cls,
        start: int,
        finish: int,
        current_point: tuple[float, float],
        target: tuple[float, float],
        action: dict[str, Any],
        profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        action_name = str(action.get("action") or "").lower() if isinstance(action, dict) else ""
        for rule in profile.get("cumulative_time_penalty_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            rate = _as_float(rule.get("rate_yuan_per_minute"))
            window_start = _as_int(rule.get("window_start_minute"))
            window_end = _as_int(rule.get("window_end_minute"))
            point = rule.get("required_point")
            if rate is None or rate <= 0 or window_start is None:
                continue
            if window_end is None or window_end <= window_start:
                window_end = window_start + 1440
            overlap = max(0, min(finish, window_end) - max(start, window_start))
            if overlap <= 0:
                continue
            radius = float(rule.get("radius_km", 1.0) or 1.0)
            at_start_required = cls._point(point) and _haversine_km(current_point[0], current_point[1], float(point[0]), float(point[1])) <= radius
            # Per-minute windows require continuously staying at the required point.
            # Ending an order/reposition at the point does not erase minutes spent away.
            if action_name == "wait" and at_start_required:
                continue
            reports.append(
                {
                    "type": "not_at_required_point_during_penalty_window",
                    "source_text": rule.get("source_text"),
                    "action_name": action_name,
                    "rate_yuan_per_minute": round(rate, 4),
                    "overlap_minutes": overlap,
                    "penalty_yuan": round(overlap * rate, 2),
                }
            )
        return reports

    @staticmethod
    def _daily_rest_conflicts(start: int, finish: int, action: dict[str, Any], time_task_report: dict[str, Any]) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        action_name = str(action.get("action") or "").lower() if isinstance(action, dict) else ""
        wait_duration = max(0, finish - start) if action_name == "wait" else 0
        for item in time_task_report.get("periodic_tasks", []) or []:
            if not isinstance(item, dict) or item.get("type") != "daily_continuous_rest":
                continue
            if item.get("status") == "done":
                continue
            day_end = _as_int(item.get("day_end_minute"))
            remaining = _as_int(item.get("remaining_rest_minutes")) or 0
            latest_start = _as_int(item.get("latest_start_minute"))
            if day_end is None or latest_start is None or remaining <= 0:
                continue
            if action_name == "wait" and wait_duration >= remaining:
                continue
            # If this action finishes after the latest start, there may not be
            # enough continuous time left in the day to finish the rest block.
            insufficient_after_action = finish + remaining > day_end
            crosses_latest_start = start < latest_start < finish
            if insufficient_after_action or crosses_latest_start:
                reports.append(
                    {
                        "type": "daily_continuous_rest_conflict",
                        "task_id": item.get("id"),
                        "required_rest_minutes": item.get("required_rest_minutes"),
                        "remaining_rest_minutes": remaining,
                        "latest_start_minute": latest_start,
                        "action_finish_minute": finish,
                        "estimated_penalty_yuan": float(item.get("estimated_penalty_yuan", 300.0) or 300.0),
                    }
                )
        return reports

    @classmethod
    def _deadline_conflicts(
        cls,
        finish: int,
        target: tuple[float, float],
        profile: dict[str, Any],
        time_task_report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        tasks = []
        tasks.extend(time_task_report.get("urgent_tasks", []) or [])
        tasks.extend(time_task_report.get("overdue_tasks", []) or [])
        tasks.extend(time_task_report.get("progress_board", []) or [])
        seen: set[str] = set()
        for task in tasks:
            if not isinstance(task, dict):
                continue
            task_id = str(task.get("id") or "")
            if task_id in seen:
                continue
            seen.add(task_id)
            point = task.get("point")
            deadline = _as_int(task.get("deadline_minute"))
            if not cls._point(point) or deadline is None or task.get("status") == "done":
                continue
            if task.get("type") == "cumulative_time_penalty_window":
                continue
            travel = _minutes_for_km(_haversine_km(target[0], target[1], float(point[0]), float(point[1])))
            arrival = finish + travel
            late = max(0, arrival - deadline)
            if late <= 0:
                continue
            rate = cls._matching_rate_for_point(profile, point) or cls._default_late_rate(profile)
            reports.append(
                {
                    "task_id": task_id,
                    "task_type": task.get("type"),
                    "deadline_minute": deadline,
                    "arrival_after_action_minute": arrival,
                    "late_minutes": late,
                    "rate_yuan_per_minute": round(rate, 4),
                    "estimated_late_penalty_yuan": round(late * rate, 2),
                    "point": point,
                }
            )
        return sorted(reports, key=lambda item: -float(item.get("estimated_late_penalty_yuan", 0.0) or 0.0))

    @classmethod
    def _conflict_tradeoffs(cls, current: int, current_point: tuple[float, float], profile: dict[str, Any], time_task_report: dict[str, Any]) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        active_windows = [
            item
            for item in (time_task_report.get("progress_board", []) or [])
            if isinstance(item, dict)
            and item.get("status") != "blocked"
            and _as_int(item.get("release_minute")) is not None
            and _as_int(item.get("earliest_minute")) is not None
            and (_as_int(item.get("earliest_minute")) or 0) <= current < (_as_int(item.get("release_minute")) or 0)
            and cls._point(item.get("point"))
        ]
        future_tasks = [
            item
            for item in (time_task_report.get("progress_board", []) or []) + (time_task_report.get("periodic_tasks", []) or [])
            if isinstance(item, dict) and item.get("status") not in {"done", "blocked"} and _as_int(item.get("deadline_minute")) is not None and cls._point(item.get("point"))
        ]
        for active in active_windows:
            active_point = active.get("point")
            active_release = _as_int(active.get("release_minute"))
            active_rate = cls._matching_rate_for_point(profile, active_point) or 0.0
            if active_release is None or not cls._point(active_point):
                continue
            for future in future_tasks:
                if future.get("id") == active.get("id"):
                    continue
                future_point = future.get("point")
                deadline = _as_int(future.get("deadline_minute"))
                if deadline is None or not cls._point(future_point):
                    continue
                if _haversine_km(float(active_point[0]), float(active_point[1]), float(future_point[0]), float(future_point[1])) <= 1.0:
                    continue
                travel = _minutes_for_km(_haversine_km(float(active_point[0]), float(active_point[1]), float(future_point[0]), float(future_point[1])))
                must_leave_by = deadline - travel
                if active_release <= must_leave_by:
                    continue
                early_leave_minutes = max(0, active_release - max(current, must_leave_by))
                late_if_stay = max(0, active_release + travel - deadline)
                future_rate = cls._matching_rate_for_point(profile, future_point) or cls._default_late_rate(profile)
                early_penalty = early_leave_minutes * active_rate
                late_penalty = late_if_stay * future_rate
                reports.append(
                    {
                        "type": "incompatible_time_tasks",
                        "active_task_id": active.get("id"),
                        "future_task_id": future.get("id"),
                        "must_leave_by_minute": must_leave_by,
                        "active_release_minute": active_release,
                        "travel_minutes": travel,
                        "early_leave_minutes": early_leave_minutes,
                        "late_if_stay_minutes": late_if_stay,
                        "early_leave_penalty_yuan": round(early_penalty, 2),
                        "late_if_stay_penalty_yuan": round(late_penalty, 2),
                        "lower_loss_choice": "leave_early" if early_penalty < late_penalty else "stay_and_accept_late_penalty",
                    }
                )
        return sorted(reports, key=lambda item: min(float(item.get("early_leave_penalty_yuan", 0.0) or 0.0), float(item.get("late_if_stay_penalty_yuan", 0.0) or 0.0)))

    @staticmethod
    def _best_tradeoff(tradeoffs: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not tradeoffs:
            return None
        item = tradeoffs[0]
        return {
            "choice": item.get("lower_loss_choice"),
            "active_task_id": item.get("active_task_id"),
            "future_task_id": item.get("future_task_id"),
            "early_leave_penalty_yuan": item.get("early_leave_penalty_yuan"),
            "late_if_stay_penalty_yuan": item.get("late_if_stay_penalty_yuan"),
            "must_leave_by_minute": item.get("must_leave_by_minute"),
        }

    @classmethod
    def _matching_rate_for_point(cls, profile: dict[str, Any], point: Any) -> float | None:
        if not cls._point(point):
            return None
        best: float | None = None
        for rule in profile.get("cumulative_time_penalty_rules", []) or []:
            if not isinstance(rule, dict) or not cls._point(rule.get("required_point")):
                continue
            target = rule.get("required_point")
            if _haversine_km(float(point[0]), float(point[1]), float(target[0]), float(target[1])) <= float(rule.get("radius_km", 1.0) or 1.0):
                rate = _as_float(rule.get("rate_yuan_per_minute"))
                if rate is not None:
                    best = max(best or 0.0, rate)
        return best

    @staticmethod
    def _default_late_rate(profile: dict[str, Any]) -> float:
        values = []
        for card in profile.get("preference_cards", []) or []:
            if isinstance(card, dict) and card.get("severity") in {"critical", "high"}:
                amount = _as_float(card.get("penalty_amount"))
                if amount is not None and amount > 0:
                    values.append(amount / 60.0)
        return max(values) if values else 10.0

    @staticmethod
    def _risk_flags(
        total: float,
        direct: list[dict[str, Any]],
        deadlines: list[dict[str, Any]],
        tradeoffs: list[dict[str, Any]],
        rest_conflicts: list[dict[str, Any]],
    ) -> list[str]:
        flags: list[str] = []
        if total > 0:
            flags.append("task_penalty_expected")
        if rest_conflicts:
            flags.append("daily_rest_task_conflict")
        if direct:
            flags.append("direct_cumulative_task_penalty")
        if deadlines:
            flags.append("future_task_deadline_conflict")
        if total > 0 and tradeoffs:
            flags.append("task_conflict_loss_minimization_needed")
        if total >= 1000:
            flags.append("large_task_penalty_risk")
        return flags

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and _as_float(value[0]) is not None and _as_float(value[1]) is not None
