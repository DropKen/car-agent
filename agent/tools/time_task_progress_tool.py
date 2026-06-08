"""Time-task progress tracker for driver preferences.

The tool is generic: it converts profile commitments into a progress board for
one-month planning and reports urgent/periodic tasks before every action.
"""

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


def _minutes_for_km(km: float, speed_kmh: float = 60.0) -> int:
    return int(math.ceil(max(0.0, km) / max(1.0, speed_kmh) * 60.0))


def _point_to_segment_km(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    lat_scale = 111.0
    lng_scale = 111.0 * max(0.2, math.cos(math.radians(point[0])))
    px, py = point[1] * lng_scale, point[0] * lat_scale
    sx, sy = start[1] * lng_scale, start[0] * lat_scale
    ex, ey = end[1] * lng_scale, end[0] * lat_scale
    dx, dy = ex - sx, ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    return math.hypot(px - (sx + t * dx), py - (sy + t * dy))


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class TimeTaskProgressTool:
    """Builds a time-task progress board and recommends urgent reminders."""

    DEFAULT_BUFFER_MINUTES = 60

    @staticmethod
    def _horizon_days() -> int:
        try:
            return max(1, int(os.environ.get("AGENT_HORIZON_DAYS", "31") or 31))
        except ValueError:
            return 31

    @classmethod
    def progress_report(
        cls,
        *,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tasks = cls._all_tasks(profile, history, current_minute)
        urgent: list[dict[str, Any]] = []
        overdue: list[dict[str, Any]] = []
        for task in tasks:
            task = cls._annotate_task(task, current_minute, lat, lng)
            if task.get("status") == "overdue":
                overdue.append(task)
            elif task.get("is_urgent"):
                urgent.append(task)
        periodic = cls._periodic_reports(profile, history, current_minute, lat, lng)
        for item in periodic:
            if item.get("status") == "overdue":
                overdue.append(item)
            elif item.get("is_urgent"):
                urgent.append(item)
        recommended = cls._recommended_action(urgent + overdue, lat, lng)
        return {
            "tool_name": "time_task_progress_tool",
            "current_minute": current_minute,
            "progress_board": tasks[:24],
            "periodic_tasks": periodic[:16],
            "urgent_tasks": urgent[:12],
            "overdue_tasks": overdue[:12],
            "recommended_action": recommended,
            "llm_review_required": bool(urgent or overdue),
            "reminder": "LLM should compare each action against overdue/urgent one-shot, ordered, periodic and cumulative time tasks.",
        }

    @classmethod
    def _all_tasks(cls, profile: dict[str, Any], history: list[dict[str, Any]], current_minute: int) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                continue
            previous_incomplete = False
            for index, step in enumerate(sequence.get("steps", []) or [], start=1):
                if not isinstance(step, dict):
                    continue
                done = cls._step_done(step, current_minute, history)
                status = "done" if done else ("blocked" if previous_incomplete else "pending")
                task = {
                    "id": f"{sequence.get('id', 'SEQ')}.{step.get('id', index)}",
                    "type": "ordered_step",
                    "parent_id": sequence.get("id"),
                    "source_text": sequence.get("source_text"),
                    "step_index": index,
                    "step": step,
                    "point": step.get("point"),
                    "deadline_minute": _as_int(step.get("deadline_minute")) or _as_int(step.get("hold_until_minute")),
                    "earliest_minute": _as_int(step.get("earliest_minute")),
                    "wait_minutes": _as_int(step.get("wait_minutes")) or (1 if step.get("step_type") == "visit_and_wait" else 0),
                    "status": status,
                    "sequence_order_must_hold": True,
                    "blocked_by_previous_step": previous_incomplete and not done,
                }
                tasks.append(task)
                if not done:
                    previous_incomplete = True
        for event in profile.get("temporary_events", []) or []:
            if isinstance(event, dict):
                pickup_minute = _as_int(event.get("pickup_minute")) or 0
                release_minute = _as_int(event.get("release_minute"))
                event_done = current_minute > (release_minute if release_minute is not None else pickup_minute + 24 * 60)
                tasks.append(
                    {
                        "id": f"TEMP{len(tasks) + 1:03d}",
                        "type": "temporary_event",
                        "point": event.get("pickup_point"),
                        "deadline_minute": pickup_minute,
                        "release_minute": release_minute,
                        "status": "done" if event_done or cls._visited_after(history, event.get("pickup_point"), pickup_minute, 1.0) else "pending",
                    }
                )
        for item in profile.get("required_cargos", []) or []:
            if isinstance(item, dict):
                cargo_id = str(item.get("cargo_id") or "")
                tasks.append(
                    {
                        "id": f"CARGO:{cargo_id}",
                        "type": "required_cargo",
                        "cargo_id": cargo_id,
                        "point": item.get("pickup_point"),
                        "deadline_minute": _as_int(item.get("online_minute")),
                        "status": "done" if cargo_id and cls._cargo_seen(history, cargo_id) else "pending",
                    }
                )
        for visit in profile.get("scheduled_visits", []) or []:
            if not isinstance(visit, dict):
                continue
            day = _as_int(visit.get("day"))
            if day is None:
                continue
            deadline = _as_int(visit.get("arrive_before_minute"))
            target_minute = day * 1440 + (deadline if deadline is not None else 20 * 60)
            point = visit.get("point")
            expired = current_minute > target_minute + 12 * 60
            tasks.append(
                {
                    "id": f"VISIT:{day}:{len(tasks)}",
                    "type": "scheduled_visit",
                    "point": point,
                    "deadline_minute": target_minute,
                    "wait_minutes": _as_int(visit.get("wait_minutes")) or 0,
                    "status": "done" if expired or cls._visited_after(history, point, day * 1440, float(visit.get("radius_km", 1.0) or 1.0)) else "pending",
                }
            )
        for rule in profile.get("cumulative_time_penalty_rules", []) or []:
            if isinstance(rule, dict):
                tasks.append(
                    {
                        "id": str(rule.get("id") or f"TIMEPEN:{len(tasks)}"),
                        "type": "cumulative_time_penalty_window",
                        "source_text": rule.get("source_text"),
                        "point": rule.get("required_point"),
                        "earliest_minute": _as_int(rule.get("window_start_minute")),
                        "deadline_minute": _as_int(rule.get("window_start_minute")),
                        "release_minute": _as_int(rule.get("window_end_minute")),
                        "rate_yuan_per_minute": _as_float(rule.get("rate_yuan_per_minute")),
                        "status": "pending" if current_minute < (_as_int(rule.get("window_end_minute")) or 10**9) else "done",
                    }
                )
        return tasks

    @classmethod
    def _annotate_task(cls, task: dict[str, Any], current_minute: int, lat: float, lng: float) -> dict[str, Any]:
        task = dict(task)
        if task.get("status") in {"done", "blocked"}:
            return task
        point = task.get("point")
        distance = cls._distance_to_point(lat, lng, point)
        travel = None if distance is None else _minutes_for_km(distance)
        deadline = _as_int(task.get("deadline_minute"))
        buffer_minutes = cls.DEFAULT_BUFFER_MINUTES
        leave_by = None if deadline is None or travel is None else deadline - travel - buffer_minutes
        task.update(
            {
                "distance_km": None if distance is None else round(distance, 2),
                "travel_minutes": travel,
                "leave_by_minute": leave_by,
                "is_urgent": leave_by is not None and current_minute >= leave_by,
                "status": "overdue" if deadline is not None and current_minute > deadline and task.get("status") != "done" else task.get("status", "pending"),
            }
        )
        return task

    @classmethod
    def _periodic_reports(cls, profile: dict[str, Any], history: list[dict[str, Any]], current_minute: int, lat: float, lng: float) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        visit = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        if cls._point(visit.get("point")):
            point = visit.get("point")
            assert isinstance(point, (list, tuple))
            target_lat = float(point[0])
            target_lng = float(point[1])
            required_days = _as_int(visit.get("required_days")) or 0
            radius = float(visit.get("radius_km", 1.0) or 1.0)
            done_days = cls._distinct_visit_days(history, point, radius)
            current_day = current_minute // 1440
            distance = cls._distance_to_point(lat, lng, point)
            at_point_now = distance is not None and distance <= radius
            credited_days = set(done_days)
            if at_point_now:
                credited_days.add(current_day)
            horizon_days = cls._horizon_days()
            remaining_days = max(0, horizon_days - current_day)
            missing = max(0, required_days - len(credited_days))
            today_already_credited = current_day in credited_days
            same_day_more_wait_value = 0 if today_already_credited else 1
            next_useful_visit_day = None if missing <= 0 else (current_day + (1 if today_already_credited else 0))
            target_done_by_now = min(required_days, max(1, math.ceil(required_days * (current_day + 1) / horizon_days))) if required_days > 0 else 0
            days_elapsed_without_progress = current_day + 1 if not credited_days else max(0, current_day - max(credited_days))
            behind_pace = missing > 0 and len(credited_days) < target_done_by_now
            low_cost_visit_km = float(os.environ.get("AGENT_LOW_COST_VISIT_KM", "120") or 120)
            should_plan_low_cost_visit = (
                missing > 0
                and not today_already_credited
                and distance is not None
                and distance <= low_cost_visit_km
                and (behind_pace or days_elapsed_without_progress >= 3)
            )
            reports.append(
                {
                    "id": "PERIODIC:visit_frequency",
                    "type": "monthly_visit_frequency",
                    "point": point,
                    "required_days": required_days,
                    "completed_days": sorted(credited_days),
                    "history_completed_days": sorted(done_days),
                    "missing_days": missing,
                    "remaining_calendar_days": remaining_days,
                    "distance_km": None if distance is None else round(distance, 2),
                    "status": "done" if missing <= 0 else ("overdue" if missing > remaining_days else "pending"),
                    "is_urgent": missing > 0 and not today_already_credited and missing >= max(1, remaining_days - 2),
                    "today_already_credited": today_already_credited,
                    "same_day_more_wait_value": same_day_more_wait_value,
                    "next_useful_visit_day": next_useful_visit_day,
                    "release_to_earn_money": missing <= 0 or today_already_credited,
                    "target_completed_days_by_now": target_done_by_now,
                    "days_elapsed_without_progress": days_elapsed_without_progress,
                    "visit_completion_pressure": "behind_pace" if behind_pace else ("urgent" if missing > 0 and missing >= max(1, remaining_days - 2) else "normal"),
                    "should_plan_low_cost_visit": should_plan_low_cost_visit,
                    "suggested_action_if_idle": (
                        {"action": "reposition", "params": {"latitude": target_lat, "longitude": target_lng}, "reason_code": "monthly_visit_low_cost_idle"}
                        if should_plan_low_cost_visit and not at_point_now
                        else ({"action": "wait", "params": {"duration_minutes": 1}, "reason_code": "monthly_visit_credit_today"} if should_plan_low_cost_visit else None)
                    ),
                    "llm_guidance": (
                        "This is a periodic visit task, not a stay task. Complete missing visit days early when idle or when the point is low-cost, "
                        "instead of waiting until month-end. Same-day repeated waiting does not add progress; once today's visit is credited, "
                        "leave for positive action_value orders unless another urgent task requires staying."
                    ),
                }
            )
        rest_report = cls._daily_rest_report(profile, history, current_minute)
        if rest_report:
            reports.append(rest_report)
        off_day_report = cls._monthly_off_day_report(profile, history, current_minute)
        if off_day_report:
            reports.append(off_day_report)
        for rule in profile.get("dynamic_preference_rules", []) or []:
            match = rule.get("match", {}) if isinstance(rule, dict) and isinstance(rule.get("match"), dict) else {}
            periodic = match.get("periodic_stop_required") if isinstance(match.get("periodic_stop_required"), dict) else None
            if not periodic:
                continue
            period_days = _as_int(periodic.get("period_days")) or 0
            min_wait = _as_int(periodic.get("min_wait_minutes")) or 0
            if period_days <= 0:
                continue
            last_wait_day = cls._last_wait_day(history, min_wait)
            current_day = current_minute // 1440
            days_since = None if last_wait_day is None else current_day - last_wait_day
            due = (last_wait_day is None and current_day + 1 >= period_days) or (days_since is not None and days_since >= period_days)
            reports.append(
                {
                    "id": str(rule.get("id") or "PERIODIC:dynamic"),
                    "type": "periodic_stop_required",
                    "period_days": period_days,
                    "min_wait_minutes": min_wait,
                    "last_completed_day": last_wait_day,
                    "days_since_last": days_since,
                    "status": "overdue" if due else "pending",
                    "is_urgent": due,
                    "source_rule_label": rule.get("label"),
                }
            )
        return reports

    @classmethod
    def _monthly_off_day_report(cls, profile: dict[str, Any], history: list[dict[str, Any]], current_minute: int) -> dict[str, Any] | None:
        required = _as_int(profile.get("required_off_days")) or 0
        if required <= 0:
            return None
        current_day = current_minute // 1440
        active_days = cls._active_days(history)
        completed_days = [day for day in range(current_day) if day not in active_days]
        today_has_activity = current_day in active_days
        completed = min(required, len(completed_days))
        missing = max(0, required - completed)
        horizon_days = cls._horizon_days()
        remaining_days = max(0, horizon_days - current_day)
        target_done_by_now = min(required, max(0, math.floor(required * current_day / horizon_days)))
        behind_pace = completed < target_done_by_now
        urgent = missing > 0 and remaining_days <= missing + 1
        can_credit_today = missing > 0 and not today_has_activity
        should_plan_idle_off_day = can_credit_today and (urgent or (behind_pace and current_minute % 1440 >= 18 * 60))
        status = "done" if missing <= 0 else ("overdue" if missing > remaining_days else "pending")
        return {
            "id": "PERIODIC:monthly_full_off_day",
            "type": "monthly_full_off_day_no_order",
            "required_off_days": required,
            "completed_off_days": completed,
            "completed_day_indices": completed_days[:required],
            "missing_days": missing,
            "remaining_calendar_days": remaining_days,
            "today_has_take_or_reposition": today_has_activity,
            "today_can_still_credit": can_credit_today,
            "target_completed_days_by_now": target_done_by_now,
            "off_day_completion_pressure": "urgent" if urgent else ("behind_pace" if behind_pace else "normal"),
            "status": status,
            "is_urgent": urgent and can_credit_today,
            "release_to_earn_money": missing <= 0 or today_has_activity or (not urgent and not should_plan_idle_off_day),
            "should_plan_idle_off_day": should_plan_idle_off_day,
            "suggested_action_if_idle": (
                {"action": "wait", "params": {"duration_minutes": max(1, (current_day + 1) * 1440 - current_minute)}, "reason_code": "monthly_full_off_day_idle"}
                if should_plan_idle_off_day else None
            ),
            "llm_guidance": (
                "Full off-day/no-order preferences are monthly progress tasks, not a command to stop earning from the first day. "
                "Do not block a positive low-risk cargo early in the month. A day only counts if there has been no take_order or reposition that day; "
                "once today has activity, release to earn money and schedule remaining off-days later on low-value or urgent days."
            ),
        }

    @staticmethod
    def _active_days(history: list[dict[str, Any]]) -> set[int]:
        days: set[int] = set()
        for record in history:
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            name = str(action_obj.get("action") or "")
            if name not in {"take_order", "reposition"}:
                continue
            end = TimeTaskProgressTool._record_end_minute(record)
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            if end is not None:
                start = max(0, end - max(1, elapsed))
                day = start // 1440
                while day <= end // 1440:
                    days.add(day)
                    day += 1
        return days

    @staticmethod
    def _record_end_minute(record: dict[str, Any]) -> int | None:
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        return _as_int(record.get("simulation_end_minute", result.get("simulation_progress_minutes")))

    @classmethod
    def _daily_rest_report(cls, profile: dict[str, Any], history: list[dict[str, Any]], current_minute: int) -> dict[str, Any] | None:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        hours = _as_float(rest.get("hours"))
        if hours is None or hours <= 0:
            return None
        required = int(math.ceil(hours * 60))
        current_day = current_minute // 1440
        day_start = current_day * 1440
        day_end = day_start + 1440
        max_wait = cls._max_continuous_wait_minutes_for_day(history, current_day)
        remaining = max(0, required - max_wait)
        latest_start = day_end - required
        must_start_now = current_minute >= latest_start and remaining > 0
        impossible_today = current_minute + remaining > day_end
        status = "done" if remaining <= 0 else ("overdue" if impossible_today else "pending")
        return {
            "id": f"REST:{current_day}",
            "type": "daily_continuous_rest",
            "required_rest_minutes": required,
            "completed_continuous_rest_minutes": max_wait,
            "remaining_rest_minutes": remaining,
            "day_start_minute": day_start,
            "day_end_minute": day_end,
            "latest_start_minute": latest_start,
            "status": status,
            "is_urgent": must_start_now or impossible_today,
            "release_to_earn_money": remaining <= 0,
            "estimated_penalty_yuan": 0.0 if remaining <= 0 else cls._penalty_for_types(profile, {"rest_or_no_action"}, 300.0),
            "llm_guidance": (
                "Daily rest is a repeated required task. If remaining_rest_minutes > 0 and latest_start_minute has passed, "
                "wait long enough to complete it unless a higher mandatory penalty conflict exists."
            ),
        }

    @staticmethod
    def _max_continuous_wait_minutes_for_day(history: list[dict[str, Any]], day: int) -> int:
        day_start = day * 1440
        day_end = day_start + 1440
        waits: list[tuple[int, int]] = []
        for record in history:
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "wait":
                continue
            end = TimeTaskProgressTool._record_end_minute(record)
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            if end is None or elapsed <= 0:
                continue
            start = end - elapsed
            overlap_start = max(start, day_start)
            overlap_end = min(end, day_end)
            if overlap_end > overlap_start:
                waits.append((overlap_start, overlap_end))
        if not waits:
            return 0
        waits.sort()
        max_len = 0
        cur_start, cur_end = waits[0]
        for start, end in waits[1:]:
            if start <= cur_end + 1:
                cur_end = max(cur_end, end)
            else:
                max_len = max(max_len, cur_end - cur_start)
                cur_start, cur_end = start, end
        max_len = max(max_len, cur_end - cur_start)
        return max_len

    @staticmethod
    def _penalty_for_types(profile: dict[str, Any], types: set[str], default: float) -> float:
        values: list[float] = []
        for card in profile.get("preference_cards", []) or []:
            if not isinstance(card, dict):
                continue
            card_types = set(card.get("types") or [])
            if card_types & types:
                amount = _as_float(card.get("penalty_amount"))
                if amount is not None and amount > 0:
                    values.append(amount)
        return max(values) if values else default

    @classmethod
    def _recommended_action(cls, tasks: list[dict[str, Any]], lat: float, lng: float) -> dict[str, Any] | None:
        rest_tasks = [task for task in tasks if task.get("type") == "daily_continuous_rest" and task.get("status") != "done"]
        if rest_tasks:
            task = sorted(rest_tasks, key=lambda item: 0 if item.get("status") == "overdue" else 1)[0]
            wait = max(30, min(240, _as_int(task.get("remaining_rest_minutes")) or _as_int(task.get("required_rest_minutes")) or 180))
            return {"action": "wait", "params": {"duration_minutes": wait}, "reason_code": "daily_rest_required", "task_id": task.get("id")}
        actionable = [task for task in tasks if task.get("status") != "done" and cls._point(task.get("point"))]
        if not actionable:
            return None
        task = min(actionable, key=lambda item: _as_int(item.get("leave_by_minute")) or 10**9)
        point = task.get("point")
        if not cls._point(point):
            return None
        assert isinstance(point, (list, tuple))
        distance = _haversine_km(lat, lng, float(point[0]), float(point[1]))
        if distance > 1.0:
            return {"action": "reposition", "params": {"latitude": float(point[0]), "longitude": float(point[1])}, "reason_code": "time_task_progress_due", "task_id": task.get("id")}
        wait = max(1, min(240, _as_int(task.get("wait_minutes")) or 30))
        return {"action": "wait", "params": {"duration_minutes": wait}, "reason_code": "time_task_progress_wait", "task_id": task.get("id")}

    @classmethod
    def _step_done(cls, step: dict[str, Any], current_minute: int, history: list[dict[str, Any]]) -> bool:
        step_type = str(step.get("step_type") or "")
        if step_type == "take_cargo":
            cargo_id = str(step.get("cargo_id") or "")
            return bool(cargo_id) and cls._cargo_seen(history, cargo_id)
        point = step.get("point")
        earliest = _as_int(step.get("earliest_minute")) or 0
        if not cls._point(point):
            return False
        if step_type == "stay_until":
            hold = _as_int(step.get("hold_until_minute"))
            if hold is not None and current_minute >= hold:
                return True
            hold_start = _as_int(step.get("deadline_minute")) or earliest
            return hold is not None and current_minute >= hold and cls._waited_near_after(
                history,
                point,
                hold_start,
                max(1, hold - hold_start),
                float(step.get("radius_km", 1.0) or 1.0),
            )
        wait = _as_int(step.get("wait_minutes")) or (1 if step_type == "visit_and_wait" else 0)
        if step_type == "visit_and_wait" and wait > 0 and current_minute >= earliest + wait:
            return True
        if wait > 1:
            return cls._waited_near_after(history, point, earliest, wait, float(step.get("radius_km", 1.0) or 1.0))
        return cls._visited_after(history, point, earliest, float(step.get("radius_km", 1.0) or 1.0))

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and _as_float(value[0]) is not None and _as_float(value[1]) is not None

    @classmethod
    def _distance_to_point(cls, lat: float, lng: float, point: Any) -> float | None:
        if not cls._point(point):
            return None
        return _haversine_km(lat, lng, float(point[0]), float(point[1]))

    @classmethod
    def _visited_after(cls, history: list[dict[str, Any]], point: Any, minute: int, radius_km: float) -> bool:
        if not cls._point(point):
            return False
        for record in history:
            end = TimeTaskProgressTool._record_end_minute(record)
            if end is not None and end < minute:
                continue
            if cls._record_passes_point(record, point, radius_km):
                return True
        return False

    @classmethod
    def _waited_near_after(cls, history: list[dict[str, Any]], point: Any, minute: int, wait_minutes: int, radius_km: float) -> bool:
        if not cls._point(point):
            return False
        run = 0
        for record in history:
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "wait":
                pos = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
                pos_lat = _as_float(pos.get("lat")) if isinstance(pos, dict) else None
                pos_lng = _as_float(pos.get("lng")) if isinstance(pos, dict) else None
                point_lat = _as_float(point[0])
                point_lng = _as_float(point[1])
                if pos_lat is None or pos_lng is None or point_lat is None or point_lng is None:
                    run = 0
                else:
                    if _haversine_km(pos_lat, pos_lng, point_lat, point_lng) > radius_km:
                        run = 0
                continue
            end = TimeTaskProgressTool._record_end_minute(record)
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            if end is None or end < minute:
                continue
            start = max(0, end - max(0, elapsed))
            pos_before = record.get("position_before") if isinstance(record.get("position_before"), dict) else {}
            pos_after = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            before_lat = _as_float(pos_before.get("lat")) if isinstance(pos_before, dict) else None
            before_lng = _as_float(pos_before.get("lng")) if isinstance(pos_before, dict) else None
            after_lat = _as_float(pos_after.get("lat")) if isinstance(pos_after, dict) else None
            after_lng = _as_float(pos_after.get("lng")) if isinstance(pos_after, dict) else None
            point_lat = _as_float(point[0])
            point_lng = _as_float(point[1])
            if None in (before_lat, before_lng, after_lat, after_lng, point_lat, point_lng):
                near = False
            else:
                assert before_lat is not None and before_lng is not None and after_lat is not None and after_lng is not None
                assert point_lat is not None and point_lng is not None
                near = (
                    _haversine_km(before_lat, before_lng, point_lat, point_lng) <= radius_km
                    and _haversine_km(after_lat, after_lng, point_lat, point_lng) <= radius_km
                )
            if near:
                run += max(0, end - max(start, minute))
                if run >= wait_minutes:
                    return True
            else:
                run = 0
        return False

    @staticmethod
    def _cargo_seen(history: list[dict[str, Any]], cargo_id: str) -> bool:
        for record in history:
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            params = action_obj.get("params", {}) if isinstance(action_obj.get("params"), dict) else {}
            if str(params.get("cargo_id") or "") == str(cargo_id) and (record.get("accepted") is True or (record.get("result", {}) if isinstance(record.get("result"), dict) else {}).get("accepted") is True):
                return True
        return False

    @classmethod
    def _distinct_visit_days(cls, history: list[dict[str, Any]], point: Any, radius_km: float) -> set[int]:
        days: set[int] = set()
        if not cls._point(point):
            return days
        for record in history:
            end = TimeTaskProgressTool._record_end_minute(record)
            if end is None:
                continue
            if cls._record_passes_point(record, point, radius_km):
                days.add(end // 1440)
        return days

    @staticmethod
    def _last_wait_day(history: list[dict[str, Any]], min_wait_minutes: int) -> int | None:
        last: int | None = None
        for record in history:
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "wait":
                continue
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            end = TimeTaskProgressTool._record_end_minute(record)
            if end is not None and elapsed >= min_wait_minutes:
                last = end // 1440
        return last

    @classmethod
    def _record_passes_point(cls, record: dict[str, Any], point: Any, radius_km: float) -> bool:
        if not cls._point(point):
            return False
        before = record.get("position_before") if isinstance(record.get("position_before"), dict) else {}
        after = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
        point_lat = _as_float(point[0])
        point_lng = _as_float(point[1])
        before_lat = _as_float(before.get("lat")) if isinstance(before, dict) else None
        before_lng = _as_float(before.get("lng")) if isinstance(before, dict) else None
        after_lat = _as_float(after.get("lat")) if isinstance(after, dict) else None
        after_lng = _as_float(after.get("lng")) if isinstance(after, dict) else None
        if None in (point_lat, point_lng, before_lat, before_lng, after_lat, after_lng):
            return False
        assert point_lat is not None and point_lng is not None
        assert before_lat is not None and before_lng is not None and after_lat is not None and after_lng is not None
        target = (point_lat, point_lng)
        start = (before_lat, before_lng)
        end = (after_lat, after_lng)
        return (
            _haversine_km(start[0], start[1], target[0], target[1]) <= radius_km
            or _haversine_km(end[0], end[1], target[0], target[1]) <= radius_km
            or _point_to_segment_km(target, start, end) <= radius_km
        )
