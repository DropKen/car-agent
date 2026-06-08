"""Rolling task calendar for preference-derived commitments.

The tool keeps preference tasks out of immediate decisions until their
activation window. Near the task it turns them into concrete actions or
candidate blocks, so profitable orders do not accidentally consume the time
reserved for a fixed-date commitment.
"""

from __future__ import annotations

import math
import os
from typing import Any


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def _minutes_for_km(km: float, speed_kmh: float = 60.0) -> int:
    return int(math.ceil(max(0.0, km) / max(1.0, speed_kmh) * 60.0))


class TaskCalendarTool:
    """Creates a rolling calendar from parsed preference tasks."""

    DEFAULT_BUFFER_MINUTES = 90

    @classmethod
    def calendar_report(
        cls,
        *,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        fixed = cls._fixed_tasks(profile, history, current_minute, lat, lng)
        monthly = cls._monthly_off_day_plan(profile, history, current_minute)
        no_action = cls._no_action_window_plan(profile, current_minute)
        daily_rest = cls._daily_rest_plan(profile, history, current_minute)
        visit_plan = cls._monthly_visit_plan(profile, history, current_minute, lat, lng)
        region_cargo_plan = cls._required_region_cargo_plan(profile, current_minute, lat, lng)
        active = [item for item in fixed if item.get("active_now")]
        next_task = min(fixed, key=lambda item: item.get("target_minute", 10**12), default=None)
        action = cls._recommended_action(active, monthly, no_action, daily_rest, visit_plan, region_cargo_plan, current_minute, lat, lng)
        candidate_deadline = cls._candidate_finish_deadline(active, next_task, monthly, no_action, daily_rest, visit_plan, region_cargo_plan, current_minute)
        return {
            "tool_name": "task_calendar_tool",
            "current_minute": current_minute,
            "active_tasks": active[:8],
            "next_task": next_task,
            "monthly_plan": monthly,
            "no_action_window_plan": no_action,
            "daily_rest_plan": daily_rest,
            "monthly_visit_plan": visit_plan,
            "required_region_cargo_plan": region_cargo_plan,
            "recommended_action": action,
            "candidate_finish_deadline": candidate_deadline,
            "llm_review_required": bool(
                active
                or monthly.get("lock_today")
                or no_action.get("inside_now")
                or daily_rest.get("lock_now")
                or visit_plan.get("active_today")
                or region_cargo_plan.get("active_today")
            ),
        }

    @classmethod
    def blocks_candidate(cls, report: dict[str, Any], finish_minute: int) -> bool:
        deadline = _as_int(report.get("candidate_finish_deadline")) if isinstance(report, dict) else None
        return deadline is not None and finish_minute > deadline

    @classmethod
    def _fixed_tasks(
        cls,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
        current_minute: int,
        lat: float,
        lng: float,
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                continue
            for step in sequence.get("steps", []) or []:
                if not isinstance(step, dict):
                    continue
                if cls._step_done(step, current_minute, history):
                    continue
                item = cls._task_from_step(step, sequence, current_minute, lat, lng)
                if item:
                    tasks.append(item)
                break
        for visit in profile.get("scheduled_visits", []) or []:
            if not isinstance(visit, dict):
                continue
            day = _as_int(visit.get("day"))
            point = visit.get("point")
            if day is None or not cls._point(point):
                continue
            deadline = day * 1440 + (_as_int(visit.get("arrive_before_minute")) or 20 * 60)
            if current_minute > deadline + 12 * 60:
                continue
            wait_minutes = max(1, _as_int(visit.get("wait_minutes")) or 1)
            radius_km = float(visit.get("radius_km", 1.5) or 1.5)
            if wait_minutes > 1:
                completed = cls._waited_near_after(history, point, day * 1440, wait_minutes, radius_km)
            else:
                completed = cls._visited_after(history, point, day * 1440, radius_km)
            if completed:
                continue
            item = cls._task_from_point(
                task_type="scheduled_visit",
                task_id=f"VISIT:{day}",
                source_text=str(visit.get("source_text") or ""),
                point=point,
                target_minute=deadline,
                wait_minutes=wait_minutes,
                current_minute=current_minute,
                lat=lat,
                lng=lng,
                buffer_minutes=cls.DEFAULT_BUFFER_MINUTES,
            )
            if item:
                tasks.append(item)
        tasks.sort(key=lambda item: item.get("target_minute", 10**12))
        return tasks

    @classmethod
    def _task_from_step(
        cls,
        step: dict[str, Any],
        sequence: dict[str, Any],
        current_minute: int,
        lat: float,
        lng: float,
    ) -> dict[str, Any] | None:
        point = step.get("point")
        if not cls._point(point):
            return None
        step_type = str(step.get("step_type") or "")
        earliest = _as_int(step.get("earliest_minute"))
        hold_until = _as_int(step.get("hold_until_minute"))
        if step_type == "stay_until" and hold_until is not None and current_minute >= hold_until:
            return None
        deadline = _as_int(step.get("deadline_minute")) or hold_until
        target = earliest if step_type == "visit_and_wait" and earliest is not None else (deadline if deadline is not None else earliest)
        if target is None:
            return None
        return cls._task_from_point(
            task_type="ordered_step",
            task_id=f"{sequence.get('id', 'SEQ')}:{step.get('id', 'step')}",
            source_text=str(sequence.get("source_text") or ""),
            point=point,
            target_minute=target,
            wait_minutes=_as_int(step.get("wait_minutes")) or (60 if step_type == "stay_until" else 1),
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            buffer_minutes=max(cls.DEFAULT_BUFFER_MINUTES, _as_int(sequence.get("buffer_minutes")) or 0),
        )

    @classmethod
    def _task_from_point(
        cls,
        *,
        task_type: str,
        task_id: str,
        source_text: str,
        point: Any,
        target_minute: int,
        wait_minutes: int,
        current_minute: int,
        lat: float,
        lng: float,
        buffer_minutes: int,
    ) -> dict[str, Any] | None:
        if not cls._point(point):
            return None
        p = (float(point[0]), float(point[1]))
        travel = _minutes_for_km(_haversine_km(lat, lng, p[0], p[1]))
        leave_by = target_minute - travel - buffer_minutes
        notice = cls._notice_minutes()
        return {
            "type": task_type,
            "id": task_id,
            "source_text": source_text,
            "point": [p[0], p[1]],
            "target_minute": target_minute,
            "leave_by_minute": leave_by,
            "travel_minutes": travel,
            "wait_minutes": max(1, wait_minutes),
            "active_now": current_minute >= leave_by or target_minute - current_minute <= notice,
            "too_far_future": target_minute - current_minute > notice,
        }

    @classmethod
    def _monthly_off_day_plan(cls, profile: dict[str, Any], history: list[dict[str, Any]], current_minute: int) -> dict[str, Any]:
        required = max(0, _as_int(profile.get("required_off_days")) or 0)
        if required <= 0:
            return {"required_off_days": 0, "lock_today": False}
        horizon = cls._horizon_days()
        current_day = current_minute // 1440
        active_days = cls._active_days(history)
        completed = sum(1 for day in range(current_day) if day not in active_days)
        remaining_needed = max(0, required - completed)
        remaining_days = max(1, horizon - current_day)
        planned_days = cls._planned_quota_days(required, horizon)
        due_by_now = min(required, sum(1 for day in planned_days if day <= current_day))
        behind_plan = completed < due_by_now
        no_activity_today = current_day not in active_days
        urgent = remaining_days <= remaining_needed + 1
        # Keep the plan advisory unless today is explicitly selected or the
        # remaining calendar can no longer absorb the quota.
        lock_today = remaining_needed > 0 and no_activity_today and (current_day in planned_days or behind_plan or urgent)
        future_lock_days = [day for day in planned_days if day >= current_day and day not in active_days]
        next_lock_day = min(future_lock_days, default=current_day if urgent else None)
        deadline = current_minute if lock_today else None
        return {
            "required_off_days": required,
            "completed_off_days": completed,
            "remaining_needed": remaining_needed,
            "planned_day_indices": planned_days,
            "lock_today": lock_today,
            "next_lock_day": next_lock_day,
            "candidate_finish_deadline": deadline,
            "reason": "planned_or_urgent_monthly_off_day" if lock_today else "not_due",
        }

    @classmethod
    def _no_action_window_plan(cls, profile: dict[str, Any], current_minute: int) -> dict[str, Any]:
        windows = cls._no_action_windows(profile)
        if not windows:
            return {"inside_now": False}
        minute = current_minute % 1440
        current_day = current_minute // 1440
        next_start: int | None = None
        active_end: int | None = None
        for start, end in windows:
            if cls._inside_window(minute, start, end):
                active_end = current_day * 1440 + end
                if end <= start and minute >= start:
                    active_end += 1440
                break
            abs_start = current_day * 1440 + start
            if abs_start <= current_minute:
                abs_start += 1440
            if next_start is None or abs_start < next_start:
                next_start = abs_start
        if active_end is not None:
            return {
                "inside_now": True,
                "window_end_minute": active_end,
                "candidate_finish_deadline": current_minute,
            }
        deadline = next_start if next_start is not None and next_start - current_minute <= 12 * 60 else None
        return {
            "inside_now": False,
            "next_start_minute": next_start,
            "candidate_finish_deadline": deadline,
        }

    @classmethod
    def _daily_rest_plan(cls, profile: dict[str, Any], history: list[dict[str, Any]], current_minute: int) -> dict[str, Any]:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        if _as_int(rest.get("window_start_minute")) is not None and _as_int(rest.get("window_end_minute")) is not None:
            return {"required_minutes": 0, "lock_now": False, "reason": "handled_as_no_action_window"}
        try:
            hours = float(rest.get("hours")) if rest.get("hours") is not None else 0.0
        except (TypeError, ValueError):
            hours = 0.0
        if hours <= 0:
            return {"required_minutes": 0, "lock_now": False}
        required = int(math.ceil(hours * 60))
        current_day = current_minute // 1440
        day_end = (current_day + 1) * 1440
        completed = cls._max_wait_for_day(history, current_day)
        remaining = max(0, required - completed)
        latest_start = day_end - required
        minute_of_day = current_minute % 1440
        preferred_start = current_day * 1440 + cls._preferred_rest_start_minute(required)
        buffer_minutes = int(os.environ.get("AGENT_TASK_CALENDAR_REST_BUFFER_MINUTES", "120") or 120)
        lock_start = min(latest_start - buffer_minutes, preferred_start)
        lock_now = remaining > 0 and current_minute >= lock_start
        # Flexible "continuous rest N hours" is planned as a fixed evening
        # block. For long rest requirements, start protecting the evening block
        # around noon so a midday long-haul order cannot consume the whole day.
        protect_from = int(os.environ.get("AGENT_TASK_CALENDAR_REST_PROTECT_FROM_8H_MINUTE", str(12 * 60)) or 12 * 60) if required >= 8 * 60 else 14 * 60
        candidate_deadline = preferred_start if remaining > 0 and minute_of_day >= protect_from else None
        return {
            "required_minutes": required,
            "completed_continuous_minutes": completed,
            "remaining_minutes": remaining,
            "latest_start_minute": latest_start,
            "preferred_start_minute": preferred_start,
            "candidate_finish_deadline": candidate_deadline,
            "lock_now": lock_now,
        }

    @classmethod
    def _required_region_cargo_plan(
        cls,
        profile: dict[str, Any],
        current_minute: int,
        lat: float,
        lng: float,
    ) -> dict[str, Any]:
        rule = profile.get("required_region_cargo_days", {}) if isinstance(profile.get("required_region_cargo_days"), dict) else {}
        required = max(0, _as_int(rule.get("min_days")) or 0)
        point = rule.get("point")
        region = str(rule.get("region") or "")
        if required <= 0 or not region or not cls._point(point):
            return {"required_days": 0, "active_today": False}
        current_day = current_minute // 1440
        completed_days_raw = rule.get("completed_day_indices")
        completed_days = {int(day) for day in completed_days_raw or [] if _as_int(day) is not None} if isinstance(completed_days_raw, list) else set()
        completed = max(len(completed_days), _as_int(rule.get("completed_days")) or 0)
        remaining = max(0, required - completed)
        horizon = cls._horizon_days()
        planned_days = cls._planned_quota_days(required, horizon)
        due_by_now = min(required, sum(1 for day in planned_days if day <= current_day))
        behind = completed < due_by_now
        urgent = horizon - current_day <= remaining + 5
        already_done_today = current_day in completed_days
        active_today = remaining > 0 and not already_done_today and (current_day in planned_days or behind or urgent)
        target = (float(point[0]), float(point[1]))
        distance = _haversine_km(lat, lng, target[0], target[1])
        max_distance = float(os.environ.get("AGENT_TASK_CALENDAR_REGION_MAX_KM", "260") or 260)
        if active_today and not urgent and distance > max_distance:
            active_today = False
        return {
            "type": "required_region_cargo_days",
            "region": region,
            "required_days": required,
            "completed_days": completed,
            "completed_day_indices": sorted(completed_days),
            "remaining_days": remaining,
            "planned_day_indices": planned_days,
            "active_today": active_today,
            "point": [target[0], target[1]],
            "distance_km": round(distance, 2),
            "near_region_point": distance <= float(os.environ.get("AGENT_TASK_CALENDAR_REGION_NEAR_KM", "35") or 35),
            "candidate_finish_deadline": current_day * 1440 + 22 * 60 if active_today else None,
            "behind_plan": behind,
            "urgent": urgent,
        }

    @classmethod
    def _monthly_visit_plan(
        cls,
        profile: dict[str, Any],
        history: list[dict[str, Any]],
        current_minute: int,
        lat: float,
        lng: float,
    ) -> dict[str, Any]:
        visit = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        required = max(0, _as_int(visit.get("required_days")) or 0)
        point = visit.get("point")
        if required <= 0 or not cls._point(point):
            return {"required_days": 0, "active_today": False}
        radius = float(visit.get("radius_km", 1.0) or 1.0)
        current_day = current_minute // 1440
        visited_days = cls._visited_days_near(history, point, radius)
        at_point = _haversine_km(lat, lng, float(point[0]), float(point[1])) <= radius
        credited = len(visited_days | ({current_day} if at_point else set()))
        remaining = max(0, required - credited)
        horizon = cls._horizon_days()
        planned_days = cls._planned_quota_days(required, horizon)
        due_by_now = min(required, sum(1 for day in planned_days if day <= current_day))
        behind = credited < due_by_now
        urgent = horizon - current_day <= remaining + 3
        active_today = remaining > 0 and current_day not in visited_days and (behind or urgent)
        target = (float(point[0]), float(point[1]))
        distance = _haversine_km(lat, lng, target[0], target[1])
        max_distance = float(os.environ.get("AGENT_TASK_CALENDAR_VISIT_MAX_KM", "180") or 180)
        if active_today and not urgent and distance > max_distance:
            active_today = False
        return {
            "required_days": required,
            "credited_days": credited,
            "remaining_days": remaining,
            "visited_day_indices": sorted(visited_days),
            "planned_day_indices": planned_days,
            "active_today": active_today,
            "at_point_now": at_point,
            "point": [target[0], target[1]],
            "distance_km": round(distance, 2),
            "candidate_finish_deadline": (current_day * 1440 + 20 * 60) if active_today and urgent else None,
            "behind_plan": behind,
            "urgent": urgent,
        }

    @staticmethod
    def _recommended_action(
        active: list[dict[str, Any]],
        monthly: dict[str, Any],
        no_action: dict[str, Any],
        daily_rest: dict[str, Any],
        visit_plan: dict[str, Any],
        region_cargo_plan: dict[str, Any],
        current_minute: int,
        lat: float,
        lng: float,
    ) -> dict[str, Any] | None:
        if no_action.get("inside_now"):
            end = _as_int(no_action.get("window_end_minute")) or current_minute + 60
            return {"action": "wait", "params": {"duration_minutes": max(1, min(720, end - current_minute))}, "reason_code": "task_calendar_no_action_window"}
        if active:
            task = min(active, key=lambda item: item.get("leave_by_minute", item.get("target_minute", 10**12)))
            point = task.get("point")
            if isinstance(point, list) and len(point) >= 2:
                dist = _haversine_km(lat, lng, float(point[0]), float(point[1]))
                if dist > 1.2:
                    return {"action": "reposition", "params": {"latitude": float(point[0]), "longitude": float(point[1])}, "reason_code": "task_calendar_reposition", "task_id": task.get("id")}
            target = _as_int(task.get("target_minute"))
            if target is not None and current_minute < target:
                return {"action": "wait", "params": {"duration_minutes": max(1, min(240, target - current_minute))}, "reason_code": "task_calendar_wait_until_task", "task_id": task.get("id")}
            return {"action": "wait", "params": {"duration_minutes": max(1, min(240, _as_int(task.get("wait_minutes")) or 60))}, "reason_code": "task_calendar_task_wait", "task_id": task.get("id")}
        if monthly.get("lock_today"):
            end_of_day = (current_minute // 1440 + 1) * 1440
            return {"action": "wait", "params": {"duration_minutes": max(1, min(720, end_of_day - current_minute))}, "reason_code": "task_calendar_monthly_off_day"}
        if visit_plan.get("active_today"):
            point = visit_plan.get("point")
            if isinstance(point, list) and len(point) >= 2 and not visit_plan.get("at_point_now"):
                return {"action": "reposition", "params": {"latitude": float(point[0]), "longitude": float(point[1])}, "reason_code": "task_calendar_monthly_visit_position"}
            return {"action": "wait", "params": {"duration_minutes": 1}, "reason_code": "task_calendar_monthly_visit_credit"}
        if region_cargo_plan.get("active_today"):
            point = region_cargo_plan.get("point")
            if isinstance(point, list) and len(point) >= 2 and not region_cargo_plan.get("near_region_point"):
                return {"action": "reposition", "params": {"latitude": float(point[0]), "longitude": float(point[1])}, "reason_code": "task_calendar_region_cargo_position"}
        if daily_rest.get("lock_now"):
            remaining = max(1, _as_int(daily_rest.get("remaining_minutes")) or _as_int(daily_rest.get("required_minutes")) or 60)
            end_of_day = (current_minute // 1440 + 1) * 1440
            return {"action": "wait", "params": {"duration_minutes": max(1, min(720, remaining, end_of_day - current_minute))}, "reason_code": "task_calendar_daily_rest"}
        return None

    @staticmethod
    def _candidate_finish_deadline(
        active: list[dict[str, Any]],
        next_task: dict[str, Any] | None,
        monthly: dict[str, Any],
        no_action: dict[str, Any],
        daily_rest: dict[str, Any],
        visit_plan: dict[str, Any],
        region_cargo_plan: dict[str, Any],
        current_minute: int,
    ) -> int | None:
        guarded = active or ([next_task] if isinstance(next_task, dict) and (next_task.get("target_minute", 10**12) - current_minute) <= TaskCalendarTool._notice_minutes() else [])
        leave_values = [_as_int(item.get("leave_by_minute")) for item in guarded if isinstance(item, dict)]
        leave_values = [value for value in leave_values if value is not None]
        for plan in (monthly, no_action, daily_rest, visit_plan, region_cargo_plan):
            value = _as_int(plan.get("candidate_finish_deadline")) if isinstance(plan, dict) else None
            if value is not None:
                leave_values.append(value)
        return min(leave_values) if leave_values else None

    @staticmethod
    def _inside_window(minute_of_day: int, start: int, end: int) -> bool:
        if start <= end:
            return start <= minute_of_day < end
        return minute_of_day >= start or minute_of_day < end

    @classmethod
    def _no_action_windows(cls, profile: dict[str, Any]) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        start = _as_int(rest.get("window_start_minute"))
        end = _as_int(rest.get("window_end_minute"))
        if start is not None and end is not None and start != end:
            windows.append((max(0, min(1439, start)), max(1, min(1440, end))))
        for rule in profile.get("dynamic_preference_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}
            window = match.get("daily_time_window") if isinstance(match.get("daily_time_window"), dict) else {}
            start = _as_int(window.get("start_minute_of_day"))
            end = _as_int(window.get("end_minute_of_day"))
            source_text = str(rule.get("source_text") or "")
            if start is None or end is None or start == end:
                continue
            if any(word in source_text for word in ("不接单", "不空驶", "不空车", "睡觉", "禁行", "不赶路")):
                windows.append((max(0, min(1439, start)), max(1, min(1440, end))))
        result: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for window in windows:
            if window not in seen:
                seen.add(window)
                result.append(window)
        return result

    @staticmethod
    def _notice_minutes() -> int:
        try:
            return max(6 * 60, int(os.environ.get("AGENT_TASK_CALENDAR_NOTICE_MINUTES", str(36 * 60)) or 36 * 60))
        except ValueError:
            return 36 * 60

    @staticmethod
    def _horizon_days() -> int:
        try:
            return max(1, int(os.environ.get("AGENT_HORIZON_DAYS", "31") or 31))
        except ValueError:
            return 31

    @staticmethod
    def _planned_quota_days(required: int, horizon: int) -> list[int]:
        if required <= 0:
            return []
        return sorted({min(horizon - 1, max(0, round((idx + 1) * horizon / (required + 1)))) for idx in range(required)})

    @staticmethod
    def _preferred_rest_start_minute(required_minutes: int) -> int:
        if required_minutes >= 8 * 60:
            return max(0, min(23 * 60, int(os.environ.get("AGENT_TASK_CALENDAR_REST_START_8H_MINUTE", str(16 * 60)) or 16 * 60)))
        if required_minutes >= 4 * 60:
            return max(0, min(23 * 60, int(os.environ.get("AGENT_TASK_CALENDAR_REST_START_4H_MINUTE", str(20 * 60)) or 20 * 60)))
        return 21 * 60

    @staticmethod
    def _max_wait_for_day(history: list[dict[str, Any]], day: int) -> int:
        day_start = day * 1440
        day_end = (day + 1) * 1440
        best = 0
        run = 0
        last_end: int | None = None
        for record in history:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") != "wait":
                if last_end is not None:
                    run = 0
                    last_end = None
                continue
            end = TaskCalendarTool._record_end_minute(record)
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            if end is None or elapsed <= 0:
                continue
            start = max(0, end - elapsed)
            overlap = max(0, min(end, day_end) - max(start, day_start))
            if overlap <= 0:
                continue
            if last_end is not None and start > last_end + 1:
                run = 0
            run += overlap
            best = max(best, run)
            last_end = end
        return best

    @staticmethod
    def _visited_days_near(history: list[dict[str, Any]], point: Any, radius_km: float) -> set[int]:
        days: set[int] = set()
        if not TaskCalendarTool._point(point):
            return days
        p = (float(point[0]), float(point[1]))
        for record in history:
            if not isinstance(record, dict):
                continue
            end = TaskCalendarTool._record_end_minute(record)
            if end is None:
                continue
            pos = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            try:
                if _haversine_km(float(pos.get("lat")), float(pos.get("lng")), p[0], p[1]) <= radius_km:
                    days.add(end // 1440)
            except (TypeError, ValueError):
                continue
        return days

    @staticmethod
    def _active_days(history: list[dict[str, Any]]) -> set[int]:
        active: set[int] = set()
        for record in history:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            name = str(act.get("action") or "")
            if name not in {"take_order", "reposition"}:
                continue
            end = TaskCalendarTool._record_end_minute(record)
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 1
            if end is None:
                continue
            start = max(0, end - max(1, elapsed))
            day = start // 1440
            last_active_minute = max(start, end - 1)
            while day <= last_active_minute // 1440:
                active.add(day)
                day += 1
        return active

    @staticmethod
    def _step_done(step: dict[str, Any], current_minute: int, history: list[dict[str, Any]]) -> bool:
        point = step.get("point")
        if not TaskCalendarTool._point(point):
            return False
        earliest = _as_int(step.get("earliest_minute")) or 0
        step_type = str(step.get("step_type") or "")
        if step_type == "stay_until":
            hold_until = _as_int(step.get("hold_until_minute"))
            if hold_until is None or current_minute < hold_until:
                return False
            return True
        if step_type == "visit_and_wait":
            wait_minutes = max(1, _as_int(step.get("wait_minutes")) or 1)
            if current_minute >= earliest + wait_minutes:
                return True
            return TaskCalendarTool._waited_near_after(history, point, earliest, max(1, _as_int(step.get("wait_minutes")) or 1), float(step.get("radius_km", 1.0) or 1.0))
        return TaskCalendarTool._visited_after(history, point, earliest, float(step.get("radius_km", 1.0) or 1.0))

    @staticmethod
    def _visited_after(history: list[dict[str, Any]], point: Any, minute: int, radius_km: float) -> bool:
        if not TaskCalendarTool._point(point):
            return False
        p = (float(point[0]), float(point[1]))
        for record in history:
            end = TaskCalendarTool._record_end_minute(record)
            if end is None or end < minute:
                continue
            pos = record.get("position_after") if isinstance(record, dict) and isinstance(record.get("position_after"), dict) else {}
            lat = pos.get("lat")
            lng = pos.get("lng")
            try:
                if _haversine_km(float(lat), float(lng), p[0], p[1]) <= radius_km:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _waited_near_after(history: list[dict[str, Any]], point: Any, minute: int, wait_minutes: int, radius_km: float) -> bool:
        if not TaskCalendarTool._point(point):
            return False
        p = (float(point[0]), float(point[1]))
        run = 0
        for record in history:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") != "wait":
                continue
            end = TaskCalendarTool._record_end_minute(record)
            elapsed = _as_int(record.get("action_exec_cost_minutes")) or _as_int(record.get("step_elapsed_minutes")) or 0
            if end is None or end < minute:
                continue
            pos = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            try:
                near = _haversine_km(float(pos.get("lat")), float(pos.get("lng")), p[0], p[1]) <= radius_km
            except (TypeError, ValueError):
                near = False
            run = run + max(0, elapsed) if near else 0
            if run >= wait_minutes:
                return True
        return False

    @staticmethod
    def _record_end_minute(record: dict[str, Any]) -> int | None:
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        return _as_int(record.get("simulation_end_minute", result.get("simulation_progress_minutes")))

    @staticmethod
    def _point(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2
