"""Official-compatible dynamic LLM planning agent.

The agent is data-driven: it asks the model to transform visible preference text
into a structured planning profile, then evaluates runtime cargo candidates with
generic algorithms. It never reads server/data files or precomputed routes.
"""

from __future__ import annotations

import calendar
import json
import logging
import math
import os
import queue
import re
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from typing import Any

from simkit.ports import SimulationApiPort

try:
    from .tools import (
        ActionPreferenceGuardTool,
        CargoEvaluationTool,
        CommitmentSequenceTool,
        DecisionSupportTools,
        DriverProfileTool,
        MemoryTool,
        PreferenceClassificationTool,
        PromptTemplates,
        RegionPreferenceTool,
        RouteComplianceTool,
        TaskCalendarTool,
        TaskPenaltyOptimizerTool,
        TimeTaskProgressTool,
    )
except ImportError:  # Official runner may put demo/agent directly on sys.path.
    from tools import (
        ActionPreferenceGuardTool,
        CargoEvaluationTool,
        CommitmentSequenceTool,
        DecisionSupportTools,
        DriverProfileTool,
        MemoryTool,
        PreferenceClassificationTool,
        PromptTemplates,
        RegionPreferenceTool,
        RouteComplianceTool,
        TaskCalendarTool,
        TaskPenaltyOptimizerTool,
        TimeTaskProgressTool,
    )

_CN_NUMBERS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _parse_cn_number(value: str) -> int | None:
    value = str(value).strip().replace("个", "")
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value in _CN_NUMBERS:
        return _CN_NUMBERS[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + _CN_NUMBERS.get(value[1], 0)
    if "十" in value:
        left, right = value.split("十", 1)
        return _CN_NUMBERS.get(left, 1) * 10 + (_CN_NUMBERS.get(right, 0) if right else 0)
    return None


def _parse_base_time() -> datetime:
    raw = os.environ.get("AGENT_BASE_TIME") or os.environ.get("SIMULATION_BASE_TIME")
    for value in (raw,):
        if not value:
            continue
        text = str(value).replace("T", " ").strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
    return datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)


BASE_TIME_FROM_ENV = bool(os.environ.get("AGENT_BASE_TIME") or os.environ.get("SIMULATION_BASE_TIME"))
BASE_TIME = _parse_base_time()
HORIZON_DAYS = int(os.environ.get("AGENT_HORIZON_DAYS") or calendar.monthrange(BASE_TIME.year, BASE_TIME.month)[1])
MONTH_HORIZON_MINUTES = HORIZON_DAYS * 24 * 60
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
GLOBAL_RAW_PREFERENCE_CACHE: dict[str, list[dict[str, Any]]] = {}

DEFAULT_AGENT_CONFIG: dict[str, Any] = {
    "llm_core": {
        "disable_env": "AGENT_DISABLE_LLM",
        "choose_env": "AGENT_LLM_CHOOSE",
        "profile_env": "AGENT_LLM_PROFILE",
        "strict_json": True,
    },
    "context_tools": {
        "status_reader": "get_driver_status(driver_id)",
        "cargo_scanner": "query_cargo(driver_id, latitude, longitude, k)",
        "history_reader": "query_decision_history(driver_id, step)",
        "llm_gateway": "model_chat_completion(payload)",
        "driver_profile_tool": "tools/driver_profile_tool.py parses visible preferences into a safety-floor profile",
        "profit_calculator": "price - (pickup_km + haul_km) * cost_per_km",
        "empty_return_calculator": "subtract empty driving cost back to home/preference anchor",
        "route_simulation_tool": "evaluate one-step relay: finish current order, reposition to better area/anchor, then return",
        "time_window_checker": "load_time, daily rest, scheduled visit deadlines",
        "route_feasibility_checker": "finish time, return time, visit margins, month horizon",
        "risk_checker": "forbidden cargo, forbidden regions/days, deadhead, rest, visits",
        "time_task_progress_tool": "tracks one-shot, ordered and periodic driver tasks across the month",
        "region_preference_tool": "checks anchors, required visit points, forbidden regions and candidate destination impact",
        "task_penalty_optimizer_tool": "compares early-leave, late-arrival and not-at-point losses when tasks conflict",
        "candidate_compressor": "top 5 candidates by net_after_return and current_action_penalty Pareto tradeoff",
    },
    "agent_roles": {
        "profile_agent": "Convert visible preference text into a structured planning profile.",
        "profit_agent": "Estimate net income, hourly value, deadhead cost and opportunity quality for each candidate.",
        "risk_agent": "Reject or flag candidates that violate cargo, region, rest, off-day or scheduled-visit constraints.",
        "positioning_agent": "Evaluate whether the destination helps future visits, home/anchor return, and nearby cargo density.",
        "planner_agent": "Compare deterministic tool reports and choose the next safe action.",
        "supervisor_agent": "Review schedule/rest/off-day actions but cannot override safety guardrails.",
        "executor_agent": "Emit only official actions: take_order, wait, reposition.",
        "feedback_agent": "Read recent history and remember failed/expired cargos.",
    },
    "runtime_limits": {
        "cargo_k_default": 120,
        "cargo_k_min": 20,
        "cargo_k_max": 600,
        "history_steps": 80,
        "llm_candidate_limit": 3,
        "candidate_pool_limit": 40,
        "llm_max_calls_per_driver": 20000,
        "llm_min_interval_minutes": 0,
        "llm_decision_timeout_seconds": 20,
        "llm_wall_budget_seconds_per_driver": 28800,
        "llm_token_budget_per_driver": 20000000,
    },
    "guardrails": {
        "keep_local_profile_as_safety_floor": True,
        "llm_cannot_remove_constraints": True,
        "llm_cannot_override_safety_actions": True,
        "llm_take_order_score_gap_max": 300.0,
        "llm_take_order_net_gap_max": 500.0,
        "reject_risks": ["miss_scheduled_visit", "negative_net"],
    },
    "planning_calendar": {
        "base_time": BASE_TIME.strftime("%Y-%m-%d %H:%M:%S"),
        "horizon_days": HORIZON_DAYS,
        "day_index_rule": "0-based day index from base_time",
    },
}


def action(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"action": name, "params": params or {}, "model_usage": dict(ZERO_USAGE)}


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def distance_to_minutes(distance_km: float, speed_kmph: float | None = None) -> int:
    if distance_km <= 1e-6:
        return 0
    if speed_kmph is None:
        try:
            speed_kmph = float(os.environ.get("AGENT_REPOSITION_SPEED_KMPH", "60") or 60)
        except (TypeError, ValueError):
            speed_kmph = 60.0
    speed_kmph = max(1.0, float(speed_kmph))
    return max(1, math.ceil(distance_km / speed_kmph * 60.0))


def point_to_segment_km(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
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


def minute_offset(value: str | None) -> int | None:
    if not value:
        return None
    text = str(value).strip()
    cn_match = re.match(
        r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月([0-9一二两三四五六七八九十]{1,3})日\s*"
        r"([0-9一二两三四五六七八九十]{1,3})(?:点|:|：)([0-9]{1,2})?",
        text,
    )
    if cn_match:
        year_text, month_text, day_text, hour_text, minute_text = cn_match.groups()
        month = _parse_cn_number(month_text)
        day = _parse_cn_number(day_text)
        hour = _parse_cn_number(hour_text)
        minute = int(minute_text or 0)
        if month is not None and day is not None and hour is not None:
            year = int(year_text) if year_text else BASE_TIME.year
            text = f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return int((datetime.strptime(text, fmt) - BASE_TIME).total_seconds() // 60)
        except ValueError:
            pass
    return None


def _date_to_day_index(year_text: str | None, month_text: str, day_text: str) -> int | None:
    month = _parse_cn_number(month_text)
    day = _parse_cn_number(day_text)
    if month is None or day is None:
        return None
    year = int(year_text) if year_text else BASE_TIME.year
    try:
        return (datetime(year, month, day) - BASE_TIME).days
    except ValueError:
        return None


def _day_indices_from_text(text: str) -> list[int]:
    days: set[int] = set()
    compact = re.sub(r"\s+", "", text)
    range_pattern = (
        r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月([0-9一二两三四五六七八九十]{1,3})[号日]?"
        r"[到至\-~、和跟]"
        r"(?:(\d{4})年)?(?:(?P<end_month>[0-9一二两三四五六七八九十]{1,3})月)?([0-9一二两三四五六七八九十]{1,3})[号日]"
    )
    for match in re.finditer(range_pattern, compact):
        start_year, start_month, start_day, end_year, end_month, end_day = match.groups()
        start = _date_to_day_index(start_year, start_month, start_day)
        end = _date_to_day_index(end_year or start_year, end_month or start_month, end_day)
        if start is not None and end is not None:
            days.update(range(min(start, end), max(start, end) + 1))
    adjacent_pattern = (
        r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月"
        r"([0-9一二两三四五六七八九十]{1,3})[号日]"
        r"([0-9一二两三四五六七八九十]{1,3})[号日]"
    )
    for match in re.finditer(adjacent_pattern, compact):
        year, month, start_day, end_day = match.groups()
        start = _date_to_day_index(year, month, start_day)
        end = _date_to_day_index(year, month, end_day)
        if start is not None and end is not None:
            days.update(range(min(start, end), max(start, end) + 1))
    date_pattern = r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月([0-9一二两三四五六七八九十]{1,3})\s*[号日]"
    for match in re.finditer(date_pattern, text):
        day_index = _date_to_day_index(*match.groups())
        if day_index is not None:
            days.add(day_index)
    return sorted(day for day in days if 0 <= day < HORIZON_DAYS)


def wall_time(minute: int) -> str:
    return (BASE_TIME + timedelta(minutes=int(minute))).strftime("%Y-%m-%d %H:%M")


def extract_coordinates(text: str) -> list[tuple[float, float]]:
    return [
        (float(lat), float(lng))
        for lat, lng in re.findall(r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]", text)
    ]


def preference_items(preferences: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in preferences:
        if isinstance(item, dict):
            result.append(
                {
                    "content": str(item.get("content") or item.get("preference") or item.get("text") or item),
                    "penalty_amount": item.get("penalty_amount"),
                    "penalty_cap": item.get("penalty_cap"),
                }
            )
        else:
            result.append({"content": str(item), "penalty_amount": None, "penalty_cap": None})
    return result


class ModelDecisionService:
    """Official entry point called by the simulator for every decision step.

    Design note for future readers:
    This class grew into a hybrid planner because the competition environment
    has two conflicting requirements. The agent must generalize from free-form
    Chinese preferences, but every action also has a tight runtime budget and
    a small set of legal outputs. The implementation therefore uses LLMs mainly
    to parse/compare ambiguous preferences, while deterministic tools protect
    hard constraints and keep the agent from timing out.
    """

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.dynamic_profile_agent")
        self._config = self._load_agent_config()
        llm_config = self._config["llm_core"]
        limits = self._config["runtime_limits"]
        self._cargo_k = self._int_env(
            "AGENT_CARGO_K",
            int(limits["cargo_k_default"]),
            int(limits["cargo_k_min"]),
            int(limits["cargo_k_max"]),
        )
        self._llm_profile_cache: dict[str, tuple[str, dict[str, Any]]] = {}
        self._cargo_memory: dict[str, dict[str, Any]] = {}
        self._failed_cargos_by_driver: dict[str, set[str]] = {}
        self._dynamic_rule_memory: dict[str, list[dict[str, Any]]] = self._load_dynamic_rule_memory()
        self._driver_tool_memory: dict[str, dict[str, Any]] = {}
        self._driver_runtime: dict[str, dict[str, Any]] = {}
        self._decision_history_cache: dict[str, Any] = {}
        self._current_driver_id: str | None = None
        requested_runtime_mode = os.environ.get("AGENT_RUNTIME_MODE", "llm_budgeted").strip().lower()
        force_fast = os.environ.get("AGENT_FORCE_FAST", "0").strip().lower() in {"1", "true", "yes"}
        self._runtime_mode = "fast" if force_fast else requested_runtime_mode
        self._disable_llm = os.environ.get(str(llm_config["disable_env"]), "").strip().lower() in {"1", "true", "yes"}
        llm_runtime_allowed = self._runtime_mode not in {"fast", "heuristic", "rules", "no_llm", "offline"}
        self._enable_llm_profile = llm_runtime_allowed and os.environ.get(str(llm_config.get("profile_env", "AGENT_LLM_PROFILE")), "1").strip().lower() in {"1", "true", "yes"}
        self._enable_llm_choice = llm_runtime_allowed and os.environ.get(str(llm_config["choose_env"]), "1").strip().lower() in {"1", "true", "yes"}
        self._enable_llm_review = llm_runtime_allowed and os.environ.get("AGENT_LLM_REVIEW_SAFETY", "0").strip().lower() in {"1", "true", "yes"}
        self._driver_profile_tool = DriverProfileTool
        self._cargo_evaluation_tool = CargoEvaluationTool
        self._memory_tool = MemoryTool
        self._prompt_templates = PromptTemplates
        self._action_guard_tool = ActionPreferenceGuardTool
        self._commitment_sequence_tool = CommitmentSequenceTool
        self._preference_classification_tool = PreferenceClassificationTool
        self._route_compliance_tool = RouteComplianceTool
        self._decision_support_tools = DecisionSupportTools
        self._time_task_progress_tool = TimeTaskProgressTool
        self._region_preference_tool = RegionPreferenceTool
        self._task_penalty_optimizer_tool = TaskPenaltyOptimizerTool
        self._task_calendar_tool = TaskCalendarTool

    @staticmethod
    def _load_agent_config() -> dict[str, Any]:
        config = json.loads(json.dumps(DEFAULT_AGENT_CONFIG, ensure_ascii=False))
        raw = os.environ.get("AGENT_CONFIG_JSON")
        if not raw:
            return config
        try:
            override = json.loads(raw)
        except json.JSONDecodeError:
            return config
        if isinstance(override, dict):
            ModelDecisionService._deep_update(config, override)
        return config

    @staticmethod
    def _dynamic_rule_file_path() -> str:
        return os.environ.get(
            "AGENT_DYNAMIC_RULE_FILE",
            os.path.join(tempfile.gettempdir(), "tianchi_agent_dynamic_rules.json"),
        )

    def _load_dynamic_rule_memory(self) -> dict[str, list[dict[str, Any]]]:
        if os.environ.get("AGENT_ENABLE_PERSISTED_TOOL_MEMORY", "0").strip().lower() not in {"1", "true", "yes"}:
            return {}
        path = self._dynamic_rule_file_path()
        try:
            with open(path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        memory: dict[str, list[dict[str, Any]]] = {}
        for driver_id, rules in raw.items():
            if isinstance(rules, list):
                memory[str(driver_id)] = [rule for rule in rules if isinstance(rule, dict)][:80]
        return memory

    def _save_dynamic_rule_memory(self) -> None:
        if os.environ.get("AGENT_ENABLE_PERSISTED_TOOL_MEMORY", "0").strip().lower() not in {"1", "true", "yes"}:
            return
        if os.environ.get("AGENT_DISABLE_DYNAMIC_RULE_FILE", "0").strip().lower() in {"1", "true", "yes"}:
            return
        path = self._dynamic_rule_file_path()
        try:
            self._atomic_json_dump(path, self._dynamic_rule_memory)
        except Exception as exc:
            self._logger.info("dynamic rule memory save skipped: %s", exc)

    @staticmethod
    def _atomic_json_dump(path: str, payload: Any) -> None:
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _driver_tool_dir() -> str:
        return os.environ.get(
            "AGENT_DRIVER_TOOL_DIR",
            os.path.join(tempfile.gettempdir(), "tianchi_agent_driver_tools"),
        )

    def _driver_tool_file_path(self, driver_id: str) -> str:
        safe_driver = re.sub(r"[^0-9A-Za-z_.-]", "_", str(driver_id))[:80] or "unknown"
        return os.path.join(self._driver_tool_dir(), f"{safe_driver}.json")

    def _load_driver_tool_memory(self, driver_id: str) -> dict[str, Any]:
        cached = self._driver_tool_memory.get(driver_id)
        if isinstance(cached, dict):
            return cached
        if os.environ.get("AGENT_ENABLE_PERSISTED_TOOL_MEMORY", "0").strip().lower() not in {"1", "true", "yes"}:
            raw = {"tool_name": "driver_dynamic_rule_tool", "driver_id": driver_id, "dynamic_preference_rules": [], "planner_updates": [], "raw_preferences": []}
            self._driver_tool_memory[driver_id] = raw
            return raw
        path = self._driver_tool_file_path(driver_id)
        try:
            with open(path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except Exception:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        raw.setdefault("tool_name", "driver_dynamic_rule_tool")
        raw.setdefault("driver_id", driver_id)
        raw.setdefault("dynamic_preference_rules", [])
        raw.setdefault("penalty_anomalies", [])
        raw.setdefault("planner_updates", [])
        raw.setdefault("raw_preferences", [])
        self._driver_tool_memory[driver_id] = raw
        return raw

    def _save_driver_tool_memory(self, driver_id: str, memory: dict[str, Any]) -> None:
        if os.environ.get("AGENT_ENABLE_PERSISTED_TOOL_MEMORY", "0").strip().lower() not in {"1", "true", "yes"}:
            self._driver_tool_memory[driver_id] = memory
            return
        if os.environ.get("AGENT_DISABLE_DRIVER_TOOL_FILE", "0").strip().lower() in {"1", "true", "yes"}:
            return
        path = self._driver_tool_file_path(driver_id)
        try:
            self._atomic_json_dump(path, memory)
            self._driver_tool_memory[driver_id] = memory
            self._prune_runtime_caches(driver_id)
        except Exception as exc:
            self._logger.info("driver tool memory save skipped: %s", exc)

    def _record_penalty_anomaly(self, driver_id: str, anomaly: dict[str, Any]) -> None:
        if not anomaly:
            return
        memory = self._load_driver_tool_memory(driver_id)
        events = [item for item in memory.get("penalty_anomalies", []) if isinstance(item, dict)]
        events.append(anomaly)
        memory["penalty_anomalies"] = events[-40:]
        self._save_driver_tool_memory(driver_id, memory)

    def _remember_raw_preferences(self, driver_id: str, preferences: list[dict[str, Any]]) -> None:
        if not preferences:
            return
        clean = [
            {
                "content": str(item.get("content") or ""),
                "penalty_amount": item.get("penalty_amount"),
                "penalty_cap": item.get("penalty_cap"),
            }
            for item in preferences
            if isinstance(item, dict) and item.get("content")
        ][:80]
        if not clean:
            return
        memory = self._load_driver_tool_memory(driver_id)
        merged = self._merge_list(
            self._merge_list(memory.get("raw_preferences", []), GLOBAL_RAW_PREFERENCE_CACHE.get(driver_id, [])),
            self._merge_list(self._driver_state(driver_id).get("raw_preferences", []), clean),
        )[:80]
        self._driver_state(driver_id)["raw_preferences"] = list(merged)
        GLOBAL_RAW_PREFERENCE_CACHE[driver_id] = list(merged)
        memory["raw_preferences"] = list(merged)
        self._save_driver_tool_memory(driver_id, memory)

    def _load_raw_preferences(self, driver_id: str) -> list[dict[str, Any]]:
        memory = self._load_driver_tool_memory(driver_id)
        raw = self._merge_list(
            self._merge_list(memory.get("raw_preferences", []), GLOBAL_RAW_PREFERENCE_CACHE.get(driver_id, [])),
            self._driver_state(driver_id).get("raw_preferences", []),
        )
        return [item for item in raw if isinstance(item, dict) and item.get("content")][:80]

    @staticmethod
    def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                ModelDecisionService._deep_update(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def _int_env(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)))
        except ValueError:
            value = default
        return max(low, min(high, value))

    def decide(self, driver_id: str) -> dict[str, Any]:
        """Return one official action for the current simulation state.

        The decision pipeline is intentionally staged:
        1. Query only official interfaces for status, history and cargo.
        2. Convert visible preference text into a structured planning profile.
        3. Execute hard safety/time actions before looking for market orders.
        4. Score cargo candidates with deterministic profit/risk tools.
        5. Use the LLM only as a bounded tie-breaker/reviewer.

        This staging prevents two common failure modes from earlier versions:
        all-wait caused by slow LLM calls, and high penalties caused by letting
        a free-form model emit actions without guardrail validation.
        """
        self._current_driver_id = driver_id
        self._driver_state(driver_id)
        self._decision_usage = dict(ZERO_USAGE)
        status = self._api.get_driver_status(driver_id)
        self._align_calendar_from_status(status)
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        self._prepare_decision_history(driver_id, current_minute)
        self._prune_runtime_caches(driver_id)
        if current_minute >= MONTH_HORIZON_MINUTES:
            return self._with_usage(action("wait", {"duration_minutes": 60}))
        self._learn_from_history(driver_id)
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)

        preferences = preference_items(list(status.get("preferences", []) or []))
        if preferences:
            self._remember_raw_preferences(driver_id, preferences)
        preferences = self._merge_list(preferences, self._load_raw_preferences(driver_id))
        # Raw text guards run before LLM/profile normalization. They cover
        # high-impact constraints such as "do not enter this circle" even when
        # the structured parser has not yet produced a clean profile.
        raw_forbidden_escape = self._raw_forbidden_circle_escape_action(lat, lng, preferences)
        if raw_forbidden_escape is not None:
            return self._with_usage(raw_forbidden_escape)
        raw_home = self._raw_night_home_action(current_minute, lat, lng, preferences)
        if raw_home is not None and raw_home.get("action") != "reposition":
            return self._with_usage(raw_home)
        visible_no_action = self._visible_daily_no_action_window_action(current_minute, preferences)
        if visible_no_action is not None and raw_home is None:
            return self._with_usage(visible_no_action)

        profile = self._profile_scoped_for_current_time(self._planning_profile(driver_id, preferences), current_minute)
        profile = self._profile_without_seen_cargo_commitments(driver_id, profile)
        profile = self._profile_without_forbidden_required_cargos(driver_id, profile)
        # The profile is a safety floor, not a complete plan. It removes already
        # satisfied/failed commitments and impossible required cargo before the
        # market scorer sees them, otherwise the agent can repeatedly chase a
        # stale cargo or drive into a forbidden area.
        forbidden_escape = self._forbidden_circle_escape_action(lat, lng, profile)
        if forbidden_escape is not None:
            return self._with_usage(forbidden_escape)
        raw_preferences = list(preferences)
        cards = profile.get("preference_cards", []) if isinstance(profile.get("preference_cards"), list) else []
        raw_preferences.extend(
            {
                "content": str(card.get("content") or ""),
                "penalty_amount": card.get("penalty_amount"),
                "penalty_cap": card.get("penalty_cap"),
            }
            for card in cards
            if isinstance(card, dict) and card.get("content")
        )
        raw_temporary = self._raw_temporary_event_action(current_minute, lat, lng, raw_preferences, driver_id=driver_id)
        if raw_temporary is not None:
            return self._with_usage(raw_temporary)
        special = self._profile_special_action(driver_id, status, profile)
        if special is not None and self._special_action_must_preempt_market(special):
            return self._with_usage(special)

        raw = self._query_cargo(driver_id, lat, lng, profile)
        status = self._api.get_driver_status(driver_id)
        items = raw.get("items", []) if isinstance(raw, dict) else []
        # Candidate construction is the core "earning money" layer. It attaches
        # profit, travel time, return cost, preference risk and schedule risk to
        # every visible cargo before any action can be selected.
        candidates = self._build_candidates(driver_id, status, items, profile)
        candidates = self._filter_raw_temporary_safe_candidates(current_minute, candidates, raw_preferences)
        notice_schedule = self._commitment_notice_action(current_minute, lat, lng, profile, raw_preferences)
        if raw_home is not None and self._home_return_schedule_preempts_market(raw_home, candidates, status):
            return self._with_usage(raw_home)
        if notice_schedule is not None and self._commitment_schedule_preempts_market(notice_schedule, candidates):
            return self._with_usage(notice_schedule)

        anomaly = self._penalty_anomaly_report(candidates)
        allow_anomaly_recheck = os.environ.get("AGENT_ENABLE_PROFILE_RECHECK_ON_ANOMALY", "0").strip().lower() in {"1", "true", "yes"}
        if allow_anomaly_recheck and anomaly.get("profile_recheck_needed") and self._enable_llm_profile and self._can_use_llm(driver_id):
            self._record_penalty_anomaly(driver_id, anomaly)
            profile = self._profile_scoped_for_current_time(
                self._planning_profile(driver_id, preferences, force_recheck=True, anomaly_report=anomaly),
                current_minute,
            )
            profile = self._profile_without_seen_cargo_commitments(driver_id, profile)
            candidates = self._build_candidates(driver_id, status, items, profile)
            candidates = self._filter_raw_temporary_safe_candidates(current_minute, candidates, raw_preferences)
            notice_schedule = self._commitment_notice_action(current_minute, lat, lng, profile, raw_preferences)
            if raw_home is not None and self._home_return_schedule_preempts_market(raw_home, candidates, status):
                return self._with_usage(raw_home)
            if notice_schedule is not None and self._commitment_schedule_preempts_market(notice_schedule, candidates):
                return self._with_usage(notice_schedule)

        confident = self._confident_income_candidate(candidates)
        if confident is not None:
            # Fast path: if tools agree a cargo is profitable and preference-safe,
            # take it without waiting for an LLM. This was added because a slow
            # or over-cautious planner can lose the whole month by returning wait.
            self._remember_cargo(confident)
            planned = action("take_order", {"cargo_id": str(confident["cargo_id"])})
            planned["reason_code"] = "deterministic_confident_income_take"
            return self._with_usage(planned)

        if not candidates:
            schedule = self._commitment_notice_action(current_minute, lat, lng, profile, raw_preferences)
            if schedule is not None:
                return self._with_usage(schedule)
            explore = self._market_explore_action(status, profile, items=items, candidates=[])
            if explore is not None:
                return self._with_usage(explore)
            if special is not None:
                reviewed_special = self._llm_review_recommended_action(driver_id, status, profile, special, "soft preference/time-task action after no feasible cargo", [], force_llm=self._force_review_every_action())
                if reviewed_special is not None:
                    return self._with_usage(reviewed_special)
                return self._with_usage(special)
            fallback = self._fallback_position_or_wait(status, profile, raw_preferences)
            reviewed = self._llm_review_recommended_action(driver_id, status, profile, fallback, "no feasible cargo candidate", force_llm=self._force_review_every_action())
            if reviewed is not None:
                return self._with_usage(reviewed)
            return self._with_usage(fallback)

        if self._enable_llm_choice and not self._disable_llm:
            # The LLM never receives authority to invent arbitrary actions. It
            # chooses among compressed tool-scored alternatives, then the result
            # is validated again before execution.
            chosen = self._llm_choose_action(driver_id, status, profile, candidates, self._history_summary(driver_id))
            if self._apply_planner_updates(driver_id, chosen, profile):
                profile = self._profile_scoped_for_current_time(
                    self._planning_profile(driver_id, preferences, force_recheck=True, anomaly_report={"reason": "planner_requested_dynamic_rule_update"}),
                    current_minute,
                )
                profile = self._profile_without_seen_cargo_commitments(driver_id, profile)
                candidates = self._build_candidates(driver_id, status, items, profile)
                candidates = self._filter_raw_temporary_safe_candidates(current_minute, candidates, raw_preferences)
                chosen = self._llm_choose_action(driver_id, status, profile, candidates, self._history_summary(driver_id)) if candidates else None
            elif self._planner_requests_profile_recheck(chosen):
                self._record_penalty_anomaly(driver_id, {"reason": "planner_requested_profile_recheck", "minute": current_minute})
                profile = self._profile_scoped_for_current_time(
                    self._planning_profile(driver_id, preferences, force_recheck=True, anomaly_report={"reason": "planner_requested_profile_recheck"}),
                    current_minute,
                )
                profile = self._profile_without_seen_cargo_commitments(driver_id, profile)
                candidates = self._build_candidates(driver_id, status, items, profile)
                candidates = self._filter_raw_temporary_safe_candidates(current_minute, candidates, raw_preferences)
                chosen = self._llm_choose_action(driver_id, status, profile, candidates, self._history_summary(driver_id)) if candidates else None
            validated = self._validate_action(chosen, candidates, current_minute=current_minute)
            if validated is not None and self._safe_to_accept_llm_choice(validated, candidates):
                if validated.get("action") != "take_order":
                    if notice_schedule is not None and self._commitment_schedule_preempts_market(notice_schedule, candidates):
                        return self._with_usage(notice_schedule)
                    if self._has_acceptable_candidate(candidates):
                        pass
                    else:
                        reviewed = self._llm_review_recommended_action(driver_id, status, profile, validated, "planner selected non-cargo action", candidates, force_llm=self._force_review_every_action())
                        if reviewed is not None:
                            return self._with_usage(reviewed)
                        return self._with_usage(validated)
                else:
                    reviewed = self._llm_review_recommended_action(driver_id, status, profile, validated, "planner selected final action", candidates, force_llm=self._force_review_every_action())
                    if reviewed is not None:
                        if reviewed.get("action") == "take_order" and self._safe_to_accept_llm_choice(reviewed, candidates):
                            return self._with_usage(reviewed)
                        if reviewed.get("action") != "take_order":
                            if notice_schedule is not None and self._commitment_schedule_preempts_market(notice_schedule, candidates):
                                return self._with_usage(notice_schedule)
                            if not self._has_acceptable_candidate(candidates):
                                return self._with_usage(reviewed)
                    return self._with_usage(validated)

        if not self._has_acceptable_candidate(candidates):
            schedule = self._commitment_notice_action(current_minute, lat, lng, profile, raw_preferences)
            if schedule is not None:
                return self._with_usage(schedule)
            rescue = self._income_rescue_candidate(candidates)
            if rescue is not None:
                self._remember_cargo(rescue)
                return self._with_usage(action("take_order", {"cargo_id": str(rescue["cargo_id"])}))
            if special is not None:
                reviewed_special = self._llm_review_recommended_action(driver_id, status, profile, special, "soft preference/time-task action after cargo tradeoffs failed", candidates, force_llm=self._force_review_every_action())
                if reviewed_special is not None:
                    return self._with_usage(reviewed_special)
                return self._with_usage(special)
            explore = self._market_explore_action(status, profile, items=items, candidates=candidates)
            if explore is not None:
                return self._with_usage(explore)
            fallback = self._fallback_position_or_wait(status, profile, raw_preferences)
            reviewed = self._llm_review_recommended_action(driver_id, status, profile, fallback, "all cargo candidates lose money or trigger too much preference risk", force_llm=self._force_review_every_action())
            if reviewed is not None:
                return self._with_usage(reviewed)
            return self._with_usage(fallback)

        best = max((item for item in candidates if self._candidate_is_acceptable(item)), key=self._profit_penalty_sort_key)
        self._remember_cargo(best)
        fallback_action = action("take_order", {"cargo_id": str(best["cargo_id"])})
        reviewed = self._llm_review_recommended_action(driver_id, status, profile, fallback_action, "tool fallback final action", candidates, force_llm=self._force_review_every_action())
        if reviewed is not None:
            if reviewed.get("action") == "take_order" and self._safe_to_accept_llm_choice(reviewed, candidates):
                return self._with_usage(reviewed)
            if reviewed.get("action") != "take_order":
                if notice_schedule is not None and self._commitment_schedule_preempts_market(notice_schedule, candidates):
                    return self._with_usage(notice_schedule)
                return self._with_usage(fallback_action)
        return self._with_usage(fallback_action)

    def _commitment_schedule_preempts_market(self, schedule: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> bool:
        if schedule is None:
            return False
        reason = str(schedule.get("reason_code") or "")
        notice_only = reason in {
            "raw_temporary_event_notice_position",
            "raw_temporary_event_notice_wait",
            "temporary_event_notice_position",
            "temporary_event_notice_wait",
        }
        if not notice_only:
            return True
        # In the early notice window, allow only already-filtered, positive-value
        # short cargos. If none exist, the mandatory schedule takes over.
        return not any(self._candidate_is_acceptable(item) for item in candidates)

    def _home_return_schedule_preempts_market(
        self,
        schedule: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        status: dict[str, Any],
    ) -> bool:
        if schedule is None:
            return False
        if schedule.get("action") != "reposition":
            return True
        params = schedule.get("params", {}) if isinstance(schedule.get("params"), dict) else {}
        try:
            home = (float(params["latitude"]), float(params["longitude"]))
        except (KeyError, TypeError, ValueError):
            return True
        deadline = self._as_int(schedule.get("home_deadline_minute"))
        cost_per_km = float(status.get("cost_per_km", 1.5) or 1.5)
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)
        empty_cost = haversine_km(lat, lng, home[0], home[1]) * cost_per_km
        margin = float(os.environ.get("AGENT_NIGHT_HOME_ORDER_VALUE_MARGIN_YUAN", "200") or 200)
        for item in candidates:
            if not self._candidate_is_acceptable(item):
                continue
            end = item.get("end")
            finish = self._as_int(item.get("finish_minute"))
            if deadline is not None and self._valid_point(end) and finish is not None:
                assert isinstance(end, (list, tuple))
                travel_home = distance_to_minutes(haversine_km(float(end[0]), float(end[1]), home[0], home[1]))
                item["home_return_after_candidate_minute"] = finish + travel_home
            if self._candidate_action_value(item) > -empty_cost + margin:
                return False
        return True

    def _finalize_action(self, result: dict[str, Any], candidates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if result.get("action") == "reposition" and not result.get("reason_code"):
            safe_candidates = [item for item in (candidates or []) if self._candidate_is_acceptable(item)]
            if safe_candidates:
                best = max(safe_candidates, key=self._profit_penalty_sort_key)
                self._remember_cargo(best)
                return action("take_order", {"cargo_id": str(best["cargo_id"])})
            return action("wait", {"duration_minutes": 60})
        return result

    @staticmethod
    def _special_action_must_preempt_market(candidate_action: dict[str, Any]) -> bool:
        """Only hard commitments should block cargo search before seeing market prices."""
        if not isinstance(candidate_action, dict):
            return False
        reason = str(candidate_action.get("reason_code") or "")
        if reason.startswith("cumulative_penalty_"):
            return True
        if reason.startswith("long_sequence_"):
            return True
        if reason == "monthly_full_off_day_urgent":
            return True
        if reason in {"daily_no_action_window_guard", "daily_rest_window_guard", "task_calendar_no_action_window"}:
            return True
        if reason in {"task_calendar_task_wait", "task_calendar_wait_until_task"}:
            return True
        if reason in {"task_calendar_monthly_off_day", "monthly_full_off_day_idle_progress"}:
            return True
        if reason in {"night_home_deadline_return_guard", "night_home_stay_guard"}:
            return True
        if reason in {
            "commitment_go_pickup_first",
            "commitment_wait_pickup",
            "commitment_return_home",
            "commitment_stay_home",
            "commitment_required_cargo_position",
            "commitment_wait_required_cargo_online",
            "commitment_required_cargo_late_position",
            "commitment_take_required_cargo",
            "mandatory_required_cargo_take_now",
        }:
            return True
        return False

    def _planning_profile(self, driver_id: str, preferences: list[dict[str, Any]], force_recheck: bool = False, anomaly_report: dict[str, Any] | None = None) -> dict[str, Any]:
        signature_payload = {"preferences": preferences, "anomaly": anomaly_report if force_recheck else None}
        signature = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
        cached = self._llm_profile_cache.get(driver_id)
        if not force_recheck and cached and cached[0] == signature:
            return cached[1]
        profile = self._llm_parse_profile(driver_id, preferences, anomaly_report) if self._enable_llm_profile and self._can_use_llm(driver_id) else None
        if not isinstance(profile, dict):
            profile = self._empty_profile(preferences)
        driver_tool = self._load_driver_tool_memory(driver_id)
        persisted_rules = self._merge_list(self._dynamic_rule_memory.get(driver_id, []), driver_tool.get("dynamic_preference_rules", []))
        if persisted_rules:
            profile["dynamic_preference_rules"] = self._merge_list(profile.get("dynamic_preference_rules"), persisted_rules)
        profile = self._normalize_profile(profile, preferences)
        rules = list(profile.get("dynamic_preference_rules", []) or [])
        self._dynamic_rule_memory[driver_id] = rules
        driver_tool["dynamic_preference_rules"] = rules
        driver_tool["last_profile_signature"] = signature
        self._save_dynamic_rule_memory()
        self._save_driver_tool_memory(driver_id, driver_tool)
        self._llm_profile_cache[driver_id] = (signature, profile)
        return profile

    def _align_calendar_from_status(self, status: dict[str, Any]) -> None:
        if BASE_TIME_FROM_ENV:
            return
        wall_time = str(status.get("simulation_wall_time") or "").strip()
        progress = self._as_int(status.get("simulation_progress_minutes"))
        if not wall_time or progress is None:
            return
        try:
            current_wall = datetime.strptime(wall_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                current_wall = datetime.strptime(wall_time, "%Y-%m-%d %H:%M")
            except ValueError:
                return
        inferred = (current_wall - timedelta(minutes=int(progress))).replace(second=0, microsecond=0)
        global BASE_TIME, HORIZON_DAYS, MONTH_HORIZON_MINUTES
        if inferred != BASE_TIME:
            BASE_TIME = inferred
            if not os.environ.get("AGENT_HORIZON_DAYS"):
                HORIZON_DAYS = calendar.monthrange(BASE_TIME.year, BASE_TIME.month)[1]
        MONTH_HORIZON_MINUTES = HORIZON_DAYS * 24 * 60
        planning = self._config.get("planning_calendar")
        if isinstance(planning, dict):
            planning["base_time"] = BASE_TIME.strftime("%Y-%m-%d %H:%M:%S")
            planning["horizon_days"] = HORIZON_DAYS
        self._sync_tool_calendar()

    def _sync_tool_calendar(self) -> None:
        for tool in (self._driver_profile_tool,):
            module = sys.modules.get(getattr(tool, "__module__", ""))
            if module is None:
                continue
            if hasattr(module, "BASE_TIME"):
                setattr(module, "BASE_TIME", BASE_TIME)
            if hasattr(module, "HORIZON_DAYS"):
                setattr(module, "HORIZON_DAYS", HORIZON_DAYS)

    def _llm_parse_profile(self, driver_id: str, preferences: list[dict[str, Any]], anomaly_report: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self._can_use_llm(driver_id):
            return None
        deterministic_profile = self._driver_profile_tool.build_profile_from_preferences(preferences)
        profile_tool_context = self._driver_profile_tool.llm_context(preferences, deterministic_profile)
        profile_tool_context["preference_classification_tool"] = self._preference_classification_tool.classify_profile(deterministic_profile)
        if anomaly_report:
            profile_tool_context["penalty_anomaly_report"] = anomaly_report
            profile_tool_context["llm_instruction"] = str(profile_tool_context.get("llm_instruction", "")) + " Re-check preference extraction because planner detected abnormal preference penalties."
        compact_preferences = self._compact_preferences_for_llm(preferences)
        payload = {
            "messages": self._prompt_templates.profile_messages(
                compact_preferences,
                self._public_agent_config(),
                self._compact_profile_tool_context(profile_tool_context),
            ),
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": int(os.environ.get("AGENT_PROFILE_MAX_TOKENS", "4096") or 4096),
        }
        try:
            response = self._call_model_chat_completion(payload, "profile")
            self._record_usage(response)
            content = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
            return self._parse_json_object(content)
        except Exception as exc:
            self._logger.info("profile parse fallback: %s", exc)
            return None

    @staticmethod
    def _compact_preferences_for_llm(preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, item in enumerate(preferences, start=1):
            content = str(item.get("content", ""))
            result.append(
                {
                    "id": f"P{index:03d}",
                    "content": content[:220],
                    "penalty_amount": item.get("penalty_amount"),
                    "penalty_cap": item.get("penalty_cap"),
                    "start_time": item.get("start_time"),
                    "end_time": item.get("end_time"),
                }
            )
        return result

    @staticmethod
    def _compact_profile_tool_context(context: dict[str, Any]) -> dict[str, Any]:
        profile = context.get("deterministic_profile", {}) if isinstance(context, dict) else {}
        if not isinstance(profile, dict):
            profile = {}
        cards = [card for card in profile.get("preference_cards", []) if isinstance(card, dict)]
        compact_cards = [
            {
                "id": card.get("id"),
                "types": card.get("types"),
                "risk_key": card.get("risk_key"),
                "severity": card.get("severity"),
                "tradeoff_mode": card.get("tradeoff_mode"),
                "penalty_amount": card.get("penalty_amount"),
                "penalty_cap": card.get("penalty_cap"),
            }
            for card in cards
        ]
        compact_profile = {
            key: value
            for key, value in profile.items()
            if key not in {"preference_cards", "tool_trace"}
        }
        compact_profile["preference_cards"] = compact_cards[:80]
        return {
            "tool_name": context.get("tool_name", "driver_profile_tool") if isinstance(context, dict) else "driver_profile_tool",
            "purpose": context.get("purpose", "") if isinstance(context, dict) else "",
            "schema": context.get("schema", {}) if isinstance(context, dict) else {},
            "preference_count": context.get("preference_count", len(cards)) if isinstance(context, dict) else len(cards),
            "deterministic_profile": compact_profile,
            "duplicate_preference_groups": profile.get("duplicate_preference_groups", [])[:12],
            "unknown_preferences": profile.get("unknown_preferences", [])[:12],
            "unknown_attribute_tags": profile.get("unknown_attribute_tags", [])[:24],
            "unknown_preference_groups": profile.get("unknown_preference_groups", [])[:12],
            "raw_preference_coverage_audit": profile.get("raw_preference_coverage_audit", [])[:24],
            "preference_classification_tool": context.get("preference_classification_tool", {}) if isinstance(context, dict) else {},
            "detected_constraint_types": context.get("detected_constraint_types", []) if isinstance(context, dict) else [],
            "llm_instruction": context.get("llm_instruction", "") if isinstance(context, dict) else "",
        }

    @staticmethod
    def _empty_profile(preferences: list[dict[str, Any]]) -> dict[str, Any]:
        # Online-safe fallback: this is the toolized version of the previous
        # driver_profile_analyzer.py and reads only visible status preferences.
        return DriverProfileTool.build_profile_from_preferences(preferences)

    @staticmethod
    def _fallback_avoid_keywords(text: str) -> list[str]:
        keywords: set[str] = set()
        negative_markers = ("不接", "不要接", "不拉", "避免", "推掉", "干不了", "不想拉", "别派", "不能碰", "拒绝")
        for quoted in re.findall(r"[「“\"']([^」“”\"']{1,20})[」”\"']", text):
            if any(word in text for word in negative_markers):
                keywords.add(quoted.strip())
        for keyword in (
            "机械设备",
            "蔬菜",
            "食品饮料",
            "服饰纺织皮革",
            "快递快运搬家",
            "冷链",
            "生鲜",
            "水果",
            "海鲜",
            "化工",
            "危险品",
            "玻璃",
            "陶瓷",
            "家具",
            "建材",
            "钢材",
        ):
            if keyword in text and any(word in text for word in negative_markers):
                keywords.add(keyword)
        for match in re.finditer(r"(?:不接|不要接|不拉|避免|推掉|干不了|拒绝|别派)([^。；;，,\n]{1,24})", text):
            segment = re.sub(r"(?:的|这类|这种|货源|货|活儿|订单|都|一律|凡是|每接.*)$", "", match.group(1)).strip()
            for token in re.split(r"[、/和与及\s]+", segment):
                token = token.strip("：“”\"'「」")
                if 2 <= len(token) <= 12 and not any(word in token for word in ("装货", "卸货", "地区", "那边", "路线")):
                    keywords.add(token)
        return sorted(k for k in keywords if k)

    @staticmethod
    def _fallback_avoid_regions(text: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        regions = set()
        for match in re.finditer(r"(?:在|去|往|进|到|跑|派进|装货地或卸货地在)([\u4e00-\u9fa5]{2,6})(?:市|区|县|镇|城|那边|的货|跑|去|接|派|查|$)", text):
            candidate = match.group(1).strip()
            candidate = re.sub(r"(跑|去|接|派|查|那边|这边)$", "", candidate).strip()
            if 2 <= len(candidate) <= 4 and not any(bad in candidate for bad in ("货地", "卸货", "这天", "交警", "连续", "每天")):
                regions.add(candidate)
        for region in regions:
            if region not in text:
                continue
            if any(bad in region for bad in ("那边", "这边", "去那", "到那")):
                continue
            around = ModelDecisionService._region_context(text, region)
            if any(word in around for word in ("一律不接", "不接", "不往", "别给我派进去", "不去", "避开", "别去", "不要去")):
                days = ModelDecisionService._fallback_days_near_region(text, region)
                result.append({"region": region, "days": days or None})
        return result

    @staticmethod
    def _region_context(text: str, region: str) -> str:
        idx = text.find(region)
        if idx < 0:
            return text
        return text[max(0, idx - 40) : idx + len(region) + 40]

    @staticmethod
    def _fallback_days_near_region(text: str, region: str) -> list[int]:
        context = ModelDecisionService._region_context(text, region)
        return _day_indices_from_text(context)

    @staticmethod
    def _fallback_daily_rest(text: str) -> dict[str, Any]:
        rest = {"hours": None, "window_start_minute": None, "window_end_minute": None}
        match = re.search(r"连续[^0-9一二三四五六七八九十]*([0-9一二三四五六七八九十]+)\s*小时", text)
        if match:
            rest["hours"] = ModelDecisionService._cn_day(match.group(1))
        window = ModelDecisionService._parse_daily_time_window(text)
        # A no-order/no-driving window is handled as a soft preference window.
        # Do not turn it into an extra daily-rest task unless the text really
        # asks for resting/sleeping/parking.
        if window is not None and any(word in text for word in ("休息", "睡觉", "熄火", "停车", "歇")):
            start, end = window
            hours = ((end - start) % 1440) / 60.0
            rest.update({"hours": max(rest["hours"] or 0, hours), "window_start_minute": start, "window_end_minute": end})
        if rest["hours"] is None:
            if "歇满4小时" in text or "满4小时" in text:
                rest["hours"] = 4
            elif "休息满5小时" in text or "满5小时" in text:
                rest["hours"] = 5
        return rest

    @staticmethod
    def _fallback_off_days(text: str) -> int:
        if not any(word in text for word in ("整天", "00:00~24:00", "停驶", "完全歇着")):
            return 0
        values: list[int] = []
        for pattern in (
            r"([0-9一二两三四五六七八九十]+)\s*(?:个)?整天",
            r"([0-9一二两三四五六七八九十]+)\s*(?:个)?整",
            r"(?:留|抽|至少|起码)[^0-9一二两三四五六七八九十]{0,6}([0-9一二两三四五六七八九十]+)\s*(?:天|日)",
        ):
            for match in re.finditer(pattern, text):
                value = ModelDecisionService._cn_day(match.group(1))
                if value is not None:
                    values.append(value)
        if values:
            return max(values)
        if "两" in text or "二" in text:
            return 2
        if "三" in text:
            return 3
        if "一" in text:
            return 1
        return 0

    @staticmethod
    def _fallback_pickup_limit(text: str) -> float | None:
        if "空驶" not in text:
            return None
        match = re.search(r"空驶[^0-9一二三四五六七八九十]*([0-9]+)", text)
        if match:
            return float(match.group(1))
        cn_match = re.search(r"空驶[^0-9一二三四五六七八九十]*([一二两三四五六七八九十百]+)", text)
        if cn_match:
            value = ModelDecisionService._cn_day(cn_match.group(1))
            if value:
                return float(value)
        return None

    @staticmethod
    def _fallback_haul_limit(text: str) -> float | None:
        if not any(word in text for word in ("装卸距离", "装货点至卸货点", "单笔货", "干线", "运距")):
            return None
        match = re.search(r"(?:装卸距离|装货点至卸货点|单笔货[^，。；;]{0,12}距离|干线|运距)[^0-9一二两三四五六七八九十百]*([0-9]+)", text)
        if match:
            return float(match.group(1))
        cn_match = re.search(r"(?:装卸距离|装货点至卸货点|单笔货[^，。；;]{0,12}距离|干线|运距)[^0-9一二两三四五六七八九十百]*([一二两三四五六七八九十百]+)", text)
        if cn_match:
            value = ModelDecisionService._cn_day(cn_match.group(1))
            if value:
                return float(value)
        return None

    @staticmethod
    def _fallback_geo_fence(text: str) -> dict[str, float] | None:
        match = re.search(
            r"北纬\s*([0-9]+(?:\.[0-9]+)?)\s*(?:至|到|-|~|—|–)\s*([0-9]+(?:\.[0-9]+)?)[\s\S]{0,20}?东经\s*([0-9]+(?:\.[0-9]+)?)\s*(?:至|到|-|~|—|–)\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        )
        if not match:
            return None
        lat1, lat2, lng1, lng2 = [float(x) for x in match.groups()]
        return {"lat_min": min(lat1, lat2), "lat_max": max(lat1, lat2), "lng_min": min(lng1, lng2), "lng_max": max(lng1, lng2)}

    @staticmethod
    def _fallback_forbidden_circles(text: str) -> list[dict[str, Any]]:
        circles: list[dict[str, Any]] = []
        if not any(word in text for word in ("不得进入", "禁止进入", "禁入", "别进")):
            return circles
        for match in re.finditer(
            r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)].{0,24}?半径\s*([0-9]+(?:\.[0-9]+)?)\s*公里",
            text,
        ):
            circles.append({"center": [float(match.group(1)), float(match.group(2))], "radius_km": float(match.group(3))})
        return circles

    @staticmethod
    def _fallback_required_cargos(text: str) -> list[dict[str, Any]]:
        if not any(word in text for word in ("熟货", "指定货源", "必须接", "必接")):
            return []
        cargo_ids = re.findall(r"(?:编号|货源编号|货源)\s*([0-9]{4,})", text)
        coords = extract_coordinates(text)
        online_minute = None
        time_match = re.search(
            r"((?:\d{4}-)?[0-9]{1,2}-[0-9]{1,2}\s+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?|"
            r"(?:\d{4}年)?[0-9一二两三四五六七八九十]{1,3}月[0-9一二两三四五六七八九十]{1,3}日\s*[0-9一二两三四五六七八九十]{1,3}(?:点|:|：)[0-9]{0,2})",
            text,
        )
        if time_match:
            online_minute = minute_offset(time_match.group(1))
        return [
            {"cargo_id": cargo_id, "pickup_point": list(coords[0]) if coords else None, "online_minute": online_minute}
            for cargo_id in cargo_ids
        ]

    @staticmethod
    def _fallback_temporary_events(text: str) -> list[dict[str, Any]]:
        if not ModelDecisionService._looks_like_personal_commitment(text):
            return []
        coords = extract_coordinates(text)
        times = [minute_offset(x) for x in ModelDecisionService._datetime_mentions(text)]
        times = [x for x in times if x is not None]
        if len(coords) < 2 or not times:
            return []
        pickup_minute = min(times)
        release_minute = max(times)
        if release_minute <= pickup_minute:
            release_minute = pickup_minute + 12 * 60
        return [{"pickup_point": list(coords[0]), "home_point": list(coords[1]), "pickup_minute": pickup_minute, "release_minute": release_minute}]

    @staticmethod
    def _datetime_mentions(text: str) -> list[str]:
        return re.findall(
            r"(?:\d{4}-[0-9]{1,2}-[0-9]{1,2}\s+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?|"
            r"(?:\d{4}年)?[0-9一二两三四五六七八九十]{1,3}月[0-9一二两三四五六七八九十]{1,3}日\s*"
            r"[0-9一二两三四五六七八九十]{1,3}(?:点|:|：)[0-9]{0,2})",
            text,
        )

    @staticmethod
    def _looks_like_personal_commitment(text: str) -> bool:
        personal_words = ("家事", "配偶", "老家", "旧家", "新家", "搬家", "孩子", "老人", "医院", "学校", "证件", "年审", "维修", "婚礼", "婚宴", "赴宴", "接亲", "接人", "赴约")
        commitment_words = ("解决前", "接上", "接到", "返回", "送到", "陪同", "必须", "不得", "之前", "待到", "停留", "释放")
        return any(word in text for word in personal_words) and any(word in text for word in commitment_words)

    @staticmethod
    def _fallback_visit_frequency(text: str) -> dict[str, Any]:
        for segment in re.split(r"[。；;\n]", text):
            if "至少" not in segment or "自然日" not in segment or "到" not in segment:
                continue
            coords = extract_coordinates(segment)
            match = re.search(r"至少\s*([0-9一二两三四五六七八九十]+)\s*个?不同", segment)
            radius_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*公里", segment)
            if match and coords:
                required = ModelDecisionService._cn_day(match.group(1)) or 0
                return {"required_days": required, "point": list(coords[0]), "radius_km": float(radius_match.group(1)) if radius_match else 1.0}
        return {"required_days": 0, "point": None, "radius_km": 1.0}

    @staticmethod
    def _fallback_required_region(text: str, coords: list[tuple[float, float]]) -> dict[str, Any]:
        if not ("不同的日子" in text and any(word in text for word in ("接够", "起码", "至少"))):
            return {"region": None, "min_days": 0, "point": None}
        region = ModelDecisionService._first_region_near(text, ("不同的日子", "接够", "起码", "至少"))
        match = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?不同的日子", text)
        return {
            "region": region,
            "min_days": ModelDecisionService._cn_day(match.group(1)) if match else 0,
            "point": list(coords[0]) if coords else None,
        }

    @staticmethod
    def _fallback_scheduled_visits(text: str, coords: list[tuple[float, float]]) -> list[dict[str, Any]]:
        visits: list[dict[str, Any]] = []
        for m in re.finditer(r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月\s*([0-9一二两三四五六七八九十]{1,3})\s*[号日]", text):
            day = _date_to_day_index(*m.groups())
            if day is None or not coords:
                continue
            line_start = max(text.rfind("\n", 0, m.start()), text.rfind("。", 0, m.start()), text.rfind("；", 0, m.start())) + 1
            line_end_candidates = [idx for idx in (text.find("\n", m.end()), text.find("。", m.end()), text.find("；", m.end())) if idx >= 0]
            line_end = min(line_end_candidates) if line_end_candidates else min(len(text), m.start() + 180)
            context = text[line_start:line_end]
            local_coords = extract_coordinates(context) or coords
            if any(word in context for word in ("先过", "再到", "赶到", "赴宴")):
                deadline = ModelDecisionService._parse_deadline_minute(context)
                if deadline is None:
                    visits.append({"day": day, "point": list(local_coords[0]), "wait_minutes": 1, "arrive_before_minute": None, "confidence": "unknown_time"})
                    continue
                first_point = coords[0] if ("先过" in context or "捎上" in context) else local_coords[0]
                final_point = local_coords[-1]
                if len(local_coords) >= 2:
                    first_point = local_coords[0]
                    final_point = local_coords[-1]
                visits.append({"day": day, "point": list(first_point), "wait_minutes": 1, "arrive_before_minute": deadline})
                visits.append({"day": day, "point": list(final_point), "wait_minutes": 1, "arrive_before_minute": deadline})
                continue
            wait = 120 if any(word in context for word in ("两小时", "2小时", "下午两点")) else 1
            deadline = ModelDecisionService._parse_deadline_minute(context)
            visits.append({"day": day, "point": list(local_coords[0]), "wait_minutes": wait, "arrive_before_minute": deadline})
        return visits

    @staticmethod
    def _cn_day(value: str) -> int | None:
        value = str(value).strip()
        value = value.replace("个", "")
        if value.isdigit():
            return int(value)
        table = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if value in table:
            return table[value]
        if value.startswith("百"):
            return 100 + (table.get(value[1:], 0) if len(value) > 1 else 0)
        if "百" in value:
            left, right = value.split("百", 1)
            return table.get(left, 1) * 100 + (ModelDecisionService._cn_day(right) or 0 if right else 0)
        if value.startswith("十") and len(value) == 2:
            return 10 + table.get(value[1], 0)
        if "十" in value:
            left, right = value.split("十", 1)
            return table.get(left, 1) * 10 + (table.get(right, 0) if right else 0)
        return None

    @staticmethod
    def _parse_daily_time_window(text: str) -> tuple[int, int] | None:
        normalized = (
            text.replace("零点", "0点")
            .replace("凌晨", "")
            .replace("清晨", "")
            .replace("早上", "")
            .replace("早", "")
            .replace("上午", "")
            .replace("中午十二点", "12点")
            .replace("中午", "")
        )
        normalized = re.sub(r"晚上\s*([0-9一二两三四五六七八九十]+)\s*点", lambda m: f"{(ModelDecisionService._cn_day(m.group(1)) or 0) + 12}点", normalized)
        normalized = re.sub(r"下午\s*([0-9一二两三四五六七八九十]+)\s*点", lambda m: f"{(ModelDecisionService._cn_day(m.group(1)) or 0) + (0 if (ModelDecisionService._cn_day(m.group(1)) or 0) >= 12 else 12)}点", normalized)
        match = re.search(
            r"([0-9一二两三四五六七八九十]{1,3})\s*(?:点|:：)?\s*(?:以后|后)?\s*(?:到|至|-|~|—)\s*(?:次日|第二天)?\s*([0-9一二两三四五六七八九十]{1,3})\s*(?:点|:：)",
            normalized,
        )
        if not match:
            return None
        start = ModelDecisionService._cn_day(match.group(1))
        end = ModelDecisionService._cn_day(match.group(2))
        if start is None or end is None:
            return None
        return ((start % 24) * 60, (end % 24) * 60)

    @staticmethod
    def _parse_deadline_minute(text: str) -> int | None:
        if "十二点前" in text or "12点前" in text:
            return 12 * 60
        match = re.search(r"([0-9一二两三四五六七八九十]{1,3})\s*点\s*前", text)
        if not match:
            return None
        hour = ModelDecisionService._cn_day(match.group(1))
        return None if hour is None else (hour % 24) * 60

    @staticmethod
    def _first_region_near(text: str, anchors: tuple[str, ...]) -> str | None:
        positions = [text.find(anchor) for anchor in anchors if text.find(anchor) >= 0]
        center = min(positions) if positions else 0
        context = text[max(0, center - 80) : center + 80]
        regions = re.findall(r"(?:在|到|去|往|进|跑|接够|至少|起码)([\u4e00-\u9fa5]{2,10})(?:市|区|县|镇|城|港|湾|岛|山|园|场|口|的货|货源|订单|方向|附近|周边)?", context)
        if regions:
            region = regions[0].strip("，,。；;、和与及 ")
            return region[:-1] if len(region) > 2 and region.endswith(("市", "区", "县")) else region
        return None

    def _normalize_profile(self, profile: dict[str, Any], preferences: list[dict[str, Any]]) -> dict[str, Any]:
        # Local parsing is a safety floor. LLM output may add structure, but must
        # not erase critical constraints like rest windows, off-days or forbidden
        # cargo/regions.
        base = self._empty_profile(preferences)
        llm_profile = {k: v for k, v in profile.items() if v is not None}
        for key, value in llm_profile.items():
            if key not in {
                "avoid_cargo_keywords",
                "avoid_regions",
                "daily_rest",
                "required_off_days",
                "pickup_deadhead_max_km",
                "monthly_deadhead_limit_km",
                "max_haul_km",
                "first_order_deadline_minute",
                "daily_order_limit",
                "geo_fence_bounds",
                "forbidden_circles",
                "required_cargos",
                "temporary_events",
                "long_sequence_commitments",
                "cumulative_time_penalty_rules",
                "raw_preference_coverage_audit",
                "visit_frequency",
                "required_region_cargo_days",
                "scheduled_visits",
                "preference_points",
            }:
                base[key] = value
        base["avoid_cargo_keywords"] = self._merge_list(base.get("avoid_cargo_keywords"), llm_profile.get("avoid_cargo_keywords"))
        base["avoid_regions"] = self._merge_list(base.get("avoid_regions"), llm_profile.get("avoid_regions"))
        base["avoid_regions"] = self._clean_avoid_regions(base.get("avoid_regions"), preferences)
        scheduled_supported = self._explicit_scheduled_visits_supported(preferences)
        base["scheduled_visits"] = self._merge_list(base.get("scheduled_visits"), llm_profile.get("scheduled_visits")) if scheduled_supported else []
        base["preference_points"] = self._merge_list(base.get("preference_points"), llm_profile.get("preference_points"))
        geo_constraints_supported = self._explicit_geo_constraints_supported(preferences)
        if geo_constraints_supported:
            base["forbidden_circles"] = self._merge_list(base.get("forbidden_circles"), llm_profile.get("forbidden_circles"))
        else:
            base["forbidden_circles"] = []
        base["required_cargos"] = self._merge_list(base.get("required_cargos"), llm_profile.get("required_cargos")) if self._explicit_required_cargo_supported(preferences) else []
        base["temporary_events"] = self._merge_list(base.get("temporary_events"), llm_profile.get("temporary_events")) if self._explicit_temporary_event_supported(preferences) else []
        base["long_sequence_commitments"] = self._merge_list(base.get("long_sequence_commitments"), llm_profile.get("long_sequence_commitments")) if self._explicit_sequence_supported(preferences) else []
        base["cumulative_time_penalty_rules"] = self._merge_list(base.get("cumulative_time_penalty_rules"), llm_profile.get("cumulative_time_penalty_rules"))
        if isinstance(llm_profile.get("daily_rest"), dict):
            base["daily_rest"] = self._merge_daily_rest(base.get("daily_rest", {}), llm_profile["daily_rest"])
        base["daily_rest"] = self._suppress_no_action_daily_rest(base.get("daily_rest", {}), preferences)
        llm_off_days = self._as_int(llm_profile.get("required_off_days")) if self._explicit_off_days_supported(preferences) else None
        base["required_off_days"] = max(
            self._as_int(base.get("required_off_days")) or 0,
            llm_off_days or 0,
        )
        base["pickup_deadhead_max_km"] = self._min_optional_float(
            base.get("pickup_deadhead_max_km"),
            llm_profile.get("pickup_deadhead_max_km"),
        )
        base["monthly_deadhead_limit_km"] = self._min_optional_float(
            base.get("monthly_deadhead_limit_km"),
            llm_profile.get("monthly_deadhead_limit_km"),
        )
        base["max_haul_km"] = self._min_optional_float(base.get("max_haul_km"), llm_profile.get("max_haul_km"))
        first_deadline = self._min_optional_int(base.get("first_order_deadline_minute"), llm_profile.get("first_order_deadline_minute"))
        base["first_order_deadline_minute"] = first_deadline if first_deadline is None else max(0, min(1439, first_deadline))
        daily_limit_values = [value for value in (self._as_int(base.get("daily_order_limit")), self._as_int(llm_profile.get("daily_order_limit"))) if value]
        base["daily_order_limit"] = min(daily_limit_values) if daily_limit_values else None
        if geo_constraints_supported and not isinstance(base.get("geo_fence_bounds"), dict) and isinstance(llm_profile.get("geo_fence_bounds"), dict):
            base["geo_fence_bounds"] = llm_profile.get("geo_fence_bounds")
        if not geo_constraints_supported:
            base["geo_fence_bounds"] = None
        if self._explicit_required_region_cargo_supported(preferences) and isinstance(llm_profile.get("required_region_cargo_days"), dict):
            base["required_region_cargo_days"] = self._merge_required_region(
                base.get("required_region_cargo_days", {}),
                llm_profile["required_region_cargo_days"],
            )
        if not isinstance(base.get("daily_rest"), dict):
            base["daily_rest"] = {"hours": None, "window_start_minute": None, "window_end_minute": None}
        if not isinstance(base.get("required_region_cargo_days"), dict):
            base["required_region_cargo_days"] = {"region": None, "min_days": 0, "point": None}
        if self._explicit_visit_frequency_supported(preferences) and isinstance(llm_profile.get("visit_frequency"), dict):
            base["visit_frequency"] = self._merge_visit_frequency(base.get("visit_frequency", {}), llm_profile["visit_frequency"])
        if not isinstance(base.get("visit_frequency"), dict):
            base["visit_frequency"] = {"required_days": 0, "point": None, "radius_km": 1.0}
        for key in (
            "avoid_cargo_keywords",
            "avoid_regions",
            "scheduled_visits",
            "preference_points",
            "forbidden_circles",
            "required_cargos",
            "temporary_events",
            "long_sequence_commitments",
            "cumulative_time_penalty_rules",
            "raw_preference_coverage_audit",
            "preference_cards",
            "duplicate_preference_groups",
            "unknown_preferences",
            "unknown_attribute_tags",
            "unknown_preference_groups",
            "dynamic_preference_rules",
        ):
            if not isinstance(base.get(key), list):
                base[key] = []
        base["dynamic_preference_rules"] = self._merge_list(
            base.get("dynamic_preference_rules", []),
            self._fallback_dynamic_preference_rules(preferences),
        )
        base["dynamic_preference_rules"] = self._normalize_dynamic_rules(base.get("dynamic_preference_rules", []), preferences)
        base["preference_points"] = [p for p in base.get("preference_points", []) if self._valid_point(p)]
        base["preference_points"] = [
            p for p in base["preference_points"]
            if self._point_allowed_by_profile(float(p[0]), float(p[1]), {**base, "preference_points": []})
        ]
        cleaned_visits = []
        for item in base.get("scheduled_visits", []):
            if isinstance(item, dict) and self._valid_point(item.get("point")):
                cleaned_visits.append(item)
        base["scheduled_visits"] = cleaned_visits
        required_region = base.get("required_region_cargo_days", {})
        if isinstance(required_region, dict) and not self._valid_point(required_region.get("point")):
            required_region["point"] = None
        try:
            base["required_off_days"] = max(0, int(base.get("required_off_days") or 0))
        except (TypeError, ValueError):
            base["required_off_days"] = 0
        return base

    @staticmethod
    def _explicit_geo_constraints_supported(preferences: list[dict[str, Any]]) -> bool:
        text = "\n".join(str(item.get("content") or "") for item in preferences if isinstance(item, dict))
        if not text:
            return False
        fence_words = ("北纬", "东经", "经纬度范围", "地理围栏", "围栏", "活动范围", "限定范围")
        circle_words = ("不得进入", "禁止进入", "禁入", "别进", "禁行区域", "半径")
        return any(word in text for word in fence_words) or any(word in text for word in circle_words)

    @staticmethod
    def _preference_text(preferences: list[dict[str, Any]]) -> str:
        return "\n".join(str(item.get("content") or "") for item in preferences if isinstance(item, dict))

    @classmethod
    def _explicit_off_days_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        if not text:
            return False
        return any(word in text for word in ("整天", "整日", "全天", "停驶", "完全歇着", "放空一整天", "自然月内")) and any(
            word in text for word in ("不接单", "不接货", "不外跑", "不空车", "休息", "歇着", "放空")
        )

    @classmethod
    def _explicit_required_cargo_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        return bool(text) and any(word in text for word in ("指定熟货", "熟货源编号", "货源编号", "指定货源", "必接", "必须接"))

    @classmethod
    def _explicit_temporary_event_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        return bool(text) and any(word in text for word in ("临时约定", "家事", "急事", "婚车", "婚礼", "婚宴", "赴宴", "搬家", "旧家", "新家", "指定日期", "须先", "再返回", "至少待到", "必须在原地"))

    @classmethod
    def _explicit_sequence_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        if not text:
            return False
        return any(
            word in text
            for word in (
                "须先",
                "先到",
                "随后",
                "再",
                "然后",
                "至少待到",
                "停到",
                "必须在原地",
                "方可再出车",
                "继续出车",
                "原地停留",
                "接上",
                "返回老家",
                "婚礼",
                "婚宴",
                "赴宴",
                "搬家",
                "旧家",
                "新家",
            )
        )

    @classmethod
    def _explicit_scheduled_visits_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        if not text:
            return False
        return any(word in text for word in ("到访", "到过", "不同的自然日", "指定日期", "几号", "号到", "当天到", "盘库", "清库存", "停一趟", "当天得"))

    @classmethod
    def _explicit_visit_frequency_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        if not text:
            return False
        return ("到过" in text or "到访" in text) and any(word in text for word in ("不同的自然日", "至少", "起码", "不少于"))

    @classmethod
    def _explicit_required_region_cargo_supported(cls, preferences: list[dict[str, Any]]) -> bool:
        text = cls._preference_text(preferences)
        if not text:
            return False
        return any(word in text for word in ("区域货源", "方向货", "那一路", "指定区域", "指定地区")) or ("至少" in text and "货" in text and "天" in text and "到过" not in text)

    @classmethod
    def _clean_avoid_regions(cls, regions: Any, preferences: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if not isinstance(regions, list):
            return []
        text = cls._preference_text(preferences or [])
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in regions:
            if not isinstance(item, dict):
                continue
            region = re.sub(r"(跑|去|接|派|查|那边|这边)$", "", str(item.get("region") or "").strip()).strip()
            if len(region) < 2 or any(bad in region for bad in ("货地", "卸货", "这天", "交警", "连续", "每天")):
                continue
            if text and cls._region_is_positive_geo_scope(region, text):
                continue
            days = item.get("days")
            cleaned = {"region": region, "days": days if isinstance(days, list) and days else None}
            key = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result

    @staticmethod
    def _region_is_positive_geo_scope(region: str, text: str) -> bool:
        """Avoid treating a preferred operating area as a forbidden region."""
        if not region or region not in text:
            return False
        for segment in re.split(r"[。；;\n]", text):
            if region not in segment:
                continue
            positive_scope = any(word in segment for word in ("范围内", "活动范围", "限定范围", "地理围栏", "围栏", "不出市", "不出城", "就在", "始终在", "须在"))
            explicit_negative = any(word in segment for word in ("不去", "不往", "避开", "别去", "不要去", "禁止进入", "不得进入", "禁入", "绕开", "远离"))
            negative_cargo = bool(re.search(rf"(?:不接|不要接|不拉|拒绝)[^。；;\n]{{0,16}}{re.escape(region)}[^。；;\n]{{0,16}}(?:货|订单|货源)", segment))
            if positive_scope and not explicit_negative and not negative_cargo:
                return True
        return False

    def _normalize_dynamic_rules(self, raw_rules: Any, preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(raw_rules, list):
            return []
        known_ids = {f"P{idx:03d}" for idx, _ in enumerate(preferences, start=1)}
        preference_by_id = {f"P{idx:03d}": item for idx, item in enumerate(preferences, start=1)}
        normalized: list[dict[str, Any]] = []
        seen_rule_keys: set[str] = set()
        for index, rule in enumerate(raw_rules, start=1):
            if not isinstance(rule, dict):
                continue
            match = rule.get("match", {})
            if not isinstance(match, dict):
                match = {}
            effect = str(rule.get("effect") or "penalize").strip().lower()
            if effect not in {"hard_reject", "penalize", "boost"}:
                effect = "penalize"
            source_id = str(rule.get("source_preference_id") or "")
            if source_id not in known_ids:
                source_id = ""
            source_pref = preference_by_id.get(source_id, {})
            source_penalty = self._as_float(source_pref.get("penalty_amount")) if isinstance(source_pref, dict) else None
            per_violation_penalty = self._as_float(rule.get("per_violation_penalty_yuan"))
            if per_violation_penalty is None:
                per_violation_penalty = self._as_float(rule.get("penalty_yuan"))
            if per_violation_penalty is None:
                per_violation_penalty = source_penalty if source_penalty is not None else 0.0
            expected_violations = self._as_float(rule.get("expected_violations_per_month"))
            if expected_violations is None:
                expected_violations = self._as_float(rule.get("repeat_count"))
            if expected_violations is None:
                expected_violations = 1.0
            expected_violations = max(1.0, min(80.0, expected_violations))
            multiplier = self._as_float(rule.get("penalty_multiplier"))
            if multiplier is None:
                multiplier = 1.0
            multiplier = max(0.1, min(50.0, multiplier))
            per_action_penalty = max(0.0, min(300000.0, per_violation_penalty * multiplier))
            monthly_risk_hint = max(0.0, min(300000.0, per_violation_penalty * expected_violations * multiplier))
            source_text = str(source_pref.get("content") or "") if isinstance(source_pref, dict) else ""
            if self._dynamic_rule_duplicates_structured_periodic_preference(source_text, match):
                continue
            normalized_match = self._normalize_dynamic_match(match)
            if self._dynamic_rule_is_broad_time_only(normalized_match) and not self._text_is_no_action_window(source_text):
                # A pure time-window hard rule without cargo/region/point scope
                # would match every order in that period. Structured rest,
                # off-day and no-action tools handle those preferences safely.
                continue
            if self._dynamic_rule_should_be_soft_daily_window(rule, normalized_match, source_text, per_action_penalty):
                effect = "penalize"
                rule = dict(rule)
                if str(rule.get("severity") or "").lower() == "critical":
                    rule["severity"] = "high"
            dedupe_key = json.dumps(
                {
                    "source": source_id,
                    "effect": effect,
                    "match": normalized_match,
                    "penalty": round(float(per_violation_penalty or 0.0), 2),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if dedupe_key in seen_rule_keys:
                continue
            seen_rule_keys.add(dedupe_key)
            normalized.append(
                {
                    "id": str(rule.get("id") or f"R{index:03d}")[:24],
                    "source_preference_id": source_id,
                    "source_unknown_tag_ids": self._string_list(rule.get("source_unknown_tag_ids"), 16),
                    "label": str(rule.get("label") or "llm_dynamic_preference")[:80],
                    "match": normalized_match,
                    "effect": effect,
                    "penalty_yuan": max(0.0, min(300000.0, per_violation_penalty)),
                    "per_violation_penalty_yuan": max(0.0, min(300000.0, per_violation_penalty)),
                    "expected_violations_per_month": expected_violations,
                    "penalty_multiplier": multiplier,
                    "per_action_penalty_yuan": per_action_penalty,
                    "effective_penalty_yuan": per_action_penalty,
                    "monthly_risk_hint_yuan": monthly_risk_hint,
                    "severity": str(rule.get("severity") or "medium").lower() if str(rule.get("severity") or "medium").lower() in {"critical", "high", "medium", "low"} else "medium",
                    "confidence": max(0.0, min(1.0, self._as_float(rule.get("confidence")) or 0.5)),
                    "repeat_count": max(1, min(80, int(math.ceil(expected_violations)))),
                    "description": str(rule.get("description") or "")[:160],
                }
            )
        return normalized[:40]

    @staticmethod
    def _dynamic_rule_duplicates_structured_periodic_preference(source_text: str, match: dict[str, Any]) -> bool:
        """Skip broad LLM rules already represented by structured periodic fields."""
        if not source_text or not isinstance(match, dict):
            return False
        is_full_off_day = any(word in source_text for word in ("整天不接单", "整天", "停驶", "完全歇着"))
        is_daily_rest = any(word in source_text for word in ("连续", "歇满", "休息", "停车歇", "熄火"))
        if not (is_full_off_day or is_daily_rest):
            return False
        has_specific_cargo_or_region = any(
            match.get(key)
            for key in ("cargo_name_contains", "start_city_contains", "end_city_contains", "start_or_end_city_contains")
        )
        has_specific_distance = match.get("max_pickup_km") is not None or match.get("max_haul_km") is not None or match.get("min_price") is not None
        has_specific_time = bool(match.get("time_window") or match.get("daily_time_window"))
        if has_specific_cargo_or_region or has_specific_distance or has_specific_time:
            return False
        # These preferences are handled by required_off_days/daily_rest progress
        # tools. A broad dynamic rule would penalize every profitable order.
        return True

    @staticmethod
    def _dynamic_rule_is_broad_time_only(match: dict[str, Any]) -> bool:
        scoped_keys = (
            "cargo_name_contains",
            "start_city_contains",
            "end_city_contains",
            "start_or_end_city_contains",
            "periodic_stop_required",
        )
        has_scope = any(bool(match.get(key)) for key in scoped_keys)
        has_numeric_scope = any(match.get(key) is not None for key in ("max_pickup_km", "max_haul_km", "min_price"))
        if has_scope or has_numeric_scope:
            return False
        return bool(match.get("time_window") or match.get("daily_time_window"))

    def _normalize_dynamic_match(self, match: dict[str, Any]) -> dict[str, Any]:
        return {
            "cargo_name_contains": self._string_list(match.get("cargo_name_contains"), 12),
            "start_city_contains": self._string_list(match.get("start_city_contains"), 12),
            "end_city_contains": self._string_list(match.get("end_city_contains"), 12),
            "start_or_end_city_contains": self._string_list(match.get("start_or_end_city_contains"), 12),
            "max_pickup_km": self._as_float(match.get("max_pickup_km")),
            "max_haul_km": self._as_float(match.get("max_haul_km")),
            "min_price": self._as_float(match.get("min_price")),
            "time_window": match.get("time_window") if isinstance(match.get("time_window"), dict) else {},
            "daily_time_window": match.get("daily_time_window") if isinstance(match.get("daily_time_window"), dict) else {},
            "periodic_stop_required": match.get("periodic_stop_required") if isinstance(match.get("periodic_stop_required"), dict) else {},
        }

    def _dynamic_rule_should_be_soft_daily_window(
        self,
        rule: dict[str, Any],
        match: dict[str, Any],
        source_text: str,
        per_action_penalty: float,
    ) -> bool:
        """Night/home daily windows are soft economic tradeoffs unless truly huge."""
        daily_window = match.get("daily_time_window") if isinstance(match.get("daily_time_window"), dict) else {}
        start = self._as_int(daily_window.get("start_minute_of_day"))
        end = self._as_int(daily_window.get("end_minute_of_day"))
        if start is None or end is None or start == end:
            return False
        has_specific_order_match = any(
            bool(match.get(key))
            for key in (
                "cargo_name_contains",
                "start_city_contains",
                "end_city_contains",
                "start_or_end_city_contains",
                "time_window",
                "periodic_stop_required",
            )
        )
        has_specific_numeric_match = any(match.get(key) is not None for key in ("max_pickup_km", "max_haul_km", "min_price"))
        if has_specific_order_match or has_specific_numeric_match:
            return False
        text = " ".join(
            str(value or "")
            for value in (
                source_text,
                rule.get("label"),
                rule.get("description"),
                rule.get("reason"),
            )
        )
        looks_like_home_or_no_action = any(
            word in text
            for word in (
                "回家",
                "到家",
                "在家",
                "进家门",
                "自家",
                "home",
                "night",
                "夜间",
                "夜里",
                "晚上",
                "不接单",
                "不空驶",
                "不空跑",
                "不赶路",
                "no-action",
            )
        )
        crosses_midnight_or_night = start >= 18 * 60 or end <= 10 * 60 or start > end
        penalty_soft_enough = float(per_action_penalty or 0.0) < float(os.environ.get("AGENT_SOFT_DAILY_WINDOW_HARD_REJECT_MAX_YUAN", "5000") or 5000)
        return penalty_soft_enough and (looks_like_home_or_no_action or crosses_midnight_or_night)

    def _fallback_dynamic_preference_rules(self, preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        for index, item in enumerate(preferences, start=1):
            text = str(item.get("content", ""))
            source_id = f"P{index:03d}"
            penalty = self._as_float(item.get("penalty_amount")) or 500.0
            cap = self._as_float(item.get("penalty_cap"))
            severity = "high" if penalty >= 1200 or (cap is not None and cap >= 10000) else "medium"
            for window_index, window in enumerate(self._extract_daily_time_windows(text), start=1):
                if any(word in text for word in ("别安排", "不要接", "不能接", "不接单", "不空车", "不空驶", "不赶路", "导致", "还在", "固定要", "每天", "每晚", "夜间", "夜里", "晚上", "凌晨", "扣")):
                    rules.append(
                        {
                            "id": f"R{index:03d}_DAYWIN_{window_index}",
                            "source_preference_id": source_id,
                            "source_unknown_tag_ids": [f"U{index:03d}_time"],
                            "label": "fallback_daily_time_window",
                            "match": {"daily_time_window": window},
                            "effect": "penalize",
                            "per_violation_penalty_yuan": penalty,
                            "expected_violations_per_month": 20,
                            "penalty_multiplier": 1.5 if severity == "high" else 1.0,
                            "severity": severity,
                            "confidence": 0.75,
                            "description": text[:120],
                        }
                    )
            keywords = self._extract_dynamic_avoid_keywords(text)
            if keywords:
                rules.append(
                    {
                        "id": f"R{index:03d}_KW",
                        "source_preference_id": source_id,
                        "source_unknown_tag_ids": [f"U{index:03d}_keyword"],
                        "label": "fallback_avoid_keywords",
                        "match": {"cargo_name_contains": keywords},
                        "effect": "hard_reject" if penalty >= 1800 else "penalize",
                        "per_violation_penalty_yuan": penalty,
                        "expected_violations_per_month": 12,
                        "penalty_multiplier": 1.2,
                        "severity": "high" if penalty >= 1200 else "medium",
                        "confidence": 0.7,
                        "description": text[:120],
                    }
                )
            periodic = self._extract_periodic_stop_requirement(text)
            if periodic:
                rules.append(
                    {
                        "id": f"R{index:03d}_PERIODIC_STOP",
                        "source_preference_id": source_id,
                        "source_unknown_tag_ids": [f"U{index:03d}_periodic_stop"],
                        "label": "fallback_periodic_stop_required",
                        "match": {"periodic_stop_required": periodic},
                        "effect": "penalize",
                        "per_violation_penalty_yuan": penalty,
                        "expected_violations_per_month": max(1, math.ceil(HORIZON_DAYS / float(periodic.get("period_days", 7)))),
                        "penalty_multiplier": 1.5,
                        "severity": "high",
                        "confidence": 0.75,
                        "description": text[:120],
                    }
                )
        return rules[:80]

    @classmethod
    def _extract_daily_time_windows(cls, text: str) -> list[dict[str, int]]:
        windows: list[dict[str, int]] = []
        for match in re.finditer(r"(\d{1,2})[:：](\d{2})\s*(?:到|至|-|~|—)\s*(\d{1,2})[:：](\d{2})", text):
            windows.append(
                {
                    "start_minute_of_day": int(match.group(1)) * 60 + int(match.group(2)),
                    "end_minute_of_day": int(match.group(3)) * 60 + int(match.group(4)),
                }
            )
        for match in re.finditer(r"(\d{1,2})[:：](\d{2})\s*(?:以后|后)", text):
            windows.append({"start_minute_of_day": int(match.group(1)) * 60 + int(match.group(2)), "end_minute_of_day": 1440})
        if "晚上九点后" in text or "晚上9点后" in text or "21点后" in text:
            windows.append({"start_minute_of_day": 21 * 60, "end_minute_of_day": 1440})
        if "上午十点到十点半" in text or "10点到10点半" in text:
            windows.append({"start_minute_of_day": 10 * 60, "end_minute_of_day": 10 * 60 + 30})
        parsed = cls._parse_daily_time_window(text)
        if parsed is not None:
            windows.append({"start_minute_of_day": parsed[0], "end_minute_of_day": parsed[1]})
        result: list[dict[str, int]] = []
        seen: set[tuple[int, int]] = set()
        for window in windows:
            start = max(0, min(1439, int(window["start_minute_of_day"])))
            end = max(1, min(1440, int(window["end_minute_of_day"])))
            key = (start, end)
            if key not in seen and start != end:
                result.append({"start_minute_of_day": start, "end_minute_of_day": end})
                seen.add(key)
        return result

    @staticmethod
    def _extract_dynamic_avoid_keywords(text: str) -> list[str]:
        negative = ("不接", "不碰", "不要", "不能", "不想", "一律", "都不", "拒绝", "别给", "扣")
        if not any(word in text for word in negative):
            return []
        keyword_aliases = {
            "海鲜": ["海鲜", "水产", "活鱼", "虾", "蟹"],
            "水产": ["海鲜", "水产", "活鱼", "虾", "蟹"],
            "冷库": ["冷库", "冷链"],
            "冷链": ["冷库", "冷链"],
            "镜面玻璃": ["玻璃", "易碎"],
            "古董家具": ["家具", "古董"],
            "易碎": ["易碎", "玻璃", "陶瓷"],
            "人工搬运": ["人工搬运", "搬运", "上楼", "背货"],
            "油漆": ["油漆", "化工"],
            "胶水": ["胶水", "化工"],
            "化工桶": ["化工", "化工塑料", "化工桶"],
            "香精": ["香精", "香料"],
            "香料": ["香精", "香料"],
            "地下": ["地下", "负一层"],
            "负一层": ["地下", "负一层"],
            "限高": ["限高", "坡道"],
            "窄巷": ["窄巷", "巷口", "村口"],
            "散件": ["散件", "裸货", "编织袋"],
            "裸货": ["散件", "裸货", "编织袋"],
            "加急": ["加急", "限时", "迟到赔付"],
            "限时": ["加急", "限时", "迟到赔付"],
            "电子料": ["电子料", "芯片", "主板", "精密仪器", "服务器"],
            "芯片": ["电子料", "芯片", "主板", "精密仪器", "服务器"],
            "菜市场": ["菜市场", "批发市场", "市场"],
            "批发市场": ["菜市场", "批发市场", "市场"],
            "轮渡": ["轮渡", "渡口", "码头", "上船"],
            "码头": ["轮渡", "渡口", "码头", "上船"],
        }
        keywords: set[str] = set()
        for marker, aliases in keyword_aliases.items():
            if marker in text:
                keywords.update(aliases)
        try:
            keywords.update(DriverProfileTool._avoid_keywords(text))
        except Exception:
            pass
        for quoted in re.findall(r"[「“\"']([^」“”\"']{1,20})[」”\"']", text):
            if 2 <= len(quoted.strip()) <= 16:
                keywords.add(quoted.strip())
        return sorted(keywords)[:16]

    @staticmethod
    def _extract_periodic_stop_requirement(text: str) -> dict[str, int] | None:
        if not any(word in text for word in ("每过三天", "每3天", "连续多天没有", "每周至少一次", "整周没有")):
            return None
        period_days = 7 if "每周" in text or "整周" in text else 3
        min_wait = 120 if any(word in text for word in ("两小时", "2小时", "二小时")) else 60
        return {"period_days": period_days, "min_wait_minutes": min_wait}

    @staticmethod
    def _string_list(value: Any, limit: int) -> list[str]:
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = [str(item) for item in value]
        else:
            values = []
        return [item.strip()[:40] for item in values if item and item.strip()][:limit]

    @staticmethod
    def _merge_list(left: Any, right: Any) -> list[Any]:
        result: list[Any] = []
        seen: set[str] = set()
        for source in (left, right):
            if not isinstance(source, list):
                continue
            for item in source:
                key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
                if key not in seen:
                    result.append(item)
                    seen.add(key)
        return result

    def _merge_daily_rest(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base) if isinstance(base, dict) else {"hours": None, "window_start_minute": None, "window_end_minute": None}
        base_hours = self._as_float(merged.get("hours"))
        extra_hours = self._as_float(extra.get("hours"))
        if extra_hours is not None:
            merged["hours"] = max(base_hours or 0.0, extra_hours)
        for key in ("window_start_minute", "window_end_minute"):
            if merged.get(key) is None and extra.get(key) is not None:
                merged[key] = extra.get(key)
        return merged

    def _suppress_no_action_daily_rest(self, rest: Any, preferences: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = dict(rest) if isinstance(rest, dict) else {"hours": None, "window_start_minute": None, "window_end_minute": None}
        hours = self._as_float(normalized.get("hours"))
        start = self._as_int(normalized.get("window_start_minute"))
        end = self._as_int(normalized.get("window_end_minute"))
        if hours is None:
            return normalized
        explicit_matching_rest_window = False
        matched_plain_no_action_window = False
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if any(word in text for word in ("休息", "睡觉", "熄火", "停车", "歇")):
                for window in self._extract_daily_time_windows(text):
                    w_start = self._as_int(window.get("start_minute_of_day"))
                    w_end = self._as_int(window.get("end_minute_of_day"))
                    if w_start == start and w_end == end:
                        explicit_matching_rest_window = True
            if not self._text_is_no_action_window(text):
                continue
            if any(word in text for word in ("休息", "睡觉", "熄火", "停车", "歇")):
                continue
            for window in self._extract_daily_time_windows(text):
                w_start = self._as_int(window.get("start_minute_of_day"))
                w_end = self._as_int(window.get("end_minute_of_day"))
                if w_start is not None and w_end is not None:
                    window_hours = ((w_end - w_start) % 1440) / 60.0
                    if abs(window_hours - float(hours)) <= 0.25:
                        matched_plain_no_action_window = True
                if w_start == start and w_end == end:
                    return {"hours": None, "window_start_minute": None, "window_end_minute": None}
        if start is None or end is None:
            if matched_plain_no_action_window and not explicit_matching_rest_window:
                return {"hours": None, "window_start_minute": None, "window_end_minute": None}
            return normalized
        if not explicit_matching_rest_window:
            normalized["window_start_minute"] = None
            normalized["window_end_minute"] = None
        return normalized

    def _merge_required_region(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base) if isinstance(base, dict) else {"region": None, "min_days": 0, "point": None}
        if not merged.get("region") and extra.get("region"):
            merged["region"] = extra.get("region")
        merged["min_days"] = max(self._as_int(merged.get("min_days")) or 0, self._as_int(extra.get("min_days")) or 0)
        if not self._valid_point(merged.get("point")) and self._valid_point(extra.get("point")):
            merged["point"] = extra.get("point")
        return merged

    def _merge_visit_frequency(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base) if isinstance(base, dict) else {"required_days": 0, "point": None, "radius_km": 1.0}
        merged["required_days"] = max(self._as_int(merged.get("required_days")) or 0, self._as_int(extra.get("required_days")) or 0)
        if not self._valid_point(merged.get("point")) and self._valid_point(extra.get("point")):
            merged["point"] = extra.get("point")
        radius = self._min_optional_float(merged.get("radius_km"), extra.get("radius_km"))
        merged["radius_km"] = radius if radius is not None else 1.0
        return merged

    def _min_optional_float(self, left: Any, right: Any) -> float | None:
        values = [value for value in (self._as_float(left), self._as_float(right)) if value is not None]
        return min(values) if values else None

    def _min_optional_int(self, left: Any, right: Any) -> int | None:
        values = [value for value in (self._as_int(left), self._as_int(right)) if value is not None]
        return min(values) if values else None

    def _profile_without_seen_cargo_commitments(self, driver_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        seen = self._runtime_seen_cargos(driver_id) | self._failed_cargos_by_driver.setdefault(driver_id, set())
        if not seen:
            return profile
        filtered = dict(profile)
        filtered["required_cargos"] = [
            item for item in (profile.get("required_cargos") or [])
            if not isinstance(item, dict) or str(item.get("cargo_id") or "") not in seen
        ]
        sequences: list[dict[str, Any]] = []
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                sequences.append(sequence)
                continue
            steps = []
            for step in sequence.get("steps", []) or []:
                if isinstance(step, dict) and str(step.get("step_type") or "").lower() == "take_cargo" and str(step.get("cargo_id") or "") in seen:
                    continue
                steps.append(step)
            copied = dict(sequence)
            copied["steps"] = steps
            sequences.append(copied)
        filtered["long_sequence_commitments"] = sequences
        return filtered

    def _profile_without_forbidden_required_cargos(self, driver_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        required = profile.get("required_cargos") or []
        if not required or not profile.get("forbidden_circles"):
            return profile
        failed = self._failed_cargos_by_driver.setdefault(driver_id, set())
        kept: list[Any] = []
        changed = False
        for item in required:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            point = item.get("pickup_point")
            if self._valid_point(point):
                assert isinstance(point, (list, tuple))
                if not self._point_allowed_by_profile(float(point[0]), float(point[1]), profile):
                    cargo_id = str(item.get("cargo_id") or "")
                    if cargo_id:
                        failed.add(cargo_id)
                    changed = True
                    continue
            kept.append(item)
        if not changed:
            return profile
        filtered = dict(profile)
        filtered["required_cargos"] = kept
        return filtered

    def _profile_scoped_for_current_time(self, profile: dict[str, Any], current_minute: int) -> dict[str, Any]:
        """Hide far-future one-shot commitments from immediate market positioning."""
        notice = self._commitment_notice_minutes()
        activation_cutoff = current_minute + notice
        scoped = dict(profile)
        inactive_points: list[tuple[float, float]] = []

        active_sequences: list[dict[str, Any]] = []
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                active_sequences.append(sequence)
                continue
            steps = [step for step in sequence.get("steps", []) or [] if isinstance(step, dict)]
            first_minute = self._first_commitment_minute(steps)
            if first_minute is not None and first_minute > activation_cutoff:
                inactive_points.extend(self._points_from_steps(steps))
                continue
            active_sequences.append(sequence)
        scoped["long_sequence_commitments"] = active_sequences

        active_events: list[dict[str, Any]] = []
        for event in profile.get("temporary_events", []) or []:
            if not isinstance(event, dict):
                active_events.append(event)
                continue
            pickup_minute = self._as_int(event.get("pickup_minute"))
            if pickup_minute is not None and pickup_minute > activation_cutoff:
                for key in ("pickup_point", "home_point"):
                    point = self._point_tuple(event.get(key))
                    if point is not None:
                        inactive_points.append(point)
                continue
            active_events.append(event)
        scoped["temporary_events"] = active_events

        active_penalties: list[dict[str, Any]] = []
        for rule in profile.get("cumulative_time_penalty_rules", []) or []:
            if not isinstance(rule, dict):
                active_penalties.append(rule)
                continue
            window_start = self._as_int(rule.get("window_start_minute"))
            if window_start is not None and window_start > activation_cutoff:
                point = self._point_tuple(rule.get("required_point"))
                if point is not None:
                    inactive_points.append(point)
                continue
            active_penalties.append(rule)
        scoped["cumulative_time_penalty_rules"] = active_penalties

        if inactive_points:
            scoped["preference_points"] = [
                point
                for point in profile.get("preference_points", []) or []
                if self._valid_point(point) and not self._point_matches_any(point, inactive_points)
            ]
        return scoped

    def _first_commitment_minute(self, steps: list[dict[str, Any]]) -> int | None:
        values: list[int] = []
        for step in steps:
            for key in ("earliest_minute", "deadline_minute", "hold_until_minute"):
                value = self._as_int(step.get(key))
                if value is not None:
                    values.append(value)
        return min(values) if values else None

    def _points_from_steps(self, steps: list[dict[str, Any]]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for step in steps:
            point = self._point_tuple(step.get("point"))
            if point is not None:
                points.append(point)
        return points

    def _point_matches_any(self, point: Any, targets: list[tuple[float, float]], radius_km: float = 1.2) -> bool:
        current = self._point_tuple(point)
        if current is None:
            return False
        lat, lng = current
        return any(haversine_km(lat, lng, target[0], target[1]) <= radius_km for target in targets)

    def _point_tuple(self, point: Any) -> tuple[float, float] | None:
        if not self._valid_point(point):
            return None
        assert isinstance(point, (list, tuple))
        return (float(point[0]), float(point[1]))

    def _profile_special_action(self, driver_id: str, status: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any] | None:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)

        no_action_window = self._dynamic_daily_no_action_window_action(current_minute, profile)
        if no_action_window is not None:
            return no_action_window
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        start = self._as_int(rest.get("window_start_minute"))
        end = self._as_int(rest.get("window_end_minute"))
        minute_of_day = current_minute % 1440
        if start is not None and end is not None:
            if self._inside_daily_window(minute_of_day, start, end):
                planned = action("wait", {"duration_minutes": max(1, self._minutes_until_window_end(minute_of_day, end))})
                planned["reason_code"] = "daily_no_action_window_guard"
                return planned
            if start == 0 and end > 0 and minute_of_day >= max(0, 1440 - min(360, end)):
                planned = action("wait", {"duration_minutes": max(1, 1440 + end - minute_of_day)})
                planned["reason_code"] = "daily_rest_window_guard"
                return planned

        calendar_profile = self._calendar_profile(driver_id, profile)
        commitment = self._commitment_sequence_tool.next_required_action(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=calendar_profile,
            history=self._recent_history_records(driver_id),
        )
        commitment_action = commitment.get("action") if isinstance(commitment, dict) else None
        if isinstance(commitment_action, dict):
            return commitment_action

        calendar_report = self._task_calendar_tool.calendar_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=calendar_profile,
            history=self._recent_history_records(driver_id),
        )
        calendar_action = calendar_report.get("recommended_action") if isinstance(calendar_report, dict) else None
        if isinstance(calendar_action, dict):
            return calendar_action

        cumulative_penalty = self._cumulative_penalty_window_action(current_minute, lat, lng, profile)
        if cumulative_penalty is not None:
            return cumulative_penalty

        daily_rest = self._daily_rest_action(driver_id, current_minute, profile)
        if daily_rest is not None:
            return daily_rest

        time_tasks = self._time_task_progress_tool.progress_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=self._profile_without_seen_cargo_commitments(driver_id, profile),
            history=self._recent_history_records(driver_id),
        )
        time_action = time_tasks.get("recommended_action") if isinstance(time_tasks, dict) else None
        if isinstance(time_action, dict):
            return time_action

        event = self._temporary_event_action(current_minute, lat, lng, profile)
        if event is not None:
            return event

        required = self._required_cargo_action(driver_id, current_minute, lat, lng, profile)
        if required is not None:
            return required

        visit_frequency = self._visit_frequency_action(driver_id, current_minute, lat, lng, profile)
        if visit_frequency is not None:
            return visit_frequency

        visit = self._scheduled_visit_action(driver_id, current_minute, lat, lng, profile)
        if visit is not None:
            return visit

        off_day = self._off_day_action(driver_id, current_minute, profile)
        if off_day is not None:
            return off_day

        periodic = self._periodic_stop_action(driver_id, current_minute, lat, lng, profile)
        if periodic is not None:
            return periodic

        night_home = self._night_home_action(current_minute, lat, lng, profile)
        if night_home is not None:
            return night_home

        safe = self._geo_safety_action(lat, lng, profile)
        if safe is not None:
            return safe

        return None

    def _calendar_profile(self, driver_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        calendar_profile = self._profile_without_seen_cargo_commitments(driver_id, profile)
        required = calendar_profile.get("required_region_cargo_days", {})
        if isinstance(required, dict):
            region = str(required.get("region") or "")
            copied = dict(required)
            if region:
                region_days = self._runtime_bucket(driver_id).get("required_region_days", {})
                days = region_days.get(region, set()) if isinstance(region_days, dict) else set()
                if isinstance(days, set):
                    copied["completed_day_indices"] = sorted(int(day) for day in days)
                    copied["completed_days"] = len(days)
            calendar_profile["required_region_cargo_days"] = copied
        return calendar_profile

    def _dynamic_daily_no_action_window_action(self, current_minute: int, profile: dict[str, Any]) -> dict[str, Any] | None:
        minute_of_day = current_minute % 1440
        for rule in profile.get("dynamic_preference_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}
            window = match.get("daily_time_window") if isinstance(match.get("daily_time_window"), dict) else {}
            start = self._as_int(window.get("start_minute_of_day"))
            end = self._as_int(window.get("end_minute_of_day"))
            if start is None or end is None:
                continue
            duration = (end - start) if end > start else (1440 - start + end)
            if duration >= 18 * 60:
                continue
            if not self._dynamic_rule_is_no_action_window(rule, profile):
                continue
            if self._inside_daily_window(minute_of_day, start, end):
                planned = action("wait", {"duration_minutes": max(1, self._minutes_until_window_end(minute_of_day, end))})
                planned["reason_code"] = "daily_no_action_window_guard"
                return planned
        return None

    def _visible_daily_no_action_window_action(self, current_minute: int, preferences: list[dict[str, Any]]) -> dict[str, Any] | None:
        minute_of_day = current_minute % 1440
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if not self._text_is_no_action_window(text):
                continue
            for window in self._extract_daily_time_windows(text):
                start = self._as_int(window.get("start_minute_of_day")) if isinstance(window, dict) else None
                end = self._as_int(window.get("end_minute_of_day")) if isinstance(window, dict) else None
                if start is None or end is None:
                    continue
                if self._inside_daily_window(minute_of_day, start, end):
                    planned = action("wait", {"duration_minutes": max(1, self._minutes_until_window_end(minute_of_day, end))})
                    planned["reason_code"] = "daily_no_action_window_guard"
                    return planned
        return None

    def _profile_no_action_windows(self, profile: dict[str, Any]) -> list[tuple[int, int]]:
        windows: list[tuple[int, int]] = []
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        start = self._as_int(rest.get("window_start_minute"))
        end = self._as_int(rest.get("window_end_minute"))
        if start is not None and end is not None and start != end:
            duration = (end - start) if end > start else (1440 - start + end)
            if duration < 18 * 60:
                windows.append((max(0, min(1439, start)), max(1, min(1440, end))))
        for rule in profile.get("dynamic_preference_rules", []) or []:
            if not isinstance(rule, dict) or not self._dynamic_rule_is_no_action_window(rule, profile):
                continue
            match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}
            window = match.get("daily_time_window") if isinstance(match.get("daily_time_window"), dict) else {}
            start = self._as_int(window.get("start_minute_of_day"))
            end = self._as_int(window.get("end_minute_of_day"))
            if start is None or end is None or start == end:
                continue
            duration = (end - start) if end > start else (1440 - start + end)
            if duration >= 18 * 60:
                continue
            windows.append((max(0, min(1439, start)), max(1, min(1440, end))))
        result: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for window in windows:
            if window not in seen:
                seen.add(window)
                result.append(window)
        return result

    def _interval_overlaps_no_action_window(self, start_minute: int, end_minute: int, profile: dict[str, Any]) -> bool:
        if end_minute <= start_minute:
            return False
        return any(self._daily_window_overlaps_interval(start_minute, end_minute, start, end) for start, end in self._profile_no_action_windows(profile))

    def _reposition_crosses_no_action_window(self, current_minute: int, travel_minutes: int, profile: dict[str, Any]) -> bool:
        buffer_minutes = int(os.environ.get("AGENT_REPOSITION_NO_ACTION_BUFFER_MINUTES", "5") or 5)
        return self._interval_overlaps_no_action_window(current_minute, current_minute + max(1, travel_minutes) + buffer_minutes, profile)

    @staticmethod
    def _text_is_no_action_window(text: str) -> bool:
        has_explicit_clock = any(
            word in text
            for word in ("每晚", "每夜", "夜间", "夜里", "晚上", "凌晨", "至次日", "到次日", "点至", "点到", "点前", "点后", "点", ":", "：")
        )
        if not has_explicit_clock:
            return False
        return any(
            word in text
            for word in ("不接单", "不得接单", "不拉活", "不接活", "不空车", "不空驶", "不赶路", "不行动", "不跑车", "睡觉", "禁行")
        )

    @staticmethod
    def _dynamic_rule_is_no_action_window(rule: dict[str, Any], profile: dict[str, Any]) -> bool:
        label = str(rule.get("label") or "").lower()
        source_id = str(rule.get("source_preference_id") or "")
        cards = profile.get("preference_cards", []) if isinstance(profile.get("preference_cards"), list) else []
        source_text = ""
        for card in cards:
            if isinstance(card, dict) and str(card.get("id") or "") == source_id:
                source_text = str(card.get("content") or "")
                break
        if source_text:
            return ModelDecisionService._text_is_no_action_window(source_text)
        return any(word in label for word in ("no-action", "no action", "sleep", "forbidden driving", "forbidden order"))

    def _night_home_action(self, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        home = self._daily_home_point(profile)
        if not self._valid_point(home):
            return None
        minute_of_day = current_minute % 1440
        night_start = int(os.environ.get("AGENT_NIGHT_HOME_START_MINUTE", str(20 * 60)) or 20 * 60)
        hard_start = int(os.environ.get("AGENT_NIGHT_HOME_HARD_START_MINUTE", str(23 * 60)) or 23 * 60)
        night_end = int(os.environ.get("AGENT_NIGHT_HOME_END_MINUTE", str(8 * 60)) or 8 * 60)
        distance = haversine_km(lat, lng, float(home[0]), float(home[1]))
        travel = distance_to_minutes(distance)
        deadline_abs = (current_minute // 1440) * 1440 + hard_start
        if minute_of_day < night_end:
            deadline_abs -= 1440
        leave_by = deadline_abs - travel - int(os.environ.get("AGENT_NIGHT_HOME_RETURN_BUFFER_MINUTES", "30") or 30)
        in_guard = minute_of_day >= night_start or minute_of_day < night_end or (distance > 1.0 and current_minute >= leave_by)
        if not in_guard:
            return None
        if distance > 1.0:
            if current_minute < leave_by and minute_of_day < hard_start:
                return None
            planned = action("reposition", {"latitude": float(home[0]), "longitude": float(home[1])})
            planned["reason_code"] = "night_home_deadline_return_guard" if current_minute >= leave_by else "night_home_return_guard"
            return planned
        wait_until = night_end if minute_of_day < night_end else 1440 + night_end
        planned = action("wait", {"duration_minutes": max(1, min(540, wait_until - minute_of_day))})
        planned["reason_code"] = "night_home_stay_guard"
        return planned

    @staticmethod
    def _home_return_is_soft_text(text: str) -> bool:
        if not any(word in text for word in ("尽量", "最好", "尽可能", "争取", "可以")):
            return False
        hard_markers = ("须", "必须", "务必", "一定", "不得不", "进家门", "车辆须", "要在", "得在")
        return not any(word in text for word in hard_markers)

    @staticmethod
    def _daily_home_point(profile: dict[str, Any]) -> list[float] | None:
        cards = profile.get("preference_cards", []) if isinstance(profile.get("preference_cards"), list) else []
        for card in cards:
            if not isinstance(card, dict):
                continue
            text = str(card.get("content") or "")
            if ModelDecisionService._home_return_is_soft_text(text):
                continue
            if not any(word in text for word in ("每天", "每日", "每晚", "夜间", "夜里", "23点前", "二十三点前")):
                continue
            if not any(word in text for word in ("自家位置", "到家", "回家", "进家门", "23点前")):
                continue
            coords = extract_coordinates(text)
            if coords:
                home_coord = coords[-1] if len(coords) >= 2 and any(word in text for word in ("老家", "进家门", "返回")) else coords[0]
                return [float(home_coord[0]), float(home_coord[1])]
        return None

    def _night_home_candidate_penalty(
        self,
        *,
        current_minute: int,
        finish_minute: int,
        end_point: tuple[float, float],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        home = self._daily_home_point(profile)
        if not self._valid_point(home):
            return {"penalty_yuan": 0.0, "risk_flags": []}
        night_start = int(os.environ.get("AGENT_NIGHT_HOME_HARD_START_MINUTE", str(23 * 60)) or 23 * 60)
        night_end = int(os.environ.get("AGENT_NIGHT_HOME_END_MINUTE", str(8 * 60)) or 8 * 60)
        penalty_unit = float(os.environ.get("AGENT_NIGHT_HOME_PENALTY_YUAN", "900") or 900)
        return_minutes = distance_to_minutes(haversine_km(end_point[0], end_point[1], float(home[0]), float(home[1])))
        home_ready_minute = finish_minute + return_minutes
        violations = 0
        details: list[dict[str, Any]] = []
        start_day = max(0, current_minute // 1440)
        end_day = min(HORIZON_DAYS, home_ready_minute // 1440 + 1)
        for day in range(start_day, end_day + 1):
            home_deadline = day * 1440 + night_start
            night_window_end = (day + 1) * 1440 + night_end
            if night_window_end <= current_minute:
                continue
            overlaps_night_order = max(current_minute, home_deadline) < min(finish_minute, night_window_end)
            cannot_return_home_by_deadline = finish_minute <= home_deadline < home_ready_minute
            still_away_after_deadline = home_deadline >= current_minute and home_deadline < finish_minute and home_ready_minute > home_deadline
            if overlaps_night_order or cannot_return_home_by_deadline or still_away_after_deadline:
                violations += 1
                details.append(
                    {
                        "day": day,
                        "home_deadline_minute": home_deadline,
                        "home_ready_minute": home_ready_minute,
                        "overlaps_night_order": overlaps_night_order,
                        "cannot_return_home_by_deadline": cannot_return_home_by_deadline,
                    }
                )
        if violations <= 0:
            return {"penalty_yuan": 0.0, "risk_flags": []}
        return {
            "penalty_yuan": round(violations * penalty_unit, 2),
            "risk_flags": ["night_home_return_risk"],
            "violations": violations,
            "details": details[:3],
            "home_point": home,
        }

    def _home_return_candidate_context(
        self,
        *,
        current_minute: int,
        finish_minute: int,
        end_point: tuple[float, float],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        home = self._daily_home_point(profile)
        if not self._valid_point(home):
            return {}
        deadline = self._next_home_return_deadline(current_minute, profile)
        if deadline is None:
            return {}
        buffer_minutes = int(os.environ.get("AGENT_HOME_RETURN_ORDER_BUFFER_MINUTES", "20") or 20)
        return_minutes = distance_to_minutes(haversine_km(end_point[0], end_point[1], float(home[0]), float(home[1]))) + buffer_minutes
        home_ready = finish_minute + return_minutes
        slack = deadline - home_ready
        return {
            "home_point": [float(home[0]), float(home[1])],
            "home_deadline_minute": deadline,
            "return_minutes_after_unload": return_minutes,
            "home_ready_minute": home_ready,
            "home_return_slack_minutes": slack,
            "can_return_home_by_deadline": slack >= 0,
        }

    def _next_home_return_deadline(self, current_minute: int, profile: dict[str, Any]) -> int | None:
        windows = self._profile_no_action_windows(profile)
        current_day = current_minute // 1440
        candidates: list[int] = []
        for start, _end in windows:
            abs_start = current_day * 1440 + start
            if abs_start <= current_minute:
                abs_start += 1440
            if abs_start - current_minute <= 36 * 60:
                candidates.append(abs_start)
        if candidates:
            return min(candidates)
        default_start = int(os.environ.get("AGENT_NIGHT_HOME_HARD_START_MINUTE", str(23 * 60)) or 23 * 60)
        deadline = current_day * 1440 + default_start
        if deadline <= current_minute:
            deadline += 1440
        return deadline

    def _cumulative_penalty_window_action(self, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        """Hard guard for visible per-minute/per-hour penalty windows."""
        selected: dict[str, Any] | None = None
        selected_leave_by: int | None = None
        for rule in profile.get("cumulative_time_penalty_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            point = rule.get("required_point")
            if not self._valid_point(point):
                continue
            start = self._as_int(rule.get("window_start_minute"))
            end = self._as_int(rule.get("window_end_minute"))
            rate = self._as_float(rule.get("rate_yuan_per_minute")) or 0.0
            if start is None or end is None or end <= start or rate <= 0:
                continue
            radius = self._as_float(rule.get("radius_km")) or 1.0
            distance = haversine_km(lat, lng, float(point[0]), float(point[1]))
            travel = distance_to_minutes(distance)
            leave_by = start - travel - 15
            active = start <= current_minute < end
            due_to_leave = leave_by <= current_minute < end
            if not active and not due_to_leave:
                continue
            if selected is not None and selected_leave_by is not None and leave_by >= selected_leave_by:
                continue
            selected_leave_by = leave_by
            if distance > radius:
                selected = action("reposition", {"latitude": float(point[0]), "longitude": float(point[1])})
                selected["reason_code"] = "cumulative_penalty_go_required_point"
            else:
                selected = action("wait", {"duration_minutes": max(1, min(240, end - current_minute))})
                selected["reason_code"] = "cumulative_penalty_stay_required_point"
        return selected

    def _visit_frequency_action(self, driver_id: str, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        vf = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        required = max(0, self._as_int(vf.get("required_days")) or 0)
        point = vf.get("point")
        if required <= 0 or not self._valid_point(point):
            return None
        assert isinstance(point, (list, tuple))
        target = (float(point[0]), float(point[1]))
        radius_km = float(vf.get("radius_km") or 1.0)
        visited = self._visited_days_near(driver_id, target, radius_km)
        current_day = current_minute // 1440
        at_point_now = haversine_km(lat, lng, target[0], target[1]) <= radius_km
        if current_day in visited:
            return None
        credited_count = len(visited) + (1 if at_point_now else 0)
        remaining_needed = required - credited_count
        if remaining_needed <= 0:
            if at_point_now:
                planned = action("wait", {"duration_minutes": 1})
                planned["reason_code"] = "monthly_visit_credit_today"
                return planned
            return None
        remaining_days = HORIZON_DAYS - current_day
        progress_target = self._monthly_visit_target_by_day(required, current_day)
        behind_pace = credited_count < progress_target
        urgent = remaining_days <= remaining_needed + 3
        distance = haversine_km(lat, lng, target[0], target[1])
        max_early_km = float(os.environ.get("AGENT_VISIT_EARLY_MAX_KM", "120") or 120)
        max_urgent_km = float(os.environ.get("AGENT_VISIT_URGENT_MAX_KM", "600") or 600)
        low_cost_idle_visit = behind_pace and distance <= max_early_km
        moderate_catchup = behind_pace and current_day >= 10 and distance <= max_urgent_km
        if not (urgent or low_cost_idle_visit or moderate_catchup):
            return None
        if not at_point_now:
            max_distance = max_urgent_km if urgent else max_early_km
            if moderate_catchup:
                max_distance = max(max_distance, max_urgent_km)
            if distance > max_distance:
                return None
            planned = action("reposition", {"latitude": target[0], "longitude": target[1]})
            planned["reason_code"] = "monthly_visit_low_cost_position" if (low_cost_idle_visit or moderate_catchup) and not urgent else "monthly_visit_due_position"
            planned["monthly_visit_context"] = {
                "completed_days": len(visited),
                "required_days": required,
                "missing_days": remaining_needed,
                "distance_km": round(distance, 2),
                "behind_pace": behind_pace,
                "progress_target_by_now": progress_target,
            }
            return planned
        planned = action("wait", {"duration_minutes": 1})
        planned["reason_code"] = "monthly_visit_credit_today"
        return planned

    def _daily_rest_action(self, driver_id: str, current_minute: int, profile: dict[str, Any]) -> dict[str, Any] | None:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        hours = self._as_float(rest.get("hours"))
        if hours is None or hours <= 0:
            return None
        required = int(math.ceil(hours * 60))
        current_day = current_minute // 1440
        day_end = (current_day + 1) * 1440
        completed = self._max_continuous_wait_minutes_for_day(driver_id, current_day)
        if completed >= required:
            return None
        remaining = max(1, required - completed)
        latest_start = day_end - required
        buffer_minutes = int(os.environ.get("AGENT_DAILY_REST_START_BUFFER_MINUTES", "60") or 60)
        if current_minute < latest_start - buffer_minutes:
            return None
        wait_minutes = min(max(remaining, required), max(1, day_end - current_minute))
        planned = action("wait", {"duration_minutes": wait_minutes})
        planned["reason_code"] = "daily_rest_required"
        planned["daily_rest_context"] = {
            "required_minutes": required,
            "completed_continuous_minutes": completed,
            "latest_start_minute": latest_start,
        }
        return planned

    def _temporary_event_action(self, current_minute: int, lat: float, lng: float, profile: dict[str, Any], force_notice: bool = False) -> dict[str, Any] | None:
        for event in profile.get("temporary_events", []):
            if not isinstance(event, dict):
                continue
            pickup = event.get("pickup_point"); home = event.get("home_point")
            pickup_minute = self._as_int(event.get("pickup_minute")); release_minute = self._as_int(event.get("release_minute"))
            if pickup_minute is None or release_minute is None or not self._valid_point(pickup) or not self._valid_point(home):
                continue
            to_pickup = distance_to_minutes(haversine_km(lat, lng, float(pickup[0]), float(pickup[1])))
            to_home = distance_to_minutes(haversine_km(float(pickup[0]), float(pickup[1]), float(home[0]), float(home[1])))
            pickup_wait = max(10, self._as_int(event.get("pickup_wait_minutes")) or 0)
            buffer_minutes = int(os.environ.get("AGENT_TEMPORARY_EVENT_RAW_BUFFER_MINUTES", "60") or 60)
            leave_by = pickup_minute - to_pickup - to_home - pickup_wait - buffer_minutes
            notice_start = pickup_minute - self._commitment_notice_minutes()
            in_window = leave_by <= current_minute < release_minute
            notice_due = force_notice and notice_start <= current_minute < release_minute
            if in_window or notice_due:
                if current_minute < pickup_minute and haversine_km(lat, lng, float(pickup[0]), float(pickup[1])) > 0.8:
                    planned = action("reposition", {"latitude": float(pickup[0]), "longitude": float(pickup[1])})
                    planned["reason_code"] = "temporary_event_notice_position" if current_minute < leave_by else "temporary_event_go_pickup"
                    return planned
                if current_minute < pickup_minute + pickup_wait:
                    planned = action("wait", {"duration_minutes": max(1, min(240, pickup_minute + pickup_wait - current_minute))})
                    planned["reason_code"] = "temporary_event_notice_wait" if current_minute < pickup_minute else "temporary_event_pickup_hold"
                    return planned
                if haversine_km(lat, lng, float(home[0]), float(home[1])) > 0.8:
                    planned = action("reposition", {"latitude": float(home[0]), "longitude": float(home[1])})
                    planned["reason_code"] = "temporary_event_return_home"
                    return planned
                planned = action("wait", {"duration_minutes": max(1, release_minute - current_minute)})
                planned["reason_code"] = "temporary_event_stay_home"
                return planned
        return None

    def _raw_night_home_action(self, current_minute: int, lat: float, lng: float, preferences: list[dict[str, Any]]) -> dict[str, Any] | None:
        minute_of_day = current_minute % 1440
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if self._home_return_is_soft_text(text):
                continue
            if not any(word in text for word in ("自家位置", "到家", "回家", "进家门", "在家")):
                continue
            if not any(word in text for word in ("每天", "每日", "每晚", "每夜", "夜间", "夜里", "当天23点至次日", "当天二十三点至次日")):
                continue
            if not any(word in text for word in ("点前", "夜间", "不接单", "不空跑", "不空驶", "至次日", "到次日")):
                continue
            coords = extract_coordinates(text)
            if not coords:
                continue
            windows = self._extract_daily_time_windows(text)
            start = self._as_int(windows[0].get("start_minute_of_day")) if windows else None
            end = self._as_int(windows[0].get("end_minute_of_day")) if windows else None
            hard_start = start if start is not None else int(os.environ.get("AGENT_NIGHT_HOME_HARD_START_MINUTE", str(23 * 60)) or 23 * 60)
            night_end = end if end is not None else int(os.environ.get("AGENT_NIGHT_HOME_END_MINUTE", str(8 * 60)) or 8 * 60)
            soft_start = max(0, hard_start - int(os.environ.get("AGENT_NIGHT_HOME_NOTICE_MINUTES", "180") or 180))
            in_guard = minute_of_day >= soft_start or minute_of_day < night_end
            home = coords[0]
            distance = haversine_km(lat, lng, float(home[0]), float(home[1]))
            if distance <= 1.0:
                if not in_guard:
                    continue
                if minute_of_day >= hard_start or minute_of_day < night_end:
                    wait_until = night_end if minute_of_day < night_end else 1440 + night_end
                    planned = action("wait", {"duration_minutes": max(1, min(540, wait_until - minute_of_day))})
                    planned["reason_code"] = "raw_night_home_stay_guard"
                    return planned
                continue
            travel = distance_to_minutes(distance)
            deadline_abs = (current_minute // 1440) * 1440 + hard_start
            if minute_of_day < night_end:
                deadline_abs -= 1440
            buffer_minutes = int(os.environ.get("AGENT_NIGHT_HOME_RETURN_BUFFER_MINUTES", "30") or 30)
            leave_by = deadline_abs - travel - buffer_minutes
            if current_minute < leave_by and not in_guard:
                continue
            if current_minute < leave_by and minute_of_day < hard_start:
                continue
            planned = action("reposition", {"latitude": float(home[0]), "longitude": float(home[1])})
            planned["reason_code"] = "raw_night_home_deadline_return_guard" if current_minute >= leave_by else "raw_night_home_return_guard"
            planned["home_deadline_minute"] = deadline_abs
            planned["home_point"] = [float(home[0]), float(home[1])]
            return planned
        return None

    def _commitment_notice_action(
        self,
        current_minute: int,
        lat: float,
        lng: float,
        profile: dict[str, Any],
        preferences: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        raw = self._raw_temporary_event_action(current_minute, lat, lng, preferences, force_notice=True)
        if raw is not None:
            return raw
        return self._temporary_event_action(current_minute, lat, lng, profile, force_notice=True)

    def _required_cargo_action(self, driver_id: str, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        for item in profile.get("required_cargos", []):
            if not isinstance(item, dict):
                continue
            cargo_id = str(item.get("cargo_id") or "")
            if not cargo_id or self._history_has_cargo(driver_id, cargo_id):
                continue
            pickup = item.get("pickup_point")
            online = self._as_int(item.get("online_minute"))
            if self._valid_point(pickup):
                pickup_distance = haversine_km(lat, lng, float(pickup[0]), float(pickup[1]))
                travel = distance_to_minutes(pickup_distance)
                buffer_minutes = max(120, int(os.environ.get("AGENT_REQUIRED_CARGO_BUFFER_MINUTES", "120") or 120))
                leave_by = None if online is None else online - travel - buffer_minutes
                if online is not None and leave_by is not None and current_minute < leave_by:
                    continue
                if online is not None and current_minute < online:
                    if pickup_distance > 3.0:
                        planned = action("reposition", {"latitude": float(pickup[0]), "longitude": float(pickup[1])})
                        planned["reason_code"] = "commitment_required_cargo_position"
                        return planned
                    planned = action("wait", {"duration_minutes": max(1, min(180, online - current_minute))})
                    planned["reason_code"] = "commitment_required_cargo_wait_window"
                    return planned
            if online is None or current_minute >= online:
                planned = action("take_order", {"cargo_id": cargo_id})
                planned["reason_code"] = "mandatory_required_cargo_take_now"
                return planned
        return None

    @staticmethod
    def _is_mandatory_required_cargo_take(candidate_action: dict[str, Any]) -> bool:
        return (
            isinstance(candidate_action, dict)
            and candidate_action.get("action") == "take_order"
            and str(candidate_action.get("reason_code") or "") == "mandatory_required_cargo_take_now"
        )

    @classmethod
    def _is_deterministic_special_action(cls, candidate_action: dict[str, Any]) -> bool:
        if cls._is_mandatory_required_cargo_take(candidate_action):
            return True
        reason = str(candidate_action.get("reason_code") or "") if isinstance(candidate_action, dict) else ""
        return reason in {
            "night_home_return_guard",
            "night_home_stay_guard",
            "cumulative_penalty_go_required_point",
            "cumulative_penalty_stay_required_point",
            "commitment_required_cargo_position",
            "commitment_required_cargo_wait_window",
            "commitment_required_cargo_wait_online",
        }

    def _geo_safety_action(self, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        bounds = profile.get("geo_fence_bounds")
        if isinstance(bounds, dict) and not self._inside_bounds(lat, lng, bounds):
            target = ((float(bounds["lat_min"]) + float(bounds["lat_max"])) / 2, (float(bounds["lng_min"]) + float(bounds["lng_max"])) / 2)
            return action("reposition", {"latitude": target[0], "longitude": target[1]})
        for circle in profile.get("forbidden_circles", []):
            if self._inside_forbidden_circle(lat, lng, circle):
                center = circle.get("center")
                radius = self._as_float(circle.get("radius_km")) or 0.0
                if self._valid_point(center):
                    return action("reposition", {"latitude": float(center[0]) + (radius + 5.0) / 111.0, "longitude": float(center[1])})
        return None

    def _scheduled_visit_action(self, driver_id: str, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        current_day = current_minute // 1440
        for item in profile.get("scheduled_visits", []):
            if not isinstance(item, dict):
                continue
            day = self._as_int(item.get("day"))
            point = item.get("point")
            if day is None or current_day != day or not self._valid_point(point):
                continue
            deadline = self._as_int(item.get("arrive_before_minute"))
            if deadline is not None and current_minute % 1440 > deadline:
                continue
            wait = max(1, self._as_int(item.get("wait_minutes")) or 1)
            p = (float(point[0]), float(point[1]))
            if self._waited_near_on_day(driver_id, day, p, wait):
                continue
            if haversine_km(lat, lng, p[0], p[1]) > 1.5:
                return action("reposition", {"latitude": p[0], "longitude": p[1]})
            return action("wait", {"duration_minutes": wait})
        return None

    def _off_day_action(self, driver_id: str, current_minute: int, profile: dict[str, Any]) -> dict[str, Any] | None:
        required = max(0, self._as_int(profile.get("required_off_days")) or 0)
        if required <= 0:
            return None
        current_day = current_minute // 1440
        active = self._active_days(driver_id)
        completed = sum(1 for day in range(current_day) if day not in active)
        remaining_needed = required - completed
        if remaining_needed <= 0:
            return None
        remaining_days = HORIZON_DAYS - current_day
        minute_of_day = current_minute % 1440
        if current_day not in active and remaining_days <= remaining_needed + 1:
            planned = action("wait", {"duration_minutes": max(1, min(MONTH_HORIZON_MINUTES, (current_day + 1) * 1440) - current_minute)})
            planned["reason_code"] = "monthly_full_off_day_urgent"
            return planned
        target_done_by_now = min(required, max(0, math.floor(required * current_day / max(1, HORIZON_DAYS))))
        behind_pace = completed < target_done_by_now
        if (
            current_day not in active
            and current_day + 1 < HORIZON_DAYS
            and minute_of_day >= 18 * 60
            and (behind_pace or remaining_days <= remaining_needed + 3)
        ):
            planned = action("wait", {"duration_minutes": max(1, min(MONTH_HORIZON_MINUTES, (current_day + 1) * 1440) - current_minute)})
            planned["reason_code"] = "monthly_full_off_day_idle_progress"
            return planned
        return None

    def _periodic_stop_action(self, driver_id: str, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        for rule in profile.get("dynamic_preference_rules", []):
            if not isinstance(rule, dict):
                continue
            match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}
            periodic = match.get("periodic_stop_required") if isinstance(match.get("periodic_stop_required"), dict) else {}
            if not periodic:
                continue
            period_days = max(1, self._as_int(periodic.get("period_days")) or 7)
            min_wait = max(30, self._as_int(periodic.get("min_wait_minutes")) or 120)
            period_minutes = period_days * 1440
            period_start = (current_minute // period_minutes) * period_minutes
            period_end = min(MONTH_HORIZON_MINUTES, period_start + period_minutes)
            if self._has_continuous_wait_in_period(driver_id, period_start, period_end, min_wait):
                continue
            latest_safe_start = max(period_start, period_end - max(min_wait, 6 * 60))
            if current_minute >= latest_safe_start:
                duration = min(min_wait, max(1, period_end - current_minute))
                home_leave_by = self._night_home_leave_by_minute(current_minute, lat, lng, profile)
                if home_leave_by is not None and current_minute < home_leave_by < current_minute + duration:
                    duration = max(1, home_leave_by - current_minute)
                return action("wait", {"duration_minutes": duration})
        return None

    def _night_home_leave_by_minute(self, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> int | None:
        home = self._daily_home_point(profile)
        if not self._valid_point(home):
            return None
        distance = haversine_km(lat, lng, float(home[0]), float(home[1]))
        if distance <= 1.0:
            return None
        hard_start = int(os.environ.get("AGENT_NIGHT_HOME_HARD_START_MINUTE", str(23 * 60)) or 23 * 60)
        night_end = int(os.environ.get("AGENT_NIGHT_HOME_END_MINUTE", str(8 * 60)) or 8 * 60)
        deadline_abs = (current_minute // 1440) * 1440 + hard_start
        if current_minute % 1440 < night_end:
            deadline_abs -= 1440
        buffer = int(os.environ.get("AGENT_NIGHT_HOME_RETURN_BUFFER_MINUTES", "30") or 30)
        return deadline_abs - distance_to_minutes(distance) - buffer

    def _query_cargo(self, driver_id: str, lat: float, lng: float, profile: dict[str, Any] | None = None) -> dict[str, Any]:
        k = self._cargo_k
        if isinstance(profile, dict) and self._needs_deeper_cargo_scan(profile):
            k = max(k, self._int_env("AGENT_CONSTRAINED_CARGO_K", 300, k, int(self._config["runtime_limits"].get("cargo_k_max", 600))))
        try:
            return self._api.query_cargo(driver_id, lat, lng, k=k)
        except TypeError:
            return self._api.query_cargo(driver_id, lat, lng)
        except Exception as exc:
            self._logger.info("query cargo failed: %s", exc)
            return {"items": []}

    @staticmethod
    def _needs_deeper_cargo_scan(profile: dict[str, Any]) -> bool:
        return bool(
            isinstance(profile.get("geo_fence_bounds"), dict)
            or profile.get("avoid_cargo_keywords")
            or profile.get("max_haul_km")
            or profile.get("pickup_deadhead_max_km")
        )

    def _build_candidates(self, driver_id: str, status: dict[str, Any], items: list[Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)
        truck_length = str(status.get("truck_length", ""))
        cost_per_km = float(status.get("cost_per_km", 1.5) or 1.5)
        failed = self._failed_cargos_by_driver.setdefault(driver_id, set())
        reject_counts: dict[str, int] = {}

        def reject(reason: str) -> None:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

        day = current_minute // 1440
        accepted_today = self._accepted_orders_on_day(driver_id, day)
        daily_order_limit = self._as_int(profile.get("daily_order_limit"))
        if daily_order_limit is not None and accepted_today >= daily_order_limit:
            self._debug_candidate_build(driver_id, current_minute, items, [], [], {"daily_order_limit": len(items)})
            return []
        first_order_deadline = self._as_int(profile.get("first_order_deadline_minute"))
        monthly_deadhead_limit = self._as_float(profile.get("monthly_deadhead_limit_km"))
        monthly_deadhead_used = self._monthly_deadhead_km(driver_id) if monthly_deadhead_limit is not None else 0.0
        required_region_rule = profile.get("required_region_cargo_days", {}) if isinstance(profile.get("required_region_cargo_days"), dict) else {}
        required_region_name = str(required_region_rule.get("region") or "")
        required_region_min_days = max(0, self._as_int(required_region_rule.get("min_days")) or 0)
        required_region_done_days = 0
        if required_region_name:
            region_days = self._runtime_bucket(driver_id).get("required_region_days", {})
            days = region_days.get(required_region_name, set()) if isinstance(region_days, dict) else set()
            required_region_done_days = len(days) if isinstance(days, set) else 0
        history_summary = self._history_summary(driver_id)
        history_records = self._history_records(driver_id)
        preferences = preference_items(list(status.get("preferences", []) or []))
        preferences = self._merge_list(preferences, self._load_raw_preferences(driver_id))
        cards = profile.get("preference_cards", []) if isinstance(profile.get("preference_cards"), list) else []
        preferences.extend(
            {
                "content": str(card.get("content") or ""),
                "penalty_amount": card.get("penalty_amount"),
                "penalty_cap": card.get("penalty_cap"),
            }
            for card in cards
            if isinstance(card, dict) and card.get("content")
        )
        calendar_profile = self._calendar_profile(driver_id, profile)
        time_task_report = self._time_task_progress_tool.progress_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=calendar_profile,
            history=history_records,
        )
        calendar_report = self._task_calendar_tool.calendar_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=calendar_profile,
            history=history_records,
        )
        region_cargo_plan = calendar_report.get("required_region_cargo_plan", {}) if isinstance(calendar_report, dict) else {}
        candidates: list[dict[str, Any]] = []
        for wrapped in items:
            cargo = wrapped.get("cargo", {}) if isinstance(wrapped, dict) else {}
            cargo_id = str(cargo.get("cargo_id", ""))
            if not cargo_id or cargo_id in failed:
                reject("failed_or_missing_id")
                continue
            truck_options = [str(x) for x in cargo.get("truck_length", []) or []]
            if truck_options and truck_length and truck_length not in truck_options:
                reject("truck_length_mismatch")
                continue
            name = str(cargo.get("cargo_name", ""))
            if any(str(k) and str(k) in name for k in profile.get("avoid_cargo_keywords", [])):
                reject("avoid_cargo_keyword")
                continue
            start = cargo.get("start", {}) or {}
            end = cargo.get("end", {}) or {}
            start_city = str(start.get("city", ""))
            end_city = str(end.get("city", ""))
            matches_required_region = bool(required_region_name and (required_region_name in start_city or required_region_name in end_city))
            cargo_meta = self._runtime_bucket(driver_id).get("cargo_meta", {})
            if isinstance(cargo_meta, dict):
                cargo_meta[cargo_id] = {
                    "start_city": start_city,
                    "end_city": end_city,
                    "required_region_match": required_region_name if matches_required_region else "",
                }
            try:
                start_lat = float(start["lat"]); start_lng = float(start["lng"])
                end_lat = float(end["lat"]); end_lng = float(end["lng"])
            except (KeyError, TypeError, ValueError):
                reject("invalid_geo")
                continue
            self._remember_region_point(profile, start_city, start_lat, start_lng)
            self._remember_region_point(profile, end_city, end_lat, end_lng)
            pickup_km = float(wrapped.get("distance_km", haversine_km(lat, lng, start_lat, start_lng)) or 0.0)
            pickup_limit = self._as_float(profile.get("pickup_deadhead_max_km"))
            if pickup_limit is not None and pickup_km > pickup_limit + 0.5:
                reject("pickup_deadhead_limit")
                continue
            pickup_minutes = distance_to_minutes(pickup_km)
            ready = current_minute + pickup_minutes
            remove_minute = minute_offset(str(cargo.get("remove_time", ""))) if cargo.get("remove_time") else None
            remove_hard_buffer = int(os.environ.get("AGENT_CARGO_REMOVE_HARD_BUFFER_MINUTES", "3") or 3)
            if remove_minute is not None and remove_minute <= current_minute + remove_hard_buffer:
                failed.add(cargo_id)
                reject("listing_expires_too_soon")
                continue
            load_time = cargo.get("load_time")
            load_end = None
            if isinstance(load_time, list) and len(load_time) == 2:
                load_start = minute_offset(str(load_time[0])); load_end = minute_offset(str(load_time[1]))
                if load_end is not None and ready > load_end:
                    failed.add(cargo_id); reject("miss_load_window"); continue
                if load_start is not None:
                    ready = max(ready, load_start)
            duration = int(cargo.get("cost_time_minutes", 0) or 0)
            finish = ready + duration
            if finish > MONTH_HORIZON_MINUTES:
                reject("beyond_month_horizon")
                continue
            if self._schedule_sensitive_long_order_block(current_minute, finish, profile):
                reject("schedule_sensitive_overnight_long_order")
                continue
            geo_allowed = self._geo_candidate_allowed(lat, lng, start_lat, start_lng, end_lat, end_lng, profile, ready, finish)
            if (
                isinstance(region_cargo_plan, dict)
                and region_cargo_plan.get("active_today")
                and required_region_name
                and required_region_done_days < required_region_min_days
                and day not in set(region_cargo_plan.get("completed_day_indices", []) or [])
                and not matches_required_region
            ):
                reject("required_region_day")
                continue
            calendar_blocks_candidate = self._task_calendar_tool.blocks_candidate(calendar_report, finish)
            calendar_soft_penalty = 0.0
            if calendar_blocks_candidate and not self._calendar_block_can_be_softened(calendar_report, profile):
                reject("task_calendar_deadline")
                continue
            if self._interval_overlaps_no_action_window(current_minute, finish, profile):
                reject("no_action_window_overlap")
                continue
            if self._profile_commitment_blocks_candidate(driver_id, current_minute, finish, end_lat, end_lng, profile):
                reject("profile_commitment_deadline")
                continue
            if self._raw_temporary_event_blocks_candidate(current_minute, finish, end_lat, end_lng, preferences):
                reject("raw_temporary_event_deadline")
                continue
            if self._blocks_scheduled_visit(end_lat, end_lng, finish, profile):
                reject("scheduled_visit_deadline")
                continue
            if self._violates_avoid_region(start_city, end_city, ready, finish, profile):
                reject("avoid_region")
                continue
            haul_km = haversine_km(start_lat, start_lng, end_lat, end_lng)
            haul_limit = self._as_float(profile.get("max_haul_km"))
            if haul_limit is not None and haul_km > haul_limit + 0.5:
                reject("haul_limit")
                continue
            price = float(cargo.get("price", 0.0) or 0.0)
            net = price - (pickup_km + haul_km) * cost_per_km
            if calendar_blocks_candidate:
                calendar_soft_penalty = self._calendar_soft_penalty_yuan(calendar_report, profile, net)
                if calendar_soft_penalty is None:
                    reject("task_calendar_deadline")
                    continue
            geo_soft_penalty = 0.0
            geo_return_cost = 0.0
            if not geo_allowed:
                geo_soft_penalty = self._geo_soft_penalty_yuan(profile, net)
                if geo_soft_penalty is None:
                    reject("geo_candidate_block")
                    continue
                geo_return_cost = self._geo_return_cost_yuan(end_lat, end_lng, profile, cost_per_km)
            hours = max(1.0, (finish - current_minute) / 60.0)
            hourly = net / hours
            hourly_weight = float(os.environ.get("AGENT_HOURLY_SCORE_WEIGHT", "10") or 10)
            score = net + hourly_weight * hourly - 0.5 * pickup_km - float(geo_soft_penalty or 0.0) - float(geo_return_cost or 0.0) - float(calendar_soft_penalty or 0.0)
            home_context = self._home_return_candidate_context(
                current_minute=current_minute,
                finish_minute=finish,
                end_point=(end_lat, end_lng),
                profile=profile,
            )
            if home_context and not home_context.get("can_return_home_by_deadline"):
                reject("daily_home_return_deadline")
                continue
            if home_context.get("can_return_home_by_deadline"):
                slack = float(home_context.get("home_return_slack_minutes", 0.0) or 0.0)
                score += float(os.environ.get("AGENT_HOME_COMPATIBLE_ORDER_BOOST_YUAN", "900") or 900) + min(600.0, slack * 2.0)
            if first_order_deadline is not None and accepted_today == 0 and ready // 1440 == day and ready % 1440 >= first_order_deadline:
                # This is a soft tradeoff: a late first order may be worth taking
                # when its profit clearly covers the preference penalty.
                score -= float(os.environ.get("AGENT_FIRST_ORDER_LATE_PENALTY", "900") or 900)
            if monthly_deadhead_limit is not None:
                over = max(0.0, monthly_deadhead_used + pickup_km - monthly_deadhead_limit)
                if over > 0:
                    monthly_deadhead_score_cap = float(os.environ.get("AGENT_MONTHLY_DEADHEAD_SCORE_CAP_YUAN", "2500") or 2500)
                    score -= min(
                        monthly_deadhead_score_cap,
                        over * float(os.environ.get("AGENT_MONTHLY_DEADHEAD_PENALTY_PER_KM", "10") or 10),
                    )
            if required_region_name and required_region_done_days < required_region_min_days:
                remaining_needed = required_region_min_days - required_region_done_days
                remaining_days = max(1, HORIZON_DAYS - day)
                behind_pace = required_region_done_days < math.floor(required_region_min_days * day / max(1, HORIZON_DAYS))
                urgent_region_progress = behind_pace or remaining_days <= remaining_needed * 6
                if matches_required_region:
                    score += 8000.0 if urgent_region_progress else 5000.0
                elif urgent_region_progress:
                    score -= 1800.0
            anchor = self._nearest_profile_point_km(end_lat, end_lng, profile)
            if anchor is not None:
                anchor_base = float(os.environ.get("AGENT_ANCHOR_SCORE_BASE", "600") or 600)
                anchor_km_penalty = float(os.environ.get("AGENT_ANCHOR_SCORE_KM_PENALTY", "25") or 25)
                score += max(0.0, anchor_base - anchor_km_penalty * anchor)
            if pickup_limit is not None and pickup_km > pickup_limit:
                score -= 400.0 * (pickup_km - pickup_limit)
            if haul_limit is not None and haul_km > haul_limit:
                score -= 200.0 * (haul_km - haul_limit)
            route_compliance = self._route_compliance_tool.evaluate(
                cargo_name=name,
                start_city=start_city,
                end_city=end_city,
                current_point=(lat, lng),
                start_point=(start_lat, start_lng),
                end_point=(end_lat, end_lng),
                profile=profile,
            )
            tool_report = self._candidate_tool_report(
                driver_id=driver_id,
                current_minute=current_minute,
                ready_minute=ready,
                finish_minute=finish,
                cargo_name=name,
                start_city=start_city,
                end_city=end_city,
                end_lat=end_lat,
                end_lng=end_lng,
                price=price,
                net=net,
                hourly=hourly,
                pickup_km=pickup_km,
                haul_km=haul_km,
                profile=profile,
                cost_per_km=cost_per_km,
                accepted_today=accepted_today,
                monthly_deadhead_used=monthly_deadhead_used,
            )
            tool_report["route_compliance_tool"] = route_compliance
            if route_compliance.get("risk_flags"):
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].extend(route_compliance.get("risk_flags", []))
            pref_risk = tool_report.get("preference_risk_assessment", {}) if isinstance(tool_report.get("preference_risk_assessment"), dict) else {}
            score += self._preference_score_adjustment(pref_risk)
            net_after_return = self._as_float(tool_report.get("net_after_return_yuan"))
            if net_after_return is None:
                net_after_return = net
            action_penalty = self._as_float(pref_risk.get("current_action_penalty_yuan"))
            if action_penalty is None:
                action_penalty = self._as_float(pref_risk.get("expected_penalty_hint_yuan")) or 0.0
            if geo_soft_penalty:
                action_penalty += float(geo_soft_penalty)
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].append("geo_fence_soft_violation")
                tool_report["geo_fence_soft_escape"] = {
                    "estimated_penalty_yuan": round(float(geo_soft_penalty), 2),
                    "return_to_allowed_area_cost_yuan": round(float(geo_return_cost), 2),
                    "reason": "high_profit_out_of_preferred_area",
                }
            if calendar_soft_penalty:
                action_penalty += float(calendar_soft_penalty)
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].append("daily_rest_soft_tradeoff")
                tool_report["calendar_soft_tradeoff"] = {
                    "estimated_penalty_yuan": round(float(calendar_soft_penalty), 2),
                    "reason": "profitable_order_over_flexible_daily_rest_plan",
                }
            candidate_snapshot = {
                "cargo_id": cargo_id,
                "finish_minute": finish,
                "estimated_net": net,
                "net_after_return": float(net_after_return),
                "end": [end_lat, end_lng],
            }
            if home_context:
                candidate_snapshot["home_return_context"] = home_context
            action_guard = self._action_guard_tool.evaluate_candidate(
                status=status,
                profile=profile,
                candidate=candidate_snapshot,
            )
            future_penalty = self._as_float(action_guard.get("estimated_future_preference_penalty_yuan")) or 0.0
            if future_penalty > 0:
                action_penalty += future_penalty
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].append("future_preference_penalty")
                    if action_guard.get("hard_block"):
                        tool_report["risk_flags"].append("future_preference_hard_block")
            tool_report["action_preference_guard"] = action_guard
            support_candidate = {
                "cargo_id": cargo_id,
                "cargo_name": name[:36],
                "start_city": start_city[:42],
                "end_city": end_city[:42],
                "pickup_km": pickup_km,
                "haul_km": haul_km,
                "price": price,
                "estimated_net": net,
                "net_after_return": float(net_after_return),
                "current_action_penalty": float(action_penalty),
                "ready_minute": ready,
                "finish_minute": finish,
                "load_end_minute": load_end,
                "remove_minute": remove_minute,
                "start": [start_lat, start_lng],
                "end": [end_lat, end_lng],
            }
            decision_support = self._decision_support_tools.evaluate_candidate(
                status=status,
                profile=profile,
                candidate=support_candidate,
                history_summary=history_summary,
            )
            region_report = self._region_preference_tool.evaluate(
                current_point=(lat, lng),
                candidate=support_candidate,
                profile=profile,
                time_task_report=time_task_report,
            )
            candidate_action_for_optimizer = action("take_order", {"cargo_id": cargo_id})
            task_penalty_report = self._task_penalty_optimizer_tool.evaluate_action(
                current_minute=current_minute,
                current_point=(lat, lng),
                action=candidate_action_for_optimizer,
                profile=profile,
                time_task_report=time_task_report,
                candidate=support_candidate,
            )
            extra_cost = self._as_float(decision_support.get("additional_future_cost_yuan")) or 0.0
            task_penalty_cost = self._as_float(task_penalty_report.get("estimated_action_task_penalty_yuan")) or 0.0
            if extra_cost > 0:
                action_penalty += extra_cost
            if task_penalty_cost > 0:
                action_penalty += task_penalty_cost
            tool_report["decision_support_tools"] = decision_support
            tool_report["region_preference_tool"] = region_report
            tool_report["task_penalty_optimizer_tool"] = task_penalty_report
            if region_report.get("risk_flags"):
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].extend(region_report.get("risk_flags", []))
            if task_penalty_report.get("risk_flags"):
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].extend(task_penalty_report.get("risk_flags", []))
            cargo_time = decision_support.get("cargo_expiry_recheck_tool", {}) if isinstance(decision_support.get("cargo_expiry_recheck_tool"), dict) else {}
            time_margin = self._as_float(cargo_time.get("load_window_margin_minutes"))
            time_reliability = self._as_float(cargo_time.get("time_reliability_score"))
            if decision_support.get("risk_flags"):
                tool_report.setdefault("risk_flags", [])
                if isinstance(tool_report["risk_flags"], list):
                    tool_report["risk_flags"].extend(decision_support.get("risk_flags", []))
            # Immediate trip profit is primary, but soft geo escapes must also
            # pay for returning to the allowed working area.
            profit_basis = min(float(net_after_return), float(net) - float(geo_return_cost)) if geo_return_cost else float(net)
            action_value = profit_basis - float(action_penalty)
            bounded_local_order = bool(isinstance(profile.get("geo_fence_bounds"), dict) and geo_allowed)
            bounded_local_score = action_value + (min(500.0, max(0.0, hourly) * 12.0) if bounded_local_order else 0.0)
            item = {
                "cargo_id": cargo_id,
                "cargo_name": name[:36],
                "start_city": start_city[:42], "end_city": end_city[:42],
                "pickup_km": round(pickup_km, 2), "haul_km": round(haul_km, 2),
                "price": round(price, 2), "estimated_net": round(net, 2), "hourly": round(hourly, 2),
                "immediate_trip_net": round(float(net), 2),
                "net_after_return": round(float(profit_basis), 2),
                "current_action_penalty": round(float(action_penalty), 2),
                "action_value": round(action_value, 2),
                "bounded_local_order": bounded_local_order,
                "bounded_local_score": round(float(bounded_local_score), 2),
                "time_margin_minutes": None if time_margin is None else round(float(time_margin), 2),
                "time_reliability_score": round(float(time_reliability if time_reliability is not None else 0.5), 4),
                "ready_minute": ready, "finish_minute": finish, "finish_time": wall_time(finish),
                "score": round(score, 2), "start": [start_lat, start_lng], "end": [end_lat, end_lng],
                "home_return_compatible": bool(home_context.get("can_return_home_by_deadline")) if home_context else False,
                "home_return_context": home_context,
                "tool_report": tool_report,
            }
            candidates.append(item)
            self._cargo_memory[cargo_id] = item
        views = self._candidate_views(candidates)
        self._debug_candidate_build(driver_id, current_minute, items, candidates, views, reject_counts)
        return views

    def _raw_temporary_event_blocks_candidate(
        self,
        current_minute: int,
        finish_minute: int,
        end_lat: float,
        end_lng: float,
        preferences: list[dict[str, Any]],
    ) -> bool:
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if not self._looks_like_personal_commitment(text):
                continue
            coords = extract_coordinates(text)
            times = self._extract_text_event_minutes(text)
            if len(coords) < 2 or not times:
                continue
            event_start = min(times)
            release = max(times)
            if current_minute >= release:
                continue
            notice_minutes = self._commitment_notice_minutes()
            if current_minute < event_start - notice_minutes:
                continue
            pickup = (float(coords[0][0]), float(coords[0][1]))
            home = (float(coords[1][0]), float(coords[1][1]))
            to_pickup = distance_to_minutes(haversine_km(end_lat, end_lng, pickup[0], pickup[1]))
            pickup_to_home = distance_to_minutes(haversine_km(pickup[0], pickup[1], home[0], home[1]))
            buffer_minutes = int(os.environ.get("AGENT_TEMPORARY_EVENT_RAW_BUFFER_MINUTES", "60") or 60)
            pickup_wait = 10 if "停留不少于10分钟" in text or "10分钟" in text else 0
            home_deadline = times[1] if len(times) >= 2 else event_start + 12 * 60
            stay_from_start = any(word in text for word in ("到家后", "原处静止", "不在家", "每迟到", "每晚到", "必须待到", "至少待到"))
            protected_arrival = event_start if stay_from_start else home_deadline
            full_return_minutes = to_pickup + pickup_wait + pickup_to_home + buffer_minutes
            if current_minute < event_start and finish_minute + to_pickup + buffer_minutes > event_start:
                return True
            if finish_minute < event_start and finish_minute + to_pickup + buffer_minutes > event_start:
                return True
            if stay_from_start and finish_minute + full_return_minutes > protected_arrival:
                return True
            if event_start <= finish_minute < release and finish_minute + to_pickup + pickup_wait + pickup_to_home > home_deadline:
                return True
            if current_minute >= event_start and finish_minute < release:
                return True
        return False

    def _schedule_sensitive_long_order_block(self, current_minute: int, finish_minute: int, profile: dict[str, Any]) -> bool:
        if os.environ.get("AGENT_BLOCK_SCHEDULE_SENSITIVE_OVERNIGHT_LONG_ORDER", "1").strip().lower() not in {"1", "true", "yes"}:
            return False
        has_schedule = bool(
            profile.get("visit_frequency")
            or profile.get("scheduled_visits")
            or profile.get("temporary_events")
            or profile.get("long_sequence_commitments")
            or profile.get("cumulative_time_penalty_rules")
        )
        if not has_schedule:
            return False
        duration = finish_minute - current_minute
        min_duration = int(os.environ.get("AGENT_SCHEDULE_LONG_ORDER_MIN_MINUTES", "720") or 720)
        if duration < min_duration:
            return False
        if finish_minute // 1440 <= current_minute // 1440:
            return False
        finish_mod = finish_minute % 1440
        latest_finish_mod = int(os.environ.get("AGENT_SCHEDULE_LONG_ORDER_LATEST_FINISH_MOD", str(12 * 60)) or 12 * 60)
        return finish_mod >= latest_finish_mod

    def _filter_raw_temporary_safe_candidates(
        self,
        current_minute: int,
        candidates: list[dict[str, Any]],
        preferences: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates or not preferences:
            return candidates
        safe: list[dict[str, Any]] = []
        for item in candidates:
            finish = self._as_int(item.get("finish_minute"))
            end = item.get("end")
            if finish is None or not self._valid_point(end):
                safe.append(item)
                continue
            assert isinstance(end, (list, tuple))
            if self._raw_temporary_event_blocks_candidate(current_minute, finish, float(end[0]), float(end[1]), preferences):
                continue
            safe.append(item)
        return safe

    @staticmethod
    def _extract_text_event_minutes(text: str) -> list[int]:
        """Parse common absolute time expressions in preference task text."""
        values: list[int] = []

        def append_time(year: int, month: int, day: int, hour: int, minute: int, meridiem: str = "") -> None:
            if any(word in meridiem for word in ("下午", "晚上", "晚间", "夜里")) and hour < 12:
                hour += 12
            if "中午" in meridiem and hour < 11:
                hour += 12
            try:
                current = datetime(year, month, day, hour, minute)
            except ValueError:
                return
            offset = int((current - BASE_TIME).total_seconds() // 60)
            if 0 <= offset <= MONTH_HORIZON_MINUTES:
                values.append(offset)

        for match in re.finditer(r"(?:(\d{4})-)?(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})(?::\d{2})?", text):
            append_time(int(match.group(1) or BASE_TIME.year), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5)))

        chinese_pattern = (
            r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月([0-9一二两三四五六七八九十]{1,3})(?:日|号)\s*"
            r"(上午|早上|中午|下午|晚上|晚间|夜里|凌晨)?\s*"
            r"([0-9一二两三四五六七八九十]{1,3})(?::|：|点|时)(\d{1,2})?"
        )
        for match in re.finditer(chinese_pattern, text):
            year = int(match.group(1) or BASE_TIME.year)
            month = _parse_cn_number(match.group(2))
            day = _parse_cn_number(match.group(3))
            hour = _parse_cn_number(match.group(5))
            minute = int(match.group(6) or 0)
            if month is not None and day is not None and hour is not None:
                append_time(year, month, day, hour, minute, match.group(4) or "")

        return sorted(set(values))

    def _raw_temporary_event_action(
        self,
        current_minute: int,
        lat: float,
        lng: float,
        preferences: list[dict[str, Any]],
        force_notice: bool = False,
        driver_id: str | None = None,
    ) -> dict[str, Any] | None:
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if not self._looks_like_personal_commitment(text):
                continue
            coords = extract_coordinates(text)
            times = self._extract_text_event_minutes(text)
            if len(coords) < 2 or not times:
                continue
            event_start = min(times)
            release = max(times)
            if current_minute >= release:
                continue
            pickup = coords[0]
            home = coords[1]
            travel_to_pickup = distance_to_minutes(haversine_km(lat, lng, pickup[0], pickup[1]))
            pickup_to_home = distance_to_minutes(haversine_km(pickup[0], pickup[1], home[0], home[1]))
            buffer_minutes = int(os.environ.get("AGENT_TEMPORARY_EVENT_RAW_BUFFER_MINUTES", "60") or 60)
            pickup_wait = 10 if "停留不少于10分钟" in text or "10分钟" in text else 0
            pickup_required = pickup_wait > 0 or any(word in text for word in ("接上", "接到", "接人", "配偶", "孩子", "老人"))
            leave_by = event_start - travel_to_pickup - pickup_to_home - pickup_wait - buffer_minutes
            notice_start = event_start - self._commitment_notice_minutes()
            near_pickup = haversine_km(lat, lng, pickup[0], pickup[1]) <= 1.0
            near_home = haversine_km(lat, lng, home[0], home[1]) <= 1.0
            picked = self._raw_temporary_pickup_completed(driver_id, pickup, event_start, max(1, pickup_wait)) if pickup_required else True
            if current_minute < leave_by:
                if current_minute < notice_start:
                    continue
                if not near_pickup:
                    planned = action("reposition", {"latitude": float(pickup[0]), "longitude": float(pickup[1])})
                    planned["reason_code"] = "raw_temporary_event_notice_position"
                    return planned
                planned = action("wait", {"duration_minutes": max(1, min(240, event_start - current_minute))})
                planned["reason_code"] = "raw_temporary_event_notice_wait"
                return planned
            if current_minute < event_start:
                if not near_pickup:
                    planned = action("reposition", {"latitude": float(pickup[0]), "longitude": float(pickup[1])})
                    planned["reason_code"] = "raw_temporary_event_go_pickup"
                    return planned
                planned = action("wait", {"duration_minutes": max(1, min(240, event_start - current_minute))})
                planned["reason_code"] = "raw_temporary_event_wait_pickup"
                return planned
            if pickup_required and not picked:
                if not near_pickup:
                    planned = action("reposition", {"latitude": float(pickup[0]), "longitude": float(pickup[1])})
                    planned["reason_code"] = "raw_temporary_event_go_pickup_first"
                    return planned
                wait_until = max(current_minute + max(1, pickup_wait), event_start + pickup_wait)
                planned = action("wait", {"duration_minutes": max(1, min(60, wait_until - current_minute))})
                planned["reason_code"] = "raw_temporary_event_pickup_hold"
                return planned
            if not near_home:
                planned = action("reposition", {"latitude": float(home[0]), "longitude": float(home[1])})
                planned["reason_code"] = "raw_temporary_event_return_home"
                return planned
            planned = action("wait", {"duration_minutes": max(1, release - current_minute)})
            planned["reason_code"] = "raw_temporary_event_stay_home"
            return planned
        return None

    def _raw_temporary_pickup_completed(
        self,
        driver_id: str | None,
        pickup: tuple[float, float],
        earliest_minute: int,
        wait_minutes: int,
    ) -> bool:
        if not driver_id:
            return False
        run = 0
        for record in self._recent_history_records(driver_id):
            if not isinstance(record, dict):
                continue
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "wait":
                run = 0
                continue
            end_minute = self._as_int(record.get("simulation_end_minute"))
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            if end_minute is None:
                end_minute = self._as_int(result.get("simulation_progress_minutes"))
            elapsed = self._as_int(record.get("action_exec_cost_minutes")) or self._as_int(record.get("step_elapsed_minutes")) or 0
            if end_minute is None or elapsed <= 0:
                continue
            start_minute = end_minute - elapsed
            pos_before = record.get("position_before") if isinstance(record.get("position_before"), dict) else {}
            pos_after = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            try:
                near = (
                    haversine_km(float(pos_before.get("lat")), float(pos_before.get("lng")), pickup[0], pickup[1]) <= 1.0
                    and haversine_km(float(pos_after.get("lat")), float(pos_after.get("lng")), pickup[0], pickup[1]) <= 1.0
                )
            except (TypeError, ValueError):
                near = False
            if not near:
                run = 0
                continue
            run += max(0, end_minute - max(start_minute, earliest_minute))
            if run >= wait_minutes:
                return True
        return False

    def _profile_commitment_blocks_candidate(
        self,
        driver_id: str,
        current_minute: int,
        finish_minute: int,
        end_lat: float,
        end_lng: float,
        profile: dict[str, Any],
    ) -> bool:
        for event in profile.get("temporary_events", []) or []:
            if not isinstance(event, dict):
                continue
            pickup = event.get("pickup_point")
            home = event.get("home_point")
            pickup_minute = self._as_int(event.get("pickup_minute"))
            release_minute = self._as_int(event.get("release_minute"))
            if pickup_minute is None or release_minute is None or not self._valid_point(pickup) or not self._valid_point(home):
                continue
            if current_minute >= release_minute:
                continue
            pickup_tuple = (float(pickup[0]), float(pickup[1]))
            pickup_wait = max(10, self._as_int(event.get("pickup_wait_minutes")) or 10)
            pickup_required = bool(event.get("pickup_required", True))
            picked = self._raw_temporary_pickup_completed(driver_id, pickup_tuple, pickup_minute, pickup_wait) if pickup_required else True
            to_pickup = distance_to_minutes(haversine_km(end_lat, end_lng, float(pickup[0]), float(pickup[1])))
            to_home = distance_to_minutes(haversine_km(float(pickup[0]), float(pickup[1]), float(home[0]), float(home[1])))
            must_be_home_by = self._temporary_event_home_deadline(event, pickup_minute)
            pickup_buffer = int(os.environ.get("AGENT_TEMPORARY_EVENT_PICKUP_BUFFER_MINUTES", "15") or 15)
            if not picked and finish_minute + to_pickup + pickup_buffer > pickup_minute:
                return True
            if not picked and finish_minute + to_pickup + pickup_wait + to_home > must_be_home_by:
                return True
            if picked and finish_minute + to_home > must_be_home_by:
                return True
        for sequence in profile.get("long_sequence_commitments", []) or []:
            if not isinstance(sequence, dict):
                continue
            for step in sequence.get("steps", []) or []:
                if not isinstance(step, dict) or not self._valid_point(step.get("point")):
                    continue
                point = step.get("point")
                step_type = str(step.get("step_type") or "").lower()
                earliest = self._as_int(step.get("earliest_minute"))
                deadline = self._as_int(step.get("deadline_minute"))
                hold_until = self._as_int(step.get("hold_until_minute"))
                if hold_until is not None and current_minute >= hold_until:
                    continue
                if step_type == "visit_and_wait" and self._long_sequence_step_completed(driver_id, step):
                    continue
                if step_type == "visit_and_wait" and earliest is not None:
                    target = earliest
                else:
                    target = deadline if deadline is not None else earliest
                if target is None:
                    target = hold_until
                if target is None:
                    continue
                travel = distance_to_minutes(haversine_km(end_lat, end_lng, float(point[0]), float(point[1])))
                wait_minutes = max(0, self._as_int(step.get("wait_minutes")) or 0)
                if finish_minute + travel + wait_minutes > target:
                    return True
                break
        return False

    def _debug_candidate_build(
        self,
        driver_id: str,
        current_minute: int,
        items: list[Any],
        candidates: list[dict[str, Any]],
        views: list[dict[str, Any]],
        reject_counts: dict[str, int],
    ) -> None:
        if os.environ.get("AGENT_DEBUG_CANDIDATES", "0").strip().lower() not in {"1", "true", "yes"}:
            return
        path = os.environ.get("AGENT_DEBUG_CANDIDATE_PATH") or os.path.join(tempfile.gettempdir(), "agent_candidate_debug.jsonl")
        top: list[dict[str, Any]] = []
        for item in sorted(candidates, key=self._profit_penalty_sort_key, reverse=True)[:5]:
            report = item.get("tool_report", {}) if isinstance(item.get("tool_report"), dict) else {}
            top.append(
                {
                    "cargo_id": item.get("cargo_id"),
                    "net": item.get("immediate_trip_net", item.get("estimated_net")),
                    "action_value": item.get("action_value"),
                    "penalty": item.get("current_action_penalty"),
                    "finish_time": item.get("finish_time"),
                    "acceptable": self._candidate_is_acceptable(item),
                    "home_return_compatible": item.get("home_return_compatible"),
                    "risk_flags": list(report.get("risk_flags", []) or [])[:8],
                }
            )
        event = {
            "driver_id": driver_id,
            "current_minute": current_minute,
            "wall_time": wall_time(current_minute),
            "base_time": BASE_TIME.strftime("%Y-%m-%d %H:%M:%S"),
            "items_count": len(items),
            "candidate_count": len(candidates),
            "view_count": len(views),
            "acceptable_count": sum(1 for item in candidates if self._candidate_is_acceptable(item)),
            "home_compatible_count": sum(1 for item in candidates if item.get("home_return_compatible")),
            "reject_counts": dict(sorted(reject_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "top_candidates": top,
        }
        try:
            with open(path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        except OSError:
            return

    def _temporary_event_home_deadline(self, event: dict[str, Any], pickup_minute: int) -> int:
        deadline = self._as_int(event.get("home_deadline_minute"))
        if deadline is not None:
            return deadline
        release = self._as_int(event.get("release_minute"))
        if release is not None:
            return min(release, pickup_minute + 12 * 60)
        return pickup_minute + 12 * 60

    def _long_sequence_step_completed(self, driver_id: str, step: dict[str, Any]) -> bool:
        point = step.get("point")
        if not self._valid_point(point):
            return False
        earliest = self._as_int(step.get("earliest_minute")) or 0
        wait_minutes = max(1, self._as_int(step.get("wait_minutes")) or 1)
        return self._raw_temporary_pickup_completed(driver_id, (float(point[0]), float(point[1])), earliest, wait_minutes)

    @staticmethod
    def _commitment_notice_minutes() -> int:
        try:
            value = int(os.environ.get("AGENT_COMMITMENT_NOTICE_MINUTES", "1440") or 1440)
        except ValueError:
            value = 1440
        return max(60, min(7 * 1440, value))

    def _candidate_views(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        candidates = self._preference_first_candidate_pool(candidates)
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        limit = int(self._config["runtime_limits"].get("candidate_pool_limit", 32))
        pareto_input_limit = max(limit * 4, self._int_env("AGENT_PARETO_INPUT_LIMIT", 96, 24, 512))
        pareto_input = sorted(candidates, key=self._profit_penalty_sort_key, reverse=True)[:pareto_input_limit]
        pareto = self._pareto_profit_penalty_candidates(pareto_input)
        views = [
            lambda x: self._profit_penalty_sort_key(x),
            lambda x: (float(x.get("net_after_return", x.get("estimated_net", 0.0))), -float(x.get("current_action_penalty", 0.0)), float(x.get("time_reliability_score", 0.5) or 0.5)),
            lambda x: (float(x.get("time_reliability_score", 0.5) or 0.5), float(x.get("action_value", -10**9))),
            lambda x: (-float(x.get("current_action_penalty", 0.0)), float(x.get("net_after_return", x.get("estimated_net", 0.0))), float(x.get("time_reliability_score", 0.5) or 0.5)),
            lambda x: (float(x.get("action_value", -10**9)), float(x.get("net_after_return", 0.0))),
            lambda x: (float(x.get("hourly", 0.0)), -float(x.get("pickup_km", 0.0))),
        ]
        for item in sorted(pareto, key=self._profit_penalty_sort_key, reverse=True):
            if item["cargo_id"] not in seen:
                selected.append(item); seen.add(item["cargo_id"])
            if len(selected) >= limit:
                return selected
        for key in views:
            for item in sorted(candidates, key=key, reverse=True)[:12]:
                if item["cargo_id"] not in seen:
                    selected.append(item); seen.add(item["cargo_id"])
                if len(selected) >= limit:
                    return selected
        return selected

    @staticmethod
    def _candidate_net_after_return(candidate: dict[str, Any]) -> float:
        value = candidate.get("net_after_return")
        if value is None:
            value = candidate.get("estimated_net")
        if value is None:
            value = candidate.get("immediate_trip_net")
        return float(value or 0.0)

    @staticmethod
    def _profit_penalty_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        net_after_return = ModelDecisionService._candidate_net_after_return(candidate)
        penalty = float(candidate.get("current_action_penalty", 0.0) or 0.0)
        action_value = float(candidate.get("action_value", net_after_return - penalty) or 0.0)
        time_reliability = float(candidate.get("time_reliability_score", 0.5) or 0.5)
        home_bonus = 1.0 if candidate.get("home_return_compatible") else 0.0
        bounded_score = float(candidate.get("bounded_local_score", action_value) or action_value) if candidate.get("bounded_local_order") else action_value
        return (bounded_score, net_after_return, action_value, home_bonus, -penalty, time_reliability, float(candidate.get("hourly", 0.0) or 0.0))

    @staticmethod
    def _pareto_profit_penalty_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pareto: list[dict[str, Any]] = []
        for item in candidates:
            net = ModelDecisionService._candidate_net_after_return(item)
            penalty = float(item.get("current_action_penalty", 0.0) or 0.0)
            time_score = float(item.get("time_reliability_score", 0.5) or 0.5)
            dominated = False
            for other in candidates:
                if other is item:
                    continue
                other_net = ModelDecisionService._candidate_net_after_return(other)
                other_penalty = float(other.get("current_action_penalty", 0.0) or 0.0)
                other_time = float(other.get("time_reliability_score", 0.5) or 0.5)
                if other_net >= net and other_penalty <= penalty and other_time >= time_score and (other_net > net or other_penalty < penalty or other_time > time_score):
                    dominated = True
                    break
            if not dominated:
                pareto.append(item)
        return pareto or candidates

    def _candidate_tool_report(
        self,
        *,
        driver_id: str,
        current_minute: int,
        ready_minute: int,
        finish_minute: int,
        cargo_name: str,
        start_city: str,
        end_city: str,
        end_lat: float,
        end_lng: float,
        price: float,
        net: float,
        hourly: float,
        pickup_km: float,
        haul_km: float,
        profile: dict[str, Any],
        cost_per_km: float = 1.5,
        accepted_today: int = 0,
        monthly_deadhead_used: float = 0.0,
    ) -> dict[str, Any]:
        risks: list[str] = []
        adjustment = 0.0
        next_visit = self._next_scheduled_visit_after(current_minute, profile)
        if next_visit is not None and self._valid_point(next_visit.get("point")):
            point = next_visit["point"]
            deadline = int(next_visit["day"]) * 1440 + int(next_visit.get("arrive_before_minute") or 20 * 60)
            return_minutes = distance_to_minutes(haversine_km(end_lat, end_lng, float(point[0]), float(point[1])))
            margin = deadline - finish_minute - return_minutes
            if margin < 0:
                risks.append("miss_scheduled_visit")
            elif margin < float(os.environ.get("AGENT_TIGHT_VISIT_MARGIN_MINUTES", "180") or 180):
                risks.append("tight_scheduled_visit")
        else:
            margin = None
            return_minutes = None

        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        rest_hours = self._as_float(rest.get("hours")) or 0.0
        minute_of_day = finish_minute % 1440
        if rest_hours >= 6 and minute_of_day > 16 * 60:
            risks.append("late_finish_before_rest")

        anchor_km = self._nearest_profile_point_km(end_lat, end_lng, profile)
        good_anchor_km = float(os.environ.get("AGENT_GOOD_ANCHOR_KM", "20") or 20)
        if anchor_km is not None and anchor_km <= good_anchor_km:
            anchor_bonus_base = float(os.environ.get("AGENT_ANCHOR_BONUS_BASE", "500") or 500)
            anchor_bonus_km_penalty = float(os.environ.get("AGENT_ANCHOR_BONUS_KM_PENALTY", "20") or 20)
            adjustment += max(0.0, anchor_bonus_base - anchor_bonus_km_penalty * anchor_km)
        if net < 0:
            risks.append("negative_net")
            adjustment -= 2000.0
        if pickup_km > float(os.environ.get("AGENT_LARGE_DEADHEAD_KM", "100") or 100):
            risks.append("large_deadhead")

        evaluation = self._cargo_evaluation_tool.evaluate_candidate(
            current_minute=current_minute,
            finish_minute=finish_minute,
            end_lat=end_lat,
            end_lng=end_lng,
            net=net,
            hourly=hourly,
            pickup_km=pickup_km,
            cost_per_km=cost_per_km,
            profile=profile,
            next_visit=next_visit,
        )
        merged_risks = list(dict.fromkeys(risks + list(evaluation.get("risk_flags", []) or [])))
        relay = self._cargo_evaluation_tool.evaluate_relay_option(
            destination=[end_lat, end_lng],
            anchors=[p for p in profile.get("preference_points", []) if self._valid_point(p)],
            net_after_return=float(evaluation.get("net_after_return_yuan", net) or net),
        )
        preference_risk = self._preference_risk_assessment(
            driver_id=driver_id,
            profile=profile,
            cargo_name=cargo_name,
            start_city=start_city,
            end_city=end_city,
            current_minute=current_minute,
            ready_minute=ready_minute,
            finish_minute=finish_minute,
            pickup_km=pickup_km,
            haul_km=haul_km,
            accepted_today=accepted_today,
            monthly_deadhead_used=monthly_deadhead_used,
            price=price,
            net=net,
        )
        return {
            "net_yuan": round(net, 2),
            "hourly_yuan": round(hourly, 2),
            "preference_anchor_km": None if anchor_km is None else round(anchor_km, 2),
            "empty_return_cost_yuan": evaluation.get("empty_return_cost_yuan"),
            "empty_return_minutes": evaluation.get("empty_return_minutes"),
            "net_after_return_yuan": evaluation.get("net_after_return_yuan"),
            "future_relay_hint": evaluation.get("future_relay_hint"),
            "relay_report": relay,
            "preference_risk_assessment": preference_risk,
            "next_visit_return_minutes": return_minutes,
            "next_visit_margin_minutes": margin,
            "risk_flags": merged_risks,
            "planner_adjustment": round(adjustment, 2),
            "tool_name": "cargo_evaluation_tool",
        }

    def _preference_risk_assessment(
        self,
        *,
        driver_id: str,
        profile: dict[str, Any],
        cargo_name: str,
        start_city: str,
        end_city: str,
        current_minute: int,
        ready_minute: int,
        finish_minute: int,
        pickup_km: float,
        haul_km: float,
        accepted_today: int,
        monthly_deadhead_used: float,
        price: float,
        net: float,
    ) -> dict[str, Any]:
        cards = [card for card in profile.get("preference_cards", []) if isinstance(card, dict)]
        triggered: list[dict[str, Any]] = []
        unknown = [item for item in profile.get("unknown_preferences", []) if isinstance(item, dict)]
        unknown_tags = [item for item in profile.get("unknown_attribute_tags", []) if isinstance(item, dict)]
        unknown_groups = [item for item in profile.get("unknown_preference_groups", []) if isinstance(item, dict)]
        duplicate_groups = [item for item in profile.get("duplicate_preference_groups", []) if isinstance(item, dict)]

        avoid_keywords = [str(k) for k in profile.get("avoid_cargo_keywords", []) if str(k)]
        for keyword in avoid_keywords:
            if keyword in cargo_name:
                triggered.append({"type": "avoid_cargo", "risk_key": f"avoid_cargo:{keyword}", "severity": "high", "reason": f"cargo_name contains {keyword}"})

        pickup_limit = self._as_float(profile.get("pickup_deadhead_max_km"))
        if pickup_limit is not None and pickup_km > pickup_limit:
            triggered.append({"type": "pickup_deadhead_limit", "risk_key": "pickup_deadhead_limit", "severity": "medium", "over_km": round(pickup_km - pickup_limit, 2)})

        monthly_limit = self._as_float(profile.get("monthly_deadhead_limit_km"))
        if monthly_limit is not None and monthly_deadhead_used + pickup_km > monthly_limit:
            triggered.append({"type": "monthly_deadhead_limit", "risk_key": "monthly_deadhead_limit", "severity": "medium", "over_km": round(monthly_deadhead_used + pickup_km - monthly_limit, 2)})

        haul_limit = self._as_float(profile.get("max_haul_km"))
        if haul_limit is not None and haul_km > haul_limit:
            triggered.append({"type": "haul_distance_limit", "risk_key": "distance_limit", "severity": "medium", "over_km": round(haul_km - haul_limit, 2)})

        daily_limit = self._as_int(profile.get("daily_order_limit"))
        if daily_limit is not None and accepted_today + 1 > daily_limit:
            triggered.append({"type": "daily_order_limit", "risk_key": "daily_order_limit", "severity": "critical", "reason": "accepted order count would exceed limit"})

        first_deadline = self._as_int(profile.get("first_order_deadline_minute"))
        if first_deadline is not None and accepted_today == 0 and ready_minute // 1440 == current_minute // 1440 and ready_minute % 1440 >= first_deadline:
            triggered.append({"type": "first_order_deadline", "risk_key": "first_order_deadline", "severity": "medium", "late_minutes": ready_minute % 1440 - first_deadline})

        rest_penalty = self._soft_daily_rest_penalty(current_minute, finish_minute, profile)
        if rest_penalty > 0:
            triggered.append({"type": "rest_or_no_action", "risk_key": "rest_or_no_action", "severity": "medium", "penalty_yuan": round(rest_penalty, 2)})

        for item in profile.get("avoid_regions", []):
            if not isinstance(item, dict):
                continue
            region = str(item.get("region") or "")
            days = item.get("days")
            active_day = ready_minute // 1440
            if region and (region in start_city or region in end_city) and (not isinstance(days, list) or active_day in days):
                triggered.append({"type": "avoid_region", "risk_key": f"avoid_region:{region}", "severity": "high", "region": region})

        dynamic_hits = self._evaluate_dynamic_preference_rules(
            driver_id=driver_id,
            profile=profile,
            cargo_name=cargo_name,
            start_city=start_city,
            end_city=end_city,
            current_minute=current_minute,
            ready_minute=ready_minute,
            finish_minute=finish_minute,
            pickup_km=pickup_km,
            haul_km=haul_km,
            price=price,
        )
        for hit in dynamic_hits:
            triggered.append(
                {
                    "type": "dynamic_preference_rule",
                    "risk_key": f"dynamic_rule:{hit.get('id')}",
                    "severity": hit.get("severity", "medium"),
                    "label": hit.get("label"),
                    "effect": hit.get("effect"),
                }
            )

        hard_ids = set(profile.get("risk_policy", {}).get("hard_constraint_ids", []) if isinstance(profile.get("risk_policy"), dict) else [])
        triggered_keys = {str(item.get("risk_key")) for item in triggered if item.get("risk_key")}
        triggered_types = {item["type"] for item in triggered}
        exact_cards = [
            card
            for card in cards
            if str(card.get("risk_key") or "") in triggered_keys
        ]
        fallback_cards = [
            card
            for card in cards
            if card not in exact_cards and set(card.get("types", []) or []) & triggered_types
        ]
        all_related_cards = exact_cards + fallback_cards
        related_cards = [
            {"id": card.get("id"), "types": card.get("types"), "severity": card.get("severity"), "tradeoff_mode": card.get("tradeoff_mode")}
            for card in all_related_cards
        ][:8]
        matching_duplicate_groups = [
            group for group in duplicate_groups
            if str(group.get("risk_key") or "") in triggered_keys or bool(set(group.get("types", []) or []) & triggered_types)
        ][:5]
        hard_hit = any(str(card.get("id")) in hard_ids for card in all_related_cards)
        unknown_review = bool(unknown)
        expected_penalty_hint = sum(float(card.get("penalty_amount") or 0.0) for card in all_related_cards)
        dynamic_penalty = sum(float(hit.get("per_action_penalty_yuan") or hit.get("effective_penalty_yuan") or hit.get("penalty_yuan") or 0.0) for hit in dynamic_hits if hit.get("effect") != "boost")
        unresolved_unknown_penalty = self._unresolved_unknown_penalty(unknown, unknown_groups)
        if matching_duplicate_groups:
            expected_penalty_hint = max(
                expected_penalty_hint,
                sum(float(group.get("stacked_penalty_amount") or 0.0) for group in matching_duplicate_groups),
            )
        expected_penalty_hint += rest_penalty
        expected_penalty_hint += dynamic_penalty
        hard_types = {"temporary_event", "required_cargo", "geo_fence", "forbidden_circle"}
        hard_hit = hard_hit and bool(triggered_types & hard_types)
        hard_hit = hard_hit or any(hit.get("effect") == "hard_reject" and str(hit.get("severity")) == "critical" for hit in dynamic_hits)
        return {
            "checked_preference_count": len(cards),
            "triggered_risks": triggered,
            "related_preference_cards": related_cards,
            "duplicate_preference_groups_hit": matching_duplicate_groups,
            "duplicate_stack_detected": bool(matching_duplicate_groups),
            "dynamic_rule_hits": dynamic_hits[:8],
            "dynamic_rule_count": len(profile.get("dynamic_preference_rules", []) if isinstance(profile.get("dynamic_preference_rules"), list) else []),
            "unknown_review_needed": unknown_review,
            "unknown_preference_count": len(unknown),
            "unknown_attribute_tag_count": len(unknown_tags),
            "unknown_attribute_tags": unknown_tags[:8],
            "unknown_preference_groups": unknown_groups[:5],
            "unresolved_unknown_penalty_hint_yuan": round(unresolved_unknown_penalty, 2),
            "unknown_preview": [{"id": item.get("id"), "reason": item.get("reason")} for item in unknown[:3]],
            "hard_constraint_maybe_violated": hard_hit,
            "expected_penalty_hint_yuan": round(expected_penalty_hint, 2),
            "current_action_penalty_yuan": round(expected_penalty_hint, 2),
            "tradeoff_hint": "hard_commitment_or_safety_block" if hard_hit else ("money_tradeoff_pay_penalty_if_profit_covers_it" if triggered or unknown_review else "low_detected_preference_risk"),
            "net_yuan": round(net, 2),
            "time_window": {"start_minute": current_minute, "ready_minute": ready_minute, "finish_minute": finish_minute},
        }

    def _soft_daily_rest_penalty(self, start_minute: int, finish_minute: int, profile: dict[str, Any]) -> float:
        if not self._interval_overlaps_no_action_window(start_minute, finish_minute, profile):
            return 0.0
        base = 0.0
        for card in profile.get("preference_cards", []) if isinstance(profile.get("preference_cards"), list) else []:
            if not isinstance(card, dict):
                continue
            if "rest_or_no_action" in set(card.get("types") or []):
                base = max(base, float(card.get("penalty_amount") or 0.0))
        if base <= 0.0:
            base = float(os.environ.get("AGENT_REST_WINDOW_DEFAULT_PENALTY_YUAN", "300") or 300)
        return base

    def _unresolved_unknown_penalty(self, unknown: list[dict[str, Any]], unknown_groups: list[dict[str, Any]]) -> float:
        if not unknown:
            return 0.0
        multiplier = float(os.environ.get("AGENT_UNKNOWN_PREF_PENALTY_MULTIPLIER", "2.0") or 2.0)
        multiplier = max(0.0, min(20.0, multiplier))
        group_penalty = sum(float(group.get("stacked_penalty_amount") or 0.0) for group in unknown_groups)
        if group_penalty <= 0:
            group_penalty = 500.0 * len(unknown)
        repeat_boost = max(1.0, max((float(group.get("count") or 1.0) for group in unknown_groups), default=1.0))
        return min(150000.0, group_penalty * multiplier * repeat_boost)

    def _evaluate_dynamic_preference_rules(
        self,
        *,
        driver_id: str,
        profile: dict[str, Any],
        cargo_name: str,
        start_city: str,
        end_city: str,
        current_minute: int,
        ready_minute: int,
        finish_minute: int,
        pickup_km: float,
        haul_km: float,
        price: float,
    ) -> list[dict[str, Any]]:
        rules = [rule for rule in profile.get("dynamic_preference_rules", []) if isinstance(rule, dict)]
        hits: list[dict[str, Any]] = []
        for rule in rules:
            match = rule.get("match", {}) if isinstance(rule.get("match"), dict) else {}
            if not self._dynamic_rule_matches(
                driver_id=driver_id,
                match=match,
                cargo_name=cargo_name,
                start_city=start_city,
                end_city=end_city,
                current_minute=current_minute,
                ready_minute=ready_minute,
                finish_minute=finish_minute,
                pickup_km=pickup_km,
                haul_km=haul_km,
                price=price,
            ):
                continue
            hits.append(
                {
                    "id": rule.get("id"),
                    "source_preference_id": rule.get("source_preference_id"),
                    "source_unknown_tag_ids": rule.get("source_unknown_tag_ids", []),
                    "label": rule.get("label"),
                    "effect": rule.get("effect"),
                    "penalty_yuan": round(float(rule.get("penalty_yuan") or 0.0), 2),
                    "per_violation_penalty_yuan": round(float(rule.get("per_violation_penalty_yuan") or rule.get("penalty_yuan") or 0.0), 2),
                    "expected_violations_per_month": rule.get("expected_violations_per_month"),
                    "penalty_multiplier": rule.get("penalty_multiplier"),
                    "per_action_penalty_yuan": round(float(rule.get("per_action_penalty_yuan") or rule.get("effective_penalty_yuan") or rule.get("penalty_yuan") or 0.0), 2),
                    "effective_penalty_yuan": round(float(rule.get("effective_penalty_yuan") or rule.get("penalty_yuan") or 0.0), 2),
                    "monthly_risk_hint_yuan": round(float(rule.get("monthly_risk_hint_yuan") or rule.get("effective_penalty_yuan") or 0.0), 2),
                    "repeat_count": rule.get("repeat_count"),
                    "severity": rule.get("severity"),
                    "confidence": rule.get("confidence"),
                }
            )
        return hits

    def _dynamic_rule_matches(
        self,
        *,
        driver_id: str,
        match: dict[str, Any],
        cargo_name: str,
        start_city: str,
        end_city: str,
        current_minute: int,
        ready_minute: int,
        finish_minute: int,
        pickup_km: float,
        haul_km: float,
        price: float,
    ) -> bool:
        checks: list[bool] = []
        for key, text in (
            ("cargo_name_contains", cargo_name),
            ("start_city_contains", start_city),
            ("end_city_contains", end_city),
        ):
            words = [str(item) for item in match.get(key, []) if str(item)]
            if words:
                checks.append(any(word in text for word in words))
        words = [str(item) for item in match.get("start_or_end_city_contains", []) if str(item)]
        if words:
            checks.append(any(word in start_city or word in end_city for word in words))
        max_pickup = self._as_float(match.get("max_pickup_km"))
        if max_pickup is not None:
            checks.append(pickup_km > max_pickup)
        max_haul = self._as_float(match.get("max_haul_km"))
        if max_haul is not None:
            checks.append(haul_km > max_haul)
        min_price = self._as_float(match.get("min_price"))
        if min_price is not None:
            checks.append(price < min_price)
        window = match.get("time_window") if isinstance(match.get("time_window"), dict) else {}
        start = self._as_int(window.get("start_minute"))
        end = self._as_int(window.get("end_minute"))
        if start is not None or end is not None:
            lower = start if start is not None else -10**9
            upper = end if end is not None else 10**9
            checks.append(not (finish_minute < lower or ready_minute > upper or current_minute > upper))
        daily_window = match.get("daily_time_window") if isinstance(match.get("daily_time_window"), dict) else {}
        daily_start = self._as_int(daily_window.get("start_minute_of_day"))
        daily_end = self._as_int(daily_window.get("end_minute_of_day"))
        if daily_start is not None and daily_end is not None:
            checks.append(self._daily_window_overlaps_interval(current_minute, finish_minute, daily_start, daily_end))
        periodic = match.get("periodic_stop_required") if isinstance(match.get("periodic_stop_required"), dict) else {}
        if periodic:
            checks.append(self._periodic_rule_candidate_risky(driver_id, current_minute, finish_minute, periodic))
        return bool(checks) and all(checks)

    @staticmethod
    def _preference_score_adjustment(preference_risk: dict[str, Any]) -> float:
        if not isinstance(preference_risk, dict):
            return 0.0
        penalty = float(preference_risk.get("expected_penalty_hint_yuan") or 0.0)
        adjustment = -min(150000.0, penalty)
        dynamic_hits = preference_risk.get("dynamic_rule_hits", [])
        if isinstance(dynamic_hits, list):
            for hit in dynamic_hits:
                if not isinstance(hit, dict):
                    continue
                if hit.get("effect") == "hard_reject" and str(hit.get("severity")) == "critical":
                    adjustment -= 150000.0
                elif hit.get("effect") == "boost":
                    adjustment += min(10000.0, float(hit.get("per_action_penalty_yuan") or hit.get("effective_penalty_yuan") or hit.get("penalty_yuan") or 0.0))
        if preference_risk.get("hard_constraint_maybe_violated"):
            adjustment -= 150000.0
        return adjustment

    @staticmethod
    def _calendar_block_can_be_softened(calendar_report: dict[str, Any], profile: dict[str, Any]) -> bool:
        if os.environ.get("AGENT_ALLOW_DAILY_REST_SOFT_TRADEOFF", "0").strip().lower() not in {"1", "true", "yes"}:
            return False
        if not isinstance(calendar_report, dict) or not isinstance(profile.get("geo_fence_bounds"), dict):
            return False
        if calendar_report.get("active_tasks"):
            return False
        no_action = calendar_report.get("no_action_window_plan", {}) if isinstance(calendar_report.get("no_action_window_plan"), dict) else {}
        if no_action.get("inside_now"):
            return False
        monthly = calendar_report.get("monthly_plan", {}) if isinstance(calendar_report.get("monthly_plan"), dict) else {}
        if monthly.get("lock_today"):
            return False
        visit = calendar_report.get("monthly_visit_plan", {}) if isinstance(calendar_report.get("monthly_visit_plan"), dict) else {}
        region = calendar_report.get("required_region_cargo_plan", {}) if isinstance(calendar_report.get("required_region_cargo_plan"), dict) else {}
        if visit.get("active_today") or region.get("active_today"):
            return False
        daily = calendar_report.get("daily_rest_plan", {}) if isinstance(calendar_report.get("daily_rest_plan"), dict) else {}
        return bool(daily.get("candidate_finish_deadline"))

    def _calendar_soft_penalty_yuan(self, calendar_report: dict[str, Any], profile: dict[str, Any], immediate_net: float) -> float | None:
        if not self._calendar_block_can_be_softened(calendar_report, profile):
            return None
        min_net = float(os.environ.get("AGENT_DAILY_REST_SOFT_TRADEOFF_MIN_NET_YUAN", "260") or 260)
        if immediate_net < min_net:
            return None
        daily = calendar_report.get("daily_rest_plan", {}) if isinstance(calendar_report.get("daily_rest_plan"), dict) else {}
        remaining = self._as_float(daily.get("remaining_minutes")) or 0.0
        base = float(os.environ.get("AGENT_DAILY_REST_SOFT_TRADEOFF_BASE_PENALTY_YUAN", "120") or 120)
        per_hour = float(os.environ.get("AGENT_DAILY_REST_SOFT_TRADEOFF_PER_HOUR_YUAN", "80") or 80)
        penalty = base + per_hour * min(8.0, max(0.0, remaining) / 60.0)
        return min(penalty, float(os.environ.get("AGENT_DAILY_REST_SOFT_TRADEOFF_MAX_PENALTY_YUAN", "900") or 900))

    def _has_acceptable_candidate(self, candidates: list[dict[str, Any]]) -> bool:
        return any(self._candidate_is_acceptable(item) for item in candidates)

    def _confident_income_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        acceptable = [item for item in candidates if self._candidate_is_acceptable(item)]
        if not acceptable:
            return None
        acceptable = self._preference_first_candidate_pool(acceptable)
        best = max(acceptable, key=self._profit_penalty_sort_key)
        immediate_net = float(best.get("immediate_trip_net", best.get("estimated_net", 0.0)) or 0.0)
        action_value = self._candidate_action_value(best)
        penalty = float(best.get("current_action_penalty", 0.0) or 0.0)
        report = best.get("tool_report", {}) if isinstance(best.get("tool_report"), dict) else {}
        risk_flags = set(report.get("risk_flags", []) or [])
        defer_flags = {"tight_scheduled_visit", "future_preference_penalty", "future_preference_hard_block"}
        home_compatible = bool(best.get("home_return_compatible"))
        min_net = float(os.environ.get("AGENT_CONFIDENT_TAKE_MIN_NET_YUAN", "80") or 80)
        min_value = float(os.environ.get("AGENT_CONFIDENT_TAKE_MIN_ACTION_VALUE_YUAN", "-300") or -300)
        if home_compatible:
            min_net = min(min_net, float(os.environ.get("AGENT_HOME_COMPATIBLE_MIN_NET_YUAN", "20") or 20))
            min_value = min(min_value, float(os.environ.get("AGENT_HOME_COMPATIBLE_MIN_ACTION_VALUE_YUAN", "-900") or -900))
        max_penalty = float(os.environ.get("AGENT_CONFIDENT_TAKE_MAX_SOFT_PENALTY_YUAN", "1500") or 1500)
        if risk_flags & defer_flags:
            return None
        if immediate_net >= min_net and action_value >= min_value and penalty <= max_penalty:
            return best
        return None

    @staticmethod
    def _preference_first_candidate_pool(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compliant = []
        soft_tradeoff = []
        for item in candidates:
            report = item.get("tool_report", {}) if isinstance(item.get("tool_report"), dict) else {}
            flags = set(report.get("risk_flags", []) or [])
            if not (flags & {"geo_fence_soft_violation", "daily_rest_soft_tradeoff"}):
                compliant.append(item)
            else:
                soft_tradeoff.append(item)
        if not compliant or not soft_tradeoff:
            return compliant or candidates
        best_compliant = max(ModelDecisionService._candidate_action_value_static(item) for item in compliant)
        best_soft = max(ModelDecisionService._candidate_action_value_static(item) for item in soft_tradeoff)
        margin = float(os.environ.get("AGENT_SOFT_TRADEOFF_BEAT_COMPLIANT_MARGIN_YUAN", "600") or 600)
        if best_soft >= best_compliant + margin:
            return candidates
        visibility_margin = float(os.environ.get("AGENT_SOFT_TRADEOFF_VISIBILITY_MARGIN_YUAN", "250") or 250)
        visible_soft = [
            item
            for item in soft_tradeoff
            if ModelDecisionService._candidate_action_value_static(item)
            >= max(
                0.0,
                best_compliant
                - (
                    float(os.environ.get("AGENT_GEO_SOFT_TRADEOFF_VISIBILITY_MARGIN_YUAN", "900") or 900)
                    if ModelDecisionService._candidate_has_geo_soft_tradeoff(item)
                    else visibility_margin
                ),
            )
        ]
        if not visible_soft:
            return compliant
        visible_soft = sorted(visible_soft, key=ModelDecisionService._candidate_action_value_static, reverse=True)[:4]
        return compliant + visible_soft

    @staticmethod
    def _candidate_has_geo_soft_tradeoff(candidate: dict[str, Any]) -> bool:
        report = candidate.get("tool_report", {}) if isinstance(candidate.get("tool_report"), dict) else {}
        return "geo_fence_soft_violation" in set(report.get("risk_flags", []) or []) or isinstance(report.get("geo_fence_soft_escape"), dict)

    @staticmethod
    def _candidate_action_value_static(candidate: dict[str, Any]) -> float:
        value = candidate.get("action_value")
        if value is not None:
            return float(value or 0.0)
        return ModelDecisionService._candidate_net_after_return(candidate) - float(candidate.get("current_action_penalty", 0.0) or 0.0)

    def _candidate_is_acceptable(self, candidate: dict[str, Any]) -> bool:
        report = candidate.get("tool_report", {}) if isinstance(candidate.get("tool_report"), dict) else {}
        risk_flags = set(report.get("risk_flags", []) or [])
        guardrails = self._config.get("guardrails", {})
        reject_risks = set(guardrails.get("reject_risks", ["miss_scheduled_visit", "negative_net"]))
        reject_risks.add("cargo_time_hard_block")
        reject_risks.add("cargo_time_window_too_tight")
        reject_risks.add("cargo_listing_expires_too_soon")
        reject_risks.add("direct_cumulative_task_penalty")
        reject_risks.add("not_at_required_point_during_penalty_window")
        reject_risks.add("deadline_task_penalty")
        if risk_flags & reject_risks:
            return False
        task_penalty = report.get("task_penalty_optimizer_tool", {}) if isinstance(report.get("task_penalty_optimizer_tool"), dict) else {}
        hard_task_penalty = float(os.environ.get("AGENT_HARD_TASK_PENALTY_REJECT_YUAN", "9000") or 9000)
        if self._as_float(task_penalty.get("estimated_action_task_penalty_yuan")) and float(task_penalty.get("estimated_action_task_penalty_yuan") or 0.0) >= hard_task_penalty:
            return False
        pref_risk = report.get("preference_risk_assessment", {}) if isinstance(report.get("preference_risk_assessment"), dict) else {}
        if pref_risk.get("hard_constraint_maybe_violated"):
            return False
        immediate_net = self._candidate_net_after_return(candidate)
        action_value = self._candidate_action_value(candidate)
        min_value = float(os.environ.get("AGENT_MIN_ACTION_VALUE_YUAN", "-200") or -200)
        soft_loss_tolerance = float(os.environ.get("AGENT_SOFT_PENALTY_LOSS_TOLERANCE_YUAN", "1200") or 1200)
        if immediate_net <= 0:
            return False
        if candidate.get("home_return_compatible"):
            home_min_net = float(os.environ.get("AGENT_HOME_COMPATIBLE_ACCEPT_MIN_NET_YUAN", "20") or 20)
            home_min_value = float(os.environ.get("AGENT_HOME_COMPATIBLE_ACCEPT_MIN_VALUE_YUAN", "-1200") or -1200)
            if immediate_net >= home_min_net and action_value >= home_min_value:
                return True
        if action_value >= min_value - soft_loss_tolerance:
            return True
        # Hidden drivers often combine soft preferences in unseen ways. When no
        # hard task is violated, keep earning from clearly profitable orders
        # instead of idling all month on over-estimated soft penalties.
        penalty = float(candidate.get("current_action_penalty", 0.0) or 0.0)
        profit_backstop = float(os.environ.get("AGENT_PROFIT_BACKSTOP_NET_YUAN", "450") or 450)
        max_soft_penalty_ratio = float(os.environ.get("AGENT_MAX_SOFT_PENALTY_TO_NET_RATIO", "2.25") or 2.25)
        return immediate_net >= profit_backstop and penalty <= max(1500.0, immediate_net * max_soft_penalty_ratio)

    def _income_rescue_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        rescue: list[dict[str, Any]] = []
        for item in candidates:
            report = item.get("tool_report", {}) if isinstance(item.get("tool_report"), dict) else {}
            risk_flags = set(report.get("risk_flags", []) or [])
            hard = {
                "cargo_time_hard_block",
                "cargo_time_window_too_tight",
                "cargo_listing_expires_too_soon",
                "direct_cumulative_task_penalty",
                "not_at_required_point_during_penalty_window",
                "deadline_task_penalty",
                "miss_scheduled_visit",
                "negative_net",
            }
            if risk_flags & hard:
                continue
            task_penalty = report.get("task_penalty_optimizer_tool", {}) if isinstance(report.get("task_penalty_optimizer_tool"), dict) else {}
            hard_task_penalty = float(os.environ.get("AGENT_HARD_TASK_PENALTY_REJECT_YUAN", "9000") or 9000)
            if (self._as_float(task_penalty.get("estimated_action_task_penalty_yuan")) or 0.0) >= hard_task_penalty:
                continue
            net_after_return = self._candidate_net_after_return(item)
            action_value = self._candidate_action_value(item)
            soft_loss_tolerance = float(os.environ.get("AGENT_RESCUE_SOFT_LOSS_TOLERANCE_YUAN", "1800") or 1800)
            penalty = float(item.get("current_action_penalty", 0.0) or 0.0)
            penalty_ratio = penalty / max(1.0, net_after_return)
            if net_after_return > 0.0 and action_value >= -soft_loss_tolerance and penalty_ratio < 2.5:
                rescue.append(item)
        if not rescue:
            return None
        return max(self._preference_first_candidate_pool(rescue), key=lambda item: (self._candidate_net_after_return(item), self._candidate_action_value(item)))

    def _candidate_action_value(self, candidate: dict[str, Any]) -> float:
        report = candidate.get("tool_report", {}) if isinstance(candidate.get("tool_report"), dict) else {}
        pref_risk = report.get("preference_risk_assessment", {}) if isinstance(report.get("preference_risk_assessment"), dict) else {}
        immediate_net = self._candidate_net_after_return(candidate)
        penalty = self._as_float(pref_risk.get("current_action_penalty_yuan"))
        if penalty is None:
            penalty = self._as_float(pref_risk.get("expected_penalty_hint_yuan")) or 0.0
        item_penalty = self._as_float(candidate.get("current_action_penalty"))
        if item_penalty is not None:
            penalty = max(float(penalty), float(item_penalty))
        return float(immediate_net) - float(penalty)

    def _market_explore_action(
        self,
        status: dict[str, Any],
        profile: dict[str, Any],
        *,
        items: list[Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        minute_of_day = current_minute % 1440
        if not (8 * 60 <= minute_of_day < 18 * 60):
            return None
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)
        driver_id = str(status.get("driver_id") or getattr(self, "_current_driver_id", "") or "")
        if self._daily_rest_should_block_explore(driver_id, current_minute, lat, lng, profile):
            return None
        visit_action = self._monthly_visit_idle_action(status, profile, candidates)
        if visit_action is not None:
            if visit_action.get("action") == "reposition":
                params = visit_action.get("params", {}) if isinstance(visit_action.get("params"), dict) else {}
                try:
                    target_lat = float(params["latitude"])
                    target_lng = float(params["longitude"])
                except (KeyError, TypeError, ValueError):
                    return None
                travel = distance_to_minutes(haversine_km(lat, lng, target_lat, target_lng))
                if self._reposition_crosses_no_action_window(current_minute, travel, profile):
                    return None
            return visit_action
        reposition_guard = self._recent_reposition_guard(driver_id)
        if int(reposition_guard.get("consecutive_tail") or 0) >= 1:
            return None
        target: tuple[float, float] | None = None
        allow_market_probe = os.environ.get("AGENT_ALLOW_GENERIC_MARKET_REPOSITION", "0").strip().lower() in {"1", "true", "yes"}
        if not allow_market_probe and isinstance(profile.get("geo_fence_bounds"), dict):
            allow_market_probe = os.environ.get("AGENT_ALLOW_BOUNDED_MARKET_REPOSITION", "0").strip().lower() in {"1", "true", "yes"}
        if allow_market_probe and candidates:
            ranked = sorted(
                candidates,
                key=lambda item: (
                    float(item.get("net_after_return", item.get("estimated_net", 0.0)) or 0.0),
                    -float(item.get("pickup_km", 0.0) or 0.0),
                ),
                reverse=True,
            )
            viable = [item for item in ranked if self._market_candidate_anchor_worth_exploring(item)]
            start = viable[0].get("start") if viable else None
            if self._valid_point(start):
                assert isinstance(start, (list, tuple))
                candidate_target = (float(start[0]), float(start[1]))
                if self._explore_target_allowed(lat, lng, candidate_target, profile, reposition_guard, current_minute=current_minute):
                    target = candidate_target
        if allow_market_probe and target is None:
            best_score = -10**18
            for wrapped in items:
                cargo = wrapped.get("cargo", {}) if isinstance(wrapped, dict) else {}
                start = cargo.get("start", {}) if isinstance(cargo, dict) else {}
                end = cargo.get("end", {}) if isinstance(cargo, dict) else {}
                try:
                    point = (float(start["lat"]), float(start["lng"]))
                    end_point = (float(end["lat"]), float(end["lng"]))
                    price = float(cargo.get("price", 0.0) or 0.0)
                    distance = float(wrapped.get("distance_km", haversine_km(lat, lng, point[0], point[1])) or 0.0)
                except (KeyError, TypeError, ValueError):
                    continue
                name = str(cargo.get("cargo_name", ""))
                if any(str(k) and str(k) in name for k in profile.get("avoid_cargo_keywords", [])):
                    continue
                if not self._geo_candidate_allowed(lat, lng, point[0], point[1], end_point[0], end_point[1], profile, current_minute, current_minute + distance_to_minutes(distance)):
                    continue
                if not self._explore_target_allowed(lat, lng, point, profile, reposition_guard, current_minute=current_minute):
                    continue
                density = self._market_target_density_score(point, items, profile)
                min_density_count = int(os.environ.get("AGENT_EXPLORE_MIN_NEARBY_CARGO_COUNT", "2") or 2)
                if int(density.get("count", 0)) < min_density_count and distance > float(os.environ.get("AGENT_EXPLORE_SINGLE_TARGET_MAX_KM", "25") or 25):
                    continue
                score = price + float(density.get("score", 0.0)) - distance * 4.0
                if score > best_score:
                    best_score = score
                    target = point
        if target is not None and haversine_km(lat, lng, target[0], target[1]) < 5.0:
            target = None
        if target is None:
            home = self._daily_home_point(profile)
            anchors: list[tuple[float, float]] = []
            for cargo_rule in profile.get("required_cargos", []) or []:
                if not isinstance(cargo_rule, dict):
                    continue
                point = cargo_rule.get("pickup_point")
                if self._valid_point(point):
                    anchor = (float(point[0]), float(point[1]))
                    if not self._required_cargo_anchor_ready(current_minute, anchor, cargo_rule):
                        continue
                    if self._explore_target_allowed(lat, lng, anchor, profile, reposition_guard, current_minute=current_minute):
                        anchors.append(anchor)
            scored: list[tuple[float, tuple[float, float]]] = []
            for point in anchors:
                if self._valid_point(home) and haversine_km(point[0], point[1], float(home[0]), float(home[1])) < 2.0:
                    continue
                dist = haversine_km(lat, lng, point[0], point[1])
                scored.append((dist, point))
            if scored:
                scored.sort(reverse=True)
                target = scored[0][1]
        if target is None and os.environ.get("AGENT_ALLOW_GENERIC_PREFERENCE_POINT_EXPLORE", "0").strip().lower() in {"1", "true", "yes"}:
            for point in profile.get("preference_points", []) or []:
                if not self._valid_point(point):
                    continue
                anchor = (float(point[0]), float(point[1]))
                if self._matches_future_required_cargo_anchor(current_minute, anchor, profile):
                    continue
                if self._explore_target_allowed(lat, lng, anchor, profile, reposition_guard, current_minute=current_minute):
                    target = anchor
                    break
        if target is None:
            return None
        distance = haversine_km(lat, lng, target[0], target[1])
        if distance < 5.0:
            return None
        max_explore_km = float(os.environ.get("AGENT_MARKET_EXPLORE_MAX_KM", "80") or 80)
        if distance > max_explore_km:
            return None
        travel_minutes = distance_to_minutes(distance)
        if self._reposition_crosses_no_action_window(current_minute, travel_minutes, profile):
            return None
        first_order_deadline = self._as_int(profile.get("first_order_deadline_minute"))
        if first_order_deadline is not None and self._accepted_orders_on_day(driver_id, current_minute // 1440) == 0:
            buffer_minutes = int(os.environ.get("AGENT_FIRST_ORDER_EXPLORE_BUFFER_MINUTES", "15") or 15)
            if current_minute % 1440 < first_order_deadline and (current_minute + travel_minutes) % 1440 > first_order_deadline - buffer_minutes:
                return None
        planned = action("reposition", {"latitude": target[0], "longitude": target[1]})
        planned["reason_code"] = "bounded_market_reposition" if isinstance(profile.get("geo_fence_bounds"), dict) else "task_anchor_reposition"
        return planned

    def _market_candidate_anchor_worth_exploring(self, candidate: dict[str, Any]) -> bool:
        net = self._as_float(candidate.get("net_after_return"))
        if net is None:
            net = self._as_float(candidate.get("estimated_net"))
        immediate_net = self._as_float(candidate.get("immediate_trip_net"))
        if immediate_net is None:
            immediate_net = self._as_float(candidate.get("estimated_net")) or 0.0
        action_value = self._candidate_action_value(candidate)
        min_net = float(os.environ.get("AGENT_EXPLORE_ANCHOR_MIN_NET_YUAN", "50") or 50)
        return max(float(net or 0.0), float(immediate_net or 0.0)) >= min_net and action_value >= -float(os.environ.get("AGENT_EXPLORE_ANCHOR_MAX_LOSS_YUAN", "300") or 300)

    def _market_target_density_score(self, target: tuple[float, float], items: list[Any], profile: dict[str, Any]) -> dict[str, Any]:
        radius_km = float(os.environ.get("AGENT_EXPLORE_DENSITY_RADIUS_KM", "18") or 18)
        count = 0
        score = 0.0
        for wrapped in items:
            cargo = wrapped.get("cargo", {}) if isinstance(wrapped, dict) else {}
            start = cargo.get("start", {}) if isinstance(cargo, dict) else {}
            end = cargo.get("end", {}) if isinstance(cargo, dict) else {}
            try:
                point = (float(start["lat"]), float(start["lng"]))
                end_point = (float(end["lat"]), float(end["lng"]))
                price = float(cargo.get("price", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            name = str(cargo.get("cargo_name", ""))
            if any(str(k) and str(k) in name for k in profile.get("avoid_cargo_keywords", [])):
                continue
            distance = haversine_km(target[0], target[1], point[0], point[1])
            if distance > radius_km:
                continue
            if not self._geo_candidate_allowed(target[0], target[1], point[0], point[1], end_point[0], end_point[1], profile):
                continue
            count += 1
            score += max(0.0, price - distance * 3.0)
        return {"count": count, "score": score}

    def _required_cargo_anchor_ready(self, current_minute: int, anchor: tuple[float, float], cargo_rule: dict[str, Any]) -> bool:
        online = self._as_int(cargo_rule.get("online_minute"))
        if online is None:
            return True
        max_lead = int(os.environ.get("AGENT_REQUIRED_CARGO_EXPLORE_LEAD_MINUTES", "480") or 480)
        return current_minute >= online - max_lead

    def _matches_future_required_cargo_anchor(self, current_minute: int, anchor: tuple[float, float], profile: dict[str, Any]) -> bool:
        for cargo_rule in profile.get("required_cargos", []) or []:
            if not isinstance(cargo_rule, dict):
                continue
            point = cargo_rule.get("pickup_point")
            if not self._valid_point(point):
                continue
            candidate = (float(point[0]), float(point[1]))
            if haversine_km(anchor[0], anchor[1], candidate[0], candidate[1]) > 2.0:
                continue
            if not self._required_cargo_anchor_ready(current_minute, candidate, cargo_rule):
                return True
        return False

    def _daily_rest_should_block_explore(self, driver_id: str, current_minute: int, lat: float, lng: float, profile: dict[str, Any]) -> bool:
        report = self._time_task_progress_tool.progress_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=profile,
            history=self._history_records(driver_id),
        )
        for task in report.get("periodic_tasks", []) if isinstance(report, dict) else []:
            if not isinstance(task, dict) or task.get("type") != "daily_continuous_rest":
                continue
            if task.get("status") == "done":
                return False
            latest_start = self._as_int(task.get("latest_start_minute"))
            remaining = self._as_int(task.get("remaining_rest_minutes")) or 0
            rest_buffer = int(os.environ.get("AGENT_REST_EXPLORE_BUFFER_MINUTES", "45") or 45)
            if remaining > 0 and latest_start is not None and current_minute >= latest_start - rest_buffer:
                return True
        return False

    def _explore_target_allowed(
        self,
        lat: float,
        lng: float,
        target: tuple[float, float],
        profile: dict[str, Any],
        reposition_guard: dict[str, Any],
        current_minute: int | None = None,
    ) -> bool:
        if not self._point_allowed_by_profile(target[0], target[1], profile):
            return False
        if current_minute is not None:
            travel = distance_to_minutes(haversine_km(lat, lng, target[0], target[1]))
            if self._point_blocked_by_avoid_region(target[0], target[1], current_minute, current_minute + travel, profile):
                return False
        if self._point_allowed_by_profile(lat, lng, profile) and not self._geo_segment_allowed(lat, lng, target[0], target[1], profile):
            return False
        if self._recent_reposition_backtrack(target, reposition_guard):
            return False
        return True

    def _recent_reposition_guard(self, driver_id: str) -> dict[str, Any]:
        if not driver_id:
            return {}
        records = self._history_records(driver_id)[-8:]
        recent_targets: list[tuple[float, float]] = []
        last_reposition_before: tuple[float, float] | None = None
        last_reposition_after: tuple[float, float] | None = None
        for record in records:
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "reposition":
                continue
            before = record.get("position_before") if isinstance(record.get("position_before"), dict) else {}
            after = record.get("position_after") if isinstance(record.get("position_after"), dict) else {}
            try:
                before_point = (float(before.get("lat")), float(before.get("lng")))
                after_point = (float(after.get("lat")), float(after.get("lng")))
            except (TypeError, ValueError):
                continue
            last_reposition_before = before_point
            last_reposition_after = after_point
            recent_targets.append(after_point)
        consecutive_tail = 0
        for record in reversed(records):
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "reposition":
                break
            consecutive_tail += 1
        return {
            "last_before": last_reposition_before,
            "last_after": last_reposition_after,
            "recent_targets": recent_targets,
            "consecutive_tail": consecutive_tail,
        }

    @staticmethod
    def _recent_reposition_backtrack(target: tuple[float, float], reposition_guard: dict[str, Any]) -> bool:
        last_reposition_before = reposition_guard.get("last_before")
        last_reposition_after = reposition_guard.get("last_after")
        recent_targets = reposition_guard.get("recent_targets", [])
        if int(reposition_guard.get("consecutive_tail") or 0) >= 2:
            return True
        if last_reposition_before is not None and last_reposition_after is not None:
            if haversine_km(target[0], target[1], last_reposition_before[0], last_reposition_before[1]) <= 1.0:
                return True
            if haversine_km(target[0], target[1], last_reposition_after[0], last_reposition_after[1]) <= 3.0:
                return True
        near_recent_count = sum(1 for point in recent_targets if haversine_km(target[0], target[1], point[0], point[1]) <= 2.0)
        return near_recent_count >= 2

    def _monthly_visit_idle_action(self, status: dict[str, Any], profile: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Use idle daytime to complete periodic visit days without blocking good cargo."""
        driver_id = str(status.get("driver_id") or getattr(self, "_current_driver_id", "") or "")
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)
        if any(self._candidate_is_acceptable(item) for item in candidates):
            return None
        vf = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        required = max(0, self._as_int(vf.get("required_days")) or 0)
        point = vf.get("point")
        if required <= 0 or not self._valid_point(point):
            return None
        assert isinstance(point, (list, tuple))
        target = (float(point[0]), float(point[1]))
        radius_km = float(vf.get("radius_km") or 1.0)
        current_day = current_minute // 1440
        visited = self._visited_days_near(driver_id, target, radius_km)
        at_point_now = haversine_km(lat, lng, target[0], target[1]) <= radius_km
        if at_point_now:
            if current_day in visited:
                return None
            planned = action("wait", {"duration_minutes": 1})
            planned["reason_code"] = "monthly_visit_credit_today"
            return planned
        if current_day in visited or len(visited) >= required:
            return None
        missing = required - len(visited)
        remaining_days = HORIZON_DAYS - current_day
        target_done_by_now = self._monthly_visit_target_by_day(required, current_day)
        behind_pace = len(visited) < target_done_by_now
        urgent = missing > 0 and remaining_days <= missing + 3
        if not (behind_pace or urgent):
            return None
        distance = haversine_km(lat, lng, target[0], target[1])
        max_idle_km = float(os.environ.get("AGENT_VISIT_IDLE_MAX_KM", os.environ.get("AGENT_VISIT_EARLY_MAX_KM", "120")) or 120)
        max_urgent_km = float(os.environ.get("AGENT_VISIT_URGENT_MAX_KM", "600") or 600)
        max_distance = max_urgent_km if urgent else max_idle_km
        if behind_pace and current_day >= 10:
            max_distance = max(max_distance, max_urgent_km)
        if distance > max_distance:
            return None
        planned = action("reposition", {"latitude": target[0], "longitude": target[1]})
        planned["reason_code"] = "monthly_visit_idle_position" if behind_pace and not urgent else "monthly_visit_due_position"
        planned["monthly_visit_context"] = {
            "completed_days": len(visited),
            "required_days": required,
            "missing_days": missing,
            "distance_km": round(distance, 2),
            "behind_pace": behind_pace,
        }
        return planned

    def _planning_board(self, status: dict[str, Any], profile: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        driver_id = str(status.get("driver_id") or getattr(self, "_current_driver_id", ""))
        safe = [item for item in candidates if not item.get("tool_report", {}).get("risk_flags")]
        top = max(candidates, key=self._profit_penalty_sort_key) if candidates else None
        top_safe = max(safe, key=self._profit_penalty_sort_key) if safe else None
        next_visit = self._next_scheduled_visit_after(current_minute, profile)
        commitment_report = self._commitment_sequence_tool.commitment_report(
            current_minute=current_minute,
            lat=float(status.get("current_lat", 0.0) or 0.0),
            lng=float(status.get("current_lng", 0.0) or 0.0),
            profile=profile,
            history=self._recent_history_records(driver_id),
        )
        history_records = self._recent_history_records(driver_id)
        time_task_report = self._time_task_progress_tool.progress_report(
            current_minute=current_minute,
            lat=float(status.get("current_lat", 0.0) or 0.0),
            lng=float(status.get("current_lng", 0.0) or 0.0),
            profile=profile,
            history=history_records,
        )
        region_report = self._region_preference_tool.evaluate(
            current_point=(float(status.get("current_lat", 0.0) or 0.0), float(status.get("current_lng", 0.0) or 0.0)),
            candidate=None,
            profile=profile,
            time_task_report=time_task_report,
        )
        baseline_task_penalty = self._task_penalty_optimizer_tool.evaluate_action(
            current_minute=current_minute,
            current_point=(float(status.get("current_lat", 0.0) or 0.0), float(status.get("current_lng", 0.0) or 0.0)),
            action=action("wait", {"duration_minutes": 30}),
            profile=profile,
            time_task_report=time_task_report,
            candidate=None,
        )
        classification_report = self._preference_classification_tool.classify_profile(profile)
        inactivity_report = self._inactivity_opportunity_report(
            driver_id=driver_id,
            current_minute=current_minute,
            candidates=candidates,
            history_records=history_records,
        )
        return {
            "current_time": wall_time(current_minute),
            "agent_config": self._public_agent_config(),
            "candidate_count": len(candidates),
            "safe_candidate_count": len(safe),
            "top_algorithm_cargo_id": None if top is None else top.get("cargo_id"),
            "top_safe_cargo_id": None if top_safe is None else top_safe.get("cargo_id"),
            "next_scheduled_visit": next_visit,
            "commitment_sequence_report": commitment_report,
            "time_task_progress_tool": time_task_report,
            "region_preference_tool": region_report,
            "task_penalty_optimizer_tool": baseline_task_penalty,
            "inactivity_opportunity_tool": inactivity_report,
            "preference_classification_tool": {
                "priority_order": classification_report.get("priority_order", []),
                "universal_audit_checklist": classification_report.get("universal_audit_checklist", []),
                "coverage_gaps": classification_report.get("coverage_gaps", []),
                "unknown_handling_policy": classification_report.get("unknown_handling_policy", ""),
            },
            "preference_card_count": len(profile.get("preference_cards", []) if isinstance(profile.get("preference_cards"), list) else []),
            "duplicate_preference_group_count": len(profile.get("duplicate_preference_groups", []) if isinstance(profile.get("duplicate_preference_groups"), list) else []),
            "unknown_preference_count": len(profile.get("unknown_preferences", []) if isinstance(profile.get("unknown_preferences"), list) else []),
            "unknown_preference_group_count": len(profile.get("unknown_preference_groups", []) if isinstance(profile.get("unknown_preference_groups"), list) else []),
            "dynamic_preference_rule_count": len(profile.get("dynamic_preference_rules", []) if isinstance(profile.get("dynamic_preference_rules"), list) else []),
            "risk_policy": profile.get("risk_policy", {}),
            "decision_policy": (
                "Compare net_after_return and total current_action_penalty as a pair; current_action_penalty includes future preference penalties from action_preference_guard_tool. "
                "A lower-profit cargo can be better if it sharply reduces current or future preference penalty. "
                "Waiting is acceptable when all available profit/penalty tradeoffs are poor, a hard time/preference block exists, or profile extraction looks suspicious. "
                "Do not keep waiting if inactivity_opportunity_tool says the driver has waited too long and there is a positive low-risk action_value cargo. Never choose cargo with miss_scheduled_visit. "
                "When commitment_sequence_report recommends an action, treat it as a high-priority sequence constraint."
            ),
        }

    def _next_scheduled_visit_after(self, current_minute: int, profile: dict[str, Any]) -> dict[str, Any] | None:
        future: list[dict[str, Any]] = []
        for item in profile.get("scheduled_visits", []):
            if not isinstance(item, dict):
                continue
            day = self._as_int(item.get("day"))
            if day is None:
                continue
            deadline = self._as_int(item.get("arrive_before_minute"))
            target_minute = day * 1440 + (deadline if deadline is not None else 20 * 60)
            if target_minute >= current_minute:
                copied = dict(item)
                copied["target_minute"] = target_minute
                copied["target_time"] = wall_time(target_minute)
                future.append(copied)
        return min(future, key=lambda item: int(item["target_minute"])) if future else None

    def _public_agent_config(self) -> dict[str, Any]:
        return {
            "context_tools": self._config.get("context_tools", {}),
            "agent_roles": self._config.get("agent_roles", {}),
            "runtime_limits": self._config.get("runtime_limits", {}),
            "guardrails": self._config.get("guardrails", {}),
            "llm_core": {
                "api_first": self._config.get("llm_core", {}).get("api_first"),
                "strict_json": self._config.get("llm_core", {}).get("strict_json"),
                "llm_choice_enabled": self._enable_llm_choice,
                "llm_disabled": self._disable_llm,
            },
        }

    def _record_usage(self, response: dict[str, Any]) -> None:
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        if not isinstance(usage, dict):
            return
        current = getattr(self, "_decision_usage", None)
        if not isinstance(current, dict):
            current = dict(ZERO_USAGE)
            self._decision_usage = current
        for key in ZERO_USAGE:
            try:
                current[key] = int(current.get(key, 0) or 0) + int(usage.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
        driver_id = getattr(self, "_current_driver_id", None)
        if driver_id:
            state = self._driver_state(str(driver_id))
            state["llm_calls"] = int(state.get("llm_calls", 0) or 0) + 1
            state["llm_total_tokens"] = int(state.get("llm_total_tokens", 0) or 0) + int(usage.get("total_tokens", 0) or 0)

    def _driver_state(self, driver_id: str) -> dict[str, Any]:
        state = self._driver_runtime.get(driver_id)
        if state is None:
            state = {
                "wall_start": time.monotonic(),
                "llm_calls": 0,
                "llm_total_tokens": 0,
                "last_planner_llm_minute": -10**9,
            }
            self._driver_runtime[driver_id] = state
        return state

    def _can_use_llm(self, driver_id: str) -> bool:
        if self._disable_llm:
            return False
        limits = self._config.get("runtime_limits", {})
        state = self._driver_state(driver_id)
        wall_budget = float(os.environ.get("AGENT_LLM_WALL_BUDGET_SECONDS_PER_DRIVER", str(limits.get("llm_wall_budget_seconds_per_driver", 3300))) or 3300)
        token_budget = int(os.environ.get("AGENT_LLM_TOKEN_BUDGET_PER_DRIVER", str(limits.get("llm_token_budget_per_driver", 120000))) or 120000)
        max_calls = int(os.environ.get("AGENT_LLM_MAX_CALLS_PER_DRIVER", str(limits.get("llm_max_calls_per_driver", 12))) or 12)
        if time.monotonic() - float(state.get("wall_start", time.monotonic())) > wall_budget:
            return False
        if int(state.get("llm_total_tokens", 0) or 0) >= token_budget:
            return False
        return int(state.get("llm_calls", 0) or 0) < max_calls

    def _llm_decision_timeout_seconds(self) -> float:
        limits = self._config.get("runtime_limits", {})
        raw = os.environ.get(
            "AGENT_LLM_DECISION_TIMEOUT_SECONDS",
            str(limits.get("llm_decision_timeout_seconds", 55)),
        )
        try:
            value = float(raw or 55)
        except (TypeError, ValueError):
            value = 55.0
        return max(3.0, min(25.0, value))

    def _call_model_chat_completion(self, payload: dict[str, Any], purpose: str) -> dict[str, Any]:
        timeout_seconds = self._llm_decision_timeout_seconds()
        return self._call_model_chat_completion_with_thread_timeout(payload, purpose, timeout_seconds)

    def _call_model_chat_completion_with_thread_timeout(self, payload: dict[str, Any], purpose: str, timeout_seconds: float) -> dict[str, Any]:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def _target() -> None:
            try:
                result_queue.put((True, self._api.model_chat_completion(payload)), block=False)
            except Exception as exc:
                try:
                    result_queue.put((False, exc), block=False)
                except queue.Full:
                    pass

        worker = threading.Thread(target=_target, name=f"agent-llm-{purpose}", daemon=True)
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            raise TimeoutError(f"{purpose} LLM call exceeded {timeout_seconds:.1f}s")
        ok, value = result_queue.get_nowait()
        if ok:
            return value
        raise value

    def _with_usage(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("action") == "reposition" and not result.get("reason_code"):
            result = action("wait", {"duration_minutes": 60})
        usage = getattr(self, "_decision_usage", None)
        if isinstance(usage, dict) and any(int(usage.get(key, 0) or 0) for key in ZERO_USAGE):
            result = dict(result)
            result["model_usage"] = {key: int(usage.get(key, 0) or 0) for key in ZERO_USAGE}
        return result

    def _planner_profile_view(self, profile: dict[str, Any]) -> dict[str, Any]:
        cards = [card for card in profile.get("preference_cards", []) if isinstance(card, dict)]
        high_cards = [
            {
                "id": card.get("id"),
                "types": card.get("types"),
                "severity": card.get("severity"),
                "tradeoff_mode": card.get("tradeoff_mode"),
                "penalty_amount": card.get("penalty_amount"),
                "penalty_cap": card.get("penalty_cap"),
                "risk_key": card.get("risk_key"),
                "planner_hint": card.get("planner_hint"),
            }
            for card in cards
            if card.get("severity") in {"critical", "high"} or card.get("tradeoff_mode") == "unknown"
        ][:6]
        return {
            "avoid_cargo_keywords": profile.get("avoid_cargo_keywords", []),
            "avoid_regions": profile.get("avoid_regions", []),
            "daily_rest": profile.get("daily_rest", {}),
            "required_off_days": profile.get("required_off_days", 0),
            "pickup_deadhead_max_km": profile.get("pickup_deadhead_max_km"),
            "monthly_deadhead_limit_km": profile.get("monthly_deadhead_limit_km"),
            "max_haul_km": profile.get("max_haul_km"),
            "first_order_deadline_minute": profile.get("first_order_deadline_minute"),
            "daily_order_limit": profile.get("daily_order_limit"),
            "geo_fence_bounds": profile.get("geo_fence_bounds"),
            "forbidden_circles": profile.get("forbidden_circles", []),
            "required_cargos": profile.get("required_cargos", []),
            "temporary_events": profile.get("temporary_events", []),
            "long_sequence_commitments": profile.get("long_sequence_commitments", [])[:3],
            "cumulative_time_penalty_rules": profile.get("cumulative_time_penalty_rules", [])[:3],
            "visit_frequency": profile.get("visit_frequency", {}),
            "scheduled_visits": profile.get("scheduled_visits", [])[:3],
            "preference_card_count": len(cards),
            "high_priority_preference_cards": high_cards,
            "duplicate_preference_groups": profile.get("duplicate_preference_groups", [])[:3],
            "unknown_preferences": profile.get("unknown_preferences", [])[:3],
            "unknown_attribute_tags": profile.get("unknown_attribute_tags", [])[:6],
            "dynamic_preference_rules": profile.get("dynamic_preference_rules", [])[:5],
            "risk_policy": profile.get("risk_policy", {}),
        }

    def _compact_planner_profile_view(self, profile: dict[str, Any]) -> dict[str, Any]:
        rules = []
        for rule in profile.get("dynamic_preference_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            rules.append(
                {
                    "label": rule.get("label"),
                    "effect": rule.get("effect"),
                    "severity": rule.get("severity"),
                    "penalty": rule.get("per_violation_penalty_yuan"),
                }
            )
        return {
            "avoid_cargo_keywords": profile.get("avoid_cargo_keywords", []),
            "daily_rest": profile.get("daily_rest", {}),
            "required_cargo_ids": [item.get("cargo_id") for item in profile.get("required_cargos", []) if isinstance(item, dict)][:5],
            "temporary_event_count": len(profile.get("temporary_events", []) or []),
            "long_sequence_count": len(profile.get("long_sequence_commitments", []) or []),
            "visit_frequency": profile.get("visit_frequency", {}),
            "dynamic_rules": rules[:4],
            "unknown_count": len(profile.get("unknown_preferences", []) or []),
        }

    @staticmethod
    def _force_review_every_action() -> bool:
        return str(os.environ.get("AGENT_FORCE_LLM_REVIEW_EVERY_ACTION", "0")).strip().lower() not in {"0", "false", "no"}

    def _should_use_planner_llm(self, driver_id: str, current_minute: int, candidates: list[dict[str, Any]]) -> bool:
        if not self._can_use_llm(driver_id):
            return False
        if not candidates:
            return False
        acceptable = [item for item in candidates if self._candidate_is_acceptable(item)]
        if not acceptable:
            return False
        if os.environ.get("AGENT_LLM_ALWAYS_PLAN", "0").strip().lower() in {"1", "true", "yes"}:
            return True
        best = max(acceptable, key=self._profit_penalty_sort_key)
        action_value = self._candidate_action_value(best)
        immediate_net = float(best.get("immediate_trip_net", best.get("estimated_net", 0.0)) or 0.0)
        penalty = float(best.get("current_action_penalty", 0.0) or 0.0)
        state = self._driver_state(driver_id)
        min_interval = int(os.environ.get("AGENT_PLANNER_LLM_MIN_INTERVAL_MINUTES", "360") or 360)
        interval_ready = current_minute - int(state.get("last_planner_llm_minute", -10**9) or -10**9) >= min_interval
        if interval_ready and self._geo_soft_tradeoff_needs_llm(acceptable):
            return True
        if action_value >= 0:
            return False
        safe_direct_value = float(os.environ.get("AGENT_SKIP_PLANNER_MIN_ACTION_VALUE_YUAN", "150") or 150)
        if action_value >= safe_direct_value and penalty <= max(1500.0, immediate_net * 0.75):
            return False
        if not interval_ready:
            return False
        if self._profile_has_unknown_or_dynamic_risk(driver_id):
            return True
        if len(acceptable) >= 2:
            ordered = sorted(acceptable, key=self._profit_penalty_sort_key, reverse=True)
            top_value = self._candidate_action_value(ordered[0])
            second_value = self._candidate_action_value(ordered[1])
            return abs(top_value - second_value) <= float(os.environ.get("AGENT_PLANNER_LLM_CLOSE_VALUE_GAP_YUAN", "250") or 250)
        return action_value < safe_direct_value

    def _geo_soft_tradeoff_needs_llm(self, acceptable: list[dict[str, Any]]) -> bool:
        if os.environ.get("AGENT_LLM_REVIEW_GEO_SOFT_TRADEOFF", "1").strip().lower() not in {"1", "true", "yes"}:
            return False
        soft = [item for item in acceptable if self._candidate_has_geo_soft_tradeoff(item)]
        if not soft:
            return False
        compliant = [item for item in acceptable if not self._candidate_has_geo_soft_tradeoff(item)]
        best_soft = max(self._candidate_action_value(item) for item in soft)
        if not compliant:
            return best_soft >= float(os.environ.get("AGENT_GEO_SOFT_LLM_MIN_ACTION_VALUE_YUAN", "300") or 300)
        best_compliant = max(self._candidate_action_value(item) for item in compliant)
        margin = float(os.environ.get("AGENT_GEO_SOFT_LLM_COMPARE_MARGIN_YUAN", "700") or 700)
        return best_soft >= best_compliant - margin

    def _profile_has_unknown_or_dynamic_risk(self, driver_id: str) -> bool:
        cached = self._llm_profile_cache.get(driver_id)
        profile = cached[1] if cached else {}
        if not isinstance(profile, dict):
            return False
        if profile.get("unknown_preferences") or profile.get("unknown_attribute_tags") or profile.get("unknown_preference_groups"):
            return True
        for rule in profile.get("dynamic_preference_rules", []) or []:
            if isinstance(rule, dict) and rule.get("effect") == "hard_reject":
                return True
        return False

    def _planner_evidence_board(self, planning_board: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        commitment = planning_board.get("commitment_sequence_report", {}) if isinstance(planning_board.get("commitment_sequence_report"), dict) else {}
        time_tasks = planning_board.get("time_task_progress_tool", {}) if isinstance(planning_board.get("time_task_progress_tool"), dict) else {}
        baseline_penalty = planning_board.get("task_penalty_optimizer_tool", {}) if isinstance(planning_board.get("task_penalty_optimizer_tool"), dict) else {}
        classification = planning_board.get("preference_classification_tool", {}) if isinstance(planning_board.get("preference_classification_tool"), dict) else {}
        inactivity = planning_board.get("inactivity_opportunity_tool", {}) if isinstance(planning_board.get("inactivity_opportunity_tool"), dict) else {}
        hard_blocks: list[dict[str, Any]] = []
        must_do_now: list[dict[str, Any]] = []
        commitment_action = commitment.get("recommended_action")
        if isinstance(commitment_action, dict):
            must_do_now.append(
                {
                    "source": "commitment_sequence_tool",
                    "action": commitment_action,
                    "why": "ordered/mandatory commitment has a current next step; do not skip previous steps or required wait/stay phases",
                    "source_text": (commitment.get("commitments") or [{}])[0].get("source_text") if commitment.get("commitments") else None,
                }
            )
        for task in (time_tasks.get("urgent_tasks", []) or [])[:3]:
            if isinstance(task, dict):
                must_do_now.append(
                    {
                        "source": "time_task_progress_tool",
                        "task_id": task.get("id"),
                        "task_type": task.get("type"),
                        "deadline_minute": task.get("deadline_minute"),
                        "leave_by_minute": task.get("leave_by_minute"),
                        "point": task.get("point"),
                    }
                )
        for task in (time_tasks.get("overdue_tasks", []) or [])[:3]:
            if isinstance(task, dict):
                must_do_now.append(
                    {
                        "source": "time_task_progress_tool",
                        "task_id": task.get("id"),
                        "task_type": task.get("type"),
                        "point": task.get("point"),
                    }
                )
        for item in candidates[:5]:
            report = item.get("tool_report", {}) if isinstance(item.get("tool_report"), dict) else {}
            flags = list(dict.fromkeys(report.get("risk_flags", []) or []))
            blocked = [
                flag
                for flag in flags
                if flag
                in {
                    "repeat_daily_rest_violation_risk",
                    "cargo_time_hard_block",
                    "cargo_time_window_too_tight",
                    "cargo_listing_expires_too_soon",
                    "direct_cumulative_task_penalty",
                    "not_at_required_point_during_penalty_window",
                }
            ]
            if blocked:
                hard_blocks.append({"cargo_id": item.get("cargo_id"), "risk_flags": blocked, "why": "candidate should not be selected unless no lower-loss legal action exists"})
        return {
            "read_first": True,
            "hard_blocks": hard_blocks[:6],
            "must_do_now": must_do_now[:6],
            "penalty_ledger": {
                "baseline_wait_30min_penalty": baseline_penalty.get("estimated_action_task_penalty_yuan"),
                "baseline_wait_risk_flags": baseline_penalty.get("risk_flags", []),
                "inactivity_opportunity_cost_yuan": inactivity.get("estimated_opportunity_loss_yuan"),
                "inactivity_risk_flags": inactivity.get("risk_flags", []),
            },
            "candidate_comparison": [self._candidate_decision_summary(item) for item in candidates[:5]],
            "decision_checklist": [
                "Treat hard_blocks as true timing/legal blockers; compare soft preference losses economically.",
                "Honor must_do_now if present.",
                "Choose max positive action_value after penalties.",
                "Wait only for mandatory task/rest or poor tradeoff.",
            ],
        }

    def _inactivity_opportunity_report(
        self,
        *,
        driver_id: str,
        current_minute: int,
        candidates: list[dict[str, Any]],
        history_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        accepted_records = []
        wait_minutes_since_accept = 0
        idle_minutes_since_productive = 0
        for record in history_records:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            name = str(act.get("action") or "")
            end = self._as_int(result.get("simulation_progress_minutes"))
            elapsed = self._as_int(record.get("action_exec_cost_minutes", record.get("step_elapsed_minutes", 0))) or 0
            if name == "take_order" and bool(result.get("accepted", False)):
                accepted_records.append(record)
                wait_minutes_since_accept = 0
                idle_minutes_since_productive = 0
            elif name == "wait":
                wait_minutes_since_accept += max(0, elapsed)
                idle_minutes_since_productive += max(0, elapsed)
            elif name in {"take_order", "reposition"}:
                idle_minutes_since_productive = 0
            if end is not None and end > current_minute:
                current_minute = end
        day = current_minute // 1440
        accepted_today = self._accepted_orders_on_day(driver_id, day) if driver_id else 0
        acceptable = [item for item in candidates if self._candidate_is_acceptable(item)]
        best = max(acceptable, key=self._profit_penalty_sort_key) if acceptable else None
        best_value = self._candidate_action_value(best) if best is not None else 0.0
        best_net = float(best.get("net_after_return", best.get("estimated_net", 0.0)) or 0.0) if best is not None else 0.0
        best_penalty = float(best.get("current_action_penalty", 0.0) or 0.0) if best is not None else 0.0
        force_take_value = float(os.environ.get("AGENT_FORCE_TAKE_MIN_ACTION_VALUE", "120") or 120)
        long_idle_minutes = int(os.environ.get("AGENT_LONG_IDLE_MINUTES", "360") or 360)
        severe_idle_minutes = int(os.environ.get("AGENT_SEVERE_IDLE_MINUTES", "720") or 720)
        flags: list[str] = []
        if wait_minutes_since_accept >= long_idle_minutes:
            flags.append("long_time_without_accepted_order")
        if wait_minutes_since_accept >= severe_idle_minutes:
            flags.append("severe_income_starvation_risk")
        if accepted_today <= 0 and current_minute % 1440 >= 12 * 60:
            flags.append("no_order_today_after_noon")
        should_take = best is not None and best_value >= force_take_value and ("severe_income_starvation_risk" in flags or "long_time_without_accepted_order" in flags or accepted_today == 0)
        return {
            "tool_name": "inactivity_opportunity_tool",
            "accepted_orders_total": len(accepted_records),
            "accepted_orders_today": accepted_today,
            "wait_minutes_since_last_accepted_order": wait_minutes_since_accept,
            "idle_minutes_since_productive_action": idle_minutes_since_productive,
            "best_acceptable_cargo_id": None if best is None else best.get("cargo_id"),
            "best_acceptable_action_value_yuan": round(best_value, 2),
            "best_acceptable_net_after_return_yuan": round(best_net, 2),
            "best_acceptable_penalty_yuan": round(best_penalty, 2),
            "force_take_min_action_value_yuan": force_take_value,
            "estimated_opportunity_loss_yuan": round(max(0.0, wait_minutes_since_accept / 60.0 * 80.0), 2),
            "recommendation": "take_positive_order_now" if should_take else "wait_allowed_if_tradeoff_poor",
            "risk_flags": flags,
        }

    @staticmethod
    def _candidate_decision_summary(candidate: dict[str, Any]) -> dict[str, Any]:
        report = candidate.get("tool_report", {}) if isinstance(candidate.get("tool_report"), dict) else {}
        support = report.get("decision_support_tools", {}) if isinstance(report.get("decision_support_tools"), dict) else {}
        task_penalty = report.get("task_penalty_optimizer_tool", {}) if isinstance(report.get("task_penalty_optimizer_tool"), dict) else {}
        action_guard = report.get("action_preference_guard", {}) if isinstance(report.get("action_preference_guard"), dict) else {}
        pref = report.get("preference_risk_assessment", {}) if isinstance(report.get("preference_risk_assessment"), dict) else {}
        flags = list(dict.fromkeys(report.get("risk_flags", []) or []))
        hard_flags = [
            flag
            for flag in flags
            if flag
            in {
                "cargo_time_hard_block",
                "cargo_time_window_too_tight",
                "cargo_listing_expires_too_soon",
                "direct_cumulative_task_penalty",
            }
        ]
        action_value = float(candidate.get("action_value", 0.0) or 0.0)
        task_penalty_yuan = float(task_penalty.get("estimated_action_task_penalty_yuan", 0.0) or 0.0)
        hard_task_penalty = float(os.environ.get("AGENT_HARD_TASK_PENALTY_REJECT_YUAN", "9000") or 9000)
        if hard_flags or task_penalty_yuan >= hard_task_penalty or pref.get("hard_constraint_maybe_violated"):
            verdict = "reject"
        elif action_value < 0:
            verdict = "avoid_negative_value"
        elif flags or action_guard.get("estimated_future_preference_penalty_yuan"):
            verdict = "caution_only_if_best_tradeoff"
        else:
            verdict = "candidate_ok_if_top_value"
        reasons = []
        if hard_flags:
            reasons.append(f"hard_flags={hard_flags[:4]}")
        if task_penalty_yuan:
            reasons.append(f"task_penalty_yuan={round(task_penalty_yuan, 2)}")
        if action_guard.get("estimated_future_preference_penalty_yuan"):
            reasons.append(f"future_preference_penalty={action_guard.get('estimated_future_preference_penalty_yuan')}")
        support_flags = support.get("risk_flags", []) if isinstance(support.get("risk_flags"), list) else []
        if support_flags:
            reasons.append(f"support_risks={support_flags[:4]}")
        if not reasons:
            reasons.append("no major tool risk after compact checks")
        return {
            "cargo_id": candidate.get("cargo_id"),
            "verdict": verdict,
            "action_value": round(action_value, 2),
            "net_after_return": candidate.get("net_after_return"),
            "current_action_penalty": candidate.get("current_action_penalty"),
            "time_reliability_score": candidate.get("time_reliability_score"),
            "finish_time": candidate.get("finish_time"),
            "top_reasons": reasons[:5],
            "risk_flags": flags[:8],
        }

    @staticmethod
    def _compact_candidate_for_llm(candidate: dict[str, Any]) -> dict[str, Any]:
        report = candidate.get("tool_report", {}) if isinstance(candidate.get("tool_report"), dict) else {}
        pref = report.get("preference_risk_assessment", {}) if isinstance(report.get("preference_risk_assessment"), dict) else {}
        net_after_return = ModelDecisionService._candidate_net_after_return(candidate)
        current_penalty = float(candidate.get("current_action_penalty") or pref.get("current_action_penalty_yuan") or pref.get("expected_penalty_hint_yuan") or 0.0)
        task_penalty = report.get("task_penalty_optimizer_tool", {}) if isinstance(report.get("task_penalty_optimizer_tool"), dict) else {}
        geo_escape = report.get("geo_fence_soft_escape", {}) if isinstance(report.get("geo_fence_soft_escape"), dict) else {}
        flags = list(dict.fromkeys(report.get("risk_flags", []) or []))
        compact = {
            "cargo_id": candidate.get("cargo_id"),
            "av": round(net_after_return - current_penalty, 2),
            "net": round(net_after_return, 2),
            "trip_net": round(float(candidate.get("immediate_trip_net", candidate.get("estimated_net", 0.0)) or 0.0), 2),
            "pen": round(current_penalty, 2),
            "hourly": round(float(candidate.get("hourly", 0.0) or 0.0), 2),
            "pickup_km": round(float(candidate.get("pickup_km", 0.0) or 0.0), 2),
            "haul_km": round(float(candidate.get("haul_km", 0.0) or candidate.get("distance_km", 0.0) or 0.0), 2),
            "finish": candidate.get("finish_minute"),
            "task_pen": task_penalty.get("estimated_action_task_penalty_yuan"),
            "verdict": ModelDecisionService._candidate_decision_summary(candidate).get("verdict"),
            "flags": flags[:5],
            "preference_risk": {
                "hard": bool(pref.get("hard_constraint_maybe_violated")),
                "unknown": bool(pref.get("unknown_review_needed")),
                "tradeoff_hint": pref.get("tradeoff_hint"),
                "triggered": pref.get("triggered_risks", [])[:2],
            },
        }
        if geo_escape:
            compact["geo_tradeoff"] = {
                "out_of_preferred_area": True,
                "return_to_allowed_area_cost": geo_escape.get("return_to_allowed_area_cost_yuan"),
                "estimated_geo_penalty": geo_escape.get("estimated_penalty_yuan"),
                "rule": "choose only if return-adjusted action value beats staying compliant",
            }
        return compact

    def _llm_choose_action(self, driver_id: str, status: dict[str, Any], profile: dict[str, Any], candidates: list[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any] | None:
        # LLM chooses only among already compressed, tool-scored alternatives.
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        if not self._should_use_planner_llm(driver_id, current_minute, candidates):
            return None
        planning_board = self._planning_board(status, profile, candidates)
        default_limit = int(self._config["runtime_limits"].get("llm_candidate_limit", 3))
        llm_limit = max(1, min(5, int(os.environ.get("AGENT_LLM_CANDIDATE_LIMIT", str(default_limit)) or default_limit)))
        compact_candidates = [self._compact_candidate_for_llm(item) for item in candidates[:llm_limit]]
        compact_history = [
            {
                "action": (record.get("action", {}) if isinstance(record.get("action"), dict) else {}).get("action"),
                "cargo_id": (record.get("action", {}) if isinstance(record.get("action"), dict) else {}).get("params", {}).get("cargo_id"),
                "accepted": (record.get("result", {}) if isinstance(record.get("result"), dict) else {}).get("accepted"),
                "end_minute": (record.get("result", {}) if isinstance(record.get("result"), dict) else {}).get("simulation_progress_minutes"),
            }
            for record in history[-3:]
            if isinstance(record, dict)
        ]
        compact_planning_board = {
            "candidate_count": planning_board.get("candidate_count"),
            "safe_candidate_count": planning_board.get("safe_candidate_count"),
            "top_algorithm_cargo_id": planning_board.get("top_algorithm_cargo_id"),
            "top_safe_cargo_id": planning_board.get("top_safe_cargo_id"),
            "next_scheduled_visit": planning_board.get("next_scheduled_visit"),
            "preference_counts": {
                "cards": planning_board.get("preference_card_count"),
                "unknown": planning_board.get("unknown_preference_count"),
                "dynamic_rules": planning_board.get("dynamic_preference_rule_count"),
            },
        }
        payload = {
            "messages": self._prompt_templates.planner_messages(
                time_text=wall_time(int(status.get("simulation_progress_minutes", 0) or 0)),
                minute=int(status.get("simulation_progress_minutes", 0) or 0),
                status={"lat": status.get("current_lat"), "lng": status.get("current_lng"), "cost_per_km": status.get("cost_per_km")},
                profile=self._compact_planner_profile_view(profile),
                agent_config={"actions": ["take_order", "wait", "reposition"], "llm_gateway": "official_interface_only"},
                history=compact_history,
                planning_board=compact_planning_board,
                evidence_board=self._planner_evidence_board(planning_board, candidates[:llm_limit]),
                candidates=compact_candidates,
            ),
            "enable_thinking": False,
            "extra_body": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": int(os.environ.get("AGENT_PLANNER_MAX_TOKENS", "1024") or 1024),
        }
        self._driver_state(driver_id)["last_planner_llm_minute"] = current_minute
        try:
            response = self._call_model_chat_completion(payload, "planner")
            self._record_usage(response)
            content = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
            return self._parse_json_object(content)
        except Exception as exc:
            self._logger.info("llm choose fallback: %s", exc)
            return None

    def _action_audit_board(
        self,
        recommended: dict[str, Any],
        candidate: dict[str, Any] | None,
        action_guard_report: dict[str, Any],
        commitment_report: dict[str, Any],
        time_task_report: dict[str, Any],
        region_report: dict[str, Any],
        task_penalty_report: dict[str, Any],
        inactivity_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        task_penalty = self._as_float(task_penalty_report.get("estimated_action_task_penalty_yuan")) or 0.0
        task_flags = list(dict.fromkeys(task_penalty_report.get("risk_flags", []) or []))
        guard_flags = list(dict.fromkeys(action_guard_report.get("risk_flags", []) or []))
        region_flags = list(dict.fromkeys(region_report.get("risk_flags", []) or []))
        hard_reasons: list[str] = []
        caution_reasons: list[str] = []
        hard_task_penalty = float(os.environ.get("AGENT_HARD_TASK_PENALTY_REJECT_YUAN", "9000") or 9000)
        guard_penalty = self._as_float(action_guard_report.get("estimated_future_preference_penalty_yuan")) or 0.0
        if action_guard_report.get("hard_block") and guard_penalty >= hard_task_penalty:
            hard_reasons.append("action_preference_guard_tool.hard_block")
        if region_report.get("hard_block"):
            hard_reasons.append("region_preference_tool.hard_block")
        for flag in task_flags:
            if flag in {"direct_cumulative_task_penalty"} and task_penalty >= hard_task_penalty:
                hard_reasons.append(f"task_penalty_optimizer_tool.{flag}")
        if task_penalty >= hard_task_penalty:
            hard_reasons.append(f"task_penalty_yuan>={round(task_penalty, 2)}")
        if guard_flags:
            caution_reasons.append(f"action_guard_flags={guard_flags[:5]}")
        if region_flags:
            caution_reasons.append(f"region_flags={region_flags[:5]}")
        if task_flags:
            caution_reasons.append(f"task_flags={task_flags[:5]}")
        commitment_action = commitment_report.get("recommended_action") if isinstance(commitment_report, dict) else None
        recommended_name = str(recommended.get("action") or "")
        if isinstance(commitment_action, dict) and commitment_action.get("action") and commitment_action.get("action") != recommended_name:
            caution_reasons.append("recommended action differs from current commitment_sequence_tool recommendation")
        urgent_tasks = [
            {
                "id": task.get("id"),
                "type": task.get("type"),
                "deadline_minute": task.get("deadline_minute"),
                "point": task.get("point"),
            }
            for task in (time_task_report.get("urgent_tasks", []) or [])[:6]
            if isinstance(task, dict)
        ]
        overdue_tasks = [
            {
                "id": task.get("id"),
                "type": task.get("type"),
                "deadline_minute": task.get("deadline_minute"),
                "point": task.get("point"),
            }
            for task in (time_task_report.get("overdue_tasks", []) or [])[:6]
            if isinstance(task, dict)
        ]
        if overdue_tasks:
            caution_reasons.append("overdue time tasks exist")
        inactivity = inactivity_report if isinstance(inactivity_report, dict) else {}
        if inactivity.get("recommendation") == "take_positive_order_now":
            caution_reasons.append("long idle time; do not reject positive safe cargo for soft preference tradeoffs")
        verdict = "reject" if hard_reasons else ("caution" if caution_reasons or urgent_tasks or overdue_tasks else "approve")
        allowed_actions = ["wait", "reposition"]
        if verdict != "reject" and candidate is not None and recommended_name == "take_order":
            allowed_actions.append("take_order")
        elif isinstance(commitment_action, dict) and commitment_action.get("action"):
            allowed_actions.append(str(commitment_action.get("action")))
        return {
            "verdict": verdict,
            "recommended_action": recommended,
            "candidate_summary": None if candidate is None else self._candidate_decision_summary(candidate),
            "hard_reject_reasons": list(dict.fromkeys(hard_reasons))[:8],
            "caution_reasons": caution_reasons[:8],
            "allowed_actions": list(dict.fromkeys(allowed_actions)),
            "must_compare_before_approval": {
                "commitment_recommended_action": commitment_action,
                "urgent_tasks": urgent_tasks,
                "overdue_tasks": overdue_tasks,
                "task_penalty_yuan": task_penalty,
                "task_penalty_flags": task_flags,
                "inactivity_opportunity_tool": inactivity,
                "cumulative_penalty_rule": "If action overlaps a per-minute/hour required-point window, approve only continuous wait at that point or the lower-loss mandatory repair action.",
                "soft_preference_rule": "Soft home/night/anchor penalties are already deducted; they are not reject reasons when candidate action_value stays strongly positive.",
            },
            "decision_rule": "Approve only if no hard_reject_reasons and the action is the lowest total-loss positive-value option after all preference penalties.",
        }

    def _llm_review_recommended_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        profile: dict[str, Any],
        recommended: dict[str, Any],
        reason: str,
        candidates: list[dict[str, Any]] | None = None,
        force_llm: bool = False,
    ) -> dict[str, Any] | None:
        if self._disable_llm or not self._enable_llm_choice or not self._enable_llm_review:
            return None
        if str(recommended.get("reason_code") or "") in {
            "night_home_deadline_return_guard",
            "night_home_preemptive_return_guard",
            "wait_until_home_return_guard",
            "raw_night_home_preemptive_return_guard",
            "raw_wait_until_home_return_guard",
            "raw_night_home_stay_guard",
            "night_home_stay_guard",
        }:
            return None
        if not force_llm and not self._can_use_llm(driver_id):
            return None
        candidate = self._candidate_for_action(recommended, candidates or [])
        action_guard_report = self._action_guard_tool.review_action(
            status=status,
            profile=profile,
            action=recommended,
            candidate=candidate,
        )
        commitment_report = self._commitment_sequence_tool.commitment_report(
            current_minute=int(status.get("simulation_progress_minutes", 0) or 0),
            lat=float(status.get("current_lat", 0.0) or 0.0),
            lng=float(status.get("current_lng", 0.0) or 0.0),
            profile=profile,
            history=self._recent_history_records(driver_id),
        )
        time_task_report = self._time_task_progress_tool.progress_report(
            current_minute=int(status.get("simulation_progress_minutes", 0) or 0),
            lat=float(status.get("current_lat", 0.0) or 0.0),
            lng=float(status.get("current_lng", 0.0) or 0.0),
            profile=profile,
            history=self._recent_history_records(driver_id),
        )
        region_report = self._region_preference_tool.evaluate(
            current_point=(float(status.get("current_lat", 0.0) or 0.0), float(status.get("current_lng", 0.0) or 0.0)),
            candidate=candidate,
            profile=profile,
            time_task_report=time_task_report,
        )
        task_penalty_report = self._task_penalty_optimizer_tool.evaluate_action(
            current_minute=int(status.get("simulation_progress_minutes", 0) or 0),
            current_point=(float(status.get("current_lat", 0.0) or 0.0), float(status.get("current_lng", 0.0) or 0.0)),
            action=recommended,
            profile=profile,
            time_task_report=time_task_report,
            candidate=candidate,
        )
        payload = {
            "messages": self._prompt_templates.supervisor_messages(
                driver_id=driver_id,
                time_text=wall_time(int(status.get("simulation_progress_minutes", 0) or 0)),
                status={
                    "lat": status.get("current_lat"),
                    "lng": status.get("current_lng"),
                    "cost_per_km": status.get("cost_per_km"),
                },
                profile=self._planner_profile_view(profile),
                recommended=recommended,
                reason=reason,
                history=self._history_summary(driver_id),
                candidates=[self._compact_candidate_for_llm(item) for item in (candidates or [])[:8]],
                action_guard_report=action_guard_report,
                commitment_report=commitment_report,
                time_task_report=time_task_report,
                region_report=region_report,
                task_penalty_report=task_penalty_report,
                action_audit_board=self._action_audit_board(
                    recommended,
                    candidate,
                    action_guard_report,
                    commitment_report,
                    time_task_report,
                    region_report,
                    task_penalty_report,
                    self._inactivity_opportunity_report(
                        driver_id=driver_id,
                        current_minute=int(status.get("simulation_progress_minutes", 0) or 0),
                        candidates=candidates or [],
                        history_records=self._history_records(driver_id),
                    ),
                ),
            ),
            "enable_thinking": False,
            "temperature": 0,
            "max_tokens": int(os.environ.get("AGENT_SUPERVISOR_MAX_TOKENS", "1536") or 1536),
        }
        try:
            response = self._call_model_chat_completion(payload, "supervisor")
            self._record_usage(response)
            content = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
            parsed = self._parse_json_object(content)
            allowed_take_order_ids = {
                str(item.get("cargo_id"))
                for item in (candidates or [])
                if isinstance(item, dict) and item.get("cargo_id") is not None
            }
            recommended_params = recommended.get("params", {}) if isinstance(recommended.get("params"), dict) else {}
            if recommended.get("action") == "take_order" and recommended_params.get("cargo_id") is not None:
                allowed_take_order_ids.add(str(recommended_params.get("cargo_id")))
            commitment_action = commitment_report.get("recommended_action") if isinstance(commitment_report, dict) else None
            if isinstance(commitment_action, dict) and commitment_action.get("action") == "take_order":
                params = commitment_action.get("params", {}) if isinstance(commitment_action.get("params"), dict) else {}
                if params.get("cargo_id") is not None:
                    allowed_take_order_ids.add(str(params.get("cargo_id")))
            validated = self._validate_action(
                parsed,
                candidates or [],
                allowed_take_order_ids=allowed_take_order_ids,
                allow_reposition=True,
                current_minute=int(status.get("simulation_progress_minutes", 0) or 0),
            )
            if validated is None:
                return None
            if validated.get("action") == "reposition":
                if recommended.get("action") != "reposition":
                    return None
                rec_params = recommended.get("params", {}) if isinstance(recommended.get("params"), dict) else {}
                val_params = validated.get("params", {}) if isinstance(validated.get("params"), dict) else {}
                try:
                    distance = haversine_km(
                        float(rec_params.get("latitude")),
                        float(rec_params.get("longitude")),
                        float(val_params.get("latitude")),
                        float(val_params.get("longitude")),
                    )
                except (TypeError, ValueError):
                    return None
                if distance > 0.2:
                    return None
                validated = action(
                    "reposition",
                    {
                        "latitude": float(rec_params.get("latitude")),
                        "longitude": float(rec_params.get("longitude")),
                    },
                )
                for key in ("reason_code", "monthly_visit_context"):
                    if recommended.get(key) is not None:
                        validated[key] = recommended.get(key)
            elif recommended.get("reason_code") is not None and validated.get("reason_code") is None:
                validated["reason_code"] = recommended.get("reason_code")
            validated_candidate = self._candidate_for_action(validated, candidates or [])
            validated_task_penalty = self._task_penalty_optimizer_tool.evaluate_action(
                current_minute=int(status.get("simulation_progress_minutes", 0) or 0),
                current_point=(float(status.get("current_lat", 0.0) or 0.0), float(status.get("current_lng", 0.0) or 0.0)),
                action=validated,
                profile=profile,
                time_task_report=time_task_report,
                candidate=validated_candidate,
            )
            validated_flags = set(validated_task_penalty.get("risk_flags", []) or [])
            hard_task_penalty = float(os.environ.get("AGENT_HARD_TASK_PENALTY_REJECT_YUAN", "9000") or 9000)
            penalty = self._as_float(validated_task_penalty.get("estimated_action_task_penalty_yuan")) or 0.0
            if validated_flags & {"direct_cumulative_task_penalty"} and penalty >= hard_task_penalty:
                return None
            if penalty >= hard_task_penalty:
                return None
            return validated
        except Exception as exc:
            self._logger.info("llm review fallback: %s", exc)
            return None

    def _validate_action(
        self,
        chosen: dict[str, Any] | None,
        candidates: list[dict[str, Any]],
        allowed_take_order_ids: set[str] | None = None,
        allow_reposition: bool = False,
        current_minute: int | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(chosen, dict):
            return None
        name = str(chosen.get("action", "")).lower().strip()
        if name == "take_order":
            allowed = {str(x["cargo_id"]) for x in candidates}
            if allowed_take_order_ids:
                allowed.update(str(item) for item in allowed_take_order_ids)
            cargo_id = str(chosen.get("cargo_id", ""))
            if cargo_id in allowed:
                matched = next((x for x in candidates if str(x["cargo_id"]) == cargo_id), None)
                if matched is not None:
                    self._remember_cargo(matched)
                return action("take_order", {"cargo_id": cargo_id})
        if name == "wait":
            duration = max(1, min(720, self._as_int(chosen.get("duration_minutes")) or 60))
            if current_minute is not None:
                minute_of_day = current_minute % 1440
                if self._should_coalesce_idle_wait(candidates, current_minute, duration):
                    duration = self._coalesced_idle_wait_minutes(current_minute, duration)
                elif 8 * 60 <= minute_of_day < 20 * 60:
                    duration = max(30, min(duration, 60))
            return action("wait", {"duration_minutes": duration})
        if name == "reposition":
            if not allow_reposition:
                return None
            try:
                lat = float(chosen["latitude"]); lng = float(chosen["longitude"])
            except (KeyError, TypeError, ValueError):
                return None
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                return action("reposition", {"latitude": round(lat, 6), "longitude": round(lng, 6)})
        return None

    def _should_coalesce_idle_wait(self, candidates: list[dict[str, Any]], current_minute: int, duration: int) -> bool:
        if duration > 90:
            return False
        acceptable = [item for item in candidates if self._candidate_is_acceptable(item)]
        if acceptable:
            best = max(acceptable, key=self._profit_penalty_sort_key)
            if self._candidate_action_value(best) > 0:
                return False
        minute_of_day = current_minute % 1440
        return minute_of_day < 6 * 60 or minute_of_day >= 16 * 60 or not acceptable

    @staticmethod
    def _coalesced_idle_wait_minutes(current_minute: int, duration: int) -> int:
        minute_of_day = current_minute % 1440
        if minute_of_day < 6 * 60:
            target = 6 * 60 - minute_of_day
        elif minute_of_day >= 20 * 60:
            target = 24 * 60 + 6 * 60 - minute_of_day
        elif minute_of_day >= 16 * 60:
            target = max(duration, 20 * 60 - minute_of_day)
        else:
            target = max(duration, 120)
        return max(duration, min(240, target))

    def _safe_to_accept_llm_choice(self, candidate_action: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
        if candidate_action.get("action") != "take_order":
            return self._llm_wait_or_reposition_allowed(candidates)
        cargo_id = str(candidate_action.get("params", {}).get("cargo_id", ""))
        if not cargo_id:
            return False
        by_id = {str(item.get("cargo_id")): item for item in candidates}
        chosen = by_id.get(cargo_id)
        if chosen is None:
            return False
        if not self._candidate_is_acceptable(chosen):
            return False
        risk_flags = set(chosen.get("tool_report", {}).get("risk_flags", []) or [])
        guardrails = self._config.get("guardrails", {})
        reject_risks = set(guardrails.get("reject_risks", ["miss_scheduled_visit", "negative_net"]))
        reject_risks.add("cargo_time_hard_block")
        reject_risks.add("cargo_time_window_too_tight")
        reject_risks.add("cargo_listing_expires_too_soon")
        reject_risks.add("direct_cumulative_task_penalty")
        reject_risks.add("not_at_required_point_during_penalty_window")
        reject_risks.add("deadline_task_penalty")
        if risk_flags & reject_risks:
            return False
        task_penalty = chosen.get("tool_report", {}).get("task_penalty_optimizer_tool", {}) if isinstance(chosen.get("tool_report", {}).get("task_penalty_optimizer_tool"), dict) else {}
        hard_task_penalty = float(os.environ.get("AGENT_HARD_TASK_PENALTY_REJECT_YUAN", "9000") or 9000)
        if self._as_float(task_penalty.get("estimated_action_task_penalty_yuan")) and float(task_penalty.get("estimated_action_task_penalty_yuan") or 0.0) >= hard_task_penalty:
            return False
        pref_risk = chosen.get("tool_report", {}).get("preference_risk_assessment", {})
        if isinstance(pref_risk, dict) and pref_risk.get("hard_constraint_maybe_violated"):
            return False
        best = max(candidates, key=self._profit_penalty_sort_key)
        chosen_value = self._candidate_action_value(chosen)
        best_value = self._candidate_action_value(best)
        chosen_net = float(chosen.get("net_after_return", chosen.get("estimated_net", 0.0)) or 0.0)
        best_net = float(best.get("net_after_return", best.get("estimated_net", 0.0)) or 0.0)
        chosen_penalty = float(chosen.get("current_action_penalty", 0.0) or 0.0)
        best_penalty = float(best.get("current_action_penalty", 0.0) or 0.0)
        value_gap = float(os.environ.get("AGENT_LLM_ACTION_VALUE_GAP_MAX", str(guardrails.get("llm_take_order_score_gap_max", 300.0))) or 300.0)
        net_gap = float(os.environ.get("AGENT_LLM_NET_AFTER_RETURN_GAP_MAX", str(guardrails.get("llm_take_order_net_gap_max", 500.0))) or 500.0)
        if chosen_value < best_value - value_gap:
            return False
        # Allow a lower net order when it materially reduces preference penalty.
        penalty_saved = max(0.0, best_penalty - chosen_penalty)
        return chosen_net >= max(0.0, best_net - net_gap - penalty_saved)

    def _llm_wait_or_reposition_allowed(self, candidates: list[dict[str, Any]]) -> bool:
        if not candidates:
            return True
        if not self._has_acceptable_candidate(candidates):
            return self._income_rescue_candidate(candidates) is None
        acceptable = [item for item in candidates if self._candidate_is_acceptable(item)]
        if not acceptable:
            return self._income_rescue_candidate(candidates) is None
        best = max(acceptable, key=self._profit_penalty_sort_key)
        best_value = self._candidate_action_value(best)
        best_net = float(best.get("net_after_return", best.get("estimated_net", 0.0)) or 0.0)
        best_penalty = float(best.get("current_action_penalty", 0.0) or 0.0)
        if best_value > 0:
            return False
        force_take_value = float(os.environ.get("AGENT_FORCE_TAKE_MIN_ACTION_VALUE", "60") or 60)
        strong_take_value = float(os.environ.get("AGENT_STRONG_TAKE_MIN_ACTION_VALUE", "150") or 150)
        if best_value >= strong_take_value:
            return False
        penalty_ratio = best_penalty / max(1.0, abs(best_net))
        max_ratio = float(os.environ.get("AGENT_LLM_ALLOW_WAIT_PENALTY_RATIO", "0.35") or 0.35)
        min_value = float(os.environ.get("AGENT_LLM_ALLOW_WAIT_MAX_ACTION_VALUE", "0") or 0)
        if best_value >= force_take_value and penalty_ratio < 0.75:
            return False
        return penalty_ratio >= max_ratio or best_value <= min_value

    def _penalty_anomaly_report(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {}
        penalties = [float(item.get("current_action_penalty", 0.0) or 0.0) for item in candidates]
        nets = [float(item.get("net_after_return", item.get("estimated_net", 0.0)) or 0.0) for item in candidates]
        unknown_count = 0
        hard_count = 0
        dynamic_hits = 0
        for item in candidates:
            pref = item.get("tool_report", {}).get("preference_risk_assessment", {}) if isinstance(item.get("tool_report"), dict) else {}
            if not isinstance(pref, dict):
                continue
            unknown_count += 1 if pref.get("unknown_review_needed") else 0
            hard_count += 1 if pref.get("hard_constraint_maybe_violated") else 0
            dynamic_hits += len(pref.get("dynamic_rule_hits", []) if isinstance(pref.get("dynamic_rule_hits"), list) else [])
        max_penalty = max(penalties or [0.0])
        avg_penalty = sum(penalties) / max(1, len(penalties))
        best_net = max(nets or [0.0])
        reasons: list[str] = []
        if max_penalty > max(1500.0, best_net * 1.5):
            reasons.append("penalty_exceeds_profit")
        if unknown_count >= max(3, len(candidates) // 2):
            reasons.append("many_unknown_preferences")
        if hard_count >= max(3, len(candidates) // 2):
            reasons.append("many_hard_hits")
        if dynamic_hits >= max(8, len(candidates)) and avg_penalty > 500:
            reasons.append("dynamic_rule_over_triggered")
        if not reasons:
            return {}
        return {
            "profile_recheck_needed": True,
            "reasons": reasons,
            "candidate_count": len(candidates),
            "max_penalty_yuan": round(max_penalty, 2),
            "avg_penalty_yuan": round(avg_penalty, 2),
            "best_net_after_return_yuan": round(best_net, 2),
            "unknown_candidate_count": unknown_count,
            "hard_hit_candidate_count": hard_count,
            "dynamic_rule_hit_count": dynamic_hits,
        }

    @staticmethod
    def _planner_requests_profile_recheck(chosen: dict[str, Any] | None) -> bool:
        if not isinstance(chosen, dict):
            return False
        return bool(chosen.get("profile_recheck_needed") or chosen.get("tool_recheck_needed"))

    def _apply_planner_updates(self, driver_id: str, chosen: dict[str, Any] | None, profile: dict[str, Any]) -> bool:
        if not isinstance(chosen, dict):
            return False
        updates = chosen.get("dynamic_rule_updates") or chosen.get("dynamic_preference_rules")
        if not isinstance(updates, list) or not updates:
            return False
        normalized = self._normalize_dynamic_rules(updates, [])
        if not normalized:
            return False
        merged = self._merge_list(profile.get("dynamic_preference_rules", []), normalized)
        profile["dynamic_preference_rules"] = merged
        self._dynamic_rule_memory[driver_id] = merged
        self._save_dynamic_rule_memory()
        memory = self._load_driver_tool_memory(driver_id)
        memory["dynamic_preference_rules"] = self._merge_list(memory.get("dynamic_preference_rules", []), normalized)
        planner_updates = [item for item in memory.get("planner_updates", []) if isinstance(item, dict)]
        planner_updates.append({"rules_added": normalized[:8], "reason": chosen.get("reason_code") or "planner_dynamic_rule_update"})
        memory["planner_updates"] = planner_updates[-40:]
        self._save_driver_tool_memory(driver_id, memory)
        return True

    def _fallback_position_or_wait(self, status: dict[str, Any], profile: dict[str, Any], preferences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        driver_id = str(status.get("driver_id") or getattr(self, "_current_driver_id", "") or "")
        lat = float(status.get("current_lat", 0.0) or 0.0)
        lng = float(status.get("current_lng", 0.0) or 0.0)
        current_minute = int(status.get("simulation_progress_minutes", 0) or 0)
        time_task_report = self._time_task_progress_tool.progress_report(
            current_minute=current_minute,
            lat=lat,
            lng=lng,
            profile=profile,
            history=self._history_records(driver_id),
        )
        recommended = time_task_report.get("recommended_action") if isinstance(time_task_report, dict) else None
        if isinstance(recommended, dict):
            return recommended
        raw_wait = self._raw_home_bounded_wait_action(current_minute, lat, lng, preferences or [], default_minutes=60)
        if raw_wait is not None:
            return raw_wait
        released_visit_point = self._released_periodic_visit_point(time_task_report)
        points = []
        bounds = profile.get("geo_fence_bounds")
        inside_geo_scope = isinstance(bounds, dict) and self._inside_bounds(lat, lng, bounds)
        # For a pure geographic working area, any in-scope point is compliant.
        # Do not burn empty miles returning to the original anchor unless a
        # released/urgent visit task explicitly requires it.
        if not inside_geo_scope or released_visit_point:
            for p in profile.get("preference_points", []):
                if not self._valid_point(p):
                    continue
                if released_visit_point and haversine_km(float(p[0]), float(p[1]), released_visit_point[0], released_visit_point[1]) <= 1.0:
                    continue
                if not self._fallback_point_keeps_home_return_feasible(current_minute, lat, lng, p, profile):
                    continue
                points.append(p)
        if points:
            target = min(points, key=lambda p: haversine_km(lat, lng, float(p[0]), float(p[1])))
            if haversine_km(lat, lng, float(target[0]), float(target[1])) > 3.0:
                planned = action("reposition", {"latitude": float(target[0]), "longitude": float(target[1])})
                planned["reason_code"] = "fallback_preference_point_reposition"
                return planned
        return self._home_bounded_wait_action(current_minute, lat, lng, profile, default_minutes=60)

    def _fallback_point_keeps_home_return_feasible(
        self,
        current_minute: int,
        lat: float,
        lng: float,
        point: Any,
        profile: dict[str, Any],
    ) -> bool:
        if not self._valid_point(point):
            return False
        home = self._daily_home_point(profile)
        if not self._valid_point(home):
            return True
        assert isinstance(point, (list, tuple))
        assert isinstance(home, (list, tuple))
        target = (float(point[0]), float(point[1]))
        home_point = (float(home[0]), float(home[1]))
        if haversine_km(target[0], target[1], home_point[0], home_point[1]) <= 1.0:
            return True
        deadline = self._next_home_return_deadline(current_minute, profile)
        if deadline is None:
            return True
        buffer_minutes = int(os.environ.get("AGENT_NIGHT_HOME_RETURN_BUFFER_MINUTES", "30") or 30)
        travel_to_point = distance_to_minutes(haversine_km(lat, lng, target[0], target[1]))
        travel_home = distance_to_minutes(haversine_km(target[0], target[1], home_point[0], home_point[1]))
        return current_minute + travel_to_point + travel_home + buffer_minutes <= deadline

    def _raw_home_bounded_wait_action(
        self,
        current_minute: int,
        lat: float,
        lng: float,
        preferences: list[dict[str, Any]],
        *,
        default_minutes: int,
    ) -> dict[str, Any] | None:
        minute_of_day = current_minute % 1440
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if self._home_return_is_soft_text(text):
                continue
            if not any(word in text for word in ("自家位置", "到家", "回家", "进家门", "在家")):
                continue
            if not any(word in text for word in ("每天", "每日", "每晚", "每夜", "夜间", "夜里", "当天23点至次日", "当天二十三点至次日")):
                continue
            coords = extract_coordinates(text)
            if not coords:
                continue
            windows = self._extract_daily_time_windows(text)
            start = self._as_int(windows[0].get("start_minute_of_day")) if windows else None
            hard_start = start if start is not None else int(os.environ.get("AGENT_NIGHT_HOME_HARD_START_MINUTE", str(23 * 60)) or 23 * 60)
            home = coords[0]
            distance = haversine_km(lat, lng, float(home[0]), float(home[1]))
            if distance <= 1.0:
                continue
            travel = distance_to_minutes(distance)
            deadline_abs = (current_minute // 1440) * 1440 + hard_start
            if minute_of_day < int(os.environ.get("AGENT_NIGHT_HOME_END_MINUTE", str(8 * 60)) or 8 * 60):
                deadline_abs -= 1440
            leave_by = deadline_abs - travel - int(os.environ.get("AGENT_NIGHT_HOME_RETURN_BUFFER_MINUTES", "30") or 30)
            decision_buffer = int(os.environ.get("AGENT_HOME_RETURN_DECISION_BUFFER_MINUTES", "20") or 20)
            if current_minute >= leave_by - decision_buffer:
                planned = action("reposition", {"latitude": float(home[0]), "longitude": float(home[1])})
                planned["reason_code"] = "raw_night_home_preemptive_return_guard"
                planned["home_leave_by_minute"] = leave_by
                return planned
            if current_minute + int(default_minutes) > leave_by - decision_buffer:
                duration = max(1, leave_by - decision_buffer - current_minute)
                planned = action("wait", {"duration_minutes": duration})
                planned["reason_code"] = "raw_wait_until_home_return_guard"
                planned["home_leave_by_minute"] = leave_by
                return planned
        return None

    def _home_bounded_wait_action(self, current_minute: int, lat: float, lng: float, profile: dict[str, Any], *, default_minutes: int) -> dict[str, Any]:
        leave_by = self._night_home_leave_by_minute(current_minute, lat, lng, profile)
        if leave_by is None:
            return action("wait", {"duration_minutes": max(1, int(default_minutes))})
        decision_buffer = int(os.environ.get("AGENT_HOME_RETURN_DECISION_BUFFER_MINUTES", "20") or 20)
        home = self._daily_home_point(profile)
        if current_minute >= leave_by - decision_buffer and self._valid_point(home):
            planned = action("reposition", {"latitude": float(home[0]), "longitude": float(home[1])})
            planned["reason_code"] = "night_home_preemptive_return_guard"
            planned["home_leave_by_minute"] = leave_by
            return planned
        duration = min(max(1, int(default_minutes)), max(1, leave_by - decision_buffer - current_minute))
        planned = action("wait", {"duration_minutes": duration})
        planned["reason_code"] = "wait_until_home_return_guard"
        planned["home_leave_by_minute"] = leave_by
        return planned

    @staticmethod
    def _released_periodic_visit_point(time_task_report: dict[str, Any]) -> tuple[float, float] | None:
        if not isinstance(time_task_report, dict):
            return None
        for item in time_task_report.get("periodic_tasks", []) or []:
            if not isinstance(item, dict) or item.get("type") != "monthly_visit_frequency":
                continue
            if item.get("status") == "done" or item.get("release_to_earn_money") is True:
                point = item.get("point")
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        return (float(point[0]), float(point[1]))
                    except (TypeError, ValueError):
                        return None
        return None

    def _remember_cargo(self, item: dict[str, Any]) -> None:
        cargo_id = str(item.get("cargo_id", ""))
        if cargo_id:
            self._cargo_memory[f"{self._current_driver_id or 'unknown'}:{cargo_id}"] = dict(item)
            self._prune_runtime_caches(str(self._current_driver_id or ""))

    def _prune_runtime_caches(self, driver_id: str | None = None) -> None:
        cargo_limit = self._int_env("AGENT_CARGO_MEMORY_LIMIT", 512, 20, 10000)
        if len(self._cargo_memory) > cargo_limit:
            drop_count = len(self._cargo_memory) - cargo_limit
            for key in list(self._cargo_memory)[:drop_count]:
                self._cargo_memory.pop(key, None)
        failed_limit = self._int_env("AGENT_FAILED_CARGO_LIMIT_PER_DRIVER", 512, 20, 10000)
        if driver_id:
            failed = self._failed_cargos_by_driver.get(driver_id)
            if isinstance(failed, set) and len(failed) > failed_limit:
                self._failed_cargos_by_driver[driver_id] = set(list(failed)[-failed_limit:])
        driver_limit = self._int_env("AGENT_DRIVER_CACHE_LIMIT", 128, 1, 1000)
        for mapping in (self._driver_tool_memory, self._driver_runtime, self._llm_profile_cache, self._failed_cargos_by_driver):
            if len(mapping) <= driver_limit:
                continue
            protected = {driver_id} if driver_id else set()
            for key in list(mapping):
                if len(mapping) <= driver_limit:
                    break
                if key not in protected:
                    mapping.pop(key, None)

    @staticmethod
    def _candidate_for_action(candidate_action: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if candidate_action.get("action") != "take_order":
            return None
        cargo_id = str(candidate_action.get("params", {}).get("cargo_id", ""))
        return next((item for item in candidates if str(item.get("cargo_id")) == cargo_id), None)

    def _learn_from_history(self, driver_id: str) -> None:
        failed = self._failed_cargos_by_driver.setdefault(driver_id, set())
        records = self._history_records(driver_id)[-int(self._config["runtime_limits"].get("history_steps", 80)):]
        failed.update(self._memory_tool.failed_cargo_ids(records or []))
        self._update_runtime_history_stats(driver_id, records)
        self._prune_runtime_caches(driver_id)

    def _prepare_decision_history(self, driver_id: str, current_minute: int) -> None:
        step_limit = max(
            int(self._config["runtime_limits"].get("history_steps", 80)),
            self._int_env("AGENT_HISTORY_STEP_LIMIT", 240, 24, 2000),
        )
        try:
            raw = self._api.query_decision_history(driver_id=driver_id, step=step_limit)
        except Exception:
            records: list[dict[str, Any]] = []
        else:
            raw_records = raw.get("records", []) if isinstance(raw, dict) else []
            records = [record for record in raw_records if isinstance(record, dict)]
        self._update_runtime_history_stats(driver_id, records)
        filtered = self._filter_history_window(records, current_minute)
        self._decision_history_cache = {
            "driver_id": driver_id,
            "current_minute": current_minute,
            "records": filtered,
            "recent_records": records,
            "summary": self._memory_tool.compact_history(filtered, limit=24),
        }

    def _runtime_bucket(self, driver_id: str) -> dict[str, Any]:
        bucket = self._driver_runtime.setdefault(driver_id, {})
        bucket.setdefault("seen_record_keys", set())
        bucket.setdefault("active_days", set())
        bucket.setdefault("accepted_orders_by_day", {})
        bucket.setdefault("monthly_deadhead_km", 0.0)
        bucket.setdefault("seen_cargos", set())
        bucket.setdefault("cargo_meta", {})
        bucket.setdefault("required_region_days", {})
        bucket.setdefault("position_samples", [])
        return bucket

    def _runtime_seen_cargos(self, driver_id: str) -> set[str]:
        seen = self._runtime_bucket(driver_id).get("seen_cargos", set())
        return set(seen) if isinstance(seen, set) else set()

    def _update_runtime_history_stats(self, driver_id: str, records: list[dict[str, Any]]) -> None:
        if not driver_id or not records:
            return
        bucket = self._runtime_bucket(driver_id)
        seen_keys = bucket["seen_record_keys"]
        active_days = bucket["active_days"]
        accepted_by_day = bucket["accepted_orders_by_day"]
        seen_cargos = bucket["seen_cargos"]
        failed = self._failed_cargos_by_driver.setdefault(driver_id, set())
        for record in records:
            if not isinstance(record, dict):
                continue
            key = self._history_record_key(record)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            name = str(act.get("action") or "")
            params = act.get("params", {}) if isinstance(act.get("params"), dict) else {}
            cargo_id = str(params.get("cargo_id") or result.get("cargo_id") or "")
            if cargo_id:
                seen_cargos.add(cargo_id)
            accepted = name == "take_order" and bool(result.get("accepted", False))
            if name == "take_order" and not accepted and cargo_id:
                failed.add(cargo_id)
            end = int(result.get("simulation_progress_minutes", 0) or 0)
            elapsed = int(record.get("action_exec_cost_minutes", record.get("step_elapsed_minutes", 0)) or 0)
            start = max(0, end - max(1, elapsed))
            if name in {"take_order", "reposition"}:
                day = start // 1440
                last_active_minute = max(start, end - 1)
                while day <= last_active_minute // 1440:
                    active_days.add(day)
                    day += 1
            if accepted:
                day = start // 1440
                accepted_by_day[day] = int(accepted_by_day.get(day, 0) or 0) + 1
                bucket["monthly_deadhead_km"] = float(bucket.get("monthly_deadhead_km", 0.0) or 0.0) + float(result.get("pickup_deadhead_km", 0.0) or 0.0)
                cargo_meta = bucket.get("cargo_meta", {}) if isinstance(bucket.get("cargo_meta"), dict) else {}
                meta = cargo_meta.get(cargo_id, {}) if isinstance(cargo_meta.get(cargo_id), dict) else {}
                required_region = str(meta.get("required_region_match") or "")
                if required_region:
                    region_days = bucket.get("required_region_days", {})
                    if not isinstance(region_days, dict):
                        region_days = {}
                        bucket["required_region_days"] = region_days
                    days = region_days.setdefault(required_region, set())
                    if isinstance(days, set):
                        days.add(day)
            elif name == "reposition":
                bucket["monthly_deadhead_km"] = float(bucket.get("monthly_deadhead_km", 0.0) or 0.0) + float(result.get("distance_km", 0.0) or 0.0)
            samples = bucket.get("position_samples")
            if isinstance(samples, list):
                after = record.get("position_after", {}) if isinstance(record.get("position_after"), dict) else {}
                try:
                    sample = {
                        "minute": end,
                        "day": end // 1440,
                        "lat": float(after.get("lat")),
                        "lng": float(after.get("lng")),
                    }
                except (TypeError, ValueError):
                    sample = None
                if sample is not None:
                    samples.append(sample)
                    if len(samples) > 5000:
                        del samples[: len(samples) - 5000]

    @staticmethod
    def _history_record_key(record: dict[str, Any]) -> str:
        act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        return "|".join(
            [
                str(record.get("step") or ""),
                str(result.get("simulation_progress_minutes") or ""),
                str(act.get("action") or ""),
                json.dumps(act.get("params", {}), ensure_ascii=False, sort_keys=True),
            ]
        )

    def _filter_history_window(self, records: list[dict[str, Any]], current_minute: int) -> list[dict[str, Any]]:
        lookback_days = self._int_env("AGENT_HISTORY_LOOKBACK_DAYS", 0, 0, HORIZON_DAYS)
        window_start = max(0, (current_minute // 1440 - lookback_days) * 1440)
        return [record for record in records if self._record_touches_history_window(record, window_start)]

    @staticmethod
    def _record_touches_history_window(record: dict[str, Any], window_start: int) -> bool:
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        end = int(result.get("simulation_progress_minutes", 0) or 0)
        elapsed = int(record.get("action_exec_cost_minutes", record.get("step_elapsed_minutes", 0)) or 0)
        start = max(0, end - max(0, elapsed))
        return end >= window_start or start >= window_start

    def _history_records(self, driver_id: str) -> list[dict[str, Any]]:
        if not driver_id:
            return []
        cache = self._decision_history_cache
        if cache.get("driver_id") == driver_id and isinstance(cache.get("records"), list):
            return list(cache["records"])
        return []

    def _recent_history_records(self, driver_id: str) -> list[dict[str, Any]]:
        if not driver_id:
            return []
        cache = self._decision_history_cache
        if cache.get("driver_id") == driver_id and isinstance(cache.get("recent_records"), list):
            return list(cache["recent_records"])
        return self._history_records(driver_id)

    def _history_summary(self, driver_id: str) -> list[dict[str, Any]]:
        cache = self._decision_history_cache
        if cache.get("driver_id") == driver_id and isinstance(cache.get("summary"), list):
            return list(cache["summary"])
        return self._memory_tool.compact_history(self._history_records(driver_id), limit=24)

    def _active_days(self, driver_id: str) -> set[int]:
        active: set[int] = set(self._runtime_bucket(driver_id).get("active_days", set()))
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") not in {"take_order", "reposition"}:
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            end = int(result.get("simulation_progress_minutes", 0) or 0)
            elapsed = int(record.get("step_elapsed_minutes", 0) or 0)
            start = max(0, end - max(1, elapsed))
            day = start // 1440
            last_active_minute = max(start, end - 1)
            while day <= last_active_minute // 1440:
                active.add(day); day += 1
        return active

    def _accepted_orders_on_day(self, driver_id: str, day: int) -> int:
        count = int(self._runtime_bucket(driver_id).get("accepted_orders_by_day", {}).get(day, 0) or 0)
        if count:
            return count
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") != "take_order":
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            if not bool(result.get("accepted", False)):
                continue
            end = int(result.get("simulation_progress_minutes", 0) or 0)
            elapsed = int(record.get("step_elapsed_minutes", 0) or 0)
            start = max(0, end - max(1, elapsed))
            if start // 1440 == day:
                count += 1
        return count

    def _monthly_deadhead_km(self, driver_id: str) -> float:
        total = float(self._runtime_bucket(driver_id).get("monthly_deadhead_km", 0.0) or 0.0)
        if total > 0:
            return total
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            name = str(act.get("action", ""))
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            if name == "reposition":
                total += float(result.get("distance_km", 0.0) or 0.0)
            elif name == "take_order" and bool(result.get("accepted", False)):
                total += float(result.get("pickup_deadhead_km", 0.0) or 0.0)
        return total

    def _waited_near_on_day(self, driver_id: str, day: int, point: tuple[float, float], min_minutes: int) -> bool:
        waited = 0
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") != "wait":
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            end = int(result.get("simulation_progress_minutes", 0) or 0)
            elapsed = int(record.get("action_exec_cost_minutes", record.get("step_elapsed_minutes", 0)) or 0)
            start = max(0, end - elapsed)
            if start // 1440 != day and end // 1440 != day:
                continue
            pos = record.get("position_after", {})
            if isinstance(pos, dict) and haversine_km(float(pos.get("lat", 0.0)), float(pos.get("lng", 0.0)), point[0], point[1]) <= 2.0:
                waited += elapsed
        return waited >= min_minutes

    def _visited_days_near(self, driver_id: str, point: tuple[float, float], radius_km: float) -> set[int]:
        visited: set[int] = set()
        samples = self._runtime_bucket(driver_id).get("position_samples", [])
        if isinstance(samples, list):
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                try:
                    if haversine_km(float(sample.get("lat")), float(sample.get("lng")), point[0], point[1]) <= radius_km:
                        day = self._as_int(sample.get("day"))
                        if day is not None:
                            visited.add(day)
                except (TypeError, ValueError):
                    continue
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            before = record.get("position_before", {}) if isinstance(record.get("position_before"), dict) else {}
            after = record.get("position_after", {}) if isinstance(record.get("position_after"), dict) else {}
            try:
                start = (float(before.get("lat")), float(before.get("lng")))
                end = (float(after.get("lat")), float(after.get("lng")))
                passed = (
                    haversine_km(start[0], start[1], point[0], point[1]) <= radius_km
                    or haversine_km(end[0], end[1], point[0], point[1]) <= radius_km
                    or point_to_segment_km(point, start, end) <= radius_km
                )
            except (TypeError, ValueError):
                passed = False
            if passed:
                minute = int(result.get("simulation_progress_minutes", 0) or 0)
                visited.add(minute // 1440)
        return visited

    def _monthly_visit_target_by_day(self, required: int, current_day: int) -> int:
        if required <= 0:
            return 0
        # Keep monthly progress ahead of the final crunch without requiring an
        # exact fixed calendar. The thresholds scale with required visit count.
        month_progress = max(0.0, min(1.0, (current_day + 1) / max(1, HORIZON_DAYS)))
        return min(required, max(1, math.ceil(required * month_progress)))

    def _candidate_blocks_visit_progress(self, driver_id: str, finish_minute: int, end_lat: float, end_lng: float, profile: dict[str, Any]) -> bool:
        vf = profile.get("visit_frequency", {}) if isinstance(profile.get("visit_frequency"), dict) else {}
        required = max(0, self._as_int(vf.get("required_days")) or 0)
        point = vf.get("point")
        if required <= 0 or not self._valid_point(point):
            return False
        target = (float(point[0]), float(point[1]))
        radius_km = float(vf.get("radius_km") or 1.0)
        visited = self._visited_days_near(driver_id, target, radius_km)
        if len(visited) >= required:
            return False
        finish_day = min(max(0, HORIZON_DAYS - 1), max(0, finish_minute // 1440))
        projected = len(visited)
        if haversine_km(end_lat, end_lng, target[0], target[1]) <= radius_km:
            projected += 0 if finish_day in visited else 1
        target_by_finish = self._monthly_visit_target_by_day(required, finish_day)
        if projected >= target_by_finish:
            return False
        remaining_days = HORIZON_DAYS - finish_day
        missing_after = required - projected
        return remaining_days <= missing_after + 3

    def _candidate_blocks_daily_rest(self, driver_id: str, current_minute: int, finish_minute: int, profile: dict[str, Any]) -> bool:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        hours = self._as_float(rest.get("hours"))
        if hours is None or hours <= 0:
            return False
        required = int(math.ceil(hours * 60))
        if required <= 0:
            return False
        start_day = max(0, current_minute // 1440)
        end_day = min(max(0, HORIZON_DAYS - 1), max(start_day, finish_minute // 1440))
        for day in range(start_day, end_day + 1):
            day_start = day * 1440
            day_end = day_start + 1440
            if current_minute >= day_end:
                continue
            completed = self._max_continuous_wait_minutes_for_day(driver_id, day)
            if completed >= required:
                continue
            latest_start = day_end - required
            if finish_minute > latest_start and current_minute < day_end:
                return True
        return False

    def _max_continuous_wait_minutes_for_day(self, driver_id: str, day: int) -> int:
        day_start = day * 1440
        day_end = day_start + 1440
        waits: list[tuple[int, int]] = []
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            action_obj = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if action_obj.get("action") != "wait":
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            end = self._as_int(record.get("simulation_end_minute"))
            if end is None:
                end = self._as_int(result.get("simulation_progress_minutes"))
            elapsed = self._as_int(record.get("action_exec_cost_minutes")) or self._as_int(record.get("step_elapsed_minutes")) or 0
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
        best = 0
        cur_start, cur_end = waits[0]
        for start, end in waits[1:]:
            if start <= cur_end + 1:
                cur_end = max(cur_end, end)
            else:
                best = max(best, cur_end - cur_start)
                cur_start, cur_end = start, end
        return max(best, cur_end - cur_start)

    def _history_has_cargo(self, driver_id: str, cargo_id: str) -> bool:
        if not cargo_id:
            return False
        if cargo_id in self._runtime_seen_cargos(driver_id) or cargo_id in self._failed_cargos_by_driver.setdefault(driver_id, set()):
            return True
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            if act.get("action") == "take_order" and str((act.get("params") or {}).get("cargo_id", "")) == cargo_id and result.get("accepted") is not False:
                return True
        return False

    def _geo_candidate_allowed(
        self,
        current_lat: float,
        current_lng: float,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        profile: dict[str, Any],
        start_minute: int | None = None,
        end_minute: int | None = None,
    ) -> bool:
        if not self._point_allowed_by_profile(start_lat, start_lng, profile):
            return False
        if not self._point_allowed_by_profile(end_lat, end_lng, profile):
            return False
        if start_minute is not None:
            finish = end_minute if end_minute is not None else start_minute
            if self._point_blocked_by_avoid_region(start_lat, start_lng, start_minute, finish, profile):
                return False
            if self._point_blocked_by_avoid_region(end_lat, end_lng, start_minute, finish, profile):
                return False
        return self._geo_segment_allowed(current_lat, current_lng, start_lat, start_lng, profile) and self._geo_segment_allowed(start_lat, start_lng, end_lat, end_lng, profile)

    def _geo_soft_penalty_yuan(self, profile: dict[str, Any], immediate_net: float) -> float | None:
        # Explicit no-entry circles are treated as hard safety constraints.
        if profile.get("forbidden_circles"):
            return None
        if not isinstance(profile.get("geo_fence_bounds"), dict):
            return None
        if os.environ.get("AGENT_ALLOW_PROFITABLE_GEOFENCE_SOFT_ESCAPE", "1").strip().lower() not in {"1", "true", "yes"}:
            return None
        min_net = float(os.environ.get("AGENT_GEOFENCE_SOFT_ESCAPE_MIN_NET_YUAN", "2200") or 2200)
        if immediate_net < min_net:
            return None
        base_penalty = float(os.environ.get("AGENT_GEOFENCE_SOFT_ESCAPE_PENALTY_YUAN", "600") or 600)
        ratio_penalty = immediate_net * float(os.environ.get("AGENT_GEOFENCE_SOFT_ESCAPE_PENALTY_RATIO", "0.35") or 0.35)
        return min(max(base_penalty, ratio_penalty), float(os.environ.get("AGENT_GEOFENCE_SOFT_ESCAPE_MAX_PENALTY_YUAN", "2600") or 2600))

    def _geo_return_cost_yuan(self, lat: float, lng: float, profile: dict[str, Any], cost_per_km: float) -> float:
        bounds = profile.get("geo_fence_bounds")
        if not isinstance(bounds, dict):
            return 0.0
        try:
            lat_min = float(bounds["lat_min"])
            lat_max = float(bounds["lat_max"])
            lng_min = float(bounds["lng_min"])
            lng_max = float(bounds["lng_max"])
        except (KeyError, TypeError, ValueError):
            return 0.0
        if lat_min <= lat <= lat_max and lng_min <= lng <= lng_max:
            return 0.0
        nearest_lat = max(lat_min, min(lat_max, lat))
        nearest_lng = max(lng_min, min(lng_max, lng))
        return haversine_km(lat, lng, nearest_lat, nearest_lng) * float(cost_per_km)

    def _point_blocked_by_avoid_region(self, lat: float, lng: float, start_minute: int, end_minute: int, profile: dict[str, Any]) -> bool:
        for item in profile.get("avoid_regions", []):
            if not isinstance(item, dict):
                continue
            region = str(item.get("region") or "")
            if not region:
                continue
            hint = self._region_center_hint(region, profile)
            if hint is None:
                continue
            center_lat, center_lng, radius_km = hint
            if haversine_km(lat, lng, center_lat, center_lng) > radius_km:
                continue
            days = item.get("days")
            if not isinstance(days, list) or not days:
                return True
            for day in days:
                d = self._as_int(day)
                if d is not None and start_minute < (d + 1) * 1440 and end_minute > d * 1440:
                    return True
        return False

    def _remember_region_point(self, profile: dict[str, Any], city: str, lat: float, lng: float) -> None:
        city = str(city or "").strip()
        if len(city) < 2:
            return
        hints = profile.setdefault("_runtime_region_hints", {})
        if not isinstance(hints, dict):
            return
        keys = {city}
        if len(city) > 2 and city.endswith(("市", "县", "区")):
            keys.add(city[:-1])
        for key in keys:
            entry = hints.get(key)
            if not isinstance(entry, dict):
                hints[key] = {"lat": lat, "lng": lng, "count": 1, "radius_km": 35.0}
                continue
            count = max(1, int(entry.get("count", 1) or 1))
            old_lat = float(entry.get("lat", lat) or lat)
            old_lng = float(entry.get("lng", lng) or lng)
            new_count = min(count + 1, 50)
            new_lat = (old_lat * count + lat) / (count + 1)
            new_lng = (old_lng * count + lng) / (count + 1)
            spread = haversine_km(old_lat, old_lng, lat, lng) + 20.0
            entry.update({
                "lat": new_lat,
                "lng": new_lng,
                "count": new_count,
                "radius_km": min(90.0, max(float(entry.get("radius_km", 35.0) or 35.0), spread, 35.0)),
            })

    @staticmethod
    def _region_center_hint(region: str, profile: dict[str, Any]) -> tuple[float, float, float] | None:
        hints = profile.get("_runtime_region_hints")
        if not isinstance(hints, dict):
            return None
        for key, value in hints.items():
            if not isinstance(value, dict):
                continue
            key_text = str(key)
            if key_text and (key_text in region or region in key_text):
                try:
                    return (
                        float(value["lat"]),
                        float(value["lng"]),
                        float(value.get("radius_km", 55.0) or 55.0),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return None

    def _geo_segment_allowed(self, from_lat: float, from_lng: float, to_lat: float, to_lng: float, profile: dict[str, Any]) -> bool:
        if not isinstance(profile.get("geo_fence_bounds"), dict) and not profile.get("forbidden_circles"):
            return True
        steps = max(4, min(24, int(math.ceil(haversine_km(from_lat, from_lng, to_lat, to_lng) / 40.0))))
        for index in range(steps + 1):
            ratio = index / steps
            lat = from_lat + (to_lat - from_lat) * ratio
            lng = from_lng + (to_lng - from_lng) * ratio
            if not self._point_allowed_by_profile(lat, lng, profile):
                return False
        return True

    def _point_allowed_by_profile(self, lat: float, lng: float, profile: dict[str, Any]) -> bool:
        bounds = profile.get("geo_fence_bounds")
        if isinstance(bounds, dict) and not self._inside_bounds(lat, lng, bounds):
            return False
        for circle in profile.get("forbidden_circles", []):
            if self._inside_forbidden_circle(lat, lng, circle):
                return False
        return True

    def _inside_bounds(self, lat: float, lng: float, bounds: dict[str, Any]) -> bool:
        try:
            return float(bounds["lat_min"]) <= lat <= float(bounds["lat_max"]) and float(bounds["lng_min"]) <= lng <= float(bounds["lng_max"])
        except (KeyError, TypeError, ValueError):
            return True

    def _inside_forbidden_circle(self, lat: float, lng: float, circle: Any) -> bool:
        if not isinstance(circle, dict):
            return False
        center = circle.get("center")
        radius = self._as_float(circle.get("radius_km"))
        if radius is None or not self._valid_point(center):
            return False
        return haversine_km(lat, lng, float(center[0]), float(center[1])) <= radius

    def _log_stable_reposition_point(self, lat: float, lng: float) -> tuple[float, float]:
        return (round(float(lat), 2), round(float(lng), 2))

    def _forbidden_circle_escape_action(self, lat: float, lng: float, profile: dict[str, Any]) -> dict[str, Any] | None:
        for circle in profile.get("forbidden_circles", []) or []:
            if not isinstance(circle, dict) or not self._inside_forbidden_circle(lat, lng, circle):
                continue
            center = circle.get("center")
            radius = self._as_float(circle.get("radius_km"))
            if radius is None or not self._valid_point(center):
                continue
            assert isinstance(center, (list, tuple))
            center_lat = float(center[0])
            center_lng = float(center[1])
            escape_margin = float(os.environ.get("AGENT_FORBIDDEN_CIRCLE_ESCAPE_MARGIN_KM", "5") or 5)
            target_distance = float(radius) + max(1.0, escape_margin)
            current_distance = haversine_km(center_lat, center_lng, lat, lng)
            if current_distance > 0.2:
                lat_scale = max(0.2, math.cos(math.radians(center_lat)))
                unit_lat = (lat - center_lat) * 111.0 / current_distance
                unit_lng = (lng - center_lng) * 111.0 * lat_scale / current_distance
                directions = [(unit_lat, unit_lng)]
            else:
                directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
            for dlat, dlng in directions:
                target_lat = center_lat + dlat * target_distance / 111.0
                lat_scale = max(0.2, math.cos(math.radians(center_lat)))
                target_lng = center_lng + dlng * target_distance / (111.0 * lat_scale)
                target_lat, target_lng = self._log_stable_reposition_point(target_lat, target_lng)
                if self._point_allowed_by_profile(target_lat, target_lng, profile):
                    planned = action("reposition", {"latitude": target_lat, "longitude": target_lng})
                    planned["reason_code"] = "forbidden_circle_escape"
                    return planned
        return None

    def _raw_forbidden_circle_escape_action(self, lat: float, lng: float, preferences: list[dict[str, Any]]) -> dict[str, Any] | None:
        for item in preferences:
            text = str(item.get("content") or "") if isinstance(item, dict) else ""
            if not any(word in text for word in ("不得进入", "禁止进入", "禁入", "别进")):
                continue
            for match in re.finditer(r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)].{0,24}?半径\s*([0-9]+(?:\.[0-9]+)?)\s*公里", text):
                center_lat = float(match.group(1))
                center_lng = float(match.group(2))
                radius = float(match.group(3))
                distance = haversine_km(lat, lng, center_lat, center_lng)
                if distance > radius:
                    continue
                margin = float(os.environ.get("AGENT_FORBIDDEN_CIRCLE_ESCAPE_MARGIN_KM", "5") or 5)
                target_distance = radius + max(1.0, margin)
                lat_scale = max(0.2, math.cos(math.radians(center_lat)))
                if distance > 0.2:
                    unit_lat = (lat - center_lat) * 111.0 / distance
                    unit_lng = (lng - center_lng) * 111.0 * lat_scale / distance
                else:
                    unit_lat, unit_lng = 1.0, 0.0
                for extra_km in (0.0, 2.0, 5.0):
                    stable_distance = target_distance + extra_km
                    target_lat = center_lat + unit_lat * stable_distance / 111.0
                    target_lng = center_lng + unit_lng * stable_distance / (111.0 * lat_scale)
                    target_lat, target_lng = self._log_stable_reposition_point(target_lat, target_lng)
                    if haversine_km(target_lat, target_lng, center_lat, center_lng) <= radius + 0.5:
                        continue
                    planned = action("reposition", {"latitude": target_lat, "longitude": target_lng})
                    planned["reason_code"] = "raw_forbidden_circle_escape"
                    return planned
        return None

    def _violates_avoid_region(self, start_city: str, end_city: str, start_minute: int, end_minute: int, profile: dict[str, Any]) -> bool:
        for item in profile.get("avoid_regions", []):
            if not isinstance(item, dict):
                continue
            region = str(item.get("region") or "")
            if not region or (region not in start_city and region not in end_city):
                continue
            days = item.get("days")
            if not isinstance(days, list) or not days:
                return True
            for day in days:
                d = self._as_int(day)
                if d is not None and start_minute < (d + 1) * 1440 and end_minute > d * 1440:
                    return True
        return False

    def _violates_daily_rest(self, start_minute: int, finish_minute: int, profile: dict[str, Any]) -> bool:
        rest = profile.get("daily_rest", {}) if isinstance(profile.get("daily_rest"), dict) else {}
        hours = self._as_float(rest.get("hours"))
        window_start = self._as_int(rest.get("window_start_minute"))
        window_end = self._as_int(rest.get("window_end_minute"))
        if window_start is not None and window_end is not None:
            day = start_minute // 1440
            while day <= finish_minute // 1440:
                ws = day * 1440 + window_start
                we = day * 1440 + window_end
                if window_start >= window_end:
                    we += 1440
                if start_minute < we and finish_minute > ws:
                    return True
                day += 1
        if hours is not None and hours >= 7.5:
            day = start_minute // 1440
            while day <= finish_minute // 1440:
                protected_start = day * 1440 + 16 * 60
                protected_end = (day + 1) * 1440
                if start_minute < protected_end and finish_minute > protected_start:
                    return True
                day += 1
        return False

    def _blocks_scheduled_visit(self, end_lat: float, end_lng: float, finish: int, profile: dict[str, Any]) -> bool:
        for item in profile.get("scheduled_visits", []):
            if not isinstance(item, dict) or not self._valid_point(item.get("point")):
                continue
            day = self._as_int(item.get("day")); deadline = self._as_int(item.get("arrive_before_minute"))
            if day is None or finish >= (day + 1) * 1440:
                continue
            target_minute = day * 1440 + (deadline if deadline is not None else 20 * 60)
            p = item["point"]
            travel = distance_to_minutes(haversine_km(end_lat, end_lng, float(p[0]), float(p[1])))
            if finish + travel + 30 > target_minute:
                return True
        return False

    def _nearest_profile_point_km(self, lat: float, lng: float, profile: dict[str, Any]) -> float | None:
        points = [p for p in profile.get("preference_points", []) if self._valid_point(p)]
        required = profile.get("required_region_cargo_days", {}) if isinstance(profile.get("required_region_cargo_days"), dict) else {}
        if self._valid_point(required.get("point")):
            points.append(required["point"])
        if not points:
            return None
        return min(haversine_km(lat, lng, float(p[0]), float(p[1])) for p in points)

    @staticmethod
    def _inside_daily_window(minute_of_day: int, start: int, end: int) -> bool:
        if start <= end:
            return start <= minute_of_day < end
        return minute_of_day >= start or minute_of_day < end

    @staticmethod
    def _daily_window_overlaps_interval(start_minute: int, end_minute: int, window_start: int, window_end: int) -> bool:
        day = max(0, start_minute // 1440)
        while day <= end_minute // 1440:
            ws = day * 1440 + window_start
            we = day * 1440 + window_end
            if window_start >= window_end:
                we += 1440
            if start_minute < we and end_minute > ws:
                return True
            day += 1
        return False

    def _periodic_rule_candidate_risky(self, driver_id: str, current_minute: int, finish_minute: int, periodic: dict[str, Any]) -> bool:
        period_days = max(1, self._as_int(periodic.get("period_days")) or 7)
        min_wait = max(30, self._as_int(periodic.get("min_wait_minutes")) or 120)
        period_minutes = period_days * 1440
        period_start = (current_minute // period_minutes) * period_minutes
        period_end = min(MONTH_HORIZON_MINUTES, period_start + period_minutes)
        if self._has_continuous_wait_in_period(driver_id, period_start, period_end, min_wait):
            return False
        pressure_start = max(period_start, period_end - max(6 * 60, min_wait))
        return finish_minute >= pressure_start or current_minute >= pressure_start

    def _waited_minutes_in_period(self, driver_id: str, period_start: int, period_end: int) -> int:
        waited = 0
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") != "wait":
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            end = int(result.get("simulation_progress_minutes", 0) or 0)
            elapsed = int(record.get("action_exec_cost_minutes", record.get("step_elapsed_minutes", 0)) or 0)
            start = max(0, end - elapsed)
            overlap = max(0, min(end, period_end) - max(start, period_start))
            waited += overlap
        return waited

    def _has_continuous_wait_in_period(self, driver_id: str, period_start: int, period_end: int, min_wait: int) -> bool:
        for record in self._history_records(driver_id):
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            if act.get("action") != "wait":
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            end = int(result.get("simulation_progress_minutes", 0) or 0)
            elapsed = int(record.get("action_exec_cost_minutes", record.get("step_elapsed_minutes", 0)) or 0)
            start = max(0, end - elapsed)
            if min(end, period_end) - max(start, period_start) >= min_wait:
                return True
        return False

    @staticmethod
    def _minutes_until_window_end(minute_of_day: int, end: int) -> int:
        return max(1, (end - minute_of_day) % 1440 or 60)

    @staticmethod
    def _valid_point(point: Any) -> bool:
        if not (isinstance(point, (list, tuple)) and len(point) == 2 and all(isinstance(x, (int, float)) for x in point)):
            return False
        lat = float(point[0])
        lng = float(point[1])
        return 3.0 <= lat <= 54.0 and 73.0 <= lng <= 136.0

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        for candidate in ModelDecisionService._json_object_candidates(text):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        return None

    @staticmethod
    def _json_object_candidates(text: str) -> list[str]:
        candidates: list[str] = [text]
        start = text.find("{")
        while start >= 0:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : index + 1])
                        break
            start = text.find("{", start + 1)
        return candidates
