"""Driver preference analysis tool used by the LLM planner.

This module is the online-safe version of the earlier driver_profile_analyzer.py:
it never reads drivers.json. It only transforms the visible preferences returned by
get_driver_status into a structured profile and a compact LLM context.
"""

from __future__ import annotations

import calendar
import os
import re
from datetime import datetime
from typing import Any

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


BASE_TIME = _parse_base_time()
HORIZON_DAYS = int(os.environ.get("AGENT_HORIZON_DAYS") or calendar.monthrange(BASE_TIME.year, BASE_TIME.month)[1])

NEGATIVE_CARGO_MARKERS = (
    "不接",
    "不要接",
    "不拉",
    "避免",
    "推掉",
    "干不了",
    "不想拉",
    "别派",
    "不能碰",
    "拒绝",
    "绕开",
)
REGION_NEGATIVE_MARKERS = (
    "一律不接",
    "不接",
    "不往",
    "不去",
    "避开",
    "别去",
    "不要去",
    "禁止进入",
    "不得进入",
    "禁入",
    "绕开",
    "远离",
)
REGION_HINT_SUFFIXES = ("市", "区", "县", "镇", "城", "港", "湾", "岛", "山", "园", "场", "口")
CARGO_KEYWORD_LEXICON: dict[str, tuple[str, ...]] = {
    "机械设备": ("机械设备", "机器", "设备", "机床", "机械"),
    "蔬菜": ("蔬菜", "菜", "青菜", "果蔬"),
    "食品饮料": ("食品饮料", "食品", "饮料", "酒水", "粮油"),
    "服饰纺织皮革": ("服饰纺织皮革", "服饰", "纺织", "皮革", "布料", "衣服"),
    "快递快运搬家": ("快递快运搬家", "快递", "快运", "搬家", "包裹"),
    "冷链": ("冷链", "冷库", "冻品", "冷藏", "冷冻"),
    "生鲜": ("生鲜", "水果", "海鲜", "水产", "活鱼", "鲜活"),
    "化工": ("化工", "油漆", "胶水", "香精", "香料", "化工桶"),
    "危险品": ("危险品", "危化", "易燃", "易爆", "电池"),
    "易碎": ("易碎", "玻璃", "陶瓷", "镜面玻璃"),
    "家具建材": ("家具", "古董家具", "建材", "板材", "瓷砖"),
    "金属钢材": ("钢材", "金属", "型材", "管材"),
    "电子精密": ("电子料", "芯片", "主板", "精密仪器", "服务器"),
    "人工装卸": ("人工搬运", "搬运", "上楼", "背货", "散件", "裸货"),
}
TOKEN_STOPWORDS = {
    "货",
    "货源",
    "订单",
    "活儿",
    "路线",
    "地区",
    "装货",
    "卸货",
    "接单",
    "空驶",
    "空车",
    "不空驶",
    "不空车",
    "午休",
    "吃饭",
    "这种",
    "这类",
    "一律",
}


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


def day_indices_from_text(text: str) -> list[int]:
    days: set[int] = set()
    compact = re.sub(r"\s+", "", text)
    adjacent_pattern = (
        r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月"
        r"([0-9一二两三四五六七八九十]{1,3})[号日]"
        r"([0-9一二两三四五六七八九十]{1,3})[号日]"
    )
    for year_text, month_text, start_text, end_text in re.findall(adjacent_pattern, compact):
        month = _parse_cn_number(month_text)
        start_day = _parse_cn_number(start_text)
        end_day = _parse_cn_number(end_text)
        if month is None or start_day is None or end_day is None:
            continue
        year = int(year_text) if year_text else BASE_TIME.year
        try:
            start = (datetime(year, month, start_day) - BASE_TIME).days
            end = (datetime(year, month, end_day) - BASE_TIME).days
        except ValueError:
            continue
        days.update(range(min(start, end), max(start, end) + 1))
    date_pattern = r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月([0-9一二两三四五六七八九十]{1,3})\s*[号日]"
    for year_text, month_text, day_text in re.findall(date_pattern, text):
        month = _parse_cn_number(month_text)
        day = _parse_cn_number(day_text)
        if month is None or day is None:
            continue
        year = int(year_text) if year_text else BASE_TIME.year
        try:
            days.add((datetime(year, month, day) - BASE_TIME).days)
        except ValueError:
            continue
    return sorted(day for day in days if 0 <= day < HORIZON_DAYS)


def datetime_mentions(text: str) -> list[str]:
    return re.findall(
        r"(?:\d{4}-[0-9]{1,2}-[0-9]{1,2}\s+[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?|"
        r"(?:\d{4}年)?[0-9一二两三四五六七八九十]{1,3}月[0-9一二两三四五六七八九十]{1,3}日\s*"
        r"[0-9一二两三四五六七八九十]{1,3}(?:点|:|：)[0-9]{0,2})",
        text,
    )


def extract_coordinates(text: str) -> list[tuple[float, float]]:
    return [
        (float(lat), float(lng))
        for lat, lng in re.findall(r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)]", text)
    ]


def _flatten_lexicon(lexicon: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    words: list[str] = []
    seen: set[str] = set()
    for key, aliases in lexicon.items():
        for word in (key, *aliases):
            if word and word not in seen:
                seen.add(word)
                words.append(word)
    return tuple(words)


class DriverProfileTool:
    """Builds a planning profile and an LLM-readable tool report from preferences.

    This parser is intentionally conservative. It should extract obvious
    structured constraints, but it must not hallucinate tasks or hard rules from
    vague text. The LLM can supplement the profile later, but this deterministic
    profile remains the safety floor used by the action guards.
    """

    PROFILE_SCHEMA: dict[str, Any] = {
        "avoid_cargo_keywords": ["cargo category/name keywords to avoid"],
        "avoid_regions": [{"region": "city/district keyword", "days": "null or list of 0-based days"}],
        "daily_rest": {"hours": "number or null", "window_start_minute": "minute-of-day or null", "window_end_minute": "minute-of-day or null"},
        "required_off_days": "integer, full natural days with no take_order/reposition",
        "pickup_deadhead_max_km": "number or null",
        "monthly_deadhead_limit_km": "number or null, total monthly empty driving limit",
        "max_haul_km": "number or null, pickup-to-delivery distance limit",
        "first_order_deadline_minute": "minute-of-day or null, if a day has orders then first accepted order must start before this",
        "daily_order_limit": "integer or null, max accepted orders per natural day",
        "geo_fence_bounds": {"lat_min": "number", "lat_max": "number", "lng_min": "number", "lng_max": "number"},
        "forbidden_circles": [{"center": "[lat,lng]", "radius_km": "number"}],
        "required_cargos": [{"cargo_id": "string", "pickup_point": "[lat,lng] or null", "online_minute": "minute offset or null"}],
        "temporary_events": [{"pickup_point": "[lat,lng]", "home_point": "[lat,lng]", "pickup_minute": "minute offset", "release_minute": "minute offset"}],
        "long_sequence_commitments": [
            {
                "id": "SEQ001",
                "source_text": "original long preference text",
                "commitment_type": "ordered_long_event|wedding_service|family_event|multi_day_service",
                "buffer_minutes": 60,
                "steps": [
                    {
                        "id": "step id",
                        "label": "human-readable phase",
                        "step_type": "visit|visit_and_wait|stay_until|take_cargo",
                        "point": "[lat,lng] when location-bound",
                        "cargo_id": "string when cargo-bound",
                        "earliest_minute": "minute offset or null",
                        "deadline_minute": "minute offset or null",
                        "wait_minutes": "required continuous wait minutes",
                        "hold_until_minute": "minute offset for stay_until",
                        "radius_km": "completion radius",
                    }
                ],
            }
        ],
        "cumulative_time_penalty_rules": [
            {
                "id": "TIMEPEN001",
                "source_text": "original text",
                "rate_yuan_per_minute": "number",
                "window_start_minute": "minute offset or null",
                "window_end_minute": "minute offset or null",
                "required_point": "[lat,lng] or null",
                "radius_km": "number",
                "trigger": "not_at_required_point|late|active_during_window|unknown",
            }
        ],
        "visit_frequency": {"required_days": "integer", "point": "[lat,lng] or null", "radius_km": "number"},
        "required_region_cargo_days": {"region": "keyword or null", "min_days": "integer", "point": "[lat,lng] or null"},
        "scheduled_visits": [{"day": "0-based day", "point": "[lat,lng]", "wait_minutes": 0, "arrive_before_minute": "minute-of-day or null"}],
        "preference_points": ["real [lat,lng] from preference text only"],
        "preference_cards": [{"id": "P001", "content": "original text", "types": ["constraint type"], "risk_key": "normalized duplicate-risk key", "severity": "critical|high|medium|low", "tradeoff_mode": "hard|soft|unknown", "penalty_amount": "number", "penalty_cap": "number|null"}],
        "duplicate_preference_groups": [{"risk_key": "same risk point", "preference_ids": ["P001"], "count": 2, "stacked_penalty_amount": "number"}],
        "unknown_preferences": [{"id": "Pxxx", "content": "preference text that deterministic tools cannot confidently convert"}],
        "unknown_attribute_tags": [{"id": "U001", "source_preference_id": "P001", "label": "unknown tag inferred from a span inside one preference", "content_span": "text span", "penalty_amount": "number"}],
        "unknown_preference_groups": [{"unknown_key": "normalized unknown text cluster", "preference_ids": ["P001"], "count": 2, "stacked_penalty_amount": "number"}],
        "dynamic_preference_rules": [
            {
                "id": "R001",
                "source_preference_id": "Pxxx",
                "source_unknown_tag_ids": ["U001"],
                "label": "new attribute tag inferred by LLM",
                "match": {
                    "cargo_name_contains": ["keyword"],
                    "start_city_contains": ["keyword"],
                    "end_city_contains": ["keyword"],
                    "start_or_end_city_contains": ["keyword"],
                    "max_pickup_km": "number or null",
                    "max_haul_km": "number or null",
                    "time_window": {"start_minute": "minute offset or null", "end_minute": "minute offset or null"},
                    "daily_time_window": {"start_minute_of_day": "0..1439", "end_minute_of_day": "1..1440"},
                    "periodic_stop_required": {"period_days": "integer", "min_wait_minutes": "integer"},
                },
                "effect": "hard_reject|penalize|boost",
                "per_violation_penalty_yuan": "number, one failed preference/tag violation",
                "expected_violations_per_month": "number, estimated repeat count in the configured planning horizon",
                "penalty_multiplier": "number, optional LLM risk weight",
                "severity": "critical|high|medium|low",
                "confidence": "0..1",
            }
        ],
    }

    @classmethod
    def build_profile_from_preferences(cls, preferences: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(p.get("content", "")) for p in preferences]
        text = "\n".join(texts)
        coords = extract_coordinates(text)
        pickup_limits = [cls._pickup_limit(item) for item in texts]
        monthly_limits = [cls._monthly_deadhead_limit(item) for item in texts]
        haul_limits = [cls._haul_limit(item) for item in texts]
        first_deadlines = [cls._first_order_deadline(item) for item in texts]
        daily_limits = [cls._daily_order_limit(item) for item in texts]
        off_days = [cls._off_days(item) for item in texts]
        rests = [cls._daily_rest(item) for item in texts]
        required_cargos: list[dict[str, Any]] = []
        # Required cargo must be parsed per preference item. Parsing the full
        # concatenated text can accidentally attach the first coordinate in the
        # entire profile, for example a forbidden-circle center, to a later
        # required cargo. That was one of the biggest hidden-driver failure modes.
        for item_text in texts:
            required_cargos.extend(cls._required_cargos(item_text))
        avoid_keywords: set[str] = set()
        # Avoid keywords are also item-scoped so explanatory clauses such as
        # "不接则损失客户" are not confused with "do not accept this cargo".
        for item_text in texts:
            avoid_keywords.update(cls._avoid_keywords(item_text))
        profile = {
            "avoid_cargo_keywords": sorted(avoid_keywords),
            "avoid_regions": cls._avoid_regions(text),
            "daily_rest": cls._merge_daily_rest(rests),
            "required_off_days": max(off_days) if off_days else 0,
            "pickup_deadhead_max_km": cls._min_not_none(pickup_limits),
            "monthly_deadhead_limit_km": cls._min_not_none(monthly_limits),
            "max_haul_km": cls._min_not_none(haul_limits),
            "first_order_deadline_minute": cls._min_not_none(first_deadlines),
            "daily_order_limit": cls._min_not_none(daily_limits),
            "geo_fence_bounds": cls._geo_fence(text),
            "forbidden_circles": cls._forbidden_circles(text),
            "required_cargos": required_cargos,
            "temporary_events": cls._temporary_events(text),
            "long_sequence_commitments": cls._long_sequence_commitments(text),
            "cumulative_time_penalty_rules": cls._cumulative_time_penalty_rules(text),
            "visit_frequency": cls._visit_frequency(text),
            "required_region_cargo_days": cls._required_region(text, coords),
            "scheduled_visits": cls._scheduled_visits(text, coords),
            "preference_points": [list(p) for p in coords],
            "tool_trace": cls._tool_trace(text),
        }
        cards = cls._preference_cards(preferences)
        profile["preference_cards"] = cards
        profile["duplicate_preference_groups"] = cls._duplicate_preference_groups(cards)
        profile["unknown_preferences"] = [
            {"id": card["id"], "content": card["content"], "reason": "low_deterministic_parse_confidence"}
            for card in cards
            if card.get("tradeoff_mode") == "unknown"
        ]
        profile["unknown_attribute_tags"] = cls._unknown_attribute_tags(cards)
        profile["unknown_preference_groups"] = cls._unknown_preference_groups(cards)
        profile["raw_preference_coverage_audit"] = cls._raw_preference_coverage_audit(cards, profile)
        profile["dynamic_preference_rules"] = []
        profile["risk_policy"] = cls._risk_policy(cards)
        return profile

    @staticmethod
    def _min_not_none(values: list[Any]) -> Any:
        clean = [value for value in values if value is not None]
        return min(clean) if clean else None

    @staticmethod
    def _merge_daily_rest(rests: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {"hours": None, "window_start_minute": None, "window_end_minute": None}
        for rest in rests:
            if not isinstance(rest, dict):
                continue
            hours = rest.get("hours")
            if hours is not None and (merged["hours"] is None or float(hours) > float(merged["hours"])):
                merged.update(rest)
        return merged

    @classmethod
    def llm_context(cls, preferences: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool_name": "driver_profile_tool",
            "purpose": "parse visible driver preferences into constraints, soft goals, penalties and planning hints",
            "input_source": "get_driver_status(driver_id).preferences only",
            "schema": cls.PROFILE_SCHEMA,
            "deterministic_profile": profile,
            "preference_count": len(preferences),
            "duplicate_preference_groups": profile.get("duplicate_preference_groups", []),
            "high_priority_preferences": [
                card for card in profile.get("preference_cards", [])
                if isinstance(card, dict) and card.get("severity") in {"critical", "high"}
            ][:12],
            "unknown_preferences": profile.get("unknown_preferences", []),
            "unknown_attribute_tags": profile.get("unknown_attribute_tags", []),
            "unknown_preference_groups": profile.get("unknown_preference_groups", []),
            "raw_preference_coverage_audit": profile.get("raw_preference_coverage_audit", []),
            "dynamic_preference_rules": profile.get("dynamic_preference_rules", []),
            "detected_constraint_types": sorted({item.get("type", "unknown") for item in profile.get("tool_trace", []) if isinstance(item, dict)}),
            "llm_instruction": (
                "Use this tool output as the safety floor. When there are many preferences, first reason over preference_cards and "
                "duplicate_preference_groups, unknown_attribute_tags and unknown_preference_groups, then convert high-risk unknown tags into conservative planning constraints. "
                "A single preference can contain multiple unknown tags; create one rule per tag when needed. "
                "For preferences not covered by fixed fields, create dynamic_preference_rules as JSON rule specs with new labels, match conditions, effect, per_violation_penalty_yuan, expected_violations_per_month and penalty_multiplier. "
                "You may add missing structure from the preference text, but do not remove forbidden cargo, forbidden regions, rest/off-day, required cargo, visit, or geo constraints."
            ),
        }

    @classmethod
    def _preference_cards(cls, preferences: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for idx, pref in enumerate(preferences, start=1):
            content = str(pref.get("content", ""))
            types = cls._detect_types(content)
            penalty_amount = cls._as_float(pref.get("penalty_amount"))
            penalty_cap = cls._as_float(pref.get("penalty_cap"))
            severity = cls._severity(types, penalty_amount, penalty_cap)
            risk_key = cls._risk_key(content, types)
            cards.append(
                {
                    "id": f"P{idx:03d}",
                    "content": content,
                    "types": types,
                    "risk_key": risk_key,
                    "severity": severity,
                    "tradeoff_mode": cls._tradeoff_mode(content, types, penalty_amount, penalty_cap),
                    "penalty_amount": penalty_amount,
                    "penalty_cap": penalty_cap,
                    "time_window": {"start_time": pref.get("start_time"), "end_time": pref.get("end_time")},
                    "planner_hint": cls._planner_hint(types, severity),
                }
            )
        return cards

    @classmethod
    def _duplicate_preference_groups(cls, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for card in cards:
            key = str(card.get("risk_key") or "")
            if not key or key == "unknown":
                continue
            grouped.setdefault(key, []).append(card)
        result: list[dict[str, Any]] = []
        for key, items in grouped.items():
            if len(items) < 2:
                continue
            penalty_amount = sum(float(item.get("penalty_amount") or 0.0) for item in items)
            finite_caps = [float(value) for item in items for value in [item.get("penalty_cap")] if value is not None]
            result.append(
                {
                    "risk_key": key,
                    "preference_ids": [str(item.get("id")) for item in items],
                    "count": len(items),
                    "types": sorted({t for item in items for t in item.get("types", []) if isinstance(t, str)}),
                    "max_severity": cls._max_severity(str(item.get("severity", "low")) for item in items),
                    "hard_count": sum(1 for item in items if item.get("tradeoff_mode") == "hard"),
                    "stacked_penalty_amount": round(penalty_amount, 2),
                    "stacked_penalty_cap": None if len(finite_caps) != len(items) else round(sum(finite_caps), 2),
                    "planner_hint": "Repeated preference risk: treat one violation as potentially triggering multiple penalties.",
                }
            )
        return sorted(result, key=lambda item: (-int(item["count"]), -float(item["stacked_penalty_amount"])))

    @classmethod
    def _unknown_preference_groups(cls, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for card in cards:
            if card.get("tradeoff_mode") != "unknown":
                continue
            key = cls._unknown_key(str(card.get("content") or ""))
            grouped.setdefault(key, []).append(card)
        result: list[dict[str, Any]] = []
        for key, items in grouped.items():
            if len(items) < 2:
                continue
            penalty_amount = sum(float(item.get("penalty_amount") or 0.0) for item in items)
            result.append(
                {
                    "unknown_key": key,
                    "preference_ids": [str(item.get("id")) for item in items],
                    "count": len(items),
                    "stacked_penalty_amount": round(penalty_amount, 2),
                    "max_severity": cls._max_severity(str(item.get("severity", "low")) for item in items),
                    "planner_hint": "Repeated unknown preference: ask LLM for one shared label/rule and stack penalty weight.",
                }
            )
        return sorted(result, key=lambda item: (-int(item["count"]), -float(item["stacked_penalty_amount"])))

    @staticmethod
    def _raw_preference_coverage_audit(cards: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
        audits: list[dict[str, Any]] = []
        signals = {
            "ordered_or_long_event": ("先", "再", "然后", "直到", "方可", "婚车", "接亲", "包车", "驻场", "连续"),
            "cumulative_time_penalty": ("每", "分钟", "小时", "上不封顶", "迟到", "不在"),
            "point_visit_or_frequency": ("自然月", "不同的自然日", "同日", "到过", "至少", "每周", "每月"),
            "family_or_personal": ("配偶", "孩子", "老人", "医院", "学校", "证件", "年审", "维修", "宗教", "祈祷"),
            "route_or_region": ("不进", "别去", "禁止", "范围", "山区", "港口", "市区", "高速", "跨省"),
            "cargo_or_loading": ("不接", "不拉", "危险品", "冷链", "活体", "装卸", "搬", "叉车", "超重", "超长"),
            "rest_or_health": ("休息", "睡", "熄火", "停车", "疲劳", "吃饭", "服药", "夜间", "凌晨"),
        }
        coverage = {
            "ordered_or_long_event": bool(profile.get("long_sequence_commitments") or profile.get("temporary_events")),
            "cumulative_time_penalty": bool(profile.get("cumulative_time_penalty_rules")),
            "route_or_region": bool(profile.get("avoid_regions") or profile.get("geo_fence_bounds") or profile.get("forbidden_circles")),
            "point_visit_or_frequency": bool(profile.get("visit_frequency") or profile.get("scheduled_visits")),
            "rest_or_health": bool((profile.get("daily_rest") or {}).get("hours") or profile.get("required_off_days")),
            "cargo_or_loading": bool(profile.get("avoid_cargo_keywords")),
            "family_or_personal": bool(profile.get("temporary_events") or profile.get("long_sequence_commitments") or profile.get("scheduled_visits")),
        }
        for card in cards:
            content = str(card.get("content") or "")
            matched = [key for key, words in signals.items() if any(word in content for word in words)]
            missing = [key for key in matched if not coverage.get(key)]
            if missing or card.get("tradeoff_mode") == "unknown":
                audits.append(
                    {
                        "id": card.get("id"),
                        "content": content[:260],
                        "signals": matched,
                        "missing_tool_coverage": missing,
                        "severity": card.get("severity"),
                        "penalty_amount": card.get("penalty_amount"),
                        "penalty_cap": card.get("penalty_cap"),
                        "llm_review_required": True,
                    }
                )
        return audits[:24]

    @classmethod
    def _unknown_attribute_tags(cls, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tags: list[dict[str, Any]] = []
        for card in cards:
            if card.get("tradeoff_mode") != "unknown":
                continue
            content = str(card.get("content") or "")
            tags.append(
                {
                    "id": f"U{len(tags) + 1:03d}",
                    "source_preference_id": card.get("id"),
                    "label": "unknown_preference_tag",
                    "content_span": content[:160],
                    "unknown_key": cls._unknown_key(content),
                    "penalty_amount": card.get("penalty_amount"),
                    "penalty_cap": card.get("penalty_cap"),
                    "planner_hint": "LLM may split this preference into multiple unknown tags if the text contains several independent likes/dislikes.",
                }
            )
        return tags

    @staticmethod
    def _unknown_key(text: str) -> str:
        cleaned = re.sub(r"\s+", "", text)
        cleaned = re.sub(r"\d+(?:\.\d+)?", "#", cleaned)
        cleaned = re.sub(r"[（(]\s*#\s*[，,]\s*#\s*[）)]", "(#,#)", cleaned)
        return cleaned[:80] or "unknown"

    @staticmethod
    def _max_severity(values: Any) -> str:
        rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        best = "low"
        for value in values:
            if rank.get(str(value), 0) > rank[best]:
                best = str(value)
        return best

    @classmethod
    def _risk_key(cls, text: str, types: list[str]) -> str:
        if not types:
            return "unknown"
        quoted = re.findall(r"[「“\"']([^」“”\"']{1,20})[」”\"']", text)
        coords = extract_coordinates(text)
        for risk_type in (
            "required_cargo",
            "temporary_event",
            "visit_frequency",
            "forbidden_circle",
            "geo_fence",
            "home_return",
            "avoid_region",
            "first_order_deadline",
            "daily_order_limit",
            "pickup_deadhead_limit",
            "monthly_deadhead_limit",
            "distance_limit",
            "rest_or_no_action",
            "avoid_cargo",
        ):
            if risk_type not in types:
                continue
            if risk_type == "avoid_cargo":
                target = quoted[0] if quoted else cls._first_known_keyword(text)
                return f"avoid_cargo:{target or 'generic'}"
            if risk_type in {"visit_frequency", "forbidden_circle", "temporary_event", "home_return"} and coords:
                lat, lng = coords[0]
                return f"{risk_type}:{lat:.3f},{lng:.3f}"
            if risk_type == "required_cargo":
                cargo_ids = re.findall(r"(?:编号|货源编号|货源)\s*([0-9]{4,})", text)
                return f"required_cargo:{cargo_ids[0] if cargo_ids else 'generic'}"
            if risk_type == "geo_fence":
                fence = cls._geo_fence(text)
                if fence:
                    return "geo_fence:{lat_min:.2f}-{lat_max:.2f}:{lng_min:.2f}-{lng_max:.2f}".format(**fence)
            return risk_type
        return str(types[0])

    @staticmethod
    def _first_known_keyword(text: str) -> str | None:
        for keyword in _flatten_lexicon(CARGO_KEYWORD_LEXICON):
            if keyword in text:
                return keyword
        quoted = re.findall(r"[「“\"']([^」“”\"']{1,20})[」”\"']", text)
        if quoted:
            return quoted[0].strip()
        return None

    @staticmethod
    def _looks_like_cargo_avoidance(text: str) -> bool:
        if any(marker in text for marker in ("货源品类", "品类", "不拉", "尽量不拉", "货物", "货品", "货名", "装卸", "搬运")):
            return True
        if re.search(r"(?:不接|拒绝|避免|别派|不碰|不要|不能).{0,16}(?:订单|货源|货|品类|货品|货物|装卸|搬运)", text):
            return True
        return False

    @staticmethod
    def _looks_like_visit_frequency(text: str) -> bool:
        if "自然日" not in text:
            return False
        if "到过" in text:
            return True
        if extract_coordinates(text) and ("一公里内" in text or re.search(r"到.{0,16}[（(]", text)):
            return True
        return False

    @staticmethod
    def _looks_like_monthly_deadhead_limit(text: str) -> bool:
        return "空驶" in text and any(word in text for word in ("总和", "总里程", "月内", "一个月", "自然月"))

    @staticmethod
    def _looks_like_pickup_deadhead_limit(text: str) -> bool:
        if "空驶" not in text:
            return False
        return any(word in text for word in ("赴装货点", "接单后", "装货点空驶", "空驶距离不得超过"))

    @classmethod
    def _detect_types(cls, text: str) -> list[str]:
        checks = [
            ("avoid_cargo", ("不接", "不拉", "避免", "拒绝", "尽量不拉", "不碰", "别派", "绕开")),
            ("rest_or_no_action", ("休息", "歇满", "停车", "不行动", "不接单", "不空跑", "不空车", "不跑车", "禁行", "午休", "吃饭")),
            ("geo_fence", ("北纬", "东经", "范围内")),
            ("forbidden_circle", ("不得进入", "禁止进入", "禁入", "半径")),
            ("required_cargo", ("熟货", "指定货源", "必接", "必须接", "一定要接")),
            ("temporary_event", ("家事", "配偶", "老家", "解决前", "孩子", "老人", "医院", "学校", "证件", "婚礼")),
            ("visit_frequency", ("自然日", "到过", "至少", "一公里内", "每周", "每月", "打卡")),
            ("distance_limit", ("装货点至卸货点", "装卸距离", "运距", "单笔货")),
            ("pickup_deadhead_limit", ("赴装货点", "接单后", "空驶距离")),
            ("monthly_deadhead_limit", ("空驶", "总和", "一个月", "自然月")),
            ("first_order_deadline", ("首单", "第一单", "首个订单", "开工")),
            ("daily_order_limit", ("同一天", "每天", "每日", "接单不得超过", "每天最多", "最多接")),
            ("home_return", ("回家", "自家位置", "进家门", "23点前")),
        ]
        detected = [name for name, words in checks if any(w in text for w in words)]
        if "avoid_cargo" in detected and not cls._looks_like_cargo_avoidance(text):
            detected.remove("avoid_cargo")
        if "visit_frequency" in detected and not cls._looks_like_visit_frequency(text):
            detected.remove("visit_frequency")
        if "monthly_deadhead_limit" in detected and not cls._looks_like_monthly_deadhead_limit(text):
            detected.remove("monthly_deadhead_limit")
        if "pickup_deadhead_limit" in detected and not cls._looks_like_pickup_deadhead_limit(text):
            detected.remove("pickup_deadhead_limit")
        return detected

    @classmethod
    def _risk_policy(cls, cards: list[dict[str, Any]]) -> dict[str, Any]:
        hard = [c["id"] for c in cards if c.get("tradeoff_mode") == "hard"]
        soft = [c["id"] for c in cards if c.get("tradeoff_mode") == "soft"]
        unknown = [c["id"] for c in cards if c.get("tradeoff_mode") == "unknown"]
        return {
            "hard_constraint_ids": hard,
            "soft_tradeoff_ids": soft,
            "unknown_constraint_ids": unknown,
            "duplicate_risk_keys": sorted({str(card.get("risk_key")) for card in cards if str(card.get("risk_key") or "") not in {"", "unknown"} and sum(1 for other in cards if other.get("risk_key") == card.get("risk_key")) > 1}),
            "strategy": "Maximize expected money: mandatory commitments/geofences remain hard, while ordinary preferences are penalty costs that a high-profit order may rationally pay.",
        }

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _severity(cls, types: list[str], penalty_amount: float | None, penalty_cap: float | None) -> str:
        if penalty_cap is None and (penalty_amount or 0) >= 500:
            return "critical"
        if any(t in types for t in ("temporary_event", "required_cargo")) or (penalty_cap or 0) >= 9000 or (penalty_amount or 0) >= 3000:
            return "critical"
        if any(t in types for t in ("geo_fence", "forbidden_circle", "home_return")) or (penalty_cap or 0) >= 4000 or (penalty_amount or 0) >= 800:
            return "high"
        if types or (penalty_amount or 0) >= 200:
            return "medium"
        return "low"

    @classmethod
    def _tradeoff_mode(cls, content: str, types: list[str], penalty_amount: float | None, penalty_cap: float | None) -> str:
        if not types:
            return "unknown"
        if "尽量" in content or "希望" in content:
            return "soft"
        hard_types = {"temporary_event", "required_cargo", "geo_fence", "forbidden_circle", "home_return"}
        if any(t in types for t in hard_types):
            return "hard"
        if (penalty_cap or 0) >= 9000 and any(t in types for t in hard_types):
            return "hard"
        return "soft"

    @staticmethod
    def _planner_hint(types: list[str], severity: str) -> str:
        if not types:
            return "Ask LLM to infer operational risk from text; treat high-penalty unknown text conservatively."
        if any(t in types for t in ("temporary_event", "required_cargo", "geo_fence", "forbidden_circle", "home_return")):
            return "Mandatory commitment or safety/geofence constraint: avoid unless no legal alternative."
        return "Treat as a monetary penalty in candidate scoring; a high-profit order may justify paying it."

    @staticmethod
    def _tool_trace(text: str) -> list[dict[str, Any]]:
        return [{"type": name, "matched": True} for name in DriverProfileTool._detect_types(text)]

    @staticmethod
    def _cn_num(value: str) -> int | None:
        value = str(value).strip().replace("个", "")
        if value.isdigit():
            return int(value)
        table = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if value in table:
            return table[value]
        if value.startswith("十") and len(value) == 2:
            return 10 + table.get(value[1], 0)
        if "十" in value:
            left, right = value.split("十", 1)
            return table.get(left, 1) * 10 + (table.get(right, 0) if right else 0)
        if "百" in value:
            left, right = value.split("百", 1)
            return table.get(left, 1) * 100 + (DriverProfileTool._cn_num(right) or 0 if right else 0)
        return None

    @classmethod
    def _avoid_keywords(cls, text: str) -> list[str]:
        text = re.sub(r"不接则[^。；;\n]*", "", text)
        keywords: set[str] = set()
        for quoted in re.findall(r"[「“\"']([^」“”\"']{1,20})[」”\"']", text):
            if any(word in text for word in NEGATIVE_CARGO_MARKERS):
                keywords.add(quoted.strip())
        if any(word in text for word in NEGATIVE_CARGO_MARKERS):
            for canonical, aliases in CARGO_KEYWORD_LEXICON.items():
                if canonical in text or any(alias in text for alias in aliases):
                    keywords.update(aliases)
                    keywords.add(canonical)
        pattern = r"(?:不接|不要接|不拉|避免|推掉|干不了|拒绝|别派|不能碰|不碰|绕开)([^。；;，,\n]{1,30})"
        for match in re.finditer(pattern, text):
            segment = re.sub(r"(?:的|这类|这种|货源|货|货品|货物|活儿|订单|都|一律|凡是|每接.*)$", "", match.group(1)).strip()
            for token in re.split(r"[、/和与及\s]+", segment):
                token = token.strip("：“”\"'「」")
                route_token = token.startswith(("去", "往", "到", "进"))
                token = re.sub(r"^(?:去|往|到|进)", "", token)
                token = re.sub(r"(?:的货|货源|货品|货物|订单|的)$", "", token)
                if (
                    2 <= len(token) <= 12
                    and not route_token
                    and not any(stop in token for stop in TOKEN_STOPWORDS)
                    and not any(word in token for word in ("装货点", "卸货点", "地区路线"))
                ):
                    keywords.add(token)
        return sorted(keywords)

    @classmethod
    def _avoid_regions(cls, text: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for segment in re.split(r"[。；;\n]", text):
            if not any(word in segment for word in REGION_NEGATIVE_MARKERS):
                continue
            candidates = cls._region_mentions(segment)
            for region in candidates:
                if region in seen:
                    continue
                seen.add(region)
                result.append({"region": region, "days": cls._days_near(segment) or None})
        return result

    @staticmethod
    def _region_mentions(text: str) -> list[str]:
        mentions: list[str] = []
        patterns = [
            r"(?:不往|不去|避开|别去|不要去|禁止进入|不得进入|禁入|绕开|远离|不进)([\u4e00-\u9fa5]{2,10}(?:市|区|县|镇|城|港|湾|岛|山|园|场|口)?)",
            r"(?:装货地|卸货地|装货地或卸货地|起点|终点|出发地|目的地)?(?:在|到|去|往)?([\u4e00-\u9fa5]{2,4})(?:的货|货源|订单)",
            r"([\u4e00-\u9fa5]{2,4})(?:那一路|那边|这一路|这边)",
            r"([\u4e00-\u9fa5]{2,10}(?:市|区|县|镇|城|港|湾|岛|山|园|场|口))(?:方向|周边|附近|里面|范围|路线|的货|订单)?",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                raw = match.group(1).strip("，,。；;、和与及 ")
                raw = re.sub(r"^(?:一律)?(?:不接|不要接|不去|不往|避开|别去|不要去|禁止进入|不得进入|禁入|绕开|远离|去|往|到|进)+", "", raw)
                raw = re.sub(r"(?:的货|货源|货品|货物|订单|路线|方向|附近|周边|里面|范围)$", "", raw)
                parts = re.split(r"[、/和与及\s]+", raw)
                for part in parts:
                    part = part.strip()
                    if 2 <= len(part) <= 10 and not any(stop in part for stop in ("货源", "订单", "装货", "卸货", "空驶")):
                        mentions.append(part)
        result: list[str] = []
        seen: set[str] = set()
        for item in mentions:
            normalized = item[:-1] if len(item) > 2 and item.endswith(("市", "县")) else item
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result[:12]

    @classmethod
    def _days_near(cls, text: str) -> list[int]:
        return day_indices_from_text(text)

    @classmethod
    def _daily_rest(cls, text: str) -> dict[str, Any]:
        rest: dict[str, Any] = {"hours": None, "window_start_minute": None, "window_end_minute": None}
        match = re.search(r"(?:连续|至少|满|歇满)[^0-9一二两三四五六七八九十]*([0-9一二两三四五六七八九十]+)\s*小时", text)
        if match:
            rest["hours"] = cls._cn_num(match.group(1))
        window = cls._parse_daily_window(text)
        # Plain no-order/no-driving windows are handled separately as
        # no-action windows. Do not convert them into an extra flexible daily
        # rest requirement unless the text explicitly asks for rest/sleep.
        if window is not None and any(word in text for word in ("休息", "睡觉", "熄火", "歇")):
            start, end = window
            hours = ((end - start) % 1440) / 60.0
            rest.update({"hours": max(rest["hours"] or 0, hours), "window_start_minute": start, "window_end_minute": end})
        return rest

    @classmethod
    def _parse_daily_window(cls, text: str) -> tuple[int, int] | None:
        normalized = text.replace("零点", "0点").replace("凌晨", "").replace("清晨", "").replace("早上", "").replace("早", "").replace("上午", "")
        normalized = re.sub(r"晚上\s*([0-9一二两三四五六七八九十]+)\s*点", lambda m: f"{(cls._cn_num(m.group(1)) or 0) + 12}点", normalized)
        normalized = re.sub(r"下午\s*([0-9一二两三四五六七八九十]+)\s*点", lambda m: f"{(cls._cn_num(m.group(1)) or 0) + 12}点", normalized)
        match = re.search(r"([0-9一二两三四五六七八九十]{1,3})\s*(?:点|:：)?\s*(?:以后|后)?\s*(?:到|至|-|~|—)\s*(?:次日|第二天)?\s*([0-9一二两三四五六七八九十]{1,3})\s*(?:点|:：)", normalized)
        if not match:
            return None
        start = cls._cn_num(match.group(1))
        end = cls._cn_num(match.group(2))
        return None if start is None or end is None else ((start % 24) * 60, (end % 24) * 60)

    @classmethod
    def _off_days(cls, text: str) -> int:
        if not any(word in text for word in ("整天", "停驶", "完全歇着", "放空一整天")):
            return 0
        values = [cls._cn_num(m.group(1)) for m in re.finditer(r"([0-9一二两三四五六七八九十]+)\s*(?:个)?(?:整天|天|日)", text)]
        values = [v for v in values if v is not None]
        return max(values) if values else (2 if "两" in text or "二" in text else 1)

    @classmethod
    def _pickup_limit(cls, text: str) -> float | None:
        if "空驶" not in text or any(word in text for word in ("总和", "总里程", "月内", "一个月", "自然月")):
            return None
        match = re.search(r"空驶[^0-9一二三四五六七八九十百]*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
        cn = re.search(r"空驶[^0-9一二三四五六七八九十百]*([一二两三四五六七八九十百]+)", text)
        value = cls._cn_num(cn.group(1)) if cn else None
        return float(value) if value else None

    @classmethod
    def _monthly_deadhead_limit(cls, text: str) -> float | None:
        if "空驶" not in text or not any(word in text for word in ("总和", "总里程", "月内", "一个月", "自然月")):
            return None
        match = re.search(r"空驶[^。；;\n]{0,24}?(?:不得超过|不超过|控制在|少于|低于)\s*([0-9]+(?:\.[0-9]+)?)\s*公里", text)
        if match:
            return float(match.group(1))
        cn = re.search(r"空驶[^。；;\n]{0,24}?(?:不得超过|不超过|控制在|少于|低于)\s*([一二两三四五六七八九十百]+)\s*公里", text)
        value = cls._cn_num(cn.group(1)) if cn else None
        return float(value) if value else None

    @classmethod
    def _first_order_deadline(cls, text: str) -> int | None:
        if not any(word in text for word in ("首单", "第一单", "首个订单")) or not any(word in text for word in ("不得晚于", "不晚于", "前", "之前")):
            return None
        if "中午12点" in text or "12点" in text or "十二点" in text:
            return 12 * 60
        match = re.search(r"(?:首单|第一单|首个订单)[^。；;\n]{0,30}?([0-9一二两三四五六七八九十]{1,3})\s*点", text)
        if not match:
            return None
        hour = cls._cn_num(match.group(1))
        if hour is None:
            return None
        if "下午" in text and hour < 12:
            hour += 12
        return (hour % 24) * 60

    @classmethod
    def _daily_order_limit(cls, text: str) -> int | None:
        if not any(word in text for word in ("同一天", "每天", "每日")) or not any(word in text for word in ("不得超过", "不超过", "最多")) or "接单" not in text:
            return None
        match = re.search(r"(?:不得超过|不超过|最多)\s*([0-9一二两三四五六七八九十]+)\s*单", text)
        value = cls._cn_num(match.group(1)) if match else None
        return value if value and value > 0 else None

    @classmethod
    def _haul_limit(cls, text: str) -> float | None:
        if not any(word in text for word in ("装卸距离", "装货点至卸货点", "单笔货", "干线", "运距")):
            return None
        match = re.search(r"(?:装卸距离|装货点至卸货点|单笔货[^，。；;]{0,12}距离|干线|运距)[^0-9一二两三四五六七八九十百]*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1))
        cn = re.search(r"(?:装卸距离|装货点至卸货点|单笔货[^，。；;]{0,12}距离|干线|运距)[^0-9一二两三四五六七八九十百]*([一二两三四五六七八九十百]+)", text)
        value = cls._cn_num(cn.group(1)) if cn else None
        return float(value) if value else None

    @staticmethod
    def _geo_fence(text: str) -> dict[str, float] | None:
        match = re.search(r"北纬\s*([0-9]+(?:\.[0-9]+)?)\s*(?:至|到|-|~|—|–)\s*([0-9]+(?:\.[0-9]+)?)[\s\S]{0,20}?东经\s*([0-9]+(?:\.[0-9]+)?)\s*(?:至|到|-|~|—|–)\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return None
        lat1, lat2, lng1, lng2 = [float(x) for x in match.groups()]
        return {"lat_min": min(lat1, lat2), "lat_max": max(lat1, lat2), "lng_min": min(lng1, lng2), "lng_max": max(lng1, lng2)}

    @staticmethod
    def _forbidden_circles(text: str) -> list[dict[str, Any]]:
        if not any(word in text for word in ("不得进入", "禁止进入", "禁入", "别进")):
            return []
        return [
            {"center": [float(m.group(1)), float(m.group(2))], "radius_km": float(m.group(3))}
            for m in re.finditer(r"[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*[，,]\s*([0-9]+(?:\.[0-9]+)?)\s*[）)].{0,24}?半径\s*([0-9]+(?:\.[0-9]+)?)\s*公里", text)
        ]

    @staticmethod
    def _required_cargos(text: str) -> list[dict[str, Any]]:
        if not any(word in text for word in ("熟货", "指定货源", "必须接", "必接")):
            return []
        coords = extract_coordinates(text)
        times = datetime_mentions(text)
        online = minute_offset(times[0]) if times else None
        return [{"cargo_id": cargo_id, "pickup_point": list(coords[0]) if coords else None, "online_minute": online} for cargo_id in re.findall(r"(?:编号|货源编号|货源)\s*([0-9]{4,})", text)]

    @staticmethod
    def _temporary_events(text: str) -> list[dict[str, Any]]:
        if not DriverProfileTool._looks_like_personal_commitment(text):
            return []
        coords = extract_coordinates(text)
        times = [minute_offset(x) for x in datetime_mentions(text)]
        times = [x for x in times if x is not None]
        if len(coords) < 2 or not times:
            return []
        start = min(times)
        end = max(times)
        return [{"pickup_point": list(coords[0]), "home_point": list(coords[1]), "pickup_minute": start, "release_minute": end if end > start else start + 12 * 60}]

    @staticmethod
    def _looks_like_personal_commitment(text: str) -> bool:
        personal_words = ("家事", "配偶", "老家", "旧家", "新家", "搬家", "孩子", "老人", "医院", "学校", "证件", "年审", "维修", "婚礼", "婚宴", "赴宴", "接亲", "接人", "赴约")
        commitment_words = ("解决前", "接上", "接到", "返回", "送到", "陪同", "必须", "不得", "之前", "待到", "停留", "释放")
        return any(word in text for word in personal_words) and any(word in text for word in commitment_words)

    @classmethod
    def _long_sequence_commitments(cls, text: str) -> list[dict[str, Any]]:
        """Convert long ordered preference text into explicit executable phases."""
        commitments: list[dict[str, Any]] = []
        context_coords = extract_coordinates(text)
        for segment in re.split(r"\n+", text):
            if not segment.strip():
                continue
            has_order = any(word in segment for word in ("先", "再", "然后", "之后", "解决前", "至少待到", "方可", "全程", "连续"))
            has_long_event = cls._looks_like_personal_commitment(segment) or any(word in segment for word in ("婚车", "婚礼", "婚宴", "赴宴", "接亲", "搬家", "旧家", "新家", "长途包车", "包车", "长周期"))
            has_mandatory_tone = any(word in segment for word in ("必须", "不得", "禁止", "否则", "罚", "违约", "待到", "停留", "驻留", "释放"))
            if not has_order:
                continue
            ordered_event = cls._ordered_scheduled_event_sequence(segment, len(commitments) + 1, context_coords)
            if ordered_event is not None:
                commitments.append(ordered_event)
                continue
            coords = extract_coordinates(segment)
            times = [
                minute_offset(value)
                for value in datetime_mentions(segment)
            ]
            times = sorted({int(value) for value in times if value is not None})
            if not coords or not times or (not has_long_event and not has_mandatory_tone):
                continue
            start = times[0]
            release = times[-1] if len(times) > 1 else start + 12 * 60
            if release <= start:
                release = start + 12 * 60
            intermediate_deadlines = [value for value in times[1:-1] if value > start]
            home_deadline = intermediate_deadlines[0] if intermediate_deadlines else min(release, start + 12 * 60)
            pickup_wait = cls._explicit_wait_minutes(segment) or (10 if any(word in segment for word in ("停留不少于10分钟", "停留不少于十分钟", "原地停留", "接上")) else 1)
            commitment_type = "personal_event" if cls._looks_like_personal_commitment(segment) else "ordered_long_event"
            steps: list[dict[str, Any]] = []
            if len(coords) == 1:
                steps = [
                    {
                        "id": "single_stop_hold",
                        "label": "arrive required stop and stay until release",
                        "step_type": "stay_until",
                        "point": list(coords[0]),
                        "earliest_minute": start,
                        "deadline_minute": start,
                        "hold_until_minute": release,
                        "radius_km": 1.0,
                    }
                ]
            for index, point in enumerate(coords):
                if len(coords) == 1:
                    break
                is_first = index == 0
                is_last = index == len(coords) - 1
                if is_last:
                    steps.append(
                        {
                            "id": "final_stop_hold",
                            "label": "arrive final stop and stay until release",
                            "step_type": "stay_until",
                            "point": list(point),
                            "earliest_minute": start,
                            "deadline_minute": home_deadline,
                            "hold_until_minute": release,
                            "radius_km": 1.0,
                        }
                    )
                    continue
                deadline = home_deadline
                if index < len(intermediate_deadlines):
                    deadline = intermediate_deadlines[index]
                steps.append(
                    {
                        "id": "pickup_or_first_stop" if is_first else f"ordered_stop_{index + 1}",
                        "label": "complete ordered stop before moving to the next phase",
                        "step_type": "visit_and_wait",
                        "point": list(point),
                        "earliest_minute": start,
                        "deadline_minute": deadline,
                        "wait_minutes": pickup_wait if is_first else 1,
                        "radius_km": 1.0,
                    }
                )
            commitments.append(
                {
                    "id": f"SEQ{len(commitments) + 1:03d}",
                    "commitment_type": commitment_type,
                    "source_text": segment[:260],
                    "buffer_minutes": 60,
                    "steps": steps,
                }
            )
        return commitments

    @classmethod
    def _ordered_scheduled_event_sequence(cls, segment: str, index: int, context_coords: list[tuple[float, float]]) -> dict[str, Any] | None:
        """Parse one-shot tasks like: on a date, first visit A, then reach B and stay until a time."""
        if not any(word in segment for word in ("赴宴", "婚宴", "婚礼", "寿宴", "宴席", "搬家", "旧家", "新家", "捎上", "带上", "先过", "先到", "随后", "再到", "赶到")):
            return None
        coords = extract_coordinates(segment)
        if len(coords) == 1 and any(word in segment for word in ("先过", "先到", "捎上", "带上", "档口", "老档口")):
            for candidate in context_coords:
                if abs(candidate[0] - coords[0][0]) + abs(candidate[1] - coords[0][1]) > 0.02:
                    coords = [candidate, coords[0]]
                    break
        days = day_indices_from_text(segment)
        if len(coords) < 2 or not days:
            return None
        deadlines = cls._deadline_minutes(segment)
        deadline_minute = deadlines[-1] if deadlines else None
        release_minute = cls._release_minute_of_day(segment)
        if deadline_minute is None:
            return None
        if release_minute is None:
            release_minute = max(deadline_minute + 60, 18 * 60)
        if release_minute <= deadline_minute:
            release_minute += 12 * 60
        day = days[0]
        final_deadline = day * 1440 + deadline_minute
        release = day * 1440 + release_minute
        first_deadline = day * 1440 + deadlines[0] if len(deadlines) >= 2 else max(day * 1440, final_deadline - 120)
        first_earliest = max(day * 1440, first_deadline - 180)
        return {
            "id": f"SEQ{index:03d}",
            "commitment_type": "ordered_scheduled_event",
            "source_text": segment[:260],
            "buffer_minutes": 90,
            "steps": [
                {
                    "id": "first_required_stop",
                    "label": "visit first required stop before final event",
                    "step_type": "visit_and_wait",
                    "point": list(coords[0]),
                    "earliest_minute": first_earliest,
                    "deadline_minute": first_deadline,
                    "wait_minutes": cls._explicit_wait_minutes(segment) or 1,
                    "radius_km": 1.0,
                },
                {
                    "id": "final_event_stay",
                    "label": "arrive final event and stay until release time",
                    "step_type": "stay_until",
                    "point": list(coords[-1]),
                    "earliest_minute": max(day * 1440, final_deadline - 60),
                    "deadline_minute": final_deadline,
                    "hold_until_minute": release,
                    "radius_km": 1.0,
                },
            ],
        }

    @classmethod
    def _release_minute_of_day(cls, text: str) -> int | None:
        patterns = [
            r"(?:直到|待到|必须待到|停到|停留到|赴宴到|待至|留到)\s*(下午|上午|中午|晚上|夜里|凌晨)?\s*([0-9一二两三四五六七八九十]{1,3})\s*点",
            r"(?:下午|上午|中午|晚上|夜里|凌晨)\s*([0-9一二两三四五六七八九十]{1,3})\s*点\s*(?:后|之后|再出车|才能|方可)",
        ]
        candidates: list[int] = []
        for pattern in patterns:
            for match in re.findall(pattern, text):
                if isinstance(match, tuple):
                    meridiem, hour_text = match
                else:
                    meridiem, hour_text = "", match
                hour = cls._cn_num(hour_text)
                if hour is None:
                    continue
                if meridiem in ("下午", "晚上", "夜里") and hour < 12:
                    hour += 12
                if meridiem == "中午" and hour < 11:
                    hour += 12
                candidates.append((hour % 24) * 60)
        return max(candidates) if candidates else None

    @classmethod
    def _explicit_wait_minutes(cls, text: str) -> int | None:
        match = re.search(r"(?:停留|待|原地停留)[^0-9一二两三四五六七八九十]{0,8}(?:不少于|至少)?\s*([0-9一二两三四五六七八九十]{1,3})\s*分钟", text)
        if not match:
            return None
        value = cls._cn_num(match.group(1))
        return max(1, int(value)) if value is not None else None

    @classmethod
    def _cumulative_time_penalty_rules(cls, text: str) -> list[dict[str, Any]]:
        rules: list[dict[str, Any]] = []
        for segment in re.split(r"\n+", text):
            if not any(word in segment for word in ("每", "罚")) or not any(unit in segment for unit in ("分钟", "小时")):
                continue
            rate = cls._time_penalty_rate_per_minute(segment)
            if rate is None or rate <= 0:
                continue
            coords = extract_coordinates(segment)
            times = [
                minute_offset(value)
                for value in datetime_mentions(segment)
            ]
            times = sorted({int(value) for value in times if value is not None})
            required_point = list(coords[-1]) if coords and any(word in segment for word in ("在家", "老家", "到家", "指定点", "原处", "现场")) else None
            trigger = "not_at_required_point" if required_point else ("late" if "迟到" in segment else "unknown")
            rules.append(
                {
                    "id": f"TIMEPEN{len(rules) + 1:03d}",
                    "source_text": segment[:260],
                    "rate_yuan_per_minute": round(rate, 4),
                    "window_start_minute": times[-2] if len(times) >= 2 else (times[0] if times else None),
                    "window_end_minute": times[-1] if len(times) >= 2 else None,
                    "required_point": required_point,
                    "radius_km": 1.0,
                    "trigger": trigger,
                }
            )
        return rules

    @classmethod
    def _time_penalty_rate_per_minute(cls, text: str) -> float | None:
        compact = re.sub(r"\s+", "", text)
        match = re.search(r"每[^，。；;]{0,16}?([0-9一二两三四五六七八九十百]+)\s*(分钟|小时)[^，。；;]{0,12}?罚\s*([0-9]+(?:\.[0-9]+)?)\s*元", compact)
        if not match:
            match = re.search(r"每[^，。；;]{0,16}?(分钟|小时)[^，。；;]{0,12}?罚\s*([0-9]+(?:\.[0-9]+)?)\s*元", compact)
            if not match:
                return None
            unit = match.group(1)
            amount = float(match.group(2))
            quantity = 1
        else:
            quantity = cls._cn_num(match.group(1)) or 1
            unit = match.group(2)
            amount = float(match.group(3))
        minutes = max(1, quantity * (60 if unit == "小时" else 1))
        return amount / minutes

    @classmethod
    def _visit_frequency(cls, text: str) -> dict[str, Any]:
        for segment in re.split(r"[。；;\n]", text):
            if "至少" not in segment or "自然日" not in segment or "到" not in segment:
                continue
            coords = extract_coordinates(segment)
            match = re.search(r"至少\s*([0-9一二两三四五六七八九十]+)\s*个?不同", segment)
            radius = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*公里", segment)
            if match and coords:
                return {"required_days": cls._cn_num(match.group(1)) or 0, "point": list(coords[0]), "radius_km": float(radius.group(1)) if radius else 1.0}
        return {"required_days": 0, "point": None, "radius_km": 1.0}

    @classmethod
    def _required_region(cls, text: str, coords: list[tuple[float, float]]) -> dict[str, Any]:
        if not ("不同的日子" in text and any(word in text for word in ("接够", "起码", "至少"))):
            return {"region": None, "min_days": 0, "point": None}
        segments = [segment for segment in re.split(r"[。；;\n]", text) if "不同的日子" in segment]
        context = segments[0] if segments else text
        match = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?不同的日子", context)
        context_match = re.search(r"(?:在|到|去|往|跑|接够|起码|至少)([\u4e00-\u9fa5]{2,10}?)(?:的货|货源|订单|方向|周边|附近|不同的日子)", context)
        region = context_match.group(1) if context_match else (cls._region_mentions(context)[0] if cls._region_mentions(context) else None)
        return {"region": region, "min_days": cls._cn_num(match.group(1)) if match else 0, "point": list(coords[0]) if coords else None}

    @classmethod
    def _scheduled_visits(cls, text: str, coords: list[tuple[float, float]]) -> list[dict[str, Any]]:
        visits: list[dict[str, Any]] = []
        for m in re.finditer(r"(?:(\d{4})年)?([0-9一二两三四五六七八九十]{1,3})月\s*([0-9一二两三四五六七八九十]{1,3})\s*[号日]", text):
            year_text, month_text, day_text = m.groups()
            month = _parse_cn_number(month_text)
            day_value = _parse_cn_number(day_text)
            if month is None or day_value is None:
                continue
            try:
                day = (datetime(int(year_text) if year_text else BASE_TIME.year, month, day_value) - BASE_TIME).days
            except ValueError:
                continue
            if day is None or not coords:
                continue
            line_start = max(text.rfind("\n", 0, m.start()), text.rfind("。", 0, m.start()), text.rfind("；", 0, m.start())) + 1
            line_end_candidates = [idx for idx in (text.find("\n", m.end()), text.find("。", m.end()), text.find("；", m.end())) if idx >= 0]
            context = text[line_start:(min(line_end_candidates) if line_end_candidates else min(len(text), m.start() + 180))]
            local = extract_coordinates(context) or coords
            deadline = cls._deadline_minute(context)
            if any(word in context for word in ("先过", "再到", "赶到", "赴宴")) and len(local) >= 2:
                if deadline is None:
                    visits.append({"day": day, "point": list(local[0]), "wait_minutes": 1, "arrive_before_minute": None, "confidence": "unknown_time"})
                    continue
                visits.append({"day": day, "point": list(local[0]), "wait_minutes": 1, "arrive_before_minute": deadline})
                visits.append({"day": day, "point": list(local[-1]), "wait_minutes": 1, "arrive_before_minute": deadline})
            else:
                visits.append({"day": day, "point": list(local[0]), "wait_minutes": 120 if "两小时" in context or "2小时" in context else 1, "arrive_before_minute": deadline})
        return visits

    @classmethod
    def _deadline_minute(cls, text: str) -> int | None:
        values = cls._deadline_minutes(text)
        return values[0] if values else None

    @classmethod
    def _deadline_minutes(cls, text: str) -> list[int]:
        values: list[int] = []
        for match in re.finditer(r"(上午|早上|中午|下午|晚上|夜里|凌晨)?\s*([0-9一二两三四五六七八九十]{1,3})\s*点\s*前", text):
            meridiem, hour_text = match.groups()
            hour = cls._cn_num(hour_text)
            if hour is None:
                continue
            if meridiem in ("下午", "晚上", "夜里") and hour < 12:
                hour += 12
            if meridiem == "中午" and hour < 11:
                hour += 12
            value = (hour % 24) * 60
            if value not in values:
                values.append(value)
        return values
