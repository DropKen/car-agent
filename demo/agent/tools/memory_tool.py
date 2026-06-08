"""History and feedback memory tool for the online agent."""

from __future__ import annotations

from typing import Any


class MemoryTool:
    """Keeps only online runtime memory derived from query_decision_history."""

    @staticmethod
    def failed_cargo_ids(records: list[dict[str, Any]]) -> set[str]:
        failed: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            cargo_id = str((act.get("params") or {}).get("cargo_id", ""))
            if act.get("action") == "take_order" and cargo_id and result.get("accepted") is False:
                failed.add(cargo_id)
        return failed

    @staticmethod
    def compact_history(records: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for record in list(records or [])[-limit:]:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            params = act.get("params") if isinstance(act.get("params"), dict) else {}
            out.append({
                "action": act.get("action"),
                "params": params,
                "elapsed": record.get("step_elapsed_minutes"),
                "end_minute": result.get("simulation_progress_minutes"),
                "accepted": result.get("accepted"),
                "cargo_id": params.get("cargo_id"),
                "distance_km": result.get("distance_km"),
                "pickup_deadhead_km": result.get("pickup_deadhead_km"),
                "income": result.get("income") or result.get("price"),
                "preference_penalty": result.get("preference_penalty") or result.get("penalty"),
                "position_after": record.get("position_after"),
            })
        return out

    @staticmethod
    def penalty_anomaly_summary(records: list[dict[str, Any]], limit: int = 80) -> dict[str, Any]:
        checked = list(records or [])[-limit:]
        penalties: list[float] = []
        rejected_take_orders = 0
        accepted_take_orders = 0
        for record in checked:
            if not isinstance(record, dict):
                continue
            act = record.get("action", {}) if isinstance(record.get("action"), dict) else {}
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            if act.get("action") == "take_order":
                if result.get("accepted") is False:
                    rejected_take_orders += 1
                elif result.get("accepted") is True:
                    accepted_take_orders += 1
            raw_penalty = result.get("preference_penalty") or result.get("penalty")
            try:
                penalties.append(float(raw_penalty))
            except (TypeError, ValueError):
                continue
        return {
            "checked_records": len(checked),
            "accepted_take_orders": accepted_take_orders,
            "rejected_take_orders": rejected_take_orders,
            "max_observed_penalty": max(penalties or [0.0]),
            "avg_observed_penalty": (sum(penalties) / len(penalties)) if penalties else 0.0,
        }
