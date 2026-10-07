"""
Pure decision logic. No I/O, no time, no Redis. Fully unit-testable.
"""
from __future__ import annotations

from domain.models import AlertAction, AlertDecision, SafetySnapshot


def decide_alert(snapshot: SafetySnapshot, thresholds: dict) -> AlertDecision:
    """
    thresholds keys:
        person_is_active_threshold
        person_absence_threshold
        startup_person_timeout
    """
    if not snapshot.gas_flowing:
        return AlertDecision(AlertAction.NONE, "gas not flowing")

    if snapshot.startup_active:
        secs = snapshot.seconds_since_last_seen
        if secs is not None and secs < thresholds["startup_person_timeout"]:
            return AlertDecision(AlertAction.EXIT_STARTUP, "person seen during startup")
        return AlertDecision(AlertAction.NONE, "startup waiting")

    secs = snapshot.seconds_since_last_seen
    if secs is not None and secs < thresholds["person_is_active_threshold"]:
        if snapshot.alert_active:
            return AlertDecision(AlertAction.CLEAR_ALERT, "person returned")
        return AlertDecision(AlertAction.NONE, "person present")

    if secs is None or secs >= thresholds["person_absence_threshold"]:
        if snapshot.alert_active:
            return AlertDecision(AlertAction.SKIP_ALREADY_ACTIVE, "alert already active")
        if snapshot.cooldown_active:
            return AlertDecision(AlertAction.SKIP_COOLDOWN, "cooldown active")
        return AlertDecision(AlertAction.SEND_ALERT, "person missing")

    return AlertDecision(AlertAction.NONE, "person missing but not critical")
