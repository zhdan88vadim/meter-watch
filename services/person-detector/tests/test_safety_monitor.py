from app.services.safety_monitor import SafetyMonitor
from tests.fakes import FakeNotifier, FakeStore

KEYS = {
    "gas_flow": "gas",
    "startup": "startup",
    "alert_triggered": "alert",
    "alert_cooldown": "cooldown",
    "human_last_seen": "last_seen",
    "human_last_seen_str": "last_seen_str",
}

TH = {
    "person_is_active_threshold": 5,
    "person_absence_threshold": 600,
    "startup_person_timeout": 60,
    "alert_cooldown": 30,
}


def make_monitor():
    store = FakeStore()
    notifier = FakeNotifier()
    m = SafetyMonitor(
        store, notifier, KEYS, TH, check_interval=1, clock=lambda: store.now
    )
    return m, store, notifier


def test_no_gas_no_alert():
    m, store, notifier = make_monitor()
    store.set("gas", "0")
    m.check_once()
    assert notifier.alerts == []


def test_person_missing_sends_alert():
    m, store, notifier = make_monitor()
    store.set("gas", "1")
    store.set_timestamp("last_seen")
    store.now = 700
    m.check_once()
    assert notifier.alerts == ["gas_alert"]
    assert store.exists("alert")


def test_cooldown_prevents_second_alert():
    m, store, notifier = make_monitor()
    store.set("gas", "1")
    store.set("cooldown", "1")
    store.set_timestamp("last_seen")
    store.now = 700
    m.check_once()
    assert notifier.alerts == []


def test_person_return_clears_alert():
    m, store, notifier = make_monitor()
    store.set("gas", "1")
    store.set("alert", "1")
    store.set_timestamp("last_seen")
    store.now = 1
    m.check_once()
    assert not store.exists("alert")


def test_startup_exit_when_person_seen():
    m, store, notifier = make_monitor()
    store.set("gas", "1")
    store.set("startup", "1")
    store.set_timestamp("last_seen")
    store.now = 10
    m.check_once()
    assert not store.exists("startup")
