from app.services.safety_monitor import SafetyMonitor
from app.domain.models import Thresholds

from app.protocol_models import AlertType
from tests.fakes import FakeNotifier, FakeStore
from meter_watch_shared.config import RedisKeys

REDIS_KEYS: RedisKeys = RedisKeys()

TH = Thresholds(
        person_is_active_threshold= 5,
        person_absence_threshold= 600,
        startup_person_timeout= 60,
        alert_cooldown= 30,
    )


def make_monitor():
    store = FakeStore()
    notifier = FakeNotifier()
    m = SafetyMonitor(
        store, notifier, REDIS_KEYS, TH, check_interval=1, clock=lambda: store.now
    )
    return m, store, notifier


def test_no_gas_no_alert():
    m, store, notifier = make_monitor()
    store.set(REDIS_KEYS.gas_flow, "0")
    m.check_once()
    assert notifier.alerts == []


def test_person_missing_sends_alert():
    m, store, notifier = make_monitor()
    store.set(REDIS_KEYS.gas_flow, "1")
    store.set_timestamp(REDIS_KEYS.human_last_seen)
    store.now = 700
    m.check_once()
    assert notifier.alerts == [AlertType.GAS_ALERT]
    assert store.exists(REDIS_KEYS.alert_triggered)


def test_cooldown_prevents_second_alert():
    m, store, notifier = make_monitor()
    store.set(REDIS_KEYS.gas_flow, "1")
    store.set(REDIS_KEYS.alert_cooldown, "1")
    store.set_timestamp(REDIS_KEYS.human_last_seen)
    store.now = 700
    m.check_once()
    assert notifier.alerts == []


def test_person_return_clears_alert():
    m, store, notifier = make_monitor()
    store.set(REDIS_KEYS.gas_flow, "1")
    store.set(REDIS_KEYS.alert_triggered, "1")
    store.set_timestamp(REDIS_KEYS.human_last_seen)
    store.now = 1
    m.check_once()
    assert not store.exists(REDIS_KEYS.alert_triggered)


def test_startup_exit_when_person_seen():
    m, store, notifier = make_monitor()
    store.set(REDIS_KEYS.gas_flow, "1")
    store.set(REDIS_KEYS.startup, "1")
    store.set_timestamp(REDIS_KEYS.human_last_seen)
    store.now = 10
    m.check_once()
    assert not store.exists(REDIS_KEYS.startup)
