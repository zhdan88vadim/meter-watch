from app.rate_limiter import SimpleRateLimiter


def test_first_call_passes():
    rl = SimpleRateLimiter(30, clock=lambda: 0.0)
    assert rl.can_save() is True


def test_second_call_blocked():
    t = {"v": 0.0}
    rl = SimpleRateLimiter(30, clock=lambda: t["v"])
    rl.can_save()
    t["v"] = 10
    assert rl.can_save() is False


def test_after_interval_passes():
    t = {"v": 0.0}
    rl = SimpleRateLimiter(30, clock=lambda: t["v"])
    rl.can_save()
    t["v"] = 31
    assert rl.can_save() is True
