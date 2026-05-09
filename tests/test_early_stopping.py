"""Tests for early_stopping.py."""

from early_stopping import EarlyStopping


class TestEarlyStopping:
    def test_no_stop_while_improving(self):
        es = EarlyStopping(patience=3)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert es.step(loss) is False

    def test_stops_after_patience_exhausted(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)  # best
        es.step(1.1)  # worse 1
        es.step(1.2)  # worse 2
        assert es.step(1.3) is True  # worse 3 → stop

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        es.step(1.1)  # worse 1
        es.step(1.2)  # worse 2
        es.step(0.5)  # improved → reset
        assert es.counter == 0
        assert es.best_loss == 0.5

    def test_min_delta(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)
        # 0.95 is better but not by min_delta=0.1
        es.step(0.95)
        assert es.counter == 1
        # 0.89 is better by > 0.1
        es.step(0.89)
        assert es.counter == 0

    def test_patience_zero_stops_immediately(self):
        es = EarlyStopping(patience=0)
        es.step(1.0)  # sets best
        assert es.step(1.1) is True  # first non-improvement → stop

    def test_initial_state(self):
        es = EarlyStopping(patience=5)
        assert es.best_loss is None
        assert es.counter == 0
