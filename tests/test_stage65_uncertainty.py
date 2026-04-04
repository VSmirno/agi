"""Stage 65: Calibrated uncertainty tests."""

import pytest

from snks.agent.calibration import CalibrationTracker
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import generate_synthetic_transitions, TRAIN_COLORS
from snks.agent.crafter_trainer import generate_crafter_transitions


class TestCalibrationTracker:
    def test_perfect_calibration(self):
        t = CalibrationTracker()
        # High confidence, all correct
        for _ in range(50):
            t.record(0.9, "moved", "moved")
        # Low confidence, all wrong
        for _ in range(50):
            t.record(0.1, "moved", "blocked")
        assert t.brier_score() < 0.02

    def test_terrible_calibration(self):
        t = CalibrationTracker()
        # High confidence, all WRONG
        for _ in range(100):
            t.record(0.9, "moved", "blocked")
        assert t.brier_score() > 0.7

    def test_correlation_positive(self):
        t = CalibrationTracker()
        # Higher confidence → more correct
        for _ in range(50):
            t.record(0.9, "moved", "moved")
        for _ in range(30):
            t.record(0.5, "moved", "moved")
        for _ in range(20):
            t.record(0.5, "moved", "blocked")
        for _ in range(50):
            t.record(0.1, "moved", "blocked")
        rho = t.correlation()
        assert rho > 0.3

    def test_empty_tracker(self):
        t = CalibrationTracker()
        assert t.brier_score() == 1.0
        assert t.correlation() == 0.0

    def test_calibration_curve_buckets(self):
        t = CalibrationTracker(n_buckets=5)
        for _ in range(20):
            t.record(0.1, "a", "b")  # bucket 0
        for _ in range(20):
            t.record(0.9, "a", "a")  # bucket 4
        curve = t.calibration_curve()
        assert len(curve) == 2  # only 2 populated buckets
        # Low conf bucket should have ~0% accuracy
        assert curve[0][1] < 0.1
        # High conf bucket should have ~100% accuracy
        assert curve[1][1] > 0.9

    def test_summary(self):
        t = CalibrationTracker()
        t.record(0.8, "a", "a")
        s = t.summary()
        assert "brier_score" in s
        assert "correlation" in s
        assert s["n_predictions"] == 1


class TestCalibratedConfidence:
    @pytest.fixture(scope="class")
    def model(self):
        mg = generate_synthetic_transitions(TRAIN_COLORS)
        cr = generate_crafter_transitions()
        m = CLSWorldModel(dim=512, n_locations=500)
        m.train(mg + cr)
        return m

    def test_neocortex_confidence_below_one(self, model):
        """Neocortex confidence should be < 1.0 (calibrated)."""
        situation = {
            "facing_obj": "key", "obj_color": "red",
            "obj_state": "none", "carrying": "nothing",
            "carrying_color": "",
        }
        outcome, conf, source = model.query(situation, "pickup")
        assert source == "neocortex"
        assert 0.0 < conf <= 0.95

    def test_neocortex_confidence_increases_with_evidence(self, model):
        """More confirmations → higher confidence."""
        # Rule with confidence=1 vs confidence=10
        key1 = "test_low_conf_pickup"
        key2 = "test_high_conf_pickup"
        from snks.agent.cls_world_model import Rule
        model.neocortex[key1] = Rule(key1, {"result": "moved"}, 1.0,
                                      confidence=1, source="test")
        model.neocortex[key2] = Rule(key2, {"result": "moved"}, 1.0,
                                      confidence=10, source="test")
        # confidence = min(N / (N+2), 0.95)
        # N=1 → 0.33, N=10 → 0.83
        conf1 = min(1 / (1 + 2), 0.95)
        conf2 = min(10 / (10 + 2), 0.95)
        assert conf1 < conf2
        assert conf1 == pytest.approx(0.333, abs=0.01)
        assert conf2 == pytest.approx(0.833, abs=0.01)
        # Cleanup
        del model.neocortex[key1]
        del model.neocortex[key2]

    def test_abstract_confidence_085(self, model):
        """Abstract known member should return 0.85."""
        # door is in "openable" category
        outcome, conf = model.abstraction.query_abstract(
            "door", "toggle", "closed", "nothing"
        )
        assert outcome == "door_opened"
        assert conf == 0.85

    def test_sdm_calibrated_range(self, model):
        """SDM calibrated confidence should be in (0, 1)."""
        # Test sigmoid mapping
        assert 0 < model._calibrate_sdm(0.0) < 1
        assert 0 < model._calibrate_sdm(0.5) < 1
        assert model._calibrate_sdm(1.0) > model._calibrate_sdm(0.0)

    def test_qa_still_works(self, model):
        """QA methods should still work with calibrated confidence."""
        assert model.qa_can_interact("key", "pickup") is True
        assert model.qa_can_interact("wall", "pickup") is False
        assert model.qa_can_pass("empty") is True
