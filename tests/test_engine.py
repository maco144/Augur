"""Tests for the Augur forecasting engine."""
from augur.engine import consensus_label, weighted_average


class TestWeightedAverage:

    def test_single_estimate_returns_its_probability(self):
        estimates = [{"probability": 0.7, "confidence": 0.8, "status": "success"}]
        prob, conf = weighted_average(estimates)
        assert abs(prob - 0.7) < 0.001
        assert abs(conf - 0.8) < 0.001

    def test_equal_confidence_is_simple_average(self):
        estimates = [
            {"probability": 0.6, "confidence": 0.5, "status": "success"},
            {"probability": 0.8, "confidence": 0.5, "status": "success"},
        ]
        prob, _ = weighted_average(estimates)
        assert abs(prob - 0.7) < 0.001

    def test_higher_confidence_weights_more(self):
        estimates = [
            {"probability": 0.9, "confidence": 0.9, "status": "success"},
            {"probability": 0.1, "confidence": 0.1, "status": "success"},
        ]
        prob, _ = weighted_average(estimates)
        assert prob > 0.7

    def test_failed_estimates_excluded(self):
        estimates = [
            {"probability": 0.9, "confidence": 0.9, "status": "success"},
            {"probability": 0.1, "confidence": 0.0, "status": "timeout"},
        ]
        prob, _ = weighted_average(estimates)
        assert abs(prob - 0.9) < 0.001

    def test_all_failed_returns_neutral(self):
        estimates = [{"probability": 0.5, "confidence": 0.0, "status": "timeout"}]
        prob, conf = weighted_average(estimates)
        assert prob == 0.5
        assert conf == 0.0


class TestConsensusLabel:

    def test_strongly_yes(self):
        assert consensus_label(0.85) == "strongly_yes"
        assert consensus_label(1.0) == "strongly_yes"
        assert consensus_label(0.80) == "strongly_yes"

    def test_lean_yes(self):
        assert consensus_label(0.65) == "lean_yes"
        assert consensus_label(0.60) == "lean_yes"

    def test_uncertain(self):
        assert consensus_label(0.5) == "uncertain"
        assert consensus_label(0.40) == "uncertain"

    def test_lean_no(self):
        assert consensus_label(0.35) == "lean_no"
        assert consensus_label(0.20) == "lean_no"

    def test_strongly_no(self):
        assert consensus_label(0.15) == "strongly_no"
        assert consensus_label(0.0) == "strongly_no"
