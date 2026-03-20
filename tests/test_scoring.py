"""Unit tests for the scoring engine."""
from augur.scoring import (
    brier,
    combined_score,
    dark_multiplier,
    early_multiplier,
    reputation_weight,
    resolved_novelty,
    score_submission,
)


class TestBrier:
    def test_perfect_yes(self):
        assert brier(1.0, True) == 0.0

    def test_perfect_no(self):
        assert brier(0.0, False) == 0.0

    def test_worst_yes(self):
        assert brier(0.0, True) == 1.0

    def test_worst_no(self):
        assert brier(1.0, False) == 1.0

    def test_midpoint(self):
        assert brier(0.5, True) == 0.25
        assert brier(0.5, False) == 0.25

    def test_calibrated_estimate(self):
        score = brier(0.7, True)
        assert abs(score - 0.09) < 0.001


class TestResolvedNovelty:
    def test_correct_divergence_up(self):
        score = resolved_novelty(0.8, 0.5, True)
        assert abs(score - 0.3) < 0.001

    def test_correct_divergence_down(self):
        score = resolved_novelty(0.2, 0.5, False)
        assert abs(score - 0.3) < 0.001

    def test_wrong_divergence_up(self):
        assert resolved_novelty(0.8, 0.5, False) == 0.0

    def test_wrong_divergence_down(self):
        assert resolved_novelty(0.2, 0.5, True) == 0.0

    def test_no_divergence(self):
        assert resolved_novelty(0.5, 0.5, True) == 0.0
        assert resolved_novelty(0.5, 0.5, False) == 0.0


class TestDarkMultiplier:
    def test_zero_confidence(self):
        assert dark_multiplier(0.0) == 1.0

    def test_full_confidence(self):
        assert dark_multiplier(1.0) == 2.0

    def test_mid_confidence(self):
        assert abs(dark_multiplier(0.5) - 1.5) < 0.001


class TestEarlyMultiplier:
    def test_submit_at_creation(self):
        # Submit right when question is created = max bonus
        m = early_multiplier(submitted_at=100.0, question_created_at=100.0, deadline=200.0)
        assert abs(m - 2.0) < 0.001

    def test_submit_at_deadline(self):
        # Submit right at deadline = no bonus
        m = early_multiplier(submitted_at=200.0, question_created_at=100.0, deadline=200.0)
        assert abs(m - 1.0) < 0.001

    def test_submit_at_midpoint(self):
        m = early_multiplier(submitted_at=150.0, question_created_at=100.0, deadline=200.0)
        assert abs(m - 1.5) < 0.001

    def test_zero_window(self):
        m = early_multiplier(submitted_at=100.0, question_created_at=100.0, deadline=100.0)
        assert m == 1.0


class TestScoreSubmission:
    def test_dark_pool_applies_dark_multiplier(self):
        result = score_submission(
            probability=0.9, confidence=0.8, consensus_at_time=0.5,
            outcome=True, pool="dark",
        )
        assert result["pool_multiplier"] == 1.8  # 1.0 + 0.8
        assert result["brier_score"] == 0.01  # (0.9 - 1.0)^2
        assert result["novelty_score"] == 0.4  # diverged 0.4 in correct direction
        # combined = (0.7 * 0.99 + 0.3 * 0.4) * 1.8
        expected = round((0.7 * 0.99 + 0.3 * 0.4) * 1.8, 4)
        assert result["combined_score"] == expected

    def test_open_pool_applies_early_multiplier(self):
        result = score_submission(
            probability=0.9, confidence=0.8, consensus_at_time=0.5,
            outcome=True, pool="open",
            submitted_at=100.0, question_created_at=100.0, deadline=200.0,
        )
        assert result["pool_multiplier"] == 2.0  # submitted at creation

    def test_open_pool_late_submission(self):
        result = score_submission(
            probability=0.9, confidence=0.8, consensus_at_time=0.5,
            outcome=True, pool="open",
            submitted_at=200.0, question_created_at=100.0, deadline=200.0,
        )
        assert result["pool_multiplier"] == 1.0  # submitted at deadline


class TestCombinedScore:
    def test_perfect_calibration_no_novelty(self):
        score = combined_score(0.0, 0.0, alpha=0.7)
        assert abs(score - 0.7) < 0.001

    def test_perfect_both(self):
        score = combined_score(0.0, 1.0, alpha=0.7)
        assert abs(score - 1.0) < 0.001

    def test_worst_calibration(self):
        score = combined_score(1.0, 0.0, alpha=0.7)
        assert abs(score - 0.0) < 0.001

    def test_pure_novelty_mode(self):
        score = combined_score(0.5, 0.8, alpha=0.0)
        assert abs(score - 0.8) < 0.001


class TestReputationWeight:
    def test_new_participant(self):
        assert reputation_weight(0, None) == 1.0
        assert reputation_weight(4, 0.9) == 1.0

    def test_established_participant(self):
        w = reputation_weight(10, 0.85)
        assert abs(w - 0.85) < 0.001

    def test_floor(self):
        w = reputation_weight(10, 0.05)
        assert w == 0.1
