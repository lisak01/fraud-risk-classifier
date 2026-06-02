"""
test_predict.py
----------------
Unit tests for the inference pipeline.
Tests run without a trained model by mocking the HuggingFace pipeline.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from predict import classify_text, LABEL_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_clf():
    """Mock HuggingFace pipeline that returns a fixed prediction."""
    clf = MagicMock()
    clf.return_value = [{"label": "FRAUD", "score": 0.93}]
    return clf


@pytest.fixture
def mock_clf_low_risk():
    """
    Fixture that simulates a classifier always predicting LOW_RISK with 98% confidence.
    Used for tests that specifically check low-risk behavior.
    """
    clf = MagicMock()
    clf.return_value = [{"label": "LOW_RISK", "score": 0.98}]
    return clf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestClassifyText:
    def test_returns_expected_keys(self, mock_clf):
        """Checks that the result dictionary contains all required keys."""
        result = classify_text("Suspicious transfer detected.", mock_clf)
        assert "label" in result
        assert "confidence" in result
        assert "description" in result
        assert "text" in result

    def test_label_is_valid(self, mock_clf):
        """Checks that the predicted label is one of the 5 defined risk categories."""
        result = classify_text("Suspicious transfer detected.", mock_clf)
        assert result["label"] in LABEL_MAP.keys()

    def test_confidence_is_float_in_range(self, mock_clf):
        """Checks that confidence score is a float between 0.0 and 1.0."""
        result = classify_text("Test text.", mock_clf)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_fraud_label(self, mock_clf):
        """Checks that a fraud-related text returns the FRAUD label."""
        result = classify_text("Unauthorized wire transfer to unknown account.", mock_clf)
        assert result["label"] == "FRAUD"

    def test_low_risk_label(self, mock_clf_low_risk):
        """Checks that a normal banking text returns the LOW_RISK label."""
        result = classify_text("Quarterly dividend payment processed.", mock_clf_low_risk)
        assert result["label"] == "LOW_RISK"

    def test_description_maps_correctly(self, mock_clf):
        """Checks that the description matches the expected value from LABEL_MAP."""
        result = classify_text("Phishing attempt on corporate account.", mock_clf)
        assert result["description"] == LABEL_MAP["FRAUD"]

    def test_empty_string_does_not_crash(self, mock_clf):
        """Edge case: empty input should not raise an exception."""
        mock_clf.return_value = [{"label": "LOW_RISK", "score": 0.51}]
        result = classify_text("", mock_clf)
        assert result is not None

    def test_very_long_text(self, mock_clf):
        """Edge case: very long text should not crash (truncation handled by pipeline)."""
        long_text = "suspicious transaction " * 200
        result = classify_text(long_text, mock_clf)
        assert result["label"] in LABEL_MAP.keys()


class TestLabelMap:
    def test_all_five_labels_present(self):
        """Checks that LABEL_MAP contains exactly the 5 expected risk categories."""
        expected = {"FRAUD", "AML_RISK", "CREDIT_RISK", "OPERATIONAL_RISK", "LOW_RISK"}
        assert set(LABEL_MAP.keys()) == expected

    def test_descriptions_are_non_empty(self):
        """Checks that every label has a non-empty description string."""
        for label, desc in LABEL_MAP.items():
            assert isinstance(desc, str) and len(desc) > 0, f"Empty description for {label}"
