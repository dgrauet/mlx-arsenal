"""Tests for mlx_arsenal.attention.profile."""

import mlx.core as mx
import pytest

from mlx_arsenal.attention import Kind, classify


class TestClassify:
    def test_labels(self):
        scores = mx.array([[0.9, 0.1], [0.1, 0.9], [0.2, 0.2], [0.6, 0.7]])
        labels = classify(scores)
        # Last row: both over threshold → SPATIAL (tie-break).
        assert labels == [Kind.SPATIAL, Kind.TEMPORAL, Kind.OTHER, Kind.SPATIAL]

    def test_tie_break_spatial_wins(self):
        scores = mx.array([[0.8, 0.8]])
        assert classify(scores) == [Kind.SPATIAL]

    def test_thresholds_respected(self):
        scores = mx.array([[0.5, 0.0], [0.0, 0.5]])
        assert classify(scores) == [Kind.SPATIAL, Kind.TEMPORAL]
        assert classify(scores, spatial_threshold=0.51) == [Kind.OTHER, Kind.TEMPORAL]

    def test_validation_shape(self):
        with pytest.raises(ValueError):
            classify(mx.array([0.5, 0.5]))  # 1D
        with pytest.raises(ValueError):
            classify(mx.array([[0.5, 0.5, 0.5]]))  # second dim != 2

    def test_validation_thresholds(self):
        scores = mx.array([[0.5, 0.5]])
        with pytest.raises(ValueError):
            classify(scores, spatial_threshold=-0.1)
        with pytest.raises(ValueError):
            classify(scores, temporal_threshold=1.5)
