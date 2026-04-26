import pytest
from pydantic import ValidationError

from llm_values.types import VerdictFormat


def test_verdict_format_scale_rejects_mismatched_point_labels():
    with pytest.raises(ValidationError, match="point_labels must have"):
        VerdictFormat(type="scale", min=1, max=5, point_labels=["a", "b", "c"])  # 3 labels for 5-point scale


def test_verdict_format_scale_accepts_matched_point_labels():
    vf = VerdictFormat(type="scale", min=1, max=5, point_labels=["a", "b", "c", "d", "e"])
    assert vf.point_labels == ["a", "b", "c", "d", "e"]


def test_verdict_format_scale_accepts_no_point_labels():
    vf = VerdictFormat(type="scale", min=1, max=5)
    assert vf.point_labels is None


def test_verdict_format_binary_requires_two_options():
    with pytest.raises(ValidationError, match="binary verdict_format requires exactly 2 options"):
        VerdictFormat(type="binary", options=["only_one"])


def test_verdict_format_categorical_requires_options():
    with pytest.raises(ValidationError, match="categorical verdict_format requires"):
        VerdictFormat(type="categorical", options=["only_one"])
