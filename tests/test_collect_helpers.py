"""Pure helpers in the collection stage + hook-layer parsing."""

from __future__ import annotations

from ada.models.extraction import parse_layer_list
from ada.probe.collect import DEPTHS, find_last_subseq


def test_depths_schedule_matches_paper():
    # "truncate to 500 tokens, sample every 25" -> 21 depths (0..500).
    assert DEPTHS == list(range(0, 501, 25))
    assert len(DEPTHS) == 21 and DEPTHS[0] == 0 and DEPTHS[-1] == 500


def test_find_last_subseq():
    assert find_last_subseq([1, 2, 3, 2, 3], [2, 3]) == 3
    assert find_last_subseq([1, 2, 3], [9]) == -1
    assert find_last_subseq([1, 2, 3], []) == -1


def test_parse_layer_list_literal():
    assert parse_layer_list("13,15,19") == [13, 15, 19]
    assert parse_layer_list("15") == [15]
    assert parse_layer_list("13 15 19") == [13, 15, 19]


def test_parse_layer_list_all_requires_model():
    import pytest
    with pytest.raises(ValueError):
        parse_layer_list("all", model=None)
