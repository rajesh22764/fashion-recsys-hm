import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from src.twotower import last_items, training_examples  # noqa: E402


def _df():
    # user 0 buys items 5, 6, 7 on days 0, 1, 2; user 1 buys item 8 once
    return pd.DataFrame({"u": [0, 0, 0, 1], "i": [5, 6, 7, 8], "timestamp": [0, 1, 2, 0]})


def test_last_items_are_oldest_first_and_left_padded():
    out = last_items(_df(), n_users=2, n_items=10, length=4)
    assert out[0].tolist() == [10, 5, 6, 7]
    assert out[1].tolist() == [10, 10, 10, 8]


def test_training_examples_use_only_earlier_purchases():
    hist, target = training_examples(_df(), n_items=10, length=3)
    # user 0 yields targets 6 and 7; user 1 has no earlier purchase so yields none
    assert target.tolist() == [6, 7]
    assert hist.tolist() == [[10, 10, 5], [10, 5, 6]]
