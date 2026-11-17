import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import eval_metrics  # noqa: E402


def test_eval_metrics_known_values():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    mse, r2 = eval_metrics(y_true, y_pred)
    assert abs(mse - 0.375) < 1e-6
    assert abs(r2 - 0.9486081370449679) < 1e-6
