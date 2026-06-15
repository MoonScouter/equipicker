import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from equipilot_app import _compute_position_size


class PositionSizingComputeTests(unittest.TestCase):
    def test_shares_and_cost_follow_atr_risk_formula(self) -> None:
        result = _compute_position_size(
            ticker="AAPL.US",
            total_portfolio_value=16000.0,
            atr_size=2.0,
            max_loss=160.0,
            atr_value=2.0,
            close_value=190.0,
        )
        # max_loss / (atr_size * atr) = 160 / (2 * 2) = 40 shares.
        self.assertIsNone(result["error"])
        self.assertEqual(result["shares_nb"], 40)
        self.assertAlmostEqual(result["estimated_cost"], 7600.0)
        self.assertFalse(result["over_budget"])

    def test_shares_are_floored_to_whole_units(self) -> None:
        result = _compute_position_size(
            ticker="X.US",
            total_portfolio_value=16000.0,
            atr_size=2.0,
            max_loss=170.0,
            atr_value=2.0,
            close_value=10.0,
        )
        # 170 / 4 = 42.5 -> floored to 42 shares.
        self.assertEqual(result["shares_nb"], 42)
        self.assertAlmostEqual(result["estimated_cost"], 420.0)

    def test_over_budget_flag_uses_105_percent_ceiling(self) -> None:
        result = _compute_position_size(
            ticker="Y.US",
            total_portfolio_value=1000.0,
            atr_size=1.0,
            max_loss=500.0,
            atr_value=1.0,
            close_value=50.0,
        )
        self.assertEqual(result["shares_nb"], 500)
        self.assertAlmostEqual(result["estimated_cost"], 25000.0)
        self.assertAlmostEqual(result["budget_ceiling"], 1050.0)
        self.assertTrue(result["over_budget"])

    def test_missing_atr_returns_error(self) -> None:
        result = _compute_position_size(
            ticker="Z.US",
            total_portfolio_value=16000.0,
            atr_size=2.0,
            max_loss=160.0,
            atr_value=np.nan,
            close_value=190.0,
        )
        self.assertIsNotNone(result["error"])
        self.assertNotIn("shares_nb", result)

    def test_missing_close_returns_error(self) -> None:
        result = _compute_position_size(
            ticker="Z.US",
            total_portfolio_value=16000.0,
            atr_size=2.0,
            max_loss=160.0,
            atr_value=2.0,
            close_value=np.nan,
        )
        self.assertIsNotNone(result["error"])


if __name__ == "__main__":
    unittest.main()
