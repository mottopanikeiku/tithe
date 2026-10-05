from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignmenttax.calibration_stage import _plot_stage2_bars
from alignmenttax.presentation_artifacts import _plot_headline


class FigureLabelsTest(unittest.TestCase):
    def setUp(self) -> None:
        # Inspect the plotting calls without requiring the optional plotting dependency.
        matplotlib = ModuleType("matplotlib")
        pyplot = ModuleType("matplotlib.pyplot")
        matplotlib.use = Mock()
        matplotlib.pyplot = pyplot
        self.figure = Mock()
        self.axes = Mock()
        pyplot.subplots = Mock(return_value=(self.figure, self.axes))
        pyplot.close = Mock()
        modules = patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot})
        modules.start()
        self.addCleanup(modules.stop)

    def test_headline_titles_cover_improvements_worsening_and_ties(self) -> None:
        protocol = "shared_plain_ab_label"
        for accuracy_delta in (-0.1, 0.0, 0.1):
            for ece_delta in (-0.1, 0.0, 0.1):
                with self.subTest(accuracy_delta=accuracy_delta, ece_delta=ece_delta):
                    self.axes.reset_mock()
                    base = [0.5, 0.7, 0.2]
                    instruct = [0.5 + accuracy_delta, 0.7, 0.2 + ece_delta]
                    summary = {
                        (protocol, "base_accuracy"): base[0],
                        (protocol, "base_mean_confidence"): base[1],
                        (protocol, "base_ece"): base[2],
                        (protocol, "instruct_accuracy"): instruct[0],
                        (protocol, "instruct_mean_confidence"): instruct[1],
                        (protocol, "instruct_ece"): instruct[2],
                        (protocol, "delta_accuracy"): accuracy_delta,
                        (protocol, "delta_mean_confidence"): 0.0,
                        (protocol, "delta_overconfidence_gap"): -accuracy_delta,
                        (protocol, "delta_ece"): ece_delta,
                    }

                    _plot_headline(summary, Path("reports"))

                    self.assertEqual(
                        self.axes.set_title.call_args_list,
                        [
                            call("Base and instruct metrics on the shared prompt"),
                            call("Metric differences on the shared prompt (instruct - base)"),
                        ],
                    )
                    bars = self.axes.bar.call_args_list
                    self.assertEqual(bars[0].args[1], base)
                    self.assertEqual(bars[1].args[1], instruct)
                    self.assertEqual(bars[2].args[1], [accuracy_delta, 0.0, -accuracy_delta, ece_delta])
                    self.axes.set_ylabel.assert_any_call("Instruct - base")

    def test_calibration_title_covers_improvements_worsening_and_ties(self) -> None:
        metrics = ["mean_confidence", "overconfidence_gap", "brier", "nll", "ece"]
        raw = [0.7, 0.2, 0.3, 0.8, 0.2]
        for delta in (-0.1, 0.0, 0.1):
            with self.subTest(calibrated_minus_raw=delta):
                self.axes.reset_mock()
                calibrated = [value + delta for value in raw]
                summary_rows = [
                    {
                        "prompt_protocol": "shared_plain_ab_label",
                        "split": "test",
                        "model_key": "instruct",
                        "stage": stage,
                        **dict(zip(metrics, values)),
                    }
                    for stage, values in (("raw", raw), ("calibrated", calibrated))
                ]

                _plot_stage2_bars(summary_rows, Path("reports"))

                self.axes.set_title.assert_called_once_with(
                    "Raw and temperature-scaled instruct metrics on held-out test split"
                )
                bars = self.axes.bar.call_args_list
                self.assertEqual(bars[0].args[1], raw)
                self.assertEqual(bars[1].args[1], calibrated)
                self.assertEqual(bars[0].kwargs["label"], "Raw instruct")
                self.assertEqual(bars[1].kwargs["label"], "Temperature scaled")


if __name__ == "__main__":
    unittest.main()
