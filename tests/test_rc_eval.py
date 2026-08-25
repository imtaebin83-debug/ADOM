from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "rc_eval"


def load_module(name: str, filename: str):
    import sys

    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    previous_common = sys.modules.get("_common")
    if filename != "_common.py" and "common" in globals():
        sys.modules["_common"] = common
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_common is None:
            sys.modules.pop("_common", None)
        else:
            sys.modules["_common"] = previous_common
    return module


common = load_module("rc_common", "_common.py")
planner = load_module("rc_planner", "create_trial_plan.py")
analysis = load_module("rc_analysis", "analyze_trials.py")
logger = load_module("rc_logger", "start_trial_logger.py")


class RCEvalTests(unittest.TestCase):
    def test_balanced_deterministic_plan(self):
        first = planner.build_plan(7, 10)
        second = planner.build_plan(7, 10)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 40)
        counts = {}
        for row in first:
            key = (row["model"], row["hazard_present"])
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(set(counts.values()), {10})

    def test_wilson_interval_and_confusion_metrics(self):
        lower, upper = common.wilson_interval(5, 10)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            for index, (present, stopped, expected) in enumerate(
                ((True, True, "TP"), (True, False, "FN"), (False, True, "FP"), (False, False, "TN")), start=1
            ):
                trial = root / "trials" / f"T{index:03d}"
                trial.mkdir(parents=True)
                metadata = {
                    "trial_id": trial.name,
                    "operator": "tester",
                    "model": "b0-e0",
                    "scene_id": "fixed-scene",
                    "hazard_type": "log" if present else "none",
                    "hazard_present": present,
                    "start_position_marker": "A",
                    "commanded_speed_mps": 0.1,
                }
                annotation = {
                    "physical_stop_before_boundary": stopped,
                    "stop_decision_observed": stopped,
                    "hazard_detection_observed": present and expected == "TP",
                    "trial_completed": True,
                    "emergency_intervention": False,
                }
                (trial / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                (trial / "human_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
            result = analysis.analyze(root, root / "analysis")
            self.assertEqual(result["trial_counts"], {"TP": 1, "FN": 1, "FP": 1, "TN": 1})
            self.assertEqual(result["stop_success_rate"]["percent"], 50.0)
            self.assertEqual(result["miss_rate"]["percent"], 50.0)
            self.assertEqual(result["false_stop_rate"]["percent"], 50.0)
            self.assertEqual(result["perception_hazard_detection_rate"]["percent"], 50.0)

    def test_interrupted_trial_is_not_counted_as_model_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "experiment"
            trial = root / "trials" / "T001"
            trial.mkdir(parents=True)
            metadata = {
                "trial_id": "T001",
                "operator": "tester",
                "model": "b0-e0",
                "scene_id": "fixed-scene",
                "hazard_type": "log",
                "hazard_present": True,
                "start_position_marker": "A",
                "commanded_speed_mps": 0.1,
            }
            annotation = {
                "physical_stop_before_boundary": False,
                "stop_decision_observed": False,
                "trial_completed": False,
                "emergency_intervention": False,
            }
            (trial / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (trial / "human_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
            result = analysis.analyze(root, root / "analysis")
            self.assertEqual(result["trial_counts"], {"INTERRUPTED": 1})
            self.assertEqual(result["analyzed_trial_count"], 0)

    def test_unverified_topics_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text(json.dumps({"topics": {"camera": None}}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "BLOCKED_UNVERIFIED_TOPICS"):
                logger._load_topics(config)

    def test_malformed_metadata_is_rejected(self):
        errors = common.validate_metadata({"model": "unknown", "hazard_present": "maybe"})
        self.assertTrue(any("missing required field" in error for error in errors))
        self.assertTrue(any("model must" in error for error in errors))
        self.assertTrue(any("boolean" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
