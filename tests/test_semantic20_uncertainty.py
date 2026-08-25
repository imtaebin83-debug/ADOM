from __future__ import annotations

import csv
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from adom.analysis.semantic20_uncertainty import (
    NUM_CLASSES,
    average_precision,
    binary_auroc,
    evaluate_threshold,
    load_manifest,
    run_analysis,
    score_maps,
    select_f1_threshold,
    stable_softmax,
)


class Semantic20UncertaintyTests(unittest.TestCase):
    def test_peaked_softmax_has_larger_variance_than_uniform(self) -> None:
        uniform_logits = np.zeros((NUM_CLASSES, 1, 1), dtype=np.float64)
        peaked_logits = uniform_logits.copy()
        peaked_logits[5, 0, 0] = 10.0
        uniform = stable_softmax(uniform_logits)
        peaked = stable_softmax(peaked_logits)
        self.assertGreater(float(np.var(peaked)), float(np.var(uniform)))

    def test_scores_are_oriented_toward_uncertainty(self) -> None:
        logits = np.zeros((NUM_CLASSES, 1, 2), dtype=np.float64)
        logits[5, 0, 0] = 10.0
        logits[5, 0, 1] = 0.1
        means = np.zeros(NUM_CLASSES)
        stds = np.ones(NUM_CLASSES)
        _, scores = score_maps(logits, means, stds)
        for name in (
            "entropy",
            "msp_uncertainty",
            "margin_uncertainty",
            "negative_softmax_variance",
            "negative_logit_variance",
        ):
            self.assertGreater(scores[name][0, 1], scores[name][0, 0], name)
        self.assertGreater(scores["sml_uncertainty"][0, 1], scores["sml_uncertainty"][0, 0])

    def test_binary_metrics_and_validation_threshold(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        positive = np.asarray([True, True, False, False])
        self.assertEqual(binary_auroc(scores, positive), 1.0)
        self.assertEqual(average_precision(scores, positive), 1.0)
        selected = select_f1_threshold(scores, positive)
        self.assertEqual(selected["threshold"], 0.8)
        evaluated = evaluate_threshold(scores, positive, selected["threshold"])
        self.assertEqual(evaluated["f1"], 1.0)

    def test_manifest_rejects_sequence_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logits = np.zeros((1, NUM_CLASSES, 2, 2), dtype=np.float32)
            np.save(root / "x.npy", logits)
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(root / "x.png")
            manifest = root / "manifest.csv"
            rows = [
                ["a", "same", "reference", "x.npy", "x.png"],
                ["b", "same", "validation", "x.npy", "x.png"],
                ["c", "test-seq", "test", "x.npy", "x.png"],
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["sample_id", "sequence_id", "split", "logits_path", "label_path"]
                )
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Sequence leakage"):
                load_manifest(manifest)

    def test_end_to_end_uses_validation_threshold_on_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            reference_logits = np.zeros((1, NUM_CLASSES, 1, NUM_CLASSES * 2))
            reference_label = np.zeros((1, NUM_CLASSES * 2), dtype=np.uint8)
            for class_id in range(NUM_CLASSES):
                for offset, value in enumerate((2.0, 3.0)):
                    column = class_id * 2 + offset
                    reference_logits[0, class_id, 0, column] = value
                    reference_label[0, column] = class_id
            np.save(root / "reference.npy", reference_logits)
            Image.fromarray(reference_label).save(root / "reference.png")

            def write_field(prefix: str) -> None:
                logits = np.zeros((1, NUM_CLASSES, 1, 4), dtype=np.float64)
                label = np.asarray([[0, 5, 1, 1]], dtype=np.uint8)
                logits[0, 5, 0, 0] = 0.5  # dirt -> sky, low max logit
                logits[0, 5, 0, 1] = 3.0  # sky -> sky
                logits[0, 1, 0, 2:] = 3.0
                np.save(root / f"{prefix}.npy", logits)
                Image.fromarray(label).save(root / f"{prefix}.png")

            write_field("validation")
            write_field("test")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["sample_id", "sequence_id", "split", "logits_path", "label_path"]
                )
                writer.writerows(
                    [
                        ["ref", "ref-seq", "reference", "reference.npy", "reference.png"],
                        ["val", "val-seq", "validation", "validation.npy", "validation.png"],
                        ["test", "test-seq", "test", "test.npy", "test.png"],
                    ]
                )
            output = root / "output"
            report = run_analysis(
                argparse.Namespace(
                    manifest=manifest,
                    output_dir=output,
                    true_class=0,
                    predicted_class=5,
                    temperature=1.0,
                    minimum_reference_pixels=2,
                    max_pixels_per_stratum_per_frame=100,
                    visualization_count=1,
                )
            )
            pair = report["results"]["dirt_to_sky"]["sml_uncertainty"]
            threshold = pair["validation"]["threshold_selection"]["threshold"]
            self.assertEqual(pair["test"]["fixed_validation_threshold"], threshold)
            self.assertEqual(pair["test"]["fixed_threshold_metrics"]["f1"], 1.0)
            self.assertTrue((output / "uncertainty-report.json").is_file())


if __name__ == "__main__":
    unittest.main()
