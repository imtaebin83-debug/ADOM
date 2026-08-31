from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adom.runtime import b5_capacity_domain_contract, semantic20_cycle
from adom.runtime.b5_gate import (
    B5_GO_SCHEMA,
    FROZEN_EVALUATION_CONTRACT_SHA256,
    FROZEN_KOREAN_TEST_MANIFEST_SHA256,
    FROZEN_RELLIS_TEST_MANIFEST_SHA256,
    validate_b5_go_decision,
)
from adom.runtime.doctor import GPU_PROFILES, validate_gpu_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "adom" / "phase1_semantic20"
HAS_MMENGINE = importlib.util.find_spec("mmengine") is not None


def _decision_payload() -> dict:
    return {
        "schema_version": B5_GO_SCHEMA,
        "decision": "GO",
        "trigger": "abs_b2_difference_in_differences_ge_10pp",
        "primary_split": "matched-legacy-4568",
        "korean_heldout_used_for_selection": False,
        "metrics_pp": {
            "capacity_only_common_miou": 3.0,
            "b2_difference_in_differences": 12.0,
            "log_capacity_effect": -2.0,
            "rubble_capacity_effect": 8.0,
        },
        "provenance": {
            "b2_e0_checkpoint_sha256": "a" * 64,
            "b2_eadom_checkpoint_sha256": "b" * 64,
            "evaluation_contract_sha256": FROZEN_EVALUATION_CONTRACT_SHA256,
            "rellis_test_manifest_sha256": FROZEN_RELLIS_TEST_MANIFEST_SHA256,
            "korean_test_manifest_sha256": FROZEN_KOREAN_TEST_MANIFEST_SHA256,
        },
    }


class B5SourceContractTests(unittest.TestCase):
    def test_stage_configs_change_only_b2_model_base(self) -> None:
        for condition in ("e0_rellis", "eadom"):
            for stage in ("stage1", "stage2"):
                b2 = (
                    CONFIG_ROOT / f"segformer_b2_{stage}_{condition}.py"
                ).read_text(encoding="utf-8")
                b5 = (
                    CONFIG_ROOT / f"segformer_b5_{stage}_{condition}.py"
                ).read_text(encoding="utf-8")
                self.assertEqual(
                    b2.replace("models/segformer_b2.py", "models/segformer_b5.py"),
                    b5,
                )

    def test_official_b5_initialization_is_explicit(self) -> None:
        source = (
            CONFIG_ROOT / "_base_" / "models" / "segformer_b5.py"
        ).read_text(encoding="utf-8")
        self.assertIn("./segformer_b2.py", source)
        self.assertIn("mit_b5_20220624-658746d9.pth", source)
        self.assertIn("num_layers=[3, 6, 40, 3]", source)
        self.assertNotIn("B0-E0", source)
        self.assertNotIn("B2-E0", source)

    def test_architecture_allowlist_is_b2_to_b5_only(self) -> None:
        self.assertEqual(
            set(b5_capacity_domain_contract.B2_TO_B5_ARCHITECTURE_DIFFS),
            {
                "checkpoint",
                "model.backbone.init_cfg.checkpoint",
                "model.backbone.num_layers",
            },
        )

    def test_b5_runtime_scope_and_profile_batch_proposals(self) -> None:
        self.assertEqual(semantic20_cycle._requested_models("b5", "e0"), ["b5"])
        self.assertEqual(semantic20_cycle._requested_models("b5", "eadom"), ["b5"])
        with self.assertRaisesRegex(RuntimeError, "only for E0 and E-ADOM"):
            semantic20_cycle._requested_models("b5", "e1")
        with self.assertRaisesRegex(RuntimeError, "exact --gpu-profile"):
            semantic20_cycle._batch_candidates("b5")
        for profile, contract in GPU_PROFILES.items():
            candidates = semantic20_cycle._batch_candidates("b5", profile)
            self.assertEqual(candidates, contract["proposed_micro_batches"])
            self.assertEqual(
                [value * (16 // value) for value in candidates],
                [16] * len(candidates),
            )

    def test_canonical_source_split_and_mapping_lock(self) -> None:
        self.assertEqual(
            b5_capacity_domain_contract._canonical_source_contract(),
            b5_capacity_domain_contract.FROZEN_CANONICAL_SOURCE,
        )


class B5GpuProfileTests(unittest.TestCase):
    def test_profiles_distinguish_ambiguous_hardware_names_and_vram(self) -> None:
        cases = {
            "a100-40gb": ("NVIDIA A100-SXM4-40GB", 39.5, (8, 0)),
            "a100-80gb": ("NVIDIA A100-SXM4-80GB", 79.2, (8, 0)),
            "rtx-a6000-48gb": ("NVIDIA RTX A6000", 47.5, (8, 6)),
            "rtx-pro-6000-blackwell-96gb": (
                "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                95.5,
                (12, 0),
            ),
            "rtx-pro-4500-blackwell-32gb": (
                "NVIDIA RTX PRO 4500 Blackwell",
                31.86,
                (12, 0),
            ),
            "rtx-5090-32gb": ("NVIDIA GeForce RTX 5090", 31.84, (12, 0)),
        }
        for profile, (name, memory_gib, capability) in cases.items():
            with self.subTest(profile=profile):
                report = validate_gpu_profile(
                    profile,
                    name,
                    int(memory_gib * 1024**3),
                    capability,
                    [f"sm_{capability[0]}{capability[1]}"],
                )
                self.assertTrue(report["name_matches"])
                self.assertTrue(report["memory_matches"])
                self.assertTrue(report["compute_capability_matches"])
                self.assertTrue(report["native_arch_supported"])
                self.assertEqual(report["errors"], [])

    def test_profile_rejects_same_family_wrong_vram_or_model(self) -> None:
        wrong_vram = validate_gpu_profile(
            "a100-80gb", "NVIDIA A100-SXM4-80GB", int(39.5 * 1024**3)
        )
        self.assertFalse(wrong_vram["memory_matches"])
        wrong_model = validate_gpu_profile(
            "rtx-a6000-48gb", "NVIDIA RTX 6000 Ada Generation", int(47.5 * 1024**3)
        )
        self.assertFalse(wrong_model["name_matches"])

    def test_blackwell_profiles_reject_mig_or_wrong_product(self) -> None:
        mig = validate_gpu_profile(
            "rtx-pro-4500-blackwell-32gb",
            "NVIDIA RTX PRO 4500 Blackwell",
            int(15.9 * 1024**3),
            (12, 0),
            ["compute_90"],
        )
        self.assertFalse(mig["memory_matches"])
        wrong_product = validate_gpu_profile(
            "rtx-5090-32gb",
            "NVIDIA RTX PRO 4500 Blackwell",
            int(31.86 * 1024**3),
            (12, 0),
            ["compute_90"],
        )
        self.assertFalse(wrong_product["name_matches"])

    def test_blackwell_ptx_fallback_is_explicitly_provisional(self) -> None:
        report = validate_gpu_profile(
            "rtx-5090-32gb",
            "NVIDIA GeForce RTX 5090",
            int(31.84 * 1024**3),
            (12, 0),
            ["sm_90", "compute_90"],
        )
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["native_arch_supported"])
        self.assertRegex(report["warnings"][0], "PTX JIT compatibility")

    def test_profile_rejects_wrong_compute_capability(self) -> None:
        report = validate_gpu_profile(
            "rtx-5090-32gb",
            "NVIDIA GeForce RTX 5090",
            int(31.84 * 1024**3),
            (8, 9),
            ["sm_89"],
        )
        self.assertFalse(report["compute_capability_matches"])
        self.assertRegex(report["errors"][-1], "compute capability 12.0")


class B5GoDecisionTests(unittest.TestCase):
    def _write(self, root: Path, payload: dict) -> Path:
        path = root / "b5-go.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_go_decision_accepts_registered_threshold_and_frozen_manifests(self) -> None:
        replacements = {
            "FROZEN_EVALUATION_CONTRACT_SHA256": "c" * 64,
            "FROZEN_RELLIS_TEST_MANIFEST_SHA256": "d" * 64,
            "FROZEN_KOREAN_TEST_MANIFEST_SHA256": "e" * 64,
        }
        payload = _decision_payload()
        payload["provenance"].update(
            {
                "evaluation_contract_sha256": "c" * 64,
                "rellis_test_manifest_sha256": "d" * 64,
                "korean_test_manifest_sha256": "e" * 64,
            }
        )
        with patch.multiple(
            "adom.runtime.b5_gate", **replacements
        ), tempfile.TemporaryDirectory() as directory:
            report = validate_b5_go_decision(self._write(Path(directory), payload))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["decision"], "GO")

    def test_go_decision_rejects_test_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaked = _decision_payload()
            leaked["korean_heldout_used_for_selection"] = True
            with self.assertRaisesRegex(RuntimeError, "test-only"):
                validate_b5_go_decision(self._write(root, leaked))

    def test_go_decision_rejects_unsupported_trigger(self) -> None:
        replacements = {
            "FROZEN_EVALUATION_CONTRACT_SHA256": "c" * 64,
            "FROZEN_RELLIS_TEST_MANIFEST_SHA256": "d" * 64,
            "FROZEN_KOREAN_TEST_MANIFEST_SHA256": "e" * 64,
        }
        with patch.multiple(
            "adom.runtime.b5_gate", **replacements
        ), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            below_threshold = _decision_payload()
            below_threshold["provenance"].update(
                {
                    "evaluation_contract_sha256": "c" * 64,
                    "rellis_test_manifest_sha256": "d" * 64,
                    "korean_test_manifest_sha256": "e" * 64,
                }
            )
            below_threshold["metrics_pp"]["b2_difference_in_differences"] = 9.99
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                validate_b5_go_decision(self._write(root, below_threshold))

    def test_preserved_b2_evaluation_identifiers_block_b5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not a valid 64-character"):
                validate_b5_go_decision(
                    self._write(Path(directory), _decision_payload())
                )

    def test_preserved_primary_dataset_identifiers_block_static_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "preserved B2 primary dataset"):
                b5_capacity_domain_contract._frozen_primary_dataset(Path(directory))

    def test_committed_go_template_is_safe_no_go(self) -> None:
        template = json.loads(
            (
                REPO_ROOT
                / "experiments"
                / "segformer"
                / "b5-go-decision.template.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(template["schema_version"], B5_GO_SCHEMA)
        self.assertEqual(template["decision"], "NO_GO")
        self.assertTrue(
            all(
                value.startswith("REPLACE_WITH_")
                for value in template["provenance"].values()
            )
        )


class Stage2HandoffTests(unittest.TestCase):
    def test_stage2_handoff_requires_selected_stage1_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stage1 = output / "b5" / "stage1"
            stage1.mkdir(parents=True)
            checkpoint = stage1 / "best_clean_selection_iter_4000.pth"
            checkpoint.write_bytes(b"stage1")
            selection = {
                "schema_version": "semantic20-clean-v1",
                "rule": "RELLIS-val-only synthetic rule",
                "selected": {
                    "iteration": 4000,
                    "checkpoint": str(checkpoint.resolve()),
                },
            }
            (stage1 / "checkpoint_selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            report = semantic20_cycle.validate_stage2_handoff(
                output, "b5", checkpoint
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["stage2_load_from"], str(checkpoint.resolve()))

            other = stage1 / "iter_4000.pth"
            other.write_bytes(b"other")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                semantic20_cycle.validate_stage2_handoff(output, "b5", other)


@unittest.skipUnless(HAS_MMENGINE, "MMEngine config import runs in training image")
class B5ResolvedConfigTests(unittest.TestCase):
    def test_e0_and_eadom_are_architecture_only_and_recipe_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "ADOM_DATA_ROOT": Path(directory).as_posix(),
                "ADOM_MICRO_BATCH": "4",
                "ADOM_ACCUMULATIVE_COUNTS": "4",
                "ADOM_SEED": "42",
                "ADOM_DETERMINISTIC": "true",
            },
            clear=False,
        ):
            reports = [
                b5_capacity_domain_contract._stage_condition_contract(condition, stage)
                for condition in ("e0", "eadom")
                for stage in ("stage1", "stage2")
            ]
        self.assertEqual(len(reports), 4)
        for report in reports:
            self.assertEqual(report["checks"]["effective_batch"], 16)
            self.assertIsNone(report["checks"]["load_from"])
            self.assertTrue(report["checks"]["canonical_test_lock"])
            self.assertTrue(report["checks"]["rellis_val_selection"])


if __name__ == "__main__":
    unittest.main()
