import unittest
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from scripts.ci.runpod_client import build_pod_payload
from scripts.ci.runpod_contract import (
    ContractError,
    configured_min_cuda_version,
    configured_gpu_tiers,
    extract_gpu_type_ids,
    resolve_local_ref,
    validate_min_cuda_version,
    validate_gpu_type_ids,
    validate_network_volume,
)


FIXTURES = Path(__file__).with_name("fixtures")
SCHEMA = json.loads((FIXTURES / "runpod_v2_openapi.json").read_text())
CATALOG = {"gpus": [{"id": "NVIDIA A40"}, {"id": "NVIDIA GeForce RTX 4090"}]}


class RunPodContractTests(unittest.TestCase):
    def test_minimum_cuda_version_requires_explicit_major_minor(self) -> None:
        self.assertEqual(configured_min_cuda_version({"min_cuda_version": "12.8"}), "12.8")
        for value in (None, [], ["12.8"], "12", "", "12.x"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                configured_min_cuda_version({"min_cuda_version": value})
        with self.assertRaises(ContractError):
            configured_min_cuda_version({"min_cuda_version": "12.8", "allowed_cuda_versions": ["13.0"]})

    def test_configured_gpu_tiers_preserve_order(self) -> None:
        tiers = configured_gpu_tiers(
            {
                "gpu_tiers": [
                    {
                        "name": "cheap",
                        "gpu_type_ids": ["NVIDIA A40"],
                    },
                    {
                        "name": "premium",
                        "gpu_type_ids": ["NVIDIA L40S"],
                    },
                ]
            }
        )
        self.assertEqual(
            tiers,
            [
                ("cheap", ("NVIDIA A40",)),
                ("premium", ("NVIDIA L40S",)),
            ],
        )

    def test_configured_gpu_tiers_reject_duplicates(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate GPU ID"):
            configured_gpu_tiers(
                {
                    "gpu_tiers": [
                        {
                            "name": "cheap",
                            "gpu_type_ids": ["NVIDIA A40"],
                        },
                        {
                            "name": "premium",
                            "gpu_type_ids": ["NVIDIA A40"],
                        },
                    ]
                }
            )

    def test_extract_gpu_ids_uses_catalog_not_schema_enum(self):
        self.assertEqual(
            extract_gpu_type_ids(CATALOG),
            frozenset({"NVIDIA A40", "NVIDIA GeForce RTX 4090"}),
        )
        validate_gpu_type_ids(SCHEMA, ["NVIDIA A40"], CATALOG)

    def test_invalid_catalog_ids_are_reported(self):
        with self.assertRaisesRegex(ContractError, "Tesla T4.*Unknown GPU"):
            validate_gpu_type_ids(SCHEMA, ["Tesla T4", "Unknown GPU"], CATALOG)

    def test_cuda_validation_checks_v2_pattern_not_old_enum(self):
        validate_min_cuda_version(SCHEMA, "12.8")
        with self.assertRaisesRegex(ContractError, "13"):
            validate_min_cuda_version(SCHEMA, "13")

    def test_missing_v2_post_schema_is_a_hard_error(self):
        with self.assertRaisesRegex(ContractError, "POST /v2/pods"):
            validate_gpu_type_ids({"paths": {}}, ["NVIDIA A40"], CATALOG)

    def test_non_local_reference_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "non-local"):
            resolve_local_ref(SCHEMA, "https://example.test/CreatePod")

    def test_malformed_catalog_is_rejected(self):
        for catalog in ({}, {"gpus": []}, {"gpus": [{"id": 7}]}):
            with self.assertRaises(ContractError):
                extract_gpu_type_ids(catalog)

    def test_generated_payload_matches_published_v2_schema(self):
        from scripts.ci.runpod_client import build_pod_payload

        config = json.loads(
            (Path(__file__).resolve().parents[1] / "runpod_config.json").read_text()
        )
        payload = build_pod_payload(config, "image", "startup", "jit", ["NVIDIA A40"])
        document = {**SCHEMA, "$ref": "#/components/schemas/CreatePodRequest"}
        Draft202012Validator(document).validate(payload)
        invalid = {**payload, "gpuTypeIds": ["NVIDIA A40"]}
        self.assertTrue(list(Draft202012Validator(document).iter_errors(invalid)))


class NetworkVolumePayloadTests(unittest.TestCase):
    """The JAX compilation cache lives on an optional network volume."""

    def config(self):
        return json.loads(
            (Path(__file__).resolve().parents[1] / "runpod_config.json").read_text()
        )

    def test_committed_configuration_attaches_no_volume(self):
        config = self.config()
        self.assertEqual(config["network_volume_id"], "")
        self.assertEqual(config["network_volume_mount_path"], "/opt/bench-cache")
        payload = build_pod_payload(config, "image", "startup", "jit", ["NVIDIA A40"])
        self.assertNotIn("network", payload["mounts"])
        self.assertEqual(payload["mounts"]["persistent"]["path"], config["volume_mount_path"])

    def test_configured_volume_replaces_the_persistent_mount(self):
        config = {**self.config(), "network_volume_id": "vol12345"}
        payload = build_pod_payload(config, "image", "startup", "jit", ["NVIDIA A40"])
        # REST v2 accepts at most one mount kind per pod.
        self.assertEqual(payload["mounts"],
                         {"network": [{"volumeId": "vol12345", "path": "/opt/bench-cache"}]})
        document = {**SCHEMA, "$ref": "#/components/schemas/CreatePodRequest"}
        Draft202012Validator(document).validate(payload)

    def test_explicit_mount_path_is_carried_through(self):
        config = {**self.config(), "network_volume_id": "vol12345",
                  "network_volume_mount_path": "/mnt/cache"}
        payload = build_pod_payload(config, "image", "startup", "jit", ["NVIDIA A40"])
        self.assertEqual(payload["mounts"]["network"][0]["path"], "/mnt/cache")


class NetworkVolumeContractTests(unittest.TestCase):
    def test_unset_volume_is_skipped(self):
        self.assertIsNone(validate_network_volume({"network_volume_id": ""}))
        self.assertIsNone(validate_network_volume({}))

    def test_malformed_volume_id_is_rejected(self):
        for value in ("not a volume", "vol/1234", "x" * 65):
            with self.subTest(value=value[:12]):
                with self.assertRaises(ContractError):
                    validate_network_volume({"network_volume_id": value})

    def test_volume_type_must_be_a_string(self):
        with self.assertRaises(ContractError):
            validate_network_volume({"network_volume_id": 7})

    def test_valid_volume_returns_id_and_mount_path(self):
        self.assertEqual(
            validate_network_volume({"network_volume_id": "vol12345"}),
            ("vol12345", "/opt/bench-cache"),
        )
        self.assertEqual(
            validate_network_volume({"network_volume_id": "vol12345",
                                     "network_volume_mount_path": "/mnt/cache"}),
            ("vol12345", "/mnt/cache"),
        )

    def test_mount_path_must_be_absolute_and_below_root(self):
        for path in ("cache", "/", ""):
            with self.assertRaises(ContractError):
                validate_network_volume({"network_volume_id": "vol12345",
                                         "network_volume_mount_path": path})



if __name__ == "__main__":
    unittest.main()
