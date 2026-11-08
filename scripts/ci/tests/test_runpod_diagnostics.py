import json
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from scripts.ci.runpod_diagnostics import (
    catalog_url, data_centers_url, diagnose, inspect_catalog, inspect_data_centers, summary,
)
from scripts.ci.runpod_http import Response


CONFIG = json.loads(Path("scripts/ci/runpod_config.json").read_text())
GPU = CONFIG["gpu_tiers"][-1]["gpu_type_ids"][0]
TIERS = [("a100", CONFIG["gpu_tiers"][-1]["gpu_type_ids"])]


class DiagnosticsTests(unittest.TestCase):
    def test_queries_differ_only_by_cuda_filter(self):
        filtered = parse_qs(urlsplit(catalog_url(CONFIG, cuda_filter=True)).query)
        plain = parse_qs(urlsplit(catalog_url(CONFIG, cuda_filter=False)).query)
        self.assertEqual(filtered.pop("minCudaVersion"), ["12.8"])
        self.assertEqual(filtered, plain)
        self.assertEqual(plain["cloud"], ["SECURE"])

    def test_reports_missing_and_out_of_stock_models_and_static_fallback(self):
        body = json.dumps({"gpus": [{
            "id": GPU, "secure": True, "availability": "NONE",
            "memory": 80, "price": {"secure": 1.5}, "env": "do not archive",
        }]}).encode()
        result = inspect_catalog(body, CONFIG, TIERS)
        self.assertEqual(result["gpus"][0]["availability"], "NONE")
        self.assertFalse(result["gpus"][1]["present"])
        self.assertEqual(result["candidate_plan"], [("a100", [gpu]) for gpu in TIERS[0][1]])
        self.assertNotIn("do not archive", json.dumps(result))

    def test_only_gets_and_preserves_errors_without_response_bodies(self):
        requests = []

        def fetch(request, **kwargs):
            requests.append(request)
            return Response(403, {"Authorization": "private"}, b"private account details")

        report = diagnose(CONFIG, "test-key", fetch=fetch)
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(request.get_method() == "GET" for request in requests))
        self.assertEqual(len(report["errors"]), 4)
        self.assertNotIn("private", json.dumps(report))
        self.assertNotIn("test-key", json.dumps(report))
        self.assertIn("HTTP 403", summary(report))

    def test_cuda_filtered_stock_can_differ_and_contract_is_checked(self):
        fixture = Path("scripts/ci/tests/fixtures/runpod_v2_openapi.json").read_bytes()

        def fetch(request, **kwargs):
            if "openapi" in request.full_url:
                return Response(200, {}, fixture)
            if "datacenters" in request.full_url:
                return Response(200, {}, json.dumps({"dataCenters": [
                    {"id": "US-TX-3", "networkVolumeTypes": ["STANDARD"],
                     "gpuAvailability": [{"id": TIERS[0][1][0], "availability": "HIGH"}]},
                ]}).encode())
            entries = [{"id": gpu, "secure": True, "memory": 80,
                        "price": {"secure": 1.5}, "availability": "HIGH"}
                       for gpu in TIERS[0][1]]
            if "minCudaVersion=" in request.full_url:
                entries = []
            return Response(200, {}, json.dumps({"gpus": entries}).encode())

        report = diagnose(CONFIG, "test-key", fetch=fetch)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["contract"].startswith("passed"))
        self.assertFalse(report["checks"]["production"]["gpus"][0]["present"])
        self.assertTrue(report["checks"]["without_cuda_filter"]["gpus"][0]["present"])

    def test_malformed_catalog_is_reported_and_next_check_still_runs(self):
        report = diagnose(CONFIG, "test-key", fetch=lambda *a, **kw: Response(200, {}, b"[]"))
        self.assertEqual(len(report["checks"]), 4)
        self.assertTrue(report["errors"])

    def test_data_centers_list_a100_stock_best_first(self):
        body = json.dumps({"dataCenters": [
            {"id": "EU-RO-1", "networkVolumeTypes": ["STANDARD"],
             "gpuAvailability": [{"id": GPU, "availability": "LOW"}]},
            {"id": "US-TX-3", "networkVolumeTypes": ["STANDARD"],
             "gpuAvailability": [{"id": GPU, "availability": "HIGH"}]},
            {"id": "AP-JP-1", "networkVolumeTypes": ["STANDARD"],
             "gpuAvailability": [{"id": "NVIDIA L40S", "availability": "HIGH"}]},
        ]}).encode()
        rows = inspect_data_centers(body, TIERS)["data_centers"]
        self.assertEqual([row["id"] for row in rows], ["US-TX-3", "EU-RO-1"])
        self.assertEqual(rows[0]["best_a100"], "HIGH")
        self.assertEqual(rows[0]["a100"], {GPU: "HIGH"})
        self.assertEqual(rows[1]["network_volume_types"], ["STANDARD"])

    def test_the_per_data_center_query_asks_for_gpu_availability(self):
        query = parse_qs(urlsplit(data_centers_url(CONFIG)).query)
        self.assertEqual(query, {"include": ["GPU_AVAILABILITY"]})
        self.assertTrue(data_centers_url(CONFIG).startswith("https://api.runpod.io/v2/catalog/datacenters"))


if __name__ == "__main__":
    unittest.main()
