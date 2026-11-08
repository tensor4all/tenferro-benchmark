import contextlib
import io
import json
import unittest

from scripts.ci import runpod_network_volume as volume
from scripts.ci.runpod_http import Response


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def transport_replies(replies, calls):
    iter_replies = iter(replies)

    def transport(request, *, timeout):
        calls.append(request)
        return next(iter_replies)

    return transport


def listing(*volumes, has_next=False, cursor=None):
    return Response(200, {}, json.dumps({
        "networkVolumes": list(volumes),
        "pagination": {"hasNextPage": has_next, "nextCursor": cursor},
    }).encode())


VOLUME = {"id": "vol-1", "name": "jax-compile-cache", "size": 10, "dataCenter": "US-KS-2"}


class NetworkVolumeTests(unittest.TestCase):
    def run_listing(self, replies):
        clock, calls = Clock(), []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = volume.list_network_volumes(
                "https://example.test/v2", "secret",
                transport=transport_replies(replies, calls),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )
        return result, calls

    def test_list_reads_the_wrapped_array_and_the_fields_the_pipeline_needs(self):
        result, calls = self.run_listing([listing(VOLUME)])
        self.assertEqual(result, [VOLUME])
        self.assertEqual(calls[0].get_method(), "GET")
        self.assertEqual(calls[0].full_url, "https://example.test/v2/network-volumes")
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer secret")

    def test_list_follows_cursor_pagination_and_stops(self):
        result, calls = self.run_listing([
            listing(VOLUME, has_next=True, cursor="c1"),
            listing({"id": "vol-2", "name": "other", "size": 20, "dataCenter": "EU-RO-1"}),
        ])
        self.assertEqual([v["id"] for v in result], ["vol-1", "vol-2"])
        self.assertIn("cursor=c1", calls[1].full_url)

    def test_pagination_that_repeats_a_cursor_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_listing([
                listing(VOLUME, has_next=True, cursor="c1"),
                listing(VOLUME, has_next=True, cursor="c1"),
            ])

    def test_list_requires_the_wrapped_shape(self):
        clock, calls = Clock(), []
        with self.assertRaises(ValueError):
            volume.list_network_volumes(
                "https://example.test/v2", "secret",
                transport=transport_replies([Response(200, {}, b'{"volumes": []}')], calls),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )

    def test_create_posts_name_size_and_data_center_and_returns_the_record(self):
        clock, calls = Clock(), []
        created = dict(VOLUME, id="vol-new")
        payload = volume.create_network_volume(
            "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
            data_center="US-KS-2",
            transport=transport_replies([Response(201, {}, json.dumps(created).encode())], calls),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.assertEqual(payload["id"], "vol-new")
        self.assertEqual(calls[0].get_method(), "POST")
        self.assertEqual(json.loads(calls[0].data), {"name": "jax-compile-cache", "size": 10,
                                                     "dataCenter": "US-KS-2"})

    def test_a_data_center_is_required_because_the_binding_is_immutable(self):
        with self.assertRaises(ValueError):
            volume.create_network_volume("https://example.test/v2", "secret",
                                         name="x", size_gb=10, data_center="")

    def test_size_outside_the_provider_bounds_is_refused_before_the_request(self):
        for size in (9, 4097, 0, -10):
            with self.assertRaises(ValueError):
                volume.create_network_volume("https://example.test/v2", "secret",
                                             name="x", size_gb=size, data_center="US-KS-2")

    def test_creation_failure_reports_the_status_and_the_body(self):
        clock, calls = Clock(), []
        with self.assertRaises(ValueError) as error:
            volume.create_network_volume(
                "https://example.test/v2", "secret", name="x", size_gb=10, data_center="US-KS-2",
                transport=transport_replies([Response(400, {}, b'{"error":"no capacity"}')], calls),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )
        self.assertIn("400", str(error.exception))
        self.assertIn("no capacity", str(error.exception))

    def test_ensure_reuses_a_volume_with_the_same_name_instead_of_duplicating_it(self):
        clock, calls = Clock(), []
        result = volume.ensure_network_volume(
            "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
            data_center="US-KS-2",
            transport=transport_replies([listing(VOLUME)], calls),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.assertFalse(result["created"])
        self.assertEqual(result["volume"]["id"], "vol-1")
        self.assertEqual([c.get_method() for c in calls], ["GET"])

    def test_dry_run_describes_the_request_and_creates_nothing(self):
        clock, calls = Clock(), []
        result = volume.ensure_network_volume(
            "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
            data_center="US-KS-2", dry_run=True,
            transport=transport_replies([listing()], calls),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.assertFalse(result["created"])
        self.assertIsNone(result["volume"])
        self.assertEqual(result["request"]["dataCenter"], "US-KS-2")
        self.assertEqual([c.get_method() for c in calls], ["GET"])

    def test_ensure_creates_when_the_name_is_absent(self):
        clock, calls = Clock(), []
        result = volume.ensure_network_volume(
            "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
            data_center="US-KS-2",
            transport=transport_replies([listing(), Response(201, {}, json.dumps(VOLUME).encode())],
                                        calls),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.assertTrue(result["created"])
        self.assertEqual([c.get_method() for c in calls], ["GET", "POST"])

    def test_api_base_comes_from_the_pod_url_in_the_config(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runpod_config.json"
            path.write_text(json.dumps({"api_url": "https://api.runpod.io/v2/pods"}))
            self.assertEqual(volume._api_base_from_config(str(path), None), "https://api.runpod.io/v2")
            self.assertEqual(volume._api_base_from_config(str(path), "https://x/v9"),
                             "https://x/v9")
        self.assertEqual(volume._api_base_from_config("/nonexistent.json", None),
                         volume.DEFAULT_API_BASE)


CATALOG = json.dumps({
    "dataCenters": [
        {
            "id": "US-KS-2",
            "name": "US Kansas 2",
            "networkVolumeTypes": ["STANDARD"],
            "gpuAvailability": [
                {"id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM4", "availability": "LOW"},
            ],
        },
        {
            "id": "EU-RO-1",
            "name": "EU Romania 1",
            "networkVolumeTypes": ["STANDARD"],
            "gpuAvailability": [
                {"id": "NVIDIA GeForce RTX 4090", "name": "RTX 4090", "availability": "HIGH"},
            ],
        },
        {
            "id": "AP-JP-1",
            "name": "AP Japan 1",
            "networkVolumeTypes": [],
            "gpuAvailability": [
                {"id": "NVIDIA A100-SXM4-80GB", "name": "A100 SXM4", "availability": "HIGH"},
            ],
        },
    ]
}).encode()


class NetworkVolumeValidationTests(unittest.TestCase):
    """Input the provider would refuse must never reach it, on any path."""

    def test_dry_run_refuses_the_same_invalid_input_as_a_real_create(self):
        for kwargs in (
            {"name": "", "size_gb": 10, "data_center": "US-KS-2"},
            {"name": "x", "size_gb": 4097, "data_center": "US-KS-2"},
            {"name": "x", "size_gb": 10, "data_center": ""},
        ):
            calls = []
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                volume.ensure_network_volume(
                    "https://example.test/v2", "secret", dry_run=True,
                    transport=transport_replies([listing()], calls), **kwargs,
                )
            self.assertEqual(calls, [])

    def test_an_existing_volume_in_another_data_center_is_refused(self):
        elsewhere = dict(VOLUME, dataCenter="EU-RO-1")
        with self.assertRaisesRegex(ValueError, "data center"):
            volume.ensure_network_volume(
                "https://example.test/v2", "secret", name="jax-compile-cache",
                size_gb=10, data_center="US-KS-2",
                transport=transport_replies([listing(elsewhere)], []),
            )

    def test_an_existing_volume_with_another_size_is_refused(self):
        with self.assertRaisesRegex(ValueError, "size"):
            volume.ensure_network_volume(
                "https://example.test/v2", "secret", name="jax-compile-cache",
                size_gb=100, data_center="US-KS-2",
                transport=transport_replies([listing(VOLUME)], []),
            )

    def test_two_volumes_with_the_same_name_are_ambiguous_and_refused(self):
        duplicate = dict(VOLUME, id="vol-2")
        with self.assertRaisesRegex(ValueError, "vol-1, vol-2"):
            volume.ensure_network_volume(
                "https://example.test/v2", "secret", name="jax-compile-cache",
                size_gb=10, data_center="US-KS-2",
                transport=transport_replies([listing(VOLUME, duplicate)], []),
            )

    def test_a_next_page_without_a_cursor_is_refused(self):
        broken = Response(200, {}, json.dumps({
            "networkVolumes": [],
            "pagination": {"hasNextPage": True, "nextCursor": None},
        }).encode())
        with self.assertRaisesRegex(ValueError, "cursor"):
            volume.list_network_volumes(
                "https://example.test/v2", "secret",
                transport=transport_replies([broken], []),
            )


class NetworkVolumeCreationTests(unittest.TestCase):
    """A lost POST response must not become a second billable volume."""

    def test_an_ambiguous_failure_adopts_the_volume_once_it_is_visible(self):
        clock, calls = Clock(), []
        result = volume.create_network_volume(
            "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
            data_center="US-KS-2",
            transport=transport_replies(
                [Response(503, {}, b""), listing(), listing(VOLUME)], calls
            ),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.assertEqual(result["id"], "vol-1")
        # The POST is never repeated: only reads follow it.
        self.assertEqual([c.get_method() for c in calls], ["POST", "GET", "GET"])

    def test_an_ambiguous_failure_with_no_visible_volume_stops(self):
        clock, calls = Clock(), []
        with self.assertRaisesRegex(ValueError, "re-run `list`"):
            volume.create_network_volume(
                "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
                data_center="US-KS-2",
                transport=transport_replies([Response(503, {}, b"")] + [listing()] * 3, calls),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )
        self.assertEqual([c.get_method() for c in calls], ["POST", "GET", "GET", "GET"])

    def test_an_adopted_volume_must_have_the_requested_bindings(self):
        clock, calls = Clock(), []
        elsewhere = dict(VOLUME, dataCenter="EU-RO-1")
        with self.assertRaisesRegex(ValueError, "data center"):
            volume.create_network_volume(
                "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
                data_center="US-KS-2",
                transport=transport_replies(
                    [Response(503, {}, b""), listing(elsewhere)], calls
                ),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )

    def test_two_adopted_volumes_with_the_same_name_are_refused(self):
        clock, calls = Clock(), []
        with self.assertRaisesRegex(ValueError, "vol-1, vol-2"):
            volume.create_network_volume(
                "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
                data_center="US-KS-2",
                transport=transport_replies(
                    [Response(503, {}, b""), listing(VOLUME, dict(VOLUME, id="vol-2"))], calls
                ),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )

    def test_a_rate_limited_post_created_nothing_so_it_is_retried(self):
        clock, calls = Clock(), []
        result = volume.create_network_volume(
            "https://example.test/v2", "secret", name="jax-compile-cache", size_gb=10,
            data_center="US-KS-2",
            transport=transport_replies(
                [Response(429, {"Retry-After": "5"}, b""),
                 Response(201, {}, json.dumps(VOLUME).encode())],
                calls,
            ),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.assertEqual(result["id"], "vol-1")
        self.assertEqual([c.get_method() for c in calls], ["POST", "POST"])

    def test_the_error_body_cannot_leak_the_api_key(self):
        clock, calls = Clock(), []
        with self.assertRaises(ValueError) as error:
            volume.create_network_volume(
                "https://example.test/v2", "super-secret-key", name="x", size_gb=10,
                data_center="US-KS-2",
                transport=transport_replies(
                    [Response(400, {}, b'{"error":"bad key super-secret-key"}')], calls
                ),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )
        self.assertNotIn("super-secret-key", str(error.exception))
        self.assertIn("***redacted***", str(error.exception))


class NetworkVolumeReadTests(unittest.TestCase):
    def test_the_pod_url_yields_the_volume_api_base(self):
        self.assertEqual(
            volume.pod_api_base("https://api.runpod.io/v2/pods"), "https://api.runpod.io/v2"
        )
        self.assertEqual(volume.pod_api_base(""), volume.DEFAULT_API_BASE)

    def test_the_data_center_is_read_back_from_the_volume(self):
        clock, calls = Clock(), []
        self.assertEqual(
            volume.pod_data_center(
                "https://example.test/v2", "secret", "vol-1",
                transport=transport_replies(
                    [Response(200, {}, json.dumps(VOLUME).encode())], calls
                ),
                monotonic=clock.monotonic, sleep=clock.sleep,
            ),
            "US-KS-2",
        )
        self.assertEqual(calls[0].full_url, "https://example.test/v2/network-volumes/vol-1")

    def test_a_volume_without_a_data_center_is_an_error_not_a_guess(self):
        clock, calls = Clock(), []
        payload = dict(VOLUME)
        payload.pop("dataCenter")
        with self.assertRaisesRegex(ValueError, "data center"):
            volume.pod_data_center(
                "https://example.test/v2", "secret", "vol-1",
                transport=transport_replies(
                    [Response(200, {}, json.dumps(payload).encode())], calls
                ),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )


class DataCenterCatalogTests(unittest.TestCase):
    def check(self, data_center, gpu_ids=("NVIDIA A100-SXM4-80GB",)):
        clock, calls = Clock(), []
        result = volume.check_data_center(
            "https://example.test/v2", "secret", data_center,
            required_gpu_type_ids=gpu_ids,
            transport=transport_replies([Response(200, {}, CATALOG)], calls),
            monotonic=clock.monotonic, sleep=clock.sleep,
        )
        return result, calls

    def test_the_canonical_spelling_is_returned(self):
        self.assertEqual(self.check("us-ks-2")[0], "US-KS-2")

    def test_an_unknown_data_center_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not an available data center"):
            self.check("US-NOWHERE-9")

    def test_a_data_center_without_network_volumes_is_refused(self):
        with self.assertRaisesRegex(ValueError, "network volumes"):
            self.check("AP-JP-1")

    def test_a_data_center_without_the_benchmark_gpu_is_refused(self):
        with self.assertRaisesRegex(ValueError, "offers none of NVIDIA A100-SXM4-80GB"):
            self.check("EU-RO-1")
        # Any configured A100 variant is enough: a real data center offers a subset.
        self.assertEqual(
            self.check("US-KS-2", gpu_ids=("NVIDIA A100 80GB PCIe",
                                           "NVIDIA A100-SXM4-80GB"))[0],
            "US-KS-2",
        )
        with self.assertRaisesRegex(ValueError, "offers none of NVIDIA A100 80GB PCIe"):
            self.check("US-KS-2", gpu_ids=("NVIDIA A100 80GB PCIe",))


class NetworkVolumeDeletionTests(unittest.TestCase):
    def delete(self, replies, **kwargs):
        clock, calls = Clock(), []
        result = volume.delete_network_volume(
            "https://example.test/v2", "secret", "vol-1",
            transport=transport_replies(replies, calls),
            monotonic=clock.monotonic, sleep=clock.sleep, **kwargs,
        )
        return result, calls

    def test_the_name_is_confirmed_before_the_delete(self):
        result, calls = self.delete(
            [Response(200, {}, json.dumps(VOLUME).encode()), Response(204, {}, b"")],
            expect_name="jax-compile-cache",
        )
        self.assertTrue(result["deleted"])
        self.assertEqual([c.get_method() for c in calls], ["GET", "DELETE"])
        self.assertEqual(calls[1].full_url, "https://example.test/v2/network-volumes/vol-1")

    def test_a_name_mismatch_refuses_before_deleting(self):
        clock, calls = Clock(), []
        with self.assertRaisesRegex(ValueError, "refusing to delete"):
            volume.delete_network_volume(
                "https://example.test/v2", "secret", "vol-1", expect_name="other",
                transport=transport_replies(
                    [Response(200, {}, json.dumps(VOLUME).encode())], calls
                ),
                monotonic=clock.monotonic, sleep=clock.sleep,
            )
        self.assertEqual([c.get_method() for c in calls], ["GET"])

    def test_a_dry_run_only_reads(self):
        result, calls = self.delete(
            [Response(200, {}, json.dumps(VOLUME).encode())], dry_run=True
        )
        self.assertFalse(result["deleted"])
        self.assertEqual([c.get_method() for c in calls], ["GET"])

    def test_an_already_deleted_volume_is_not_an_error(self):
        result, calls = self.delete(
            [Response(200, {}, json.dumps(VOLUME).encode()), Response(404, {}, b"")]
        )
        self.assertFalse(result["deleted"])


if __name__ == "__main__":
    unittest.main()
