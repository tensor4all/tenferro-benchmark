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


if __name__ == "__main__":
    unittest.main()
