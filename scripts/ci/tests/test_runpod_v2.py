import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from jsonschema import Draft202012Validator

from scripts.ci.runpod_client import (
    CreateRequest,
    PermanentRunPodError,
    RetryableRunPodError,
    build_pod_payload,
    create_pod,
    parse_create_response,
)
from scripts.ci.runpod_contract import validate_gpu_type_ids, validate_min_cuda_version
from scripts.ci.runpod_http import Cooldown, Response, delete_pod, list_pods
from scripts.ci.runpod_pricing import candidate_plan, http_transport, parse_gpu_offers
from scripts.ci.runpod_provision import PodState, _pod_state_checker, parse_cost_per_hr
from scripts.ci.tests.test_runpod_http import Clock

FIXTURES = Path(__file__).with_name('fixtures')
ROOT = Path(__file__).resolve().parents[3]
CONFIG = json.loads((ROOT / 'scripts/ci/runpod_config.json').read_text())
SCHEMA = json.loads((FIXTURES / 'runpod_v2_openapi.json').read_text())


def fixture(name):
    return (FIXTURES / f'runpod_v2_{name}.json').read_bytes()


class V2FixtureTests(unittest.TestCase):
    def test_published_response_examples_validate_against_schema(self):
        for name, path, method, status in (
            ('create', '/v2/pods', 'post', '201'),
            ('get', '/v2/pods/{id}', 'get', '200'),
            ('list', '/v2/pods', 'get', '200'),
            ('catalog', '/v2/catalog/gpus', 'get', '200'),
        ):
            shape = SCHEMA['paths'][path][method]['responses'][status]['content'][
                'application/json'
            ]['schema']
            Draft202012Validator({**SCHEMA, **shape}).validate(
                json.loads(fixture(name))
            )

    def test_created_gpu_and_hourly_cost_use_v2_fields(self):
        result = parse_create_response(201, fixture('create'))
        self.assertEqual(result.gpu_type_id, 'NVIDIA GeForce RTX 4090')
        self.assertEqual(parse_cost_per_hr(result.body), 0.44)

    def test_catalog_replaces_pricing_query_and_enforces_allowlist(self):
        offers = parse_gpu_offers(
            fixture('catalog'), ['NVIDIA GeForce RTX 4090'], min_vram_gb=8
        )
        self.assertEqual(offers[0].price_per_hr, 0.44)
        self.assertEqual(offers[0].stock_status, 'HIGH')
        self.assertEqual(
            parse_gpu_offers(fixture('catalog'), ['NVIDIA A40'], min_vram_gb=8), []
        )

    def test_catalog_get_is_authenticated_and_requests_pod_cuda_availability(self):
        calls = []

        def transport(request, *, timeout):
            calls.append(request)
            return Response(200, {}, fixture('catalog'))

        send = http_transport(
            CONFIG['catalog_url'],
            api_key='fixture-key',
            config=CONFIG,
            transport=transport,
        )
        self.assertEqual(send()[0], 200)
        request = calls[0]
        self.assertEqual(request.method, 'GET')
        self.assertIsNone(request.data)
        self.assertEqual(request.get_header('Authorization'), 'Bearer fixture-key')
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(
            query,
            {
                'include': ['AVAILABILITY'],
                'product': ['POD'],
                'cloud': ['SECURE'],
                'count': ['1'],
                'minCudaVersion': ['12.8'],
            },
        )

    def test_static_fallback_is_expanded_into_single_gpu_requests(self):
        plan = candidate_plan(
            {}, [('static', ['GPU-A', 'GPU-B'])], transport=lambda: (429, b'')
        )
        self.assertEqual(plan, [('static', ['GPU-A']), ('static', ['GPU-B'])])
        for _, ids in plan:
            Draft202012Validator(
                {**SCHEMA, '$ref': '#/components/schemas/CreatePodRequest'}
            ).validate(build_pod_payload(CONFIG, 'image', 'startup', 'jit', ids))

    def test_multi_gpu_type_payload_is_rejected(self):
        with self.assertRaisesRegex(PermanentRunPodError, 'exactly one GPU type'):
            build_pod_payload(CONFIG, 'image', 'startup', 'jit', ['GPU-A', 'GPU-B'])

    def test_runtime_and_terminal_status_replace_state_query(self):
        replies = iter(
            [
                json.loads(fixture('get')),
                {**json.loads(fixture('get')), 'status': 'ERROR', 'runtime': None},
            ]
        )
        requests = []

        def transport(request, *, timeout):
            requests.append(request)
            return Response(200, {}, json.dumps(next(replies)).encode())

        checker = _pod_state_checker(CONFIG['api_url'], 'key', transport=transport)
        self.assertEqual(checker('pod-1'), PodState('RUNNING', True))
        self.assertEqual(checker('pod-1'), PodState('ERROR', False))
        self.assertEqual(requests[0].full_url, CONFIG['api_url'] + '/pod-1')
        self.assertEqual(requests[0].method, 'GET')
        self.assertIsNone(requests[0].data)

    def test_wrapped_pod_listing_follows_cursor_and_omits_sensitive_fields(self):
        page = json.loads(fixture('list'))
        first = {**page, 'pagination': {'hasNextPage': True, 'nextCursor': 'opaque+/='}}
        replies = iter([first, page])
        requests = []

        def transport(request, *, timeout):
            requests.append(request)
            return Response(200, {}, json.dumps(next(replies)).encode())

        pods = list_pods(CONFIG['api_url'], 'key', transport=transport)
        self.assertEqual(len(pods), 2)
        self.assertEqual(set(pods[0]), {'id', 'status', 'cost'})
        self.assertEqual(
            parse_qs(urlparse(requests[1].full_url).query), {'cursor': ['opaque+/=']}
        )
        self.assertNotIn('env', json.dumps(pods))

    def test_listing_rejects_bare_arrays_bad_pagination_and_cycles(self):
        for payload in (
            [],
            {'pods': []},
            {'pods': [], 'pagination': {'hasNextPage': True, 'nextCursor': None}},
        ):
            with self.assertRaises(ValueError):
                list_pods(
                    CONFIG['api_url'],
                    'key',
                    transport=lambda request, **kw: Response(
                        200, {}, json.dumps(payload).encode()
                    ),
                )
        cyclic = {'pods': [], 'pagination': {'hasNextPage': True, 'nextCursor': 'same'}}
        with self.assertRaisesRegex(ValueError, 'repeated cursor'):
            list_pods(
                CONFIG['api_url'],
                'key',
                transport=lambda request, **kw: Response(
                    200, {}, json.dumps(cyclic).encode()
                ),
            )

    def test_listing_deadline_respects_long_retry_after_without_resend(self):
        calls = []
        clock = Clock()

        def transport(request, *, timeout):
            calls.append(request)
            return Response(429, {'Retry-After': '300'}, b'')

        with self.assertRaises(ValueError):
            list_pods(
                CONFIG['api_url'],
                'key',
                deadline_seconds=30,
                transport=transport,
                sleep=clock.sleep,
                monotonic=clock.monotonic,
            )
        self.assertEqual(len(calls), 1)

    def test_v2_problem_capacity_moves_to_next_candidate_but_validation_does_not(self):
        requests = [
            CreateRequest('first', b'first'),
            CreateRequest('second', b'second'),
        ]
        for failure in (
            (
                400,
                {},
                b'{"title":"Bad Request","status":400,"detail":"No available machine"}',
            ),
            (
                403,
                {},
                b'{"title":"Forbidden","status":403,"detail":"pool inaccessible"}',
            ),
        ):
            responses = iter([failure, (201, {}, fixture('create'))])
            result = create_pod(
                CONFIG,
                requests,
                transport=lambda payload, timeout: next(responses),
                sleep=lambda seconds: self.fail(
                    'candidate rejection must not retry in place'
                ),
            )
            self.assertEqual(result.gpu_tier, 'second')
        from scripts.ci.runpod_client import PermanentRunPodError

        with self.assertRaises(PermanentRunPodError):
            create_pod(
                CONFIG,
                requests,
                transport=lambda payload, timeout: (
                    422,
                    {},
                    b'{"title":"Unprocessable Entity","status":422,"detail":"invalid body","errors":["gpu required"]}',
                ),
            )

    def test_create_rate_limit_cannot_be_bypassed_by_next_candidate_or_poll(self):
        clock = Clock()
        cooldown = Cooldown()
        calls = []

        def transport(payload, timeout):
            calls.append(payload)
            return (
                429,
                {'Retry-After': '120'},
                b'{"title":"Too Many Requests","status":429,"detail":"limited"}',
            )

        for name in ('first', 'second'):
            with self.assertRaises(RetryableRunPodError):
                create_pod(
                    CONFIG,
                    [CreateRequest(name, name.encode())],
                    cooldown=cooldown,
                    transport=transport,
                    sleep=clock.sleep,
                    monotonic=clock.monotonic,
                )
        self.assertEqual(calls, [b'first'])
        checker = _pod_state_checker(
            CONFIG['api_url'],
            'key',
            cooldown=cooldown,
            monotonic=clock.monotonic,
            transport=lambda *a, **kw: self.fail('cooldown must prevent GET'),
        )
        self.assertEqual(checker('pod-1', deadline=420).retry_seconds, 120)

    def test_delete_waits_for_shared_cooldown_before_request(self):
        clock = Clock()
        times = []

        def transport(request, *, timeout):
            times.append(clock.now)
            return Response(
                404, {}, b'{"title":"Not Found","status":404,"detail":"gone"}'
            )

        self.assertTrue(
            delete_pod(
                CONFIG['api_url'],
                'key',
                'pod-1',
                cooldown=Cooldown(30),
                monotonic=clock.monotonic,
                sleep=clock.sleep,
                transport=transport,
            )
        )
        self.assertEqual(times, [30])

    def test_contract_cli_validates_offline_schema_and_catalog_fixtures(self):
        config = {
            **CONFIG,
            'gpu_tiers': [
                {'name': 'fixture', 'gpu_type_ids': ['NVIDIA GeForce RTX 4090']}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.json'
            config_path.write_text(json.dumps(config))
            result = subprocess.run(
                [
                    'python3',
                    '-m',
                    'scripts.ci.runpod_contract',
                    '--config',
                    str(config_path),
                    '--schema-file',
                    str(FIXTURES / 'runpod_v2_openapi.json'),
                    '--catalog-file',
                    str(FIXTURES / 'runpod_v2_catalog.json'),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('CUDA version syntax is valid', result.stdout)
        self.assertIn('Live capacity is checked separately', result.stdout)

    def test_operational_paths_have_no_retired_api_dependency(self):
        config = json.dumps(CONFIG)
        self.assertNotIn('graphql', config.lower())
        self.assertNotIn('/v1', config)
        for name in (
            'runpod_client.py',
            'runpod_provision.py',
            'runpod_pricing.py',
            'runpod_http.py',
        ):
            source = (ROOT / 'scripts/ci' / name).read_text()
            self.assertNotIn('api.runpod.io/graphql', source)
            self.assertNotIn('rest.runpod.io/v1', source)


if __name__ == '__main__':
    unittest.main()
