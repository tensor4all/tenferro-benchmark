import json
import unittest
from unittest import mock

from scripts.ci.runpod_client import CreateResult, PermanentRunPodError
from scripts.ci.runpod_http import Cooldown, Response
from scripts.ci.runpod_provision import (
    PodState,
    ProvisionExhaustedError,
    _pod_state_checker,
    provision,
)
from scripts.ci.runpod_pricing import (
    candidate_plan,
    http_transport as pricing_transport,
)
from scripts.ci.tests.test_runpod_http import Clock


def live(runtime=True):
    return Response(
        200,
        {},
        json.dumps({'status': 'RUNNING', 'runtime': {} if runtime else None}).encode(),
    )


class PollingTests(unittest.TestCase):
    def checker(self, responses):
        clock = Clock()
        calls = []
        replies = iter(responses)

        def transport(request, *, timeout):
            calls.append((clock.now, timeout))
            return next(replies)

        return (
            _pod_state_checker(
                'https://example.test/v2/pods',
                'secret',
                monotonic=clock.monotonic,
                transport=transport,
            ),
            clock,
            calls,
        )

    def test_retry_after_cooldown_survives_candidate_change_and_success_resets_backoff(
        self,
    ):
        checker, clock, calls = self.checker(
            [
                Response(429, {'Retry-After': '40'}, b''),
                Response(503, {}, b''),
                live(),
                Response(503, {}, b''),
            ]
        )
        first = checker('pod-1', deadline=100)
        self.assertIsNone(first.status)
        self.assertGreaterEqual(first.retry_seconds, 40)
        # Even a fresh candidate may not query during the same endpoint cooldown.
        checker('pod-2', deadline=100)
        self.assertEqual(len(calls), 1)
        clock.sleep(first.retry_seconds)
        second = checker('pod-2', deadline=100)
        self.assertIsNone(second.has_runtime)
        clock.sleep(second.retry_seconds)
        self.assertEqual(checker('pod-2', deadline=100), PodState('RUNNING', True))
        third = checker('pod-2', deadline=100)
        self.assertLessEqual(third.retry_seconds, 12)

    def test_consecutive_failures_back_off_and_cap(self):
        checker, clock, _ = self.checker([Response(503, {}, b'')] * 7)
        delays = []
        for _ in range(7):
            state = checker('pod-1', deadline=1000)
            delays.append(state.retry_seconds)
            clock.sleep(state.retry_seconds)
        self.assertGreaterEqual(delays[1], 20)
        self.assertEqual(delays[-1], 60)

    def test_http_date_retry_after_in_polling(self):
        from email.utils import formatdate
        import time

        checker, _, _ = self.checker(
            [
                Response(
                    429,
                    {'Retry-After': formatdate(time.time() + 120, usegmt=True)},
                    b'',
                )
            ]
        )
        self.assertGreater(checker('pod-1', deadline=200).retry_seconds, 118)

    def test_observation_errors_never_report_container_death(self):
        for response in (
            Response(0, {}, b''),
            Response(200, {}, b'bad json'),
            Response(200, {}, b'{"errors":[{"message":"limited"}]}'),
            Response(200, {}, b'{"status":"RUNNING"}'),
        ):
            checker, _, _ = self.checker([response])
            state = checker('pod-1', deadline=100)
            self.assertIsNone(state.has_runtime)
            self.assertIsNone(state.status)

    def test_permanent_http_failure_is_explicit(self):
        checker, _, calls = self.checker([Response(401, {}, b'secret')])
        with self.assertRaisesRegex(PermanentRunPodError, 'HTTP 401'):
            checker('pod-1', deadline=100)
        self.assertEqual(len(calls), 1)

    def test_expired_deadline_makes_no_request_and_remaining_time_caps_timeout(self):
        checker, clock, calls = self.checker([live()])
        checker('pod-1', deadline=0)
        self.assertEqual(calls, [])
        checker('pod-1', deadline=3)
        self.assertEqual(calls, [(0, 3)])

    def run_provision(self, checker, clock, *, online=None, timeout=100, polls=10):
        deleted = []
        self.deleted = deleted
        result = provision(
            {
                'max_provision_attempts': 1,
                'startup_timeout_seconds': timeout,
                'startup_poll_seconds': polls,
            },
            [('test', ['GPU'])],
            label_prefix='label',
            mint_runner=lambda label: 'jit',
            create=lambda request, jit: CreateResult('pod-1', 'GPU', 'test', b'{}'),
            runner_online=online or (lambda label, **kwargs: True),
            pod_status=checker,
            delete_pod=lambda pod: (deleted.append(pod), True)[1],
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        return result, deleted

    def test_retry_after_longer_than_startup_budget_times_out_without_resend(self):
        checker, clock, calls = self.checker(
            [Response(429, {'Retry-After': '120'}, b'')]
        )
        with self.assertRaises(ProvisionExhaustedError):
            self.run_provision(checker, clock, timeout=30)
        self.assertEqual(self.deleted, ['pod-1'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(clock.now, 30)

    def test_v2_provisioning_is_not_accepted_before_running(self):
        checker, clock, calls = self.checker(
            [Response(200, {}, b'{"status":"PROVISIONING","runtime":null}'), live()]
        )
        online = mock.Mock(return_value=True)
        result, deleted = self.run_provision(checker, clock, online=online)
        self.assertEqual(result.pod_id, 'pod-1')
        self.assertEqual([call[0] for call in calls], [0, 10])
        self.assertEqual(online.call_count, 1)
        self.assertEqual(deleted, [])

    def test_v2_error_status_fails_fast_and_cleans_up(self):
        checker, clock, calls = self.checker(
            [Response(200, {}, b'{"status":"ERROR","runtime":null}')]
        )
        with self.assertRaisesRegex(ProvisionExhaustedError, 'status ERROR'):
            self.run_provision(checker, clock)
        self.assertEqual(clock.now, 0)
        self.assertEqual(self.deleted, ['pod-1'])

    def test_malformed_status_and_runtime_remain_unknown(self):
        for payload in (
            b'{"status":{},"runtime":null}',
            b'{"status":"RUNNING","runtime":"invalid"}',
        ):
            checker, clock, calls = self.checker([Response(200, {}, payload)])
            state = checker('pod-1', deadline=100)
            self.assertIsNone(state.status)
            self.assertIsNone(state.has_runtime)

    def test_recovery_restores_normal_poll_interval(self):
        checker, clock, calls = self.checker(
            [Response(429, {'Retry-After': '40'}, b''), live(), live()]
        )
        online_calls = []

        def online(label, **kwargs):
            online_calls.append(clock.now)
            return len(online_calls) == 2

        result, deleted = self.run_provision(checker, clock, online=online)
        self.assertEqual(result.pod_id, 'pod-1')
        self.assertEqual([call[0] for call in calls], [0, 40, 50])
        self.assertEqual(deleted, [])

    def test_container_death_after_transient_error_is_still_detected(self):
        checker, clock, calls = self.checker(
            [live(), Response(503, {}, b''), live(False)]
        )
        with self.assertRaisesRegex(ProvisionExhaustedError, 'container stopped'):
            self.run_provision(checker, clock, online=lambda label, **kwargs: False)
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.deleted, ['pod-1'])

    def test_permanent_poll_failure_cleans_up(self):
        checker, clock, _ = self.checker([Response(403, {}, b'')])
        with self.assertRaisesRegex(
            ProvisionExhaustedError, 'permanent pod state API failure'
        ):
            self.run_provision(checker, clock)
        self.assertEqual(self.deleted, ['pod-1'])

    def test_slow_observation_cannot_accept_a_runner_after_deadline(self):
        clock = Clock()

        def transport(request, *, timeout):
            clock.sleep(timeout)
            return live()

        checker = _pod_state_checker(
            'https://example.test/v2/pods',
            'key',
            monotonic=clock.monotonic,
            transport=transport,
        )
        online = mock.Mock(return_value=True)
        with self.assertRaises(ProvisionExhaustedError):
            self.run_provision(checker, clock, timeout=3, online=online)
        online.assert_not_called()
        self.assertEqual(self.deleted, ['pod-1'])

    def test_github_registry_request_uses_remaining_startup_budget(self):
        from scripts.ci.runpod_provision import _runner_online_checker

        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b'{"runners": []}'
        with (
            mock.patch('scripts.ci.runpod_provision.time.monotonic', return_value=10),
            mock.patch('urllib.request.urlopen', return_value=response) as urlopen,
        ):
            self.assertFalse(_runner_online_checker('key', 'org')('label', deadline=13))
        self.assertEqual(urlopen.call_args.kwargs['timeout'], 3)

    def test_runner_minting_keeps_its_independent_request_timeout(self):
        from scripts.ci.runpod_provision import _runner_minter

        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = (
            b'{"encoded_jit_config": "fixture"}'
        )
        with mock.patch('urllib.request.urlopen', return_value=response) as urlopen:
            self.assertEqual(_runner_minter('key', 'org', 4)('label'), 'fixture')
        self.assertEqual(urlopen.call_args.kwargs['timeout'], 30)

    def test_pricing_retry_after_is_shared_with_first_pod_observation(self):
        clock = Clock()
        cooldown = Cooldown()
        pricing_calls = []
        pod_calls = []

        def pricing_request(request, *, timeout):
            pricing_calls.append(clock.now)
            return Response(429, {'Retry-After': '120'}, b'')

        def pod_request(request, *, timeout):
            pod_calls.append(clock.now)
            return live()

        pricing = pricing_transport(
            'https://example.test/v2/catalog/gpus',
            api_key='key',
            cooldown=cooldown,
            monotonic=clock.monotonic,
            transport=pricing_request,
        )
        self.assertEqual(
            candidate_plan({}, [('static', ['GPU'])], transport=pricing),
            [('static', ['GPU'])],
        )
        checker = _pod_state_checker(
            'https://example.test/v2/pods',
            'key',
            cooldown=cooldown,
            monotonic=clock.monotonic,
            transport=pod_request,
        )
        state = checker('pod-1', deadline=420)
        self.assertEqual(state.retry_seconds, 120)
        self.assertEqual(pod_calls, [])
        # No other pricing query may bypass the endpoint cooldown either.
        pricing()
        self.assertEqual(pricing_calls, [0])
        clock.sleep(state.retry_seconds)
        self.assertEqual(checker('pod-1', deadline=420), PodState('RUNNING', True))
        self.assertEqual(pod_calls, [120])

    def test_pricing_is_one_query_then_static_fallback_without_retry_storm(self):
        calls = []

        def transport():
            calls.append('catalog')
            return 429, b''

        plan = candidate_plan({}, [('static', ['GPU'])], transport=transport)
        self.assertEqual(len(calls), 1)
        self.assertEqual(plan, [('static', ['GPU'])])


if __name__ == '__main__':
    unittest.main()
