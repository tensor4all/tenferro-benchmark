"""JAX compilation cache: key identity, generation pruning, and row recording."""
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import jax_cache_stats  # noqa: E402
from scripts.ci import jax_cache  # noqa: E402

MANIFEST = ROOT / 'benchmarks/gpu/manifests/routine.yaml'


class CacheKeyTests(unittest.TestCase):
    def test_key_is_deterministic_and_pin_independent(self):
        matrix = jax_cache.case_matrix(MANIFEST)
        self.assertTrue(matrix)
        # The matrix is sorted, so the key cannot depend on manifest order.
        self.assertEqual(matrix, sorted(matrix))
        first = jax_cache.cache_key('0.10.1', '12.8', '570.1', matrix)
        self.assertEqual(jax_cache.cache_key('0.10.1', '12.8', '570.1', list(matrix)), first)
        with tempfile.TemporaryDirectory() as directory:
            pinned = Path(directory) / 'routine.yaml'
            document = yaml.safe_load(MANIFEST.read_text())
            for spec in document['suites'].values():
                # A tenferro re-pin must not invalidate compiled programs.
                spec['tenferro_pin'] = 'a' * 40
            pinned.write_text(yaml.safe_dump(document))
            pinned_matrix = jax_cache.case_matrix(pinned)
        self.assertEqual(pinned_matrix, matrix)
        self.assertEqual(jax_cache.cache_key('0.10.1', '12.8', '570.1', pinned_matrix), first)

    def test_every_compile_input_changes_the_key(self):
        matrix = jax_cache.case_matrix(MANIFEST)
        base = jax_cache.cache_key('0.10.1', '12.8', '570.1', matrix)
        self.assertNotEqual(jax_cache.cache_key('0.10.2', '12.8', '570.1', matrix), base)
        self.assertNotEqual(jax_cache.cache_key('0.10.1', '12.9', '570.1', matrix), base)
        self.assertNotEqual(jax_cache.cache_key('0.10.1', '12.8', '575.0', matrix), base)
        self.assertNotEqual(jax_cache.cache_key('0.10.1', '12.8', '570.1', matrix[:-1]), base)
        self.assertNotEqual(
            jax_cache.cache_key('0.10.1', '12.8', '570.1',
                                matrix + ['gpu/dense:new_case:jax-cuda:{}']), base)

    def test_key_requires_a_jax_version(self):
        with self.assertRaises(ValueError):
            jax_cache.cache_key('', '12.8', '570.1', [])

    def test_case_matrix_drops_non_compile_fields(self):
        """run/verify/notes cannot change a compiled program, so they stay out."""
        matrix = jax_cache.case_matrix(MANIFEST)
        self.assertTrue(matrix)
        self.assertNotIn('"warmups"', '\n'.join(matrix))
        self.assertNotIn('"rtol"', '\n'.join(matrix))


class ConfigureTests(unittest.TestCase):
    def test_unconfigured_volume_disables_the_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / 'github_env'
            result = jax_cache.configure('', directory, jax_version='0.10.1', github_env=env_file)
            self.assertEqual(result, {'enabled': False})
            self.assertFalse(env_file.exists())

    def test_configured_volume_exports_the_documented_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / 'github_env'
            result = jax_cache.configure('vol-1234', directory, jax_version='0.10.1',
                                         cuda_version='12.8', driver_version='570.1',
                                         github_env=env_file)
            self.assertTrue(result['enabled'])
            key = result['key']
            exported = dict(line.split('=', 1) for line in env_file.read_text().splitlines())
            self.assertEqual(exported['JAX_COMPILATION_CACHE_DIR'],
                             str(Path(directory) / 'jax' / key))
            self.assertEqual(exported['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'], '1')
            self.assertEqual(exported['JAX_PERSISTENT_CACHE_MAX_SIZE_BYTES'], str(4 * 1024 ** 3))
            self.assertEqual(exported['BENCH_JAX_CACHE_KEY'], key)
            self.assertTrue((Path(directory) / 'jax' / key).is_dir())

    def test_prune_keeps_the_current_and_newest_previous_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'jax'
            generations = ['newest', 'current', 'older']
            for offset, name in enumerate(generations):
                path = root / name
                path.mkdir(parents=True)
                (path / 'entry').write_text('x')
                stamp = time.time() - offset * 3600
                os.utime(path, (stamp, stamp))
            removed = jax_cache.prune_generations(root, 'current')
            self.assertEqual(removed, ['older'])
            self.assertEqual(sorted(entry.name for entry in root.iterdir()), ['current', 'newest'])
            # Deletion is per key directory; kept generations keep their content.
            self.assertTrue((root / 'newest' / 'entry').exists())

    def test_prune_tolerates_a_missing_root(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(jax_cache.prune_generations(Path(directory) / 'absent', 'current'), [])

    def test_an_unusable_volume_continues_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            not_a_volume = Path(directory) / 'file'
            not_a_volume.write_text('not a mount')
            status = jax_cache.main(['configure', '--volume-id', 'vol-1234',
                                     '--mount-path', str(not_a_volume)])
            self.assertEqual(status, 0)


class _Device:
    pass


DEVICE = _Device()


class _Array:
    """Stands in for a JAX array: placement, dtype, size and numpy conversion."""

    def __init__(self, value):
        self._value = np.asarray(value)

    def devices(self):
        return {DEVICE}

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def shape(self):
        return self._value.shape

    @property
    def size(self):
        return self._value.size

    def __array__(self, dtype=None):
        return self._value if dtype is None else self._value.astype(dtype)


class _Compiled:
    def __call__(self, *arguments):
        return _Array(np.ones((2, 2), dtype=np.float64))


class _Lowered:
    def __init__(self, sink=None):
        self._sink = sink

    def compile(self, compiler_options=None):
        time.sleep(0.02)  # a compile that costs something, like a real one
        if self._sink is not None:
            self._sink.write_text('compiled')
        return _Compiled()


class _Function:
    def __init__(self, sink=None):
        self._sink = sink

    def lower(self, *arguments):
        return _Lowered(self._sink)

    def __call__(self, *arguments):
        return _Array(np.ones((2, 2), dtype=np.float64))


def _fake_jax():
    module = types.ModuleType('jax')
    module.config = types.SimpleNamespace(update=lambda *args, **kwargs: None)
    module.devices = lambda kind=None: [DEVICE]
    module.block_until_ready = lambda value: value
    module.__version__ = '0.10.1'
    return module


class FirstCallRecordTests(unittest.TestCase):
    """The row carries first_call_ms, and the timed region stays untouched."""

    def run_case(self, *, cached, warmups=0, runs=1):
        import benchmark_gpu_python
        import gpu_framework_runner

        with tempfile.TemporaryDirectory() as directory:
            sink = Path(directory) / 'entry-cache' if cached else None
            function = _Function(sink)
            legacy = types.SimpleNamespace(
                _make_data_np=lambda problem: {'A': np.eye(2)},
                _to_jax=lambda data, device: dict(data),
                _to_jax_cpu=lambda data: dict(data),
                _jax_fn=lambda op, data: (function, ()),
                _verify_jax=lambda op, result, reference, data, rtol, atol: ('passed', 0.0, 0.0),
                _stub=lambda *args, **kwargs: {'status': 'stub'},
                _ok_record=benchmark_gpu_python._ok_record,
                _timing_stats=benchmark_gpu_python._timing_stats,
                _base_env=lambda *args: {},
            )
            problem = {'id': 'unit_case', 'op': 'matmul', 'run': {'warmups': warmups, 'runs': runs},
                       'verify': {'rtol': 1e-8, 'atol': 1e-10}}
            with mock.patch.dict(sys.modules, {'jax': _fake_jax()}), \
                    mock.patch.dict(os.environ, {'JAX_COMPILATION_CACHE_DIR': directory}):
                record = gpu_framework_runner.run(legacy, 'gpu/unit', problem, 'jax-cuda', 0,
                                                  ts='t', bc='b', tc='c')
        return record

    def test_first_call_ms_includes_compilation_and_leaves_timing_alone(self):
        record = self.run_case(cached=True)
        self.assertEqual(record['status'], 'ok')
        timing = record['timing']
        self.assertIn('first_call_ms', timing)
        self.assertIsNotNone(timing['first_call_ms'])
        self.assertGreaterEqual(timing['first_call_ms'], timing['compile_time_ms'])
        self.assertGreaterEqual(timing['compile_time_ms'], 20.0)
        # The sampled region is still reported unchanged.
        self.assertEqual(timing['warmup_runs'], 0)
        self.assertEqual(timing['timed_runs'], 1)
        self.assertEqual(timing['timing_scope'], 'steady_state_host_api_plus_device_sync')

    def test_the_row_records_the_untimed_verification_work(self):
        record = self.run_case(cached=True)
        timing = record['timing']
        self.assertIn('verification_time_ms', timing)
        self.assertGreaterEqual(timing['verification_time_ms'], 0.0)
        # Proving the numbers must not leak into the timed region.
        self.assertEqual(timing['timed_runs'], 1)
        self.assertEqual(timing['timing_scope'], 'steady_state_host_api_plus_device_sync')

    def test_first_call_is_reported_once_and_consistently(self):
        record = self.run_case(cached=False)
        timing = record['timing']
        notes = json.loads(record['execution']['notes'])
        self.assertEqual(notes['first_call_ms'], timing['first_call_ms'])
        self.assertEqual(notes['compile_time_ms'], timing['compile_time_ms'])
        # Every JAX arm compiles explicitly, so the cold path is never cheaper
        # than the compilation it includes.
        self.assertGreaterEqual(timing['first_call_ms'], timing['compile_time_ms'])

    def test_the_record_validates_against_the_result_schema(self):
        import jsonschema

        schema = json.loads((ROOT / 'schemas/benchmark-result.schema.json').read_text())
        record = self.run_case(cached=True)
        jsonschema.validate(record, schema)
        # first_call_ms stays optional: collectors that never measure a cold
        # path keep emitting valid rows.
        without = json.loads(json.dumps(record))
        without['timing'].pop('first_call_ms')
        without['timing'].pop('verification_time_ms')
        jsonschema.validate(without, schema)

    def test_cache_growth_is_recorded_as_a_delta_not_a_hit_claim(self):
        record = self.run_case(cached=True)
        notes = json.loads(record['execution']['notes'])
        self.assertEqual(notes['cache_entries_delta'], 1)
        self.assertGreater(notes['cache_bytes_delta'], 0)
        self.assertIn('not proof of one', notes['jax_compilation_cache_note'])

    def test_cache_statistics_are_absent_when_no_cache_is_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = jax_cache_stats.stats(Path(directory) / 'absent')
            self.assertEqual(entries['available'], False)
            self.assertEqual(jax_cache_stats.delta(entries, entries),
                             {'cache_bytes_delta': None, 'cache_entries_delta': None})


if __name__ == '__main__':
    unittest.main()
