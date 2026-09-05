import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import small_work
import small_work_provenance


def case(case_id, contract_id, family='core', surface='concrete', operation='add', phase='execution'):
    return {
        'id': case_id, 'contract_id': contract_id, 'family': family, 'surface': surface,
        'operation': operation, 'phase': phase, 'api_tier': 'concrete-fresh', 'backend': 'cpu',
        'dtype': 'f64', 'layout': 'contiguous', 'shape': [2], 'calls_per_workflow': 1,
        'workflow': 'x', 'setup': {'includes': [], 'excludes': []},
        'scope': {'timer': ['op'], 'outside_timer': []},
    }


class SmallWorkBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'docs/internals').mkdir(parents=True)
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts/check-public-boundary-inventory.py').write_text('')
        self.export = {'schema': 'tenferro.public-boundary-benchmarks.v1', 'cases': [
            {'id': 'core.add', 'family': 'core', 'operation': 'add', 'surface': 'concrete', 'phase': 'execution'}]}
        self.inventory = {'families': {'core': {'selectors': [
            {'operations': ['add'], 'cases': [{'id': 'core.add', 'surface': 'concrete'}],
             'surfaces': {'concrete': {'disposition': 'follow-up'}}}]}}}
        (self.root / 'docs/internals/public-boundary-benchmarks.json').write_text(json.dumps(self.export))
        (self.root / 'docs/internals/public-boundary-overhead-inventory.json').write_text(json.dumps(self.inventory))
        self.state = {'path': str(self.root), 'head': 'a' * 40, 'dirty': False, 'untracked': False}

    def tearDown(self):
        self.tmp.cleanup()

    def test_contract_requires_canonical_fields(self):
        value = case('x', 'core.add')
        for key in ('contract_id', 'family', 'surface'):
            missing = dict(value); missing.pop(key)
            with self.assertRaises(small_work.ContractError):
                small_work.CaseContract.from_mapping(missing)

    def test_four_cases_can_share_contract_reference(self):
        cases = [small_work.CaseContract.from_mapping(case(str(i), 'core.add')) for i in range(4)]
        with patch.object(small_work, 'run_inventory_checker', return_value={'returncode': 0}), \
             patch.object(small_work_provenance, '_git_state', return_value=self.state):
            snapshot = small_work.verify_canonical_binding(self.root, cases, check_inventory=True)
        self.assertEqual(snapshot['checkout']['head'], 'a' * 40)

    def test_unknown_and_unsupported_fail(self):
        unknown = small_work.CaseContract.from_mapping(case('x', 'core.nope'))
        with patch.object(small_work_provenance, '_git_state', return_value=self.state):
            with self.assertRaises(small_work.ContractError):
                small_work.verify_canonical_binding(self.root, [unknown], check_inventory=False)
        self.inventory['families']['core']['selectors'][0]['surfaces']['concrete']['disposition'] = 'unsupported'
        (self.root / 'docs/internals/public-boundary-overhead-inventory.json').write_text(json.dumps(self.inventory))
        valid = small_work.CaseContract.from_mapping(case('x', 'core.add'))
        with patch.object(small_work_provenance, '_git_state', return_value=self.state):
            with self.assertRaises(small_work.ContractError):
                small_work.verify_canonical_binding(self.root, [valid], check_inventory=False)

    def test_checker_failure_preserves_diagnostics(self):
        with patch('small_work.subprocess.run', side_effect=small_work.subprocess.TimeoutExpired('x', 1, output='out', stderr='err')):
            with self.assertRaises(small_work.ContractError) as raised:
                small_work.run_inventory_checker(self.root)
        self.assertEqual(raised.exception.details['stdout'], 'out')
        self.assertEqual(raised.exception.details['stderr'], 'err')

    def test_snapshot_rejects_checkout_change(self):
        valid = small_work.CaseContract.from_mapping(case('x', 'core.add'))
        with patch.object(small_work_provenance, '_git_state', return_value=self.state):
            snap = small_work.canonical_snapshot(self.root, [valid])
        changed = dict(self.state, head='b' * 40)
        with patch.object(small_work_provenance, '_git_state', return_value=changed):
            with self.assertRaises(small_work.ContractError):
                small_work.verify_canonical_snapshot(snap, self.root, [valid])


if __name__ == '__main__':
    unittest.main()
