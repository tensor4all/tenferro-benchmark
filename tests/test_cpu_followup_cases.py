"""Failures and stale references must never become performance evidence."""
import ctypes
import json
import os
import runpy
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import cpu_followup_cases as adapter
import generate_perf_issue_cases as generator
import mkl_fft

class FollowupTest(unittest.TestCase):
    def test_mismatching_reference_prevents_collection(self):
        sig=dict(shape=[1],count=1,sum=[1.,0.],sum_abs=1.,sum_sq=1.,probes=[[0,1.,0.]])
        bad=dict(sig,sum=[9.,0.],probes=[[0,9.,0.]])
        case=next(c for c in generator.generate_cases() if c['kind']=='cpu_followup_mwe')
        with patch.object(adapter,'execute',side_effect=[dict(status='ok',outputs=[sig]),dict(status='ok',outputs=[bad])]) as execute:
            result=adapter.run(case,1,dict(min_runtime_ms=10,runs=15))
        self.assertEqual(execute.call_count,2)
        self.assertEqual(result['correctness_status'],'failed')
        self.assertEqual(result['samples'],[])

    def test_skipped_reference_prevents_rust_and_sampling(self):
        case=next(c for c in generator.generate_cases() if c['kind']=='cpu_followup_mwe')
        skipped=dict(status='skipped',correctness_status='skipped',samples=[],skip_reason='no MKL runtime')
        with patch.object(adapter,'execute',return_value=skipped) as execute:
            result=adapter.run(case,1,dict(min_runtime_ms=10,runs=15))
        self.assertEqual(execute.call_count,1)
        self.assertEqual(result,dict(correctness_status='skipped',samples=[],skip_reason='no MKL runtime'))

    def test_pending_rust_signature_check_still_gets_validated(self):
        sig=dict(shape=[1],count=1,sum=[1.,0.],sum_abs=1.,sum_sq=1.,probes=[[0,1.,0.]])
        # The standalone Rust MWE emits this marker before the adapter verifies it.
        payload=dict(correctness_status='awaiting cross-implementation signature check',outputs=[sig],samples=[])
        case=next(c for c in generator.generate_cases() if c['kind']=='cpu_followup_mwe')
        with patch.object(adapter.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(payload),'')):
            self.assertEqual(adapter.execute('fft.fft','eager-shared',1,0,1)['status'],'ok')
            self.assertEqual(adapter.run(case,1,dict(min_runtime_ms=10,runs=0),True)['correctness_status'],'passed')

    def test_execute_preserves_skip_and_rejects_process_failure(self):
        payload=dict(correctness_status='skipped',samples=[],skip_reason='no MKL runtime')
        with patch.object(adapter.subprocess,'run',return_value=subprocess.CompletedProcess([],0,json.dumps(payload),'clear')):
            self.assertEqual(adapter.execute('fft.fft','mkl-dfti',1,0,1)['status'],'skipped')
        with patch.object(adapter.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','DFTI failed')):
            with self.assertRaisesRegex(RuntimeError,'DFTI failed'):
                adapter.execute('fft.fft','mkl-dfti',1,0,1)

    def test_standalone_campaign_preserves_skips_without_running_rust(self):
        skipped=dict(correctness_status='skipped',samples=[],skip_reason='no MKL runtime')
        commands=[]
        def run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command,0,json.dumps(skipped),'')
        with tempfile.TemporaryDirectory() as tmp, patch.object(sys,'argv',[
                'collect.py','--container','fixture','--output',tmp,'--case','fft.fft']), patch.object(
                subprocess,'check_output',return_value='fixture\n'), patch.object(subprocess,'run',side_effect=run):
            runpy.run_path(str(ROOT/'mwe/cpu_followup/collect.py'),run_name='__main__')
            rows=json.loads((Path(tmp)/'scan.json').read_text())
            self.assertTrue(rows)
            self.assertTrue(all(row['status']=='skipped' and row['reason']=='no MKL runtime' for row in rows))
            docker=[c for c in commands if c[0]=='docker']
            self.assertEqual(len(docker),2)  # one reference at each thread count
            self.assertTrue(all('reference.py' in c[-1] and 'mkl-dfti' in c[-1] for c in docker))

    def test_confirmation_failure_takes_precedence_over_missing_reference(self):
        skipped=dict(correctness_status='skipped',samples=[],skip_reason='no MKL runtime')
        for rust_fails in (False,True):
            with self.subTest(rust_fails=rust_fails), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp)/'scan.json').write_text(json.dumps([dict(
                    case_id='fft.fft',correct=True,ratio=2.,threads=1,path='eager-shared')]))
                def run(command, **kwargs):
                    shell=command[-1]
                    if command[0]!='docker':
                        return subprocess.CompletedProcess(command,0,'','')
                    if 'reference.py' in shell:
                        return subprocess.CompletedProcess(command,0,json.dumps(skipped),'')
                    payload=dict(correctness_status='awaiting cross-implementation signature check',
                                 samples=[dict(elapsed_ns=1,iterations=1)])
                    return subprocess.CompletedProcess(command,1 if rust_fails else 0,json.dumps(payload),'Rust failed')
                with patch.object(sys,'argv',['collect.py','--container','fixture','--output',tmp,
                        '--phase','confirm','--case','fft.fft']), patch.object(subprocess,'check_output',
                        return_value='fixture\n'), patch.object(subprocess,'run',side_effect=run):
                    runpy.run_path(str(ROOT/'mwe/cpu_followup/collect.py'),run_name='__main__')
                row=json.loads((Path(tmp)/'confirm.json').read_text())[0]
                self.assertEqual(row['status'],'failed' if rust_fails else 'skipped')


class JuliaDiscoveryTest(unittest.TestCase):
    def test_env_then_path_then_home(self):
        with patch.dict(os.environ,{'JULIA':'/explicit/julia'}), patch.object(adapter.shutil,'which',return_value='/explicit/julia') as which:
            self.assertEqual(adapter.command('case','julia-base',1,0)[0],'/explicit/julia')
            which.assert_called_once_with('/explicit/julia')
        with patch.dict(os.environ,{'JULIA':''}), patch.object(adapter.shutil,'which',return_value='/path/julia'):
            self.assertEqual(adapter.resolve_julia(),'/path/julia')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'JULIA':''}), patch.object(adapter.shutil,'which',return_value=None), patch.object(adapter.Path,'home',return_value=Path(tmp)):
            julia=Path(tmp)/'.juliaup/bin/julia'
            julia.parent.mkdir(parents=True)
            julia.write_text('#!/bin/sh\nexit 0\n')
            with self.assertRaisesRegex(RuntimeError,'julia not found: set JULIA or install juliaup'):
                adapter.resolve_julia()
            julia.chmod(0o755)
            self.assertEqual(adapter.resolve_julia(),str(julia))
            julia.unlink()
            with self.assertRaisesRegex(RuntimeError,'julia not found: set JULIA or install juliaup'):
                adapter.resolve_julia()
        with patch.dict(os.environ,{'JULIA':'/invalid'}), patch.object(adapter.shutil,'which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'JULIA=.*not executable'):
                adapter.resolve_julia()


class MklDiscoveryTest(unittest.TestCase):
    def test_order_fallback_and_environment_is_unchanged(self):
        candidates=['/explicit/lib/libmkl_rt.so','/explicit/lib/intel64/libmkl_rt.so',
                    'loader-mkl', '/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_rt.so',
                    '/usr/lib/x86_64-linux-gnu/libmkl_rt.so']
        for found in candidates:
            with self.subTest(found=found), patch.dict(os.environ,{'MKLROOT':'/explicit'}), patch.object(ctypes.util,'find_library',return_value='loader-mkl'), patch.object(mkl_fft.C,'CDLL') as load:
                before=dict(os.environ)
                sentinel=object()
                load.side_effect=lambda name: sentinel if name==found else self.fail_load(name)
                self.assertEqual(mkl_fft.load_runtime(),(sentinel,found))
                self.assertEqual([c.args[0] for c in load.call_args_list],candidates[:candidates.index(found)+1])
                self.assertEqual(dict(os.environ),before)

    @staticmethod
    def fail_load(name):
        raise OSError('cannot load '+name)

    def test_no_mklroot_or_runtime_is_explicit(self):
        with patch.dict(os.environ,{},clear=True), patch.object(ctypes.util,'find_library',return_value=None), patch.object(mkl_fft.C,'CDLL',side_effect=OSError('missing')) as load:
            with self.assertRaisesRegex(mkl_fft.MklRuntimeUnavailable,'oneMKL runtime not found'):
                mkl_fft.load_runtime()
            self.assertEqual(load.call_count,2)
            self.assertNotIn('MKLROOT',os.environ)

    def test_loaded_but_invalid_runtime_is_not_skipped(self):
        with patch.object(mkl_fft,'load_runtime',return_value=(object(),'bad-library')):
            with self.assertRaises(AttributeError):
                mkl_fft.prepared_fft(None,None,'fft',8,1)


if __name__=='__main__':unittest.main()
